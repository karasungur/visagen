"""
DFL Lightning Module for Visagen.

Main training module that combines encoder and decoder,
manages the training loop, and handles optimization.
Supports optional GAN training with PatchGAN discriminator
and temporal training with 3D Conv discriminator.
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from visagen.models.decoders.decoder import Decoder
from visagen.models.encoders.convnext import ConvNeXtEncoder
from visagen.training.losses import DSSIMLoss, MultiScaleDSSIMLoss


class DFLModule(pl.LightningModule):
    """
    DeepFaceLab Lightning Module with optional GAN training.

    Combines ConvNeXt encoder and decoder for face swapping training.
    Supports multiple loss functions including DSSIM, LPIPS, ID loss,
    and optional adversarial training with PatchGAN discriminator.

    Args:
        image_size: Input/output image size. Default: 256.
        in_channels: Number of input channels. Default: 3.
        encoder_dims: Channel dims for encoder stages. Default: [64, 128, 256, 512].
        encoder_depths: Block depths per encoder stage. Default: [2, 2, 4, 2].
        decoder_dims: Channel dims for decoder stages. Default: [512, 256, 128, 64].
        latent_dim: Latent space dimension. Default: 512.
        learning_rate: Learning rate for optimizer. Default: 1e-4.
        weight_decay: Weight decay for AdamW. Default: 0.01.
        drop_path_rate: Stochastic depth rate. Default: 0.1.
        dssim_weight: Weight for DSSIM loss. Default: 10.0.
        l1_weight: Weight for L1 loss. Default: 10.0.
        lpips_weight: Weight for LPIPS loss. Default: 0.0.
        id_weight: Weight for identity loss. Default: 0.0.
        use_multiscale_dssim: Use multi-scale DSSIM. Default: True.
        gan_power: GAN loss weight. 0 disables GAN training. Default: 0.0.
        gan_patch_size: Discriminator target receptive field. Default: 70.
        gan_mode: GAN loss mode ('vanilla', 'lsgan', 'hinge'). Default: 'vanilla'.
        gan_base_ch: Discriminator base channels. Default: 16.
        use_spectral_norm: Use spectral normalization in discriminator. Default: None (auto-enable for GAN/temporal).
        temporal_enabled: Enable temporal training with 3D discriminator. Default: False.
        temporal_power: Temporal GAN loss weight. Default: 0.1.
        temporal_sequence_length: Number of frames per sequence. Default: 5.
        temporal_consistency_weight: Weight for frame-to-frame consistency loss. Default: 1.0.
        temporal_base_ch: Base channels for temporal discriminator. Default: 32.
        model_type: Model architecture type ("standard", "diffusion", "eg3d"). Default: "standard".
        diffusion_texture_weight: Texture consistency loss weight for diffusion model. Default: 0.0.
        use_pretrained_vae: Use pretrained SD VAE for diffusion model. Default: True.
        eg3d_latent_dim: Latent dimension for EG3D model. Default: 512.
        eg3d_plane_channels: Number of channels per tri-plane. Default: 32.
        eg3d_render_resolution: Neural render resolution for EG3D. Default: 64.

    Example:
        >>> module = DFLModule()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = module(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])

        # With GAN training:
        >>> module = DFLModule(gan_power=0.1, gan_patch_size=70)

        # With temporal training:
        >>> module = DFLModule(temporal_enabled=True, temporal_power=0.1)
        >>> seq = torch.randn(2, 3, 5, 256, 256)  # (B, C, T, H, W)
        >>> out_seq = module.forward_sequence(seq)
        >>> out_seq.shape
        torch.Size([2, 3, 5, 256, 256])
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        encoder_dims: list[int] | None = None,
        encoder_depths: list[int] | None = None,
        decoder_dims: list[int] | None = None,
        latent_dim: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        drop_path_rate: float = 0.1,
        # Loss weights
        dssim_weight: float = 10.0,
        l1_weight: float = 10.0,
        lpips_weight: float = 0.0,
        id_weight: float = 0.0,
        eyes_mouth_weight: float = 0.0,
        gaze_weight: float = 0.0,
        use_multiscale_dssim: bool = True,
        # GAN parameters
        gan_power: float = 0.0,
        gan_patch_size: int = 70,
        gan_mode: str = "vanilla",
        gan_base_ch: int = 16,
        use_spectral_norm: bool | None = None,
        # Temporal parameters
        temporal_enabled: bool = False,
        temporal_power: float = 0.1,
        temporal_sequence_length: int = 5,
        temporal_consistency_weight: float = 1.0,
        temporal_base_ch: int = 32,
        # Experimental model parameters
        model_type: str = "standard",  # "standard", "diffusion", "eg3d"
        diffusion_texture_weight: float = 0.0,
        use_pretrained_vae: bool = True,
        eg3d_latent_dim: int = 512,
        eg3d_plane_channels: int = 32,
        eg3d_render_resolution: int = 64,
    ) -> None:
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Auto-enable spectral norm for GAN/Temporal training if not specified
        if use_spectral_norm is None:
            use_spectral_norm = (gan_power > 0) or temporal_enabled

        # Default dimensions
        if encoder_dims is None:
            encoder_dims = [64, 128, 256, 512]
        if encoder_depths is None:
            encoder_depths = [2, 2, 4, 2]
        if decoder_dims is None:
            decoder_dims = [512, 256, 128, 64]

        # Store config
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gan_power = gan_power
        self.gan_mode = gan_mode
        self.temporal_enabled = temporal_enabled
        self.temporal_power = temporal_power
        self.temporal_sequence_length = temporal_sequence_length
        self.temporal_consistency_weight = temporal_consistency_weight
        self.model_type = model_type

        # Calculate skip connection dimensions
        skip_dims = encoder_dims[:-1][::-1] + [encoder_dims[0]]

        # Build model based on type
        self._build_model(
            encoder_dims=encoder_dims,
            encoder_depths=encoder_depths,
            decoder_dims=decoder_dims,
            drop_path_rate=drop_path_rate,
            in_channels=in_channels,
            skip_dims=skip_dims,
            image_size=image_size,
        )

        # Build loss functions
        self._init_losses(
            dssim_weight=dssim_weight,
            l1_weight=l1_weight,
            lpips_weight=lpips_weight,
            id_weight=id_weight,
            eyes_mouth_weight=eyes_mouth_weight,
            gaze_weight=gaze_weight,
            use_multiscale_dssim=use_multiscale_dssim,
        )

        # Initialize GAN components if enabled
        if gan_power > 0:
            self._init_gan(
                in_channels=in_channels,
                patch_size=gan_patch_size,
                base_ch=gan_base_ch,
                use_spectral_norm=use_spectral_norm,
            )
            # Manual optimization for GAN training
            self.automatic_optimization = False
        else:
            self.discriminator = None
            self.gan_loss = None
            self.d_loss_fn = None
            self.tv_loss = None

        # Initialize temporal components if enabled
        if temporal_enabled:
            self._init_temporal(
                in_channels=in_channels,
                sequence_length=temporal_sequence_length,
                base_ch=temporal_base_ch,
                use_spectral_norm=use_spectral_norm,
            )
            # Manual optimization for temporal training
            self.automatic_optimization = False
        else:
            self.temporal_discriminator = None
            self.temporal_gan_loss = None
            self.temporal_d_loss_fn = None
            self.temporal_consistency_loss = None

    def _build_model(
        self,
        encoder_dims: list[int],
        encoder_depths: list[int],
        decoder_dims: list[int],
        drop_path_rate: float,
        in_channels: int,
        skip_dims: list[int],
        image_size: int,
    ) -> None:
        """
        Build encoder/decoder based on model_type.

        Args:
            encoder_dims: Channel dimensions for encoder stages.
            encoder_depths: Block depths per encoder stage.
            decoder_dims: Channel dimensions for decoder stages.
            drop_path_rate: Stochastic depth rate.
            in_channels: Number of input channels.
            skip_dims: Skip connection dimensions.
            image_size: Input/output image size.
        """
        if self.model_type == "diffusion":
            from visagen.models.experimental.diffusion import DiffusionAutoEncoder

            self.model = DiffusionAutoEncoder(
                image_size=image_size,
                structure_dims=encoder_dims,
                structure_depths=encoder_depths,
                texture_dim=encoder_dims[-1],  # Match structure latent dim
                decoder_dims=decoder_dims,
                use_pretrained_vae=self.hparams.use_pretrained_vae,
                use_attention=True,
            )
            self.encoder = None
            self.decoder = None
        elif self.model_type == "eg3d":
            from visagen.models.experimental.eg3d import EG3DEncoder, EG3DGenerator

            self.encoder = EG3DEncoder(
                latent_dim=self.hparams.eg3d_latent_dim,
                backbone_dims=encoder_dims,
                backbone_depths=encoder_depths,
            )
            self.model = EG3DGenerator(
                latent_dim=self.hparams.eg3d_latent_dim,
                plane_channels=self.hparams.eg3d_plane_channels,
                render_resolution=self.hparams.eg3d_render_resolution,
                output_resolution=image_size,
            )
            self.decoder = None
        else:
            # Standard encoder-decoder
            self.encoder = ConvNeXtEncoder(
                in_channels=in_channels,
                dims=encoder_dims,
                depths=encoder_depths,
                drop_path_rate=drop_path_rate,
            )
            self.decoder = Decoder(
                latent_channels=encoder_dims[-1],
                dims=decoder_dims,
                skip_dims=skip_dims,
                out_channels=in_channels,
                use_attention=True,
            )
            self.model = None

    def _init_losses(
        self,
        dssim_weight: float,
        l1_weight: float,
        lpips_weight: float,
        id_weight: float,
        eyes_mouth_weight: float,
        gaze_weight: float,
        use_multiscale_dssim: bool,
    ) -> None:
        """Initialize loss functions."""
        self.dssim_weight = dssim_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.id_weight = id_weight
        self.eyes_mouth_weight = eyes_mouth_weight
        self.gaze_weight = gaze_weight
        self.texture_weight = self.hparams.get("diffusion_texture_weight", 0.0)

        # DSSIM loss
        if use_multiscale_dssim:
            self.dssim_loss = MultiScaleDSSIMLoss()
        else:
            self.dssim_loss = DSSIMLoss()

        # LPIPS loss (lazy loaded)
        self._lpips_loss = None

        # ID loss (lazy loaded)
        self._id_loss = None

        # Eyes/Mouth loss (lazy loaded)
        self._eyes_mouth_loss = None

        # Gaze loss (lazy loaded)
        self._gaze_loss = None

        # Texture loss (lazy loaded - for diffusion model)
        self._texture_loss = None

    def _init_gan(
        self,
        in_channels: int,
        patch_size: int,
        base_ch: int,
        use_spectral_norm: bool,
    ) -> None:
        """Initialize GAN components."""
        from visagen.models.discriminators import UNetPatchDiscriminator
        from visagen.training.losses import (
            DiscriminatorLoss,
            GANLoss,
            TotalVariationLoss,
        )

        self.discriminator = UNetPatchDiscriminator(
            in_channels=in_channels,
            patch_size=patch_size,
            base_ch=base_ch,
            use_spectral_norm=use_spectral_norm,
        )

        self.gan_loss = GANLoss(mode=self.gan_mode)
        self.d_loss_fn = DiscriminatorLoss(mode=self.gan_mode)
        self.tv_loss = TotalVariationLoss(weight=1e-6)

    def _init_temporal(
        self,
        in_channels: int,
        sequence_length: int,
        base_ch: int,
        use_spectral_norm: bool,
    ) -> None:
        """Initialize temporal discriminator and losses."""
        from visagen.models.discriminators import TemporalDiscriminator
        from visagen.training.losses import (
            TemporalConsistencyLoss,
            TemporalDiscriminatorLoss,
            TemporalGANLoss,
            TotalVariationLoss,
        )

        self.temporal_discriminator = TemporalDiscriminator(
            in_channels=in_channels,
            base_ch=base_ch,
            sequence_length=sequence_length,
            use_spectral_norm=use_spectral_norm,
        )

        self.temporal_gan_loss = TemporalGANLoss(mode=self.gan_mode)
        self.temporal_d_loss_fn = TemporalDiscriminatorLoss(mode=self.gan_mode)
        self.temporal_consistency_loss = TemporalConsistencyLoss(
            weight=self.temporal_consistency_weight
        )

        # Also need TV loss for temporal if not already initialized
        if self.tv_loss is None:
            self.tv_loss = TotalVariationLoss(weight=1e-6)

    @property
    def lpips_loss(self):
        """Lazy load LPIPS loss."""
        if self._lpips_loss is None and self.lpips_weight > 0:
            from visagen.training.losses import LPIPSLoss

            self._lpips_loss = LPIPSLoss()
        return self._lpips_loss

    @property
    def id_loss(self):
        """Lazy load ID loss."""
        if self._id_loss is None and self.id_weight > 0:
            from visagen.training.losses import IDLoss

            self._id_loss = IDLoss()
        return self._id_loss

    @property
    def eyes_mouth_loss(self):
        """Lazy load Eyes/Mouth loss."""
        if self._eyes_mouth_loss is None and self.eyes_mouth_weight > 0:
            from visagen.training.losses import EyesMouthLoss

            self._eyes_mouth_loss = EyesMouthLoss()
        return self._eyes_mouth_loss

    @property
    def gaze_loss_fn(self):
        """Lazy load Gaze loss."""
        if self._gaze_loss is None and self.gaze_weight > 0:
            from visagen.training.losses import GazeLoss

            self._gaze_loss = GazeLoss()
        return self._gaze_loss

    @property
    def texture_loss(self):
        """Lazy load Texture Consistency loss (for diffusion model)."""
        if self._texture_loss is None and self.texture_weight > 0:
            from visagen.training.diffusion_losses import TextureConsistencyLoss

            self._texture_loss = TextureConsistencyLoss()
        return self._texture_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        if self.model_type == "diffusion":
            # DiffusionAutoEncoder handles both encoding and decoding
            return self.model(x)
        elif self.model_type == "eg3d":
            # EG3D: encode to latent, then generate 3D-aware image
            z = self.encoder(x)
            return self.model(z)
        else:
            # Standard encoder-decoder
            features, latent = self.encoder(x)

            # Prepare skip connections (reverse order for decoder)
            # features: [stage0, stage1, stage2, stage3]
            # decoder needs: [stage2, stage1, stage0, stage0] (deep to shallow)
            skip_features = features[:-1][::-1] + [features[0]]

            # Decode
            output = self.decoder(latent, skip_features)

            return output

    def forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a sequence of frames.

        Processes each frame independently through encoder/decoder.

        Args:
            sequence: Input sequence tensor of shape (B, C, T, H, W).

        Returns:
            Reconstructed sequence tensor of shape (B, C, T, H, W).
        """
        B, C, T, H, W = sequence.shape

        # Reshape to (B*T, C, H, W) for batch processing
        frames = sequence.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Process all frames through encoder/decoder
        output_frames = self(frames)

        # Reshape back to (B, C, T, H, W)
        output = output_frames.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return output

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined reconstruction loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).
            landmarks: Optional facial landmarks (B, 68, 2) for region-based losses.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # DSSIM loss
        if self.dssim_weight > 0:
            loss_dssim = self.dssim_loss(pred, target)
            losses["dssim"] = loss_dssim
            total = total + self.dssim_weight * loss_dssim

        # L1 loss
        if self.l1_weight > 0:
            loss_l1 = F.l1_loss(pred, target)
            losses["l1"] = loss_l1
            total = total + self.l1_weight * loss_l1

        # LPIPS loss
        if self.lpips_weight > 0 and self.lpips_loss is not None:
            loss_lpips = self.lpips_loss(pred, target)
            losses["lpips"] = loss_lpips
            total = total + self.lpips_weight * loss_lpips

        # ID loss
        if self.id_weight > 0 and self.id_loss is not None:
            loss_id = self.id_loss(pred, target)
            losses["id"] = loss_id
            total = total + self.id_weight * loss_id

        # Eyes/Mouth loss (requires landmarks)
        if self.eyes_mouth_weight > 0 and self.eyes_mouth_loss is not None:
            if landmarks is not None:
                loss_em = self.eyes_mouth_loss(pred, target, landmarks)
                losses["eyes_mouth"] = loss_em
                total = total + self.eyes_mouth_weight * loss_em

        # Gaze loss (requires landmarks)
        if self.gaze_weight > 0 and self.gaze_loss_fn is not None:
            if landmarks is not None:
                loss_gaze = self.gaze_loss_fn(pred, target, landmarks)
                losses["gaze"] = loss_gaze
                total = total + self.gaze_weight * loss_gaze

        # Texture consistency loss (for diffusion model)
        if self.texture_weight > 0 and self.texture_loss is not None:
            loss_texture = self.texture_loss(pred, target)
            losses["texture"] = loss_texture
            total = total + self.texture_weight * loss_texture

        losses["total"] = total
        return total, losses

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor | None:
        """
        Training step with optional GAN or temporal training.

        Training modes (checked in priority order):
        1. Temporal training (temporal_enabled=True): Uses sequence data (B, C, T, H, W)
        2. GAN training (gan_power > 0): Uses adversarial loss
        3. Autoencoder training: Standard reconstruction loss only

        Args:
            batch: Tuple of (src_dict, dst_dict) each containing:
                   - 'image': (B, C, H, W) tensor or (B, C, T, H, W) for temporal
                   - 'landmarks': (B, 68, 2) tensor (optional)
                   - 'mask': (B, 1, H, W) tensor (optional)
            batch_idx: Batch index.

        Returns:
            Loss value (only when GAN and temporal are disabled).
        """
        src_dict, dst_dict = batch

        # Extract images and landmarks
        src = src_dict["image"]
        dst = dst_dict["image"]
        src_landmarks = src_dict.get("landmarks")

        if self.temporal_enabled:
            return self._training_step_temporal(src, dst, src_landmarks, batch_idx)
        elif self.gan_power > 0:
            return self._training_step_gan(src, dst, src_landmarks, batch_idx)
        else:
            return self._training_step_ae(src, dst, src_landmarks, batch_idx)

    def _training_step_ae(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        landmarks: torch.Tensor | None,
        batch_idx: int,
    ) -> torch.Tensor:
        """Standard autoencoder training step."""
        # Forward pass (reconstruct source from source)
        pred = self(src)

        # Compute losses (with landmarks for eyes_mouth/gaze losses)
        total_loss, loss_dict = self.compute_loss(pred, src, landmarks)

        # Log all losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def _training_step_gan(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        landmarks: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """GAN training step with manual optimization."""
        g_opt, d_opt = self.optimizers()

        # Forward pass
        pred = self(src)

        # === GENERATOR STEP ===
        g_opt.zero_grad()

        # Reconstruction loss (with landmarks for eyes_mouth/gaze losses)
        total_loss, loss_dict = self.compute_loss(pred, src, landmarks)

        # Adversarial loss for generator
        # Generator wants discriminator to classify fake as real
        d_fake_center, d_fake_final = self.discriminator(pred)

        g_adv_loss = self.gan_loss(d_fake_center, target_is_real=True) + self.gan_loss(
            d_fake_final, target_is_real=True
        )

        # Total variation to suppress artifacts
        tv_loss = self.tv_loss(pred)

        # Combined generator loss
        g_total = total_loss + self.gan_power * g_adv_loss + tv_loss

        self.manual_backward(g_total)
        g_opt.step()

        loss_dict["g_adv"] = g_adv_loss
        loss_dict["tv"] = tv_loss
        loss_dict["g_total"] = g_total

        # === DISCRIMINATOR STEP ===
        d_opt.zero_grad()

        # Detach generated images to avoid backprop through generator
        pred_detached = pred.detach()

        # Discriminator outputs for real and fake
        d_real_center, d_real_final = self.discriminator(src)
        d_fake_center, d_fake_final = self.discriminator(pred_detached)

        # Discriminator loss (real=1, fake=0)
        d_loss = (
            self.d_loss_fn(d_real_center, d_fake_center)
            + self.d_loss_fn(d_real_final, d_fake_final)
        ) * 0.5

        self.manual_backward(d_loss)
        d_opt.step()

        loss_dict["d_loss"] = d_loss

        # Log losses
        for name, value in loss_dict.items():
            prog_bar = name in ["total", "d_loss", "g_adv"]
            self.log(f"train_{name}", value, prog_bar=prog_bar)

    def _training_step_temporal(
        self,
        src_seq: torch.Tensor,
        dst_seq: torch.Tensor,
        landmarks: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """
        Temporal training step with 3D discriminator.

        Trains on frame sequences for temporal consistency.

        Args:
            src_seq: Source sequence (B, C, T, H, W).
            dst_seq: Destination sequence (B, C, T, H, W).
            landmarks: Source landmarks (B, 68, 2) - Note: per-batch, not per-frame.
            batch_idx: Batch index.
        """
        # Get optimizers based on whether spatial GAN is also enabled
        if self.gan_power > 0:
            g_opt, d_opt, t_opt = self.optimizers()
        else:
            g_opt, t_opt = self.optimizers()

        B, C, T, H, W = src_seq.shape

        # Forward pass through sequence
        pred_seq = self.forward_sequence(src_seq)

        # === GENERATOR STEP ===
        g_opt.zero_grad()

        # Per-frame reconstruction loss
        # Reshape sequences to (B*T, C, H, W) for loss computation
        src_flat = src_seq.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        pred_flat = pred_seq.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Note: landmarks are per-batch, not per-frame, so we don't pass them
        # for temporal training. Eyes/mouth/gaze losses work best with
        # standard image training, not sequences.
        total_loss, loss_dict = self.compute_loss(pred_flat, src_flat, None)

        # Temporal consistency loss
        temp_cons_loss = self.temporal_consistency_loss(pred_seq)
        loss_dict["temp_cons"] = temp_cons_loss

        # Temporal adversarial loss
        t_fake_score = self.temporal_discriminator(pred_seq)
        t_adv_loss = self.temporal_gan_loss(t_fake_score, target_is_real=True)
        loss_dict["t_adv"] = t_adv_loss

        # TV loss on predictions
        tv_loss = self.tv_loss(pred_flat)
        loss_dict["tv"] = tv_loss

        # Spatial GAN loss if enabled
        g_adv_loss = torch.tensor(0.0, device=pred_seq.device)
        if self.gan_power > 0 and self.discriminator is not None:
            d_fake_center, d_fake_final = self.discriminator(pred_flat)
            g_adv_loss = self.gan_loss(
                d_fake_center, target_is_real=True
            ) + self.gan_loss(d_fake_final, target_is_real=True)
            loss_dict["g_adv"] = g_adv_loss

        # Combined generator loss
        g_total = (
            total_loss
            + self.temporal_power * t_adv_loss
            + temp_cons_loss
            + tv_loss
            + self.gan_power * g_adv_loss
        )
        loss_dict["g_total"] = g_total

        self.manual_backward(g_total)
        g_opt.step()

        # === SPATIAL DISCRIMINATOR STEP (if enabled) ===
        if self.gan_power > 0 and self.discriminator is not None:
            d_opt.zero_grad()

            pred_detached = pred_flat.detach()
            d_real_center, d_real_final = self.discriminator(src_flat)
            d_fake_center, d_fake_final = self.discriminator(pred_detached)

            d_loss = (
                self.d_loss_fn(d_real_center, d_fake_center)
                + self.d_loss_fn(d_real_final, d_fake_final)
            ) * 0.5

            self.manual_backward(d_loss)
            d_opt.step()
            loss_dict["d_loss"] = d_loss

        # === TEMPORAL DISCRIMINATOR STEP ===
        t_opt.zero_grad()

        pred_seq_detached = pred_seq.detach()
        t_real_score = self.temporal_discriminator(src_seq)
        t_fake_score = self.temporal_discriminator(pred_seq_detached)

        t_d_loss = self.temporal_d_loss_fn(t_real_score, t_fake_score)

        self.manual_backward(t_d_loss)
        t_opt.step()
        loss_dict["t_d_loss"] = t_d_loss

        # Log losses
        for name, value in loss_dict.items():
            prog_bar = name in ["total", "t_adv", "t_d_loss", "temp_cons"]
            self.log(f"train_{name}", value, prog_bar=prog_bar)

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Tuple of (src_dict, dst_dict) each containing:
                   - 'image': (B, C, H, W) tensor or (B, C, T, H, W) for temporal
                   - 'landmarks': (B, 68, 2) tensor (optional)
                   - 'mask': (B, 1, H, W) tensor (optional)
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        src_dict, dst_dict = batch
        src = src_dict["image"]
        src_landmarks = src_dict.get("landmarks")

        if self.temporal_enabled:
            # Temporal validation
            B, C, T, H, W = src.shape
            pred_seq = self.forward_sequence(src)

            # Flatten for loss computation
            src_flat = src.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            pred_flat = pred_seq.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

            # Landmarks not used in temporal validation
            total_loss, loss_dict = self.compute_loss(pred_flat, src_flat, None)

            # Add temporal consistency loss
            temp_cons_loss = self.temporal_consistency_loss(pred_seq)
            loss_dict["temp_cons"] = temp_cons_loss
            total_loss = total_loss + temp_cons_loss
        else:
            # Standard validation
            pred = self(src)
            total_loss, loss_dict = self.compute_loss(pred, src, src_landmarks)

        # Log validation losses
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def configure_optimizers(self) -> dict[str, Any] | tuple[list, list]:
        """
        Configure optimizer(s) and scheduler(s).

        Training modes and optimizers:
        - Temporal + GAN: 3 optimizers (generator, spatial discriminator, temporal discriminator)
        - Temporal only: 2 optimizers (generator, temporal discriminator)
        - GAN only: 2 optimizers (generator, spatial discriminator)
        - AE only: 1 optimizer

        Returns:
            Single optimizer dict (AE mode) or tuple of optimizer/scheduler lists (GAN/temporal mode).
        """
        max_epochs = self.trainer.max_epochs if self.trainer else 100

        # Generator optimizer - model type dependent
        if self.model_type == "diffusion":
            g_params = list(self.model.parameters())
        elif self.model_type == "eg3d":
            g_params = list(self.encoder.parameters()) + list(self.model.parameters())
        else:
            # Standard encoder-decoder
            g_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        g_optimizer = torch.optim.AdamW(
            g_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer, T_max=max_epochs, eta_min=1e-6
        )

        if self.temporal_enabled and self.gan_power > 0:
            # Temporal + GAN: 3 optimizers
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                d_optimizer, T_max=max_epochs, eta_min=1e-6
            )

            t_optimizer = torch.optim.AdamW(
                self.temporal_discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
            t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                t_optimizer, T_max=max_epochs, eta_min=1e-6
            )

            return (
                [g_optimizer, d_optimizer, t_optimizer],
                [
                    {"scheduler": g_scheduler, "interval": "epoch"},
                    {"scheduler": d_scheduler, "interval": "epoch"},
                    {"scheduler": t_scheduler, "interval": "epoch"},
                ],
            )

        elif self.temporal_enabled:
            # Temporal only: 2 optimizers (generator + temporal discriminator)
            t_optimizer = torch.optim.AdamW(
                self.temporal_discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
            t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                t_optimizer, T_max=max_epochs, eta_min=1e-6
            )

            return (
                [g_optimizer, t_optimizer],
                [
                    {"scheduler": g_scheduler, "interval": "epoch"},
                    {"scheduler": t_scheduler, "interval": "epoch"},
                ],
            )

        elif self.gan_power > 0:
            # GAN only: 2 optimizers (generator + spatial discriminator)
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                d_optimizer, T_max=max_epochs, eta_min=1e-6
            )

            return (
                [g_optimizer, d_optimizer],
                [
                    {"scheduler": g_scheduler, "interval": "epoch"},
                    {"scheduler": d_scheduler, "interval": "epoch"},
                ],
            )
        else:
            # AE only: single optimizer - use the already created g_optimizer
            return {
                "optimizer": g_optimizer,
                "lr_scheduler": {
                    "scheduler": g_scheduler,
                    "interval": "epoch",
                },
            }
