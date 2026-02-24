"""
Training Lightning Module for Visagen.

Main training module that combines encoder and decoder,
manages the training loop, and handles optimization.
Supports optional GAN training with PatchGAN discriminator
and temporal training with 3D Conv discriminator.
"""

import logging
import math
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from visagen.models.decoders.decoder import Decoder
from visagen.models.encoders.convnext import ConvNeXtEncoder
from visagen.training.losses import DSSIMLoss, MultiScaleDSSIMLoss

logger = logging.getLogger(__name__)


class TrainingModule(pl.LightningModule):
    """
    Visagen Training Lightning Module with optional GAN training.

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
        >>> module = TrainingModule()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = module(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])

        # With GAN training:
        >>> module = TrainingModule(gan_power=0.1, gan_patch_size=70)

        # With temporal training:
        >>> module = TrainingModule(temporal_enabled=True, temporal_power=0.1)
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
        # Optimizer settings
        optimizer_type: str = "adamw",  # "adamw", "adabelief"
        lr_dropout: float = 1.0,  # 1.0 = no dropout (for AdaBelief)
        lr_cos_period: int = 0,  # 0 = disabled (for AdaBelief)
        clipnorm: float = 0.0,  # 0.0 = disabled (for AdaBelief)
        # Scheduler settings
        warmup_epochs: int = 0,  # 0 = disabled
        scheduler_type: str = "cosine",  # "cosine", "plateau", "constant"
        # Loss weights
        dssim_weight: float = 10.0,
        l1_weight: float = 10.0,
        lpips_weight: float = 0.0,
        id_weight: float = 0.0,
        eyes_mouth_weight: float = 0.0,
        gaze_weight: float = 0.0,
        use_multiscale_dssim: bool = True,
        # Style loss weights
        face_style_weight: float = 0.0,
        bg_style_weight: float = 0.0,
        # Augmentation options
        blur_out_mask: bool = False,
        # Mask training options
        mask_weight: float = 10.0,
        masked_training: bool = False,
        # true_face_power for identity preservation
        true_face_power: float = 0.0,
        # GAN parameters
        gan_power: float = 0.0,
        gan_patch_size: int = 70,
        gan_mode: str = "vanilla",
        gan_base_ch: int = 16,
        feature_matching_weight: float = 0.0,
        d_betas: tuple[float, float] = (0.5, 0.999),
        use_spectral_norm: bool | None = None,
        # Temporal parameters
        temporal_enabled: bool = False,
        temporal_power: float = 0.1,
        temporal_sequence_length: int = 5,
        temporal_consistency_weight: float = 1.0,
        temporal_base_ch: int = 32,
        temporal_checkpoint: bool = True,
        # Gradient accumulation for large batch training
        gradient_accumulation_steps: int = 1,
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
        self.image_size = image_size

        # Store new feature flags
        self.blur_out_mask_enabled = blur_out_mask
        self.true_face_power = true_face_power
        self.face_style_weight = face_style_weight
        self.bg_style_weight = bg_style_weight
        self.mask_weight = mask_weight
        self.masked_training = masked_training
        self.feature_matching_weight = feature_matching_weight
        self.d_betas = d_betas

        # Optional components are initialized lazily or conditionally.
        self.model: Any | None = None
        self.encoder: Any | None = None
        self.decoder: Any | None = None
        self.discriminator: Any | None = None
        self.gan_loss: Any | None = None
        self.d_loss_fn: Any | None = None
        self.tv_loss: Any | None = None
        self.feature_matching_loss: Any | None = None
        self.temporal_discriminator: Any | None = None
        self.temporal_gan_loss: Any | None = None
        self.temporal_d_loss_fn: Any | None = None
        self.temporal_consistency_loss: Any | None = None
        self.code_discriminator: Any | None = None

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

        # Initialize CodeDiscriminator for true_face_power
        if true_face_power > 0:
            from visagen.models.discriminators import CodeDiscriminator

            self.code_discriminator = CodeDiscriminator(
                in_ch=encoder_dims[-1],
                hidden_ch=encoder_dims[-1],
            )
            # Enable manual optimization if not already enabled
            if self.automatic_optimization:
                self.automatic_optimization = False
        else:
            self.code_discriminator = None

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

            use_pretrained_vae = bool(self.hparams.get("use_pretrained_vae", True))
            self.model = DiffusionAutoEncoder(
                image_size=image_size,
                structure_dims=encoder_dims,
                structure_depths=encoder_depths,
                texture_dim=encoder_dims[-1],  # Match structure latent dim
                decoder_dims=decoder_dims,
                use_pretrained_vae=use_pretrained_vae,
                use_attention=True,
            )
            self.encoder = None
            self.decoder = None
        elif self.model_type == "eg3d":
            from visagen.models.experimental.eg3d import EG3DEncoder, EG3DGenerator

            eg3d_latent_dim = int(self.hparams.get("eg3d_latent_dim", 512))
            eg3d_plane_channels = int(self.hparams.get("eg3d_plane_channels", 32))
            eg3d_render_resolution = int(self.hparams.get("eg3d_render_resolution", 64))

            self.encoder = EG3DEncoder(
                latent_dim=eg3d_latent_dim,
                backbone_dims=encoder_dims,
                backbone_depths=encoder_depths,
            )
            self.model = EG3DGenerator(
                latent_dim=eg3d_latent_dim,
                plane_channels=eg3d_plane_channels,
                render_resolution=eg3d_render_resolution,
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
                mask_output=True,
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
        self.dssim_loss: DSSIMLoss | MultiScaleDSSIMLoss
        if use_multiscale_dssim:
            self.dssim_loss = MultiScaleDSSIMLoss(resolution=self.image_size)
        else:
            filter_size = max(3, int(self.image_size / 11.6)) | 1
            self.dssim_loss = DSSIMLoss(filter_size=filter_size)

        # LPIPS loss (lazy loaded)
        self._lpips_loss: Any | None = None

        # ID loss (lazy loaded)
        self._id_loss: Any | None = None

        # Eyes/Mouth loss (lazy loaded)
        self._eyes_mouth_loss: Any | None = None

        # Gaze loss (lazy loaded)
        self._gaze_loss: Any | None = None

        # Texture loss (lazy loaded - for diffusion model)
        self._texture_loss: Any | None = None

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

        # Feature matching loss (if enabled)
        if self.feature_matching_weight > 0:
            from visagen.training.losses import FeatureMatchingLoss

            self.feature_matching_loss = FeatureMatchingLoss(
                weight=self.feature_matching_weight
            )

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
    def lpips_loss(self) -> Any | None:
        """Lazy load LPIPS loss."""
        if self._lpips_loss is None and self.lpips_weight > 0:
            from visagen.training.losses import LPIPSLoss

            self._lpips_loss = LPIPSLoss()
        return self._lpips_loss

    @property
    def id_loss(self) -> Any | None:
        """Lazy load ID loss."""
        if self._id_loss is None and self.id_weight > 0:
            from visagen.training.losses import IDLoss

            self._id_loss = IDLoss()
        return self._id_loss

    @property
    def eyes_mouth_loss(self) -> Any | None:
        """Lazy load Eyes/Mouth loss."""
        if self._eyes_mouth_loss is None and self.eyes_mouth_weight > 0:
            from visagen.training.losses import EyesMouthLoss

            self._eyes_mouth_loss = EyesMouthLoss()
        return self._eyes_mouth_loss

    @property
    def gaze_loss_fn(self) -> Any | None:
        """Lazy load Gaze loss."""
        if self._gaze_loss is None and self.gaze_weight > 0:
            from visagen.training.losses import GazeLoss

            self._gaze_loss = GazeLoss()
        return self._gaze_loss

    @property
    def texture_loss(self) -> Any | None:
        """Lazy load Texture Consistency loss (for diffusion model)."""
        if self._texture_loss is None and self.texture_weight > 0:
            from visagen.training.diffusion_losses import TextureConsistencyLoss

            self._texture_loss = TextureConsistencyLoss()
        return self._texture_loss

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            For standard mode with mask: tuple of (image, mask) tensors.
            For diffusion/eg3d modes: image tensor only.
        """
        if self.model_type == "diffusion":
            # DiffusionAutoEncoder handles both encoding and decoding
            assert self.model is not None
            return cast(torch.Tensor, self.model(x))
        elif self.model_type == "eg3d":
            # EG3D: encode to latent, then generate 3D-aware image
            assert self.encoder is not None
            assert self.model is not None
            z = self.encoder(x)
            return cast(torch.Tensor, self.model(z))
        else:
            # Standard encoder-decoder
            assert self.encoder is not None
            assert self.decoder is not None
            features, latent = self.encoder(x)

            # Prepare skip connections (reverse order for decoder)
            # features: [stage0, stage1, stage2, stage3]
            # decoder needs: [stage2, stage1, stage0, stage0] (deep to shallow)
            skip_features = features[:-1][::-1] + [features[0]]

            # Decode — returns (image, mask) tuple with mask_output=True
            result = self.decoder(latent, skip_features)
            if isinstance(result, tuple):
                return cast(torch.Tensor, result[0]), cast(torch.Tensor, result[1])
            return cast(torch.Tensor, result)

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
        result = self(frames)

        # Unpack tuple if decoder returns (image, mask)
        if isinstance(result, tuple):
            output_frames = result[0]
        else:
            output_frames = result

        # Reshape back to (B, C, T, H, W)
        output = output_frames.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return cast(torch.Tensor, output)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pred_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined reconstruction loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).
            landmarks: Optional facial landmarks (B, 68, 2) for region-based losses.
            mask: Optional face mask (B, 1, H, W) for style losses.
            pred_mask: Optional predicted mask (B, 1, H, W) from decoder.
            target_mask: Optional ground truth mask (B, 1, H, W) for mask loss.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Determine effective inputs for masked training
        pred_eff = pred
        target_eff = target
        if self.masked_training and mask is not None:
            pred_eff = pred * mask
            target_eff = target * mask

        # DSSIM loss
        if self.dssim_weight > 0:
            loss_dssim = self.dssim_loss(pred_eff, target_eff)
            losses["dssim"] = loss_dssim
            total = total + self.dssim_weight * loss_dssim

        # L1 loss
        if self.l1_weight > 0:
            loss_l1 = F.l1_loss(pred_eff, target_eff)
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

        # Mask reconstruction loss
        if self.mask_weight > 0 and pred_mask is not None and target_mask is not None:
            loss_mask = self.mask_weight * F.mse_loss(pred_mask, target_mask)
            losses["mask"] = loss_mask
            total = total + loss_mask

        # Face style loss (requires mask)
        if self.face_style_weight > 0 and mask is not None:
            from visagen.training.losses import face_style_loss

            loss_face_style = face_style_loss(
                pred, target, mask, mask, blur_radius=self.image_size // 8
            )
            losses["face_style"] = loss_face_style
            total = total + self.face_style_weight * loss_face_style

        # Background style loss (requires mask)
        if self.bg_style_weight > 0 and mask is not None:
            from visagen.training.losses import bg_style_loss

            loss_bg_style = bg_style_loss(
                pred, target, mask, resolution=self.image_size
            )
            losses["bg_style"] = loss_bg_style
            total = total + self.bg_style_weight * loss_bg_style

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

        # Extract images, landmarks, and masks
        src = src_dict["image"]
        dst = dst_dict["image"]
        src_landmarks = src_dict.get("landmarks")
        src_mask = src_dict.get("mask")
        dst_mask = dst_dict.get("mask")

        # Apply blur_out_mask augmentation if enabled
        if self.blur_out_mask_enabled and src_mask is not None:
            from visagen.data.augmentations import blur_out_mask

            src = blur_out_mask(src, src_mask, resolution=self.image_size)

        if self.temporal_enabled:
            self._training_step_temporal(
                src, dst, src_landmarks, src_mask, dst_mask, batch_idx
            )
            return None
        if self.gan_power > 0 or self.true_face_power > 0:
            self._training_step_gan(
                src, dst, src_landmarks, src_mask, dst_mask, batch_idx
            )
            return None
        return self._training_step_ae(src, dst, src_landmarks, src_mask, batch_idx)

    def _training_step_ae(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        landmarks: torch.Tensor | None,
        mask: torch.Tensor | None,
        batch_idx: int,
    ) -> torch.Tensor:
        """Standard autoencoder training step."""
        # Forward pass (reconstruct source from source)
        result = self(src)

        # Unpack (image, mask) tuple if returned by standard decoder
        pred_mask: torch.Tensor | None = None
        if isinstance(result, tuple):
            pred, pred_mask = result
        else:
            pred = result

        # Compute losses (with landmarks, mask, and predicted mask)
        total_loss, loss_dict = self.compute_loss(
            pred,
            src,
            landmarks,
            mask,
            pred_mask=pred_mask,
            target_mask=mask,
        )

        # NaN/Inf guard to prevent training corruption
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(
                f"Loss is NaN/Inf at step {self.global_step}. Loss dict: {loss_dict}"
            )
            # Create gradient-safe dummy loss instead of zero
            dummy_loss = torch.tensor(
                1e-6, device=total_loss.device, requires_grad=True
            )
            return cast(torch.Tensor, dummy_loss * pred.mean().detach())

        # Log all losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def _training_step_gan(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        landmarks: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        dst_mask: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """GAN training step with manual optimization."""
        # Get optimizers based on what's enabled
        optimizers_obj = self.optimizers()
        if isinstance(optimizers_obj, (list, tuple)):
            optimizers = list(optimizers_obj)
        else:
            optimizers = [optimizers_obj]

        g_opt = cast(Any, optimizers[0])
        d_opt: Any | None = None
        code_opt: Any | None = None

        if self.true_face_power > 0 and self.code_discriminator is not None:
            if self.gan_power > 0:
                d_opt = cast(Any, optimizers[1])
                code_opt = cast(Any, optimizers[2])
            else:
                code_opt = cast(Any, optimizers[1])
        elif self.gan_power > 0:
            d_opt = cast(Any, optimizers[1])

        # Forward pass
        result = self(src)

        # Unpack (image, mask) tuple if returned by standard decoder
        pred_mask: torch.Tensor | None = None
        if isinstance(result, tuple):
            pred, pred_mask = result
        else:
            pred = result

        # === GENERATOR STEP ===
        g_opt.zero_grad()

        # Reconstruction loss (with landmarks, mask, and predicted mask)
        total_loss, loss_dict = self.compute_loss(
            pred,
            src,
            landmarks,
            src_mask,
            pred_mask=pred_mask,
            target_mask=src_mask,
        )

        # Adversarial loss for generator (if GAN enabled)
        g_adv_loss = torch.tensor(0.0, device=pred.device)
        fm_loss = torch.tensor(0.0, device=pred.device)
        if self.gan_power > 0 and self.discriminator is not None:
            # Generator wants discriminator to classify fake as real
            assert self.gan_loss is not None
            d_fake_center, d_fake_final = self.discriminator(pred)

            g_adv_loss = self.gan_loss(
                d_fake_center, target_is_real=True
            ) + self.gan_loss(d_fake_final, target_is_real=True)
            loss_dict["g_adv"] = g_adv_loss

            # Feature matching loss (compare discriminator features on real vs fake)
            if self.feature_matching_loss is not None:
                with torch.no_grad():
                    d_real_center, d_real_final = self.discriminator(src)
                fm_loss = self.feature_matching_loss(
                    [d_real_center, d_real_final],
                    [d_fake_center, d_fake_final],
                )
                loss_dict["fm"] = fm_loss

        # Total variation to suppress artifacts
        tv_loss = torch.tensor(0.0, device=pred.device)
        if self.tv_loss is not None:
            tv_loss = self.tv_loss(pred)
            loss_dict["tv"] = tv_loss

        # CodeDiscriminator loss for generator (true_face_power)
        g_code_loss = torch.tensor(0.0, device=pred.device)
        if self.true_face_power > 0 and self.code_discriminator is not None:
            # Get latent codes from encoder
            assert self.encoder is not None
            _, src_code = self.encoder(src)
            # Generator wants src_code to look like real (dst) codes
            src_code_pred = self.code_discriminator(src_code)
            g_code_loss = F.binary_cross_entropy_with_logits(
                src_code_pred, torch.ones_like(src_code_pred)
            )
            loss_dict["g_code"] = g_code_loss

        # Combined generator loss
        g_total = (
            total_loss
            + self.gan_power * g_adv_loss
            + tv_loss
            + fm_loss
            + self.true_face_power * g_code_loss
        )

        # NaN/Inf guard for generator loss
        if torch.isnan(g_total) or torch.isinf(g_total):
            logger.error(
                f"Generator loss is NaN/Inf at step {self.global_step}. "
                f"Loss dict: {loss_dict}"
            )
            return  # Skip this step

        # Apply gradient accumulation scaling if enabled
        accum_steps = self.hparams.get("gradient_accumulation_steps", 1)
        if accum_steps > 1:
            g_total = g_total / accum_steps

        self.manual_backward(g_total)

        # Only step optimizer at accumulation boundary
        if accum_steps > 1:
            if (batch_idx + 1) % accum_steps == 0:
                g_opt.step()
                g_opt.zero_grad()
        else:
            g_opt.step()

        loss_dict["g_total"] = g_total

        # === DISCRIMINATOR STEP (if GAN enabled) ===
        if self.gan_power > 0 and d_opt is not None and self.discriminator is not None:
            d_opt.zero_grad()

            # Detach generated images to avoid backprop through generator
            pred_detached = pred.detach()

            # Discriminator outputs for real and fake
            d_real_center, d_real_final = self.discriminator(src)
            d_fake_center, d_fake_final = self.discriminator(pred_detached)

            # Discriminator loss (real=1, fake=0)
            assert self.d_loss_fn is not None
            d_loss = (
                self.d_loss_fn(d_real_center, d_fake_center)
                + self.d_loss_fn(d_real_final, d_fake_final)
            ) * 0.5

            # Background regularization using dst_mask
            if dst_mask is not None and self.bg_style_weight > 0:
                from visagen.training.losses import bg_style_loss

                bg_reg = bg_style_loss(
                    pred_detached, dst, dst_mask, resolution=self.image_size
                )
                d_loss = d_loss + bg_reg
                loss_dict["d_bg_reg"] = bg_reg

            self.manual_backward(d_loss)
            d_opt.step()

            loss_dict["d_loss"] = d_loss

        # === CODE DISCRIMINATOR STEP (if true_face_power enabled) ===
        if self.true_face_power > 0 and self.code_discriminator is not None:
            assert code_opt is not None
            assert self.encoder is not None
            code_opt.zero_grad()

            # Get latent codes (detached)
            with torch.no_grad():
                _, src_code = self.encoder(src)
                _, dst_code = self.encoder(dst)

            # Code discriminator: dst is real, src is fake
            dst_code_pred = self.code_discriminator(dst_code)
            src_code_pred = self.code_discriminator(src_code)

            code_d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    dst_code_pred, torch.ones_like(dst_code_pred)
                )
                + F.binary_cross_entropy_with_logits(
                    src_code_pred, torch.zeros_like(src_code_pred)
                )
            )

            self.manual_backward(code_d_loss)
            code_opt.step()

            loss_dict["code_d"] = code_d_loss

        # Log losses
        for name, value in loss_dict.items():
            prog_bar = name in ["total", "d_loss", "g_adv"]
            self.log(f"train_{name}", value, prog_bar=prog_bar)

    def _training_step_temporal(
        self,
        src_seq: torch.Tensor,
        dst_seq: torch.Tensor,
        landmarks: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        dst_mask: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """
        Temporal training step with 3D discriminator.

        Trains on frame sequences for temporal consistency.

        Args:
            src_seq: Source sequence (B, C, T, H, W).
            dst_seq: Destination sequence (B, C, T, H, W).
            landmarks: Source landmarks (B, 68, 2) - Note: per-batch, not per-frame.
            src_mask: Source mask (B, 1, H, W) - optional.
            dst_mask: Destination mask (B, 1, H, W) - optional.
            batch_idx: Batch index.
        """
        # Get optimizers based on whether spatial GAN is also enabled
        optimizers_obj = self.optimizers()
        if isinstance(optimizers_obj, (list, tuple)):
            optimizers = list(optimizers_obj)
        else:
            optimizers = [optimizers_obj]
        g_opt = cast(Any, optimizers[0])
        d_opt: Any | None = None
        if self.gan_power > 0:
            d_opt = cast(Any, optimizers[1])
            t_opt = cast(Any, optimizers[2])
        else:
            t_opt = cast(Any, optimizers[1])

        B, C, T, H, W = src_seq.shape

        # Forward pass through sequence with optional gradient checkpointing
        if self.training and self.hparams.get("temporal_checkpoint", True):
            from torch.utils.checkpoint import checkpoint

            pred_seq = checkpoint(self.forward_sequence, src_seq, use_reentrant=False)
        else:
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
        assert self.temporal_consistency_loss is not None
        temp_cons_loss = self.temporal_consistency_loss(pred_seq)
        loss_dict["temp_cons"] = temp_cons_loss

        # Temporal adversarial loss
        assert self.temporal_discriminator is not None
        assert self.temporal_gan_loss is not None
        t_fake_score = self.temporal_discriminator(pred_seq)
        t_adv_loss = self.temporal_gan_loss(t_fake_score, target_is_real=True)
        loss_dict["t_adv"] = t_adv_loss

        # TV loss on predictions
        assert self.tv_loss is not None
        tv_loss = self.tv_loss(pred_flat)
        loss_dict["tv"] = tv_loss

        # Spatial GAN loss if enabled
        g_adv_loss = torch.tensor(0.0, device=pred_seq.device)
        if self.gan_power > 0 and self.discriminator is not None:
            assert self.gan_loss is not None
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
            assert d_opt is not None
            d_opt.zero_grad()

            pred_detached = pred_flat.detach()
            d_real_center, d_real_final = self.discriminator(src_flat)
            d_fake_center, d_fake_final = self.discriminator(pred_detached)

            assert self.d_loss_fn is not None
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

        assert self.temporal_d_loss_fn is not None
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
            assert self.temporal_consistency_loss is not None
            temp_cons_loss = self.temporal_consistency_loss(pred_seq)
            loss_dict["temp_cons"] = temp_cons_loss
            total_loss = total_loss + temp_cons_loss
        else:
            # Standard validation
            result = self(src)
            src_mask = src_dict.get("mask")

            # Unpack (image, mask) tuple if returned
            pred_mask: torch.Tensor | None = None
            if isinstance(result, tuple):
                pred, pred_mask = result
            else:
                pred = result

            total_loss, loss_dict = self.compute_loss(
                pred,
                src,
                src_landmarks,
                src_mask,
                pred_mask=pred_mask,
                target_mask=src_mask,
            )

        # Log validation losses
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def _get_optimizer(
        self, params: Any, betas: tuple[float, float] | None = None
    ) -> torch.optim.Optimizer:
        """Helper to create optimizer based on config.

        Args:
            params: Model parameters to optimize.
            betas: Optional beta coefficients override. If None, uses (0.9, 0.999).
        """
        betas = betas or (0.9, 0.999)
        optimizer_type = str(self.hparams.get("optimizer_type", "adamw"))
        if optimizer_type == "adabelief":
            from visagen.training.optimizers.adabelief import AdaBelief

            return AdaBelief(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
                lr_dropout=float(self.hparams.get("lr_dropout", 1.0)),
                lr_cos_period=int(self.hparams.get("lr_cos_period", 0)),
                clipnorm=float(self.hparams.get("clipnorm", 0.0)),
            )
        else:
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
            )

    def _get_max_epochs(self) -> int:
        """Resolve max epochs from trainer or fallback to 100."""
        try:
            trainer_max_epochs = self.trainer.max_epochs
        except RuntimeError:
            trainer_max_epochs = 100
        return (
            int(trainer_max_epochs)
            if trainer_max_epochs is not None and int(trainer_max_epochs) > 0
            else 100
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Build LR scheduler with optional warmup.

        All schedulers operate on epoch-level intervals. The warmup parameter
        specifies the number of warmup epochs (not steps).

        Supports three scheduler types:
        - "cosine": CosineAnnealing with optional linear warmup phase
        - "plateau": ReduceLROnPlateau monitoring val_total
        - "constant": No scheduling (returns dummy constant scheduler)

        Args:
            optimizer: The optimizer to attach the scheduler to.

        Returns:
            Scheduler config dict compatible with PyTorch Lightning.
        """
        scheduler_type = str(self.hparams.get("scheduler_type", "cosine"))
        warmup_epochs = int(self.hparams.get("warmup_epochs", 0))

        if scheduler_type == "plateau":
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "scheduler": plateau_scheduler,
                "monitor": "val_total",
                "interval": "epoch",
            }

        if scheduler_type == "constant":
            constant_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _epoch: 1.0
            )
            return {"scheduler": constant_scheduler, "interval": "epoch"}

        # Cosine (with optional warmup)
        max_epochs = self._get_max_epochs()
        cosine_scheduler: torch.optim.lr_scheduler.LRScheduler
        if warmup_epochs > 0:

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
                return float(0.5 * (1 + math.cos(math.pi * progress)))

            cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )
        return {"scheduler": cosine_scheduler, "interval": "epoch"}

    def configure_optimizers(self) -> Any:
        """
        Configure optimizer(s) and scheduler(s).

        Training modes and optimizers:
        - Temporal + GAN: 3+ optimizers (generator, spatial discriminator, temporal discriminator, [code_discriminator])
        - Temporal only: 2+ optimizers (generator, temporal discriminator, [code_discriminator])
        - GAN only: 2+ optimizers (generator, spatial discriminator, [code_discriminator])
        - GAN + true_face: 3 optimizers (generator, spatial discriminator, code_discriminator)
        - true_face only: 2 optimizers (generator, code_discriminator)
        - AE only: 1 optimizer

        Returns:
            Single optimizer dict (AE mode) or tuple of optimizer/scheduler lists (GAN/temporal mode).
        """
        # Generator optimizer - model type dependent
        if self.model_type == "diffusion":
            assert self.model is not None
            g_params = list(self.model.parameters())
        elif self.model_type == "eg3d":
            assert self.encoder is not None
            assert self.model is not None
            g_params = list(self.encoder.parameters()) + list(self.model.parameters())
        else:
            # Standard encoder-decoder
            assert self.encoder is not None
            assert self.decoder is not None
            g_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        g_optimizer = self._get_optimizer(g_params)
        g_scheduler = self._build_scheduler(g_optimizer)

        # Code discriminator optimizer (if true_face_power enabled)
        code_optimizer: torch.optim.Optimizer | None = None
        code_scheduler: dict[str, Any] | None = None
        if self.true_face_power > 0 and self.code_discriminator is not None:
            code_optimizer = self._get_optimizer(
                self.code_discriminator.parameters(), betas=self.d_betas
            )
            code_scheduler = self._build_scheduler(code_optimizer)

        if self.temporal_enabled and self.gan_power > 0:
            # Temporal + GAN: 3+ optimizers
            assert self.discriminator is not None
            d_optimizer = self._get_optimizer(
                self.discriminator.parameters(), betas=self.d_betas
            )
            d_scheduler = self._build_scheduler(d_optimizer)

            assert self.temporal_discriminator is not None
            t_optimizer = self._get_optimizer(
                self.temporal_discriminator.parameters(), betas=self.d_betas
            )
            t_scheduler = self._build_scheduler(t_optimizer)

            optimizers = [g_optimizer, d_optimizer, t_optimizer]
            schedulers = [g_scheduler, d_scheduler, t_scheduler]

            if code_optimizer is not None:
                assert code_scheduler is not None
                optimizers.append(code_optimizer)
                schedulers.append(code_scheduler)

            return (optimizers, schedulers)

        elif self.temporal_enabled:
            # Temporal only: 2+ optimizers (generator + temporal discriminator)
            assert self.temporal_discriminator is not None
            t_optimizer = self._get_optimizer(
                self.temporal_discriminator.parameters(), betas=self.d_betas
            )
            t_scheduler = self._build_scheduler(t_optimizer)

            optimizers = [g_optimizer, t_optimizer]
            schedulers = [g_scheduler, t_scheduler]

            if code_optimizer is not None:
                assert code_scheduler is not None
                optimizers.append(code_optimizer)
                schedulers.append(code_scheduler)

            return (optimizers, schedulers)

        elif self.gan_power > 0:
            # GAN only: 2+ optimizers (generator + spatial discriminator)
            assert self.discriminator is not None
            d_optimizer = self._get_optimizer(
                self.discriminator.parameters(), betas=self.d_betas
            )
            d_scheduler = self._build_scheduler(d_optimizer)

            optimizers = [g_optimizer, d_optimizer]
            schedulers = [g_scheduler, d_scheduler]

            if code_optimizer is not None:
                assert code_scheduler is not None
                optimizers.append(code_optimizer)
                schedulers.append(code_scheduler)

            return (optimizers, schedulers)

        elif self.true_face_power > 0 and code_optimizer is not None:
            # true_face only: 2 optimizers (generator + code_discriminator)
            assert code_scheduler is not None
            return (
                [g_optimizer, code_optimizer],
                [g_scheduler, code_scheduler],
            )

        else:
            # AE only: single optimizer
            return {
                "optimizer": g_optimizer,
                "lr_scheduler": g_scheduler,
            }

    def generate_preview(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_samples: int = 4,
    ) -> torch.Tensor:
        """
        Generate preview grid for visualization.

        Creates a 4-row grid showing:
        - Row 1: Source original images
        - Row 2: Source reconstructed images
        - Row 3: Destination original images
        - Row 4: Destination with swapped face

        Args:
            src: Source images tensor (B, C, H, W).
            dst: Destination images tensor (B, C, H, W).
            num_samples: Number of samples to include in grid.

        Returns:
            Preview grid tensor (C, H_grid, W_grid).

        Example:
            >>> model = TrainingModule()
            >>> src = torch.randn(4, 3, 256, 256)
            >>> dst = torch.randn(4, 3, 256, 256)
            >>> grid = model.generate_preview(src, dst, num_samples=4)
            >>> grid.shape
            torch.Size([3, 1032, 1032])  # Approximate, depends on padding
        """
        try:
            import torchvision.utils as vutils
        except ImportError:
            raise ImportError(
                "torchvision required for preview generation. "
                "Install with: pip install torchvision"
            )

        # Limit to num_samples
        actual_samples = min(src.shape[0], dst.shape[0], num_samples)
        src = src[:actual_samples]
        dst = dst[:actual_samples]

        # Store training state and switch to eval
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                src_pred = self(src)
                dst_pred = self(dst)

            # Create grid: 4 rows x num_samples cols
            grid = vutils.make_grid(
                torch.cat([src, src_pred, dst, dst_pred], dim=0),
                nrow=actual_samples,
                normalize=True,
                value_range=(0, 1),
                padding=2,
            )

            return cast(torch.Tensor, grid)

        finally:
            # Restore training mode
            if was_training:
                self.train()
