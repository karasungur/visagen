"""
DFL Lightning Module for Visagen.

Main training module that combines encoder and decoder,
manages the training loop, and handles optimization.
Supports optional GAN training with PatchGAN discriminator.
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
        use_spectral_norm: Use spectral normalization in discriminator. Default: False.

    Example:
        >>> module = DFLModule()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = module(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])

        # With GAN training:
        >>> module = DFLModule(gan_power=0.1, gan_patch_size=70)
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
        use_multiscale_dssim: bool = True,
        # GAN parameters
        gan_power: float = 0.0,
        gan_patch_size: int = 70,
        gan_mode: str = "vanilla",
        gan_base_ch: int = 16,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

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

        # Build encoder
        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            dims=encoder_dims,
            depths=encoder_depths,
            drop_path_rate=drop_path_rate,
        )

        # Calculate skip connection dimensions
        # Skip features are from encoder stages (reversed for decoder)
        skip_dims = encoder_dims[:-1][::-1] + [encoder_dims[0]]

        # Build decoder
        self.decoder = Decoder(
            latent_channels=encoder_dims[-1],
            dims=decoder_dims,
            skip_dims=skip_dims,
            out_channels=in_channels,
            use_attention=True,
        )

        # Build loss functions
        self._init_losses(
            dssim_weight=dssim_weight,
            l1_weight=l1_weight,
            lpips_weight=lpips_weight,
            id_weight=id_weight,
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

    def _init_losses(
        self,
        dssim_weight: float,
        l1_weight: float,
        lpips_weight: float,
        id_weight: float,
        use_multiscale_dssim: bool,
    ) -> None:
        """Initialize loss functions."""
        self.dssim_weight = dssim_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.id_weight = id_weight

        # DSSIM loss
        if use_multiscale_dssim:
            self.dssim_loss = MultiScaleDSSIMLoss()
        else:
            self.dssim_loss = DSSIMLoss()

        # LPIPS loss (lazy loaded)
        self._lpips_loss = None

        # ID loss (lazy loaded)
        self._id_loss = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        # Encode
        features, latent = self.encoder(x)

        # Prepare skip connections (reverse order for decoder)
        # features: [stage0, stage1, stage2, stage3]
        # decoder needs: [stage2, stage1, stage0, stage0] (deep to shallow)
        skip_features = features[:-1][::-1] + [features[0]]

        # Decode
        output = self.decoder(latent, skip_features)

        return output

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined reconstruction loss.

        Args:
            pred: Predicted image (B, C, H, W).
            target: Target image (B, C, H, W).

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

        losses["total"] = total
        return total, losses

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        """
        Training step with optional GAN training.

        When GAN is enabled (gan_power > 0):
        1. Generator step: reconstruction loss + adversarial loss
        2. Discriminator step: real vs fake classification

        Args:
            batch: Tuple of (source, target) tensors.
            batch_idx: Batch index.

        Returns:
            Loss value (only when GAN is disabled).
        """
        src, dst = batch

        if self.gan_power > 0:
            return self._training_step_gan(src, dst, batch_idx)
        else:
            return self._training_step_ae(src, dst, batch_idx)

    def _training_step_ae(
        self, src: torch.Tensor, dst: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Standard autoencoder training step."""
        # Forward pass (reconstruct source from source)
        pred = self(src)

        # Compute losses
        total_loss, loss_dict = self.compute_loss(pred, src)

        # Log all losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def _training_step_gan(
        self, src: torch.Tensor, dst: torch.Tensor, batch_idx: int
    ) -> None:
        """GAN training step with manual optimization."""
        g_opt, d_opt = self.optimizers()

        # Forward pass
        pred = self(src)

        # === GENERATOR STEP ===
        g_opt.zero_grad()

        # Reconstruction loss
        total_loss, loss_dict = self.compute_loss(pred, src)

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

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Tuple of (source, target) tensors.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        src, dst = batch
        pred = self(src)

        total_loss, loss_dict = self.compute_loss(pred, src)

        # Log validation losses
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def configure_optimizers(self) -> dict[str, Any] | tuple[list, list]:
        """
        Configure optimizer(s) and scheduler(s).

        When GAN is enabled, returns two optimizers:
        - Generator optimizer (encoder + decoder)
        - Discriminator optimizer

        Returns:
            Single optimizer dict (AE mode) or tuple of optimizer/scheduler lists (GAN mode).
        """
        max_epochs = self.trainer.max_epochs if self.trainer else 100

        if self.gan_power > 0:
            # Generator optimizer (encoder + decoder)
            g_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            g_optimizer = torch.optim.AdamW(
                g_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )

            # Discriminator optimizer
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )

            # Schedulers
            g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                g_optimizer, T_max=max_epochs, eta_min=1e-6
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
            # Standard single optimizer
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
