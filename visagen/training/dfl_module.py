"""
DFL Lightning Module for Visagen.

Main training module that combines encoder and decoder,
manages the training loop, and handles optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List

from visagen.models.encoders.convnext import ConvNeXtEncoder
from visagen.models.decoders.decoder import Decoder
from visagen.training.losses import DSSIMLoss, MultiScaleDSSIMLoss, CombinedLoss


class DFLModule(pl.LightningModule):
    """
    DeepFaceLab Lightning Module.

    Combines ConvNeXt encoder and decoder for face swapping training.
    Supports multiple loss functions including DSSIM, LPIPS, and ID loss.

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

    Example:
        >>> module = DFLModule()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = module(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        encoder_dims: List[int] = None,
        encoder_depths: List[int] = None,
        decoder_dims: List[int] = None,
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (source, target) tensors.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        src, dst = batch

        # Forward pass (reconstruct source from source)
        pred = self(src)

        # Compute losses
        total_loss, loss_dict = self.compute_loss(pred, src)

        # Log all losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
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

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and scheduler.

        Returns:
            Dict with optimizer and lr_scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
