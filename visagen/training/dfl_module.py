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


class DFLModule(pl.LightningModule):
    """
    DeepFaceLab Lightning Module.

    Combines ConvNeXt encoder and decoder for face swapping training.
    Uses reconstruction loss for initial skeleton testing.

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

        # For skeleton testing: reconstruct source from source
        # (Real training will use src->dst mapping)
        output = self(src)

        # Reconstruction loss (L1)
        loss = F.l1_loss(output, src)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        return loss

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
        output = self(src)
        loss = F.l1_loss(output, src)

        self.log("val_loss", loss, prog_bar=True)

        return loss

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
