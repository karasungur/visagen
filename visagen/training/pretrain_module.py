"""
Pretrain Lightning Module for Visagen.

Self-supervised pretraining module for face reconstruction.
Trains on large generic face datasets (FFHQ, CelebA) before fine-tuning.
"""

from pathlib import Path
from typing import Any

import torch

from visagen.training.dfl_module import DFLModule


class PretrainModule(DFLModule):
    """
    Pretrain Lightning Module for self-supervised face reconstruction.

    Extends DFLModule with pretrain-specific behavior:
    - Self-reconstruction training (encode-decode same face)
    - GAN disabled by default (gan_power=0)
    - LPIPS/ID loss disabled for faster training
    - Optimized loss weights for generic face learning

    This module trains on large face datasets to learn general face
    reconstruction. The pretrained weights can then be loaded into
    DFLModule for fine-tuning on specific face pairs.

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
        lpips_weight: Weight for LPIPS loss. Default: 0.0 (disabled).
        id_weight: Weight for identity loss. Default: 0.0 (disabled).
        use_multiscale_dssim: Use multi-scale DSSIM. Default: True.

    Example:
        >>> module = PretrainModule()
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = module(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])

        # Training on FFHQ:
        >>> from visagen.data.pretrain_datamodule import PretrainDataModule
        >>> datamodule = PretrainDataModule(data_dir="/data/ffhq")
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
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
        # Pretrain loss weights (LPIPS and ID disabled)
        dssim_weight: float = 10.0,
        l1_weight: float = 10.0,
        lpips_weight: float = 0.0,  # Disabled for pretrain
        id_weight: float = 0.0,  # Disabled for pretrain
        use_multiscale_dssim: bool = True,
    ) -> None:
        # Initialize parent with GAN disabled (gan_power=0)
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            encoder_dims=encoder_dims,
            encoder_depths=encoder_depths,
            decoder_dims=decoder_dims,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            drop_path_rate=drop_path_rate,
            dssim_weight=dssim_weight,
            l1_weight=l1_weight,
            lpips_weight=lpips_weight,
            id_weight=id_weight,
            use_multiscale_dssim=use_multiscale_dssim,
            # GAN disabled for pretrain
            gan_power=0.0,
            gan_patch_size=70,
            gan_mode="vanilla",
            gan_base_ch=16,
            use_spectral_norm=False,
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Self-reconstruction training step.

        In pretrain mode, src and target are the same image.
        We encode the image, decode it, and compute reconstruction loss.

        Args:
            batch: Tuple of (src, target) tensors (identical in pretrain).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        src, target = batch  # In pretrain: src == target

        # Forward pass: encode and decode
        pred = self(src)

        # Compute reconstruction losses
        total_loss, loss_dict = self.compute_loss(pred, target)

        # Log all losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Tuple of (src, target) tensors.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        src, target = batch
        pred = self(src)

        total_loss, loss_dict = self.compute_loss(pred, target)

        # Log validation losses
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, prog_bar=(name == "total"))

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and scheduler for pretrain.

        Uses AdamW with CosineAnnealing scheduler.
        Single optimizer since GAN is disabled.

        Returns:
            Optimizer and scheduler configuration dict.
        """
        max_epochs = self.trainer.max_epochs if self.trainer else 100

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

    @classmethod
    def load_for_finetune(
        cls,
        checkpoint_path: str | Path,
        strict: bool = False,
        **override_kwargs,
    ) -> "DFLModule":
        """
        Load pretrained weights for fine-tuning.

        Loads encoder/decoder weights from pretrain checkpoint and
        creates a DFLModule ready for fine-tuning. Optimizer state
        is discarded, and training starts fresh from epoch 0.

        Args:
            checkpoint_path: Path to pretrained .ckpt file.
            strict: Require exact key matching. Default: False.
            **override_kwargs: Override hyperparameters for fine-tune.

        Returns:
            DFLModule with pretrained weights loaded.

        Example:
            >>> # Load pretrained model for fine-tuning
            >>> model = PretrainModule.load_for_finetune(
            ...     "pretrain/checkpoints/last.ckpt",
            ...     learning_rate=5e-5,
            ...     gan_power=0.1,
            ... )
            >>> # Now fine-tune on specific faces
            >>> trainer.fit(model, face_datamodule)
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Get hyperparameters from checkpoint
        hparams = checkpoint.get("hyper_parameters", {})

        # Override with provided kwargs
        hparams.update(override_kwargs)

        # Remove pretrain-specific settings that shouldn't carry over
        hparams.pop("lpips_weight", None)
        hparams.pop("id_weight", None)

        # Set reasonable defaults for fine-tuning if not provided
        if "lpips_weight" not in override_kwargs:
            hparams["lpips_weight"] = 0.0
        if "id_weight" not in override_kwargs:
            hparams["id_weight"] = 0.0
        if "gan_power" not in override_kwargs:
            hparams["gan_power"] = 0.0

        # Create new DFLModule with loaded hyperparameters
        model = DFLModule(**hparams)

        # Load state dict (weights only)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if missing:
            print(f"Note: Missing keys (expected for fine-tune): {len(missing)}")
        if unexpected:
            print(f"Note: Unexpected keys (will be ignored): {len(unexpected)}")

        return model

    @staticmethod
    def extract_weights(
        checkpoint_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Extract model weights from checkpoint (without optimizer state).

        Useful for creating smaller checkpoint files for distribution.

        Args:
            checkpoint_path: Path to full .ckpt file.
            output_path: Optional path to save weights-only file.

        Returns:
            State dict with model weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract only model weights
        state_dict = checkpoint.get("state_dict", checkpoint)
        hparams = checkpoint.get("hyper_parameters", {})

        weights_only = {
            "state_dict": state_dict,
            "hyper_parameters": hparams,
        }

        if output_path is not None:
            torch.save(weights_only, output_path)
            print(f"Saved weights to: {output_path}")

        return weights_only
