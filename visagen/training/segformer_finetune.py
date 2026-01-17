"""
SegFormer LoRA Fine-tuning Module.

Provides PyTorch Lightning module and trainer for efficient
LoRA fine-tuning of SegFormer face parsing model.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from visagen.data.mask_dataset import MaskDataModule
from visagen.vision.segmenter import DEFAULT_MODEL_PATH
from visagen.vision.segmenter_lora import LoRAConfig, SegFormerLoRA

logger = logging.getLogger(__name__)


@dataclass
class SegFormerFinetuneConfig:
    """
    Configuration for SegFormer LoRA fine-tuning.

    Attributes:
        learning_rate: Learning rate for AdamW optimizer. Default: 1e-4.
        max_epochs: Maximum training epochs. Default: 50.
        batch_size: Batch size for training. Default: 4.
        lora_rank: LoRA decomposition rank. Default: 8.
        lora_alpha: LoRA scaling factor. Default: 16.0.
        lora_dropout: LoRA dropout probability. Default: 0.1.
        ce_weight: Cross-entropy loss weight. Default: 1.0.
        dice_weight: Dice loss weight. Default: 1.0.
        val_split: Validation data fraction. Default: 0.1.
        num_workers: DataLoader workers. Default: 0.
        target_size: Training image size. Default: 512.
        num_classes: Number of output classes. Default: 2 (binary).
            Use 19 for full CelebAMask-HQ multi-class training.
        mask_mode: Training mask mode ("binary" or "multiclass"). Default: "binary".
        precision: Training precision. Default: "16-mixed".
            Options: "32" (full), "16-mixed" (FP16 AMP), "bf16-mixed" (BF16 AMP).
        early_stopping_patience: Epochs to wait before early stopping. Default: 10.
        save_top_k: Number of best checkpoints to keep. Default: 3.
        warmup_epochs: Number of epochs for learning rate warmup. Default: 3.
    """

    learning_rate: float = 1e-4
    max_epochs: int = 50
    batch_size: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    val_split: float = 0.1
    num_workers: int = 0
    target_size: int = 512
    num_classes: int = 2
    mask_mode: str = "binary"
    precision: str = "16-mixed"
    early_stopping_patience: int = 10
    save_top_k: int = 3
    warmup_epochs: int = 3


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Better at handling class imbalance and boundary learning
    compared to cross-entropy alone.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Model output logits (B, C, H, W).
            labels: Ground truth labels (B, H, W).

        Returns:
            Dice loss value.
        """
        num_classes = logits.shape[1]

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode labels
        labels_one_hot = F.one_hot(labels, num_classes)  # (B, H, W, C)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)  # (B, C, N)
        labels_flat = labels_one_hot.view(
            labels_one_hot.shape[0], labels_one_hot.shape[1], -1
        )  # (B, C, N)

        # Compute intersection and union
        intersection = (probs_flat * labels_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + labels_flat.sum(dim=2)

        # Compute Dice score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - mean Dice (loss)
        return 1.0 - dice.mean()


class SegFormerFinetuneModule(pl.LightningModule):
    """
    PyTorch Lightning module for SegFormer LoRA fine-tuning.

    Combines cross-entropy and Dice loss for optimal segmentation training.
    Only trains LoRA adapter parameters while keeping base model frozen.

    Args:
        config: Fine-tuning configuration.
        model_path: Path to base SegFormer model.

    Example:
        >>> config = SegFormerFinetuneConfig(max_epochs=50, lora_rank=8)
        >>> module = SegFormerFinetuneModule(config)
        >>> trainer = pl.Trainer(max_epochs=50)
        >>> trainer.fit(module, train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        config: SegFormerFinetuneConfig,
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        # Load base model
        model_source = self._resolve_model_source(model_path)
        self.processor = SegformerImageProcessor.from_pretrained(model_source)
        base_model = SegformerForSemanticSegmentation.from_pretrained(model_source)

        # Create LoRA configuration
        lora_config = LoRAConfig(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

        # Wrap with LoRA
        self.lora_model = SegFormerLoRA(base_model, lora_config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice_loss = DiceLoss()

    def _resolve_model_source(self, model_path: str | Path | None) -> str:
        """Resolve model source path."""
        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                return str(path)
            return str(model_path)

        if DEFAULT_MODEL_PATH.exists():
            return str(DEFAULT_MODEL_PATH)

        return "jonathandinu/face-parsing"

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through LoRA model."""
        return self.lora_model(pixel_values=pixel_values, labels=labels)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step with combined CE + Dice loss.

        Args:
            batch: Dictionary with pixel_values and labels.
            batch_idx: Batch index.

        Returns:
            Combined loss value.
        """
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Forward pass
        outputs = self(pixel_values=pixel_values)
        logits = outputs.logits

        # Upsample logits to match label size
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Compute losses
        ce_loss = self.ce_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)

        # Combined loss
        total_loss = (
            self.config.ce_weight * ce_loss + self.config.dice_weight * dice_loss
        )

        # Log losses
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_ce", ce_loss)
        self.log("train_dice", dice_loss)

        return total_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step with comprehensive metrics.

        Args:
            batch: Dictionary with pixel_values and labels.
            batch_idx: Batch index.

        Returns:
            Validation loss value.
        """
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values=pixel_values)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        ce_loss = self.ce_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)
        total_loss = (
            self.config.ce_weight * ce_loss + self.config.dice_weight * dice_loss
        )

        # Compute comprehensive metrics
        metrics = self._compute_metrics(logits, labels)

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_iou", metrics["iou_binary"], prog_bar=True)
        self.log("val_dice", metrics["dice"], prog_bar=True)
        self.log("val_pixel_acc", metrics["pixel_acc"])

        if "miou" in metrics:
            self.log("val_miou", metrics["miou"], prog_bar=True)

        return total_loss

    def _compute_iou(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean Intersection over Union (binary foreground)."""
        preds = logits.argmax(dim=1)

        # Binary mask comparison (foreground vs background)
        pred_fg = (preds > 0).float()
        label_fg = (labels > 0).float()

        intersection = (pred_fg * label_fg).sum()
        union = pred_fg.sum() + label_fg.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def _compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Compute comprehensive segmentation metrics."""
        preds = logits.argmax(dim=1)

        metrics = {}

        # Binary IoU (backward compat)
        pred_fg = (preds > 0).float()
        label_fg = (labels > 0).float()
        intersection = (pred_fg * label_fg).sum()
        union = pred_fg.sum() + label_fg.sum() - intersection
        metrics["iou_binary"] = float((intersection + 1e-6) / (union + 1e-6))

        # Per-class IoU (for multiclass)
        num_classes = logits.shape[1]
        if num_classes > 2:
            class_ious = []
            for c in range(num_classes):
                pred_c = (preds == c).float()
                label_c = (labels == c).float()
                inter = (pred_c * label_c).sum()
                uni = pred_c.sum() + label_c.sum() - inter
                if uni > 0:
                    class_ious.append(float((inter + 1e-6) / (uni + 1e-6)))
            metrics["miou"] = np.mean(class_ious) if class_ious else 0.0

        # Dice score
        dice_inter = (pred_fg * label_fg).sum() * 2
        dice_union = pred_fg.sum() + label_fg.sum()
        metrics["dice"] = float((dice_inter + 1e-6) / (dice_union + 1e-6))

        # Pixel accuracy
        correct = (preds == labels).float().sum()
        total = labels.numel()
        metrics["pixel_acc"] = float(correct / total)

        return metrics

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer with warmup + cosine annealing scheduler."""
        # Only optimize LoRA parameters
        optimizer = AdamW(
            self.lora_model.get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Warmup + Cosine annealing
        warmup_epochs = self.config.warmup_epochs
        total_epochs = self.config.max_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def save_lora_weights(self, path: Path | str) -> None:
        """Save LoRA weights to file."""
        self.lora_model.save_lora_weights(path)

    def get_trainable_param_count(self) -> int:
        """Get count of trainable parameters."""
        return self.lora_model.get_trainable_param_count()


class SegFormerTrainer:
    """
    High-level trainer API for SegFormer LoRA fine-tuning.

    Provides a simple interface for training with progress callbacks
    and automatic checkpoint management.

    Args:
        config: Fine-tuning configuration.
        model_path: Optional path to base model.

    Example:
        >>> trainer = SegFormerTrainer(SegFormerFinetuneConfig())
        >>> output_path = trainer.train(
        ...     samples_dir="./training_samples",
        ...     output_dir="./lora_output",
        ... )
    """

    def __init__(
        self,
        config: SegFormerFinetuneConfig | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.config = config or SegFormerFinetuneConfig()
        self.model_path = model_path
        self._trainer: pl.Trainer | None = None
        self._module: SegFormerFinetuneModule | None = None
        self._is_training = False

    def train(
        self,
        samples_dir: Path | str,
        output_dir: Path | str,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> Path:
        """
        Train LoRA adapters on provided samples.

        Args:
            samples_dir: Directory containing training samples.
            output_dir: Directory for output weights.
            progress_callback: Optional callback(epoch, max_epochs, loss).

        Returns:
            Path to saved LoRA weights file.
        """
        samples_dir = Path(samples_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create module
        self._module = SegFormerFinetuneModule(
            config=self.config,
            model_path=self.model_path,
        )

        logger.info(
            f"Training with {self._module.get_trainable_param_count():,} "
            f"trainable parameters (LoRA rank={self.config.lora_rank})"
        )

        # Create data module
        data_module = MaskDataModule(
            samples_dir=samples_dir,
            processor=self._module.processor,
            batch_size=self.config.batch_size,
            val_split=self.config.val_split,
            num_workers=self.config.num_workers,
            target_size=self.config.target_size,
        )
        data_module.setup()

        # Create callbacks
        callbacks = []

        if progress_callback is not None:
            callbacks.append(
                _ProgressCallback(progress_callback, self.config.max_epochs)
            )

        # Early stopping
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop)

        # Model checkpointing
        checkpoint = ModelCheckpoint(
            dirpath=output_dir,
            filename="segformer-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=self.config.save_top_k,
            save_last=True,
        )
        callbacks.append(checkpoint)

        # Configure trainer
        self._trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            precision=self.config.precision,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=progress_callback is None,
            logger=False,
        )

        # Train
        self._is_training = True
        try:
            self._trainer.fit(
                self._module,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader(),
            )
        finally:
            self._is_training = False

        # Save LoRA weights
        weights_path = output_dir / "segformer_lora.pt"
        self._module.save_lora_weights(weights_path)

        logger.info(f"LoRA weights saved to: {weights_path}")

        return weights_path

    def stop(self) -> None:
        """Stop training if in progress."""
        if self._trainer is not None and self._is_training:
            self._trainer.should_stop = True

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training


class _ProgressCallback(pl.Callback):
    """Internal callback for progress reporting."""

    def __init__(
        self,
        callback: Callable[[int, int, float], None],
        max_epochs: int,
    ) -> None:
        self.callback = callback
        self.max_epochs = max_epochs
        self._last_loss = 0.0

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at end of each training epoch."""
        epoch = trainer.current_epoch + 1
        loss = trainer.callback_metrics.get("train_loss", 0.0)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self._last_loss = loss

        self.callback(epoch, self.max_epochs, loss)


def quick_finetune(
    samples_dir: Path | str,
    output_dir: Path | str,
    epochs: int = 50,
    rank: int = 8,
    learning_rate: float = 1e-4,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    """
    Quick fine-tuning function for simple use cases.

    Args:
        samples_dir: Directory containing training samples.
        output_dir: Directory for output weights.
        epochs: Number of training epochs. Default: 50.
        rank: LoRA rank. Default: 8.
        learning_rate: Learning rate. Default: 1e-4.
        progress_callback: Optional progress callback.

    Returns:
        Path to saved LoRA weights file.
    """
    config = SegFormerFinetuneConfig(
        max_epochs=epochs,
        lora_rank=rank,
        learning_rate=learning_rate,
    )

    trainer = SegFormerTrainer(config)
    return trainer.train(
        samples_dir=samples_dir,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
