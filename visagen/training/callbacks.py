"""
Training callbacks for Visagen.

Includes PreviewCallback for generating training previews during training.
"""

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

try:
    import torchvision.utils as vutils

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class PreviewCallback(Callback):
    """
    Generate and save training previews at regular intervals.

    Creates a grid showing:
    - Row 1: Source images (original)
    - Row 2: Source reconstructed
    - Row 3: Destination images (original)
    - Row 4: Destination reconstructed (face swap result)

    The preview is saved to disk as `latest.png` and optionally logged to
    TensorBoard. Metadata (step, epoch, loss) is saved to `latest.json`.

    Args:
        preview_dir: Directory to save previews.
        interval: Generate preview every N training steps. Default: 500.
        num_samples: Number of samples to show in preview grid. Default: 4.
        save_history: Keep historical previews (step_XXXXX.png). Default: False.

    Example:
        >>> from visagen.training.callbacks import PreviewCallback
        >>> callback = PreviewCallback(
        ...     preview_dir="./workspace/model/previews",
        ...     interval=500,
        ...     num_samples=4,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        preview_dir: str | Path,
        interval: int = 500,
        num_samples: int = 4,
        save_history: bool = False,
    ) -> None:
        super().__init__()
        self.preview_dir = Path(preview_dir)
        self.interval = interval
        self.num_samples = num_samples
        self.save_history = save_history

        # Loss history tracking
        self.loss_history: list[dict[str, float]] = []
        self._last_batch: tuple | None = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Create preview directory on training start."""
        if stage == "fit":
            self.preview_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        """Store last batch and generate preview at intervals."""
        self._last_batch = batch

        # Track loss from outputs
        loss_value = self._extract_loss(outputs)
        if loss_value is not None:
            step_loss = {"step": trainer.global_step, "loss": loss_value}
            self.loss_history.append(step_loss)
            # Keep only last 1000 entries to avoid memory bloat
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]

        # Generate preview at interval
        if trainer.global_step > 0 and trainer.global_step % self.interval == 0:
            self._generate_preview(trainer, pl_module)

    def _extract_loss(self, outputs: Any) -> float | None:
        """Extract loss value from training step outputs."""
        if outputs is None:
            return None

        # Handle dict output
        if isinstance(outputs, dict):
            if "loss" in outputs:
                return float(outputs["loss"])
            if "total" in outputs:
                return float(outputs["total"])

        # Handle tensor output
        if isinstance(outputs, torch.Tensor):
            return float(outputs)

        return None

    def _generate_preview(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate and save preview image."""
        if self._last_batch is None:
            return

        if not TORCHVISION_AVAILABLE:
            return

        try:
            src_dict, dst_dict = self._last_batch
            src = src_dict["image"][: self.num_samples]
            dst = dst_dict["image"][: self.num_samples]
        except (KeyError, TypeError, IndexError):
            # Batch format unexpected, skip preview
            return

        # Ensure we have enough samples
        actual_samples = min(src.shape[0], dst.shape[0], self.num_samples)
        if actual_samples == 0:
            return

        src = src[:actual_samples]
        dst = dst[:actual_samples]

        # Switch to eval mode for inference
        was_training = pl_module.training
        pl_module.eval()

        try:
            with torch.no_grad():
                src_pred = pl_module(src)
                dst_pred = pl_module(dst)

            # Create grid: 4 rows x num_samples cols
            # Row 1: Source original
            # Row 2: Source reconstructed
            # Row 3: Destination original
            # Row 4: Destination swapped (face swap result)
            grid = vutils.make_grid(
                torch.cat([src, src_pred, dst, dst_pred], dim=0),
                nrow=actual_samples,
                normalize=True,
                value_range=(0, 1),
                padding=2,
            )

            # Save to disk
            latest_path = self.preview_dir / "latest.png"
            vutils.save_image(grid, latest_path)

            # Save history if enabled
            if self.save_history:
                step_path = (
                    self.preview_dir / f"step_{trainer.global_step:08d}.png"
                )
                vutils.save_image(grid, step_path)

            # Save metadata
            current_loss = (
                self.loss_history[-1]["loss"] if self.loss_history else None
            )
            metadata = {
                "step": trainer.global_step,
                "epoch": trainer.current_epoch,
                "loss": current_loss,
                "loss_history": self.loss_history[-100:],  # Keep last 100
                "num_samples": actual_samples,
            }
            with open(self.preview_dir / "latest.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Log to TensorBoard
            if trainer.logger is not None:
                try:
                    trainer.logger.experiment.add_image(
                        "preview/training",
                        grid,
                        trainer.global_step,
                    )
                except AttributeError:
                    # Logger doesn't support add_image (e.g., CSVLogger)
                    pass

        except Exception:
            # Don't crash training on preview errors
            pass

        finally:
            # Restore training mode
            if was_training:
                pl_module.train()

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate final preview at training end."""
        if self._last_batch is not None:
            self._generate_preview(trainer, pl_module)
