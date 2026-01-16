"""
Training callbacks for Visagen.

Includes:
- PreviewCallback: Generate training previews at intervals
- AutoBackupCallback: Automatic checkpoint backups with rotation
- TargetStepCallback: Stop training at target step or loss
- CommandFileReaderCallback: Read and apply live parameter updates from JSON file
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from visagen.utils.io import read_json_locked

try:
    import torchvision.utils as vutils

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)


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
                step_path = self.preview_dir / f"step_{trainer.global_step:08d}.png"
                vutils.save_image(grid, step_path)

            # Save metadata
            current_loss = self.loss_history[-1]["loss"] if self.loss_history else None
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

        except Exception as e:
            # Don't crash training on preview errors, but log them
            logger.warning(
                f"Preview generation failed at step {trainer.global_step}: {e}"
            )

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


class AutoBackupCallback(Callback):
    """
    Automatically backup checkpoints at regular intervals with rotation.

    Creates rotating backups in: output_dir/backups/01, 02, ..., max_backups
    Each backup contains the full checkpoint and metadata.

    The rotation works like this:
    - When a new backup is created, existing backups shift up (01 -> 02, etc.)
    - The oldest backup (max_backups) is deleted
    - New backup is saved to slot 01

    Args:
        backup_dir: Directory to save backups.
        interval_steps: Create backup every N steps. Default: 5000.
        interval_minutes: Alternative: backup every N minutes. Default: None.
        max_backups: Maximum number of backups to keep. Default: 5.

    Example:
        >>> from visagen.training.callbacks import AutoBackupCallback
        >>> callback = AutoBackupCallback(
        ...     backup_dir="./workspace/model/backups",
        ...     interval_steps=5000,
        ...     max_backups=5,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        backup_dir: str | Path,
        interval_steps: int = 5000,
        interval_minutes: int | None = None,
        max_backups: int = 5,
    ) -> None:
        super().__init__()
        self.backup_dir = Path(backup_dir)
        self.interval_steps = interval_steps
        self.interval_minutes = interval_minutes
        self.max_backups = max_backups

        self._last_backup_step = 0
        self._last_backup_time: float | None = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Create backup directory on training start."""
        if stage == "fit":
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self._last_backup_time = time.time()
            # Initialize last backup step from trainer if resuming
            self._last_backup_step = trainer.global_step

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        """Check if backup is needed and create it."""
        should_backup = False

        # Step-based backup
        if self.interval_steps > 0:
            steps_since_backup = trainer.global_step - self._last_backup_step
            if steps_since_backup >= self.interval_steps:
                should_backup = True

        # Time-based backup (optional)
        if self.interval_minutes and self._last_backup_time:
            elapsed_minutes = (time.time() - self._last_backup_time) / 60
            if elapsed_minutes >= self.interval_minutes:
                should_backup = True

        if should_backup and trainer.global_step > 0:
            self._create_backup(trainer)
            self._last_backup_step = trainer.global_step
            self._last_backup_time = time.time()

    def _create_backup(self, trainer: pl.Trainer) -> None:
        """Create backup with rotation."""
        # Rotate existing backups (max -> delete, max-1 -> max, ... 01 -> 02)
        for i in range(self.max_backups, 0, -1):
            current_path = self.backup_dir / f"{i:02d}"
            next_path = self.backup_dir / f"{i + 1:02d}"

            if current_path.exists():
                if i == self.max_backups:
                    # Delete oldest backup
                    shutil.rmtree(current_path)
                else:
                    # Move to next slot
                    current_path.rename(next_path)

        # Save new backup to slot 01
        slot_path = self.backup_dir / "01"
        slot_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = slot_path / f"backup_step_{trainer.global_step}.ckpt"

        try:
            trainer.save_checkpoint(str(checkpoint_path))

            # Save metadata
            metadata = {
                "step": trainer.global_step,
                "epoch": trainer.current_epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(slot_path / "backup_info.json", "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            # Don't crash training on backup errors, but log them
            logger.error(f"Backup failed at step {trainer.global_step}: {e}")


class TargetStepCallback(Callback):
    """
    Stop training when target step count or loss value is reached.

    This callback monitors training progress and gracefully stops training
    when the specified conditions are met.

    Args:
        target_step: Stop training at this step. 0 = unlimited. Default: 0.
        target_loss: Stop when loss reaches this value. None = disabled.

    Example:
        >>> from visagen.training.callbacks import TargetStepCallback
        >>> # Stop at 100000 steps
        >>> callback = TargetStepCallback(target_step=100000)
        >>> # Or stop when loss reaches 0.05
        >>> callback = TargetStepCallback(target_loss=0.05)
        >>> # Or both (whichever comes first)
        >>> callback = TargetStepCallback(target_step=100000, target_loss=0.05)
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        target_step: int = 0,
        target_loss: float | None = None,
    ) -> None:
        super().__init__()
        self.target_step = target_step
        self.target_loss = target_loss
        self._current_loss: float | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        """Check if target conditions are met."""
        # Extract current loss from outputs
        self._current_loss = self._extract_loss(outputs)

        # Check target step
        if self.target_step > 0 and trainer.global_step >= self.target_step:
            print(f"\nTarget step {self.target_step} reached. Stopping training.")
            trainer.should_stop = True
            return

        # Check target loss
        if self.target_loss is not None and self._current_loss is not None:
            if self._current_loss <= self.target_loss:
                print(
                    f"\nTarget loss {self.target_loss} reached "
                    f"(current: {self._current_loss:.4f}). Stopping training."
                )
                trainer.should_stop = True

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


class CommandFileReaderCallback(Callback):
    """
    Read and apply live parameter updates from a JSON command file.

    This callback enables real-time parameter adjustment during training by
    polling a JSON file for changes. When the file is modified, the callback
    reads new parameter values and applies them to the training module.

    The command file format:
        {
            "params": {
                "learning_rate": 0.0001,
                "gan_power": 0.1,
                "dssim_weight": 10.0,
                ...
            }
        }

    Supported parameters:
        - learning_rate: Updates optimizer learning rate
        - gan_power, true_face_power: GAN-related weights
        - face_style_weight, bg_style_weight: Style transfer weights
        - dssim_weight, l1_weight, lpips_weight: Reconstruction loss weights
        - eyes_mouth_weight, gaze_weight, texture_weight: Feature-specific weights

    Args:
        command_file: Path to the JSON command file. Default: "cmd_training.json".
        check_interval: Check for updates every N training steps. Default: 10.

    Example:
        >>> from visagen.training.callbacks import CommandFileReaderCallback
        >>> callback = CommandFileReaderCallback(
        ...     command_file="./workspace/model/cmd_training.json",
        ...     check_interval=10,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    # Parameters that can be updated at runtime
    UPDATABLE_PARAMS = frozenset(
        {
            "learning_rate",
            "gan_power",
            "true_face_power",
            "face_style_weight",
            "bg_style_weight",
            "dssim_weight",
            "l1_weight",
            "lpips_weight",
            "eyes_mouth_weight",
            "gaze_weight",
            "texture_weight",
            "id_weight",
            "temporal_power",
            "temporal_consistency_weight",
        }
    )

    def __init__(
        self,
        command_file: str | Path = "cmd_training.json",
        check_interval: int = 10,
    ) -> None:
        super().__init__()
        self.command_file = Path(command_file)
        self.check_interval = check_interval
        self._last_check_step = 0
        self._last_mtime: float | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        """Check for command file updates and apply parameter changes."""
        # Only check at specified interval
        if trainer.global_step - self._last_check_step < self.check_interval:
            return

        self._last_check_step = trainer.global_step

        # Skip if command file doesn't exist
        if not self.command_file.exists():
            return

        # Check if file was modified since last read
        try:
            current_mtime = self.command_file.stat().st_mtime
        except OSError:
            return

        if self._last_mtime is not None and current_mtime <= self._last_mtime:
            return

        self._last_mtime = current_mtime

        # Read and apply parameters
        self._apply_params(trainer, pl_module)

    def _apply_params(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Read command file and apply parameter updates."""
        data = read_json_locked(self.command_file)
        if data is None:
            return

        params = data.get("params", {})
        if not params:
            return

        for key, value in params.items():
            if key not in self.UPDATABLE_PARAMS:
                logger.warning(f"Unknown parameter '{key}' in command file, skipping")
                continue

            try:
                float_value = float(value)
            except (TypeError, ValueError):
                logger.warning(f"Invalid value for '{key}': {value}")
                continue

            if key == "learning_rate":
                self._update_learning_rate(trainer, pl_module, float_value)
            else:
                self._update_module_param(trainer, pl_module, key, float_value)

    def _update_learning_rate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        new_lr: float,
    ) -> None:
        """Update learning rate across all optimizer param groups."""
        optimizers = pl_module.optimizers()
        if optimizers is None:
            return

        # Handle single optimizer or list of optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for opt in optimizers:
            for param_group in opt.param_groups:
                old_lr = param_group.get("lr", 0)
                if abs(old_lr - new_lr) > 1e-9:
                    param_group["lr"] = new_lr

        logger.info(f"[Step {trainer.global_step}] Learning rate updated to {new_lr}")

    def _update_module_param(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        key: str,
        value: float,
    ) -> None:
        """Update a parameter on the training module."""
        if hasattr(pl_module, key):
            old_value = getattr(pl_module, key, None)
            if old_value != value:
                setattr(pl_module, key, value)
                logger.info(f"[Step {trainer.global_step}] {key} updated to {value}")
        else:
            logger.warning(f"Module does not have attribute '{key}', cannot update")
