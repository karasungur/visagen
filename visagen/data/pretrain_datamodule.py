"""
Pretrain Data Module for PyTorch Lightning.

Provides DataModule for pretraining on large face datasets (FFHQ, CelebA, etc.)
with self-reconstruction training (input = target).
"""

from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

pl: Any
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from visagen.data.augmentations import FaceAugmentationPipeline  # noqa: E402

# Pretrain-specific augmentation config
PRETRAIN_AUGMENTATION_CONFIG = {
    "random_flip_prob": 0.5,  # random_flips=True
    "random_warp": False,
    "rotation_range": (-10, 10),
    "scale_range": (-0.05, 0.05),
    "translation_range": (-0.05, 0.05),
    "hsv_shift_amount": 0.1,
    "brightness_range": 0.1,
    "contrast_range": 0.1,
}


class PretrainDataset(Dataset):
    """
    Dataset for pretraining with self-reconstruction.

    Returns (image, image) pairs where input and target are the same.
    This is used for autoencoder pretraining on large face datasets.

    Args:
        image_paths: List of image file paths.
        target_size: Output image size. Default: 256.
        transform: Optional augmentation transform.

    Example:
        >>> dataset = PretrainDataset(image_paths, target_size=256)
        >>> src, target = dataset[0]
        >>> src.shape
        torch.Size([3, 256, 256])
    """

    def __init__(
        self,
        image_paths: list[Path],
        target_size: int = 256,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample for self-reconstruction training.

        Returns:
            Tuple of (image, image) where both are the same image.
            Images are in [-1, 1] range with shape (C, H, W).
        """
        path = self.image_paths[idx]

        # Load image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            # Return black image if load fails (will be filtered in training)
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        else:
            image = image.astype(np.float32) / 255.0

        # Resize to target size
        h, w = image.shape[:2]
        if h != self.target_size or w != self.target_size:
            image = cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_CUBIC,
            )

        # BGR to RGB, HWC to CHW
        image = image[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        # Apply augmentation if provided
        if self.transform is not None:
            image_tensor, _ = self.transform(image_tensor, None)

        # Normalize to [-1, 1]
        if image_tensor.max() <= 1.0:
            image_tensor = image_tensor * 2 - 1

        # Return same image for both src and target (self-reconstruction)
        return image_tensor, image_tensor.clone()


def scan_images_recursive(root_dir: Path) -> list[Path]:
    """
    Recursively scan directory for image files.

    Supports FFHQ-style nested folder structure (00000/, 01000/, etc.)
    and flat CelebA-style structure.

    Args:
        root_dir: Root directory to scan.

    Returns:
        Sorted list of image file paths.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths: list[Path] = []

    for ext in extensions:
        paths.extend(root_dir.rglob(f"*{ext}"))
        paths.extend(root_dir.rglob(f"*{ext.upper()}"))

    return sorted(paths)


def scan_images_flat(root_dir: Path) -> list[Path]:
    """
    Scan directory for image files (non-recursive).

    Args:
        root_dir: Directory to scan.

    Returns:
        Sorted list of image file paths.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths: list[Path] = []

    for ext in extensions:
        paths.extend(root_dir.glob(f"*{ext}"))
        paths.extend(root_dir.glob(f"*{ext.upper()}"))

    return sorted(paths)


if pl is not None:

    class PretrainDataModule(pl.LightningDataModule):
        """
        Lightning DataModule for pretraining on large face datasets.

        Manages dataset loading with proper train/val splits,
        pretrain-specific augmentation, and DataLoader configuration.

        Supports:
        - FFHQ nested folder structure (00000/, 01000/, ...)
        - CelebA flat folder structure
        - Self-reconstruction training (input = target)
        - Pretrain-specific augmentation settings (no warp, flips enabled)

        Args:
            data_dir: Path to dataset directory.
            batch_size: Training batch size. Default: 16.
            num_workers: DataLoader workers. Default: 4.
            target_size: Output image size. Default: 256.
            val_split: Validation split ratio. Default: 0.05.
            recursive: Scan subdirectories recursively. Default: True.
            augmentation_config: Dict of augmentation parameters. Default: None.
            pin_memory: Pin memory for GPU. Default: True.

        Example:
            >>> datamodule = PretrainDataModule(
            ...     data_dir=Path("/data/ffhq/images1024x1024"),
            ...     batch_size=32,
            ...     recursive=True,
            ... )
            >>> datamodule.setup()
            >>> train_loader = datamodule.train_dataloader()
        """

        def __init__(
            self,
            data_dir: str | Path,
            batch_size: int = 16,
            num_workers: int = 4,
            target_size: int = 256,
            val_split: float = 0.05,
            recursive: bool = True,
            augmentation_config: dict | None = None,
            pin_memory: bool = True,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()

            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.target_size = target_size
            self.val_split = val_split
            self.recursive = recursive
            self.pin_memory = pin_memory

            # Use pretrain-specific augmentation config by default
            self.aug_config = augmentation_config or PRETRAIN_AUGMENTATION_CONFIG.copy()

            # Will be set in setup()
            self.train_dataset: Dataset | None = None
            self.val_dataset: Dataset | None = None
            self._image_paths: list[Path] = []

        def prepare_data(self) -> None:
            """Verify dataset exists (called once on main process)."""
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        def setup(self, stage: str | None = None) -> None:
            """
            Setup datasets for training/validation.

            Args:
                stage: Current stage ('fit', 'validate', 'test', 'predict').
            """
            if stage == "fit" or stage is None:
                # Scan for images
                if self.recursive:
                    self._image_paths = scan_images_recursive(self.data_dir)
                else:
                    self._image_paths = scan_images_flat(self.data_dir)

                if len(self._image_paths) == 0:
                    raise ValueError(f"No images found in {self.data_dir}")

                # Create augmentation pipeline
                train_transform = FaceAugmentationPipeline(
                    target_size=self.target_size,
                    **self.aug_config,
                )

                # Create full dataset with augmentation
                full_dataset = PretrainDataset(
                    image_paths=self._image_paths,
                    target_size=self.target_size,
                    transform=train_transform,
                )

                # Split into train/val
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val

                # Deterministic split
                generator = torch.Generator().manual_seed(42)
                train_subset, val_subset = random_split(
                    full_dataset,
                    [n_train, n_val],
                    generator=generator,
                )

                self.train_dataset = train_subset

                # Validation dataset without augmentation
                val_dataset_no_aug = PretrainDataset(
                    image_paths=[self._image_paths[i] for i in val_subset.indices],
                    target_size=self.target_size,
                    transform=None,  # No augmentation for validation
                )
                self.val_dataset = val_dataset_no_aug

        def train_dataloader(self) -> DataLoader:
            """Get training DataLoader."""
            assert self.train_dataset is not None
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                drop_last=True,
            )

        def val_dataloader(self) -> DataLoader:
            """Get validation DataLoader."""
            assert self.val_dataset is not None
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        @property
        def num_train_samples(self) -> int:
            """Return number of training samples."""
            if self.train_dataset is not None and isinstance(self.train_dataset, Sized):
                return int(len(self.train_dataset))
            return 0

        @property
        def num_val_samples(self) -> int:
            """Return number of validation samples."""
            if self.val_dataset is not None and isinstance(self.val_dataset, Sized):
                return int(len(self.val_dataset))
            return 0

        @property
        def num_total_images(self) -> int:
            """Return total number of images found."""
            return len(self._image_paths)

else:
    # Fallback when Lightning is not installed
    class PretrainDataModule:  # type: ignore
        """PretrainDataModule requires pytorch-lightning."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PretrainDataModule requires pytorch-lightning. "
                "Install with: pip install pytorch-lightning"
            )
