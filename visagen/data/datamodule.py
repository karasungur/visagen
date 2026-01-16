"""
Face Data Module for PyTorch Lightning.

Provides DataModules for training face swapping models.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from visagen.data.augmentations import FaceAugmentationPipeline
from visagen.data.face_dataset import FaceDataset


class PairedFaceDataset(Dataset):
    """
    Dataset returning paired (src, dst) samples for training.

    Each __getitem__ returns one sample from src and one from dst
    datasets, with independent augmentations applied to each.

    Args:
        src_dataset: Source person face dataset.
        dst_dataset: Destination person face dataset.
        return_dict: If True, return full dicts with landmarks/mask.
                     If False, return only image tensors (legacy). Default: True.

    Example:
        >>> src_data = FaceDataset(Path("data_src/aligned"))
        >>> dst_data = FaceDataset(Path("data_dst/aligned"))
        >>> paired = PairedFaceDataset(src_data, dst_data)
        >>> src_dict, dst_dict = paired[0]
        >>> src_dict["image"].shape, src_dict["landmarks"].shape
        (torch.Size([3, 256, 256]), torch.Size([68, 2]))
    """

    def __init__(
        self,
        src_dataset: Dataset,
        dst_dataset: Dataset,
        return_dict: bool = True,
    ) -> None:
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset
        self.return_dict = return_dict

        # Use larger dataset size, wrap smaller
        self._length = max(len(src_dataset), len(dst_dataset))

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Get paired sample.

        Args:
            idx: Sample index.

        Returns:
            If return_dict=True: Tuple of (src_dict, dst_dict) each containing
                'image', 'landmarks', and optionally 'mask'.
            If return_dict=False: Tuple of (src_image, dst_image) tensors.
        """
        # Wrap indices for smaller dataset
        src_idx = idx % len(self.src_dataset)
        dst_idx = idx % len(self.dst_dataset)

        src_sample = self.src_dataset[src_idx]
        dst_sample = self.dst_dataset[dst_idx]

        if self.return_dict:
            return src_sample, dst_sample
        else:
            # Legacy mode: return only images
            return src_sample["image"], dst_sample["image"]


class TransformWrapper(Dataset):
    """
    Wrapper to apply transforms to dataset output.

    Used to apply augmentation after dataset loading.
    Supports both dict-based and tensor-based paired datasets.

    Args:
        dataset: Base dataset.
        transform: Transform to apply.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        if self.transform is not None:
            if isinstance(item, tuple) and len(item) == 2:
                src, dst = item

                # Handle dict-based paired dataset (new format with landmarks)
                if isinstance(src, dict) and isinstance(dst, dict):
                    src_image = src["image"]
                    src_mask = src.get("mask")
                    dst_image = dst["image"]
                    dst_mask = dst.get("mask")

                    # Apply transforms to images and masks
                    src_image, src_mask = self.transform(src_image, src_mask)
                    dst_image, dst_mask = self.transform(dst_image, dst_mask)

                    # Update dicts with transformed values
                    src["image"] = src_image
                    if src_mask is not None:
                        src["mask"] = src_mask

                    dst["image"] = dst_image
                    if dst_mask is not None:
                        dst["mask"] = dst_mask

                    # Landmarks are preserved unchanged (pixel coords don't need transform)
                    return src, dst
                else:
                    # Legacy tensor-only paired dataset
                    src, _ = self.transform(src, None)
                    dst, _ = self.transform(dst, None)
                    return src, dst

            elif isinstance(item, dict):
                # Single dataset returns dict
                image = item["image"]
                mask = item.get("mask")
                image, mask = self.transform(image, mask)
                item["image"] = image
                if mask is not None:
                    item["mask"] = mask
                return item

        return item


if pl is not None:

    class FaceDataModule(pl.LightningDataModule):
        """
        Lightning DataModule for DFL face training.

        Manages src/dst datasets with proper train/val splits,
        augmentation, and DataLoader configuration.

        Args:
            src_dir: Source faces directory.
            dst_dir: Destination faces directory.
            batch_size: Training batch size. Default: 8.
            num_workers: DataLoader workers. Default: 4.
            target_size: Output image size. Default: 256.
            val_split: Validation split ratio. Default: 0.1.
            augmentation_config: Dict of augmentation parameters. Default: None.
            pin_memory: Pin memory for GPU. Default: True.

        Example:
            >>> datamodule = FaceDataModule(
            ...     src_dir=Path("data_src/aligned"),
            ...     dst_dir=Path("data_dst/aligned"),
            ...     batch_size=8,
            ... )
            >>> datamodule.setup()
            >>> train_loader = datamodule.train_dataloader()
        """

        def __init__(
            self,
            src_dir: str | Path,
            dst_dir: str | Path,
            batch_size: int = 8,
            num_workers: int = 4,
            target_size: int = 256,
            val_split: float = 0.1,
            augmentation_config: dict | None = None,
            pin_memory: bool = True,
            uniform_yaw: bool = False,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()

            self.src_dir = Path(src_dir)
            self.dst_dir = Path(dst_dir)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.target_size = target_size
            self.val_split = val_split
            self.pin_memory = pin_memory
            self.uniform_yaw = uniform_yaw

            # Default augmentation config matching legacy DFL
            self.aug_config = augmentation_config or {
                "random_flip_prob": 0.4,
                "random_warp": True,
                "rotation_range": (-10, 10),
                "scale_range": (-0.05, 0.05),
                "translation_range": (-0.05, 0.05),
                "hsv_shift_amount": 0.1,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
            }

            # Will be set in setup()
            self.train_dataset: Dataset | None = None
            self.val_dataset: Dataset | None = None
            self._src_train: Dataset | None = None
            self._src_val: Dataset | None = None
            self._dst_train: Dataset | None = None
            self._dst_val: Dataset | None = None

        def setup(self, stage: str | None = None) -> None:
            """
            Setup datasets for training/validation.

            Args:
                stage: Current stage ('fit', 'validate', 'test', 'predict').
            """
            if stage == "fit" or stage is None:
                # Load full datasets (no augmentation yet)
                src_full = FaceDataset(
                    self.src_dir,
                    target_size=self.target_size,
                    transform=None,
                    uniform_yaw=self.uniform_yaw,
                )
                dst_full = FaceDataset(
                    self.dst_dir,
                    target_size=self.target_size,
                    transform=None,
                    uniform_yaw=self.uniform_yaw,
                )

                # Calculate split sizes
                n_src_val = max(1, int(len(src_full) * self.val_split))
                n_dst_val = max(1, int(len(dst_full) * self.val_split))
                n_src_train = len(src_full) - n_src_val
                n_dst_train = len(dst_full) - n_dst_val

                # Split datasets deterministically
                generator = torch.Generator().manual_seed(42)

                self._src_train, self._src_val = random_split(
                    src_full,
                    [n_src_train, n_src_val],
                    generator=generator,
                )
                self._dst_train, self._dst_val = random_split(
                    dst_full,
                    [n_dst_train, n_dst_val],
                    generator=generator,
                )

                # Create augmentation pipeline
                train_transform = FaceAugmentationPipeline(
                    target_size=self.target_size,
                    **self.aug_config,
                )

                # Create paired datasets
                train_paired = PairedFaceDataset(self._src_train, self._dst_train)
                val_paired = PairedFaceDataset(self._src_val, self._dst_val)

                # Apply augmentation to training data
                self.train_dataset = TransformWrapper(train_paired, train_transform)
                self.val_dataset = val_paired  # No augmentation for validation

        def train_dataloader(self) -> DataLoader:
            """Get training DataLoader."""
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
            if self.train_dataset is not None:
                return len(self.train_dataset)
            return 0

        @property
        def num_val_samples(self) -> int:
            """Return number of validation samples."""
            if self.val_dataset is not None:
                return len(self.val_dataset)
            return 0

else:
    # Fallback when Lightning is not installed
    class FaceDataModule:  # type: ignore
        """FaceDataModule requires pytorch-lightning."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FaceDataModule requires pytorch-lightning. "
                "Install with: pip install pytorch-lightning"
            )
