"""
Face Data Module for PyTorch Lightning.

Provides DataModules for training face swapping models.
"""

from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split
from torch.utils.data._utils.collate import default_collate

pl: Any
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from visagen.data.augmentations import FaceAugmentationPipeline  # noqa: E402
from visagen.data.face_dataset import FaceDataset  # noqa: E402
from visagen.data.sequence_dataset import PairedSequenceDataset  # noqa: E402


def _collate_sample_dicts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate dict-based face samples while preserving non-tensor metadata.

    Tensor/number fields are collated using default behavior; variable Python
    metadata (e.g. `seg_ie_polys`) is kept as a list to avoid shape coercion.
    """
    keys = {key for sample in samples for key in sample}
    collated: dict[str, Any] = {}

    for key in keys:
        values = [sample.get(key) for sample in samples]
        non_none = [value for value in values if value is not None]

        if not non_none:
            collated[key] = None
            continue

        if len(non_none) != len(values):
            collated[key] = values
            continue

        if isinstance(non_none[0], torch.Tensor):
            collated[key] = torch.stack(cast(list[torch.Tensor], values))
            continue

        if key == "seg_ie_polys":
            collated[key] = values
            continue

        try:
            collated[key] = default_collate(values)
        except Exception:
            collated[key] = values

    return collated


def paired_face_collate_fn(batch: list[Any]) -> Any:
    """Collate `(src_dict, dst_dict)` batches used by training dataloaders."""
    if not batch:
        return batch

    first = batch[0]
    if isinstance(first, tuple) and len(first) == 2:
        src_first, dst_first = first
        if isinstance(src_first, dict) and isinstance(dst_first, dict):
            src_items = [cast(dict[str, Any], sample[0]) for sample in batch]
            dst_items = [cast(dict[str, Any], sample[1]) for sample in batch]
            return _collate_sample_dicts(src_items), _collate_sample_dicts(dst_items)

    return default_collate(batch)


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
        src_dataset: Any,
        dst_dataset: Any,
        return_dict: bool = True,
    ) -> None:
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset
        self.return_dict = return_dict

        # Use larger dataset size, wrap smaller
        self._length = max(len(src_dataset), len(dst_dataset))

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _resolve_dataset_index(dataset: Any, idx: int) -> int:
        return idx % len(dataset)

    def __getitem__(
        self, idx: int | tuple[int, int]
    ) -> (
        tuple[dict[str, Any], dict[str, Any]]
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
        if isinstance(idx, tuple):
            src_raw, dst_raw = idx
        else:
            src_raw = idx
            dst_raw = idx

        src_idx = self._resolve_dataset_index(self.src_dataset, int(src_raw))
        dst_idx = self._resolve_dataset_index(self.dst_dataset, int(dst_raw))

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
        dataset: Any,
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
                    try:
                        src_image, src_mask = self.transform(
                            src_image,
                            src_mask,
                            target_image=dst_image,
                        )
                        dst_image, dst_mask = self.transform(
                            dst_image,
                            dst_mask,
                            target_image=src["image"],
                        )
                    except TypeError:
                        src_image, src_mask = self.transform(src_image, src_mask)
                        dst_image, dst_mask = self.transform(dst_image, dst_mask)

                    # Update dicts with transformed values
                    src["image"] = src_image
                    if src_mask is not None:
                        src["mask"] = src_mask

                    dst["image"] = dst_image
                    if dst_mask is not None:
                        dst["mask"] = dst_mask

                    # Landmarks remain unchanged.
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


class UniformYawSubset(Subset):
    """Subset with optional uniform-yaw sampling support."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self._uniform_bins: list[list[int]] | None = None

        build_bins = getattr(dataset, "build_uniform_yaw_bins_for_indices", None)
        if callable(build_bins):
            try:
                bins = build_bins(indices)
                if bins:
                    self._uniform_bins = bins
            except Exception:
                self._uniform_bins = None

    def sample_uniform_yaw(self) -> int:
        if not self._uniform_bins:
            return int(np.random.randint(len(self.indices)))

        bin_idx = int(np.random.randint(len(self._uniform_bins)))
        return int(np.random.choice(self._uniform_bins[bin_idx]))


class UniformYawPairSampler(Sampler[tuple[int, int]]):
    """Sampler that yields `(src_idx, dst_idx)` pairs using uniform-yaw bins."""

    def __init__(
        self,
        src_dataset: Dataset,
        dst_dataset: Dataset,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.src_dataset = src_dataset
        self.dst_dataset = dst_dataset
        self.num_samples = num_samples
        self.seed = seed
        self._epoch = 0

    @staticmethod
    def _sample_index(dataset: Dataset, rng: np.random.Generator) -> int:
        dataset_len = len(cast(Sized, dataset))
        sampler = getattr(dataset, "sample_uniform_yaw", None)
        if callable(sampler):
            try:
                sampled = int(sampler())
                if 0 <= sampled < dataset_len:
                    return sampled
            except Exception:
                pass
        return int(rng.integers(0, dataset_len))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.num_samples):
            yield (
                self._sample_index(self.src_dataset, rng),
                self._sample_index(self.dst_dataset, rng),
            )

    def __len__(self) -> int:
        return self.num_samples


class TemporalPairDictWrapper(Dataset):
    """Wrap temporal sequences into `(src_dict, dst_dict)` batch contract."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        src_seq, dst_seq = self.dataset[idx]
        return {"image": src_seq}, {"image": dst_seq}


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
            allow_packed_faceset: bool = True,
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
            self.allow_packed_faceset = allow_packed_faceset

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
            self._train_sampler: Sampler[int | tuple[int, int]] | None = None

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
                    allow_packed_faceset=self.allow_packed_faceset,
                )
                dst_full = FaceDataset(
                    self.dst_dir,
                    target_size=self.target_size,
                    transform=None,
                    uniform_yaw=self.uniform_yaw,
                    allow_packed_faceset=self.allow_packed_faceset,
                )

                # Calculate split sizes
                n_src_val = max(1, int(len(src_full) * self.val_split))
                n_dst_val = max(1, int(len(dst_full) * self.val_split))
                n_src_train = len(src_full) - n_src_val
                n_dst_train = len(dst_full) - n_dst_val

                # Split datasets deterministically
                generator = torch.Generator().manual_seed(42)

                src_perm = torch.randperm(len(src_full), generator=generator).tolist()
                dst_perm = torch.randperm(len(dst_full), generator=generator).tolist()

                src_val_indices = src_perm[:n_src_val]
                src_train_indices = src_perm[n_src_val : n_src_val + n_src_train]
                dst_val_indices = dst_perm[:n_dst_val]
                dst_train_indices = dst_perm[n_dst_val : n_dst_val + n_dst_train]

                if self.uniform_yaw:
                    self._src_train = UniformYawSubset(src_full, src_train_indices)
                    self._dst_train = UniformYawSubset(dst_full, dst_train_indices)
                else:
                    self._src_train = Subset(src_full, src_train_indices)
                    self._dst_train = Subset(dst_full, dst_train_indices)

                self._src_val = Subset(src_full, src_val_indices)
                self._dst_val = Subset(dst_full, dst_val_indices)

                # Create augmentation pipeline
                train_transform = FaceAugmentationPipeline(
                    target_size=self.target_size,
                    **self.aug_config,
                )

                # Create paired datasets
                train_paired = PairedFaceDataset(self._src_train, self._dst_train)
                val_paired = PairedFaceDataset(self._src_val, self._dst_val)

                if self.uniform_yaw:
                    self._train_sampler = UniformYawPairSampler(
                        self._src_train,
                        self._dst_train,
                        num_samples=len(train_paired),
                        seed=42,
                    )
                else:
                    self._train_sampler = None

                # Apply augmentation to training data
                self.train_dataset = TransformWrapper(train_paired, train_transform)
                self.val_dataset = val_paired  # No augmentation for validation

        def train_dataloader(self) -> DataLoader:
            """Get training DataLoader."""
            assert self.train_dataset is not None
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self._train_sampler is None,
                sampler=self._train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=paired_face_collate_fn,
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
                collate_fn=paired_face_collate_fn,
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


    class TemporalFaceDataModule(pl.LightningDataModule):
        """
        Temporal DataModule for sequence-based training.

        Returns batches in the same `(src_dict, dst_dict)` format as FaceDataModule,
        where `dict["image"]` has shape `(B, C, T, H, W)`.
        """

        def __init__(
            self,
            src_dir: str | Path,
            dst_dir: str | Path,
            batch_size: int = 8,
            num_workers: int = 4,
            target_size: int = 256,
            val_split: float = 0.1,
            sequence_length: int = 5,
            stride: int = 1,
            pin_memory: bool = True,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()

            self.src_dir = Path(src_dir)
            self.dst_dir = Path(dst_dir)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.target_size = target_size
            self.val_split = val_split
            self.sequence_length = sequence_length
            self.stride = stride
            self.pin_memory = pin_memory

            self.train_dataset: Dataset | None = None
            self.val_dataset: Dataset | None = None

        def setup(self, stage: str | None = None) -> None:
            if stage == "fit" or stage is None:
                full_dataset = PairedSequenceDataset(
                    src_dir=self.src_dir,
                    dst_dir=self.dst_dir,
                    sequence_length=self.sequence_length,
                    target_size=self.target_size,
                    stride=self.stride,
                )

                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val

                generator = torch.Generator().manual_seed(42)
                train_subset, val_subset = random_split(
                    full_dataset,
                    [n_train, n_val],
                    generator=generator,
                )

                self.train_dataset = TemporalPairDictWrapper(train_subset)
                self.val_dataset = TemporalPairDictWrapper(val_subset)

        def train_dataloader(self) -> DataLoader:
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
            if self.train_dataset is not None and isinstance(self.train_dataset, Sized):
                return int(len(self.train_dataset))
            return 0

        @property
        def num_val_samples(self) -> int:
            if self.val_dataset is not None and isinstance(self.val_dataset, Sized):
                return int(len(self.val_dataset))
            return 0


    def create_temporal_datamodule(
        src_dir: str | Path,
        dst_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        target_size: int = 256,
        val_split: float = 0.1,
        sequence_length: int = 5,
        stride: int = 1,
    ) -> TemporalFaceDataModule:
        return TemporalFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=target_size,
            val_split=val_split,
            sequence_length=sequence_length,
            stride=stride,
        )

else:
    # Fallback when Lightning is not installed
    class FaceDataModule:  # type: ignore
        """FaceDataModule requires pytorch-lightning."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FaceDataModule requires pytorch-lightning. "
                "Install with: pip install pytorch-lightning"
            )


    class TemporalFaceDataModule:  # type: ignore
        """TemporalFaceDataModule requires pytorch-lightning."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TemporalFaceDataModule requires pytorch-lightning. "
                "Install with: pip install pytorch-lightning"
            )


    def create_temporal_datamodule(*args, **kwargs):  # type: ignore
        raise ImportError(
            "TemporalFaceDataModule requires pytorch-lightning. "
            "Install with: pip install pytorch-lightning"
        )
