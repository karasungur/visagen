"""
DALI-based DataModule for PyTorch Lightning.

Provides GPU-accelerated data loading using NVIDIA DALI.
Falls back to standard PyTorch DataLoader when DALI is unavailable.

Features:
    - GPU-based image decoding and augmentation
    - Zero CPU bottleneck for data loading
    - Seamless PyTorch Lightning integration
    - Automatic fallback for Windows/non-CUDA systems

Requires:
    nvidia-dali-cuda120 >= 1.30.0 (Linux only)
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from .dali_pipeline import check_dali_available, create_dali_iterator

# Import standard datamodule for fallback
from .datamodule import FaceDataModule


class DALIFaceDataModule(pl.LightningDataModule):
    """
    DALI-accelerated DataModule for face swap training.

    Uses NVIDIA DALI for GPU-based data loading and augmentation.
    Automatically falls back to standard PyTorch DataLoader if DALI
    is not available (e.g., on Windows).

    Args:
        src_dir: Directory containing source face images.
        dst_dir: Directory containing destination face images.
        batch_size: Batch size per GPU.
        image_size: Output image size (square).
        num_threads: Number of CPU threads for DALI pipeline.
        device_id: GPU device ID.
        augment: Whether to apply data augmentation.
        seed: Random seed for reproducibility.
        num_shards: Total number of shards (for distributed training).
        shard_id: Current shard ID.
        use_dali: Force DALI usage (False for fallback to PyTorch).
        fallback_num_workers: Number of workers for PyTorch fallback.

    Example:
        >>> dm = DALIFaceDataModule(
        ...     src_dir="data/src",
        ...     dst_dir="data/dst",
        ...     batch_size=8,
        ... )
        >>> dm.setup("fit")
        >>> for batch in dm.train_dataloader():
        ...     src_images = batch["src_images"]
        ...     dst_images = batch["dst_images"]
    """

    def __init__(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        batch_size: int = 8,
        image_size: int = 256,
        num_threads: int = 4,
        device_id: int = 0,
        augment: bool = True,
        seed: int = 42,
        num_shards: int = 1,
        shard_id: int = 0,
        use_dali: bool | None = None,
        fallback_num_workers: int = 4,
        val_split: float = 0.1,
        augmentation_config: dict | None = None,
        uniform_yaw: bool = False,
    ) -> None:
        super().__init__()

        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.augment = augment
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.fallback_num_workers = fallback_num_workers
        self.val_split = val_split
        self.augmentation_config = augmentation_config
        self.uniform_yaw = uniform_yaw

        # Determine if DALI should be used
        if use_dali is None:
            self._use_dali = check_dali_available()
        else:
            self._use_dali = use_dali and check_dali_available()

        # Store iterators
        self._train_iterator: Iterator | None = None
        self._val_iterator: Iterator | None = None

        # Fallback datamodule
        self._fallback_dm: FaceDataModule | None = None

    @property
    def using_dali(self) -> bool:
        """Check if DALI is being used."""
        return self._use_dali

    def setup(self, stage: str | None = None) -> None:
        """
        Set up data loaders for training/validation.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if not self._use_dali:
            # Use fallback PyTorch DataLoader
            self._setup_fallback(stage)
            return

        # Validate directories
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.src_dir}")
        if not self.dst_dir.exists():
            raise FileNotFoundError(f"Destination directory not found: {self.dst_dir}")

    def _setup_fallback(self, stage: str | None = None) -> None:
        """Set up fallback PyTorch DataLoader."""
        if self._fallback_dm is None:
            aug_config = self.augmentation_config
            if not self.augment:
                # Explicitly disable all augmentations for parity with `augment=False`.
                aug_config = {
                    "random_flip_prob": 0.0,
                    "random_warp": False,
                    "rotation_range": (0.0, 0.0),
                    "scale_range": (0.0, 0.0),
                    "translation_range": (0.0, 0.0),
                    "hsv_shift_amount": 0.0,
                    "brightness_range": 0.0,
                    "contrast_range": 0.0,
                }
            self._fallback_dm = FaceDataModule(
                src_dir=self.src_dir,
                dst_dir=self.dst_dir,
                batch_size=self.batch_size,
                target_size=self.image_size,
                num_workers=self.fallback_num_workers,
                val_split=self.val_split,
                augmentation_config=aug_config,
                uniform_yaw=self.uniform_yaw,
            )
        self._fallback_dm.setup(stage)

    def train_dataloader(self) -> Any:
        """
        Get training data loader.

        Returns:
            DALI iterator or PyTorch DataLoader.
        """
        if not self._use_dali:
            if self._fallback_dm is None:
                self._setup_fallback("fit")
            assert self._fallback_dm is not None
            return self._fallback_dm.train_dataloader()

        # Create DALI iterator
        self._train_iterator = create_dali_iterator(
            src_dir=self.src_dir,
            dst_dir=self.dst_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            augment=self.augment,
            seed=self.seed,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )

        return DALIIteratorWrapper(self._train_iterator)

    def val_dataloader(self) -> Any:
        """
        Get validation data loader.

        Returns:
            DALI iterator or PyTorch DataLoader (no augmentation).
        """
        if not self._use_dali:
            if self._fallback_dm is None:
                self._setup_fallback("validate")
            assert self._fallback_dm is not None
            return self._fallback_dm.val_dataloader()

        # Create DALI iterator without augmentation
        self._val_iterator = create_dali_iterator(
            src_dir=self.src_dir,
            dst_dir=self.dst_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            augment=False,  # No augmentation for validation
            seed=self.seed + 1,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )

        return DALIIteratorWrapper(self._val_iterator)

    def teardown(self, stage: str | None = None) -> None:
        """Clean up resources."""
        self._train_iterator = None
        self._val_iterator = None

        if self._fallback_dm is not None:
            self._fallback_dm.teardown(stage or "fit")


class DALIIteratorWrapper:
    """
    Wrapper to make DALI iterator compatible with Lightning.

    Converts DALI output format to match expected batch structure.

    Args:
        dali_iterator: DALI generic iterator.
    """

    def __init__(self, dali_iterator) -> None:
        self.dali_iterator = dali_iterator
        self._epoch_size = None

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self):
        """Get next batch, converting DALI format to training tuple format."""
        try:
            data = next(self.dali_iterator)
            # DALI returns list of dicts, one per pipeline
            # We use single pipeline, so take first element
            batch = data[0]
            # Match FaceDataModule contract: (src_dict, dst_dict)
            return (
                {"image": batch["src_images"]},
                {"image": batch["dst_images"]},
            )
        except StopIteration:
            self.dali_iterator.reset()
            raise

    def __len__(self):
        """Return epoch size."""
        if self._epoch_size is None:
            self._epoch_size = self.dali_iterator.epoch_size("src_reader")
        return self._epoch_size // self.dali_iterator.batch_size

    def reset(self):
        """Reset iterator for new epoch."""
        self.dali_iterator.reset()


def create_dali_datamodule(
    src_dir: str | Path,
    dst_dir: str | Path,
    batch_size: int = 8,
    image_size: int = 256,
    num_threads: int = 4,
    device_id: int = 0,
    augment: bool = True,
    seed: int = 42,
    force_pytorch: bool = False,
    val_split: float = 0.1,
    augmentation_config: dict | None = None,
    uniform_yaw: bool = False,
    **kwargs,
) -> pl.LightningDataModule:
    """
    Create a data module, preferring DALI when available.

    Factory function that returns DALIFaceDataModule when DALI is
    available, otherwise returns standard FaceDataModule.

    Args:
        src_dir: Source images directory.
        dst_dir: Destination images directory.
        batch_size: Batch size.
        image_size: Output image size.
        num_threads: Number of CPU threads.
        device_id: GPU device ID.
        augment: Enable augmentation.
        seed: Random seed.
        force_pytorch: Force PyTorch DataLoader even if DALI available.
        **kwargs: Additional arguments passed to datamodule.

    Returns:
        LightningDataModule (DALI or PyTorch based).

    Example:
        >>> dm = create_dali_datamodule(
        ...     src_dir="data/src",
        ...     dst_dir="data/dst",
        ...     batch_size=16,
        ... )
        >>> print(f"Using DALI: {dm.using_dali}")
    """
    fallback_num_workers = int(
        kwargs.pop("fallback_num_workers", kwargs.pop("num_workers", 4))
    )
    use_dali = kwargs.pop("use_dali", None)

    if force_pytorch or not check_dali_available():
        # Use standard PyTorch DataModule
        aug_config = augmentation_config
        if not augment:
            aug_config = {
                "random_flip_prob": 0.0,
                "random_warp": False,
                "rotation_range": (0.0, 0.0),
                "scale_range": (0.0, 0.0),
                "translation_range": (0.0, 0.0),
                "hsv_shift_amount": 0.0,
                "brightness_range": 0.0,
                "contrast_range": 0.0,
            }
        return FaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=batch_size,
            target_size=image_size,
            num_workers=fallback_num_workers,
            val_split=val_split,
            augmentation_config=aug_config,
            uniform_yaw=uniform_yaw,
        )

    # Use DALI DataModule
    return DALIFaceDataModule(
        src_dir=src_dir,
        dst_dir=dst_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_threads=num_threads,
        device_id=device_id,
        augment=augment,
        seed=seed,
        use_dali=use_dali,
        fallback_num_workers=fallback_num_workers,
        val_split=val_split,
        augmentation_config=augmentation_config,
        uniform_yaw=uniform_yaw,
        **kwargs,
    )


def benchmark_dataloaders(
    src_dir: str | Path,
    dst_dir: str | Path,
    batch_size: int = 16,
    image_size: int = 256,
    num_batches: int = 100,
) -> dict:
    """
    Benchmark DALI vs PyTorch data loading performance.

    Args:
        src_dir: Source images directory.
        dst_dir: Destination images directory.
        batch_size: Batch size to test.
        image_size: Image size.
        num_batches: Number of batches to measure.

    Returns:
        Dict with 'pytorch_fps', 'dali_fps', 'speedup'.
    """
    import time

    import torch

    results: dict[str, float | None] = {}

    # Benchmark PyTorch DataLoader
    pytorch_dm = FaceDataModule(
        src_dir=src_dir,
        dst_dir=dst_dir,
        batch_size=batch_size,
        target_size=image_size,
        num_workers=4,
        augmentation_config=None,
    )
    pytorch_dm.setup("fit")
    pytorch_loader = pytorch_dm.train_dataloader()

    # Warm up
    for i, batch in enumerate(pytorch_loader):
        if i >= 5:
            break
        src_dict, dst_dict = batch
        _ = src_dict["image"].cuda()
        _ = dst_dict["image"].cuda()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i, batch in enumerate(pytorch_loader):
        if i >= num_batches:
            break
        src_dict, dst_dict = batch
        _ = src_dict["image"].cuda()
        _ = dst_dict["image"].cuda()

    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    pytorch_fps = (num_batches * batch_size * 2) / pytorch_time  # *2 for src+dst
    results["pytorch_fps"] = pytorch_fps
    results["pytorch_time"] = pytorch_time

    # Benchmark DALI if available
    if check_dali_available():
        dali_dm = DALIFaceDataModule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_threads=4,
            device_id=0,
            augment=True,
        )
        dali_dm.setup("fit")
        dali_loader = dali_dm.train_dataloader()

        # Warm up
        for i, batch in enumerate(dali_loader):
            if i >= 5:
                break
            src_dict, dst_dict = batch
            _ = src_dict["image"]
            _ = dst_dict["image"]

        torch.cuda.synchronize()
        start = time.perf_counter()

        for i, batch in enumerate(dali_loader):
            if i >= num_batches:
                break
            src_dict, dst_dict = batch
            _ = src_dict["image"]
            _ = dst_dict["image"]

        torch.cuda.synchronize()
        dali_time = time.perf_counter() - start
        dali_fps = (num_batches * batch_size * 2) / dali_time
        results["dali_fps"] = dali_fps
        results["dali_time"] = dali_time
        results["speedup"] = dali_fps / pytorch_fps
    else:
        results["dali_fps"] = None
        results["dali_time"] = None
        results["speedup"] = None

    return results
