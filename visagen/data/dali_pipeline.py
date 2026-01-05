"""
NVIDIA DALI Pipeline for GPU-accelerated face data loading.

Provides high-performance data loading and augmentation for face swapping training.
All operations (decode, resize, augment) run on GPU, eliminating CPU bottlenecks.

Features:
    - GPU-based JPEG decoding
    - GPU augmentations (flip, rotate, color jitter)
    - DFL-style random warping via external source
    - Paired source/destination loading

Requires:
    nvidia-dali-cuda120 >= 1.30.0

Note:
    DALI is Linux-only. For Windows, use the standard PyTorch DataLoader fallback.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# DALI imports with graceful fallback
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.ops as ops
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    pipeline_def = None
    fn = None
    types = None
    DALIGenericIterator = None
    LastBatchPolicy = None
    ops = None


def check_dali_available() -> bool:
    """Check if DALI is available."""
    return DALI_AVAILABLE


class FaceSwapExternalSource:
    """
    External source for DALI pipeline that reads face images.

    Provides paired source/destination images for face swap training.
    Handles image path iteration and batching.

    Args:
        src_files: List of source image paths.
        dst_files: List of destination image paths.
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle the data.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        src_files: List[Path],
        dst_files: List[Path],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.src_files = [str(f) for f in src_files]
        self.dst_files = [str(f) for f in dst_files]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Create indices
        self.src_indices = np.arange(len(self.src_files))
        self.dst_indices = np.arange(len(self.dst_files))

        if shuffle:
            self.rng.shuffle(self.src_indices)
            self.rng.shuffle(self.dst_indices)

        self.src_pos = 0
        self.dst_pos = 0

    def __call__(self, sample_info) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next batch of image file contents.

        Args:
            sample_info: DALI sample info (contains idx).

        Returns:
            Tuple of (src_image_bytes, dst_image_bytes).
        """
        idx = sample_info.idx_in_epoch

        # Get source image
        src_idx = self.src_indices[idx % len(self.src_indices)]
        src_path = self.src_files[src_idx]

        # Get destination image (independent index)
        dst_idx = self.dst_indices[idx % len(self.dst_indices)]
        dst_path = self.dst_files[dst_idx]

        # Read image files as bytes
        with open(src_path, 'rb') as f:
            src_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        with open(dst_path, 'rb') as f:
            dst_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        return src_bytes, dst_bytes

    def __len__(self) -> int:
        """Return dataset length (max of src and dst)."""
        return max(len(self.src_files), len(self.dst_files))

    def reset(self) -> None:
        """Reset for new epoch."""
        self.src_pos = 0
        self.dst_pos = 0
        if self.shuffle:
            self.rng.shuffle(self.src_indices)
            self.rng.shuffle(self.dst_indices)


if DALI_AVAILABLE:

    @pipeline_def
    def face_swap_pipeline(
        src_files: List[str],
        dst_files: List[str],
        image_size: int = 256,
        # Augmentation parameters
        flip_prob: float = 0.5,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        saturation_range: Tuple[float, float] = (0.9, 1.1),
        hue_range: float = 0.05,
        # Pipeline settings
        seed: int = 42,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        """
        DALI pipeline for face swap training with GPU augmentations.

        Args:
            src_files: List of source image file paths.
            dst_files: List of destination image file paths.
            image_size: Output image size (square).
            flip_prob: Probability of horizontal flip.
            rotation_range: Maximum rotation angle in degrees.
            scale_range: Scale range (min, max).
            brightness_range: Brightness adjustment range.
            contrast_range: Contrast adjustment range.
            saturation_range: Saturation adjustment range.
            hue_range: Hue adjustment range.
            seed: Random seed.
            shard_id: Shard ID for distributed training.
            num_shards: Total number of shards.

        Returns:
            Tuple of (src_images, dst_images) as GPU tensors.
        """
        # Read source images
        src_jpegs, src_labels = fn.readers.file(
            files=src_files,
            random_shuffle=True,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            name="src_reader",
        )

        # Read destination images
        dst_jpegs, dst_labels = fn.readers.file(
            files=dst_files,
            random_shuffle=True,
            seed=seed + 1,  # Different seed for dst
            shard_id=shard_id,
            num_shards=num_shards,
            name="dst_reader",
        )

        # Decode on GPU (mixed = decode on CPU, output on GPU)
        src_images = fn.decoders.image(
            src_jpegs,
            device="mixed",
            output_type=types.RGB,
        )
        dst_images = fn.decoders.image(
            dst_jpegs,
            device="mixed",
            output_type=types.RGB,
        )

        # Resize to target size
        src_images = fn.resize(
            src_images,
            size=[image_size, image_size],
            mode="not_smaller",
            interp_type=types.INTERP_LINEAR,
        )
        dst_images = fn.resize(
            dst_images,
            size=[image_size, image_size],
            mode="not_smaller",
            interp_type=types.INTERP_LINEAR,
        )

        # Center crop to exact size
        src_images = fn.crop(
            src_images,
            crop=[image_size, image_size],
            crop_pos_x=0.5,
            crop_pos_y=0.5,
        )
        dst_images = fn.crop(
            dst_images,
            crop=[image_size, image_size],
            crop_pos_x=0.5,
            crop_pos_y=0.5,
        )

        # =====================================================================
        # Augmentations (applied independently to src and dst)
        # =====================================================================

        # Random horizontal flip
        src_flip = fn.random.coin_flip(probability=flip_prob)
        dst_flip = fn.random.coin_flip(probability=flip_prob)
        src_images = fn.flip(src_images, horizontal=src_flip)
        dst_images = fn.flip(dst_images, horizontal=dst_flip)

        # Random rotation
        src_angle = fn.random.uniform(range=[-rotation_range, rotation_range])
        dst_angle = fn.random.uniform(range=[-rotation_range, rotation_range])
        src_images = fn.rotate(
            src_images,
            angle=src_angle,
            fill_value=0,
            keep_size=True,
        )
        dst_images = fn.rotate(
            dst_images,
            angle=dst_angle,
            fill_value=0,
            keep_size=True,
        )

        # Color augmentations
        # Brightness
        src_brightness = fn.random.uniform(range=brightness_range)
        dst_brightness = fn.random.uniform(range=brightness_range)
        src_images = fn.brightness(src_images, brightness=src_brightness)
        dst_images = fn.brightness(dst_images, brightness=dst_brightness)

        # Contrast
        src_contrast = fn.random.uniform(range=contrast_range)
        dst_contrast = fn.random.uniform(range=contrast_range)
        src_images = fn.contrast(src_images, contrast=src_contrast)
        dst_images = fn.contrast(dst_images, contrast=dst_contrast)

        # HSV adjustments
        src_hue = fn.random.uniform(range=[-hue_range, hue_range])
        dst_hue = fn.random.uniform(range=[-hue_range, hue_range])
        src_saturation = fn.random.uniform(range=saturation_range)
        dst_saturation = fn.random.uniform(range=saturation_range)

        src_images = fn.hsv(
            src_images,
            hue=src_hue,
            saturation=src_saturation,
        )
        dst_images = fn.hsv(
            dst_images,
            hue=dst_hue,
            saturation=dst_saturation,
        )

        # Normalize to [-1, 1] range (from 0-255)
        src_images = fn.normalize(
            src_images,
            mean=[127.5, 127.5, 127.5],
            stddev=[127.5, 127.5, 127.5],
        )
        dst_images = fn.normalize(
            dst_images,
            mean=[127.5, 127.5, 127.5],
            stddev=[127.5, 127.5, 127.5],
        )

        # Transpose to CHW format for PyTorch
        src_images = fn.transpose(src_images, perm=[2, 0, 1])
        dst_images = fn.transpose(dst_images, perm=[2, 0, 1])

        return src_images, dst_images


    @pipeline_def
    def face_swap_pipeline_simple(
        src_files: List[str],
        dst_files: List[str],
        image_size: int = 256,
        seed: int = 42,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        """
        Simple DALI pipeline without augmentations (for validation).

        Args:
            src_files: List of source image file paths.
            dst_files: List of destination image file paths.
            image_size: Output image size.
            seed: Random seed.
            shard_id: Shard ID.
            num_shards: Total shards.

        Returns:
            Tuple of (src_images, dst_images).
        """
        # Read images
        src_jpegs, _ = fn.readers.file(
            files=src_files,
            random_shuffle=False,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            name="src_reader",
        )
        dst_jpegs, _ = fn.readers.file(
            files=dst_files,
            random_shuffle=False,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            name="dst_reader",
        )

        # Decode
        src_images = fn.decoders.image(src_jpegs, device="mixed", output_type=types.RGB)
        dst_images = fn.decoders.image(dst_jpegs, device="mixed", output_type=types.RGB)

        # Resize and crop
        src_images = fn.resize(src_images, size=[image_size, image_size])
        dst_images = fn.resize(dst_images, size=[image_size, image_size])

        # Normalize to [-1, 1]
        src_images = fn.normalize(
            src_images,
            mean=[127.5, 127.5, 127.5],
            stddev=[127.5, 127.5, 127.5],
        )
        dst_images = fn.normalize(
            dst_images,
            mean=[127.5, 127.5, 127.5],
            stddev=[127.5, 127.5, 127.5],
        )

        # Transpose to CHW
        src_images = fn.transpose(src_images, perm=[2, 0, 1])
        dst_images = fn.transpose(dst_images, perm=[2, 0, 1])

        return src_images, dst_images


def create_dali_iterator(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    batch_size: int,
    image_size: int = 256,
    num_threads: int = 4,
    device_id: int = 0,
    augment: bool = True,
    seed: int = 42,
    shard_id: int = 0,
    num_shards: int = 1,
) -> "DALIGenericIterator":
    """
    Create a DALI iterator for face swap training.

    Args:
        src_dir: Directory containing source face images.
        dst_dir: Directory containing destination face images.
        batch_size: Batch size.
        image_size: Image size.
        num_threads: Number of CPU threads for data loading.
        device_id: GPU device ID.
        augment: Whether to apply augmentations.
        seed: Random seed.
        shard_id: Shard ID for distributed training.
        num_shards: Total number of shards.

    Returns:
        DALIGenericIterator yielding {"src_images": tensor, "dst_images": tensor}.

    Raises:
        ImportError: If DALI is not available.
        FileNotFoundError: If directories don't exist.
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            "NVIDIA DALI is not available. "
            "Install with: pip install nvidia-dali-cuda120"
        )

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not dst_dir.exists():
        raise FileNotFoundError(f"Destination directory not found: {dst_dir}")

    # Get image files
    src_files = sorted([
        str(f) for f in src_dir.glob("*.jpg")
    ] + [
        str(f) for f in src_dir.glob("*.jpeg")
    ] + [
        str(f) for f in src_dir.glob("*.png")
    ])

    dst_files = sorted([
        str(f) for f in dst_dir.glob("*.jpg")
    ] + [
        str(f) for f in dst_dir.glob("*.jpeg")
    ] + [
        str(f) for f in dst_dir.glob("*.png")
    ])

    if not src_files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    if not dst_files:
        raise FileNotFoundError(f"No images found in {dst_dir}")

    # Create pipeline
    if augment:
        pipe = face_swap_pipeline(
            src_files=src_files,
            dst_files=dst_files,
            image_size=image_size,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )
    else:
        pipe = face_swap_pipeline_simple(
            src_files=src_files,
            dst_files=dst_files,
            image_size=image_size,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )

    pipe.build()

    # Create iterator
    iterator = DALIGenericIterator(
        [pipe],
        output_map=["src_images", "dst_images"],
        reader_name="src_reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )

    return iterator


def get_dataset_size(data_dir: Union[str, Path]) -> int:
    """
    Get number of images in a directory.

    Args:
        data_dir: Directory path.

    Returns:
        Number of image files.
    """
    data_dir = Path(data_dir)
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        count += len(list(data_dir.glob(ext)))
    return count
