"""
NVIDIA DALI Pipeline for GPU-accelerated face data loading.

Provides high-performance data loading and augmentation for face swapping training.
All operations (decode, resize, augment) run on GPU, eliminating CPU bottlenecks.

Features:
    - GPU-based JPEG decoding
    - GPU augmentations (flip, color jitter)
    - DFL-style random warping via external source
    - Paired source/destination loading

Requires:
    nvidia-dali-cuda120 >= 1.30.0

Note:
    DALI is Linux-only. For Windows, use the standard PyTorch DataLoader fallback.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from visagen.data.dali_warp import DALIAffineGenerator
from visagen.data.face_dataset import PACKED_FACESET_FILENAME, PackedFacesetReader
from visagen.data.face_sample import FaceSample

# DALI imports with graceful fallback
try:
    import nvidia.dali.ops as ops
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

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


SUPPORTED_DALI_WARP_MODES = {"affine", "strict"}


class FaceSwapExternalSource:
    """
    External source for DALI pipeline that reads face images.

    Provides paired source/destination images for face swap training.
    Handles image path iteration and batching.

    Args:
        src_items: List of source image paths or packed FaceSample entries.
        dst_items: List of destination image paths or packed FaceSample entries.
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle the data.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        src_items: Sequence[Path | str | FaceSample],
        dst_items: Sequence[Path | str | FaceSample],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.src_items = list(src_items)
        self.dst_items = list(dst_items)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Create indices
        self.src_indices = np.arange(len(self.src_items))
        self.dst_indices = np.arange(len(self.dst_items))

        if shuffle:
            self.rng.shuffle(self.src_indices)
            self.rng.shuffle(self.dst_indices)

        self.src_pos = 0
        self.dst_pos = 0

    def __call__(self, sample_info) -> tuple[np.ndarray, np.ndarray]:
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
        src_item = self.src_items[src_idx]

        # Get destination image (independent index)
        dst_idx = self.dst_indices[idx % len(self.dst_indices)]
        dst_item = self.dst_items[dst_idx]

        src_bytes = np.frombuffer(self._read_item_bytes(src_item), dtype=np.uint8)
        dst_bytes = np.frombuffer(self._read_item_bytes(dst_item), dtype=np.uint8)

        return src_bytes, dst_bytes

    def __len__(self) -> int:
        """Return dataset length (max of src and dst)."""
        return max(len(self.src_items), len(self.dst_items))

    def reset(self) -> None:
        """Reset for new epoch."""
        self.src_pos = 0
        self.dst_pos = 0
        if self.shuffle:
            self.rng.shuffle(self.src_indices)
            self.rng.shuffle(self.dst_indices)

    @staticmethod
    def _read_item_bytes(item: Path | str | FaceSample) -> bytes:
        """Read encoded image bytes from path or packed FaceSample."""
        if isinstance(item, FaceSample):
            return item.read_raw_bytes()

        path = Path(item)
        with open(path, "rb") as f:
            return f.read()


def _scan_image_files(root_dir: Path) -> list[Path]:
    """Collect image files from a flat aligned directory."""
    files: list[Path] = []
    for ext in (".jpg", ".jpeg", ".png"):
        files.extend(root_dir.glob(f"*{ext}"))
        files.extend(root_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def _load_packed_samples(root_dir: Path) -> list[FaceSample]:
    """Load packed faceset samples from `<root>/faceset.pak`."""
    packed_path = root_dir / PACKED_FACESET_FILENAME
    if not packed_path.exists():
        return []
    reader = PackedFacesetReader(packed_path)
    return reader.read_samples(root_dir=root_dir)


def _resolve_dali_inputs(
    root_dir: Path,
    *,
    allow_packed_faceset: bool,
) -> list[Path | FaceSample]:
    """
    Resolve DALI inputs as filesystem images or legacy packed samples.

    Preference order:
      1) plain image files
      2) faceset.pak entries (if allowed)
    """
    files = _scan_image_files(root_dir)
    if files:
        resolved_files: list[Path | FaceSample] = []
        resolved_files.extend(files)
        return resolved_files

    packed_path = root_dir / PACKED_FACESET_FILENAME
    if packed_path.exists() and not allow_packed_faceset:
        raise ValueError(
            "No images found and faceset.pak loading is disabled. "
            "Set allow_packed_faceset=True to enable legacy packed input."
        )

    if allow_packed_faceset and packed_path.exists():
        samples = _load_packed_samples(root_dir)
        if samples:
            resolved_samples: list[Path | FaceSample] = []
            resolved_samples.extend(samples)
            return resolved_samples
        raise ValueError(f"faceset.pak exists but contains no readable samples: {root_dir}")

    raise FileNotFoundError(f"No images found in {root_dir}")


if DALI_AVAILABLE:

    @pipeline_def
    def face_swap_pipeline(
        src_files: list[str],
        dst_files: list[str],
        image_size: int = 256,
        # Augmentation parameters
        flip_prob: float = 0.4,
        rotation_range: float = 10.0,
        scale_range: tuple[float, float] = (0.95, 1.05),
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
        saturation_range: tuple[float, float] = (0.9, 1.1),
        hue_range: float = 0.05,
        # Legacy-style affine warp augmentation
        warp_rotation_range: tuple[float, float] = (-10.0, 10.0),
        warp_scale_range: tuple[float, float] = (0.95, 1.05),
        warp_translation_range: tuple[float, float] = (-0.05, 0.05),
        warp_mode: str = "affine",
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
            rotation_range: Deprecated compatibility argument (unused in affine mode).
            scale_range: Deprecated compatibility argument (unused in affine mode).
            brightness_range: Brightness adjustment range.
            contrast_range: Contrast adjustment range.
            saturation_range: Saturation adjustment range.
            hue_range: Hue adjustment range.
            warp_rotation_range: Rotation range for affine warp (degrees).
            warp_scale_range: Scale range for affine warp.
            warp_translation_range: Translation range for affine warp (relative).
            warp_mode: DALI warp mode (`affine` or `strict`).
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

        if warp_mode == "affine":
            # Legacy-compatible affine approximation available in DALI.
            src_warp_mat = fn.external_source(
                source=DALIAffineGenerator(
                    size=image_size,
                    rotation_range=warp_rotation_range,
                    scale_range=warp_scale_range,
                    translation_range=warp_translation_range,
                    seed=seed + 101,
                ),
                batch=False,
                device="cpu",
            )
            dst_warp_mat = fn.external_source(
                source=DALIAffineGenerator(
                    size=image_size,
                    rotation_range=warp_rotation_range,
                    scale_range=warp_scale_range,
                    translation_range=warp_translation_range,
                    seed=seed + 202,
                ),
                batch=False,
                device="cpu",
            )

            src_images = fn.warp_affine(
                src_images,
                matrix=src_warp_mat,
                fill_value=0,
                interp_type=types.INTERP_LINEAR,
            )
            dst_images = fn.warp_affine(
                dst_images,
                matrix=dst_warp_mat,
                fill_value=0,
                interp_type=types.INTERP_LINEAR,
            )
        else:
            raise ValueError(
                "DALI strict warp mode is unsupported. "
                "Use PyTorch backend for strict legacy warp parity."
            )

        # Random horizontal flip
        src_flip = fn.random.coin_flip(probability=flip_prob)
        dst_flip = fn.random.coin_flip(probability=flip_prob)
        src_images = fn.flip(src_images, horizontal=src_flip)
        dst_images = fn.flip(dst_images, horizontal=dst_flip)

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
    def face_swap_pipeline_external(
        external_source: FaceSwapExternalSource,
        image_size: int = 256,
        # Augmentation parameters
        flip_prob: float = 0.4,
        rotation_range: float = 10.0,
        scale_range: tuple[float, float] = (0.95, 1.05),
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
        saturation_range: tuple[float, float] = (0.9, 1.1),
        hue_range: float = 0.05,
        # Legacy-style affine warp augmentation
        warp_rotation_range: tuple[float, float] = (-10.0, 10.0),
        warp_scale_range: tuple[float, float] = (0.95, 1.05),
        warp_translation_range: tuple[float, float] = (-0.05, 0.05),
        warp_mode: str = "affine",
        # Pipeline settings
        seed: int = 42,
    ):
        """
        DALI pipeline with encoded bytes fed via external source.

        Used for legacy packed faceset inputs where training samples are not
        directly addressable as regular filesystem image files.
        """
        del rotation_range
        del scale_range

        src_jpegs, dst_jpegs = fn.external_source(
            source=external_source,
            num_outputs=2,
            batch=False,
            device="cpu",
        )

        src_images = fn.decoders.image(src_jpegs, device="mixed", output_type=types.RGB)
        dst_images = fn.decoders.image(dst_jpegs, device="mixed", output_type=types.RGB)

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

        if warp_mode == "affine":
            src_warp_mat = fn.external_source(
                source=DALIAffineGenerator(
                    size=image_size,
                    rotation_range=warp_rotation_range,
                    scale_range=warp_scale_range,
                    translation_range=warp_translation_range,
                    seed=seed + 101,
                ),
                batch=False,
                device="cpu",
            )
            dst_warp_mat = fn.external_source(
                source=DALIAffineGenerator(
                    size=image_size,
                    rotation_range=warp_rotation_range,
                    scale_range=warp_scale_range,
                    translation_range=warp_translation_range,
                    seed=seed + 202,
                ),
                batch=False,
                device="cpu",
            )

            src_images = fn.warp_affine(
                src_images,
                matrix=src_warp_mat,
                fill_value=0,
                interp_type=types.INTERP_LINEAR,
            )
            dst_images = fn.warp_affine(
                dst_images,
                matrix=dst_warp_mat,
                fill_value=0,
                interp_type=types.INTERP_LINEAR,
            )
        else:
            raise ValueError(
                "DALI strict warp mode is unsupported. "
                "Use PyTorch backend for strict legacy warp parity."
            )

        src_flip = fn.random.coin_flip(probability=flip_prob)
        dst_flip = fn.random.coin_flip(probability=flip_prob)
        src_images = fn.flip(src_images, horizontal=src_flip)
        dst_images = fn.flip(dst_images, horizontal=dst_flip)

        src_brightness = fn.random.uniform(range=brightness_range)
        dst_brightness = fn.random.uniform(range=brightness_range)
        src_images = fn.brightness(src_images, brightness=src_brightness)
        dst_images = fn.brightness(dst_images, brightness=dst_brightness)

        src_contrast = fn.random.uniform(range=contrast_range)
        dst_contrast = fn.random.uniform(range=contrast_range)
        src_images = fn.contrast(src_images, contrast=src_contrast)
        dst_images = fn.contrast(dst_images, contrast=dst_contrast)

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

        src_images = fn.transpose(src_images, perm=[2, 0, 1])
        dst_images = fn.transpose(dst_images, perm=[2, 0, 1])

        return src_images, dst_images

    @pipeline_def
    def face_swap_pipeline_simple(
        src_files: list[str],
        dst_files: list[str],
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

    @pipeline_def
    def face_swap_pipeline_simple_external(
        external_source: FaceSwapExternalSource,
        image_size: int = 256,
    ):
        """Simple DALI pipeline (no augment) for external-source encoded bytes."""
        src_jpegs, dst_jpegs = fn.external_source(
            source=external_source,
            num_outputs=2,
            batch=False,
            device="cpu",
        )

        src_images = fn.decoders.image(src_jpegs, device="mixed", output_type=types.RGB)
        dst_images = fn.decoders.image(dst_jpegs, device="mixed", output_type=types.RGB)

        src_images = fn.resize(src_images, size=[image_size, image_size])
        dst_images = fn.resize(dst_images, size=[image_size, image_size])

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

        src_images = fn.transpose(src_images, perm=[2, 0, 1])
        dst_images = fn.transpose(dst_images, perm=[2, 0, 1])

        return src_images, dst_images


def create_dali_iterator(
    src_dir: str | Path,
    dst_dir: str | Path,
    batch_size: int,
    image_size: int = 256,
    num_threads: int = 4,
    device_id: int = 0,
    augment: bool = True,
    seed: int = 42,
    shard_id: int = 0,
    num_shards: int = 1,
    warp_mode: str = "affine",
    allow_packed_faceset: bool = True,
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
        warp_mode: DALI warp mode (`affine` or `strict`).
        allow_packed_faceset: Allow loading legacy packed facesets (`faceset.pak`)
            when flat image files are missing.

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
    warp_mode = warp_mode.lower().strip()
    if warp_mode not in SUPPORTED_DALI_WARP_MODES:
        raise ValueError(
            f"Unsupported warp_mode={warp_mode!r}. "
            f"Use one of {sorted(SUPPORTED_DALI_WARP_MODES)}."
        )
    if augment and warp_mode == "strict":
        raise ValueError(
            "warp_mode='strict' is not supported by DALI; "
            "use PyTorch fallback for strict legacy warp."
        )

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not dst_dir.exists():
        raise FileNotFoundError(f"Destination directory not found: {dst_dir}")

    src_inputs = _resolve_dali_inputs(
        src_dir,
        allow_packed_faceset=allow_packed_faceset,
    )
    dst_inputs = _resolve_dali_inputs(
        dst_dir,
        allow_packed_faceset=allow_packed_faceset,
    )
    use_external_source = any(isinstance(item, FaceSample) for item in src_inputs) or any(
        isinstance(item, FaceSample) for item in dst_inputs
    )

    external_source: FaceSwapExternalSource | None = None
    if use_external_source:
        external_source = FaceSwapExternalSource(
            src_inputs,
            dst_inputs,
            batch_size=batch_size,
            shuffle=augment,
            seed=seed,
        )
        if augment:
            pipe = face_swap_pipeline_external(
                external_source=external_source,
                image_size=image_size,
                seed=seed,
                warp_mode=warp_mode,
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
            )
        else:
            pipe = face_swap_pipeline_simple_external(
                external_source=external_source,
                image_size=image_size,
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
            )
    else:
        src_files: list[str] = []
        for item in src_inputs:
            if isinstance(item, FaceSample):
                raise RuntimeError("Unexpected packed sample in file-reader mode.")
            src_files.append(str(Path(item)))

        dst_files: list[str] = []
        for item in dst_inputs:
            if isinstance(item, FaceSample):
                raise RuntimeError("Unexpected packed sample in file-reader mode.")
            dst_files.append(str(Path(item)))
        if augment:
            pipe = face_swap_pipeline(
                src_files=src_files,
                dst_files=dst_files,
                image_size=image_size,
                seed=seed,
                shard_id=shard_id,
                num_shards=num_shards,
                warp_mode=warp_mode,
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
    if external_source is not None:
        # Keep callback alive for external_source pipelines.
        pipe._visagen_external_source = external_source

    # Create iterator
    iterator_kwargs: dict[str, object] = {
        "output_map": ["src_images", "dst_images"],
        "last_batch_policy": LastBatchPolicy.DROP,
        "auto_reset": True,
    }
    if use_external_source and external_source is not None:
        iterator_kwargs["size"] = len(external_source)
    else:
        iterator_kwargs["reader_name"] = "src_reader"

    iterator = DALIGenericIterator([pipe], **iterator_kwargs)

    return iterator


def get_dataset_size(data_dir: str | Path) -> int:
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
