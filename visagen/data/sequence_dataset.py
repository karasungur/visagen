"""
Sequence Dataset for Temporal Training.

Provides datasets for loading consecutive frame sequences for
temporal discriminator training.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _scan_image_paths(root_dir: Path) -> list[Path]:
    """Scan a flat directory for supported image files."""
    extensions = {".jpg", ".jpeg", ".png"}
    paths: list[Path] = []

    for ext in extensions:
        paths.extend(root_dir.glob(f"*{ext}"))
        paths.extend(root_dir.glob(f"*{ext.upper()}"))

    return sorted(set(paths))


def _source_filename_sort_key(path: Path) -> tuple[str, str]:
    """Sort key using DFL metadata source filename with filename fallback."""
    fallback = path.stem
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return fallback, path.name

    try:
        from visagen.vision.dflimg import DFLImage

        with open(path, "rb") as f:
            data = f.read()
        parsed = DFLImage._parse_jpeg_metadata(data)
        if isinstance(parsed, dict):
            source_filename = str(parsed.get("source_filename") or fallback)
            return source_filename, path.name
    except Exception:
        pass

    return fallback, path.name


def _sort_image_paths(
    paths: list[Path],
    sort_mode: Literal["source_filename", "stem"],
) -> list[Path]:
    """Sort image paths according to requested temporal ordering mode."""
    if sort_mode == "source_filename":
        return sorted(paths, key=_source_filename_sort_key)
    return sorted(paths, key=lambda p: (p.stem, p.name))


class SequenceFaceDataset(Dataset):
    """
    Dataset for loading consecutive frame sequences.

    Loads T consecutive frames from video-extracted facesets.

    Args:
        root_dir: Directory containing aligned face images.
        sequence_length: Number of consecutive frames per sample. Default: 5.
        target_size: Output image size. Default: 256.
        stride: Skip frames between sequence starts. Default: 1.
        transform: Optional augmentation transform.
        sort_mode: Temporal ordering source:
            - "source_filename": Prefer DFL metadata source filename (legacy parity)
            - "stem": Filename stem order fallback behavior
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        stride: int = 1,
        transform: Callable | None = None,
        sort_mode: Literal["source_filename", "stem"] = "source_filename",
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.stride = stride
        self.transform = transform
        self.sort_mode = sort_mode

        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")

        # Scan and sort images by requested temporal ordering mode.
        self.image_paths = self._scan_directory()

        if len(self.image_paths) < sequence_length:
            raise ValueError(
                f"Not enough images in {root_dir}. "
                f"Found {len(self.image_paths)}, need at least {sequence_length}."
            )

        # Calculate valid sequence start indices
        self.valid_starts = self._compute_valid_starts()

    def _scan_directory(self) -> list[Path]:
        """Scan directory for images and sort according to `sort_mode`."""
        paths = _scan_image_paths(self.root_dir)
        return _sort_image_paths(paths, self.sort_mode)

    def _compute_valid_starts(self) -> list[int]:
        """Compute valid sequence start indices based on stride."""
        max_start = len(self.image_paths) - self.sequence_length
        return list(range(0, max_start + 1, self.stride))

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sequence of frames.

        Args:
            idx: Sequence index.

        Returns:
            Dict with 'sequence': (C, T, H, W) tensor in [-1, 1].
        """
        start_idx = self.valid_starts[idx]
        frames = []

        for i in range(self.sequence_length):
            frame_path = self.image_paths[start_idx + i]
            frame = self._load_frame(frame_path)
            frames.append(frame)

        # Stack frames: list of (C, H, W) -> (C, T, H, W)
        sequence = torch.stack(frames, dim=1)

        # Apply transform if provided
        if self.transform is not None:
            sequence = self.transform(sequence)

        return {"sequence": sequence}

    def _load_frame(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single frame."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        if h != self.target_size or w != self.target_size:
            image = cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_CUBIC,
            )

        image = image[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image)
        tensor = tensor * 2 - 1

        return tensor


class PairedSequenceDataset(Dataset):
    """
    Paired src/dst sequence dataset for temporal training.

    Returns matching sequences from source and destination directories.
    Sequences are aligned by frame index.

    Args:
        src_dir: Directory containing source face sequences.
        dst_dir: Directory containing destination face sequences.
        sequence_length: Number of consecutive frames. Default: 5.
        target_size: Output size. Default: 256.
        stride: Skip frames between sequence starts. Default: 1.
        transform: Optional augmentation.
        sort_mode: Temporal ordering mode for both src and dst.
    """

    def __init__(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        stride: int = 1,
        transform: Callable | None = None,
        sort_mode: Literal["source_filename", "stem"] = "source_filename",
    ) -> None:
        super().__init__()

        self.src_dataset = SequenceFaceDataset(
            root_dir=src_dir,
            sequence_length=sequence_length,
            target_size=target_size,
            stride=stride,
            transform=transform,
            sort_mode=sort_mode,
        )

        self.dst_dataset = SequenceFaceDataset(
            root_dir=dst_dir,
            sequence_length=sequence_length,
            target_size=target_size,
            stride=stride,
            transform=transform,
            sort_mode=sort_mode,
        )

        # Use minimum length to ensure paired access
        self._length = min(len(self.src_dataset), len(self.dst_dataset))

    def __len__(self) -> int:
        """Return number of paired sequences."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired sequence.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (src_sequence, dst_sequence) each (C, T, H, W).
        """
        src_sample = self.src_dataset[idx]
        dst_sample = self.dst_dataset[idx]

        return src_sample["sequence"], dst_sample["sequence"]


class RandomSequenceDataset(Dataset):
    """
    Random sequence dataset for unpaired temporal training.

    Returns random sequences from a single directory.
    Useful when training on a single video's frames.

    Args:
        root_dir: Directory containing face images.
        sequence_length: Number of consecutive frames. Default: 5.
        target_size: Output size. Default: 256.
        num_samples: Number of random samples to generate. Default: 1000.
        transform: Optional augmentation.
        sort_mode: Temporal ordering mode for scanned frame list.
        stride_mode: Sequence stride behavior:
            - "fixed": contiguous frames (legacy-disabled behavior)
            - "random": random temporal multiplier in [1, max_stride]
        max_stride: Maximum temporal multiplier for random stride mode.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        num_samples: int = 1000,
        transform: Callable | None = None,
        sort_mode: Literal["source_filename", "stem"] = "source_filename",
        stride_mode: Literal["fixed", "random"] = "random",
        max_stride: int = 4,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.num_samples = num_samples
        self.transform = transform
        self.sort_mode = sort_mode
        self.stride_mode = stride_mode
        self.max_stride = max_stride

        if self.max_stride < 1:
            raise ValueError(f"max_stride must be >= 1, got {self.max_stride}")

        # Scan images
        self.image_paths = self._scan_directory()

        if len(self.image_paths) < sequence_length:
            raise ValueError(
                f"Not enough images. Found {len(self.image_paths)}, "
                f"need at least {sequence_length}."
            )

        if self.stride_mode == "fixed":
            self.max_start = self._max_start_for_stride(1)
        elif self.stride_mode == "random":
            self.max_start = self._max_start_for_stride(self.max_stride)
            if self.max_start < 0:
                raise ValueError("Not enough samples to fit temporal line.")
        else:
            raise ValueError(
                f"Unsupported stride_mode={self.stride_mode!r}. "
                "Use 'fixed' or 'random'."
            )

    def _scan_directory(self) -> list[Path]:
        """Scan directory for images and sort according to `sort_mode`."""
        paths = _scan_image_paths(self.root_dir)
        return _sort_image_paths(paths, self.sort_mode)

    def _max_start_for_stride(self, stride: int) -> int:
        """Maximum valid start index for a given intra-sequence stride."""
        return len(self.image_paths) - (self.sequence_length * stride - (stride - 1))

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a random sequence.

        Args:
            idx: Sample index (used for reproducibility with fixed seed).

        Returns:
            Dict with 'sequence': (C, T, H, W) tensor.
        """
        rng = np.random.default_rng(idx)

        stride = 1
        if self.stride_mode == "random":
            stride = int(rng.integers(1, self.max_stride + 1))

        max_start = self._max_start_for_stride(stride)
        if max_start < 0:
            raise ValueError("Not enough images for selected temporal stride.")

        start_idx = int(rng.integers(0, max_start + 1))

        frames = []
        for i in range(self.sequence_length):
            frame_path = self.image_paths[start_idx + i * stride]
            frame = self._load_frame(frame_path)
            frames.append(frame)

        sequence = torch.stack(frames, dim=1)

        if self.transform is not None:
            sequence = self.transform(sequence)

        return {"sequence": sequence}

    def _load_frame(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single frame."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        if h != self.target_size or w != self.target_size:
            image = cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_CUBIC,
            )

        image = image[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image)
        tensor = tensor * 2 - 1

        return tensor
