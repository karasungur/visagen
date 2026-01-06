"""
Sequence Dataset for Temporal Training.

Provides datasets for loading consecutive frame sequences for
temporal discriminator training.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceFaceDataset(Dataset):
    """
    Dataset for loading consecutive frame sequences.

    Loads T consecutive frames from video-extracted facesets.
    Assumes frames are named with sortable indices (e.g., 00001.jpg, 00002.jpg).

    Args:
        root_dir: Directory containing aligned face images.
        sequence_length: Number of consecutive frames per sample. Default: 5.
        target_size: Output image size. Default: 256.
        stride: Skip frames between sequences for data augmentation. Default: 1.
        transform: Optional augmentation transform.

    Example:
        >>> dataset = SequenceFaceDataset("aligned/", sequence_length=5)
        >>> sample = dataset[0]
        >>> sample['sequence'].shape
        torch.Size([3, 5, 256, 256])
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        stride: int = 1,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.stride = stride
        self.transform = transform

        # Scan and sort images by name
        self.image_paths = self._scan_directory()

        if len(self.image_paths) < sequence_length:
            raise ValueError(
                f"Not enough images in {root_dir}. "
                f"Found {len(self.image_paths)}, need at least {sequence_length}."
            )

        # Calculate valid sequence start indices
        self.valid_starts = self._compute_valid_starts()

    def _scan_directory(self) -> list[Path]:
        """Scan directory for images and sort by name."""
        extensions = {".jpg", ".jpeg", ".png"}
        paths = []

        for ext in extensions:
            paths.extend(self.root_dir.glob(f"*{ext}"))
            paths.extend(self.root_dir.glob(f"*{ext.upper()}"))

        # Sort by filename to maintain frame order
        return sorted(paths, key=lambda p: p.stem)

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
        # Load image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = image.astype(np.float32) / 255.0

        # Resize if needed
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

        # Convert to tensor and normalize to [-1, 1]
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
        stride: Skip frames between sequences. Default: 1.
        transform: Optional augmentation.

    Example:
        >>> dataset = PairedSequenceDataset("src/", "dst/", sequence_length=5)
        >>> src_seq, dst_seq = dataset[0]
        >>> src_seq.shape
        torch.Size([3, 5, 256, 256])
    """

    def __init__(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        stride: int = 1,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.src_dataset = SequenceFaceDataset(
            root_dir=src_dir,
            sequence_length=sequence_length,
            target_size=target_size,
            stride=stride,
            transform=transform,
        )

        self.dst_dataset = SequenceFaceDataset(
            root_dir=dst_dir,
            sequence_length=sequence_length,
            target_size=target_size,
            stride=stride,
            transform=transform,
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

    Example:
        >>> dataset = RandomSequenceDataset("faces/", num_samples=1000)
        >>> sample = dataset[0]
        >>> sample['sequence'].shape
        torch.Size([3, 5, 256, 256])
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 5,
        target_size: int = 256,
        num_samples: int = 1000,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.num_samples = num_samples
        self.transform = transform

        # Scan images
        self.image_paths = self._scan_directory()

        if len(self.image_paths) < sequence_length:
            raise ValueError(
                f"Not enough images. Found {len(self.image_paths)}, "
                f"need at least {sequence_length}."
            )

        # Max valid start index
        self.max_start = len(self.image_paths) - sequence_length

    def _scan_directory(self) -> list[Path]:
        """Scan directory for images and sort by name."""
        extensions = {".jpg", ".jpeg", ".png"}
        paths = []

        for ext in extensions:
            paths.extend(self.root_dir.glob(f"*{ext}"))
            paths.extend(self.root_dir.glob(f"*{ext.upper()}"))

        return sorted(paths, key=lambda p: p.stem)

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
        # Use idx as seed for reproducibility
        rng = np.random.default_rng(idx)
        start_idx = rng.integers(0, self.max_start + 1)

        frames = []
        for i in range(self.sequence_length):
            frame_path = self.image_paths[start_idx + i]
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
