"""
Face Dataset for Training.

Load DFL-aligned face images with metadata for training.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from visagen.data.face_sample import FaceSample
from visagen.vision.face_type import FaceType

logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """
    Dataset for DFL-aligned face images.

    Loads face images from a directory containing JPEG files with
    embedded DFL metadata. Supports lazy loading and augmentation.

    Args:
        root_dir: Directory containing aligned face JPEGs.
        transform: Optional augmentation transform.
        target_size: Output image size. Default: 256.
        face_type_filter: Only load faces of this type. Default: None (all).
        with_mask: Load segmentation masks if available. Default: True.
        preload_metadata: Load all metadata on init. Default: True.
        uniform_yaw: Enable uniform yaw sampling for balanced pose distribution.
            Default: False.
        yaw_bins: Number of yaw bins for uniform sampling. Default: 10.

    Example:
        >>> dataset = FaceDataset(Path("aligned_faces/"))
        >>> sample = dataset[0]
        >>> sample['image'].shape
        torch.Size([3, 256, 256])
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        target_size: int = 256,
        face_type_filter: FaceType | None = None,
        with_mask: bool = True,
        preload_metadata: bool = True,
        uniform_yaw: bool = False,
        yaw_bins: int = 10,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.face_type_filter = face_type_filter
        self.with_mask = with_mask
        self.uniform_yaw = uniform_yaw
        self.yaw_bins = yaw_bins

        # Scan directory for image files
        self.image_paths = self.scan_directory(self.root_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

        # Load metadata
        self.samples: list[FaceSample] = []
        if preload_metadata:
            self._preload_metadata()
        else:
            # Create placeholder samples with paths only
            self.samples = [
                FaceSample(
                    filepath=path,
                    face_type="unknown",
                    shape=(0, 0, 0),
                    landmarks=np.zeros((68, 2)),
                )
                for path in self.image_paths
            ]

        # Build yaw bins for uniform sampling
        self._yaw_bins_indices: list[list[int]] | None = None
        if self.uniform_yaw and preload_metadata:
            self._build_yaw_bins()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample with retry logic for corrupt images.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys:
                - 'image': (C, H, W) tensor in [-1, 1]
                - 'mask': (1, H, W) tensor in [0, 1] (if with_mask and available)
                - 'landmarks': (68, 2) tensor
                - 'face_type': int (FaceType enum value)
        """
        MAX_RETRIES = 3

        for retry in range(MAX_RETRIES):
            try:
                current_idx = (idx + retry) % len(self.samples)
                sample = self.samples[current_idx]

                # Load image with validation
                image = sample.load_image()  # (H, W, C) float32 [0, 1]

                if image is None:
                    raise ValueError(f"Image is None at {sample.filepath}")
                if not np.isfinite(image).all():
                    raise ValueError(f"Image contains NaN/Inf at {sample.filepath}")
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"Invalid image shape: {image.shape}")

                # Resize if needed
                h, w = image.shape[:2]
                if h != self.target_size or w != self.target_size:
                    image = cv2.resize(
                        image,
                        (self.target_size, self.target_size),
                        interpolation=cv2.INTER_CUBIC,
                    )

                # Convert BGR to RGB and transpose to CHW
                image = image[:, :, ::-1].copy()  # BGR to RGB
                image = np.transpose(image, (2, 0, 1))  # HWC to CHW
                image_tensor = torch.from_numpy(image)

                # Load mask if available
                mask_tensor = None
                if self.with_mask:
                    mask = sample.get_xseg_mask()
                    if mask is not None:
                        if (
                            mask.shape[0] != self.target_size
                            or mask.shape[1] != self.target_size
                        ):
                            mask = cv2.resize(
                                mask,
                                (self.target_size, self.target_size),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        if mask.ndim == 2:
                            mask = mask[..., np.newaxis]
                        mask = np.transpose(mask, (2, 0, 1))
                        mask_tensor = torch.from_numpy(mask)

                # Apply augmentation if provided
                if self.transform is not None:
                    image_tensor, mask_tensor = self.transform(
                        image_tensor, mask_tensor
                    )

                # Normalize to [-1, 1]
                if image_tensor.max() <= 1.0:
                    image_tensor = image_tensor * 2 - 1

                # Build output dict
                output = {
                    "image": image_tensor,
                    "landmarks": torch.from_numpy(sample.landmarks.copy()),
                    "face_type": self._get_face_type_int(sample.face_type),
                }

                if mask_tensor is not None:
                    output["mask"] = mask_tensor

                return output

            except Exception as e:
                if retry == 0:
                    logger.warning(f"Failed to load sample {idx}: {e}")
                if retry == MAX_RETRIES - 1:
                    logger.error(f"All retries failed for sample {idx}")
                    # Return blank sample to prevent crash
                    return {
                        "image": torch.zeros(3, self.target_size, self.target_size)
                        - 1.0,
                        "landmarks": torch.zeros(68, 2),
                        "face_type": 0,
                    }

        # Should never reach here, but just in case
        return {
            "image": torch.zeros(3, self.target_size, self.target_size) - 1.0,
            "landmarks": torch.zeros(68, 2),
            "face_type": 0,
        }

    def _preload_metadata(self) -> None:
        """Preload metadata from all images."""
        for path in self.image_paths:
            sample = FaceSample.from_dfl_image(path)

            if sample is None:
                # Skip images without DFL metadata
                continue

            # Apply face type filter
            if self.face_type_filter is not None:
                try:
                    sample_type = FaceType.from_string(sample.face_type)
                    if sample_type != self.face_type_filter:
                        continue
                except ValueError:
                    continue

            self.samples.append(sample)

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid DFL images found in {self.root_dir}. "
                "Make sure images have embedded DFL metadata."
            )

    def _build_yaw_bins(self) -> None:
        """
        Build yaw bin indices for uniform sampling.

        Divides samples into bins based on yaw angle for balanced pose distribution.
        """
        import math

        # Initialize bins
        self._yaw_bins_indices = [[] for _ in range(self.yaw_bins)]

        # Yaw range: -pi/2 to pi/2 (left to right profile)
        yaw_min = -math.pi / 2
        yaw_max = math.pi / 2
        bin_width = (yaw_max - yaw_min) / self.yaw_bins

        for idx, sample in enumerate(self.samples):
            try:
                _, yaw, _ = sample.get_pitch_yaw_roll()
                # Clamp yaw to valid range
                yaw = max(yaw_min, min(yaw_max - 1e-6, yaw))
                bin_idx = int((yaw - yaw_min) / bin_width)
                bin_idx = max(0, min(self.yaw_bins - 1, bin_idx))
                self._yaw_bins_indices[bin_idx].append(idx)
            except Exception:
                # If pose estimation fails, add to middle bin
                middle_bin = self.yaw_bins // 2
                self._yaw_bins_indices[middle_bin].append(idx)

        # Remove empty bins and log warnings
        non_empty_bins = [b for b in self._yaw_bins_indices if len(b) > 0]
        removed_count = self.yaw_bins - len(non_empty_bins)

        if removed_count > 0:
            logger.warning(f"Removed {removed_count}/{self.yaw_bins} empty yaw bins")

        if len(non_empty_bins) == 0:
            logger.error("All yaw bins empty! Falling back to random sampling.")
            self._yaw_bins_indices = None
        else:
            self._yaw_bins_indices = non_empty_bins

    def sample_uniform_yaw(self) -> int:
        """
        Sample an index with uniform yaw distribution.

        First selects a random yaw bin, then selects a random sample from that bin.
        This ensures balanced representation of different head poses.

        Returns:
            Sample index.
        """
        if self._yaw_bins_indices is None or len(self._yaw_bins_indices) == 0:
            return np.random.randint(len(self.samples))

        # Extra safety: filter empty bins at runtime
        non_empty = [b for b in self._yaw_bins_indices if len(b) > 0]
        if not non_empty:
            return np.random.randint(len(self.samples))

        # Select random bin
        bin_idx = np.random.randint(len(non_empty))
        # Select random sample from bin
        sample_idx = np.random.choice(non_empty[bin_idx])
        return sample_idx

    @staticmethod
    def scan_directory(root_dir: Path) -> list[Path]:
        """
        Find all image files in directory.

        Args:
            root_dir: Directory to scan.

        Returns:
            List of image file paths, sorted alphabetically.
        """
        extensions = {".jpg", ".jpeg", ".png"}
        paths = []

        for ext in extensions:
            paths.extend(root_dir.glob(f"*{ext}"))
            paths.extend(root_dir.glob(f"*{ext.upper()}"))

        return sorted(paths)

    @staticmethod
    def _get_face_type_int(face_type_str: str) -> int:
        """Convert face type string to int."""
        try:
            return FaceType.from_string(face_type_str).value
        except (ValueError, AttributeError):
            return FaceType.WHOLE_FACE.value


class SimpleFaceDataset(Dataset):
    """
    Simple dataset for loading face images without DFL metadata.

    Just loads images from a directory and resizes them.
    Useful for testing or when metadata is not needed.

    Args:
        root_dir: Directory containing images.
        target_size: Output size. Default: 256.
        transform: Optional transform.
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_size: int = 256,
        transform: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform

        self.image_paths = FaceDataset.scan_directory(self.root_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx]

        # Load image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = image.astype(np.float32) / 255.0

        # Resize
        image = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC,
        )

        # BGR to RGB, HWC to CHW
        image = image[:, :, ::-1].copy()
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        # Apply transform
        if self.transform is not None:
            image_tensor, _ = self.transform(image_tensor, None)

        # Normalize to [-1, 1]
        if image_tensor.max() <= 1.0:
            image_tensor = image_tensor * 2 - 1

        return {"image": image_tensor}
