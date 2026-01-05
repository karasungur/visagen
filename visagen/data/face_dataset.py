"""
Face Dataset for Training.

Load DFL-aligned face images with metadata for training.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from visagen.data.face_sample import FaceSample
from visagen.vision.face_type import FaceType


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
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.face_type_filter = face_type_filter
        self.with_mask = with_mask

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

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys:
                - 'image': (C, H, W) tensor in [-1, 1]
                - 'mask': (1, H, W) tensor in [0, 1] (if with_mask and available)
                - 'landmarks': (68, 2) tensor
                - 'face_type': int (FaceType enum value)
        """
        sample = self.samples[idx]

        # Load image
        image = sample.load_image()  # (H, W, C) float32 [0, 1]

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
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

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
