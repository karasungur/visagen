"""
Mask Dataset for SegFormer Fine-tuning.

Provides dataset class for loading image-mask pairs
with augmentation support for LoRA training.

Features:
- Binary and multi-class mask modes
- Configurable label mapping
- 8x data augmentation
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor


class MaskMode(Enum):
    """Mask training mode."""

    BINARY = "binary"  # 0=background, 1=face (default)
    MULTICLASS = "multiclass"  # Full 19-class CelebAMask labels


@dataclass
class MaskSample:
    """
    A single image-mask training sample.

    Attributes:
        image_path: Path to the face image.
        mask_path: Path to the corresponding mask.
    """

    image_path: Path
    mask_path: Path

    def load_image(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_mask(self) -> np.ndarray:
        """Load mask as single-channel numpy array."""
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {self.mask_path}")
        return mask


class MaskDataset(Dataset):
    """
    Dataset for SegFormer mask fine-tuning.

    Expected directory structure:
        samples_dir/
        ├── images/
        │   ├── sample_0001.jpg
        │   ├── sample_0002.jpg
        │   └── ...
        └── masks/
            ├── sample_0001.png
            ├── sample_0002.png
            └── ...

    Args:
        samples_dir: Directory containing images/ and masks/ subdirectories.
        processor: SegFormer image processor for preprocessing.
        augment: Whether to apply data augmentation. Default: True.
        target_size: Target image size. Default: 512.
        mask_mode: BINARY or MULTICLASS. Default: BINARY.
        label_mapping: Optional dict to remap mask label values.

    Example:
        >>> processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        >>> dataset = MaskDataset("./training_samples", processor)
        >>> sample = dataset[0]
        >>> print(sample["pixel_values"].shape)  # (3, 512, 512)
    """

    # Augmentation variants (9x data expansion with background mixing)
    NUM_AUGMENT_VARIANTS = 9

    def __init__(
        self,
        samples_dir: Path | str,
        processor: SegformerImageProcessor,
        augment: bool = True,
        target_size: int = 512,
        mask_mode: MaskMode = MaskMode.BINARY,
        label_mapping: dict[int, int] | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.processor = processor
        self.augment = augment
        self.target_size = target_size
        self.mask_mode = mask_mode
        self.label_mapping = label_mapping or {}

        # Validate directory structure
        self.images_dir = self.samples_dir / "images"
        self.masks_dir = self.samples_dir / "masks"

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        # Collect samples
        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {samples_dir}")

    def _collect_samples(self) -> list[MaskSample]:
        """Collect matching image-mask pairs."""
        samples = []

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png"}
        for image_path in self.images_dir.iterdir():
            if image_path.suffix.lower() not in image_extensions:
                continue

            # Find corresponding mask
            stem = image_path.stem
            mask_path = None

            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = self.masks_dir / f"{stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is not None:
                samples.append(MaskSample(image_path=image_path, mask_path=mask_path))

        return sorted(samples, key=lambda s: s.image_path.name)

    def __len__(self) -> int:
        """Return dataset length (with augmentation expansion)."""
        base_len = len(self.samples)
        if self.augment:
            return base_len * self.NUM_AUGMENT_VARIANTS
        return base_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - pixel_values: Preprocessed image tensor (3, H, W).
                - labels: Mask tensor (H, W) with class indices.
        """
        if self.augment:
            # Calculate base sample and augmentation variant
            base_idx = idx // self.NUM_AUGMENT_VARIANTS
            variant = idx % self.NUM_AUGMENT_VARIANTS
        else:
            base_idx = idx
            variant = 0

        sample = self.samples[base_idx]

        # Load image and mask
        image = sample.load_image()
        mask = sample.load_mask()

        # Resize to target size
        image = cv2.resize(image, (self.target_size, self.target_size))
        mask = cv2.resize(
            mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST
        )

        # Apply augmentation
        if self.augment and variant > 0:
            image, mask = self._augment(image, mask, variant)

        # Convert mask to class labels based on mode
        if self.mask_mode == MaskMode.BINARY:
            # Binary: 255 -> 1 (face), 0 -> 0 (background)
            labels = (mask > 127).astype(np.int64)
        else:
            # Multi-class: mask values are class indices
            labels = mask.astype(np.int64)
            # Apply label remapping if specified
            for src, dst in self.label_mapping.items():
                labels[labels == src] = dst

        # Process image for SegFormer
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Convert labels to tensor
        labels_tensor = torch.from_numpy(labels).long()

        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor,
        }

    def _augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        variant: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation based on variant index.

        Augmentation variants:
            0: Original (no augmentation)
            1: Horizontal flip
            2: Brightness/contrast adjustment
            3: Flip + brightness
            4: Small rotation (±10°)
            5: Flip + rotation
            6: Brightness + rotation
            7: All combined
            8: Background mixing (if multiple samples available)

        Args:
            image: Input RGB image.
            mask: Input mask.
            variant: Augmentation variant (0-8).

        Returns:
            Tuple of (augmented_image, augmented_mask).
        """
        aug_image = image.copy()
        aug_mask = mask.copy()

        # Background mixing (variant 8)
        if variant == 8:
            if len(self.samples) > 1:
                aug_image = self._apply_background_mixing(aug_image, aug_mask)
            return aug_image, aug_mask

        # Horizontal flip (variants 1, 3, 5, 7)
        if variant in [1, 3, 5, 7]:
            aug_image = cv2.flip(aug_image, 1)
            aug_mask = cv2.flip(aug_mask, 1)

        # Brightness/contrast (variants 2, 3, 6, 7)
        if variant in [2, 3, 6, 7]:
            aug_image = self._adjust_brightness_contrast(aug_image)

        # Rotation (variants 4, 5, 6, 7)
        if variant in [4, 5, 6, 7]:
            angle = np.random.uniform(-10, 10)
            aug_image, aug_mask = self._rotate(aug_image, aug_mask, angle)

        return aug_image, aug_mask

    def _apply_background_mixing(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Mix background from another random image.

        Creates synthetic training variety by replacing background
        while keeping the face region intact.

        Args:
            image: Input RGB image.
            mask: Binary mask (255 = face, 0 = background).

        Returns:
            Image with mixed background.
        """
        h, w = image.shape[:2]

        # Select random background image
        bg_idx = np.random.randint(len(self.samples))
        bg_sample = self.samples[bg_idx]
        bg_image = bg_sample.load_image()
        bg_image = cv2.resize(bg_image, (self.target_size, self.target_size))

        # Random transform background for variety
        angle = np.random.uniform(-180, 180)
        scale = np.random.uniform(0.8, 1.2)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        bg_transformed = cv2.warpAffine(
            bg_image, M, (w, h), borderMode=cv2.BORDER_REFLECT
        )

        # Create float mask for blending
        mask_float = mask.astype(np.float32) / 255.0
        if mask_float.ndim == 2:
            mask_float = mask_float[..., np.newaxis]

        # Blend ratio for subtle effect
        blend_ratio = 0.15 + np.random.rand() * 0.85

        # Extract background from transformed image
        bg_only = bg_transformed * (1 - mask_float)

        # Blend: keep face region, mix background
        result = (
            image * mask_float
            + image * (1 - mask_float) * blend_ratio
            + bg_only * (1 - blend_ratio)
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast adjustment."""
        # Random brightness delta (-30 to +30)
        brightness = np.random.uniform(-30, 30)

        # Random contrast factor (0.8 to 1.2)
        contrast = np.random.uniform(0.8, 1.2)

        # Apply adjustments
        image = image.astype(np.float32)
        image = image * contrast + brightness
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def _rotate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate image (bilinear interpolation)
        rotated_image = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

        # Rotate mask (nearest neighbor to preserve labels)
        rotated_mask = cv2.warpAffine(
            mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )

        return rotated_image, rotated_mask

    def get_base_sample_count(self) -> int:
        """Get number of base samples (without augmentation)."""
        return len(self.samples)

    def get_sample_paths(self) -> list[tuple[Path, Path]]:
        """Get list of (image_path, mask_path) tuples."""
        return [(s.image_path, s.mask_path) for s in self.samples]


class MaskDataModule:
    """
    Data module for SegFormer mask training.

    Provides train/val dataloaders with proper configuration.

    Args:
        samples_dir: Directory containing training samples.
        processor: SegFormer image processor.
        batch_size: Batch size for training. Default: 4.
        val_split: Fraction of data for validation. Default: 0.1.
        num_workers: DataLoader workers. Default: 0.
        target_size: Target image size. Default: 512.
    """

    def __init__(
        self,
        samples_dir: Path | str,
        processor: SegformerImageProcessor,
        batch_size: int = 4,
        val_split: float = 0.1,
        num_workers: int = 0,
        target_size: int = 512,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.processor = processor
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.target_size = target_size

        self._train_dataset: MaskDataset | None = None
        self._val_dataset: MaskDataset | None = None

    def setup(self) -> None:
        """Set up train and validation datasets."""
        # Create full dataset (without augmentation for splitting)
        full_dataset = MaskDataset(
            samples_dir=self.samples_dir,
            processor=self.processor,
            augment=False,
            target_size=self.target_size,
        )

        # Split into train/val
        n_samples = len(full_dataset.samples)
        n_val = max(1, int(n_samples * self.val_split))
        n_train = n_samples - n_val

        # Split samples
        train_samples = full_dataset.samples[:n_train]
        val_samples = full_dataset.samples[n_train:]

        # Create separate directories or use the samples directly
        # For simplicity, we'll create new dataset instances
        self._train_dataset = _SampleListDataset(
            samples=train_samples,
            processor=self.processor,
            augment=True,
            target_size=self.target_size,
        )

        self._val_dataset = _SampleListDataset(
            samples=val_samples,
            processor=self.processor,
            augment=False,
            target_size=self.target_size,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training dataloader."""
        if self._train_dataset is None:
            self.setup()

        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation dataloader."""
        if self._val_dataset is None:
            self.setup()

        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class _SampleListDataset(Dataset):
    """Internal dataset class that works with a list of MaskSample objects."""

    NUM_AUGMENT_VARIANTS = 9

    def __init__(
        self,
        samples: list[MaskSample],
        processor: SegformerImageProcessor,
        augment: bool = True,
        target_size: int = 512,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.augment = augment
        self.target_size = target_size

    def __len__(self) -> int:
        base_len = len(self.samples)
        if self.augment:
            return base_len * self.NUM_AUGMENT_VARIANTS
        return base_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.augment:
            base_idx = idx // self.NUM_AUGMENT_VARIANTS
            variant = idx % self.NUM_AUGMENT_VARIANTS
        else:
            base_idx = idx
            variant = 0

        sample = self.samples[base_idx]

        # Load and process
        image = sample.load_image()
        mask = sample.load_mask()

        image = cv2.resize(image, (self.target_size, self.target_size))
        mask = cv2.resize(
            mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST
        )

        # Apply augmentation
        if self.augment and variant > 0:
            image, mask = MaskDataset._augment(None, image, mask, variant)

        labels = (mask > 127).astype(np.int64)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels_tensor = torch.from_numpy(labels).long()

        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor,
        }


def create_samples_directory(base_dir: Path | str) -> tuple[Path, Path]:
    """
    Create the expected directory structure for training samples.

    Args:
        base_dir: Base directory for samples.

    Returns:
        Tuple of (images_dir, masks_dir).
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / "images"
    masks_dir = base_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    return images_dir, masks_dir


def save_training_sample(
    samples_dir: Path | str,
    image: np.ndarray,
    mask: np.ndarray,
    sample_id: str,
) -> tuple[Path, Path]:
    """
    Save an image-mask pair as a training sample.

    Args:
        samples_dir: Base directory for samples.
        image: Face image (BGR format).
        mask: Binary mask (0 or 255).
        sample_id: Unique identifier for the sample.

    Returns:
        Tuple of (image_path, mask_path).
    """
    samples_dir = Path(samples_dir)
    images_dir, masks_dir = create_samples_directory(samples_dir)

    image_path = images_dir / f"{sample_id}.jpg"
    mask_path = masks_dir / f"{sample_id}.png"

    cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(mask_path), mask)

    return image_path, mask_path
