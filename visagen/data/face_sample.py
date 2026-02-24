"""
Face Sample Dataclass.

Lightweight container for face sample data with lazy image loading.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class FaceSample:
    """
    Face sample with metadata for training.

    Stores file path and metadata, loads image lazily on demand
    for memory efficiency.

    Attributes:
        filepath: Path to the aligned JPEG file.
        face_type: Face type string (e.g., "whole_face", "head").
        shape: Image dimensions (H, W, C).
        landmarks: 68-point facial landmarks in aligned image space.
        xseg_mask: Compressed segmentation mask bytes. Default: None.
        seg_ie_polys: Interactive segmentation polygon metadata. Default: None.
        eyebrows_expand_mod: Eyebrow expansion modifier. Default: 1.0.
        source_filename: Original source file name. Default: None.
        image_to_face_mat: Affine transform matrix (2, 3). Default: None.
        packed_faceset_path: Optional path to faceset.pak.
        packed_offset: Optional byte offset in packed faceset file.
        packed_size: Optional byte size in packed faceset file.

    Example:
        >>> sample = FaceSample(
        ...     filepath=Path("aligned/face.jpg"),
        ...     face_type="whole_face",
        ...     shape=(512, 512, 3),
        ...     landmarks=np.zeros((68, 2)),
        ... )
        >>> image = sample.load_image()
        >>> image.shape
        (512, 512, 3)
    """

    filepath: Path
    face_type: str
    shape: tuple[int, int, int]
    landmarks: np.ndarray
    xseg_mask: bytes | None = None
    seg_ie_polys: Any | None = None
    eyebrows_expand_mod: float = 1.0
    source_filename: str | None = None
    image_to_face_mat: np.ndarray | None = None
    packed_faceset_path: Path | None = None
    packed_offset: int | None = None
    packed_size: int | None = None
    _pitch_yaw_roll: tuple[float, float, float] | None = field(default=None, repr=False)

    def read_raw_bytes(self) -> bytes:
        """
        Read raw encoded image bytes from disk or packed faceset.

        Returns:
            Encoded image bytes.

        Raises:
            FileNotFoundError: If source file does not exist.
            ValueError: If packed faceset entry is invalid.
        """
        if (
            self.packed_faceset_path is not None
            and self.packed_offset is not None
            and self.packed_size is not None
        ):
            with open(self.packed_faceset_path, "rb") as f:
                f.seek(self.packed_offset, 0)
                image_bytes = f.read(self.packed_size)
            if len(image_bytes) != self.packed_size:
                raise ValueError(
                    "Invalid packed sample size at "
                    f"{self.packed_faceset_path}:{self.packed_offset}"
                )
            return image_bytes

        if not self.filepath.exists():
            raise FileNotFoundError(f"Image not found: {self.filepath}")
        return self.filepath.read_bytes()

    def load_image(self) -> np.ndarray:
        """
        Load BGR image from disk.

        Returns:
            Image as float32 array in range [0, 1], shape (H, W, C).

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        image_bytes = self.read_raw_bytes()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError(f"Failed to load image: {self.filepath}")

        return image.astype(np.float32) / 255.0

    def get_xseg_mask(self) -> np.ndarray | None:
        """
        Decompress and return segmentation mask.

        Returns:
            Mask as float32 array in range [0, 1], shape (H, W, 1),
            or None if no mask is available.
        """
        if self.xseg_mask is None:
            return None

        # Decode compressed mask
        mask = cv2.imdecode(
            np.frombuffer(self.xseg_mask, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            return None

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]

        return mask.astype(np.float32) / 255.0

    def get_pitch_yaw_roll(self) -> tuple[float, float, float]:
        """
        Get face pose estimation (pitch, yaw, roll).

        Computes on first call and caches the result.

        Returns:
            Tuple of (pitch, yaw, roll) in radians.
        """
        if self._pitch_yaw_roll is None:
            from visagen.vision.aligner import FaceAligner

            aligner = FaceAligner()
            self._pitch_yaw_roll = aligner.estimate_pitch_yaw_roll(
                self.landmarks, size=self.shape[0]
            )

        return self._pitch_yaw_roll

    @classmethod
    def from_face_image(cls, filepath: Path) -> Optional["FaceSample"]:
        """
        Create FaceSample from image file.

        Args:
            filepath: Path to JPEG file with embedded metadata.

        Returns:
            FaceSample instance, or None if file has no face metadata.
        """
        from visagen.vision.face_image import FaceImage

        try:
            image, metadata = FaceImage.load(filepath)
        except Exception:
            return None

        if metadata is None:
            return None

        return cls(
            filepath=filepath,
            face_type=metadata.face_type,
            shape=image.shape,
            landmarks=metadata.landmarks,
            xseg_mask=metadata.xseg_mask,
            seg_ie_polys=metadata.seg_ie_polys,
            eyebrows_expand_mod=metadata.eyebrows_expand_mod,
            source_filename=metadata.source_filename,
            image_to_face_mat=metadata.image_to_face_mat,
        )
