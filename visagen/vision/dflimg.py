"""
DFL Image Metadata Handler.

Read and write DeepFaceLab metadata embedded in JPEG APP15 chunks.
Provides backward compatibility with legacy DFLJPG format.
"""

from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from visagen.vision.polys import PolygonSet


@dataclass
class FaceMetadata:
    """
    Face metadata stored in DFL images.

    Attributes:
        landmarks: 68-point facial landmarks in aligned image space.
        source_landmarks: Original landmarks in source image space.
        source_rect: Bounding box [x1, y1, x2, y2] in source image.
        source_filename: Original source file name.
        face_type: Face type string (e.g., "whole_face", "head").
        image_to_face_mat: Affine transform matrix (2, 3).
        eyebrows_expand_mod: Eyebrow expansion modifier. Default: 1.0.
        xseg_mask: Compressed segmentation mask bytes. Default: None.
        seg_ie_polys: Interactive editor polygon data. Default: None.
    """

    landmarks: np.ndarray
    source_landmarks: np.ndarray
    source_rect: tuple[int, int, int, int]
    source_filename: str
    face_type: str
    image_to_face_mat: np.ndarray
    eyebrows_expand_mod: float = 1.0
    xseg_mask: bytes | None = None
    seg_ie_polys: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy DFL dictionary format."""
        data = {
            "landmarks": self.landmarks.tolist()
            if isinstance(self.landmarks, np.ndarray)
            else self.landmarks,
            "source_landmarks": self.source_landmarks.tolist()
            if isinstance(self.source_landmarks, np.ndarray)
            else self.source_landmarks,
            "source_rect": list(self.source_rect),
            "source_filename": self.source_filename,
            "face_type": self.face_type,
            "image_to_face_mat": self.image_to_face_mat.tolist()
            if isinstance(self.image_to_face_mat, np.ndarray)
            else self.image_to_face_mat,
            "eyebrows_expand_mod": self.eyebrows_expand_mod,
        }

        if self.xseg_mask is not None:
            data["xseg_mask"] = self.xseg_mask

        if self.seg_ie_polys is not None:
            data["seg_ie_polys"] = self.seg_ie_polys

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceMetadata:
        """Create from legacy DFL dictionary format."""
        return cls(
            landmarks=np.array(data["landmarks"]),
            source_landmarks=np.array(data.get("source_landmarks", data["landmarks"])),
            source_rect=tuple(data.get("source_rect", [0, 0, 0, 0])),
            source_filename=data.get("source_filename", ""),
            face_type=data.get("face_type", "whole_face"),
            image_to_face_mat=np.array(data["image_to_face_mat"])
            if data.get("image_to_face_mat")
            else np.eye(2, 3),
            eyebrows_expand_mod=data.get("eyebrows_expand_mod", 1.0),
            xseg_mask=data.get("xseg_mask"),
            seg_ie_polys=data.get("seg_ie_polys"),
        )


class DFLImage:
    """
    Read and write DFL metadata in JPEG APP15 chunks.

    Provides compatibility with legacy DeepFaceLab image format
    while using modern Python patterns.

    Example:
        >>> # Load existing DFL image
        >>> image, metadata = DFLImage.load(Path("aligned_face.jpg"))
        >>> print(metadata.face_type)
        'whole_face'

        >>> # Save with metadata
        >>> DFLImage.save(Path("output.jpg"), image, metadata)
    """

    # JPEG markers
    _SOI = 0xD8  # Start of image
    _EOI = 0xD9  # End of image
    _SOS = 0xDA  # Start of scan
    _SOF0 = 0xC0  # Baseline DCT
    _SOF2 = 0xC2  # Progressive DCT
    _APP15 = 0xEF  # Application segment 15 (DFL data)

    @staticmethod
    def load(filepath: Path) -> tuple[np.ndarray, FaceMetadata | None]:
        """
        Load DFL image with metadata.

        Args:
            filepath: Path to JPEG file.

        Returns:
            Tuple of (image as BGR numpy array, metadata or None).

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is not a valid JPEG.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read raw bytes
        with open(filepath, "rb") as f:
            data = f.read()

        # Parse JPEG chunks
        dfl_dict = DFLImage._parse_jpeg_metadata(data)

        # Load image with OpenCV
        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")

        # Convert dict to metadata if present
        metadata = None
        if dfl_dict and "landmarks" in dfl_dict:
            metadata = FaceMetadata.from_dict(dfl_dict)

        return image, metadata

    @staticmethod
    def save(
        filepath: Path,
        image: np.ndarray,
        metadata: FaceMetadata,
        quality: int = 95,
    ) -> None:
        """
        Save image with DFL metadata.

        Args:
            filepath: Output path for JPEG file.
            image: Image as BGR numpy array.
            metadata: Face metadata to embed.
            quality: JPEG quality (1-100). Default: 95.
        """
        filepath = Path(filepath)

        # First save image normally
        cv2.imwrite(
            str(filepath),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        # Then inject metadata
        DFLImage._inject_metadata(filepath, metadata.to_dict())

    @staticmethod
    def has_metadata(filepath: Path) -> bool:
        """
        Check if file contains DFL metadata.

        Args:
            filepath: Path to JPEG file.

        Returns:
            True if file contains DFL metadata.
        """
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            dfl_dict = DFLImage._parse_jpeg_metadata(data)
            return dfl_dict is not None and len(dfl_dict) > 0
        except Exception:
            return False

    @staticmethod
    def _parse_jpeg_metadata(data: bytes) -> dict[str, Any] | None:
        """Parse JPEG data and extract DFL dictionary from APP15 chunk."""
        if len(data) < 2:
            return None

        # Verify JPEG SOI marker
        if data[0] != 0xFF or data[1] != DFLImage._SOI:
            return None

        pos = 2
        length = len(data)

        while pos < length - 1:
            if data[pos] != 0xFF:
                pos += 1
                continue

            marker = data[pos + 1]
            pos += 2

            # End of image
            if marker == DFLImage._EOI:
                break

            # Start of scan - skip to end
            if marker == DFLImage._SOS:
                break

            # Restart markers (no payload)
            if 0xD0 <= marker <= 0xD7:
                continue

            # SOI marker (no payload)
            if marker == DFLImage._SOI:
                continue

            # Read segment length
            if pos + 2 > length:
                break

            seg_length = struct.unpack(">H", data[pos : pos + 2])[0]
            pos += 2

            # APP15 - DFL data
            if marker == DFLImage._APP15:
                chunk_data = data[pos : pos + seg_length - 2]
                try:
                    return pickle.loads(chunk_data)
                except Exception:
                    return None

            pos += seg_length - 2

        return None

    @staticmethod
    def _inject_metadata(filepath: Path, dfl_dict: dict[str, Any]) -> None:
        """Inject DFL metadata into existing JPEG file."""
        with open(filepath, "rb") as f:
            data = f.read()

        # Verify JPEG
        if len(data) < 2 or data[0] != 0xFF or data[1] != DFLImage._SOI:
            raise ValueError("Not a valid JPEG file")

        # Parse chunks
        chunks = DFLImage._parse_jpeg_chunks(data)

        # Remove existing APP15 if present
        chunks = [c for c in chunks if c["marker"] != DFLImage._APP15]

        # Find insertion point (after APP chunks, before SOF)
        insert_idx = 1  # After SOI
        for i, chunk in enumerate(chunks):
            if chunk["marker"] & 0xF0 == 0xE0:  # APP0-APP15
                insert_idx = i + 1
            elif chunk["marker"] in (DFLImage._SOF0, DFLImage._SOF2):
                break

        # Create APP15 chunk
        app15_data = pickle.dumps(dfl_dict)
        app15_chunk = {
            "marker": DFLImage._APP15,
            "data": app15_data,
            "ex_data": None,
        }
        chunks.insert(insert_idx, app15_chunk)

        # Rebuild JPEG
        output = DFLImage._build_jpeg(chunks)

        with open(filepath, "wb") as f:
            f.write(output)

    @staticmethod
    def _parse_jpeg_chunks(data: bytes) -> list:
        """Parse JPEG into list of chunks."""
        chunks = []
        pos = 0
        length = len(data)

        while pos < length - 1:
            if data[pos] != 0xFF:
                pos += 1
                continue

            marker = data[pos + 1]
            pos += 2

            chunk = {"marker": marker, "data": None, "ex_data": None}

            # Markers without payload
            if marker in (DFLImage._SOI, DFLImage._EOI) or 0xD0 <= marker <= 0xD7:
                chunks.append(chunk)
                continue

            # Read segment length
            if pos + 2 > length:
                break

            seg_length = struct.unpack(">H", data[pos : pos + 2])[0]
            pos += 2

            chunk["data"] = data[pos : pos + seg_length - 2]
            pos += seg_length - 2

            # SOS has extra data until EOI
            if marker == DFLImage._SOS:
                end = pos
                while end < length - 1:
                    if data[end] == 0xFF and data[end + 1] == DFLImage._EOI:
                        break
                    end += 1
                chunk["ex_data"] = data[pos:end]
                pos = end

            chunks.append(chunk)

        return chunks

    @staticmethod
    def _build_jpeg(chunks: list) -> bytes:
        """Build JPEG bytes from chunks."""
        output = b""

        for chunk in chunks:
            output += struct.pack("BB", 0xFF, chunk["marker"])

            if chunk["data"] is not None:
                output += struct.pack(">H", len(chunk["data"]) + 2)
                output += chunk["data"]

            if chunk["ex_data"] is not None:
                output += chunk["ex_data"]

        return output

    @staticmethod
    def get_xseg_mask(metadata: FaceMetadata) -> np.ndarray | None:
        """
        Decode XSeg mask from metadata.

        Args:
            metadata: Face metadata containing compressed mask.

        Returns:
            Mask as float32 array (H, W, 1) in range [0, 1], or None.
        """
        if metadata.xseg_mask is None:
            return None

        # Decode compressed mask
        mask = cv2.imdecode(
            np.frombuffer(metadata.xseg_mask, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            return None

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]

        return mask.astype(np.float32) / 255.0

    @staticmethod
    def set_xseg_mask(
        metadata: FaceMetadata,
        mask: np.ndarray,
        max_size: int = 50000,
    ) -> None:
        """
        Encode and set XSeg mask in metadata.

        Args:
            metadata: Face metadata to update.
            mask: Mask as float32 array in range [0, 1].
            max_size: Maximum compressed size in bytes. Default: 50000.
        """
        if mask is None:
            metadata.xseg_mask = None
            return

        # Normalize to uint8
        mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        if len(mask_uint8.shape) == 3 and mask_uint8.shape[2] > 1:
            mask_uint8 = mask_uint8[..., 0]

        # Try PNG first
        ret, buf = cv2.imencode(".png", mask_uint8)
        if ret and len(buf) <= max_size:
            metadata.xseg_mask = buf.tobytes()
            return

        # Fall back to JPEG with decreasing quality
        for quality in range(100, -1, -5):
            ret, buf = cv2.imencode(
                ".jpg",
                mask_uint8,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if ret and len(buf) <= max_size:
                metadata.xseg_mask = buf.tobytes()
                return

        raise ValueError("Unable to compress mask within size limit")

    @staticmethod
    def get_seg_ie_polys(metadata: FaceMetadata) -> PolygonSet | None:
        """
        Get Include/Exclude polygons from metadata.

        Args:
            metadata: Face metadata containing polygon data.

        Returns:
            PolygonSet instance or None if no polygons.
        """
        from visagen.vision.polys import PolygonSet

        if metadata.seg_ie_polys is None:
            return None

        return PolygonSet.from_dict(metadata.seg_ie_polys)

    @staticmethod
    def set_seg_ie_polys(
        metadata: FaceMetadata,
        polys: PolygonSet | None,
    ) -> None:
        """
        Set Include/Exclude polygons in metadata.

        Args:
            metadata: Face metadata to update.
            polys: PolygonSet to store, or None to clear.
        """
        if polys is None or not polys.has_polys():
            metadata.seg_ie_polys = None
        else:
            metadata.seg_ie_polys = polys.to_dict()
