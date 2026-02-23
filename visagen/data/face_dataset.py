"""
Face Dataset for Training.

Load aligned face images with metadata for training.
"""

import io
import logging
import pickle
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from visagen.data.face_sample import FaceSample
from visagen.vision.face_type import FaceType

logger = logging.getLogger(__name__)
FaceDatasetItem = dict[str, Any]
PACKED_FACESET_FILENAME = "faceset.pak"


class _PackedUnpickler(pickle.Unpickler):
    """
    Unpickler with compatibility fallbacks for enum references.

    Packed `faceset.pak` files may store enum values with import paths that do not
    exist in the modern runtime. These are downcast to plain ints.
    """

    _ENUM_FALLBACKS: set[tuple[str, str]] = {
        ("facelib", "FaceType"),
        ("facelib.FaceType", "FaceType"),
        ("samplelib", "SampleType"),
        ("samplelib.Sample", "SampleType"),
        ("samplelib.SampleType", "SampleType"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._ENUM_FALLBACKS:
            return int
        return super().find_class(module, name)


def _unpickle_packed_configs(data: bytes) -> Any:
    return _PackedUnpickler(io.BytesIO(data)).load()


def _to_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, np.ndarray):
        return value.tobytes()
    return None


def _encode_mask_to_png(mask: Any) -> bytes | None:
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        return None

    mask_np = mask
    if mask_np.dtype != np.uint8:
        mask_np = np.clip(mask_np, 0, 1)
        mask_np = (mask_np * 255).astype(np.uint8)

    if mask_np.ndim == 3 and mask_np.shape[2] > 1:
        mask_np = mask_np[..., 0]

    ok, encoded = cv2.imencode(".png", mask_np)
    if not ok:
        return None
    return encoded.tobytes()


def _normalize_face_type(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, int):
        try:
            return FaceType(value).to_string()
        except ValueError:
            return FaceType.WHOLE_FACE.to_string()
    if hasattr(value, "value"):
        return _normalize_face_type(value.value)
    if hasattr(value, "name"):
        return str(value.name).lower()
    return FaceType.WHOLE_FACE.to_string()


class PackedFacesetReader:
    """
    Reader for `faceset.pak` archives.

    Parses metadata and image offsets and returns `FaceSample` entries that read
    image bytes lazily from the packed archive.
    """

    def __init__(self, packed_path: str | Path) -> None:
        self.packed_path = Path(packed_path)

    def read_samples(
        self,
        root_dir: str | Path,
        face_type_filter: FaceType | None = None,
    ) -> list[FaceSample]:
        root_dir_path = Path(root_dir)
        if not self.packed_path.exists():
            return []

        try:
            with open(self.packed_path, "rb") as f:
                version = struct.unpack("Q", f.read(8))[0]
                if version != 1:
                    raise ValueError(f"Unsupported faceset.pak version: {version}")

                configs_size = struct.unpack("Q", f.read(8))[0]
                configs_raw = f.read(configs_size)
                configs = _unpickle_packed_configs(configs_raw)
                if not isinstance(configs, list):
                    raise ValueError("faceset.pak metadata payload is not a list")

                offsets = [
                    struct.unpack("Q", f.read(8))[0] for _ in range(len(configs) + 1)
                ]
                data_start_offset = f.tell()
        except Exception as e:
            raise ValueError(
                f"Failed to parse faceset archive: {self.packed_path}"
            ) from e

        samples: list[FaceSample] = []
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                continue

            try:
                landmarks_raw = config.get("landmarks")
                if landmarks_raw is None:
                    continue
                landmarks = np.array(landmarks_raw, dtype=np.float32)

                face_type = _normalize_face_type(config.get("face_type"))
                shape_raw = config.get("shape")
                if shape_raw is None:
                    shape = (0, 0, 3)
                else:
                    shape_tuple = tuple(int(v) for v in shape_raw)
                    if len(shape_tuple) == 2:
                        shape = (shape_tuple[0], shape_tuple[1], 3)
                    elif len(shape_tuple) == 3:
                        shape = (shape_tuple[0], shape_tuple[1], shape_tuple[2])
                    else:
                        shape = (0, 0, 3)

                person_name = config.get("person_name")
                filename = str(config.get("filename", f"{i:08d}.jpg"))
                if person_name is not None:
                    filepath = root_dir_path / str(person_name) / filename
                else:
                    filepath = root_dir_path / filename

                xseg_mask = _to_bytes(config.get("xseg_mask_compressed"))
                if xseg_mask is None:
                    xseg_mask = _encode_mask_to_png(config.get("xseg_mask"))

                start_offset = int(offsets[i])
                end_offset = int(offsets[i + 1])
                if end_offset < start_offset:
                    raise ValueError("Invalid offset table in faceset.pak")

                sample = FaceSample(
                    filepath=filepath,
                    face_type=face_type,
                    shape=shape,
                    landmarks=landmarks,
                    xseg_mask=xseg_mask,
                    seg_ie_polys=config.get("seg_ie_polys"),
                    eyebrows_expand_mod=float(config.get("eyebrows_expand_mod", 1.0)),
                    source_filename=config.get("source_filename"),
                    packed_faceset_path=self.packed_path,
                    packed_offset=data_start_offset + start_offset,
                    packed_size=end_offset - start_offset,
                )

                if face_type_filter is not None:
                    try:
                        sample_type = FaceType.from_string(sample.face_type)
                        if sample_type != face_type_filter:
                            continue
                    except ValueError:
                        continue

                samples.append(sample)
            except Exception as e:
                logger.debug(
                    "Skipping packed sample %d in %s: %s",
                    i,
                    self.packed_path,
                    e,
                )

        return samples


class FaceDataset(Dataset):
    """
    Dataset for aligned face images.

    Loads face images from a directory containing JPEG files with
    embedded face metadata. Supports lazy loading and augmentation.

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
        allow_packed_faceset: Enable `faceset.pak` support. Default: True.

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
        allow_packed_faceset: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.face_type_filter = face_type_filter
        self.with_mask = with_mask
        self.uniform_yaw = uniform_yaw
        self.yaw_bins = yaw_bins
        self.allow_packed_faceset = allow_packed_faceset

        # Scan directory for image files
        self.image_paths = self.scan_directory(self.root_dir)
        self.packed_faceset_path = self.root_dir / PACKED_FACESET_FILENAME

        packed_available = (
            self.packed_faceset_path.exists() and self.allow_packed_faceset
        )
        if len(self.image_paths) == 0 and not packed_available:
            if self.packed_faceset_path.exists() and not self.allow_packed_faceset:
                raise ValueError(
                    "No images found and faceset.pak loading is disabled. "
                    "Set allow_packed_faceset=True to enable packed input."
                )
            raise ValueError(f"No images found in {self.root_dir}")

        # Load metadata
        self.samples: list[FaceSample] = []
        if preload_metadata:
            self._preload_metadata()
        else:
            if packed_available:
                # Packed facesets require metadata to locate image offsets.
                self.samples = self._load_packed_faceset(strict=True)
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

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid face samples found in {self.root_dir}. "
                "Check faceset.pak integrity."
            )

        # Build yaw bins for uniform sampling
        self._yaw_bins_indices: list[list[int]] | None = None
        if self.uniform_yaw and preload_metadata:
            self._build_yaw_bins()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> FaceDatasetItem:
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
                output: FaceDatasetItem = {
                    "image": image_tensor,
                    "landmarks": torch.from_numpy(sample.landmarks.copy()),
                    "face_type": self._get_face_type_int(sample.face_type),
                    "seg_ie_polys": sample.seg_ie_polys,
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
                        "seg_ie_polys": None,
                    }

        # Should never reach here, but just in case
        return {
            "image": torch.zeros(3, self.target_size, self.target_size) - 1.0,
            "landmarks": torch.zeros(68, 2),
            "face_type": 0,
            "seg_ie_polys": None,
        }

    def _preload_metadata(self) -> None:
        """Preload metadata from all images."""
        seen_relpaths: set[str] = set()

        if self.packed_faceset_path.exists() and self.allow_packed_faceset:
            packed_samples = self._load_packed_faceset(
                strict=len(self.image_paths) == 0
            )
            for sample in packed_samples:
                rel = str(sample.filepath.relative_to(self.root_dir))
                seen_relpaths.add(rel)
                self.samples.append(sample)

        for path in self.image_paths:
            rel = str(path.relative_to(self.root_dir))
            if rel in seen_relpaths:
                continue

            face_sample = FaceSample.from_face_image(path)

            if face_sample is None:
                # Skip images without face metadata
                continue

            # Apply face type filter
            if self.face_type_filter is not None:
                try:
                    sample_type = FaceType.from_string(face_sample.face_type)
                    if sample_type != self.face_type_filter:
                        continue
                except ValueError:
                    continue

            seen_relpaths.add(rel)
            self.samples.append(face_sample)

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid face images found in {self.root_dir}. "
                "Make sure images have embedded face metadata."
            )

    def _load_packed_faceset(self, strict: bool = False) -> list[FaceSample]:
        """
        Load FaceSample entries from `faceset.pak`.

        The packed format stores metadata configs followed by an offset table and
        raw image blobs. We use metadata for face attributes and decode image data
        lazily via packed offset/size in FaceSample.
        """
        if not self.packed_faceset_path.exists() or not self.allow_packed_faceset:
            return []

        try:
            reader = PackedFacesetReader(self.packed_faceset_path)
            return reader.read_samples(
                root_dir=self.root_dir,
                face_type_filter=self.face_type_filter,
            )
        except Exception as e:
            if strict:
                raise ValueError(
                    f"Failed to read packed faceset at {self.packed_faceset_path}"
                ) from e
            logger.warning(
                "Failed to read packed faceset at %s: %s",
                self.packed_faceset_path,
                e,
            )
            return []

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
            return int(np.random.randint(len(self.samples)))

        # Extra safety: filter empty bins at runtime
        non_empty = [b for b in self._yaw_bins_indices if len(b) > 0]
        if not non_empty:
            return int(np.random.randint(len(self.samples)))

        # Select random bin
        bin_idx = np.random.randint(len(non_empty))
        # Select random sample from bin
        sample_idx = np.random.choice(non_empty[bin_idx])
        return cast(int, int(sample_idx))

    def build_uniform_yaw_bins_for_indices(
        self, dataset_indices: list[int]
    ) -> list[list[int]] | None:
        """
        Build subset-local yaw bins from full-dataset yaw bins.

        Args:
            dataset_indices: Indices in this dataset that belong to a subset.

        Returns:
            Subset-local bin indices (0..len(dataset_indices)-1), or None if
            uniform yaw bins are unavailable.
        """
        if self._yaw_bins_indices is None:
            return None

        index_map = {
            full_idx: sub_idx for sub_idx, full_idx in enumerate(dataset_indices)
        }
        subset_bins: list[list[int]] = []

        for full_bin in self._yaw_bins_indices:
            subset_bin = [index_map[idx] for idx in full_bin if idx in index_map]
            if subset_bin:
                subset_bins.append(subset_bin)

        if not subset_bins:
            return None
        return subset_bins

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
        paths: list[Path] = []

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
    Simple dataset for loading face images without embedded metadata.

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
