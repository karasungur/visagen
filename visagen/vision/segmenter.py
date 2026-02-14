"""
Face Segmentation using SegFormer.

Provides semantic face parsing using SegFormer model
fine-tuned on CelebAMask-HQ dataset.

Features:
- Single image and batch segmentation
- Configurable interpolation modes for upsampling
- Optional LRU caching for repeated images
- Face type transformation support
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from visagen.vision.cache import SegmentationCache

logger = __import__("logging").getLogger(__name__)

if TYPE_CHECKING:
    from visagen.vision.face_type import FaceType

# Default local model path (relative to project root)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face-parsing"

# CelebAMask-HQ label mapping (19 classes)
CELEBAMASK_LABELS = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_glasses",
    4: "left_eye",
    5: "right_eye",
    6: "left_brow",
    7: "right_brow",
    8: "left_ear",
    9: "right_ear",
    10: "mouth",
    11: "upper_lip",
    12: "lower_lip",
    13: "hair",
    14: "hat",
    15: "earring",
    16: "necklace",
    17: "neck",
    18: "cloth",
}

# Inverse mapping for lookup
LABEL_TO_ID = {v: k for k, v in CELEBAMASK_LABELS.items()}

# Face mask components (excluding background, hair, hat, cloth, accessories)
FACE_COMPONENTS = {
    "skin",
    "nose",
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "left_ear",
    "right_ear",
    "mouth",
    "upper_lip",
    "lower_lip",
}


class InterpolationMode(Enum):
    """Interpolation modes for mask upsampling."""

    BILINEAR = "bilinear"  # Default, smooth but may blur edges
    BICUBIC = "bicubic"  # Smoother than bilinear
    NEAREST = "nearest"  # Preserves hard edges, may be blocky
    AREA = "area"  # Best for downsampling


class ThresholdMode(Enum):
    """Threshold modes for binary mask generation."""

    FIXED = "fixed"  # Fixed threshold (default 0.5)
    OTSU = "otsu"  # Otsu's automatic threshold selection
    ADAPTIVE = "adaptive"  # Gaussian adaptive thresholding


@dataclass
class SegmentationResult:
    """
    Face segmentation result.

    Attributes:
        mask: Binary face mask (H, W), values 0 or 255.
        parsing: Full parsing map with class indices (H, W).
        confidence: Mean confidence score for face region.
    """

    mask: np.ndarray
    parsing: np.ndarray
    confidence: float


class FaceSegmenter:
    """
    SegFormer-based face segmentation.

    Uses SegFormer model fine-tuned on CelebAMask-HQ for
    semantic face parsing with 19 classes.

    Args:
        model_name: Model path or HuggingFace identifier.
            Default: auto-detect (local first, then HuggingFace).
        device: Device for inference. Default: auto-detect.
        use_half: Use FP16 for faster inference. Default: True on CUDA.
        interpolation_mode: Upsampling interpolation mode. Default: BILINEAR.
        enable_cache: Enable LRU cache for repeated images. Default: False.
        cache_size: Maximum cache entries. Default: 100.

    Example:
        >>> segmenter = FaceSegmenter()
        >>> result = segmenter.segment(aligned_face)
        >>> mask = result.mask  # Binary face mask
    """

    def __init__(
        self,
        model_name: str | Path | None = None,
        device: str | None = None,
        use_half: bool = True,
        interpolation_mode: InterpolationMode = InterpolationMode.BILINEAR,
        enable_cache: bool = False,
        cache_size: int = 100,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.use_half = use_half and device == "cuda"
        self.interpolation_mode = interpolation_mode

        # Initialize cache if enabled
        self._cache = SegmentationCache(cache_size) if enable_cache else None

        # Resolve model source
        model_source, local_only = self._resolve_model_source(model_name)

        # Load model and processor
        self.processor = SegformerImageProcessor.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        self.model.to(device)
        self.model.eval()

        if self.use_half:
            self.model.half()

    def _resolve_model_source(self, model_name: str | Path | None) -> tuple[str, bool]:
        """
        Resolve model source with local-first priority.

        Priority:
        1. Explicit path provided → use it
        2. Default local path exists → use it
        3. Fallback to HuggingFace Hub

        Returns:
            Tuple of (model_source, local_files_only).
        """
        # Explicit path provided
        if model_name is not None:
            path = Path(model_name) if isinstance(model_name, str) else model_name
            if path.exists():
                return str(path), True
            # Assume it's a HuggingFace repo ID
            return str(model_name), False

        # Check default local path
        if DEFAULT_MODEL_PATH.exists():
            return str(DEFAULT_MODEL_PATH), True

        # Fallback to HuggingFace
        return "jonathandinu/face-parsing", False

    def segment(
        self,
        face_image: np.ndarray,
        return_soft_mask: bool = False,
        use_cache: bool = True,
        anti_alias: bool = False,
        threshold_mode: ThresholdMode = ThresholdMode.FIXED,
        threshold_value: float = 0.5,
    ) -> SegmentationResult:
        """
        Segment face from aligned face image.

        Args:
            face_image: Aligned face image (BGR, uint8).
            return_soft_mask: Return soft probabilities instead of binary.
                Default: False.
            use_cache: Use cache if enabled. Default: True.
            anti_alias: Apply anti-aliasing to binary mask edges. Default: False.
            threshold_mode: Threshold mode for binary mask. Default: FIXED.
            threshold_value: Threshold value for FIXED mode. Default: 0.5.

        Returns:
            SegmentationResult with binary mask and full parsing.
        """
        # Check cache first
        if use_cache and self._cache is not None:
            cached = self._cache.get_result(
                face_image,
                return_soft_mask,
                anti_alias,
                threshold_mode.value,
                threshold_value,
            )
            if cached is not None:
                return cast(SegmentationResult, cached)

        h, w = face_image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Upsample to original size with configurable interpolation
        interp_mode = self.interpolation_mode.value
        if interp_mode == "area":
            # Area mode only for downsampling, use bilinear for upsampling
            interp_mode = "bilinear"

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode=interp_mode,
            align_corners=False if interp_mode != "nearest" else None,
        )

        # Get predictions
        probs = F.softmax(logits, dim=1)
        parsing = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Create face mask from face components
        face_mask = np.zeros((h, w), dtype=np.float32)
        for component in FACE_COMPONENTS:
            if component in LABEL_TO_ID:
                component_id = LABEL_TO_ID[component]
                face_mask += probs[0, component_id].cpu().numpy()

        # Calculate confidence as mean probability in face region
        confidence = (
            float(face_mask[face_mask > 0.5].mean()) if (face_mask > 0.5).any() else 0.0
        )

        # Convert to binary mask
        if not return_soft_mask:
            if threshold_mode == ThresholdMode.OTSU:
                # Otsu's automatic threshold
                mask_uint8 = (face_mask * 255).astype(np.uint8)
                _, thresholded = cv2.threshold(
                    mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                face_mask = cast(np.ndarray, thresholded).astype(np.float32)
            elif threshold_mode == ThresholdMode.ADAPTIVE:
                # Gaussian adaptive threshold
                mask_uint8 = (face_mask * 255).astype(np.uint8)
                face_mask = cast(
                    np.ndarray,
                    cv2.adaptiveThreshold(
                    mask_uint8,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                    ),
                )
                face_mask = face_mask.astype(np.float32)
            else:  # FIXED
                face_mask = (face_mask > threshold_value).astype(np.float32) * 255.0

            # Apply anti-aliasing if requested
            if anti_alias:
                face_mask = cast(
                    np.ndarray,
                    cv2.GaussianBlur(face_mask.astype(np.float32), (0, 0), 1.0),
                )
                face_mask = (face_mask > 127).astype(np.float32) * 255.0

        result = SegmentationResult(
            mask=face_mask,
            parsing=parsing.astype(np.uint8),
            confidence=confidence,
        )

        # Store in cache
        if use_cache and self._cache is not None:
            self._cache.put_result(
                face_image,
                result,
                return_soft_mask,
                anti_alias,
                threshold_mode.value,
                threshold_value,
            )

        return result

    def segment_batch(
        self,
        face_images: Sequence[np.ndarray],
        batch_size: int = 8,
        return_soft_mask: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SegmentationResult]:
        """
        Batch segmentation for GPU efficiency.

        Processes multiple images together for improved throughput.

        Args:
            face_images: Sequence of aligned face images (BGR, uint8).
            batch_size: Number of images per batch. Default: 8.
            return_soft_mask: Return soft probabilities. Default: False.
            progress_callback: Optional callback(processed, total). Default: None.

        Returns:
            List of SegmentationResult for each input image.
        """
        results: list[SegmentationResult] = []
        total = len(face_images)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = list(face_images[batch_start:batch_end])

            batch_results = self._process_batch(batch, return_soft_mask)
            results.extend(batch_results)

            if progress_callback is not None:
                progress_callback(len(results), total)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return results

    def _process_batch(
        self,
        images: Sequence[np.ndarray],
        return_soft_mask: bool,
    ) -> list[SegmentationResult]:
        """
        Process a batch of images together.

        Args:
            images: Batch of images.
            return_soft_mask: Whether to return soft masks.

        Returns:
            List of SegmentationResult for batch.
        """
        if not images:
            return []

        # All images must have same size for batching
        h, w = images[0].shape[:2]

        # Convert BGR to RGB
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        # Batch preprocess
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Batch inference with OOM recovery
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM in segmentation, retrying with smaller batch")
                torch.cuda.empty_cache()
                # Split batch in half and retry
                if len(images) > 1:
                    mid = len(images) // 2
                    results_1 = self._process_batch(
                        list(images[:mid]), return_soft_mask
                    )
                    results_2 = self._process_batch(
                        list(images[mid:]), return_soft_mask
                    )
                    return results_1 + results_2
                else:
                    # Single image OOM, fall back to CPU
                    logger.warning("Single image OOM, falling back to CPU")
                    self.model = self.model.cpu()
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
            else:
                raise

        # Batch upsample
        interp_mode = self.interpolation_mode.value
        if interp_mode == "area":
            interp_mode = "bilinear"

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode=interp_mode,
            align_corners=False if interp_mode != "nearest" else None,
        )

        # Get predictions
        probs = F.softmax(logits, dim=1)
        parsings = logits.argmax(dim=1).cpu().numpy()

        # Build results for each image
        results = []
        for i in range(len(images)):
            # Create face mask from components
            face_mask = np.zeros((h, w), dtype=np.float32)
            for component in FACE_COMPONENTS:
                if component in LABEL_TO_ID:
                    component_id = LABEL_TO_ID[component]
                    face_mask += probs[i, component_id].cpu().numpy()

            # Calculate confidence
            confidence = (
                float(face_mask[face_mask > 0.5].mean())
                if (face_mask > 0.5).any()
                else 0.0
            )

            # Convert to binary
            if not return_soft_mask:
                face_mask = (face_mask > 0.5).astype(np.float32) * 255.0

            results.append(
                SegmentationResult(
                    mask=face_mask,
                    parsing=parsings[i].astype(np.uint8),
                    confidence=confidence,
                )
            )

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return results

    def clear_cache(self) -> None:
        """Clear the segmentation cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_stats(self) -> dict | None:
        """Get cache statistics if caching is enabled."""
        if self._cache is not None:
            return self._cache.stats
        return None

    def get_parsing(
        self,
        face_image: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Get detailed parsing masks for each face component.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Dictionary mapping component names to binary masks.
        """
        h, w = face_image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Get per-class masks
        parsing = logits.argmax(dim=1).squeeze().cpu().numpy()

        result = {}
        for label_id, label_name in CELEBAMASK_LABELS.items():
            mask = (parsing == label_id).astype(np.uint8) * 255
            if mask.any():
                result[label_name] = mask

        return result

    def get_hair_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get hair-only mask from face image.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Binary hair mask (H, W), values 0 or 255.
        """
        parsing = self.get_parsing(face_image)
        hair_mask = parsing.get("hair", np.zeros(face_image.shape[:2], dtype=np.uint8))
        return hair_mask

    def get_face_and_hair_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get combined face and hair mask.

        Args:
            face_image: Aligned face image (BGR, uint8).

        Returns:
            Binary mask (H, W), values 0 or 255.
        """
        result = self.segment(face_image)
        parsing = self.get_parsing(face_image)

        combined = result.mask.copy()
        if "hair" in parsing:
            combined = np.maximum(combined, parsing["hair"])

        return cast(np.ndarray, combined)

    def segment_with_face_type(
        self,
        face_image: np.ndarray,
        landmarks: np.ndarray,
        image_face_type: FaceType,
        model_face_type: FaceType | None = None,
    ) -> SegmentationResult:
        """
        Segment face with automatic face type transformation.

        When the model is trained on a different face type than the image,
        this method automatically transforms the image before segmentation
        and transforms the result mask back.

        Args:
            face_image: Aligned face image (BGR, uint8).
            landmarks: 68-point facial landmarks in aligned image space.
            image_face_type: Face type of the input image.
            model_face_type: Face type the model was trained on.
                If None or same as image_face_type, no transformation is applied.

        Returns:
            SegmentationResult with mask in original image space.
        """
        from visagen.vision.aligner import FaceAligner
        from visagen.vision.face_type import FaceType

        # If no transformation needed, use direct segmentation
        if model_face_type is None or model_face_type == image_face_type:
            return self.segment(face_image)

        h, w = face_image.shape[:2]
        aligner = FaceAligner()

        # Get transformation matrix from image face type to model face type
        forward_mat = aligner.get_face_type_transform_mat(
            landmarks, w, image_face_type, model_face_type
        )

        # Warp image to model face type space
        warped_image = cv2.warpAffine(
            face_image,
            forward_mat,
            (w, w),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Segment in model face type space
        result = self.segment(warped_image)

        # Warp mask back to image face type space
        inverse_mat = cv2.invertAffineTransform(forward_mat)
        warped_mask = cv2.warpAffine(
            result.mask,
            inverse_mat,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderValue=0,
        )

        # Re-threshold to clean up interpolation artifacts
        warped_mask = (warped_mask > 127).astype(np.uint8) * 255

        return SegmentationResult(
            mask=warped_mask,
            parsing=result.parsing,  # Note: parsing is in model space
            confidence=result.confidence,
        )

    def __del__(self) -> None:
        """Cleanup GPU resources on deletion."""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
