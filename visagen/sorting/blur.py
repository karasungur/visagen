"""
Blur-based sorting methods.

Provides sharpness estimation using CPBD and Laplacian variance.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from visagen.sorting.base import SortMethod

logger = logging.getLogger(__name__)
_CPBD_FAILURE_LOGGED = False

if TYPE_CHECKING:
    from visagen.vision.face_image import FaceMetadata


def get_face_hull_mask(
    shape: tuple[int, ...],
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Create convex hull mask from facial landmarks.

    Args:
        shape: Image shape (H, W) or (H, W, C).
        landmarks: 68-point facial landmarks.

    Returns:
        Binary mask as float32 array with same H, W as shape.
    """
    h, w = shape[:2]
    mask: np.ndarray = np.zeros((h, w, 1), dtype=np.float32)

    # Use jaw + eyebrows for face hull (points 0-16, 17-26)
    hull_points = np.concatenate([landmarks[0:17], landmarks[17:27]], axis=0)
    hull_points = hull_points.astype(np.int32)

    # Create convex hull
    hull = cv2.convexHull(hull_points)
    cv2.fillConvexPoly(mask, hull, (1.0,))

    return mask


def estimate_sharpness(image: np.ndarray) -> float:
    """
    Estimate image sharpness using Laplacian variance.

    Higher values indicate sharper images.

    Args:
        image: BGR or grayscale image.

    Returns:
        Sharpness score (Laplacian variance).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def estimate_sharpness_cpbd(image: np.ndarray) -> tuple[float, str]:
    """
    Estimate sharpness using CPBD with deterministic fallback.

    Returns:
        Tuple of (score, metric_name), where metric_name is either
        "cpbd" or "laplacian-fallback".
    """
    try:
        from visagen.sorting.cpbd import estimate_cpbd_sharpness

        return estimate_cpbd_sharpness(image), "cpbd"
    except Exception as e:
        global _CPBD_FAILURE_LOGGED
        if not _CPBD_FAILURE_LOGGED:
            logger.warning(f"CPBD unavailable, falling back to Laplacian: {e}")
            _CPBD_FAILURE_LOGGED = True
        return estimate_sharpness(image), "laplacian-fallback"


class BlurSorter(SortMethod):
    """
    Sort by image sharpness (blur detection).

    Uses CPBD to estimate sharpness.
    Higher scores indicate sharper (less blurry) images.

    If face metadata is available, applies face mask to focus
    sharpness estimation on the face region only.
    """

    name = "blur"
    description = "Sort by image sharpness (CPBD)"
    requires_face_metadata = True
    execution_profile = "cpu_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute sharpness score."""
        # Apply face mask if landmarks available
        if metadata is not None and metadata.landmarks is not None:
            try:
                mask = get_face_hull_mask(image.shape, metadata.landmarks)
                image = (image * mask).astype(np.uint8)
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        score, metric = estimate_sharpness_cpbd(image)
        if metric != "cpbd":
            logger.debug("BlurSorter used Laplacian fallback")
        return score


class BlurFastSorter(SortMethod):
    """
    Fast blur sorter using Laplacian variance.

    This mode is optimized for throughput and does not guarantee
    CPBD parity.
    """

    name = "blur-fast"
    description = "Sort by image sharpness (fast Laplacian)"
    requires_face_metadata = True
    execution_profile = "cpu_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute fast sharpness score."""
        if metadata is not None and metadata.landmarks is not None:
            try:
                mask = get_face_hull_mask(image.shape, metadata.landmarks)
                image = (image * mask).astype(np.uint8)
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        return estimate_sharpness(image)


class MotionBlurSorter(SortMethod):
    """
    Sort by motion blur detection.

    Uses Laplacian variance with larger kernel (ksize=11)
    to better detect motion blur patterns.

    Higher scores indicate less motion blur.
    """

    name = "motion-blur"
    description = "Sort by motion blur (Laplacian ksize=11)"
    requires_face_metadata = True
    execution_profile = "cpu_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute motion blur score."""
        # Apply face mask if landmarks available
        if metadata is not None and metadata.landmarks is not None:
            try:
                mask = get_face_hull_mask(image.shape, metadata.landmarks)
                image = (image * mask).astype(np.uint8)
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Laplacian with larger kernel for motion blur
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=11)
        return float(laplacian.var())
