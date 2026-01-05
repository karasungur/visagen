"""
Color-based sorting methods.

Provides sorting by brightness, hue, and black pixel count.
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from visagen.sorting.base import SortMethod

if TYPE_CHECKING:
    from visagen.vision.dflimg import FaceMetadata


class BrightnessSorter(SortMethod):
    """
    Sort by image brightness.

    Uses mean V channel value from HSV color space.
    Higher scores indicate brighter images.
    """

    name = "brightness"
    description = "Sort by brightness"
    requires_dfl_metadata = False

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute mean brightness (V channel)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[..., 2]))


class HueSorter(SortMethod):
    """
    Sort by dominant hue.

    Uses mean H channel value from HSV color space.
    Groups images with similar color tones together.
    """

    name = "hue"
    description = "Sort by hue"
    requires_dfl_metadata = False

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute mean hue (H channel)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[..., 0]))


class BlackPixelSorter(SortMethod):
    """
    Sort by amount of black pixels.

    Counts pixels with value 0. Useful for detecting
    images with large black borders or artifacts.

    Lower scores (fewer black pixels) rank first.
    """

    name = "black"
    description = "Sort by amount of black pixels"
    requires_dfl_metadata = False

    @property
    def reverse_sort(self) -> bool:
        """Fewer black pixels is better (ascending order)."""
        return False

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Count black pixels."""
        return float(np.sum(image == 0))


class SaturationSorter(SortMethod):
    """
    Sort by color saturation.

    Uses mean S channel value from HSV color space.
    Higher scores indicate more colorful images.
    """

    name = "saturation"
    description = "Sort by color saturation"
    requires_dfl_metadata = False

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute mean saturation (S channel)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[..., 1]))


class ContrastSorter(SortMethod):
    """
    Sort by image contrast.

    Uses standard deviation of grayscale values.
    Higher scores indicate higher contrast images.
    """

    name = "contrast"
    description = "Sort by contrast"
    requires_dfl_metadata = False

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute contrast (std of grayscale)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
