"""
Mask Operations for Morphological Processing.

Provides utility classes and functions for mask refinement,
brush operations, and component-based mask building.
"""

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from visagen.vision.segmenter import LABEL_TO_ID


@dataclass
class MaskRefinementConfig:
    """
    Configuration for mask refinement operations.

    Attributes:
        erode_size: Erosion kernel size. 0 means no erosion.
        dilate_size: Dilation kernel size. 0 means no dilation.
        blur_size: Gaussian blur kernel size. 0 means no blur.
        threshold: Threshold for binarization after blur. Default: 0.5.
    """

    erode_size: int = 0
    dilate_size: int = 0
    blur_size: int = 0
    threshold: float = 0.5


@dataclass
class BrushConfig:
    """
    Configuration for brush operations.

    Attributes:
        size: Brush diameter in pixels.
        mode: 'add' to add to mask, 'remove' to subtract from mask.
        hardness: Brush edge hardness (0.0 = soft, 1.0 = hard).
    """

    size: int = 20
    mode: Literal["add", "remove"] = "add"
    hardness: float = 1.0


# Default face components for mask building
DEFAULT_FACE_COMPONENTS: set[str] = {
    "skin",
    "nose",
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "mouth",
    "upper_lip",
    "lower_lip",
}

# All available components from CelebAMask-HQ
ALL_COMPONENTS: set[str] = {
    "background",
    "skin",
    "nose",
    "eye_glasses",
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "left_ear",
    "right_ear",
    "mouth",
    "upper_lip",
    "lower_lip",
    "hair",
    "hat",
    "earring",
    "necklace",
    "neck",
    "cloth",
}


class MaskOperations:
    """
    Static utility class for mask morphological operations.

    Provides methods for erosion, dilation, blur, brush application,
    and component-based mask combination.
    """

    @staticmethod
    def erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply erosion to mask.

        Args:
            mask: Input mask (H, W), uint8 or float32.
            kernel_size: Erosion kernel size. Must be odd and positive.

        Returns:
            Eroded mask with same dtype as input.
        """
        if kernel_size <= 0:
            return mask.copy()

        # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        return cv2.erode(mask, kernel, iterations=1)

    @staticmethod
    def dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply dilation to mask.

        Args:
            mask: Input mask (H, W), uint8 or float32.
            kernel_size: Dilation kernel size. Must be odd and positive.

        Returns:
            Dilated mask with same dtype as input.
        """
        if kernel_size <= 0:
            return mask.copy()

        # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        return cv2.dilate(mask, kernel, iterations=1)

    @staticmethod
    def blur(
        mask: np.ndarray,
        kernel_size: int,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Apply Gaussian blur to mask with optional thresholding.

        Args:
            mask: Input mask (H, W), uint8 (0-255) or float32 (0-1).
            kernel_size: Blur kernel size. Must be odd and positive.
            threshold: Threshold for binarization. Set to 0 to skip.

        Returns:
            Blurred (and optionally thresholded) mask.
        """
        if kernel_size <= 0:
            return mask.copy()

        # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # Determine input type
        is_uint8 = mask.dtype == np.uint8

        # Convert to float for processing
        if is_uint8:
            mask_float = mask.astype(np.float32) / 255.0
        else:
            mask_float = mask.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)

        # Apply threshold if specified
        if threshold > 0:
            blurred = (blurred > threshold).astype(np.float32)

        # Convert back to original dtype
        if is_uint8:
            return (blurred * 255).astype(np.uint8)

        return blurred

    @staticmethod
    def apply_brush(
        mask: np.ndarray,
        points: list[tuple[int, int]],
        brush_size: int,
        mode: Literal["add", "remove"] = "add",
    ) -> np.ndarray:
        """
        Apply brush strokes to mask.

        Args:
            mask: Input mask (H, W), uint8 (0-255).
            points: List of (x, y) coordinates for brush stroke.
            brush_size: Diameter of the brush in pixels.
            mode: 'add' to paint white, 'remove' to paint black.

        Returns:
            Modified mask with brush strokes applied.
        """
        result = mask.copy()

        if not points:
            return result

        color = 255 if mode == "add" else 0
        radius = max(1, brush_size // 2)

        for x, y in points:
            cv2.circle(result, (int(x), int(y)), radius, color, -1)

        # Draw lines between consecutive points for smooth strokes
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(result, pt1, pt2, color, brush_size)

        return result

    @staticmethod
    def refine(mask: np.ndarray, config: MaskRefinementConfig) -> np.ndarray:
        """
        Apply refinement pipeline to mask.

        Processing order: erode -> dilate -> blur

        Args:
            mask: Input mask (H, W), uint8 or float32.
            config: Refinement configuration.

        Returns:
            Refined mask.
        """
        result = mask.copy()

        # Apply erosion
        if config.erode_size > 0:
            result = MaskOperations.erode(result, config.erode_size)

        # Apply dilation
        if config.dilate_size > 0:
            result = MaskOperations.dilate(result, config.dilate_size)

        # Apply blur
        if config.blur_size > 0:
            result = MaskOperations.blur(result, config.blur_size, config.threshold)

        return result

    @staticmethod
    def combine_component_masks(
        parsing: np.ndarray,
        include_components: set[str],
    ) -> np.ndarray:
        """
        Combine parsing map into binary mask based on selected components.

        Args:
            parsing: Parsing map (H, W) with class indices.
            include_components: Set of component names to include.

        Returns:
            Binary mask (H, W) with values 0 or 255.
        """
        h, w = parsing.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for component in include_components:
            if component in LABEL_TO_ID:
                component_id = LABEL_TO_ID[component]
                mask[parsing == component_id] = 255

        return mask

    @staticmethod
    def invert(mask: np.ndarray) -> np.ndarray:
        """
        Invert mask values.

        Args:
            mask: Input mask (H, W), uint8 (0-255).

        Returns:
            Inverted mask.
        """
        return 255 - mask

    @staticmethod
    def smooth_edges(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Smooth mask edges using Gaussian blur.

        Args:
            mask: Input mask (H, W), uint8 (0-255).
            sigma: Gaussian blur sigma.

        Returns:
            Mask with smoothed edges.
        """
        kernel_size = int(sigma * 4) | 1  # Ensure odd
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask using morphological closing.

        Args:
            mask: Input mask (H, W), uint8 (0-255).

        Returns:
            Mask with holes filled.
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Fill all contours
        result = np.zeros_like(mask)
        cv2.drawContours(result, contours, -1, 255, -1)

        return result

    @staticmethod
    def remove_small_regions(
        mask: np.ndarray,
        min_area: int = 100,
    ) -> np.ndarray:
        """
        Remove small connected components from mask.

        Args:
            mask: Input mask (H, W), uint8 (0-255).
            min_area: Minimum area in pixels to keep.

        Returns:
            Mask with small regions removed.
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        result = np.zeros_like(mask)

        # Keep only components larger than min_area
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 255

        return result


@dataclass
class MaskBuilder:
    """
    Builder for creating masks from face parsing.

    Provides a fluent interface for building masks with
    component selection and refinement options.
    """

    parsing: np.ndarray
    components: set[str] = field(default_factory=lambda: DEFAULT_FACE_COMPONENTS.copy())
    refinement: MaskRefinementConfig = field(default_factory=MaskRefinementConfig)

    def include(self, *components: str) -> "MaskBuilder":
        """Add components to include set."""
        self.components.update(components)
        return self

    def exclude(self, *components: str) -> "MaskBuilder":
        """Remove components from include set."""
        self.components -= set(components)
        return self

    def set_components(self, components: set[str]) -> "MaskBuilder":
        """Set components directly."""
        self.components = components.copy()
        return self

    def with_refinement(self, config: MaskRefinementConfig) -> "MaskBuilder":
        """Set refinement configuration."""
        self.refinement = config
        return self

    def erode(self, size: int) -> "MaskBuilder":
        """Set erosion size."""
        self.refinement.erode_size = size
        return self

    def dilate(self, size: int) -> "MaskBuilder":
        """Set dilation size."""
        self.refinement.dilate_size = size
        return self

    def blur(self, size: int) -> "MaskBuilder":
        """Set blur size."""
        self.refinement.blur_size = size
        return self

    def build(self) -> np.ndarray:
        """
        Build the final mask.

        Returns:
            Binary mask (H, W) with values 0 or 255.
        """
        # Combine component masks
        mask = MaskOperations.combine_component_masks(
            self.parsing,
            self.components,
        )

        # Apply refinement
        mask = MaskOperations.refine(mask, self.refinement)

        return mask
