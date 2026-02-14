"""
Mask Preview Mode Generator.

Provides multiple visualization modes for mask preview,
compatible with legacy XSegEditor preview functionality.
"""

from enum import Enum
from typing import cast

import cv2
import numpy as np


class PreviewMode(Enum):
    """Available mask preview modes."""

    ORIGINAL = "original"  # Original image only
    OVERLAY = "overlay"  # Mask overlay with color tint
    MASKED_RESULT = "masked"  # Face only, background removed
    MASK_ONLY = "mask_only"  # Black and white mask
    INVERTED = "inverted"  # Background only, face removed
    SIDE_BY_SIDE = "side_by_side"  # Original and overlay side by side


class MaskPreviewGenerator:
    """
    Generate mask preview in various visualization modes.

    Supports multiple preview modes for mask editing workflow:
    - ORIGINAL: Show original image
    - OVERLAY: Show mask as colored overlay
    - MASKED_RESULT: Show masked face only
    - MASK_ONLY: Show black/white mask
    - INVERTED: Show background only
    - SIDE_BY_SIDE: Show original and overlay

    Args:
        overlay_color: Color for mask overlay (BGR). Default: green.
        overlay_alpha: Transparency for overlay. Default: 0.4.
        background_color: Color for background in masked result. Default: gray.

    Example:
        >>> generator = MaskPreviewGenerator()
        >>> preview = generator.generate(image, mask, PreviewMode.OVERLAY)
    """

    def __init__(
        self,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.4,
        background_color: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.overlay_color = overlay_color
        self.overlay_alpha = overlay_alpha
        self.background_color = background_color

    def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mode: PreviewMode,
    ) -> np.ndarray:
        """
        Generate preview in specified mode.

        Args:
            image: Base image (BGR or RGB format, uint8).
            mask: Binary mask (H, W), values 0-255.
            mode: Preview mode to generate.

        Returns:
            Preview image in same format as input.
        """
        if mode == PreviewMode.ORIGINAL:
            return self._preview_original(image)
        elif mode == PreviewMode.OVERLAY:
            return self._preview_overlay(image, mask)
        elif mode == PreviewMode.MASKED_RESULT:
            return self._preview_masked(image, mask)
        elif mode == PreviewMode.MASK_ONLY:
            return self._preview_mask_only(mask)
        elif mode == PreviewMode.INVERTED:
            return self._preview_inverted(image, mask)
        elif mode == PreviewMode.SIDE_BY_SIDE:
            return self._preview_side_by_side(image, mask)
        else:
            return image.copy()

    def _preview_original(self, image: np.ndarray) -> np.ndarray:
        """Return original image."""
        return cast(np.ndarray, image.copy())

    def _preview_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Generate overlay preview with colored mask."""
        result = image.copy()
        overlay = image.copy()

        # Create colored overlay in mask region
        mask_bool = mask > 127
        overlay[mask_bool] = self.overlay_color

        # Blend
        result = cv2.addWeighted(
            result,
            1 - self.overlay_alpha,
            overlay,
            self.overlay_alpha,
            0,
        )

        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, self.overlay_color, 2)

        return cast(np.ndarray, result)

    def _preview_masked(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Generate masked result (face only, background removed)."""
        result = np.full_like(image, self.background_color)

        # Normalize mask to 0-1 for blending
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = mask_float[..., np.newaxis]

        # Blend
        result = (
            image.astype(np.float32) * mask_float
            + result.astype(np.float32) * (1 - mask_float)
        ).astype(np.uint8)

        return cast(np.ndarray, result)

    def _preview_mask_only(self, mask: np.ndarray) -> np.ndarray:
        """Generate black and white mask preview."""
        # Convert to 3-channel for display
        if len(mask.shape) == 2:
            return cast(np.ndarray, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        return cast(np.ndarray, mask.copy())

    def _preview_inverted(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Generate inverted preview (background only)."""
        result = np.full_like(image, self.background_color)

        # Invert mask
        inv_mask = 255 - mask
        mask_float = inv_mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = mask_float[..., np.newaxis]

        # Blend
        result = (
            image.astype(np.float32) * mask_float
            + result.astype(np.float32) * (1 - mask_float)
        ).astype(np.uint8)

        return cast(np.ndarray, result)

    def _preview_side_by_side(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Generate side-by-side preview (original | overlay)."""
        original = image.copy()
        overlay = self._preview_overlay(image, mask)

        # Add divider line
        h = image.shape[0]
        divider_width = 2

        # Concatenate horizontally
        result = np.concatenate([original, overlay], axis=1)

        # Draw divider
        cv2.line(
            result,
            (image.shape[1], 0),
            (image.shape[1], h),
            (255, 255, 255),
            divider_width,
        )

        return cast(np.ndarray, result)

    @staticmethod
    def get_mode_names() -> dict[PreviewMode, str]:
        """Get display names for preview modes."""
        return {
            PreviewMode.ORIGINAL: "Original",
            PreviewMode.OVERLAY: "Overlay",
            PreviewMode.MASKED_RESULT: "Masked",
            PreviewMode.MASK_ONLY: "Mask Only",
            PreviewMode.INVERTED: "Inverted",
            PreviewMode.SIDE_BY_SIDE: "Side by Side",
        }

    @staticmethod
    def mode_from_string(name: str) -> PreviewMode:
        """Get PreviewMode from string name."""
        name_lower = name.lower().replace(" ", "_")
        for mode in PreviewMode:
            if mode.value == name_lower:
                return mode
            if mode.name.lower() == name_lower:
                return mode
        return PreviewMode.OVERLAY


def create_checkerboard_background(
    height: int,
    width: int,
    square_size: int = 16,
    color1: tuple[int, int, int] = (200, 200, 200),
    color2: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Create checkerboard pattern for transparency visualization.

    Args:
        height: Image height.
        width: Image width.
        square_size: Size of each checkerboard square.
        color1: First checkerboard color.
        color2: Second checkerboard color.

    Returns:
        Checkerboard pattern image (H, W, 3).
    """
    result = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            row_idx = y // square_size
            col_idx = x // square_size
            color = color1 if (row_idx + col_idx) % 2 == 0 else color2

            y_end = min(y + square_size, height)
            x_end = min(x + square_size, width)
            result[y:y_end, x:x_end] = color

    return result


def apply_mask_with_checkerboard(
    image: np.ndarray,
    mask: np.ndarray,
    square_size: int = 16,
) -> np.ndarray:
    """
    Apply mask showing checkerboard where transparent.

    Args:
        image: Input image (BGR or RGB).
        mask: Binary mask.
        square_size: Checkerboard square size.

    Returns:
        Image with checkerboard background where mask is 0.
    """
    h, w = image.shape[:2]
    background = create_checkerboard_background(h, w, square_size)

    mask_float = mask.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = mask_float[..., np.newaxis]

    result = (
        image.astype(np.float32) * mask_float
        + background.astype(np.float32) * (1 - mask_float)
    ).astype(np.uint8)

    return result
