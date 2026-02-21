"""Image degradation effects for matching target video quality.

Provides functions to apply various degradation effects to match
the quality characteristics of the target video, improving visual
consistency in face swap results.
"""

from typing import cast

import cv2
import numpy as np


def apply_denoise(image: np.ndarray, power: int) -> np.ndarray:
    """
    Apply progressive median blur denoising.

    Uses iterative median blur with gradual blending for smooth
    denoising effect.

    Args:
        image: Input image (H, W, 3) float32 [0, 1].
        power: Denoise power (0-500). 0 = disabled.

    Returns:
        Denoised image (H, W, 3) float32 [0, 1].
    """
    if power <= 0:
        return image

    result = image.copy()
    n = power

    while n > 0:
        # Apply median blur
        img_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, 5).astype(np.float32) / 255.0

        if int(n / 100) != 0:
            # Full application for values >= 100
            result = denoised
        else:
            # Partial blend for remainder
            blend = (n % 100) / 100.0
            result = result * (1.0 - blend) + denoised * blend

        n = max(n - 100, 0)

    return cast(np.ndarray, np.clip(result, 0, 1).astype(np.float32))


def apply_bicubic_degrade(image: np.ndarray, power: int) -> np.ndarray:
    """
    Degrade image quality by downscale/upscale cycle.

    Reduces effective resolution by scaling down then back up,
    simulating lower quality source footage.

    Args:
        image: Input image (H, W, 3) float32 [0, 1].
        power: Degrade power (0-100). 0 = disabled.

    Returns:
        Degraded image (H, W, 3) float32 [0, 1].
    """
    if power <= 0:
        return image

    h, w = image.shape[:2]

    # Calculate scale factor (power 100 = scale to ~1% of original)
    scale = 1.0 - power / 101.0
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))

    # Convert to uint8 for resize
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Downscale then upscale
    downscaled = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    return upscaled.astype(np.float32) / 255.0


def apply_color_degrade(image: np.ndarray, power: int) -> np.ndarray:
    """
    Reduce color depth through quantization.

    Simulates lower bit-depth color representation by reducing
    the number of distinct color levels with dynamic quantization.

    Args:
        image: Input image (H, W, 3) float32 [0, 1].
        power: Degrade power (0-100). 0 = disabled.

    Returns:
        Color-degraded image (H, W, 3) float32 [0, 1].
    """
    if power <= 0:
        return image

    # Convert to uint8
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Dynamic levels based on power (power=100 -> 16, power=50 -> ~130)
    levels = max(16, int(256 - (240 * power / 100)))
    step = 256 // levels
    reduced = (img_uint8 // step) * step
    reduced_f = reduced.astype(np.float32) / 255.0

    # Blend based on power
    alpha = power / 100.0
    result = image * (1.0 - alpha) + reduced_f * alpha

    return np.clip(result, 0, 1).astype(np.float32)


def apply_degradation_pipeline(
    image: np.ndarray,
    denoise_power: int = 0,
    bicubic_power: int = 0,
    color_power: int = 0,
) -> np.ndarray:
    """
    Apply full degradation pipeline.

    Applies denoising, bicubic degradation, and color degradation
    in sequence for complete quality matching.

    Args:
        image: Input image (H, W, 3) float32 [0, 1].
        denoise_power: Denoise power (0-500).
        bicubic_power: Bicubic degrade power (0-100).
        color_power: Color degrade power (0-100).

    Returns:
        Degraded image (H, W, 3) float32 [0, 1].
    """
    result = image

    if denoise_power > 0:
        result = apply_denoise(result, denoise_power)

    if bicubic_power > 0:
        result = apply_bicubic_degrade(result, bicubic_power)

    if color_power > 0:
        result = apply_color_degrade(result, color_power)

    return result
