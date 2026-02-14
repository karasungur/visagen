"""
Motion Blur for temporal consistency.

Applies directional motion blur based on optical flow analysis
to reduce flickering in video face swaps.
"""

from typing import cast

import cv2
import numpy as np


def linear_motion_blur(
    image: np.ndarray,
    kernel_size: int,
    angle: float,
) -> np.ndarray:
    """
    Apply directional motion blur using rotated line kernel.

    Creates a motion blur effect by convolving with a rotated
    line kernel, simulating camera motion in the specified direction.

    Args:
        image: Input image (H, W, C) float32 [0,1] or uint8.
        kernel_size: Motion blur length in pixels (2-50).
        angle: Motion direction in degrees.

    Returns:
        Blurred image with same dtype as input.
    """
    if kernel_size < 2:
        return image

    kernel_size = min(max(kernel_size, 2), 50)

    # Create horizontal line kernel
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[(kernel_size - 1) // 2, :] = np.ones(kernel_size, dtype=np.float32)

    # Rotate kernel by angle
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    k = cast(np.ndarray, cv2.warpAffine(k, rotation_matrix, (kernel_size, kernel_size)))

    # Normalize kernel
    k = (k / np.sum(k)).astype(np.float32)

    # Apply convolution
    return cv2.filter2D(image, -1, k)


def apply_motion_blur_to_face(
    face_image: np.ndarray,
    motion_power: float,
    motion_deg: float,
    blur_power: int = 100,
    super_resolution: bool = False,
) -> np.ndarray:
    """
    Apply motion blur to face based on detected motion.

    Args:
        face_image: Face region (H, W, C).
        motion_power: Detected motion magnitude (from optical flow).
        motion_deg: Motion direction in degrees.
        blur_power: User-configured blur power (0-100).
        super_resolution: Whether super resolution is enabled (doubles kernel).

    Returns:
        Motion-blurred face image.
    """
    if blur_power == 0:
        return face_image

    cfg_mp = blur_power / 100.0
    k_size = int(motion_power * cfg_mp)

    if k_size < 1:
        return face_image

    k_size = np.clip(k_size + 1, 2, 50)

    if super_resolution:
        k_size *= 2

    return linear_motion_blur(face_image, k_size, motion_deg)
