"""
Image Blending functions for face merging.

Implements multiple blending algorithms for seamless face compositing:

- Laplacian Pyramid Blending: Multi-band frequency-based fusion
- Poisson Blending: Gradient-domain seamless cloning (via cv2.seamlessClone)
- Feather Blending: Simple alpha blending with feathered edges

All functions expect float32 images in [0, 1] range.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal, List

BlendMode = Literal["laplacian", "poisson", "feather"]


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Gaussian pyramid from image.

    Args:
        img: Input image (H, W, C) or (H, W).
        levels: Number of downsampling levels.

    Returns:
        List of images from original resolution to smallest.
    """
    pyramid = [img.copy()]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Laplacian pyramid from image.

    Each level contains the difference between Gaussian levels,
    capturing frequency information at different scales.

    Args:
        img: Input image (H, W, C) float32.
        levels: Number of pyramid levels.

    Returns:
        List of Laplacian images, with residual at the end.
    """
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []

    for i in range(levels):
        # Upsample lower resolution and compute difference
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        diff = gaussian[i].astype(np.float32) - expanded.astype(np.float32)
        laplacian.append(diff)

    # Add residual (lowest resolution Gaussian)
    laplacian.append(gaussian[-1].astype(np.float32))

    return laplacian


def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from Laplacian pyramid.

    Args:
        pyramid: List of Laplacian images with residual at the end.

    Returns:
        Reconstructed image.
    """
    img = pyramid[-1].astype(np.float32)

    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = img + pyramid[i]

    return img


def laplacian_pyramid_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    levels: int = 6,
) -> np.ndarray:
    """
    Blend images using Laplacian pyramid.

    Creates seamless multi-band blending by combining frequency
    components at different scales. This produces smooth transitions
    without visible seams.

    Args:
        foreground: Foreground image (H, W, 3) float32 [0, 1].
        background: Background image (H, W, 3) float32 [0, 1].
        mask: Blending mask (H, W, 1) or (H, W) float32 [0, 1].
            1 = foreground, 0 = background.
        levels: Number of pyramid levels. Default: 6.
            Higher values = smoother blending but more computation.

    Returns:
        Blended image (H, W, 3) float32 [0, 1].

    Example:
        >>> fg = cv2.imread('face.png').astype(np.float32) / 255
        >>> bg = cv2.imread('scene.png').astype(np.float32) / 255
        >>> mask = cv2.imread('mask.png', 0).astype(np.float32) / 255
        >>> result = laplacian_pyramid_blend(fg, bg, mask)
    """
    # Ensure mask has 3 channels for element-wise operations
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, 3, axis=-1)

    # Limit levels based on image size
    min_dim = min(foreground.shape[0], foreground.shape[1])
    max_levels = int(np.log2(min_dim)) - 1
    levels = min(levels, max_levels)

    # Build pyramids
    fg_pyramid = build_laplacian_pyramid(foreground, levels)
    bg_pyramid = build_laplacian_pyramid(background, levels)
    mask_pyramid = build_gaussian_pyramid(mask, levels)

    # Blend at each level
    blended_pyramid = []
    for fg, bg, m in zip(fg_pyramid, bg_pyramid, mask_pyramid):
        # Ensure mask dimensions match
        if m.ndim == 2:
            m = m[..., None]
        if m.shape[-1] == 1:
            m = np.repeat(m, 3, axis=-1)

        # Blend: fg * mask + bg * (1 - mask)
        blended = fg * m + bg * (1 - m)
        blended_pyramid.append(blended)

    # Reconstruct
    result = reconstruct_from_laplacian(blended_pyramid)

    return np.clip(result, 0, 1).astype(np.float32)


def poisson_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    mode: int = cv2.NORMAL_CLONE,
) -> np.ndarray:
    """
    Blend using Poisson image editing (seamlessClone).

    Uses gradient-domain blending for seamless compositing.
    Preserves texture details while matching colors at boundaries.

    Args:
        foreground: Foreground image (H, W, 3) float32 [0, 1].
        background: Background image (H, W, 3) float32 [0, 1].
        mask: Binary mask (H, W) or (H, W, 1) indicating foreground region.
        center: Center point (x, y) for blending. Default: mask centroid.
        mode: cv2 clone mode. Default: cv2.NORMAL_CLONE.
            - cv2.NORMAL_CLONE: Preserves texture of foreground
            - cv2.MIXED_CLONE: Mixes textures at boundaries

    Returns:
        Blended image (H, W, 3) float32 [0, 1].

    Note:
        The mask should be a binary mask (0 or 255 for uint8).
        Non-binary masks will be thresholded.
    """
    # Convert to uint8
    fg_uint8 = np.clip(foreground * 255, 0, 255).astype(np.uint8)
    bg_uint8 = np.clip(background * 255, 0, 255).astype(np.uint8)

    # Prepare mask
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8[..., 0]

    # Threshold to binary
    _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Find center from mask if not provided
    if center is None:
        moments = cv2.moments(mask_uint8)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center = (cx, cy)
        else:
            # Fallback to image center
            center = (mask_uint8.shape[1] // 2, mask_uint8.shape[0] // 2)

    # Check if mask has any non-zero pixels
    if cv2.countNonZero(mask_uint8) == 0:
        # No foreground region, return background
        return background.copy()

    try:
        result = cv2.seamlessClone(fg_uint8, bg_uint8, mask_uint8, center, mode)
    except cv2.error:
        # Fallback to simple alpha blend if seamlessClone fails
        mask_float = mask_uint8.astype(np.float32) / 255.0
        if mask_float.ndim == 2:
            mask_float = mask_float[..., None]
        result = (fg_uint8 * mask_float + bg_uint8 * (1 - mask_float)).astype(np.uint8)

    return result.astype(np.float32) / 255.0


def feather_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    feather_amount: int = 15,
) -> np.ndarray:
    """
    Simple alpha blending with feathered edges.

    Applies Gaussian blur to the mask edges for smooth transitions.

    Args:
        foreground: Foreground image (H, W, 3) float32 [0, 1].
        background: Background image (H, W, 3) float32 [0, 1].
        mask: Blending mask (H, W, 1) or (H, W) float32 [0, 1].
        feather_amount: Gaussian blur kernel radius. Default: 15.
            Higher values = softer edges.

    Returns:
        Blended image (H, W, 3) float32 [0, 1].
    """
    # Ensure mask is 2D for blurring
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = mask.astype(np.float32)

    # Feather the mask edges with Gaussian blur
    if feather_amount > 0:
        kernel_size = feather_amount * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # Expand mask for blending
    mask = mask[..., None]

    # Blend
    result = foreground * mask + background * (1 - mask)

    return np.clip(result, 0, 1).astype(np.float32)


def erode_mask(
    mask: np.ndarray,
    erosion_size: int = 5,
) -> np.ndarray:
    """
    Erode mask to shrink the foreground region.

    Useful for hiding edge artifacts before blending.

    Args:
        mask: Binary or float mask (H, W) or (H, W, 1).
        erosion_size: Erosion kernel size. Default: 5.

    Returns:
        Eroded mask with same shape as input.
    """
    if mask.ndim == 3:
        mask_2d = mask[..., 0]
    else:
        mask_2d = mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )

    if mask_2d.dtype == np.float32 or mask_2d.dtype == np.float64:
        mask_uint8 = (mask_2d * 255).astype(np.uint8)
        eroded = cv2.erode(mask_uint8, kernel)
        result = eroded.astype(np.float32) / 255.0
    else:
        result = cv2.erode(mask_2d, kernel)

    if mask.ndim == 3:
        result = result[..., None]

    return result


def dilate_mask(
    mask: np.ndarray,
    dilation_size: int = 5,
) -> np.ndarray:
    """
    Dilate mask to expand the foreground region.

    Args:
        mask: Binary or float mask (H, W) or (H, W, 1).
        dilation_size: Dilation kernel size. Default: 5.

    Returns:
        Dilated mask with same shape as input.
    """
    if mask.ndim == 3:
        mask_2d = mask[..., 0]
    else:
        mask_2d = mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )

    if mask_2d.dtype == np.float32 or mask_2d.dtype == np.float64:
        mask_uint8 = (mask_2d * 255).astype(np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel)
        result = dilated.astype(np.float32) / 255.0
    else:
        result = cv2.dilate(mask_2d, kernel)

    if mask.ndim == 3:
        result = result[..., None]

    return result


def blend(
    mode: BlendMode,
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Unified blending interface.

    Args:
        mode: Blend mode ('laplacian', 'poisson', 'feather').
        foreground: Foreground image (H, W, 3) float32 [0, 1].
        background: Background image (H, W, 3) float32 [0, 1].
        mask: Blending mask.
        **kwargs: Additional arguments passed to specific function.

    Returns:
        Blended image (H, W, 3) float32 [0, 1].

    Raises:
        ValueError: If unknown mode is specified.

    Example:
        >>> result = blend('laplacian', fg, bg, mask, levels=6)
        >>> result = blend('poisson', fg, bg, mask)
        >>> result = blend('feather', fg, bg, mask, feather_amount=20)
    """
    if mode == "laplacian":
        return laplacian_pyramid_blend(foreground, background, mask, **kwargs)
    elif mode == "poisson":
        return poisson_blend(foreground, background, mask, **kwargs)
    elif mode == "feather":
        return feather_blend(foreground, background, mask, **kwargs)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
