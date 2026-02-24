"""
CPBD (Cumulative Probability of Blur Detection) sharpness metric.

This module implements CPBD (Cumulative Probability of Blur Detection) metrics.
It provides a no-reference sharpness score where higher values indicate
sharper images.
"""

from __future__ import annotations

from math import atan2, pi
from typing import cast

import cv2
import numpy as np

# Threshold to characterize blocks as edge/non-edge blocks.
THRESHOLD = 0.002
# Fitting parameter.
BETA = 3.6
# Block size.
BLOCK_HEIGHT, BLOCK_WIDTH = (64, 64)
# Just noticeable widths based on perceptual experiments.
WIDTH_JNB = np.concatenate([5 * np.ones(51), 3 * np.ones(205)])


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    """Detect edges with Sobel operator + simple thinning."""
    from scipy.ndimage import convolve
    from skimage.filters.edges import HSOBEL_WEIGHTS

    h1 = np.array(HSOBEL_WEIGHTS, dtype=np.float64)
    h1 /= np.sum(np.abs(h1))

    strength2 = np.square(convolve(image, h1.T))
    thresh2 = 2 * np.sqrt(np.mean(strength2))
    strength2[strength2 <= thresh2] = 0
    return _simple_thinning(strength2)


def _simple_thinning(strength: np.ndarray) -> np.ndarray:
    """Perform lightweight edge thinning."""
    num_rows, num_cols = strength.shape

    zero_column = np.zeros((num_rows, 1))
    zero_row = np.zeros((1, num_cols))

    x = (strength > np.c_[zero_column, strength[:, :-1]]) & (
        strength > np.c_[strength[:, 1:], zero_column]
    )
    y = (strength > np.r_[zero_row, strength[:-1, :]]) & (
        strength > np.r_[strength[1:, :], zero_row]
    )
    return cast(np.ndarray, x | y)


def _is_edge_block(block: np.ndarray, threshold: float) -> bool:
    """Decide whether the given block is an edge block."""
    return bool(np.count_nonzero(block) > (block.size * threshold))


def _get_block_contrast(block: np.ndarray) -> int:
    """Get block contrast as max-min."""
    return int(np.max(block) - np.min(block))


def _marziliano_method(edges: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Calculate edge widths with Marziliano method."""
    edge_widths = np.zeros(image.shape, dtype=np.float64)

    gradient_y, gradient_x = np.gradient(image)
    img_height, img_width = image.shape
    edge_angles = np.zeros(image.shape, dtype=np.float64)

    for row in range(img_height):
        for col in range(img_width):
            gx = gradient_x[row, col]
            gy = gradient_y[row, col]
            if gx != 0:
                edge_angles[row, col] = atan2(gy, gx) * (180 / pi)
            elif gx == 0 and gy == 0:
                edge_angles[row, col] = 0
            elif gx == 0 and gy == pi / 2:
                edge_angles[row, col] = 90

    if np.any(edge_angles):
        quantized_angles = 45 * np.round(edge_angles / 45)

        for row in range(1, img_height - 1):
            for col in range(1, img_width - 1):
                if edges[row, col] != 1:
                    continue

                # Gradient angle = 180 or -180.
                if quantized_angles[row, col] in (180, -180):
                    for margin in range(101):
                        inner_border = (col - 1) - margin
                        outer_border = (col - 2) - margin
                        if (
                            outer_border < 0
                            or (image[row, outer_border] - image[row, inner_border])
                            <= 0
                        ):
                            break
                    width_left = margin + 1

                    for margin in range(101):
                        inner_border = (col + 1) + margin
                        outer_border = (col + 2) + margin
                        if (
                            outer_border >= img_width
                            or (image[row, outer_border] - image[row, inner_border])
                            >= 0
                        ):
                            break
                    width_right = margin + 1
                    edge_widths[row, col] = width_left + width_right

                # Gradient angle = 0.
                if quantized_angles[row, col] == 0:
                    for margin in range(101):
                        inner_border = (col - 1) - margin
                        outer_border = (col - 2) - margin
                        if (
                            outer_border < 0
                            or (image[row, outer_border] - image[row, inner_border])
                            >= 0
                        ):
                            break
                    width_left = margin + 1

                    for margin in range(101):
                        inner_border = (col + 1) + margin
                        outer_border = (col + 2) + margin
                        if (
                            outer_border >= img_width
                            or (image[row, outer_border] - image[row, inner_border])
                            <= 0
                        ):
                            break
                    width_right = margin + 1
                    edge_widths[row, col] = width_right + width_left

    return cast(np.ndarray, edge_widths)


def _calculate_sharpness_metric(
    image: np.ndarray,
    edges: np.ndarray,
    edge_widths: np.ndarray,
) -> float:
    """Calculate final CPBD metric."""
    img_height, img_width = image.shape
    total_num_edges = 0
    hist_pblur: np.ndarray = np.zeros(101, dtype=np.float64)

    num_blocks_vertically = int(img_height / BLOCK_HEIGHT)
    num_blocks_horizontally = int(img_width / BLOCK_WIDTH)

    for i in range(num_blocks_vertically):
        for j in range(num_blocks_horizontally):
            rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
            cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))

            if not _is_edge_block(edges[rows, cols], THRESHOLD):
                continue

            block_widths = edge_widths[rows, cols]
            block_widths = np.rot90(np.flipud(block_widths), 3)
            block_widths = block_widths[block_widths != 0]

            block_contrast = _get_block_contrast(image[rows, cols])
            block_jnb = WIDTH_JNB[block_contrast]
            prob_blur_detection = 1 - np.exp(
                -(np.abs(block_widths / block_jnb) ** BETA)
            )

            for probability in prob_blur_detection:
                bucket = int(round(probability * 100))
                hist_pblur[bucket] += 1
                total_num_edges += 1

    if total_num_edges > 0:
        hist_pblur = hist_pblur / total_num_edges

    return float(np.sum(hist_pblur[:64]))


def estimate_cpbd_sharpness(image: np.ndarray) -> float:
    """
    Estimate sharpness using CPBD.

    Args:
        image: BGR or grayscale image.

    Returns:
        CPBD sharpness score (higher is sharper).
    """
    from skimage.feature import canny

    if image.ndim == 3:
        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image[..., 0]

    image = image.astype(np.float64)
    canny_edges = canny(image)
    sobel_edges = _sobel_edges(image)
    marziliano_widths = _marziliano_method(sobel_edges, image)
    return _calculate_sharpness_metric(image, canny_edges, marziliano_widths)
