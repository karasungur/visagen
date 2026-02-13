"""
Structural similarity (SSIM) based sorting methods.

Provides similarity and dissimilarity ordering using grayscale SSIM.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult

if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
    from visagen.vision.dflimg import FaceMetadata


def _load_ssim_image(
    filepath: Path,
    target_size: int,
) -> tuple[Path, np.ndarray | None, str | None]:
    """Load and normalize a grayscale image for SSIM comparison."""
    try:
        image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return filepath, None, "Failed to load image"

        resized = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
        return filepath, resized.astype(np.float32), None
    except Exception as e:
        return filepath, None, str(e)


def _load_images_for_ssim(
    image_paths: list[Path],
    target_size: int,
    processor: ParallelSortProcessor | None,
) -> tuple[list[tuple[Path, np.ndarray]], list[SortResult]]:
    """Load images for SSIM sorting."""
    image_data: list[tuple[Path, np.ndarray]] = []
    trash: list[SortResult] = []

    if processor is None:
        loaded = [_load_ssim_image(path, target_size) for path in image_paths]
    else:
        loaded = []
        executor_class = (
            ThreadPoolExecutor if processor.use_threads else ProcessPoolExecutor
        )
        with executor_class(max_workers=processor.max_workers) as executor:
            futures = {
                executor.submit(_load_ssim_image, path, target_size): path
                for path in image_paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    loaded.append(future.result())
                except Exception as e:
                    loaded.append((path, None, str(e)))

    for path, image, error in loaded:
        if image is None:
            trash.append(SortResult(path, 0.0, {"error": error or "Failed to load"}))
            continue
        image_data.append((path, image))

    return image_data, trash


def _ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM similarity score in [0, 1] for two grayscale float images.
    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    score = float(np.mean(ssim_map))
    return float(np.clip(score, 0.0, 1.0))


def _projection_order(matrix: np.ndarray) -> list[int]:
    """Get projection-based order for approximate similarity chain."""
    weights = np.linspace(-1.0, 1.0, matrix.shape[1], dtype=np.float32)
    projection = matrix @ weights
    return cast(list[int], np.argsort(projection).tolist())


class SSIMSimilaritySorter(SortMethod):
    """Sort by structural similarity."""

    name = "ssim"
    description = "Sort by SSIM similarity (groups similar images)"
    requires_dfl_metadata = False
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 3000, target_size: int = 96) -> None:
        self.exact_limit = exact_limit
        self.target_size = target_size

    def compute_score(
        self,
        image: np.ndarray,
        metadata: FaceMetadata | None = None,
    ) -> float:
        """Not used - this method uses custom sort."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: ParallelSortProcessor | None = None,
    ) -> SortOutput:
        """Sort by SSIM similarity."""
        start_time = time.time()
        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        image_data, trash = _load_images_for_ssim(
            image_paths, self.target_size, processor
        )
        if len(image_data) == 0:
            return SortOutput([], trash, self.name, time.time() - start_time)

        n = len(image_data)
        if n <= self.exact_limit:
            order = [0]
            remaining = set(range(1, n))
            for _ in range(n - 1):
                current_idx = order[-1]
                current_image = image_data[current_idx][1]

                next_idx = min(
                    remaining,
                    key=lambda idx: 1.0
                    - _ssim_similarity(current_image, image_data[idx][1]),
                )
                order.append(next_idx)
                remaining.remove(next_idx)
        else:
            matrix = np.stack([img.flatten() for _, img in image_data], axis=0).astype(
                np.float32
            )
            order = _projection_order(matrix)

        results = [
            SortResult(image_data[idx][0], float(i)) for i, idx in enumerate(order)
        ]
        return SortOutput(results, trash, self.name, time.time() - start_time)


class SSIMDissimilaritySorter(SortMethod):
    """Sort by structural dissimilarity."""

    name = "ssim-dissim"
    description = "Sort by SSIM dissimilarity (outliers first)"
    requires_dfl_metadata = False
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 3000, target_size: int = 96) -> None:
        self.exact_limit = exact_limit
        self.target_size = target_size

    def compute_score(
        self,
        image: np.ndarray,
        metadata: FaceMetadata | None = None,
    ) -> float:
        """Not used - this method uses custom sort."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: ParallelSortProcessor | None = None,
    ) -> SortOutput:
        """Sort by SSIM dissimilarity."""
        start_time = time.time()
        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        image_data, trash = _load_images_for_ssim(
            image_paths, self.target_size, processor
        )
        if len(image_data) == 0:
            return SortOutput([], trash, self.name, time.time() - start_time)

        n = len(image_data)
        scores: list[float] = []

        if n <= self.exact_limit:
            for i in range(n):
                total_distance = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    total_distance += 1.0 - _ssim_similarity(
                        image_data[i][1], image_data[j][1]
                    )
                scores.append(total_distance)
        else:
            matrix = np.stack([img.flatten() for _, img in image_data], axis=0).astype(
                np.float32
            )
            centroid = np.mean(matrix, axis=0, keepdims=True)
            scores = np.abs(matrix - centroid).sum(axis=1).astype(np.float32).tolist()

        order = np.argsort(np.array(scores, dtype=np.float32))[::-1].tolist()
        results = [SortResult(image_data[idx][0], float(scores[idx])) for idx in order]
        return SortOutput(results, trash, self.name, time.time() - start_time)
