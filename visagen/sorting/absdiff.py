"""
Absolute difference sorting methods.

Provides GPU-accelerated sorting by pixel-wise absolute difference.
This is a modern PyTorch reimplementation of DeepFaceLab's sort_by_absdiff.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult

if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
    from visagen.vision.dflimg import FaceMetadata

logger = logging.getLogger(__name__)


class AbsDiffSorter(SortMethod):
    """
    Sort by absolute pixel difference.

    Computes pairwise absolute differences between all images and sorts
    using a greedy nearest-neighbor (or farthest-neighbor) algorithm.

    Uses PyTorch for GPU acceleration when available.

    Args:
        similar: If True, sort by similarity (nearest neighbor).
                 If False, sort by dissimilarity (farthest neighbor).
        batch_size: Batch size for GPU processing.
        target_size: Resize images to this size for comparison.
                    Smaller = faster but less accurate.

    Example:
        >>> sorter = AbsDiffSorter(similar=True)
        >>> result = sorter.sort(image_paths)
    """

    name = "absdiff"
    description = "Sort by absolute pixel difference (GPU-accelerated)"
    requires_dfl_metadata = False
    execution_profile = "gpu_bound"

    def __init__(
        self,
        similar: bool = True,
        batch_size: int = 64,
        target_size: int = 128,
        exact_limit: int = 3000,
    ) -> None:
        self.similar = similar
        self.batch_size = batch_size
        self.target_size = target_size
        self.exact_limit = exact_limit

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Not used - this method uses custom sort."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: "ParallelSortProcessor | None" = None,
    ) -> SortOutput:
        """
        Sort images by absolute difference.

        Computes a distance matrix between all image pairs and performs
        greedy nearest/farthest neighbor sorting.

        Args:
            image_paths: List of image paths to sort.
            processor: Optional parallel processor (not used, GPU handles parallelism).

        Returns:
            SortOutput with sorted images.
        """
        start_time = time.time()

        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        # Load and preprocess all images
        images: list[tuple[Path, np.ndarray]] = []
        trash: list[SortResult] = []
        use_approx = len(image_paths) > self.exact_limit
        comparison_size = (
            self.target_size if not use_approx else min(self.target_size, 32)
        )

        logger.info(f"Loading {len(image_paths)} images...")
        for filepath in image_paths:
            try:
                image = cv2.imread(str(filepath))
                if image is None:
                    trash.append(SortResult(filepath, 0.0, {"error": "Failed to load"}))
                    continue

                # Resize for faster comparison
                resized = cv2.resize(
                    image,
                    (comparison_size, comparison_size),
                    interpolation=cv2.INTER_AREA,
                )
                # Normalize to float32 [0, 1]
                normalized = resized.astype(np.float32) / 255.0
                images.append((filepath, normalized))

            except Exception as e:
                trash.append(SortResult(filepath, 0.0, {"error": str(e)}))

        if len(images) == 0:
            return SortOutput([], trash, self.name, time.time() - start_time)

        n = len(images)
        if use_approx:
            logger.info(
                f"Large dataset detected (n={n}), using approximate path (limit={self.exact_limit})"
            )
            sorted_indices = self._approximate_sort(images, self.similar)
        else:
            logger.info(f"Computing distance matrix for {n} images...")
            try:
                distance_matrix = self._compute_distance_matrix_gpu(images)
            except Exception as e:
                logger.warning(f"GPU computation failed, falling back to CPU: {e}")
                distance_matrix = self._compute_distance_matrix_cpu(images)

            logger.info("Performing greedy sorting...")
            sorted_indices = self._greedy_sort(distance_matrix, self.similar)

        # Build results
        results = [
            SortResult(images[idx][0], float(i)) for i, idx in enumerate(sorted_indices)
        ]

        elapsed = time.time() - start_time
        logger.info(f"Sorting completed in {elapsed:.1f}s")

        return SortOutput(results, trash, self.name, elapsed)

    def _approximate_sort(
        self,
        images: list[tuple[Path, np.ndarray]],
        similar: bool,
    ) -> list[int]:
        """
        Approximate large-scale ordering without full NxN matrix allocation.

        Similar mode: order by 1D projection for locality.
        Dissimilar mode: rank by L1 distance to centroid.
        """
        flat = np.stack([img.flatten() for _, img in images], axis=0).astype(np.float32)

        if similar:
            weights = np.linspace(-1.0, 1.0, flat.shape[1], dtype=np.float32)
            projection = flat @ weights
            return cast(list[int], np.argsort(projection).tolist())

        centroid = np.mean(flat, axis=0, keepdims=True)
        scores = np.abs(flat - centroid).sum(axis=1)
        return cast(list[int], np.argsort(scores)[::-1].tolist())

    def _compute_distance_matrix_gpu(
        self,
        images: list[tuple[Path, np.ndarray]],
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix using PyTorch GPU.

        Args:
            images: List of (path, normalized_image) tuples.

        Returns:
            Distance matrix (n x n) numpy array.
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        n = len(images)

        # Stack all images into a tensor
        image_stack = np.stack([img for _, img in images], axis=0)
        # Flatten each image: (n, h, w, c) -> (n, h*w*c)
        image_flat = image_stack.reshape(n, -1)

        # Convert to torch tensor
        tensor = torch.from_numpy(image_flat).to(device)

        # Compute pairwise L1 distances in batches
        distance_matrix: np.ndarray = np.zeros((n, n), dtype=np.float32)

        batch_size = self.batch_size
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            batch_i = tensor[i:i_end]  # (batch, features)

            for j in range(i, n, batch_size):
                j_end = min(j + batch_size, n)
                batch_j = tensor[j:j_end]  # (batch, features)

                # Compute L1 distances: |batch_i - batch_j|
                # batch_i: (bi, f), batch_j: (bj, f)
                # Result: (bi, bj)
                diff = torch.abs(
                    batch_i.unsqueeze(1) - batch_j.unsqueeze(0)
                )  # (bi, bj, f)
                distances = diff.sum(dim=2).cpu().numpy()  # (bi, bj)

                distance_matrix[i:i_end, j:j_end] = distances
                # Mirror for symmetric matrix
                if i != j:
                    distance_matrix[j:j_end, i:i_end] = distances.T

        return distance_matrix

    def _compute_distance_matrix_cpu(
        self,
        images: list[tuple[Path, np.ndarray]],
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix using CPU (fallback).

        Args:
            images: List of (path, normalized_image) tuples.

        Returns:
            Distance matrix (n x n) numpy array.
        """
        n = len(images)

        # Flatten images
        image_flat = np.array([img.flatten() for _, img in images])

        # Compute pairwise L1 distances
        distance_matrix: np.ndarray = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.abs(image_flat[i] - image_flat[j]).sum()
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _greedy_sort(
        self,
        distance_matrix: np.ndarray,
        similar: bool,
    ) -> list[int]:
        """
        Perform greedy nearest/farthest neighbor sorting.

        Args:
            distance_matrix: Pairwise distance matrix.
            similar: If True, find nearest neighbor. If False, farthest.

        Returns:
            List of sorted indices.
        """
        n = distance_matrix.shape[0]
        sorted_indices = [0]
        remaining = set(range(1, n))

        for _ in range(n - 1):
            current_idx = sorted_indices[-1]

            # Get distances from current to all remaining
            distances = [(idx, distance_matrix[current_idx, idx]) for idx in remaining]

            if similar:
                # Find nearest (minimum distance)
                next_idx = min(distances, key=lambda x: x[1])[0]
            else:
                # Find farthest (maximum distance)
                next_idx = max(distances, key=lambda x: x[1])[0]

            sorted_indices.append(next_idx)
            remaining.remove(next_idx)

        return sorted_indices


class AbsDiffDissimilaritySorter(AbsDiffSorter):
    """
    Sort by absolute pixel difference (dissimilar mode).

    Sorts images so that each successive image is as different
    as possible from the previous one.

    Example:
        >>> sorter = AbsDiffDissimilaritySorter()
        >>> result = sorter.sort(image_paths)
    """

    name = "absdiff-dissim"
    description = "Sort by absolute pixel difference (dissimilar, GPU-accelerated)"

    def __init__(
        self,
        batch_size: int = 64,
        target_size: int = 128,
        exact_limit: int = 3000,
    ) -> None:
        super().__init__(
            similar=False,
            batch_size=batch_size,
            target_size=target_size,
            exact_limit=exact_limit,
        )
