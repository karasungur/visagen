"""
Histogram-based sorting methods.

Provides sorting by histogram similarity and dissimilarity.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult
from visagen.sorting.blur import get_face_hull_mask

if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
    from visagen.vision.face_image import FaceMetadata


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute color histogram for an image.

    Args:
        image: BGR image.
        bins: Number of histogram bins.

    Returns:
        Concatenated histogram for all channels.
    """
    histograms = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return cast(np.ndarray, np.concatenate(histograms))


def compute_grayscale_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute grayscale histogram for an image.

    Args:
        image: BGR or grayscale image.
        bins: Number of histogram bins.

    Returns:
        Normalized histogram array.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return cast(np.ndarray, cv2.normalize(hist, hist).flatten())


def _should_run_exact(n_items: int, exact_limit: int) -> bool:
    """Return whether exact O(n^2) strategy should run."""
    return exact_limit > 0 and n_items <= exact_limit


def _projection_values(hist_matrix: np.ndarray) -> np.ndarray:
    """Project histogram vectors onto deterministic 1D axis."""
    weights = np.linspace(-1.0, 1.0, hist_matrix.shape[1], dtype=np.float32)
    return cast(np.ndarray, (hist_matrix @ weights).reshape(-1))


def _refine_projection_chain(
    hist_matrix: np.ndarray,
    projection: np.ndarray,
    base_order: np.ndarray,
    window_size: int = 32,
) -> list[int]:
    """
    Build a local-refined similarity chain from projection ordering.

    Uses only a bounded neighborhood around current rank to avoid O(n^2) scans.
    """
    n_items = len(base_order)
    if n_items <= 2:
        return cast(list[int], base_order.tolist())

    window = max(8, min(window_size, n_items - 1))
    rank_lookup: np.ndarray = np.empty(n_items, dtype=np.int32)
    rank_lookup[base_order] = np.arange(n_items, dtype=np.int32)

    order: list[int] = [int(base_order[0])]
    remaining: set[int] = set(cast(list[int], base_order[1:].tolist()))

    while remaining:
        current_idx = order[-1]
        current_rank = int(rank_lookup[current_idx])
        left = max(0, current_rank - window)
        right = min(n_items, current_rank + window + 1)

        candidates = [
            int(base_order[pos])
            for pos in range(left, right)
            if int(base_order[pos]) in remaining
        ]

        if not candidates:
            current_projection = float(projection[current_idx])
            next_idx = min(
                remaining,
                key=lambda idx: abs(float(projection[idx]) - current_projection),
            )
        else:
            current_hist = hist_matrix[current_idx].reshape(-1, 1)
            next_idx = min(
                candidates,
                key=lambda idx: float(
                    cv2.compareHist(
                        current_hist,
                        hist_matrix[idx].reshape(-1, 1),
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                ),
            )

        order.append(next_idx)
        remaining.remove(next_idx)

    return order


def _approx_hist_dissim_scores(hist_matrix: np.ndarray) -> np.ndarray:
    """Compute scalable histogram dissimilarity scores."""
    n_items = hist_matrix.shape[0]
    centroid = np.mean(hist_matrix, axis=0, keepdims=True)
    centroid_scores = np.abs(hist_matrix - centroid).sum(axis=1).astype(np.float32)

    projection = _projection_values(hist_matrix)
    projection_order = np.argsort(projection)
    anchor_count = min(32, n_items)
    anchor_positions = np.linspace(0, n_items - 1, num=anchor_count, dtype=np.int32)
    anchor_indices = projection_order[anchor_positions]
    anchors = hist_matrix[anchor_indices]

    anchor_scores = np.zeros(n_items, dtype=np.float32)
    chunk_size = 512
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        chunk = hist_matrix[start:end]
        distances = np.abs(chunk[:, None, :] - anchors[None, :, :]).sum(axis=2)
        anchor_scores[start:end] = np.mean(distances, axis=1).astype(np.float32)

    return cast(np.ndarray, centroid_scores + (0.25 * anchor_scores))


class HistogramSimilaritySorter(SortMethod):
    """
    Sort by histogram similarity.

    Groups visually similar images together by creating a chain
    where each image is followed by its most similar neighbor.

    Uses Bhattacharyya distance for histogram comparison.
    """

    name = "hist"
    description = "Sort by histogram similarity (groups similar images)"
    requires_face_metadata = False
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 0) -> None:
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
        Sort images by histogram similarity.

        Creates a chain where each image is followed by its
        most similar neighbor (greedy nearest neighbor).
        """
        start_time = time.time()

        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        # Load all images and compute histograms
        image_data: list[tuple[Path, np.ndarray]] = []
        trash: list[SortResult] = []

        if processor is not None:
            processed = processor.load_and_process_all(
                image_paths,
                compute_sharpness=False,
                compute_pose=False,
                compute_histogram=True,
                histogram_mode="color",
            )
            for item in processed:
                if item.error is not None:
                    trash.append(SortResult(item.filepath, 0.0, {"error": item.error}))
                    continue
                if item.histogram is None:
                    trash.append(
                        SortResult(
                            item.filepath, 0.0, {"error": "Failed to build histogram"}
                        )
                    )
                    continue
                image_data.append((item.filepath, item.histogram))
        else:
            for filepath in image_paths:
                try:
                    image = cv2.imread(str(filepath))
                    if image is None:
                        trash.append(
                            SortResult(filepath, 0.0, {"error": "Failed to load"})
                        )
                        continue

                    hist = compute_histogram(image)
                    image_data.append((filepath, hist))
                except Exception as e:
                    trash.append(SortResult(filepath, 0.0, {"error": str(e)}))

        if len(image_data) == 0:
            return SortOutput([], trash, self.name, time.time() - start_time)

        n = len(image_data)
        if _should_run_exact(n, self.exact_limit):
            # Greedy nearest neighbor sorting (exact)
            sorted_indices = [0]
            remaining = set(range(1, n))

            for _ in range(n - 1):
                current_idx = sorted_indices[-1]
                current_hist = image_data[current_idx][1]

                best_idx = -1
                best_score = float("inf")

                for idx in remaining:
                    hist = image_data[idx][1]
                    score = cv2.compareHist(
                        current_hist.reshape(-1, 1),
                        hist.reshape(-1, 1),
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                    if score < best_score:
                        best_score = score
                        best_idx = idx

                if best_idx >= 0:
                    sorted_indices.append(best_idx)
                    remaining.remove(best_idx)
        else:
            # Approximate path for large datasets (O(n*k)).
            hist_matrix = np.stack([h for _, h in image_data], axis=0).astype(
                np.float32
            )
            projection = _projection_values(hist_matrix)
            base_order = np.argsort(projection).astype(np.int32)
            sorted_indices = _refine_projection_chain(
                hist_matrix,
                projection,
                base_order,
            )

        # Build results
        results = [
            SortResult(image_data[idx][0], float(i))
            for i, idx in enumerate(sorted_indices)
        ]

        elapsed = time.time() - start_time
        return SortOutput(results, trash, self.name, elapsed)


class HistogramDissimilaritySorter(SortMethod):
    """
    Sort by histogram dissimilarity.

    Images that are most different from others appear first.
    Useful for finding unique or outlier images.

    Computes total Bhattacharyya distance to all other images.
    """

    name = "hist-dissim"
    description = "Sort by histogram dissimilarity (unique images first)"
    requires_face_metadata = True  # Uses face mask
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 0) -> None:
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
        Sort images by histogram dissimilarity.

        Images with highest total distance to all others rank first.
        """
        start_time = time.time()

        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        # Load all images and compute histograms
        image_data: list[tuple[Path, np.ndarray]] = []
        trash: list[SortResult] = []

        if processor is not None:
            processed = processor.load_and_process_all(
                image_paths,
                compute_sharpness=False,
                compute_pose=False,
                compute_histogram=True,
                histogram_mode="gray",
            )
            for item in processed:
                if item.error is not None:
                    trash.append(SortResult(item.filepath, 0.0, {"error": item.error}))
                    continue
                if item.histogram is None:
                    trash.append(
                        SortResult(
                            item.filepath, 0.0, {"error": "Failed to build histogram"}
                        )
                    )
                    continue
                image_data.append((item.filepath, item.histogram))
        else:
            from visagen.vision.face_image import FaceImage

            for filepath in image_paths:
                try:
                    image, metadata = FaceImage.load(filepath)
                    if image is None:
                        trash.append(
                            SortResult(filepath, 0.0, {"error": "Failed to load"})
                        )
                        continue

                    # Apply face mask if available
                    if metadata is not None and metadata.landmarks is not None:
                        try:
                            mask = get_face_hull_mask(image.shape, metadata.landmarks)
                            image = (image * mask).astype(np.uint8)
                        except Exception:
                            pass

                    hist = compute_grayscale_histogram(image)
                    image_data.append((filepath, hist))
                except Exception as e:
                    trash.append(SortResult(filepath, 0.0, {"error": str(e)}))

        if len(image_data) == 0:
            return SortOutput([], trash, self.name, time.time() - start_time)

        n = len(image_data)

        # Compute dissimilarity scores
        scores: list[float] = []
        if _should_run_exact(n, self.exact_limit):
            for i in range(n):
                total_score = 0.0
                hist_i = image_data[i][1].reshape(-1, 1)

                for j in range(n):
                    if i == j:
                        continue
                    hist_j = image_data[j][1].reshape(-1, 1)
                    score = cv2.compareHist(hist_i, hist_j, cv2.HISTCMP_BHATTACHARYYA)
                    total_score += score

                scores.append(total_score)
        else:
            hist_matrix = np.stack([h for _, h in image_data], axis=0).astype(
                np.float32
            )
            scores = _approx_hist_dissim_scores(hist_matrix).tolist()

        # Build results sorted by dissimilarity (highest first)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            SortResult(image_data[idx][0], score) for idx, score in indexed_scores
        ]

        elapsed = time.time() - start_time
        return SortOutput(results, trash, self.name, elapsed)
