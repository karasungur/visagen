"""
Embedding-based sorting methods.

Uses identity embeddings (InsightFace/AntelopeV2) stored in DFL metadata
to sort by identity similarity and dissimilarity.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult

if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
    from visagen.vision.dflimg import FaceMetadata


def _load_embedding(
    filepath: Path,
) -> tuple[Path, np.ndarray | None, str | None]:
    """Load embedding for a single file."""
    from visagen.vision.dflimg import DFLImage

    try:
        _image, metadata = DFLImage.load(filepath)
    except Exception as e:
        return filepath, None, f"error:{e}"

    if metadata is None or metadata.embedding is None:
        return filepath, None, "missing_embedding"

    emb = np.asarray(metadata.embedding, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        return filepath, None, "invalid_embedding"

    return filepath, emb, None


def _collect_embeddings(
    image_paths: list[Path],
    processor: ParallelSortProcessor | None,
) -> tuple[list[Path], list[np.ndarray], list[SortResult]]:
    """Collect valid embeddings and trash entries."""
    valid_paths: list[Path] = []
    embeddings: list[np.ndarray] = []
    trash: list[SortResult] = []

    if processor is None:
        loaded = [_load_embedding(path) for path in image_paths]
    else:
        loaded = []
        executor_class = (
            ThreadPoolExecutor if processor.use_threads else ProcessPoolExecutor
        )
        with executor_class(max_workers=processor.max_workers) as executor:
            futures = {
                executor.submit(_load_embedding, path): path for path in image_paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    loaded.append(future.result())
                except Exception as e:
                    loaded.append((path, None, f"error:{e}"))

    for filepath, emb, error in loaded:
        if emb is not None:
            valid_paths.append(filepath)
            embeddings.append(emb)
            continue

        if error in {"missing_embedding", "invalid_embedding"}:
            trash.append(SortResult(filepath, 0.0, {"reason": error}))
        else:
            message = (
                error[6:]
                if isinstance(error, str) and error.startswith("error:")
                else "unknown_error"
            )
            trash.append(SortResult(filepath, 0.0, {"error": message}))

    return valid_paths, embeddings, trash


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize embedding matrix row-wise."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return cast(np.ndarray, x / np.maximum(norms, eps))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix (1 - cosine similarity)."""
    normed = _l2_normalize(embeddings)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    return cast(np.ndarray, 1.0 - sim)


def _principal_axis_projection(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings onto first principal axis."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return cast(np.ndarray, centered @ axis)


class IDSimilaritySorter(SortMethod):
    """
    Sort by identity similarity using metadata embeddings.

    For smaller datasets, uses exact greedy nearest-neighbor ordering.
    For larger datasets, uses principal-axis approximation for scalability.
    """

    name = "id-sim"
    description = "Sort by identity similarity (embedding-based)"
    requires_dfl_metadata = True
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 3000) -> None:
        self.exact_limit = exact_limit

    def compute_score(
        self,
        image: np.ndarray,
        metadata: FaceMetadata | None = None,
    ) -> float:
        """Not used; custom sort implementation."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: ParallelSortProcessor | None = None,
    ) -> SortOutput:
        """Sort by identity similarity."""
        start_time = time.time()
        valid_paths, embeddings, trash = _collect_embeddings(image_paths, processor)

        if not embeddings:
            return SortOutput([], trash, self.name, time.time() - start_time)

        matrix = np.stack(embeddings, axis=0)
        n = matrix.shape[0]

        if n <= self.exact_limit:
            dist = _cosine_distance_matrix(matrix)
            order = [0]
            remaining = set(range(1, n))
            for _ in range(n - 1):
                current = order[-1]
                next_idx = min(remaining, key=lambda idx: dist[current, idx])
                order.append(next_idx)
                remaining.remove(next_idx)
        else:
            projection = _principal_axis_projection(matrix)
            order = np.argsort(projection).tolist()

        results = [
            SortResult(valid_paths[idx], float(i)) for i, idx in enumerate(order)
        ]
        return SortOutput(results, trash, self.name, time.time() - start_time)


class IDDissimilaritySorter(SortMethod):
    """
    Sort by identity dissimilarity using metadata embeddings.

    For smaller datasets, uses total cosine distance to all other items.
    For larger datasets, uses distance to centroid approximation.
    """

    name = "id-dissim"
    description = "Sort by identity dissimilarity (embedding outliers first)"
    requires_dfl_metadata = True
    execution_profile = "cpu_bound"

    def __init__(self, exact_limit: int = 3000) -> None:
        self.exact_limit = exact_limit

    def compute_score(
        self,
        image: np.ndarray,
        metadata: FaceMetadata | None = None,
    ) -> float:
        """Not used; custom sort implementation."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: ParallelSortProcessor | None = None,
    ) -> SortOutput:
        """Sort by identity dissimilarity."""
        start_time = time.time()
        valid_paths, embeddings, trash = _collect_embeddings(image_paths, processor)

        if not embeddings:
            return SortOutput([], trash, self.name, time.time() - start_time)

        matrix = np.stack(embeddings, axis=0)
        n = matrix.shape[0]

        if n <= self.exact_limit:
            dist = _cosine_distance_matrix(matrix)
            scores = dist.sum(axis=1)
        else:
            normed = _l2_normalize(matrix)
            centroid = np.mean(normed, axis=0, keepdims=True)
            centroid = _l2_normalize(centroid)
            scores = (
                1.0 - np.clip((normed @ centroid.T).reshape(-1), -1.0, 1.0)
            ).astype(np.float32)

        order = np.argsort(scores)[::-1].tolist()
        results = [SortResult(valid_paths[idx], float(scores[idx])) for idx in order]
        return SortOutput(results, trash, self.name, time.time() - start_time)
