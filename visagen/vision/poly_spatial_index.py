"""Spatial indexing for fast polygon point/edge lookup."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from visagen.vision.polys import PolygonSet


class PolygonSpatialIndex:
    """
    KD-Tree based spatial index for O(log n) point lookup.

    Provides efficient nearest-point queries for polygon editing,
    replacing O(n×m) brute-force search with O(log n) lookups.

    Args:
        polys: PolygonSet to index.

    Example:
        >>> index = PolygonSpatialIndex(polys)
        >>> poly_idx, point_idx, dist = index.find_nearest_point(100, 200)
    """

    def __init__(self, polys: PolygonSet) -> None:
        self.polys = polys
        self._tree: KDTree | None = None
        self._metadata: list[tuple[int, int]] = []
        self._dirty = True

    def _rebuild(self) -> None:
        """Build KD-Tree from all polygon points."""
        all_points = []
        self._metadata = []

        for poly_idx, poly in enumerate(self.polys):
            for point_idx, pt in enumerate(poly.points):
                all_points.append(pt)
                self._metadata.append((poly_idx, point_idx))

        if all_points:
            self._tree = KDTree(np.array(all_points))
        else:
            self._tree = None

        self._dirty = False

    def invalidate(self) -> None:
        """Mark index as dirty (rebuild on next query)."""
        self._dirty = True

    def find_nearest_point(
        self, x: float, y: float, threshold: float = 15.0
    ) -> tuple[int, int, float]:
        """
        O(log n) nearest point search.

        Args:
            x: X coordinate to search from.
            y: Y coordinate to search from.
            threshold: Maximum distance to consider.

        Returns:
            Tuple of (polygon_idx, point_idx, distance).
            If no point found within threshold, returns (-1, -1, inf).
        """
        if self._dirty:
            self._rebuild()

        if self._tree is None or len(self._metadata) == 0:
            return (-1, -1, float("inf"))

        dist, idx = self._tree.query([x, y], k=1, distance_upper_bound=threshold)

        if dist == float("inf"):
            return (-1, -1, float("inf"))

        poly_idx, point_idx = self._metadata[idx]
        return (poly_idx, point_idx, float(dist))

    def find_k_nearest_points(
        self, x: float, y: float, k: int = 5, threshold: float = 50.0
    ) -> list[tuple[int, int, float]]:
        """
        Find k nearest points within threshold.

        Args:
            x: X coordinate to search from.
            y: Y coordinate to search from.
            k: Number of nearest points to return.
            threshold: Maximum distance to consider.

        Returns:
            List of (polygon_idx, point_idx, distance) tuples.
        """
        if self._dirty:
            self._rebuild()

        if self._tree is None or len(self._metadata) == 0:
            return []

        k = min(k, len(self._metadata))
        distances, indices = self._tree.query(
            [x, y], k=k, distance_upper_bound=threshold
        )

        # Handle single result case
        if k == 1:
            distances = [distances]
            indices = [indices]

        results = []
        for dist, idx in zip(distances, indices, strict=False):
            if dist != float("inf") and idx < len(self._metadata):
                poly_idx, point_idx = self._metadata[idx]
                results.append((poly_idx, point_idx, float(dist)))

        return results

    @property
    def point_count(self) -> int:
        """Get total number of indexed points."""
        if self._dirty:
            self._rebuild()
        return len(self._metadata)
