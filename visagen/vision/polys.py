"""
Polygon Data Structures for Mask Editing.

Provides Include/Exclude polygon system compatible with
legacy XSegEditor format.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import cv2
import numpy as np


class PolyType(IntEnum):
    """Polygon type for mask modification."""

    EXCLUDE = 0  # Subtract from mask (paint black)
    INCLUDE = 1  # Add to mask (paint white)


class Polygon:
    """
    Single polygon with type and points.

    Represents a closed polygon that either adds to or
    subtracts from a mask. Supports undo/redo for point editing.

    Attributes:
        type: INCLUDE to add to mask, EXCLUDE to subtract.
        points: Array of (x, y) coordinates, shape (N, 2).
    """

    def __init__(self, poly_type: PolyType = PolyType.INCLUDE) -> None:
        """
        Initialize polygon with optional type.

        Args:
            poly_type: Type of polygon (INCLUDE or EXCLUDE).
        """
        self.type = poly_type
        self._points = np.empty((0, 2), dtype=np.float32)
        self._n = 0  # Current position (undo pointer)

    @property
    def points(self) -> np.ndarray:
        """Get current valid points."""
        return self._points[: self._n].copy()

    @points.setter
    def points(self, value: np.ndarray) -> None:
        """Set points directly (for compatibility)."""
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        if value.dtype != np.float32:
            value = value.astype(np.float32)
        if len(value.shape) == 1 and len(value) > 0:
            value = value.reshape(-1, 2)
        self._points = value
        self._n = len(value)

    @property
    def n(self) -> int:
        """Current point count."""
        return self._n

    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the polygon with undo support.

        Truncates any redo history when adding new point.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        # Truncate redo history and add new point
        self._points = np.append(
            self._points[: self._n], [[float(x), float(y)]], axis=0
        ).astype(np.float32)
        self._n += 1

    def undo(self) -> bool:
        """
        Undo last point addition.

        Returns:
            True if undo was successful, False if no points to undo.
        """
        if self._n > 0:
            self._n -= 1
            return True
        return False

    def redo(self) -> bool:
        """
        Redo undone point addition.

        Returns:
            True if redo was successful, False if no points to redo.
        """
        if self._n < len(self._points):
            self._n += 1
            return True
        return False

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._n > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._n < len(self._points)

    def insert_point(self, idx: int, x: float, y: float) -> None:
        """
        Insert a point at specific index.

        Clears redo history after insertion.

        Args:
            idx: Index to insert at.
            x: X coordinate.
            y: Y coordinate.
        """
        if idx < 0 or idx > self._n:
            raise IndexError(
                f"Index {idx} out of range for polygon with {self._n} points"
            )
        new_point = np.array([[x, y]], dtype=np.float32)
        self._points = np.concatenate(
            [self._points[:idx], new_point, self._points[idx : self._n]], axis=0
        )
        self._n += 1

    def remove_point(self, idx: int) -> None:
        """
        Remove point at index.

        Args:
            idx: Index of point to remove.
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(
                f"Index {idx} out of range for polygon with {self._n} points"
            )
        self._points = np.concatenate(
            [self._points[:idx], self._points[idx + 1 : self._n]], axis=0
        )
        self._n -= 1

    def set_point(self, idx: int, x: float, y: float) -> None:
        """
        Update point coordinates.

        Args:
            idx: Index of point to update.
            x: New X coordinate.
            y: New Y coordinate.
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range")
        self._points[idx] = [x, y]

    def scale(self, factor: float) -> "Polygon":
        """
        Scale polygon points by factor.

        Args:
            factor: Scale factor.

        Returns:
            New scaled Polygon instance.
        """
        new_poly = Polygon(self.type)
        new_poly.points = self.points * factor
        return new_poly

    def translate(self, dx: float, dy: float) -> "Polygon":
        """
        Translate polygon points.

        Args:
            dx: X translation.
            dy: Y translation.

        Returns:
            New translated Polygon instance.
        """
        new_poly = Polygon(self.type)
        new_poly.points = self.points + np.array([dx, dy], dtype=np.float32)
        return new_poly

    def get_centroid(self) -> tuple[float, float]:
        """Get polygon centroid."""
        if self._n == 0:
            return (0.0, 0.0)
        pts = self._points[: self._n]
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Get bounding box of polygon.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max).
        """
        if self._n == 0:
            return (0.0, 0.0, 0.0, 0.0)
        pts = self._points[: self._n]
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    def is_valid(self) -> bool:
        """Check if polygon has enough points to be rendered."""
        return self._n >= 3

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Compatible with legacy DFL format.
        """
        return {
            "type": int(self.type),
            "pts": self.points.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Polygon":
        """
        Create from dictionary.

        Compatible with legacy DFL format.
        """
        poly = cls(poly_type=PolyType(data["type"]))
        poly.points = np.array(data.get("pts", []), dtype=np.float32)
        return poly

    def __len__(self) -> int:
        """Return number of points."""
        return self._n


@dataclass
class PolygonSet:
    """
    Collection of Include/Exclude polygons.

    Manages multiple polygons for mask modification.
    When rendering, INCLUDE polygons are drawn first (white),
    then EXCLUDE polygons are drawn on top (black).

    Attributes:
        polygons: List of Polygon objects.
    """

    polygons: list[Polygon] = field(default_factory=list)

    def add_polygon(self, poly_type: PolyType) -> Polygon:
        """
        Add new empty polygon.

        Args:
            poly_type: Type of polygon to create.

        Returns:
            The newly created Polygon.
        """
        poly = Polygon(type=poly_type)
        self.polygons.append(poly)
        return poly

    def remove_polygon(self, idx: int) -> None:
        """
        Remove polygon at index.

        Args:
            idx: Index of polygon to remove.
        """
        if 0 <= idx < len(self.polygons):
            self.polygons.pop(idx)

    def get_polygon(self, idx: int) -> Polygon | None:
        """Get polygon at index."""
        if 0 <= idx < len(self.polygons):
            return self.polygons[idx]
        return None

    def has_polys(self) -> bool:
        """Check if there are any polygons."""
        return len(self.polygons) > 0

    def get_total_points(self) -> int:
        """Get total number of points across all polygons."""
        return sum(len(poly) for poly in self.polygons)

    def sort(self) -> None:
        """
        Sort polygons so INCLUDE come before EXCLUDE.

        This ensures proper rendering order.
        """
        include_polys = [p for p in self.polygons if p.type == PolyType.INCLUDE]
        exclude_polys = [p for p in self.polygons if p.type == PolyType.EXCLUDE]
        self.polygons = include_polys + exclude_polys

    def render_to_mask(self, height: int, width: int) -> np.ndarray:
        """
        Render polygons to binary mask.

        INCLUDE polygons are drawn first (white), then
        EXCLUDE polygons are drawn (black).

        Args:
            height: Mask height.
            width: Mask width.

        Returns:
            Binary mask (H, W) with values 0 or 255.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # Sort to ensure correct order
        sorted_polys = sorted(
            self.polygons,
            key=lambda p: (0 if p.type == PolyType.INCLUDE else 1),
        )

        for poly in sorted_polys:
            if not poly.is_valid():
                continue

            pts = poly.points.astype(np.int32)
            color = 255 if poly.type == PolyType.INCLUDE else 0
            cv2.fillPoly(mask, [pts], color)

        return mask

    def overlay_on_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply polygons to existing mask.

        Args:
            mask: Existing mask to modify.

        Returns:
            Modified mask.
        """
        result = mask.copy()

        sorted_polys = sorted(
            self.polygons,
            key=lambda p: (0 if p.type == PolyType.INCLUDE else 1),
        )

        for poly in sorted_polys:
            if not poly.is_valid():
                continue

            pts = poly.points.astype(np.int32)
            color = 255 if poly.type == PolyType.INCLUDE else 0
            cv2.fillPoly(result, [pts], color)

        return result

    def scale(self, factor: float) -> "PolygonSet":
        """
        Scale all polygons by factor.

        Args:
            factor: Scale factor.

        Returns:
            New PolygonSet with scaled polygons.
        """
        return PolygonSet(polygons=[poly.scale(factor) for poly in self.polygons])

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Compatible with legacy DFL seg_ie_polys format.
        """
        return {"polys": [poly.to_dict() for poly in self.polygons]}

    @classmethod
    def from_dict(cls, data: dict | list | None) -> "PolygonSet":
        """
        Create from dictionary.

        Handles both new dict format and legacy list format.

        Args:
            data: Serialized polygon data.

        Returns:
            PolygonSet instance.
        """
        polyset = cls()

        if data is None:
            return polyset

        if isinstance(data, list):
            # Legacy format: list of (type, pts) tuples
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    poly_type, pts = item[0], item[1]
                    polyset.polygons.append(
                        Polygon(
                            type=PolyType(poly_type),
                            points=np.array(pts, dtype=np.float32),
                        )
                    )
        elif isinstance(data, dict):
            # New format: {"polys": [...]}
            for poly_data in data.get("polys", []):
                polyset.polygons.append(Polygon.from_dict(poly_data))

        polyset.sort()
        return polyset

    def __iter__(self):
        """Iterate over polygons."""
        return iter(self.polygons)

    def __len__(self) -> int:
        """Return number of polygons."""
        return len(self.polygons)

    def __getitem__(self, idx: int) -> Polygon:
        """Get polygon by index."""
        return self.polygons[idx]
