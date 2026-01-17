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


@dataclass
class Polygon:
    """
    Single polygon with type and points.

    Represents a closed polygon that either adds to or
    subtracts from a mask.

    Attributes:
        type: INCLUDE to add to mask, EXCLUDE to subtract.
        points: Array of (x, y) coordinates, shape (N, 2).
    """

    type: PolyType
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        """Ensure points array is proper type."""
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points, dtype=np.float32)
        if self.points.dtype != np.float32:
            self.points = self.points.astype(np.float32)
        if len(self.points.shape) == 1:
            self.points = self.points.reshape(-1, 2)

    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the polygon.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        new_point = np.array([[x, y]], dtype=np.float32)
        self.points = np.concatenate([self.points, new_point], axis=0)

    def insert_point(self, idx: int, x: float, y: float) -> None:
        """
        Insert a point at specific index.

        Args:
            idx: Index to insert at.
            x: X coordinate.
            y: Y coordinate.
        """
        if idx < 0 or idx > len(self.points):
            raise IndexError(
                f"Index {idx} out of range for polygon with {len(self.points)} points"
            )
        new_point = np.array([[x, y]], dtype=np.float32)
        self.points = np.concatenate(
            [self.points[:idx], new_point, self.points[idx:]], axis=0
        )

    def remove_point(self, idx: int) -> None:
        """
        Remove point at index.

        Args:
            idx: Index of point to remove.
        """
        if idx < 0 or idx >= len(self.points):
            raise IndexError(
                f"Index {idx} out of range for polygon with {len(self.points)} points"
            )
        self.points = np.concatenate(
            [self.points[:idx], self.points[idx + 1 :]], axis=0
        )

    def set_point(self, idx: int, x: float, y: float) -> None:
        """
        Update point coordinates.

        Args:
            idx: Index of point to update.
            x: New X coordinate.
            y: New Y coordinate.
        """
        if idx < 0 or idx >= len(self.points):
            raise IndexError(f"Index {idx} out of range")
        self.points[idx] = [x, y]

    def scale(self, factor: float) -> "Polygon":
        """
        Scale polygon points by factor.

        Args:
            factor: Scale factor.

        Returns:
            New scaled Polygon instance.
        """
        return Polygon(
            type=self.type,
            points=self.points * factor,
        )

    def translate(self, dx: float, dy: float) -> "Polygon":
        """
        Translate polygon points.

        Args:
            dx: X translation.
            dy: Y translation.

        Returns:
            New translated Polygon instance.
        """
        return Polygon(
            type=self.type,
            points=self.points + np.array([dx, dy], dtype=np.float32),
        )

    def get_centroid(self) -> tuple[float, float]:
        """Get polygon centroid."""
        if len(self.points) == 0:
            return (0.0, 0.0)
        return (float(self.points[:, 0].mean()), float(self.points[:, 1].mean()))

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Get bounding box of polygon.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max).
        """
        if len(self.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )

    def is_valid(self) -> bool:
        """Check if polygon has enough points to be rendered."""
        return len(self.points) >= 3

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
        return cls(
            type=PolyType(data["type"]),
            points=np.array(data.get("pts", []), dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return number of points."""
        return len(self.points)


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
