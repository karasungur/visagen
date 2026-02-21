"""
Polygon Rendering Utilities.

Provides visualization and interaction utilities for polygons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from visagen.vision.polys import Polygon, PolygonSet, PolyType

if TYPE_CHECKING:
    from visagen.vision.poly_spatial_index import PolygonSpatialIndex

# Default colors for polygon visualization (BGR format)
DEFAULT_INCLUDE_COLOR = (0, 255, 0)  # Green
DEFAULT_EXCLUDE_COLOR = (0, 0, 255)  # Red
DEFAULT_SELECTED_COLOR = (255, 255, 0)  # Cyan
DEFAULT_POINT_COLOR = (255, 255, 255)  # White
DEFAULT_ACTIVE_POINT_COLOR = (0, 255, 255)  # Yellow


def render_polygons_to_mask(
    polys: PolygonSet,
    height: int,
    width: int,
    initial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render polygons to binary mask.

    INCLUDE polygons are applied first (white), then
    EXCLUDE polygons are applied (black).

    Args:
        polys: PolygonSet to render.
        height: Output mask height.
        width: Output mask width.
        initial_mask: Optional starting mask. Default: all zeros.

    Returns:
        Binary mask (H, W) with values 0 or 255.
    """
    if initial_mask is not None:
        mask = initial_mask.copy()
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    return polys.overlay_on_mask(mask)


def overlay_polygons_on_image(
    image: np.ndarray,
    polys: PolygonSet,
    selected_idx: int = -1,
    selected_point_idx: int = -1,
    include_color: tuple[int, int, int] = DEFAULT_INCLUDE_COLOR,
    exclude_color: tuple[int, int, int] = DEFAULT_EXCLUDE_COLOR,
    selected_color: tuple[int, int, int] = DEFAULT_SELECTED_COLOR,
    point_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    active_point_color: tuple[int, int, int] = DEFAULT_ACTIVE_POINT_COLOR,
    line_thickness: int = 2,
    point_radius: int = 5,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Overlay polygon visualization on image.

    Args:
        image: Base image (BGR or RGB format).
        polys: PolygonSet to visualize.
        selected_idx: Index of selected polygon (-1 for none).
        selected_point_idx: Index of selected point in selected polygon.
        include_color: Color for INCLUDE polygons.
        exclude_color: Color for EXCLUDE polygons.
        selected_color: Color for selected polygon.
        point_color: Color for polygon points.
        active_point_color: Color for selected point.
        line_thickness: Line thickness for polygon edges.
        point_radius: Radius of point markers.
        alpha: Fill transparency (0-1).

    Returns:
        Image with polygon overlay.
    """
    result = image.copy()
    overlay = image.copy()

    for poly_idx, poly in enumerate(polys):
        if not poly.is_valid():
            # Draw incomplete polygon as lines only
            if len(poly) >= 2:
                for i in range(len(poly) - 1):
                    pt1 = tuple(poly.points[i].astype(int))
                    pt2 = tuple(poly.points[i + 1].astype(int))
                    color = (
                        selected_color
                        if poly_idx == selected_idx
                        else (
                            include_color
                            if poly.type == PolyType.INCLUDE
                            else exclude_color
                        )
                    )
                    cv2.line(overlay, pt1, pt2, color, line_thickness)
            # Draw points
            for point_idx, pt in enumerate(poly.points):
                pt_int = tuple(pt.astype(int))
                if poly_idx == selected_idx and point_idx == selected_point_idx:
                    cv2.circle(
                        overlay, pt_int, point_radius + 2, active_point_color, -1
                    )
                else:
                    cv2.circle(overlay, pt_int, point_radius, point_color, -1)
            continue

        pts = poly.points.astype(np.int32)

        # Determine color
        if poly_idx == selected_idx:
            color = selected_color
        elif poly.type == PolyType.INCLUDE:
            color = include_color
        else:
            color = exclude_color

        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], color)

        # Draw outline
        cv2.polylines(
            result, [pts], isClosed=True, color=color, thickness=line_thickness
        )

        # Draw points
        for point_idx, pt in enumerate(poly.points):
            pt_int = tuple(pt.astype(int))
            if poly_idx == selected_idx and point_idx == selected_point_idx:
                cv2.circle(result, pt_int, point_radius + 2, active_point_color, -1)
            else:
                cv2.circle(result, pt_int, point_radius, point_color, -1)

    # Blend overlay
    result = cast(np.ndarray, cv2.addWeighted(result, 1.0, overlay, alpha, 0))

    return result


def find_nearest_point(
    polys: PolygonSet,
    x: int,
    y: int,
    threshold: float = 15.0,
    spatial_index: PolygonSpatialIndex | None = None,
) -> tuple[int, int, float]:
    """
    Find the nearest point in any polygon to given coordinates.

    Args:
        polys: PolygonSet to search.
        x: X coordinate to search from.
        y: Y coordinate to search from.
        threshold: Maximum distance to consider.
        spatial_index: Optional spatial index for O(log n) lookup.

    Returns:
        Tuple of (polygon_idx, point_idx, distance).
        If no point found within threshold, returns (-1, -1, inf).
    """
    # Use spatial index if available (O(log n))
    if spatial_index is not None:
        return spatial_index.find_nearest_point(x, y, threshold)

    # Fallback: brute force (O(n×m))
    target = np.array([x, y], dtype=np.float32)
    best_poly_idx = -1
    best_point_idx = -1
    best_distance = float("inf")

    for poly_idx, poly in enumerate(polys):
        for point_idx, pt in enumerate(poly.points):
            distance = np.linalg.norm(pt - target)
            if distance < best_distance and distance <= threshold:
                best_poly_idx = poly_idx
                best_point_idx = point_idx
                best_distance = float(distance)

    return (best_poly_idx, best_point_idx, best_distance)


def find_nearest_edge(
    polys: PolygonSet,
    x: int,
    y: int,
    threshold: float = 10.0,
) -> tuple[int, int, float, tuple[float, float]]:
    """
    Find the nearest edge in any polygon to given coordinates.

    Useful for inserting points along polygon edges.

    Args:
        polys: PolygonSet to search.
        x: X coordinate to search from.
        y: Y coordinate to search from.
        threshold: Maximum distance to consider.

    Returns:
        Tuple of (polygon_idx, edge_start_idx, distance, nearest_point).
        If no edge found within threshold, returns (-1, -1, inf, (0, 0)).
    """
    target = np.array([x, y], dtype=np.float32)
    best_poly_idx = -1
    best_edge_idx = -1
    best_distance = float("inf")
    best_point = (0.0, 0.0)

    for poly_idx, poly in enumerate(polys):
        if not poly.is_valid():
            continue

        n = len(poly.points)
        for i in range(n):
            p1 = poly.points[i]
            p2 = poly.points[(i + 1) % n]

            # Project point onto line segment
            edge = p2 - p1
            edge_len_sq = np.dot(edge, edge)

            if edge_len_sq < 1e-6:
                continue

            t = np.clip(np.dot(target - p1, edge) / edge_len_sq, 0, 1)
            projection = p1 + t * edge
            distance = np.linalg.norm(target - projection)

            if distance < best_distance and distance <= threshold:
                best_poly_idx = poly_idx
                best_edge_idx = i
                best_distance = float(distance)
                best_point = (float(projection[0]), float(projection[1]))

    return (best_poly_idx, best_edge_idx, best_distance, best_point)


def point_in_polygon(poly: Polygon, x: float, y: float) -> bool:
    """
    Check if point is inside polygon.

    Uses OpenCV's pointPolygonTest.

    Args:
        poly: Polygon to test.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        True if point is inside polygon.
    """
    if not poly.is_valid():
        return False

    pts = poly.points.astype(np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(pts, (x, y), measureDist=False)
    return result >= 0


def find_polygon_at_point(
    polys: PolygonSet,
    x: float,
    y: float,
) -> int:
    """
    Find polygon containing the given point.

    If multiple polygons contain the point, returns the last one
    (topmost in rendering order).

    Args:
        polys: PolygonSet to search.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        Index of polygon containing point, or -1 if none.
    """
    result_idx = -1

    for poly_idx, poly in enumerate(polys):
        if point_in_polygon(poly, x, y):
            result_idx = poly_idx

    return result_idx


def create_preview_with_mask_overlay(
    image: np.ndarray,
    polys: PolygonSet,
    base_mask: np.ndarray | None = None,
    overlay_alpha: float = 0.4,
    mask_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Create preview showing mask effect of polygons.

    Args:
        image: Base image.
        polys: Polygons to apply.
        base_mask: Optional base mask to modify.
        overlay_alpha: Mask overlay transparency.
        mask_color: Color for mask overlay.

    Returns:
        Preview image with mask overlay.
    """
    h, w = image.shape[:2]

    # Generate mask from polygons
    if base_mask is not None:
        mask = polys.overlay_on_mask(base_mask)
    else:
        mask = polys.render_to_mask(h, w)

    # Create colored overlay
    result = image.copy()
    overlay = np.zeros_like(image)
    overlay[mask > 127] = mask_color

    result = cv2.addWeighted(result, 1.0, overlay, overlay_alpha, 0)

    # Draw mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, mask_color, 2)

    return result
