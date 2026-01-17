"""
Polygon Editor Component for Gradio.

Provides click-based polygon editing interface since
gr.ImageEditor doesn't support polygon drawing natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import cv2
import gradio as gr
import numpy as np

from visagen.vision.poly_render import (
    find_nearest_point,
    find_polygon_at_point,
    overlay_polygons_on_image,
)
from visagen.vision.polys import PolygonSet, PolyType

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


class PolygonEditorMode(Enum):
    """Editor interaction modes."""

    VIEW = "view"  # View only, no editing
    ADD_INCLUDE = "add_include"  # Adding INCLUDE polygon
    ADD_EXCLUDE = "add_exclude"  # Adding EXCLUDE polygon
    EDIT_POINTS = "edit_points"  # Editing existing points
    DELETE_POLY = "delete_poly"  # Deleting polygons


@dataclass
class PolygonEditorState:
    """
    Internal state for PolygonEditor.

    Attributes:
        polygons: Current polygon set.
        mode: Current editing mode.
        selected_poly_idx: Index of selected polygon (-1 for none).
        selected_point_idx: Index of selected point (-1 for none).
        current_image: Current base image.
    """

    polygons: PolygonSet = field(default_factory=PolygonSet)
    mode: PolygonEditorMode = PolygonEditorMode.VIEW
    selected_poly_idx: int = -1
    selected_point_idx: int = -1
    current_image: np.ndarray | None = None


class PolygonEditor:
    """
    Click-based polygon editor component.

    Provides an interactive interface for creating and editing
    Include/Exclude polygons. Uses click events on an image
    since Gradio doesn't support native polygon drawing.

    Usage:
        1. Select ADD_INCLUDE or ADD_EXCLUDE mode
        2. Click to add points to current polygon
        3. Click "Finish Polygon" to close the polygon
        4. Use EDIT_POINTS mode to move existing points
        5. Use DELETE_POLY mode to remove polygons

    Args:
        i18n: Internationalization instance.

    Example:
        >>> editor = PolygonEditor(i18n)
        >>> components = editor.build()
        >>> editor.setup_events(components)
    """

    def __init__(self, i18n: I18n) -> None:
        self.i18n = i18n
        self._state = PolygonEditorState()

    def t(self, key: str) -> str:
        """Get translation."""
        return self.i18n.t(f"mask_editor.polygon.{key}")

    def build(self) -> dict[str, Any]:
        """
        Build polygon editor UI components.

        Returns:
            Dictionary mapping component names to Gradio components.
        """
        components: dict[str, Any] = {}

        # Mode selection
        with gr.Row():
            components["mode_view"] = gr.Button("View", size="sm", variant="primary")
            components["mode_include"] = gr.Button("Add Include", size="sm")
            components["mode_exclude"] = gr.Button("Add Exclude", size="sm")
            components["mode_edit"] = gr.Button("Edit Points", size="sm")
            components["mode_delete"] = gr.Button("Delete", size="sm")

        # Canvas for polygon editing (clickable image)
        components["canvas"] = gr.Image(
            label=self.t("canvas"),
            type="numpy",
            height=400,
            interactive=True,
        )

        # Current mode indicator
        components["mode_indicator"] = gr.Textbox(
            label=self.t("current_mode"),
            value="View Mode",
            interactive=False,
        )

        # Polygon actions
        with gr.Row():
            components["finish_poly_btn"] = gr.Button(
                self.t("finish_polygon"), size="sm"
            )
            components["cancel_poly_btn"] = gr.Button(
                self.t("cancel_polygon"), size="sm"
            )
            components["clear_all_btn"] = gr.Button(self.t("clear_all"), size="sm")

        # Polygon list
        components["poly_list"] = gr.Dataframe(
            headers=["#", "Type", "Points"],
            label=self.t("polygon_list"),
            interactive=False,
        )

        # Hidden state
        components["poly_state"] = gr.State(value=None)

        return components

    def setup_events(
        self,
        components: dict[str, Any],
    ) -> None:
        """
        Set up event handlers for polygon editor.

        Args:
            components: Dictionary of Gradio components from build().
        """
        # Mode buttons
        components["mode_view"].click(
            fn=lambda: self._set_mode(PolygonEditorMode.VIEW),
            outputs=[components["mode_indicator"]],
        )
        components["mode_include"].click(
            fn=lambda: self._set_mode(PolygonEditorMode.ADD_INCLUDE),
            outputs=[components["mode_indicator"]],
        )
        components["mode_exclude"].click(
            fn=lambda: self._set_mode(PolygonEditorMode.ADD_EXCLUDE),
            outputs=[components["mode_indicator"]],
        )
        components["mode_edit"].click(
            fn=lambda: self._set_mode(PolygonEditorMode.EDIT_POINTS),
            outputs=[components["mode_indicator"]],
        )
        components["mode_delete"].click(
            fn=lambda: self._set_mode(PolygonEditorMode.DELETE_POLY),
            outputs=[components["mode_indicator"]],
        )

        # Canvas click
        components["canvas"].select(
            fn=self._handle_click,
            inputs=[components["canvas"]],
            outputs=[
                components["canvas"],
                components["poly_list"],
            ],
        )

        # Polygon actions
        components["finish_poly_btn"].click(
            fn=self._finish_polygon,
            outputs=[
                components["canvas"],
                components["poly_list"],
                components["mode_indicator"],
            ],
        )

        components["cancel_poly_btn"].click(
            fn=self._cancel_polygon,
            outputs=[
                components["canvas"],
                components["poly_list"],
            ],
        )

        components["clear_all_btn"].click(
            fn=self._clear_all,
            outputs=[
                components["canvas"],
                components["poly_list"],
            ],
        )

    def load_image(self, image: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Load image into editor.

        Args:
            image: Image to edit (BGR format).

        Returns:
            Tuple of (canvas_image, poly_list_data).
        """
        self._state.current_image = image.copy()
        self._state.polygons = PolygonSet()
        self._state.selected_poly_idx = -1
        self._state.selected_point_idx = -1
        self._state.mode = PolygonEditorMode.VIEW

        return self._render_canvas(), self._get_poly_list_data()

    def load_polygons(self, polys: PolygonSet) -> tuple[np.ndarray, list]:
        """
        Load existing polygons into editor.

        Args:
            polys: PolygonSet to load.

        Returns:
            Tuple of (canvas_image, poly_list_data).
        """
        self._state.polygons = polys
        return self._render_canvas(), self._get_poly_list_data()

    def get_polygons(self) -> PolygonSet:
        """Get current polygon set."""
        return self._state.polygons

    def _set_mode(self, mode: PolygonEditorMode) -> str:
        """Set editing mode and return indicator text."""
        self._state.mode = mode
        self._state.selected_poly_idx = -1
        self._state.selected_point_idx = -1

        mode_names = {
            PolygonEditorMode.VIEW: "View Mode",
            PolygonEditorMode.ADD_INCLUDE: "Add Include Polygon (click to add points)",
            PolygonEditorMode.ADD_EXCLUDE: "Add Exclude Polygon (click to add points)",
            PolygonEditorMode.EDIT_POINTS: "Edit Points (click to select, drag to move)",
            PolygonEditorMode.DELETE_POLY: "Delete Mode (click polygon to delete)",
        }
        return mode_names.get(mode, "Unknown Mode")

    def _handle_click(
        self,
        evt: gr.SelectData,
        image: np.ndarray | None,
    ) -> tuple[np.ndarray, list]:
        """Handle canvas click event."""
        if self._state.current_image is None:
            return image, self._get_poly_list_data()

        x, y = evt.index[0], evt.index[1]

        if self._state.mode == PolygonEditorMode.VIEW:
            # View mode: just select polygon
            poly_idx = find_polygon_at_point(self._state.polygons, x, y)
            self._state.selected_poly_idx = poly_idx

        elif self._state.mode in (
            PolygonEditorMode.ADD_INCLUDE,
            PolygonEditorMode.ADD_EXCLUDE,
        ):
            # Add point to current polygon
            self._add_point_to_current_polygon(x, y)

        elif self._state.mode == PolygonEditorMode.EDIT_POINTS:
            # Find and select nearest point
            poly_idx, point_idx, distance = find_nearest_point(
                self._state.polygons, x, y, threshold=20.0
            )
            if poly_idx >= 0:
                self._state.selected_poly_idx = poly_idx
                self._state.selected_point_idx = point_idx
            else:
                self._state.selected_poly_idx = -1
                self._state.selected_point_idx = -1

        elif self._state.mode == PolygonEditorMode.DELETE_POLY:
            # Delete polygon at click location
            poly_idx = find_polygon_at_point(self._state.polygons, x, y)
            if poly_idx >= 0:
                self._state.polygons.remove_polygon(poly_idx)
                self._state.selected_poly_idx = -1

        return self._render_canvas(), self._get_poly_list_data()

    def _add_point_to_current_polygon(self, x: float, y: float) -> None:
        """Add point to current polygon being drawn."""
        # Check if we have an incomplete polygon
        current_poly = None

        if len(self._state.polygons) > 0:
            last_poly = self._state.polygons[-1]
            if not last_poly.is_valid():
                current_poly = last_poly

        # Create new polygon if needed
        if current_poly is None:
            poly_type = (
                PolyType.INCLUDE
                if self._state.mode == PolygonEditorMode.ADD_INCLUDE
                else PolyType.EXCLUDE
            )
            current_poly = self._state.polygons.add_polygon(poly_type)
            self._state.selected_poly_idx = len(self._state.polygons) - 1

        current_poly.add_point(x, y)

    def _finish_polygon(self) -> tuple[np.ndarray, list, str]:
        """Finish current polygon."""
        if len(self._state.polygons) > 0:
            last_poly = self._state.polygons[-1]
            if not last_poly.is_valid():
                # Not enough points - remove incomplete polygon
                self._state.polygons.remove_polygon(len(self._state.polygons) - 1)

        self._state.selected_poly_idx = -1
        self._state.mode = PolygonEditorMode.VIEW

        return (
            self._render_canvas(),
            self._get_poly_list_data(),
            "View Mode",
        )

    def _cancel_polygon(self) -> tuple[np.ndarray, list]:
        """Cancel current polygon being drawn."""
        if len(self._state.polygons) > 0:
            last_poly = self._state.polygons[-1]
            if not last_poly.is_valid():
                self._state.polygons.remove_polygon(len(self._state.polygons) - 1)

        self._state.selected_poly_idx = -1

        return self._render_canvas(), self._get_poly_list_data()

    def _clear_all(self) -> tuple[np.ndarray, list]:
        """Clear all polygons."""
        self._state.polygons = PolygonSet()
        self._state.selected_poly_idx = -1
        self._state.selected_point_idx = -1

        return self._render_canvas(), self._get_poly_list_data()

    def _render_canvas(self) -> np.ndarray:
        """Render current state to canvas image."""
        if self._state.current_image is None:
            return np.zeros((400, 400, 3), dtype=np.uint8)

        # Convert BGR to RGB for display
        image = cv2.cvtColor(self._state.current_image, cv2.COLOR_BGR2RGB)

        # Overlay polygons
        if self._state.polygons.has_polys() or self._state.selected_poly_idx >= 0:
            image = overlay_polygons_on_image(
                image,
                self._state.polygons,
                selected_idx=self._state.selected_poly_idx,
                selected_point_idx=self._state.selected_point_idx,
            )

        return image

    def _get_poly_list_data(self) -> list[list]:
        """Get polygon list data for display."""
        data = []
        for i, poly in enumerate(self._state.polygons):
            type_str = "Include" if poly.type == PolyType.INCLUDE else "Exclude"
            data.append([i + 1, type_str, len(poly)])
        return data
