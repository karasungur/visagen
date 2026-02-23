"""
Mask Canvas Component for Interactive Mask Editing.

Provides an interactive canvas for editing face segmentation masks
with component toggles, brush tools, and refinement controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import cv2
import gradio as gr
import numpy as np

from visagen.gui.components.mask_preview import (
    MaskPreviewGenerator,
    PreviewMode,
)
from visagen.gui.components.poly_editor import PolygonEditor
from visagen.gui.components.workflow_steps import (
    STEP_ORDER,
    WorkflowManager,
)
from visagen.vision.mask_ops import (
    DEFAULT_FACE_COMPONENTS,
    MaskOperations,
    MaskRefinementConfig,
)
from visagen.vision.polys import PolygonSet
from visagen.vision.segmenter import LABEL_TO_ID, FaceSegmenter

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


# Component groups for UI organization
COMPONENT_GROUPS = {
    "face": ["skin", "nose"],
    "eyes": ["left_eye", "right_eye", "left_brow", "right_brow"],
    "mouth": ["mouth", "upper_lip", "lower_lip"],
    "ears": ["left_ear", "right_ear"],
    "accessories": ["eye_glasses", "earring", "necklace"],
    "other": ["hair", "hat", "neck", "cloth"],
}

# Display names for components
COMPONENT_LABELS = {
    "skin": "Skin",
    "nose": "Nose",
    "left_eye": "L.Eye",
    "right_eye": "R.Eye",
    "left_brow": "L.Brow",
    "right_brow": "R.Brow",
    "left_ear": "L.Ear",
    "right_ear": "R.Ear",
    "mouth": "Mouth",
    "upper_lip": "U.Lip",
    "lower_lip": "L.Lip",
    "hair": "Hair",
    "hat": "Hat",
    "eye_glasses": "Glasses",
    "earring": "Earring",
    "necklace": "Necklace",
    "neck": "Neck",
    "cloth": "Cloth",
}


class NavigationImageCache:
    """
    LRU cache for fast prev/next navigation.

    Preloads images around current position for smoother navigation.
    Maintains a window of cached images: prev_count before and
    next_count after the current index.

    Args:
        prev_count: Number of previous images to cache. Default: 5.
        next_count: Number of next images to cache. Default: 5.
    """

    def __init__(self, prev_count: int = 5, next_count: int = 5) -> None:
        self.prev_count = prev_count
        self.next_count = next_count
        self._cache: dict[str, np.ndarray] = {}

    def update(self, current_idx: int, image_paths: list) -> None:
        """
        Update cache around current position.

        Loads images in the window [current - prev_count, current + next_count]
        and removes images outside this range.

        Args:
            current_idx: Current image index.
            image_paths: List of all image paths.
        """
        if not image_paths:
            return

        start = max(0, current_idx - self.prev_count)
        end = min(len(image_paths), current_idx + self.next_count + 1)
        keep_paths = {str(p) for p in image_paths[start:end]}

        # Remove entries outside the window
        for key in list(self._cache.keys()):
            if key not in keep_paths:
                del self._cache[key]

        # Preload new entries
        for path in image_paths[start:end]:
            key = str(path)
            if key not in self._cache:
                img = cv2.imread(key)
                if img is not None:
                    self._cache[key] = img

    def get(self, path) -> np.ndarray | None:
        """
        Get cached image.

        Args:
            path: Image path.

        Returns:
            Cached image or None if not in cache.
        """
        return self._cache.get(str(path))

    def clear(self) -> None:
        """Clear all cached images."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached images."""
        return len(self._cache)


class MaskHistory:
    """
    Undo/redo stack for mask edits.

    Maintains a history of mask states for undo/redo functionality.
    Limited to a maximum number of states to conserve memory.

    Args:
        max_size: Maximum number of history states. Default: 20.
    """

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._history: list[np.ndarray] = []
        self._position: int = -1

    def push(self, mask: np.ndarray) -> None:
        """
        Push a new mask state to history.

        Discards any redo states after current position.

        Args:
            mask: Mask to save (will be copied).
        """
        # Discard future states if we're not at the end
        if self._position < len(self._history) - 1:
            self._history = self._history[: self._position + 1]

        # Add new state
        self._history.append(mask.copy())
        self._position = len(self._history) - 1

        # Trim if exceeds max size
        if len(self._history) > self.max_size:
            self._history.pop(0)
            self._position -= 1

    def undo(self) -> np.ndarray | None:
        """
        Undo to previous state.

        Returns:
            Previous mask state, or None if at beginning.
        """
        if self._position > 0:
            self._position -= 1
            return cast(np.ndarray, self._history[self._position].copy())
        return None

    def redo(self) -> np.ndarray | None:
        """
        Redo to next state.

        Returns:
            Next mask state, or None if at end.
        """
        if self._position < len(self._history) - 1:
            self._position += 1
            return cast(np.ndarray, self._history[self._position].copy())
        return None

    def current(self) -> np.ndarray | None:
        """Get current mask state."""
        if 0 <= self._position < len(self._history):
            return cast(np.ndarray, self._history[self._position].copy())
        return None

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._position = -1

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._position > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._position < len(self._history) - 1


@dataclass
class MaskCanvasState:
    """
    Internal state for MaskCanvas.

    Attributes:
        current_image: Current face image being edited.
        current_mask: Current mask state.
        current_parsing: Face parsing result.
        selected_components: Set of enabled component names.
        history: Undo/redo history.
        preview_mode: Current preview visualization mode.
        polygons: Include/Exclude polygons for mask editing.
    """

    current_image: np.ndarray | None = None
    current_mask: np.ndarray | None = None
    current_parsing: np.ndarray | None = None
    selected_components: set[str] = field(
        default_factory=lambda: DEFAULT_FACE_COMPONENTS.copy()
    )
    history: MaskHistory = field(default_factory=MaskHistory)
    preview_mode: PreviewMode = PreviewMode.OVERLAY
    polygons: PolygonSet = field(default_factory=PolygonSet)


class MaskCanvas:
    """
    Interactive mask editing canvas component.

    Features:
    - Step-based workflow (Auto → Components → Polygons → Brush → Refine → Preview)
    - Component toggles (skin, eyes, nose, lips, hair, etc.)
    - Polygon-based include/exclude editing
    - Brush add/remove tools via gr.ImageEditor
    - Erode/Dilate/Blur refinement sliders
    - Undo/Redo functionality
    - Multiple preview modes

    Args:
        i18n: Internationalization instance.
        segmenter: Optional pre-loaded FaceSegmenter.

    Example:
        >>> canvas = MaskCanvas(i18n)
        >>> components = canvas.build()
        >>> canvas.setup_events(components)
    """

    def __init__(
        self,
        i18n: I18n,
        segmenter: FaceSegmenter | None = None,
    ) -> None:
        self.i18n = i18n
        self._segmenter = segmenter
        self._state = MaskCanvasState()
        self._preview_generator = MaskPreviewGenerator()
        self._workflow = WorkflowManager()
        self._poly_editor: PolygonEditor | None = None
        self._nav_cache = NavigationImageCache()

    def t(self, key: str) -> str:
        """Get translation."""
        return self.i18n.t(f"mask_editor.canvas.{key}")

    @property
    def segmenter(self) -> FaceSegmenter:
        """Get or create segmenter lazily."""
        if self._segmenter is None:
            self._segmenter = FaceSegmenter()
        return self._segmenter

    def build(self) -> dict[str, Any]:
        """
        Build canvas UI components.

        Returns:
            Dictionary mapping component names to Gradio components.
        """
        components: dict[str, Any] = {}

        # Workflow step navigation
        with gr.Row():
            components["prev_step_btn"] = gr.Button(
                "◀ Previous", size="sm", interactive=False
            )
            components["step_indicator"] = gr.Textbox(
                value="Step 1/6: Auto Segment",
                interactive=False,
                show_label=False,
                container=False,
            )
            components["next_step_btn"] = gr.Button(
                "Next ▶", size="sm", interactive=True
            )

        # === Step 1: Auto Segment ===
        with gr.Group(visible=True) as step1_group:
            components["step1_group"] = step1_group
            gr.Markdown("### Step 1: Automatic Segmentation")
            gr.Markdown("Load a face image to generate initial mask using SegFormer.")

        # === Step 2: Components ===
        with gr.Group(visible=False) as step2_group:
            components["step2_group"] = step2_group
            # Component toggles section
            with gr.Accordion(self.i18n.t("mask_editor.components.title"), open=True):
                # Face components row
                with gr.Row():
                    components["toggle_skin"] = gr.Checkbox(
                        label=COMPONENT_LABELS["skin"], value=True
                    )
                    components["toggle_nose"] = gr.Checkbox(
                        label=COMPONENT_LABELS["nose"], value=True
                    )
                    components["toggle_left_eye"] = gr.Checkbox(
                        label=COMPONENT_LABELS["left_eye"], value=True
                    )
                    components["toggle_right_eye"] = gr.Checkbox(
                        label=COMPONENT_LABELS["right_eye"], value=True
                    )

                # Brows and mouth
                with gr.Row():
                    components["toggle_left_brow"] = gr.Checkbox(
                        label=COMPONENT_LABELS["left_brow"], value=True
                    )
                    components["toggle_right_brow"] = gr.Checkbox(
                        label=COMPONENT_LABELS["right_brow"], value=True
                    )
                    components["toggle_mouth"] = gr.Checkbox(
                        label=COMPONENT_LABELS["mouth"], value=True
                    )

                # Lips
                with gr.Row():
                    components["toggle_upper_lip"] = gr.Checkbox(
                        label=COMPONENT_LABELS["upper_lip"], value=True
                    )
                    components["toggle_lower_lip"] = gr.Checkbox(
                        label=COMPONENT_LABELS["lower_lip"], value=True
                    )
                    components["toggle_hair"] = gr.Checkbox(
                        label=COMPONENT_LABELS["hair"], value=False
                    )

                # Ears and extras
                with gr.Row():
                    components["toggle_left_ear"] = gr.Checkbox(
                        label=COMPONENT_LABELS["left_ear"], value=False
                    )
                    components["toggle_right_ear"] = gr.Checkbox(
                        label=COMPONENT_LABELS["right_ear"], value=False
                    )
                    components["toggle_neck"] = gr.Checkbox(
                        label=COMPONENT_LABELS["neck"], value=False
                    )

                # Rebuild button
                components["rebuild_btn"] = gr.Button(
                    self.i18n.t("mask_editor.canvas.rebuild"),
                    size="sm",
                )

        # === Step 3: Polygons ===
        with gr.Group(visible=False) as step3_group:
            components["step3_group"] = step3_group
            gr.Markdown("### Step 3: Polygon Editing")
            gr.Markdown("Add Include (green) or Exclude (red) regions.")

            # Create polygon editor
            self._poly_editor = PolygonEditor(self.i18n)
            poly_components = self._poly_editor.build()
            for key, comp in poly_components.items():
                components[f"poly_{key}"] = comp

            # Apply polygons button
            components["apply_polys_btn"] = gr.Button(
                "Apply Polygons to Mask", size="sm"
            )

        # === Step 4: Brush ===
        with gr.Group(visible=False) as step4_group:
            components["step4_group"] = step4_group
            # Main canvas area
            with gr.Row():
                # Image editor for brush operations
                with gr.Column(scale=2):
                    components["editor"] = gr.ImageEditor(
                        label=self.i18n.t("mask_editor.canvas.label"),
                        type="numpy",
                        height=400,
                        brush=gr.Brush(
                            colors=["#FFFFFF", "#000000"],
                            default_color="#FFFFFF",
                            default_size=20,
                        ),
                        eraser=gr.Eraser(default_size=20),
                        layers=False,
                    )

                # Preview column
                with gr.Column(scale=1):
                    components["preview"] = gr.Image(
                        label=self.i18n.t("mask_editor.canvas.preview"),
                        height=200,
                        interactive=False,
                    )
                    components["mask_only"] = gr.Image(
                        label=self.i18n.t("mask_editor.canvas.mask_only"),
                        height=200,
                        interactive=False,
                    )

        # === Step 5: Refine ===
        with gr.Group(visible=False) as step5_group:
            components["step5_group"] = step5_group
            # Refinement controls
            with gr.Accordion(self.i18n.t("mask_editor.refine.title"), open=True):
                with gr.Row():
                    components["erode_slider"] = gr.Slider(
                        label=self.i18n.t("mask_editor.refine.erode"),
                        minimum=0,
                        maximum=20,
                        value=0,
                        step=1,
                    )
                    components["dilate_slider"] = gr.Slider(
                        label=self.i18n.t("mask_editor.refine.dilate"),
                        minimum=0,
                        maximum=20,
                        value=0,
                        step=1,
                    )
                    components["blur_slider"] = gr.Slider(
                        label=self.i18n.t("mask_editor.refine.blur"),
                        minimum=0,
                        maximum=50,
                        value=0,
                        step=2,
                    )

                with gr.Row():
                    components["apply_refine_btn"] = gr.Button(
                        self.i18n.t("common.apply"), size="sm"
                    )
                    components["reset_btn"] = gr.Button(
                        self.i18n.t("mask_editor.canvas.reset"), size="sm"
                    )

        # === Step 6: Preview ===
        with gr.Group(visible=False) as step6_group:
            components["step6_group"] = step6_group
            gr.Markdown("### Step 6: Preview & Save")

            # Preview mode selector
            with gr.Row():
                components["preview_mode"] = gr.Radio(
                    label="Preview Mode",
                    choices=[
                        "Original",
                        "Overlay",
                        "Masked",
                        "Mask Only",
                        "Inverted",
                        "Side by Side",
                    ],
                    value="Overlay",
                )

            # Final preview
            with gr.Row():
                components["final_preview"] = gr.Image(
                    label="Final Preview",
                    height=400,
                    interactive=False,
                )

        # Undo/Redo (always visible)
        with gr.Row():
            components["undo_btn"] = gr.Button("Undo", size="sm")
            components["redo_btn"] = gr.Button("Redo", size="sm")

        return components

    def setup_events(
        self,
        components: dict[str, Any],
        face_selector: Any | None = None,
    ) -> None:
        """
        Set up event handlers for canvas components.

        Args:
            components: Dictionary of Gradio components from build().
            face_selector: Optional face gallery/selector component.
        """
        # Step group references
        step_groups = [
            components.get("step1_group"),
            components.get("step2_group"),
            components.get("step3_group"),
            components.get("step4_group"),
            components.get("step5_group"),
            components.get("step6_group"),
        ]

        # Workflow navigation handlers
        def handle_next_step():
            self._workflow.state.next_step()
            visibility = self._workflow.get_step_visibility()
            buttons = self._workflow.get_button_states()
            indicator = self._workflow.state.get_step_indicator()

            updates = []
            for step in STEP_ORDER:
                updates.append(gr.update(visible=visibility.get(step, False)))

            return (
                indicator,
                gr.update(interactive=buttons["prev_enabled"]),
                gr.update(interactive=buttons["next_enabled"]),
                *updates,
            )

        def handle_prev_step():
            self._workflow.state.prev_step()
            visibility = self._workflow.get_step_visibility()
            buttons = self._workflow.get_button_states()
            indicator = self._workflow.state.get_step_indicator()

            updates = []
            for step in STEP_ORDER:
                updates.append(gr.update(visible=visibility.get(step, False)))

            return (
                indicator,
                gr.update(interactive=buttons["prev_enabled"]),
                gr.update(interactive=buttons["next_enabled"]),
                *updates,
            )

        # Connect navigation buttons
        components["next_step_btn"].click(
            fn=handle_next_step,
            outputs=[
                components["step_indicator"],
                components["prev_step_btn"],
                components["next_step_btn"],
                *step_groups,
            ],
        )

        components["prev_step_btn"].click(
            fn=handle_prev_step,
            outputs=[
                components["step_indicator"],
                components["prev_step_btn"],
                components["next_step_btn"],
                *step_groups,
            ],
        )

        # Get all toggle components
        toggle_keys = [k for k in components if k.startswith("toggle_")]
        toggles = [components[k] for k in toggle_keys]

        # Rebuild mask from components
        components["rebuild_btn"].click(
            fn=self._rebuild_from_components,
            inputs=toggles,
            outputs=[
                components["editor"],
                components["preview"],
                components["mask_only"],
            ],
        )

        # Apply refinement
        components["apply_refine_btn"].click(
            fn=self._apply_refinement,
            inputs=[
                components["editor"],
                components["erode_slider"],
                components["dilate_slider"],
                components["blur_slider"],
            ],
            outputs=[
                components["editor"],
                components["preview"],
                components["mask_only"],
            ],
        )

        # Reset to original parsing
        components["reset_btn"].click(
            fn=self._reset_mask,
            inputs=toggles,
            outputs=[
                components["editor"],
                components["preview"],
                components["mask_only"],
            ],
        )

        # Undo/Redo
        components["undo_btn"].click(
            fn=self._undo,
            outputs=[
                components["editor"],
                components["preview"],
                components["mask_only"],
            ],
        )

        components["redo_btn"].click(
            fn=self._redo,
            outputs=[
                components["editor"],
                components["preview"],
                components["mask_only"],
            ],
        )

        # Update preview on editor change
        components["editor"].change(
            fn=self._on_editor_change,
            inputs=[components["editor"]],
            outputs=[components["preview"], components["mask_only"]],
        )

        # Preview mode change
        components["preview_mode"].change(
            fn=self._on_preview_mode_change,
            inputs=[components["preview_mode"]],
            outputs=[components.get("final_preview") or components["preview"]],
        )

        # Polygon editor events
        if self._poly_editor is not None:
            # Get polygon editor components
            poly_components = {
                k.replace("poly_", ""): v
                for k, v in components.items()
                if k.startswith("poly_")
            }
            self._poly_editor.setup_events(poly_components)

        # Apply polygons to mask
        if "apply_polys_btn" in components:
            components["apply_polys_btn"].click(
                fn=self._apply_polygons,
                outputs=[
                    components["editor"],
                    components["preview"],
                    components["mask_only"],
                ],
            )

    def load_face(self, face_image: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Load a face image and generate initial mask.

        Args:
            face_image: Face image (BGR format).

        Returns:
            Tuple of (editor_data, preview_image, mask_only).
        """
        self._state.current_image = face_image.copy()

        # Get parsing (segment call not needed, just get_parsing)
        parsing_dict = self.segmenter.get_parsing(face_image)

        # Store parsing map
        h, w = face_image.shape[:2]
        parsing = np.zeros((h, w), dtype=np.uint8)
        for name, mask in parsing_dict.items():
            if name in LABEL_TO_ID:
                parsing[mask > 127] = LABEL_TO_ID[name]

        self._state.current_parsing = parsing

        # Generate initial mask from default components
        mask = MaskOperations.combine_component_masks(
            parsing, self._state.selected_components
        )
        self._state.current_mask = mask

        # Initialize history
        self._state.history.clear()
        self._state.history.push(mask)

        # Create outputs
        return self._create_outputs(face_image, mask)

    def _create_outputs(
        self,
        image: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Create editor data, preview, and mask-only outputs."""
        # Convert to RGB for display
        image_rgb: np.ndarray
        if image is None:
            image_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image_rgb = cast(np.ndarray, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if mask is None:
            mask = np.zeros((512, 512), dtype=np.uint8)

        # Create mask as RGB for editor
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Editor data
        editor_data = {
            "background": mask_rgb,
            "layers": [],
            "composite": mask_rgb,
        }

        # Preview with overlay (using current preview mode)
        preview = self._create_preview_with_mode(image_rgb, mask)

        return editor_data, preview, mask

    def _create_preview_with_mode(
        self,
        image: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """Create preview using current preview mode."""
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        if mask is None:
            return image

        return cast(
            np.ndarray,
            self._preview_generator.generate(image, mask, self._state.preview_mode),
        )

    def _create_preview_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create preview with mask overlay."""
        if image is None or mask is None:
            return image

        # Create colored overlay
        overlay = image.copy()
        mask_bool = mask > 127

        # Green tint for mask region
        overlay[mask_bool, 1] = np.clip(
            overlay[mask_bool, 1].astype(np.int32) + 80, 0, 255
        ).astype(np.uint8)

        # Blend
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Draw mask boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def _on_preview_mode_change(self, mode_name: str) -> np.ndarray | None:
        """Handle preview mode change."""
        # Map display name to PreviewMode
        mode_map = {
            "Original": PreviewMode.ORIGINAL,
            "Overlay": PreviewMode.OVERLAY,
            "Masked": PreviewMode.MASKED_RESULT,
            "Mask Only": PreviewMode.MASK_ONLY,
            "Inverted": PreviewMode.INVERTED,
            "Side by Side": PreviewMode.SIDE_BY_SIDE,
        }
        self._state.preview_mode = mode_map.get(mode_name, PreviewMode.OVERLAY)

        # Regenerate preview
        if (
            self._state.current_image is not None
            and self._state.current_mask is not None
        ):
            image_rgb = cv2.cvtColor(self._state.current_image, cv2.COLOR_BGR2RGB)
            return self._create_preview_with_mode(image_rgb, self._state.current_mask)

        return None

    def _get_selected_components(self, *toggle_values: bool) -> set[str]:
        """Get set of selected component names from toggle values."""
        component_names = [
            "skin",
            "nose",
            "left_eye",
            "right_eye",
            "left_brow",
            "right_brow",
            "mouth",
            "upper_lip",
            "lower_lip",
            "hair",
            "left_ear",
            "right_ear",
            "neck",
        ]

        selected = set()
        for name, value in zip(component_names, toggle_values, strict=False):
            if value:
                selected.add(name)

        return selected

    def _rebuild_from_components(
        self,
        *toggle_values: bool,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Rebuild mask from component toggles."""
        if self._state.current_parsing is None or self._state.current_image is None:
            return self._create_outputs(None, None)

        # Get selected components
        selected = self._get_selected_components(*toggle_values)
        self._state.selected_components = selected

        # Rebuild mask
        mask = MaskOperations.combine_component_masks(
            self._state.current_parsing, selected
        )
        self._state.current_mask = mask

        # Save to history
        self._state.history.push(mask)

        return self._create_outputs(self._state.current_image, mask)

    def _apply_refinement(
        self,
        editor_data: dict,
        erode: int,
        dilate: int,
        blur: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Apply refinement to current mask."""
        if editor_data is None:
            return self._create_outputs(None, None)

        # Get mask from editor
        mask = self._extract_mask_from_editor(editor_data)
        if mask is None:
            return self._create_outputs(self._state.current_image, None)

        # Apply refinement
        config = MaskRefinementConfig(
            erode_size=erode,
            dilate_size=dilate,
            blur_size=blur,
        )
        refined = MaskOperations.refine(mask, config)
        self._state.current_mask = refined

        # Save to history
        self._state.history.push(refined)

        return self._create_outputs(self._state.current_image, refined)

    def _reset_mask(
        self,
        *toggle_values: bool,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Reset mask to original parsing."""
        return self._rebuild_from_components(*toggle_values)

    def _undo(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """Undo last mask edit."""
        mask = self._state.history.undo()
        if mask is not None:
            self._state.current_mask = mask

        return self._create_outputs(self._state.current_image, self._state.current_mask)

    def _redo(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """Redo last undone edit."""
        mask = self._state.history.redo()
        if mask is not None:
            self._state.current_mask = mask

        return self._create_outputs(self._state.current_image, self._state.current_mask)

    def _on_editor_change(
        self,
        editor_data: dict,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Handle editor content change."""
        if editor_data is None:
            return None, None

        mask = self._extract_mask_from_editor(editor_data)
        if mask is None:
            return None, None

        self._state.current_mask = mask

        preview = None
        if self._state.current_image is not None:
            image_rgb = cv2.cvtColor(self._state.current_image, cv2.COLOR_BGR2RGB)
            preview = self._create_preview_with_mode(image_rgb, mask)

        return preview, mask

    def _extract_mask_from_editor(self, editor_data: dict) -> np.ndarray | None:
        """Extract mask from editor data."""
        if editor_data is None:
            return None

        # Get composite or background
        composite = editor_data.get("composite")
        if composite is None:
            composite = editor_data.get("background")

        if composite is None:
            return None

        # Convert to grayscale
        if len(composite.shape) == 3:
            mask = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
        else:
            mask = composite

        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return cast(np.ndarray, mask.astype(np.uint8))

    def get_current_mask(self) -> np.ndarray | None:
        """Get the current mask."""
        return (
            self._state.current_mask.copy()
            if self._state.current_mask is not None
            else None
        )

    def get_current_image(self) -> np.ndarray | None:
        """Get the current image."""
        return (
            self._state.current_image.copy()
            if self._state.current_image is not None
            else None
        )

    def _apply_polygons(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """Apply polygon edits to current mask."""
        if self._state.current_image is None or self._state.current_mask is None:
            return self._create_outputs(None, None)

        if self._poly_editor is None:
            return self._create_outputs(
                self._state.current_image, self._state.current_mask
            )

        # Get polygons from editor
        polys = self._poly_editor.get_polygons()
        self._state.polygons = polys

        # Apply polygons to current mask
        if polys.has_polys():
            modified_mask = polys.overlay_on_mask(self._state.current_mask)
            self._state.current_mask = modified_mask
            self._state.history.push(modified_mask)
        else:
            modified_mask = self._state.current_mask

        return self._create_outputs(self._state.current_image, modified_mask)

    def get_polygons(self) -> PolygonSet:
        """Get current polygon set."""
        return self._state.polygons

    def set_polygons(self, polys: PolygonSet) -> None:
        """Set polygon set."""
        self._state.polygons = polys
        if self._poly_editor is not None:
            self._poly_editor.load_polygons(polys)
