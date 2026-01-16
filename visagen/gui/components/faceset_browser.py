"""Faceset Browser component for browsing extracted faces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import gradio as gr
import numpy as np

from visagen.data.face_sample import FaceSample
from visagen.gui.components.base import ComponentConfig
from visagen.gui.components.displays import create_mask_overlay

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


@dataclass
class FacesetBrowserConfig(ComponentConfig):
    """Configuration for faceset browser."""

    columns: int = 6
    rows: int = 3
    page_size: int = 18
    height: int = 400


class FacesetBrowser:
    """
    Browse extracted faces with masks inside an Accordion.

    Features:
    - Paginated gallery view
    - Toggle mask overlay
    - Face metadata viewer
    """

    def __init__(
        self,
        config: FacesetBrowserConfig,
        i18n: I18n,
    ) -> None:
        self.config = config
        self.i18n = i18n
        self._face_files: list[Path] = []
        self._current_page: int = 0

    def t(self, key: str) -> str:
        """Get translation for browser key."""
        return self.i18n.t(f"faceset_browser.{key}")

    def build(self) -> dict[str, Any]:
        """Build browser UI components."""
        components: dict[str, Any] = {}

        with gr.Accordion(self.t("title"), open=False) as accordion:
            components["accordion"] = accordion

            # Controls row
            with gr.Row():
                components["dir_input"] = gr.Textbox(
                    label=self.t("directory"),
                    placeholder="./workspace/extracted",
                    scale=3,
                )
                components["load_btn"] = gr.Button(
                    self.t("load"),
                    size="sm",
                    scale=1,
                )
                components["refresh_btn"] = gr.Button(
                    self.t("refresh"),
                    size="sm",
                    scale=1,
                )

            # View options
            with gr.Row():
                components["show_masks"] = gr.Checkbox(
                    label=self.t("show_masks"),
                    value=True,
                )
                components["sort_by"] = gr.Dropdown(
                    label=self.t("sort_by"),
                    choices=["name", "date"],
                    value="name",
                )

            # Gallery
            components["gallery"] = gr.Gallery(
                label=self.t("faces"),
                columns=self.config.columns,
                rows=self.config.rows,
                height=self.config.height,
                allow_preview=True,
                show_share_button=False,
            )

            # Pagination
            with gr.Row():
                components["prev_btn"] = gr.Button("<< Prev", size="sm")
                components["page_info"] = gr.Textbox(
                    value="Page 0/0",
                    interactive=False,
                    show_label=False,
                )
                components["next_btn"] = gr.Button("Next >>", size="sm")

            # Selected face info
            with gr.Row():
                components["selected_face"] = gr.Image(
                    label=self.t("selected"),
                    height=200,
                    interactive=False,
                )
                components["face_metadata"] = gr.JSON(
                    label=self.t("metadata"),
                )

        return components

    def setup_events(self, c: dict[str, Any]) -> None:
        """Wire up event handlers."""
        c["load_btn"].click(
            fn=self._load_directory,
            inputs=[c["dir_input"], c["show_masks"], c["sort_by"]],
            outputs=[c["gallery"], c["page_info"]],
        )

        c["refresh_btn"].click(
            fn=self._load_directory,
            inputs=[c["dir_input"], c["show_masks"], c["sort_by"]],
            outputs=[c["gallery"], c["page_info"]],
        )

        c["prev_btn"].click(
            fn=self._prev_page,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"]],
        )

        c["next_btn"].click(
            fn=self._next_page,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"]],
        )

        c["gallery"].select(
            fn=self._on_select,
            outputs=[c["selected_face"], c["face_metadata"]],
        )

    def _load_directory(
        self,
        directory: str,
        show_masks: bool,
        sort_by: str,
    ) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Load face images from directory."""
        if not directory:
            return [], self.t("no_directory") if hasattr(
                self, "_t_cache"
            ) else "No directory"

        dir_path = Path(directory)
        if not dir_path.exists():
            return [], self.t("not_found") if hasattr(self, "_t_cache") else "Not found"

        # Find DFL images
        self._face_files = list(dir_path.glob("*.jpg"))

        if sort_by == "date":
            self._face_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            self._face_files.sort(key=lambda p: p.name)

        self._current_page = 0
        return self._get_current_page(show_masks)

    def _get_current_page(
        self, show_masks: bool
    ) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Get current page of faces."""
        page_size = self.config.page_size
        start = self._current_page * page_size
        end = start + page_size

        page_files = self._face_files[start:end]
        gallery_items: list[tuple[np.ndarray, str]] = []

        for filepath in page_files:
            try:
                image = cv2.imread(str(filepath))
                if image is None:
                    continue

                if show_masks:
                    sample = FaceSample.from_dfl_image(filepath)
                    if sample and sample.xseg_mask:
                        mask = sample.get_xseg_mask()
                        if mask is not None:
                            mask_uint8 = (mask[:, :, 0] * 255).astype(np.uint8)
                            image = create_mask_overlay(image, mask_uint8, alpha=0.3)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gallery_items.append((image_rgb, filepath.stem))

            except Exception:
                continue

        total_pages = max(1, (len(self._face_files) + page_size - 1) // page_size)
        page_info = f"Page {self._current_page + 1}/{total_pages} ({len(self._face_files)} faces)"

        return gallery_items, page_info

    def _prev_page(self, show_masks: bool) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Go to previous page."""
        if self._current_page > 0:
            self._current_page -= 1
        return self._get_current_page(show_masks)

    def _next_page(self, show_masks: bool) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Go to next page."""
        total_pages = (
            len(self._face_files) + self.config.page_size - 1
        ) // self.config.page_size
        if self._current_page < total_pages - 1:
            self._current_page += 1
        return self._get_current_page(show_masks)

    def _on_select(
        self, evt: gr.SelectData
    ) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        """Handle gallery selection."""
        if evt.index is None or evt.index >= len(self._face_files):
            return None, None

        page_start = self._current_page * self.config.page_size
        file_idx = page_start + evt.index

        if file_idx >= len(self._face_files):
            return None, None

        filepath = self._face_files[file_idx]

        try:
            image = cv2.imread(str(filepath))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            sample = FaceSample.from_dfl_image(filepath)
            metadata: dict[str, Any] = {
                "filename": filepath.name,
                "face_type": sample.face_type if sample else "unknown",
                "has_mask": bool(sample and sample.xseg_mask),
            }

            return image_rgb, metadata

        except Exception:
            return None, None
