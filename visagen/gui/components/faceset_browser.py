"""Faceset Browser component for browsing extracted faces."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import gradio as gr
import numpy as np

from visagen.data.face_sample import FaceSample
from visagen.gui.components.base import ComponentConfig
from visagen.gui.components.displays import create_mask_overlay
from visagen.tools.dataset_trash import move_to_trash, undo_last_batch

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
        self._current_dir: Path | None = None
        self._selected_files: set[Path] = set()
        self._sort_by: str = "name"
        self._thumb_size: int = 256
        self._selected_preview_max_size: int = 1024
        self._page_size: int = config.page_size
        self._thumb_workers: int = 4
        self._cache_warm_budget: int = 8
        self._cache_warm_thread: threading.Thread | None = None

    def t(self, key: str, **kwargs: Any) -> str:
        """Get translation for browser key."""
        return cast(str, self.i18n.t(f"faceset_browser.{key}", **kwargs))

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
                components["page_size"] = gr.Dropdown(
                    label=self.t("page_size"),
                    choices=[12, 18, 24, 36],
                    value=self._page_size,
                )

            # Gallery
            components["gallery"] = gr.Gallery(
                label=self.t("faces"),
                columns=self.config.columns,
                rows=self.config.rows,
                height=self.config.height,
                allow_preview=True,
            )

            # Pagination
            with gr.Row():
                components["prev_btn"] = gr.Button(self.t("prev"), size="sm")
                components["page_info"] = gr.Textbox(
                    value="Page 0/0",
                    interactive=False,
                    show_label=False,
                )
                components["next_btn"] = gr.Button(self.t("next"), size="sm")

            with gr.Row():
                components["delete_selected_btn"] = gr.Button(
                    self.t("delete_selected"), size="sm"
                )
                components["clear_selection_btn"] = gr.Button(
                    self.t("clear_selection"), size="sm"
                )
                components["undo_delete_btn"] = gr.Button(
                    self.t("undo_last_delete"), size="sm"
                )

            components["status"] = gr.Textbox(
                value="",
                interactive=False,
                show_label=False,
            )

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
            inputs=[c["dir_input"], c["show_masks"], c["sort_by"], c["page_size"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["refresh_btn"].click(
            fn=self._load_directory,
            inputs=[c["dir_input"], c["show_masks"], c["sort_by"], c["page_size"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["page_size"].change(
            fn=self._set_page_size,
            inputs=[c["page_size"], c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["prev_btn"].click(
            fn=self._prev_page,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["next_btn"].click(
            fn=self._next_page,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["gallery"].select(
            fn=self._on_select,
            inputs=[c["show_masks"]],
            outputs=[
                c["selected_face"],
                c["face_metadata"],
                c["gallery"],
                c["page_info"],
                c["status"],
            ],
        )

        c["delete_selected_btn"].click(
            fn=self._delete_selected,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["clear_selection_btn"].click(
            fn=self._clear_selection,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

        c["undo_delete_btn"].click(
            fn=self._undo_last_delete,
            inputs=[c["show_masks"]],
            outputs=[c["gallery"], c["page_info"], c["status"]],
        )

    def _load_directory(
        self,
        directory: str,
        show_masks: bool,
        sort_by: str,
        page_size: int,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Load face images from directory."""
        if not directory:
            return [], "Page 0/0", self.t("no_directory")

        dir_path = Path(directory)
        if not dir_path.exists():
            return [], "Page 0/0", self.t("not_found")

        self._current_dir = dir_path
        self._sort_by = sort_by
        self._page_size = int(page_size)
        self._selected_files.clear()
        self._refresh_face_file_list()
        self._current_page = 0
        gallery, page_info, _ = self._get_current_page(show_masks)
        return gallery, page_info, self.t("status_loaded", count=len(self._face_files))

    def _set_page_size(
        self,
        page_size: int,
        show_masks: bool,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Set page size and refresh current view."""
        self._page_size = int(page_size)
        self._current_page = 0
        return self._get_current_page(show_masks)

    def _refresh_face_file_list(self) -> None:
        """Refresh file list from current directory."""
        if self._current_dir is None or not self._current_dir.exists():
            self._face_files = []
            return

        allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        entries: list[tuple[Path, float]] = []
        with os.scandir(self._current_dir) as scanner:
            for entry in scanner:
                if not entry.is_file():
                    continue
                suffix = Path(entry.name).suffix.lower()
                if suffix not in allowed_exts:
                    continue
                mtime = entry.stat().st_mtime
                entries.append((Path(entry.path), mtime))

        if self._sort_by == "date":
            entries.sort(key=lambda item: item[1], reverse=True)
        else:
            entries.sort(key=lambda item: item[0].name)

        self._face_files = [path for path, _mtime in entries]

    def _load_thumbnail(self, filepath: Path, show_masks: bool) -> np.ndarray | None:
        """Load thumbnail from cache or generate it."""
        if self._current_dir is None:
            return None

        cache_name = "thumbs_mask" if show_masks else "thumbs"
        cache_dir = self._current_dir / ".visagen_cache" / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        stat = filepath.stat()
        cache_file = cache_dir / (
            f"{filepath.stem}_{int(stat.st_mtime)}_{stat.st_size}_{self._thumb_size}.jpg"
        )

        if cache_file.exists():
            cached = cv2.imread(str(cache_file))
            if cached is not None:
                return cast(np.ndarray, cached)

        image = cv2.imread(str(filepath))
        if image is None:
            return None

        if show_masks:
            sample = FaceSample.from_face_image(filepath)
            if sample and sample.xseg_mask:
                mask = sample.get_xseg_mask()
                if mask is not None:
                    mask_uint8 = (mask[:, :, 0] * 255).astype(np.uint8)
                    image = create_mask_overlay(image, mask_uint8, alpha=0.3)

        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > self._thumb_size:
            scale = self._thumb_size / max_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(cache_file), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return cast(np.ndarray, image)

    def _get_current_page(
        self, show_masks: bool
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Get current page of faces."""
        page_size = self._page_size
        start = self._current_page * page_size
        end = start + page_size

        page_files = self._face_files[start:end]
        gallery_items, load_errors = self._load_page_items(page_files, show_masks)
        self._schedule_cache_warm(show_masks)

        total_pages = max(1, (len(self._face_files) + page_size - 1) // page_size)
        page_info = f"Page {self._current_page + 1}/{total_pages} ({len(self._face_files)} faces)"
        status = self.t("status_selected", count=len(self._selected_files))
        if load_errors > 0:
            status = f"{status} | {self.t('status_load_errors', count=load_errors)}"
        return gallery_items, page_info, status

    def _load_page_items(
        self,
        page_files: list[Path],
        show_masks: bool,
    ) -> tuple[list[tuple[np.ndarray, str]], int]:
        """Load page thumbnails in parallel while preserving file order."""
        if len(page_files) == 0:
            return [], 0

        max_workers = max(1, min(self._thumb_workers, len(page_files)))
        loaded_by_index: list[np.ndarray | None] = [None] * len(page_files)
        load_errors = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._load_thumbnail, filepath, show_masks): idx
                for idx, filepath in enumerate(page_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    loaded_by_index[idx] = future.result()
                except Exception:
                    loaded_by_index[idx] = None

        gallery_items: list[tuple[np.ndarray, str]] = []
        for filepath, image in zip(page_files, loaded_by_index, strict=True):
            if image is None:
                load_errors += 1
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_selected = filepath in self._selected_files
            caption = f"[x] {filepath.stem}" if is_selected else filepath.stem
            gallery_items.append((image_rgb, caption))

        return gallery_items, load_errors

    def _schedule_cache_warm(self, show_masks: bool) -> None:
        """Warm thumbnail cache for the next page in background."""
        if self._cache_warm_thread is not None and self._cache_warm_thread.is_alive():
            return

        page_size = self._page_size
        next_start = (self._current_page + 1) * page_size
        next_files = self._face_files[next_start : next_start + self._cache_warm_budget]
        if len(next_files) == 0:
            return

        def _warm(files: list[Path], masks: bool) -> None:
            for path in files:
                try:
                    self._load_thumbnail(path, masks)
                except Exception:
                    continue

        self._cache_warm_thread = threading.Thread(
            target=_warm,
            args=(next_files, show_masks),
            daemon=True,
        )
        self._cache_warm_thread.start()

    def _prev_page(
        self, show_masks: bool
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Go to previous page."""
        if self._current_page > 0:
            self._current_page -= 1
        return self._get_current_page(show_masks)

    def _next_page(
        self, show_masks: bool
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Go to next page."""
        total_pages = (len(self._face_files) + self._page_size - 1) // self._page_size
        if self._current_page < total_pages - 1:
            self._current_page += 1
        return self._get_current_page(show_masks)

    def _on_select(
        self,
        show_masks: bool,
        evt: gr.SelectData,
    ) -> tuple[
        np.ndarray | None,
        dict[str, Any] | None,
        list[tuple[np.ndarray, str]],
        str,
        str,
    ]:
        """Handle gallery selection."""
        page_start = self._current_page * self._page_size
        page_end = page_start + self._page_size
        page_files = self._face_files[page_start:page_end]

        if evt.index is None or evt.index < 0 or evt.index >= len(page_files):
            gallery, page_info, status = self._get_current_page(show_masks)
            return None, None, gallery, page_info, status

        filepath = page_files[evt.index]
        if filepath in self._selected_files:
            self._selected_files.remove(filepath)
        else:
            self._selected_files.add(filepath)

        try:
            image = cv2.imread(str(filepath))
            if image is None:
                raise ValueError("Failed to load selected image")
            image = self._resize_for_preview(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            sample = FaceSample.from_face_image(filepath)
            metadata: dict[str, Any] = {
                "filename": filepath.name,
                "face_type": sample.face_type if sample else "unknown",
                "has_mask": bool(sample and sample.xseg_mask),
                "selected": filepath in self._selected_files,
            }

            gallery, page_info, status = self._get_current_page(show_masks)
            return image_rgb, metadata, gallery, page_info, status

        except Exception:
            gallery, page_info, status = self._get_current_page(show_masks)
            return None, None, gallery, page_info, status

    def _delete_selected(
        self,
        show_masks: bool,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Delete selected files by moving them to managed trash."""
        if self._current_dir is None:
            return self._get_current_page(show_masks)

        if not self._selected_files:
            gallery, page_info, _status = self._get_current_page(show_masks)
            return gallery, page_info, self.t("no_selected_files")

        batch = move_to_trash(
            sorted(self._selected_files),
            dataset_root=self._current_dir,
            reason="browser-delete",
        )
        self._selected_files.clear()
        self._refresh_face_file_list()
        total_pages = max(
            1,
            (len(self._face_files) + self._page_size - 1) // self._page_size,
        )
        self._current_page = min(self._current_page, total_pages - 1)
        gallery, page_info, _status = self._get_current_page(show_masks)
        status = self.t(
            "trash_summary",
            batch_id=batch.batch_id,
            moved=getattr(batch, "count_moved", batch.count),
            missing=getattr(batch, "count_missing", 0),
            failed=getattr(batch, "count_failed", 0),
        )
        if getattr(batch, "errors", None):
            status = f"{status} | {batch.errors[0]}"
        return (
            gallery,
            page_info,
            status,
        )

    def _clear_selection(
        self,
        show_masks: bool,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Clear current selection."""
        self._selected_files.clear()
        gallery, page_info, _status = self._get_current_page(show_masks)
        return gallery, page_info, self.t("selection_cleared")

    def _undo_last_delete(
        self,
        show_masks: bool,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Undo most recent delete batch."""
        if self._current_dir is None:
            return self._get_current_page(show_masks)

        result = undo_last_batch(self._current_dir)
        self._refresh_face_file_list()
        total_pages = max(
            1,
            (len(self._face_files) + self._page_size - 1) // self._page_size,
        )
        self._current_page = min(self._current_page, total_pages - 1)
        gallery, page_info, _status = self._get_current_page(show_masks)

        if result.batch_id:
            status = self.t(
                "undo_summary",
                batch_id=result.batch_id,
                restored=result.restored,
                skipped=result.skipped,
                failed=getattr(result, "failed", 0),
            )
            if getattr(result, "errors", None):
                status = f"{status} | {result.errors[0]}"
            return (
                gallery,
                page_info,
                status,
            )
        return gallery, page_info, self.t("no_trash_batch")

    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        """Resize selected preview image for stable browser memory usage."""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._selected_preview_max_size:
            return image

        scale = self._selected_preview_max_size / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cast(
            np.ndarray,
            cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA),
        )
