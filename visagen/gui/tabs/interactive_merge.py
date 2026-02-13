"""Interactive merge tab for real-time face merging with preview."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import gradio as gr
import numpy as np

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    PathInput,
    PathInputConfig,
    SliderConfig,
    SliderInput,
)
from visagen.gui.tabs.base import BaseTab
from visagen.merger.interactive_config import MASK_MODES

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class InteractiveMergeTab(BaseTab):
    """
    Interactive face merge tab with real-time preview.

    Allows users to:
    - Load a session with checkpoint and frames directory
    - Adjust merge parameters in real-time
    - Navigate through frames with preview
    - Export current frame or all frames
    - Save/load session configuration
    """

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        """Initialize export worker state."""
        super().__init__(app_state, i18n)
        self._export_lock = threading.Lock()
        self._export_stop_event = threading.Event()
        self._export_thread: threading.Thread | None = None
        self._export_state: dict[str, Any] = {
            "running": False,
            "current": 0,
            "total": 0,
            "message": "Idle",
            "success": None,
        }

    def _set_export_state(self, **updates: Any) -> None:
        """Update shared export state under lock."""
        with self._export_lock:
            self._export_state.update(updates)

    def _get_export_snapshot(self) -> dict[str, Any]:
        """Get a thread-safe snapshot of export state."""
        with self._export_lock:
            return dict(self._export_state)

    @staticmethod
    def _format_export_snapshot(snapshot: dict[str, Any]) -> tuple[str, str]:
        """Format export status + progress strings for UI."""
        status = str(snapshot.get("message", "Idle"))
        total = int(snapshot.get("total", 0) or 0)
        current = int(snapshot.get("current", 0) or 0)

        if total > 0:
            current = max(0, min(current, total))
            percent = (current / total) * 100.0
            progress = f"{current}/{total} ({percent:.1f}%)"
        else:
            progress = "Idle"

        return status, progress

    def _run_export_worker(self) -> None:
        """Background worker that exports all frames with cancellation support."""
        merger = self.state._interactive_merger
        if merger is None:
            self._set_export_state(
                running=False,
                message=self.t("errors.no_session"),
                success=False,
                total=0,
                current=0,
            )
            return

        def _progress_cb(current: int, total: int) -> None:
            self._set_export_state(
                current=int(current),
                total=int(total),
                message=f"Exporting {int(current)}/{int(total)}",
            )

        try:
            success, message, exported = merger.export_all(
                progress_callback=_progress_cb,
                stop_event=self._export_stop_event,
            )
            total = len(merger.frames)
            final_current = max(0, min(int(exported), int(total)))
            self._set_export_state(
                running=False,
                current=final_current,
                total=int(total),
                message=message,
                success=bool(success),
            )
        except Exception as e:
            self._set_export_state(
                running=False,
                message=f"{self.t('export.failed')}: {e}",
                success=False,
            )
        finally:
            self._export_stop_event.clear()

    @property
    def id(self) -> str:
        return "interactive_merge"

    def _build_content(self) -> dict[str, Any]:
        """Build interactive merge tab UI components."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # === Session Setup Section ===
        gr.Markdown(f"#### {self.t('session.title')}")
        with gr.Row():
            with gr.Column():
                components["checkpoint"] = PathInput(
                    PathInputConfig(
                        key="interactive_merge.checkpoint",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["frames_dir"] = PathInput(
                    PathInputConfig(
                        key="interactive_merge.frames_dir",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["output_dir"] = PathInput(
                    PathInputConfig(
                        key="interactive_merge.output_dir",
                        path_type="directory",
                        default="./output",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["load_btn"] = gr.Button(
                    self.t("load_session"),
                    variant="primary",
                )
                components["session_status"] = gr.Textbox(
                    label=self.t("session_status.label"),
                    value=self.t("session_status.not_loaded"),
                    interactive=False,
                )

        gr.Markdown("---")

        with gr.Row():
            # === Left Column: Controls ===
            with gr.Column(scale=1):
                gr.Markdown(f"#### {self.t('settings.title')}")

                # Merge mode dropdown
                components["mode"] = DropdownInput(
                    DropdownConfig(
                        key="interactive_merge.mode",
                        choices=[
                            "original",
                            "overlay",
                            "hist-match",
                            "seamless",
                            "seamless-hist-match",
                        ],
                        default="overlay",
                    ),
                    self.i18n,
                ).build()

                # Mask mode dropdown
                components["mask_mode"] = DropdownInput(
                    DropdownConfig(
                        key="interactive_merge.mask_mode",
                        choices=list(MASK_MODES.keys()),
                        default="segmented",
                    ),
                    self.i18n,
                ).build()

                # Color transfer dropdown
                components["color_transfer"] = DropdownInput(
                    DropdownConfig(
                        key="interactive_merge.color_transfer",
                        choices=[
                            "none",
                            "rct",
                            "lct",
                            "sot",
                            "mkl",
                            "idt",
                            "mix",
                            "hist-match",
                        ],
                        default="rct",
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('mask_processing.title')}")

                components["erode_mask"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.erode_mask",
                        minimum=-100,
                        maximum=100,
                        step=1,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                components["blur_mask"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.blur_mask",
                        minimum=0,
                        maximum=100,
                        step=1,
                        default=10,
                    ),
                    self.i18n,
                ).build()

                components["face_scale"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.face_scale",
                        minimum=-50,
                        maximum=50,
                        step=1,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('sharpening.title')}")

                components["sharpen_mode"] = DropdownInput(
                    DropdownConfig(
                        key="interactive_merge.sharpen_mode",
                        choices=["none", "box", "gaussian"],
                        default="none",
                    ),
                    self.i18n,
                ).build()

                components["sharpen_amount"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.sharpen_amount",
                        minimum=-100,
                        maximum=100,
                        step=1,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('advanced.title')}")

                components["hist_threshold"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.hist_threshold",
                        minimum=0,
                        maximum=255,
                        step=1,
                        default=238,
                    ),
                    self.i18n,
                ).build()

                components["restore_face"] = gr.Checkbox(
                    label=self.t("restore_face.label"),
                    value=False,
                    info=self.t("restore_face.info"),
                )

                components["restore_strength"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.restore_strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        default=0.5,
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('super_resolution.title')}")

                components["super_resolution_power"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.super_resolution_power",
                        minimum=0,
                        maximum=100,
                        step=5,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('motion_blur.title')}")

                components["motion_blur_power"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.motion_blur_power",
                        minimum=0,
                        maximum=100,
                        step=1,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                gr.Markdown(f"#### {self.t('degradation.title')}")

                components["image_denoise_power"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.image_denoise_power",
                        minimum=0,
                        maximum=500,
                        step=10,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                components["bicubic_degrade_power"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.bicubic_degrade_power",
                        minimum=0,
                        maximum=100,
                        step=5,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                components["color_degrade_power"] = SliderInput(
                    SliderConfig(
                        key="interactive_merge.color_degrade_power",
                        minimum=0,
                        maximum=100,
                        step=5,
                        default=0,
                    ),
                    self.i18n,
                ).build()

                # Apply button
                components["apply_btn"] = gr.Button(
                    self.t("apply_settings"),
                    variant="secondary",
                )
                components["config_status"] = gr.Textbox(
                    label=self.t("config_status.label"),
                    value="Mode: overlay | Mask: segmented | Color: rct",
                    interactive=False,
                )

            # === Right Column: Preview ===
            with gr.Column(scale=2):
                gr.Markdown(f"#### {self.t('preview.title')}")

                with gr.Row():
                    components["show_original"] = gr.Checkbox(
                        label=self.t("show_original.label"),
                        value=False,
                    )

                components["preview"] = gr.Image(
                    label=self.t("preview.image_label"),
                    type="numpy",
                    height=512,
                )

                components["frame_info"] = gr.Textbox(
                    label=self.t("frame_info.label"),
                    value=self.t("frame_info.no_frame"),
                    interactive=False,
                )

                # Navigation
                with gr.Row():
                    components["prev_btn"] = gr.Button(self.t("nav.prev"))
                    components["frame_slider"] = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label=self.t("nav.frame"),
                    )
                    components["next_btn"] = gr.Button(self.t("nav.next"))

                gr.Markdown("---")

                # Export section
                gr.Markdown(f"#### {self.t('export.title')}")
                with gr.Row():
                    components["export_current_btn"] = gr.Button(
                        self.t("export.current")
                    )
                    components["export_all_btn"] = gr.Button(
                        self.t("export.all"),
                        variant="primary",
                    )
                    components["export_stop_btn"] = gr.Button(
                        "Stop export",
                        variant="stop",
                    )
                    components["save_session_btn"] = gr.Button(
                        self.t("export.save_session")
                    )

                components["export_status"] = gr.Textbox(
                    label=self.t("export.status_label"),
                    interactive=False,
                )
                components["export_progress"] = gr.Textbox(
                    label="Export progress",
                    value="Idle",
                    interactive=False,
                )
                components["export_timer"] = gr.Timer(value=1)

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up interactive merge event handlers."""

        # Load session
        c["load_btn"].click(
            fn=self._on_load_session,
            inputs=[c["checkpoint"], c["frames_dir"], c["output_dir"]],
            outputs=[
                c["session_status"],
                c["preview"],
                c["frame_slider"],
                c["frame_info"],
            ],
        )

        # Config inputs for apply settings
        config_inputs = [
            c["mode"],
            c["mask_mode"],
            c["color_transfer"],
            c["erode_mask"],
            c["blur_mask"],
            c["face_scale"],
            c["sharpen_mode"],
            c["sharpen_amount"],
            c["hist_threshold"],
            c["restore_face"],
            c["restore_strength"],
            c["super_resolution_power"],
            c["motion_blur_power"],
            c["image_denoise_power"],
            c["bicubic_degrade_power"],
            c["color_degrade_power"],
            c["show_original"],
        ]

        # Apply settings button
        c["apply_btn"].click(
            fn=self._on_apply_settings,
            inputs=config_inputs,
            outputs=[c["preview"], c["config_status"]],
        )

        # Show original toggle
        c["show_original"].change(
            fn=self._on_apply_settings,
            inputs=config_inputs,
            outputs=[c["preview"], c["config_status"]],
        )

        # Real-time updates for sliders: Use .release() instead of .change() for debouncing
        # This triggers only when user releases the slider, not during drag
        sliders = [
            "erode_mask",
            "blur_mask",
            "face_scale",
            "sharpen_amount",
            "hist_threshold",
            "restore_strength",
            "super_resolution_power",
            "motion_blur_power",
            "image_denoise_power",
            "bicubic_degrade_power",
            "color_degrade_power",
        ]
        for name in sliders:
            c[name].release(
                fn=self._on_apply_settings,
                inputs=config_inputs,
                outputs=[c["preview"], c["config_status"]],
                show_progress="hidden",
            )

        # Dropdowns trigger immediate update
        dropdowns = ["mode", "mask_mode", "color_transfer", "sharpen_mode"]
        for name in dropdowns:
            c[name].change(
                fn=self._on_apply_settings,
                inputs=config_inputs,
                outputs=[c["preview"], c["config_status"]],
                show_progress="hidden",
            )

        # Checkbox triggers immediate update
        c["restore_face"].change(
            fn=self._on_apply_settings,
            inputs=config_inputs,
            outputs=[c["preview"], c["config_status"]],
            show_progress="hidden",
        )

        # Navigation
        c["prev_btn"].click(
            fn=self._on_navigate_prev,
            outputs=[c["preview"], c["frame_slider"], c["frame_info"]],
        )

        c["next_btn"].click(
            fn=self._on_navigate_next,
            outputs=[c["preview"], c["frame_slider"], c["frame_info"]],
        )

        c["frame_slider"].release(
            fn=self._on_slider_change,
            inputs=[c["frame_slider"]],
            outputs=[c["preview"], c["frame_info"]],
        )

        # Export handlers
        c["export_current_btn"].click(
            fn=self._on_export_current,
            outputs=c["export_status"],
        )

        c["export_all_btn"].click(
            fn=self._on_export_all_start,
            outputs=[c["export_status"], c["export_progress"]],
        )

        c["export_stop_btn"].click(
            fn=self._on_export_stop,
            outputs=[c["export_status"], c["export_progress"]],
        )

        c["save_session_btn"].click(
            fn=self._on_save_session,
            outputs=c["export_status"],
        )

        c["export_timer"].tick(
            fn=self._poll_export_progress,
            outputs=[c["export_status"], c["export_progress"]],
        )

    # =========================================================================
    # Event Handler Methods
    # =========================================================================

    def _on_load_session(
        self,
        checkpoint_path: str,
        frames_dir: str,
        output_dir: str,
    ) -> tuple[str, np.ndarray | None, Any, str]:
        """Handle session load."""
        from visagen.merger.interactive import InteractiveMerger

        try:
            self._export_stop_event.set()
            self._set_export_state(
                running=False,
                current=0,
                total=0,
                message="Idle",
                success=None,
            )
            # Create new merger instance
            self.state._interactive_merger = InteractiveMerger(
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                frames_dir=frames_dir if frames_dir else None,
                output_dir=output_dir or "./output",
            )

            # Load session
            success, message = self.state._interactive_merger.load_session()

            if not success:
                return (
                    message,
                    None,
                    gr.update(maximum=0, value=0),
                    self.t("frame_info.no_frame"),
                )

            # Get first frame preview
            preview = self.state._interactive_merger.process_current_frame()
            info = self.state._interactive_merger.get_current_frame_info()

            return (
                self.t("session_status.loaded", count=info["total"], path=frames_dir),
                preview,
                gr.update(maximum=max(0, info["total"] - 1), value=info["index"]),
                self.t(
                    "frame_info.format", current=info["index"] + 1, total=info["total"]
                ),
            )

        except Exception as e:
            return (
                f"{self.t('errors.load_failed')}: {e}",
                None,
                gr.update(maximum=0, value=0),
                self.t("frame_info.no_frame"),
            )

    def _on_apply_settings(
        self,
        mode: str,
        mask_mode: str,
        color_transfer: str,
        erode_mask: int,
        blur_mask: int,
        face_scale: int,
        sharpen_mode: str,
        sharpen_amount: int,
        hist_threshold: int,
        restore_face: bool,
        restore_strength: float,
        super_resolution_power: int,
        motion_blur_power: int,
        image_denoise_power: int,
        bicubic_degrade_power: int,
        color_degrade_power: int,
        show_original: bool,
    ) -> tuple[np.ndarray | None, str]:
        """Handle settings update."""
        if self.state._interactive_merger is None:
            return None, self.t("errors.no_session")

        try:
            if show_original:
                original = self.state._interactive_merger.get_original_frame()
                return original, self.t("config_status.showing_original")

            preview = self.state._interactive_merger.update_config(
                mode=mode,
                mask_mode=mask_mode,
                color_transfer=color_transfer,
                erode_mask=erode_mask,
                blur_mask=blur_mask,
                face_scale=face_scale,
                sharpen_mode=sharpen_mode,
                sharpen_amount=sharpen_amount,
                hist_match_threshold=hist_threshold,
                restore_face=restore_face,
                restore_strength=restore_strength,
                super_resolution_power=int(super_resolution_power),
                motion_blur_power=int(motion_blur_power),
                image_denoise_power=int(image_denoise_power),
                bicubic_degrade_power=int(bicubic_degrade_power),
                color_degrade_power=int(color_degrade_power),
            )

            # Get status string from config
            status = self.state._interactive_merger.session.config.to_status_string()
            process_status = self.state._interactive_merger.get_last_process_status()
            if process_status != "ok":
                status = f"{status} | Warning: {process_status}"
            return preview, status

        except Exception as e:
            return None, f"{self.t('errors.update_failed')}: {e}"

    def _on_navigate_prev(self) -> tuple[np.ndarray | None, Any, str]:
        """Navigate to previous frame."""
        if self.state._interactive_merger is None:
            return None, 0, self.t("errors.no_session")

        try:
            preview, idx = self.state._interactive_merger.navigate(-1)
            info = self.state._interactive_merger.get_current_frame_info()

            return (
                preview,
                idx,
                self.t(
                    "frame_info.detail",
                    current=info["index"] + 1,
                    total=info["total"],
                    filename=info["filename"],
                ),
            )
        except Exception as e:
            return None, 0, f"{self.t('errors.navigate_failed')}: {e}"

    def _on_navigate_next(self) -> tuple[np.ndarray | None, Any, str]:
        """Navigate to next frame."""
        if self.state._interactive_merger is None:
            return None, 0, self.t("errors.no_session")

        try:
            preview, idx = self.state._interactive_merger.navigate(1)
            info = self.state._interactive_merger.get_current_frame_info()

            return (
                preview,
                idx,
                self.t(
                    "frame_info.detail",
                    current=info["index"] + 1,
                    total=info["total"],
                    filename=info["filename"],
                ),
            )
        except Exception as e:
            return None, 0, f"{self.t('errors.navigate_failed')}: {e}"

    def _on_slider_change(self, idx: int) -> tuple[np.ndarray | None, str]:
        """Handle frame slider change."""
        if self.state._interactive_merger is None:
            return None, self.t("errors.no_session")

        try:
            preview, actual_idx = self.state._interactive_merger.go_to_frame(int(idx))
            info = self.state._interactive_merger.get_current_frame_info()

            return (
                preview,
                self.t(
                    "frame_info.detail",
                    current=info["index"] + 1,
                    total=info["total"],
                    filename=info["filename"],
                ),
            )
        except Exception as e:
            return None, f"{self.t('errors.navigate_failed')}: {e}"

    def _on_export_current(self) -> str:
        """Export current frame."""
        if self.state._interactive_merger is None:
            return self.t("errors.no_session")

        try:
            idx = self.state._interactive_merger.session.current_idx
            success, message = self.state._interactive_merger.export_frame(idx)
            if success:
                return self.t("export.current_success", path=message)
            return f"{self.t('export.failed')}: {message}"
        except Exception as e:
            return f"{self.t('export.failed')}: {e}"

    def _on_export_all_start(self) -> tuple[str, str]:
        """Start asynchronous export worker."""
        if self.state._interactive_merger is None:
            return self.t("errors.no_session"), "Idle"

        try:
            with self._export_lock:
                if (
                    self._export_thread is not None
                    and self._export_thread.is_alive()
                    and self._export_state.get("running", False)
                ):
                    snapshot = dict(self._export_state)
                    return self._format_export_snapshot(snapshot)

                total = len(self.state._interactive_merger.frames)
                self._export_stop_event.clear()
                self._export_state.update(
                    {
                        "running": True,
                        "current": 0,
                        "total": int(total),
                        "message": f"Export started (0/{int(total)})",
                        "success": None,
                    }
                )
                self._export_thread = threading.Thread(
                    target=self._run_export_worker,
                    name="visagen-interactive-export",
                    daemon=True,
                )
                thread = self._export_thread

            thread.start()
            return self._format_export_snapshot(self._get_export_snapshot())
        except Exception as e:
            self._set_export_state(
                running=False,
                message=f"{self.t('export.failed')}: {e}",
                success=False,
            )
            return self._format_export_snapshot(self._get_export_snapshot())

    def _on_export_stop(self) -> tuple[str, str]:
        """Request cancellation for running export worker."""
        snapshot = self._get_export_snapshot()
        if not snapshot.get("running", False):
            return self._format_export_snapshot(snapshot)

        self._export_stop_event.set()
        self._set_export_state(message="Stopping export...")
        return self._format_export_snapshot(self._get_export_snapshot())

    def _poll_export_progress(self) -> tuple[str, str]:
        """Poll export state for timer-driven UI refresh."""
        with self._export_lock:
            if (
                self._export_thread is not None
                and not self._export_thread.is_alive()
                and not self._export_state.get("running", False)
            ):
                self._export_thread = None
            snapshot = dict(self._export_state)

        return self._format_export_snapshot(snapshot)

    def _on_save_session(self) -> str:
        """Save current interactive session."""
        if self.state._interactive_merger is None:
            return self.t("errors.no_session")

        try:
            success, message = self.state._interactive_merger.save_session()
            if success:
                return self.t("export.session_saved", path=message)
            return f"{self.t('export.save_failed')}: {message}"
        except Exception as e:
            return f"{self.t('export.save_failed')}: {e}"
