"""Interactive merge tab for real-time face merging with preview."""

from __future__ import annotations

from typing import Any

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
                    components["save_session_btn"] = gr.Button(
                        self.t("export.save_session")
                    )

                components["export_status"] = gr.Textbox(
                    label=self.t("export.status_label"),
                    interactive=False,
                )

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
            fn=self._on_export_all,
            outputs=c["export_status"],
        )

        c["save_session_btn"].click(
            fn=self._on_save_session,
            outputs=c["export_status"],
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

    def _on_export_all(self) -> str:
        """Export all frames with current configuration."""
        if self.state._interactive_merger is None:
            return self.t("errors.no_session")

        try:
            success, message, count = self.state._interactive_merger.export_all()
            if success:
                return self.t("export.all_success", count=count)
            return f"{self.t('export.failed')}: {message}"
        except Exception as e:
            return f"{self.t('export.failed')}: {e}"

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
