"""Postprocess demo tab for color transfer and blending."""

from __future__ import annotations

from typing import Any, Literal, cast

import gradio as gr
import numpy as np

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    ImagePreview,
    ImagePreviewConfig,
    SliderConfig,
    SliderInput,
)
from visagen.gui.tabs.base import BaseTab

NeuralColorMode = Literal["histogram", "statistics", "gram"]


class PostprocessTab(BaseTab):
    """
    Postprocess demonstration tab.

    Provides demos for:
    - Color transfer (RCT, LCT, SOT, etc.)
    - Image blending (Laplacian, Poisson, Feather)
    - Face restoration (GFPGAN, GPEN)
    - Neural color transfer
    """

    @property
    def id(self) -> str:
        return "postprocess"

    def _build_content(self) -> dict[str, Any]:
        """Build postprocess tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")

        # === Color Transfer ===
        gr.Markdown(f"#### {self.t('color_transfer.title')}")

        with gr.Row():
            components["ct_source"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.ct_source"),
                self.i18n,
            ).build()

            components["ct_target"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.ct_target"),
                self.i18n,
            ).build()

            components["ct_result"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.ct_result"),
                self.i18n,
            ).build()

        with gr.Row():
            components["ct_mode"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.ct_mode",
                    choices=["rct", "lct", "sot", "mkl", "idt"],
                    default="rct",
                ),
                self.i18n,
            ).build()

            components["ct_btn"] = gr.Button(self.t("color_transfer.apply"))

        gr.Markdown("---")

        # === Blending ===
        gr.Markdown(f"#### {self.t('blending.title')}")

        with gr.Row():
            components["bl_fg"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.bl_fg"),
                self.i18n,
            ).build()

            components["bl_bg"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.bl_bg"),
                self.i18n,
            ).build()

            components["bl_mask"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.bl_mask"),
                self.i18n,
            ).build()

            components["bl_result"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.bl_result"),
                self.i18n,
            ).build()

        with gr.Row():
            components["bl_mode"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.bl_mode",
                    choices=["laplacian", "poisson", "feather"],
                    default="laplacian",
                ),
                self.i18n,
            ).build()

            components["bl_btn"] = gr.Button(self.t("blending.apply"))

        gr.Markdown("---")

        # === Restoration ===
        gr.Markdown(f"#### {self.t('restoration.title')}")
        gr.Markdown(self.t("restoration.description"))

        with gr.Row():
            components["restore_input"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.restore_input"),
                self.i18n,
            ).build()

            components["restore_result"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.restore_result"),
                self.i18n,
            ).build()

        with gr.Row():
            components["restore_mode"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.restore_mode",
                    choices=["gfpgan", "gpen"],
                    default="gfpgan",
                ),
                self.i18n,
            ).build()

            components["restore_strength"] = SliderInput(
                SliderConfig(
                    key="postprocess.restore_strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    default=0.5,
                ),
                self.i18n,
            ).build()

        with gr.Row():
            components["restore_version"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.restore_version",
                    choices=["1.2", "1.3", "1.4"],
                    default="1.4",
                ),
                self.i18n,
            ).build()

            components["gpen_size"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.gpen_size",
                    choices=["256", "512", "1024"],
                    default="512",
                ),
                self.i18n,
            ).build()

            components["restore_btn"] = gr.Button(self.t("restoration.apply"))

        gr.Markdown("---")

        # === Neural Color ===
        gr.Markdown(f"#### {self.t('neural.title')}")
        gr.Markdown(self.t("neural.description"))

        with gr.Row():
            components["nct_source"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.nct_source"),
                self.i18n,
            ).build()

            components["nct_target"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.nct_target"),
                self.i18n,
            ).build()

            components["nct_result"] = ImagePreview(
                ImagePreviewConfig(key="postprocess.nct_result"),
                self.i18n,
            ).build()

        with gr.Row():
            components["nct_mode"] = DropdownInput(
                DropdownConfig(
                    key="postprocess.nct_mode",
                    choices=["histogram", "statistics", "gram"],
                    default="histogram",
                ),
                self.i18n,
            ).build()

            components["nct_strength"] = SliderInput(
                SliderConfig(
                    key="postprocess.nct_strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    default=0.8,
                ),
                self.i18n,
            ).build()

            components["nct_preserve"] = gr.Checkbox(
                label=self.t("nct_preserve.label"),
                value=True,
            )

            components["nct_btn"] = gr.Button(self.t("neural.apply"))

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up postprocess event handlers."""

        # Color Transfer
        c["ct_btn"].click(
            fn=self._apply_color_transfer,
            inputs=[c["ct_source"], c["ct_target"], c["ct_mode"]],
            outputs=c["ct_result"],
        )

        # Blending
        c["bl_btn"].click(
            fn=self._apply_blend,
            inputs=[c["bl_fg"], c["bl_bg"], c["bl_mask"], c["bl_mode"]],
            outputs=c["bl_result"],
        )

        # Restoration
        c["restore_btn"].click(
            fn=self._apply_restoration,
            inputs=[
                c["restore_input"],
                c["restore_strength"],
                c["restore_mode"],
                c["restore_version"],
                c["gpen_size"],
            ],
            outputs=c["restore_result"],
        )

        # Neural Color
        c["nct_btn"].click(
            fn=self._apply_neural_color,
            inputs=[
                c["nct_source"],
                c["nct_target"],
                c["nct_mode"],
                c["nct_strength"],
                c["nct_preserve"],
            ],
            outputs=c["nct_result"],
        )

    # =========================================================================
    # Operation Handlers
    # =========================================================================

    def _apply_color_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """Apply color transfer."""
        if source is None or target is None:
            raise gr.Error(self.i18n.t("errors.missing_images"))

        try:
            import cv2

            from visagen.postprocess import color_transfer
            from visagen.postprocess.color_transfer import ColorTransferMode

            # Ensure float32 [0, 1] and BGR
            if source.dtype == np.uint8:
                source = source.astype(np.float32) / 255.0
            if target.dtype == np.uint8:
                target = target.astype(np.float32) / 255.0

            source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            result_bgr = color_transfer(
                cast(ColorTransferMode, mode), target_bgr, source_bgr
            )
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            return (result_rgb * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Color transfer failed: {e}")

    def _apply_blend(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """Apply blending."""
        if foreground is None or background is None or mask is None:
            raise gr.Error(self.i18n.t("errors.missing_images"))

        try:
            import cv2

            from visagen.postprocess import blend
            from visagen.postprocess.blending import BlendMode

            if foreground.dtype == np.uint8:
                foreground = foreground.astype(np.float32) / 255.0
            if background.dtype == np.uint8:
                background = background.astype(np.float32) / 255.0
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0

            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            if mask.shape[:2] != foreground.shape[:2]:
                mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

            result = blend(cast(BlendMode, mode), foreground, background, mask)
            return (result * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Blending failed: {e}")

    def _apply_restoration(
        self,
        image: np.ndarray,
        strength: float,
        mode: str,
        version: str,
        gpen_size: str,
    ) -> np.ndarray:
        """Apply face restoration."""
        if image is None:
            raise gr.Error(self.i18n.t("errors.missing_images"))

        try:
            import cv2

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if mode == "gpen":
                from visagen.postprocess.gpen import GPENModelSize, restore_face_gpen

                model_size_int = int(gpen_size)
                if model_size_int not in (256, 512, 1024):
                    raise gr.Error(f"Unsupported GPEN model size: {gpen_size}")
                restored_bgr = restore_face_gpen(
                    image_bgr,
                    strength=strength,
                    model_size=cast(GPENModelSize, model_size_int),
                )
            else:
                from visagen.postprocess.restore import restore_face

                restored_bgr = restore_face(
                    image_bgr,
                    strength=strength,
                    model_version=float(version),
                )

            return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise gr.Error(f"Restoration failed: {e}")

    def _apply_neural_color(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mode: str,
        strength: float,
        preserve: bool,
    ) -> np.ndarray:
        """Apply neural color transfer."""
        if source is None or target is None:
            raise gr.Error(self.i18n.t("errors.missing_images"))

        try:
            import cv2

            from visagen.postprocess.neural_color import neural_color_transfer

            source_bgr = (
                cv2.cvtColor(source, cv2.COLOR_RGB2BGR).astype(np.float32) / 255
            )
            target_bgr = (
                cv2.cvtColor(target, cv2.COLOR_RGB2BGR).astype(np.float32) / 255
            )

            result_bgr = neural_color_transfer(
                target_bgr,
                source_bgr,
                mode=cast(NeuralColorMode, mode),
                strength=strength,
                preserve_luminance=preserve,
            )

            return cv2.cvtColor(
                (result_bgr * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_BGR2RGB,
            )

        except Exception as e:
            raise gr.Error(f"Neural color transfer failed: {e}")
