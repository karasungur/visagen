"""Inference tab implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gradio as gr
import numpy as np

from visagen.gui.components import (
    ImagePreview,
    ImagePreviewConfig,
    PathInput,
    PathInputConfig,
)
from visagen.gui.tabs.base import BaseTab
from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

if TYPE_CHECKING:
    pass


class InferenceTab(BaseTab):
    """
    Single image inference tab.

    Allows users to:
    - Load a trained model checkpoint
    - Perform face swapping on single images
    - Visualize results immediately
    """

    @property
    def id(self) -> str:
        return "inference"

    def _build_content(self) -> dict[str, Any]:
        """Build inference tab UI."""
        components = {}

        gr.Markdown(f"### {self.t('title')}")

        # Model Loading Section
        with gr.Row():
            components["checkpoint"] = PathInput(
                PathInputConfig(
                    key="inference.checkpoint",
                    path_type="file",
                    file_types=[".ckpt"],
                ),
                self.i18n,
            ).build()

            components["load_btn"] = gr.Button(self.t("load_model"))
            components["unload_btn"] = gr.Button(self.t("unload_model"))

        components["model_status"] = gr.Textbox(
            label=self.t("model_status.label"),
            value=self.i18n.t("status.no_model"),
            interactive=False,
        )

        gr.Markdown("---")

        # Inference Section
        with gr.Row():
            components["source_image"] = ImagePreview(
                ImagePreviewConfig(key="inference.source_image"),
                self.i18n,
            ).build()

            components["target_image"] = ImagePreview(
                ImagePreviewConfig(key="inference.target_image"),
                self.i18n,
            ).build()

            components["output_image"] = ImagePreview(
                ImagePreviewConfig(key="inference.output_image"),
                self.i18n,
            ).build()

        with gr.Row():
            components["swap_btn"] = gr.Button(
                self.t("swap"),
                variant="primary",
            )

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up inference event handlers."""

        # Model loading
        c["load_btn"].click(
            fn=self._load_model,
            inputs=[c["checkpoint"]],
            outputs=[c["model_status"]],
        )

        c["unload_btn"].click(
            fn=self._unload_model,
            outputs=[c["model_status"]],
        )

        # Inference
        c["swap_btn"].click(
            fn=self._swap_face,
            inputs=[c["source_image"], c["target_image"]],
            outputs=[c["output_image"]],
        )

    def _load_model(self, checkpoint_path: str) -> str:
        """Load model wrapper."""
        device = self.state.settings.device
        return self.state.model.load(checkpoint_path, device)

    def _unload_model(self) -> str:
        """Unload model wrapper."""
        return self.state.model.unload()

    def _swap_face(
        self,
        source_img: np.ndarray | None,
        target_img: np.ndarray,
    ) -> np.ndarray | None:
        """
        Perform face swap inference.

        Uses FrameProcessor for proper face detection, alignment, and blending.
        The model is trained to produce a specific source identity, so source_img
        is optional (for reference display only). The actual swap happens on
        target_img using the identity learned during training.

        Args:
            source_img: Optional reference image (not used in inference,
                        the model already knows the source identity).
            target_img: Target frame (where the face will be swapped).

        Returns:
            Swapped face image with model's learned identity on target frame.
        """
        if not self.state.model.is_loaded:
            raise gr.Error(self.i18n.t("errors.no_model_loaded"))

        if target_img is None:
            raise gr.Error(self.i18n.t("errors.target_image_required"))

        try:
            import cv2

            # Create frame processor with loaded model
            config = FrameProcessorConfig(
                face_type="whole_face",
                output_size=256,
                color_transfer_mode="rct",
                blend_mode="laplacian",
                mask_erode=5,
                mask_blur=5,
            )
            processor = FrameProcessor(
                model=self.state.model.model,
                config=config,
                device=self.state.model.device,
            )

            # Convert RGB to BGR (Gradio uses RGB, OpenCV uses BGR)
            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

            # Process frame (target_img is where the swap happens)
            result = processor.process_frame(target_bgr, frame_idx=0)

            if result.faces_swapped == 0:
                raise gr.Error(self.i18n.t("errors.no_face_detected"))

            # Convert BGR back to RGB for Gradio display
            output_rgb = cv2.cvtColor(result.output_image, cv2.COLOR_BGR2RGB)

            return output_rgb

        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Inference failed: {e}")
