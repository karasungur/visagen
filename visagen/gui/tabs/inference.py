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
        source_img: np.ndarray,
        target_img: np.ndarray,
    ) -> np.ndarray | None:
        """
        Perform face swap inference.

        Args:
            source_img: Source face image.
            target_img: Target face image.

        Returns:
            Swapped face image.
        """
        if not self.state.model.is_loaded:
            raise gr.Error(self.i18n.t("errors.no_model_loaded"))

        if source_img is None:
            raise gr.Error(self.i18n.t("errors.source_image_required"))

        try:
            import cv2
            import torch

            # Preprocess: resize and normalize
            img = cv2.resize(source_img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img).unsqueeze(0)  # Add batch dim

            if self.state.model.device == "cuda":
                img = img.cuda()

            # Inference
            with torch.no_grad():
                output = self.state.model.model(img)

            # Postprocess
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
            output = np.clip(output * 255, 0, 255).astype(np.uint8)

            # Resize back to original size if needed
            if source_img.shape[:2] != (256, 256):
                output = cv2.resize(output, (source_img.shape[1], source_img.shape[0]))

            return output

        except Exception as e:
            raise gr.Error(f"Inference failed: {e}")
