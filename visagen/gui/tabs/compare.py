"""Model comparison tab implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gradio as gr
import numpy as np

from visagen.gui.components import (
    PathInput,
    PathInputConfig,
)
from visagen.gui.tabs.base import BaseTab

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class CompareTab(BaseTab):
    """
    Model comparison tab.

    Compare outputs from two different checkpoints side by side.
    Includes optional SSIM/PSNR quality metrics.
    """

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        """Initialize comparison tab with model slots."""
        super().__init__(app_state, i18n)
        self._model_a: Any = None
        self._model_b: Any = None
        self._device: str = "cpu"

    @property
    def id(self) -> str:
        return "compare"

    def _build_content(self) -> dict[str, Any]:
        """Build comparison tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # Model Selection Row
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model A")
                components["ckpt_a"] = PathInput(
                    PathInputConfig(
                        key="compare.checkpoint_a",
                        path_type="file",
                        file_types=[".ckpt"],
                    ),
                    self.i18n,
                ).build()
                components["load_a_btn"] = gr.Button(self.t("load_model_a"))
                components["status_a"] = gr.Textbox(
                    label=self.t("status_a.label"),
                    value=self.i18n.t("status.no_model"),
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("#### Model B")
                components["ckpt_b"] = PathInput(
                    PathInputConfig(
                        key="compare.checkpoint_b",
                        path_type="file",
                        file_types=[".ckpt"],
                    ),
                    self.i18n,
                ).build()
                components["load_b_btn"] = gr.Button(self.t("load_model_b"))
                components["status_b"] = gr.Textbox(
                    label=self.t("status_b.label"),
                    value=self.i18n.t("status.no_model"),
                    interactive=False,
                )

        gr.Markdown("---")

        # Test Image Input
        with gr.Row():
            components["test_image"] = gr.Image(
                label=self.t("test_image.label"),
                type="numpy",
            )

        # Compare Button
        with gr.Row():
            components["compare_btn"] = gr.Button(
                self.t("compare"),
                variant="primary",
            )

        gr.Markdown("---")
        gr.Markdown(f"#### {self.t('results.title')}")

        # Results Row
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Model A Output**")
                components["output_a"] = gr.Image(
                    label="Model A",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("**Model B Output**")
                components["output_b"] = gr.Image(
                    label="Model B",
                    interactive=False,
                )

        # Metrics
        with gr.Row():
            components["metrics"] = gr.JSON(
                label=self.t("metrics.label"),
                visible=True,
            )

        # Unload button
        with gr.Row():
            components["unload_btn"] = gr.Button(
                self.t("unload_all"),
                variant="secondary",
            )

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up comparison event handlers."""
        c["load_a_btn"].click(
            fn=self._load_model_a,
            inputs=[c["ckpt_a"]],
            outputs=[c["status_a"]],
        )

        c["load_b_btn"].click(
            fn=self._load_model_b,
            inputs=[c["ckpt_b"]],
            outputs=[c["status_b"]],
        )

        c["compare_btn"].click(
            fn=self._compare,
            inputs=[c["test_image"]],
            outputs=[c["output_a"], c["output_b"], c["metrics"]],
        )

        c["unload_btn"].click(
            fn=self._unload_all,
            outputs=[c["status_a"], c["status_b"]],
        )

    def _load_model_a(self, checkpoint_path: str) -> str:
        """Load model into slot A."""
        return self._load_model(checkpoint_path, "a")

    def _load_model_b(self, checkpoint_path: str) -> str:
        """Load model into slot B."""
        return self._load_model(checkpoint_path, "b")

    def _load_model(self, checkpoint_path: str, slot: str) -> str:
        """
        Load model into specified slot.

        Args:
            checkpoint_path: Path to checkpoint file.
            slot: Either "a" or "b".

        Returns:
            Status message.
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return self.i18n.t("errors.path_not_found")

        try:
            import torch

            from visagen.training.training_module import TrainingModule

            model = TrainingModule.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            model.eval()

            # Determine device
            if torch.cuda.is_available():
                model = model.cuda()
                self._device = "cuda"
            else:
                self._device = "cpu"

            if slot == "a":
                # Unload previous model A if exists
                if self._model_a is not None:
                    del self._model_a
                self._model_a = model
            else:
                # Unload previous model B if exists
                if self._model_b is not None:
                    del self._model_b
                self._model_b = model

            return f"✅ {Path(checkpoint_path).name} ({self._device})"

        except Exception as e:
            return f"❌ Error: {e}"

    def _compare(
        self,
        test_img: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """
        Compare both models on test image.

        Args:
            test_img: Input test image.

        Returns:
            Tuple of (output_a, output_b, metrics).
        """
        if test_img is None:
            raise gr.Error(self.i18n.t("errors.source_image_required"))

        if self._model_a is None or self._model_b is None:
            raise gr.Error(self.t("errors.both_models_required"))

        try:
            result_a = self._run_inference(self._model_a, test_img)
            result_b = self._run_inference(self._model_b, test_img)

            # Calculate metrics
            metrics = self._calculate_metrics(test_img, result_a, result_b)

            return result_a, result_b, metrics

        except Exception as e:
            raise gr.Error(f"Comparison failed: {e}")

    def _run_inference(self, model: Any, img: np.ndarray) -> np.ndarray:
        """
        Run inference on a single model.

        Args:
            model: Loaded model.
            img: Input image.

        Returns:
            Output image.
        """
        import cv2
        import torch

        # Preprocess: resize to 256x256 and normalize
        img_resized = cv2.resize(img, (256, 256))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)

        if self._device == "cuda":
            img_tensor = img_tensor.cuda()

        # Inference
        with torch.no_grad():
            output = model(img_tensor)

        # Unpack tuple if decoder returns (image, mask)
        if isinstance(output, tuple):
            output = output[0]

        # Postprocess
        output = output.squeeze(0).cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        # Resize back to original size
        if img.shape[:2] != (256, 256):
            output = cv2.resize(output, (img.shape[1], img.shape[0]))

        return cast(np.ndarray, output)

    def _calculate_metrics(
        self,
        original: np.ndarray,
        result_a: np.ndarray,
        result_b: np.ndarray,
    ) -> dict:
        """
        Calculate quality metrics for comparison.

        Args:
            original: Original input image.
            result_a: Output from model A.
            result_b: Output from model B.

        Returns:
            Dictionary with SSIM, PSNR, and winner.
        """
        try:
            import cv2
            from skimage.metrics import structural_similarity as ssim

            # Resize all to same size for fair comparison
            size = (256, 256)
            orig = cv2.resize(original, size)
            a = cv2.resize(result_a, size)
            b = cv2.resize(result_b, size)

            # Convert to grayscale for SSIM
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
            a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

            ssim_a = ssim(orig_gray, a_gray)
            ssim_b = ssim(orig_gray, b_gray)

            # PSNR calculation
            mse_a = np.mean((orig.astype(float) - a.astype(float)) ** 2)
            mse_b = np.mean((orig.astype(float) - b.astype(float)) ** 2)
            psnr_a = (
                20 * np.log10(255.0 / np.sqrt(mse_a)) if mse_a > 0 else float("inf")
            )
            psnr_b = (
                20 * np.log10(255.0 / np.sqrt(mse_b)) if mse_b > 0 else float("inf")
            )

            # Determine winner based on SSIM (higher is better)
            winner = "A" if ssim_a > ssim_b else "B" if ssim_b > ssim_a else "Tie"

            return {
                "Model A": {"SSIM": round(ssim_a, 4), "PSNR": round(psnr_a, 2)},
                "Model B": {"SSIM": round(ssim_b, 4), "PSNR": round(psnr_b, 2)},
                "Winner": winner,
            }
        except ImportError:
            return {
                "note": "Install scikit-image for metrics: pip install scikit-image"
            }
        except Exception as e:
            return {"error": str(e)}

    def _unload_all(self) -> tuple[str, str]:
        """Unload both models and free memory."""
        try:
            import torch

            if self._model_a is not None:
                del self._model_a
                self._model_a = None

            if self._model_b is not None:
                del self._model_b
                self._model_b = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        no_model = self.i18n.t("status.no_model")
        return no_model, no_model
