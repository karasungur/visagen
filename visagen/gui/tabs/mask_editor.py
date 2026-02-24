"""
Mask Editor Tab.

Main tab for face segmentation mask editing, LoRA training,
and batch application of trained models.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import gradio as gr
import numpy as np

from visagen.data.mask_dataset import save_training_sample
from visagen.gui.components.mask_canvas import MaskCanvas
from visagen.gui.tabs.base import BaseTab
from visagen.vision.mask_ops import MaskOperations, MaskRefinementConfig
from visagen.vision.segmenter import FaceSegmenter

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state import AppState

logger = logging.getLogger(__name__)


class MaskEditorTab(BaseTab):
    """
    Mask Editor tab with three sub-tabs:
    1. Edit Masks - Interactive mask editing with component toggles
    2. LoRA Training - Fine-tune SegFormer with edited masks
    3. Batch Apply - Apply fine-tuned model to entire faceset

    Provides workflow for:
    - Loading faceset and browsing faces
    - Editing masks with brush and component selection
    - Saving edited masks as training samples
    - Training LoRA adapters (~2-5 min for 10-20 samples)
    - Batch application to full faceset
    """

    @property
    def id(self) -> str:
        return "mask_editor"

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        super().__init__(app_state, i18n)
        self._segmenter: FaceSegmenter | None = None
        self._canvas: MaskCanvas | None = None
        self._face_files: list[Path] = []
        self._current_face_idx: int = 0
        self._gallery_page: int = 0
        self._gallery_page_size: int = 18
        self._samples_count: int = 0
        self._training_thread: threading.Thread | None = None
        self._trainer: Any | None = None
        self._training_active: bool = False
        self._training_stop_timeout_sec: float = 5.0
        self._pending_batch_request: dict[str, Any] | None = None
        self._batch_lock = threading.Lock()

    @property
    def segmenter(self) -> FaceSegmenter:
        """Lazy-loaded segmenter."""
        if self._segmenter is None:
            self._segmenter = FaceSegmenter()
        return self._segmenter

    def _build_content(self) -> dict[str, Any]:
        """Build tab content with three sub-tabs."""
        components: dict[str, Any] = {}

        with gr.Tabs():
            # Sub-tab 1: Edit Masks
            with gr.Tab(self.t("tabs.editor")):
                components.update(self._build_editor_tab())

            # Sub-tab 2: LoRA Training
            with gr.Tab(self.t("tabs.training")):
                components.update(self._build_training_tab())

            # Sub-tab 3: Batch Apply
            with gr.Tab(self.t("tabs.batch")):
                components.update(self._build_batch_tab())

        return components

    def _build_editor_tab(self) -> dict[str, Any]:
        """Build the Edit Masks sub-tab."""
        components: dict[str, Any] = {}

        with gr.Row():
            # Left panel - Faceset browser
            with gr.Column(scale=1):
                components["faceset_dir"] = gr.Textbox(
                    label=self.t("editor.faceset_dir"),
                    placeholder="./workspace/data_src/aligned",
                )
                with gr.Row():
                    components["load_faceset_btn"] = gr.Button(
                        self.i18n.t("common.load"), size="sm"
                    )
                    components["refresh_faceset_btn"] = gr.Button(
                        self.i18n.t("common.refresh"), size="sm"
                    )

                components["face_gallery"] = gr.Gallery(
                    label=self.t("editor.faces"),
                    columns=3,
                    rows=3,
                    height=300,
                    allow_preview=False,
                )

                with gr.Row():
                    components["gallery_prev_btn"] = gr.Button("<< Page", size="sm")
                    components["gallery_page_info"] = gr.Textbox(
                        value="Page 0/0",
                        interactive=False,
                        show_label=False,
                    )
                    components["gallery_next_btn"] = gr.Button("Page >>", size="sm")

                with gr.Row():
                    components["prev_face_btn"] = gr.Button("<< Prev", size="sm")
                    components["face_info"] = gr.Textbox(
                        label=self.t("editor.selected"),
                        interactive=False,
                        lines=1,
                    )
                    components["next_face_btn"] = gr.Button("Next >>", size="sm")

            # Right panel - Canvas
            with gr.Column(scale=2):
                # Create canvas component
                self._canvas = MaskCanvas(self.i18n, self._segmenter)
                canvas_components = self._canvas.build()
                components.update(
                    {f"canvas_{k}": v for k, v in canvas_components.items()}
                )

        # Action buttons
        with gr.Row():
            components["save_mask_btn"] = gr.Button(
                self.t("actions.save_mask"),
                variant="secondary",
            )
            components["add_sample_btn"] = gr.Button(
                self.t("actions.save_sample"),
                variant="primary",
            )
            components["samples_count"] = gr.Textbox(
                label=self.t("editor.samples_count"),
                value="0",
                interactive=False,
                scale=0,
            )

        components["editor_status"] = gr.Textbox(
            label=self.t("editor.status"),
            interactive=False,
        )

        return components

    def _build_training_tab(self) -> dict[str, Any]:
        """Build the LoRA Training sub-tab."""
        components: dict[str, Any] = {}

        with gr.Row():
            with gr.Column():
                components["samples_dir"] = gr.Textbox(
                    label=self.t("lora.samples_dir"),
                    placeholder="./workspace/mask_samples",
                    value="./workspace/mask_samples",
                )
                components["lora_output_dir"] = gr.Textbox(
                    label=self.t("lora.output_dir"),
                    placeholder="./workspace/lora",
                    value="./workspace/lora",
                )

            with gr.Column():
                components["lora_epochs"] = gr.Slider(
                    label=self.t("lora.epochs"),
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                )
                components["lora_rank"] = gr.Slider(
                    label=self.t("lora.rank"),
                    minimum=2,
                    maximum=32,
                    value=8,
                    step=2,
                )
                components["lora_lr"] = gr.Number(
                    label=self.t("lora.learning_rate"),
                    value=0.0001,
                    precision=6,
                )

        with gr.Row():
            components["start_training_btn"] = gr.Button(
                self.t("lora.start"),
                variant="primary",
            )
            components["stop_training_btn"] = gr.Button(
                self.t("lora.stop"),
                variant="stop",
            )

        components["training_progress"] = gr.Textbox(
            label=self.t("lora.progress"),
            interactive=False,
            lines=3,
        )

        components["training_log"] = gr.Textbox(
            label=self.t("lora.log"),
            interactive=False,
            lines=10,
        )

        return components

    def _build_batch_tab(self) -> dict[str, Any]:
        """Build the Batch Apply sub-tab."""
        components: dict[str, Any] = {}

        with gr.Row():
            with gr.Column():
                components["batch_input_dir"] = gr.Textbox(
                    label=self.t("batch.input_dir"),
                    placeholder="./workspace/data_src/aligned",
                )
                components["batch_output_dir"] = gr.Textbox(
                    label=self.t("batch.output_dir"),
                    placeholder="./workspace/data_src/aligned_masked",
                )

            with gr.Column():
                components["use_lora"] = gr.Checkbox(
                    label=self.t("batch.use_lora"),
                    value=True,
                )
                components["lora_weights_path"] = gr.Textbox(
                    label=self.t("batch.lora_weights"),
                    placeholder="./workspace/lora/segformer_lora.pt",
                )

        # Component toggles for batch
        with gr.Accordion(self.t("batch.components"), open=True):
            with gr.Row():
                components["batch_skin"] = gr.Checkbox(label="Skin", value=True)
                components["batch_nose"] = gr.Checkbox(label="Nose", value=True)
                components["batch_eyes"] = gr.Checkbox(label="Eyes", value=True)
                components["batch_mouth"] = gr.Checkbox(label="Mouth", value=True)
                components["batch_hair"] = gr.Checkbox(label="Hair", value=False)

        # Refinement for batch
        with gr.Accordion(self.t("batch.refinement"), open=False):
            with gr.Row():
                components["batch_erode"] = gr.Slider(
                    label=self.t("mask_editor.refine.erode"),
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                )
                components["batch_dilate"] = gr.Slider(
                    label=self.t("mask_editor.refine.dilate"),
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                )
                components["batch_blur"] = gr.Slider(
                    label=self.t("mask_editor.refine.blur"),
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=2,
                )

        components["preview_batch"] = gr.Checkbox(
            label=self.t("batch.preview_before_save"),
            value=True,
        )

        with gr.Row():
            components["apply_batch_btn"] = gr.Button(
                self.t("batch.apply"),
                variant="primary",
            )
            components["confirm_batch_btn"] = gr.Button(
                self.t("batch.confirm"),
                variant="secondary",
                visible=False,
            )
            components["cancel_batch_btn"] = gr.Button(
                self.t("batch.cancel"),
                visible=False,
            )

        # Preview gallery
        components["batch_preview_gallery"] = gr.Gallery(
            label=self.t("batch.preview_gallery"),
            columns=4,
            rows=2,
            height=300,
        )

        components["batch_progress"] = gr.Textbox(
            label=self.t("batch.progress"),
            interactive=False,
        )

        with gr.Accordion(self.t("annotation.title"), open=False):
            with gr.Row():
                components["annotation_input_dir"] = gr.Textbox(
                    label=self.t("annotation.input_dir"),
                    placeholder="./workspace/data_src/aligned",
                )
                components["annotation_output_path"] = gr.Textbox(
                    label=self.t("annotation.output_path"),
                    placeholder="./workspace/mask_annotations",
                )

            with gr.Row():
                components["annotation_format"] = gr.Dropdown(
                    label=self.t("annotation.format"),
                    choices=["labelme", "coco"],
                    value="labelme",
                )
                components["annotation_include_image_data"] = gr.Checkbox(
                    label=self.t("annotation.include_image_data"),
                    value=False,
                )
                components["annotation_min_area"] = gr.Slider(
                    label=self.t("annotation.min_area"),
                    minimum=0,
                    maximum=5000,
                    value=100,
                    step=10,
                )

            components["annotation_export_btn"] = gr.Button(
                self.t("annotation.export"),
                variant="secondary",
            )

            with gr.Row():
                components["annotation_import_path"] = gr.Textbox(
                    label=self.t("annotation.import_path"),
                    placeholder="./workspace/mask_annotations",
                )
                components["annotation_import_output_dir"] = gr.Textbox(
                    label=self.t("annotation.import_output_dir"),
                    placeholder="./workspace/data_src/aligned",
                )
                components["annotation_import_format"] = gr.Dropdown(
                    label=self.t("annotation.import_format"),
                    choices=["labelme", "coco"],
                    value="labelme",
                )

            components["annotation_import_btn"] = gr.Button(
                self.t("annotation.import"),
                variant="secondary",
            )
            components["annotation_status"] = gr.Textbox(
                label=self.t("annotation.status"),
                interactive=False,
                lines=2,
            )

        return components

    def _setup_events(self, components: dict[str, Any]) -> None:
        """Set up all event handlers."""
        # Editor tab events
        self._setup_editor_events(components)

        # Training tab events
        self._setup_training_events(components)

        # Batch tab events
        self._setup_batch_events(components)

    def _setup_editor_events(self, c: dict[str, Any]) -> None:
        """Set up editor tab events."""
        # Load faceset
        c["load_faceset_btn"].click(
            fn=self._load_faceset,
            inputs=[c["faceset_dir"]],
            outputs=[
                c["face_gallery"],
                c["gallery_page_info"],
                c["face_info"],
                c["editor_status"],
            ],
        )

        c["refresh_faceset_btn"].click(
            fn=self._load_faceset,
            inputs=[c["faceset_dir"]],
            outputs=[
                c["face_gallery"],
                c["gallery_page_info"],
                c["face_info"],
                c["editor_status"],
            ],
        )

        c["gallery_prev_btn"].click(
            fn=self._prev_gallery_page,
            outputs=[c["face_gallery"], c["gallery_page_info"]],
        )

        c["gallery_next_btn"].click(
            fn=self._next_gallery_page,
            outputs=[c["face_gallery"], c["gallery_page_info"]],
        )

        # Navigation
        c["prev_face_btn"].click(
            fn=self._prev_face,
            outputs=[
                c["canvas_editor"],
                c["canvas_preview"],
                c["canvas_mask_only"],
                c["face_info"],
            ],
        )

        c["next_face_btn"].click(
            fn=self._next_face,
            outputs=[
                c["canvas_editor"],
                c["canvas_preview"],
                c["canvas_mask_only"],
                c["face_info"],
            ],
        )

        # Gallery selection
        c["face_gallery"].select(
            fn=self._on_face_select,
            outputs=[
                c["canvas_editor"],
                c["canvas_preview"],
                c["canvas_mask_only"],
                c["face_info"],
            ],
        )

        # Save mask to image
        c["save_mask_btn"].click(
            fn=self._save_mask_to_image,
            outputs=[c["editor_status"]],
        )

        # Add to training set
        c["add_sample_btn"].click(
            fn=self._add_to_training_set,
            inputs=[c["faceset_dir"]],
            outputs=[c["samples_count"], c["editor_status"]],
        )

        # Canvas events (delegated to canvas component)
        if self._canvas is not None:
            canvas_comps = {k[7:]: v for k, v in c.items() if k.startswith("canvas_")}
            self._canvas.setup_events(canvas_comps)

    def _setup_training_events(self, c: dict[str, Any]) -> None:
        """Set up training tab events."""
        c["start_training_btn"].click(
            fn=self._start_training,
            inputs=[
                c["samples_dir"],
                c["lora_output_dir"],
                c["lora_epochs"],
                c["lora_rank"],
                c["lora_lr"],
            ],
            outputs=[c["training_progress"], c["training_log"]],
        )

        c["stop_training_btn"].click(
            fn=self._stop_training,
            outputs=[c["training_progress"]],
        )

    def _setup_batch_events(self, c: dict[str, Any]) -> None:
        """Set up batch tab events."""
        c["apply_batch_btn"].click(
            fn=self._apply_batch,
            inputs=[
                c["batch_input_dir"],
                c["batch_output_dir"],
                c["use_lora"],
                c["lora_weights_path"],
                c["batch_skin"],
                c["batch_nose"],
                c["batch_eyes"],
                c["batch_mouth"],
                c["batch_hair"],
                c["batch_erode"],
                c["batch_dilate"],
                c["batch_blur"],
                c["preview_batch"],
            ],
            outputs=[
                c["batch_preview_gallery"],
                c["batch_progress"],
                c["apply_batch_btn"],
                c["confirm_batch_btn"],
                c["cancel_batch_btn"],
            ],
        )

        c["confirm_batch_btn"].click(
            fn=self._confirm_batch,
            outputs=[
                c["batch_preview_gallery"],
                c["batch_progress"],
                c["apply_batch_btn"],
                c["confirm_batch_btn"],
                c["cancel_batch_btn"],
            ],
        )

        c["cancel_batch_btn"].click(
            fn=self._cancel_batch,
            outputs=[
                c["batch_preview_gallery"],
                c["batch_progress"],
                c["apply_batch_btn"],
                c["confirm_batch_btn"],
                c["cancel_batch_btn"],
            ],
        )

        c["annotation_export_btn"].click(
            fn=self._export_annotations,
            inputs=[
                c["annotation_input_dir"],
                c["annotation_output_path"],
                c["annotation_format"],
                c["annotation_include_image_data"],
                c["annotation_min_area"],
            ],
            outputs=[c["annotation_status"]],
        )

        c["annotation_import_btn"].click(
            fn=self._import_annotations,
            inputs=[
                c["annotation_import_path"],
                c["annotation_import_output_dir"],
                c["annotation_import_format"],
            ],
            outputs=[c["annotation_status"]],
        )

    # ============ Editor Handlers ============

    def _load_faceset(
        self,
        directory: str,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str, str]:
        """Load faceset from directory."""
        if not directory:
            return [], "Page 0/0", "", "Please specify a directory"

        dir_path = Path(directory)
        if not dir_path.exists():
            return [], "Page 0/0", "", f"Directory not found: {directory}"

        # Find face images
        self._face_files = sorted(
            list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")),
            key=lambda p: p.name,
        )

        if not self._face_files:
            return [], "Page 0/0", "", "No images found in directory"

        self._gallery_page = 0
        self._current_face_idx = 0
        gallery_items, page_info = self._get_gallery_page()
        status = f"Loaded {len(self._face_files)} faces"
        face_info = self._face_files[0].name if self._face_files else ""

        return gallery_items, page_info, face_info, status

    def _get_gallery_page(self) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Get current gallery page."""
        start = self._gallery_page * self._gallery_page_size
        end = start + self._gallery_page_size
        page_files = self._face_files[start:end]

        gallery_items: list[tuple[np.ndarray, str]] = []
        for path in page_files:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > 256:
                    scale = 256 / max_dim
                    img = cv2.resize(
                        img,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gallery_items.append((img_rgb, path.stem))
            except Exception:
                continue

        total_pages = max(
            1,
            (len(self._face_files) + self._gallery_page_size - 1)
            // self._gallery_page_size,
        )
        page_info = f"Page {self._gallery_page + 1}/{total_pages}"
        return gallery_items, page_info

    def _prev_gallery_page(self) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Go to previous gallery page."""
        if self._gallery_page > 0:
            self._gallery_page -= 1
        return self._get_gallery_page()

    def _next_gallery_page(self) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Go to next gallery page."""
        total_pages = max(
            1,
            (len(self._face_files) + self._gallery_page_size - 1)
            // self._gallery_page_size,
        )
        if self._gallery_page < total_pages - 1:
            self._gallery_page += 1
        return self._get_gallery_page()

    def _prev_face(
        self,
    ) -> tuple[dict, np.ndarray | None, np.ndarray | None, str]:
        """Navigate to previous face."""
        if not self._face_files:
            return {}, None, None, ""

        self._current_face_idx = max(0, self._current_face_idx - 1)
        return self._load_current_face()

    def _next_face(
        self,
    ) -> tuple[dict, np.ndarray | None, np.ndarray | None, str]:
        """Navigate to next face."""
        if not self._face_files:
            return {}, None, None, ""

        self._current_face_idx = min(
            len(self._face_files) - 1, self._current_face_idx + 1
        )
        return self._load_current_face()

    def _on_face_select(
        self,
        evt: gr.SelectData,
    ) -> tuple[dict, np.ndarray | None, np.ndarray | None, str]:
        """Handle gallery face selection."""
        if evt.index is not None:
            page_start = self._gallery_page * self._gallery_page_size
            file_idx = page_start + evt.index
            if file_idx < len(self._face_files):
                self._current_face_idx = file_idx
        return self._load_current_face()

    def _load_current_face(
        self,
    ) -> tuple[dict, np.ndarray | None, np.ndarray | None, str]:
        """Load current face into canvas."""
        if not self._face_files or self._current_face_idx >= len(self._face_files):
            return {}, None, None, ""

        face_path = self._face_files[self._current_face_idx]
        face_image = cv2.imread(str(face_path))

        if face_image is None:
            return {}, None, None, f"Failed to load: {face_path.name}"

        if self._canvas is None:
            return {}, None, None, "Canvas not initialized"

        # Load into canvas
        editor_data, preview, mask = self._canvas.load_face(face_image)

        face_info = (
            f"{face_path.name} ({self._current_face_idx + 1}/{len(self._face_files)})"
        )

        return editor_data, preview, mask, face_info

    def _save_mask_to_image(self) -> str:
        """Save current mask to the face image's xseg_mask."""
        if self._canvas is None:
            return "Canvas not initialized"

        mask = self._canvas.get_current_mask()
        if mask is None:
            return "No mask to save"

        if not self._face_files or self._current_face_idx >= len(self._face_files):
            return "No face selected"

        face_path = self._face_files[self._current_face_idx]

        try:
            from visagen.vision.face_image import FaceImage

            image, metadata = FaceImage.load(face_path)
            if metadata is None:
                return "No face metadata in image"

            # Convert mask to float and set
            mask_float = mask.astype(np.float32) / 255.0
            if len(mask_float.shape) == 2:
                mask_float = mask_float[..., np.newaxis]

            FaceImage.set_xseg_mask(metadata, mask_float)
            FaceImage.save(face_path, image, metadata)

            return f"Mask saved to: {face_path.name}"

        except Exception as e:
            return f"Error saving mask: {e}"

    def _add_to_training_set(self, faceset_dir: str) -> tuple[str, str]:
        """Add current face+mask to training set."""
        if self._canvas is None:
            return str(self._samples_count), "Canvas not initialized"

        image = self._canvas.get_current_image()
        mask = self._canvas.get_current_mask()

        if image is None or mask is None:
            return str(self._samples_count), "No image or mask to save"

        # Determine samples directory
        samples_dir = Path(faceset_dir).parent / "mask_samples"

        try:
            # Create sample ID
            self._samples_count += 1
            sample_id = f"sample_{self._samples_count:04d}"

            # Save sample
            save_training_sample(samples_dir, image, mask, sample_id)

            return str(self._samples_count), f"Added sample: {sample_id}"

        except Exception as e:
            return str(self._samples_count), f"Error saving sample: {e}"

    # ============ Training Handlers ============

    def _start_training(
        self,
        samples_dir: str,
        output_dir: str,
        epochs: int,
        rank: int,
        lr: float,
    ) -> tuple[str, str]:
        """Start LoRA training."""
        if self._training_active:
            return "Training already in progress", ""

        samples_path = Path(samples_dir)
        if not samples_path.exists():
            return "Samples directory not found", ""

        images_dir = samples_path / "images"
        if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
            return "No training images found", ""

        try:
            from visagen.training.segformer_finetune import (
                SegFormerFinetuneConfig,
                SegFormerTrainer,
            )

            config = SegFormerFinetuneConfig(
                max_epochs=int(epochs),
                lora_rank=int(rank),
                learning_rate=lr,
            )

            trainer = SegFormerTrainer(config)
            self._trainer = trainer

            log_messages = []

            def progress_callback(epoch: int, max_epochs: int, loss: float) -> None:
                log_messages.append(f"Epoch {epoch}/{max_epochs} - Loss: {loss:.4f}")

            self._training_active = True

            # Run in thread
            def train_fn():
                try:
                    trainer.train(
                        samples_dir=samples_dir,
                        output_dir=output_dir,
                        progress_callback=progress_callback,
                    )
                except Exception as e:
                    log_messages.append(f"Error: {e}")
                finally:
                    self._training_active = False
                    self._trainer = None
                    self._training_thread = None

            self._training_thread = threading.Thread(
                target=train_fn,
                name="segformer_train_thread",
                daemon=True,
            )
            self._training_thread.start()

            return "Training started...", "\n".join(log_messages)

        except Exception as e:
            return f"Error starting training: {e}", ""

    def _stop_training(self) -> str:
        """Stop training."""
        if self._trainer is not None:
            try:
                self._trainer.stop()
            except Exception as e:
                logger.debug(f"Failed to stop trainer cleanly: {e}")

        if self._training_thread is not None and self._training_thread.is_alive():
            self._training_thread.join(timeout=self._training_stop_timeout_sec)
            if self._training_thread.is_alive():
                self._training_active = False
                return "Training stop requested (thread is still shutting down)"

        self._training_active = False
        return "Training stopped"

    # ============ Batch Handlers ============

    def _apply_batch(
        self,
        input_dir: str,
        output_dir: str,
        use_lora: bool,
        lora_weights: str,
        skin: bool,
        nose: bool,
        eyes: bool,
        mouth: bool,
        hair: bool,
        erode: int,
        dilate: int,
        blur: int,
        preview: bool,
    ) -> tuple[
        list[tuple[np.ndarray, str]],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Apply segmentation to batch of faces."""
        if not self._batch_lock.acquire(blocking=False):
            return (
                [],
                "Batch operation already in progress",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        try:
            if self._pending_batch_request is not None:
                return (
                    [],
                    "Pending preview exists. Confirm or cancel before applying again.",
                    gr.update(interactive=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            input_path = Path(input_dir)
            if not input_path.exists():
                return (
                    [],
                    "Input directory not found",
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            files = self._collect_batch_files(input_path)
            if not files:
                return (
                    [],
                    "No images found",
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            selected_components = self._build_batch_components(
                skin, nose, eyes, mouth, hair
            )
            if not selected_components:
                return (
                    [],
                    "No mask components selected",
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            resolved_output = (
                Path(output_dir)
                if output_dir
                else input_path.parent / f"{input_path.name}_masked"
            )
            request = {
                "input_path": input_path,
                "output_path": resolved_output,
                "files": files,
                "use_lora": use_lora,
                "lora_weights": lora_weights,
                "components": selected_components,
                "erode": erode,
                "dilate": dilate,
                "blur": blur,
            }

            preview_items, preview_errors = self._generate_batch_preview(
                files,
                use_lora=use_lora,
                lora_weights=lora_weights,
                selected_components=selected_components,
                erode=erode,
                dilate=dilate,
                blur=blur,
            )

            if preview:
                self._pending_batch_request = request
                msg = (
                    f"Preview ready ({len(preview_items)}/{len(files)}). "
                    f"Confirm to save results to: {resolved_output}"
                )
                if preview_errors > 0:
                    msg += f" | Skipped {preview_errors} files during preview"
                return (
                    preview_items,
                    msg,
                    gr.update(interactive=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            self._pending_batch_request = None
            saved, failed = self._save_batch_request(request)
            msg = (
                f"Saved {saved}/{len(files)} images to {resolved_output}"
                if failed == 0
                else f"Saved {saved}/{len(files)} images to {resolved_output} | Failed: {failed}"
            )
            return (
                preview_items,
                msg,
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        finally:
            self._batch_lock.release()

    def _confirm_batch(
        self,
    ) -> tuple[
        list[tuple[np.ndarray, str]],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Confirm and save pending batch request."""
        if not self._batch_lock.acquire(blocking=False):
            return (
                [],
                "Batch operation already in progress",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        try:
            request = self._pending_batch_request
            if request is None:
                return (
                    [],
                    "No pending batch preview to confirm",
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            self._pending_batch_request = None
            saved, failed = self._save_batch_request(request)
            preview_items, _ = self._generate_batch_preview(
                request["files"],
                use_lora=request["use_lora"],
                lora_weights=request["lora_weights"],
                selected_components=request["components"],
                erode=request["erode"],
                dilate=request["dilate"],
                blur=request["blur"],
            )
            total = len(request["files"])
            output_path = request["output_path"]
            msg = (
                f"Saved {saved}/{total} images to {output_path}"
                if failed == 0
                else f"Saved {saved}/{total} images to {output_path} | Failed: {failed}"
            )
            return (
                preview_items,
                msg,
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        finally:
            self._batch_lock.release()

    def _cancel_batch(
        self,
    ) -> tuple[
        list[tuple[np.ndarray, str]],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Cancel pending batch confirmation."""
        if not self._batch_lock.acquire(blocking=False):
            return (
                [],
                "Batch operation already in progress",
                gr.update(interactive=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        try:
            self._pending_batch_request = None
            return (
                [],
                "Batch preview cancelled",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        finally:
            self._batch_lock.release()

    @staticmethod
    def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Normalize input mask to uint8 binary format {0, 255}."""
        if mask.ndim == 3:
            mask = mask[..., 0]

        if mask.dtype != np.uint8:
            mask = mask.astype(np.float32)
            max_val = float(mask.max()) if mask.size > 0 else 0.0
            if max_val <= 1.0:
                mask = mask * 255.0
            mask = np.clip(mask, 0.0, 255.0).astype(np.uint8)

        return cast(np.ndarray, np.where(mask > 127, 255, 0).astype(np.uint8))

    def _load_mask_for_export(self, image_path: Path) -> np.ndarray | None:
        """Load mask from sidecar file or embedded face metadata."""
        sidecar = image_path.parent / f"{image_path.stem}_mask.png"
        if sidecar.exists():
            sidecar_mask = cv2.imread(str(sidecar), cv2.IMREAD_GRAYSCALE)
            if sidecar_mask is not None:
                return self._normalize_binary_mask(sidecar_mask)

        if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
            return None

        try:
            from visagen.vision.face_image import FaceImage

            _image, metadata = FaceImage.load(image_path)
            if metadata is None:
                return None
            mask = FaceImage.get_xseg_mask(metadata)
            if mask is None:
                return None
            return self._normalize_binary_mask(mask)
        except Exception as e:
            logger.debug("Failed to load face mask from %s: %s", image_path, e)
            return None

    def _save_imported_mask(
        self,
        *,
        image_name: str,
        output_dir: Path,
        mask: np.ndarray,
    ) -> None:
        """Persist imported mask as sidecar PNG and embed in face metadata when possible."""
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_mask = self._normalize_binary_mask(mask)
        stem = Path(image_name).stem

        sidecar = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(sidecar), normalized_mask)

        candidate_image = output_dir / Path(image_name).name
        if candidate_image.suffix.lower() not in {".jpg", ".jpeg"}:
            return
        if not candidate_image.exists():
            return

        try:
            from visagen.vision.face_image import FaceImage

            image, metadata = FaceImage.load(candidate_image)
            if metadata is None:
                return
            FaceImage.set_xseg_mask(
                metadata, normalized_mask.astype(np.float32) / 255.0
            )
            FaceImage.save(candidate_image, image, metadata)
        except Exception as e:
            logger.debug(
                "Failed to embed imported mask into %s metadata: %s",
                candidate_image,
                e,
            )

    def _export_annotations(
        self,
        input_dir: str,
        output_path: str,
        format_name: str,
        include_image_data: bool,
        min_area: int,
    ) -> str:
        """Export masks to LabelMe or COCO annotations."""
        input_path = Path(input_dir)
        if not input_path.exists():
            return f"Input directory not found: {input_dir}"

        files = self._collect_batch_files(input_path)
        if not files:
            return "No images found for annotation export"

        masks: dict[str, np.ndarray] = {}
        for file_path in files:
            mask = self._load_mask_for_export(file_path)
            if mask is not None:
                masks[file_path.name] = mask

        if not masks:
            return "No masks found in input directory (metadata or *_mask.png)"

        min_area = int(min_area)
        try:
            if format_name == "labelme":
                from visagen.vision.mask_export import export_labelme

                out_dir = (
                    Path(output_path)
                    if output_path
                    else input_path / "mask_annotations_labelme"
                )
                if out_dir.suffix.lower() == ".json":
                    out_dir = out_dir.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                exported = 0
                for image_path in files:
                    mask = masks.get(image_path.name)
                    if mask is None:
                        continue
                    export_labelme(
                        image_path=image_path,
                        mask=mask,
                        output_path=out_dir / f"{image_path.stem}.json",
                        include_image_data=include_image_data,
                        min_area=min_area,
                    )
                    exported += 1
                return f"Exported {exported} LabelMe annotation files to {out_dir}"

            if format_name == "coco":
                from visagen.vision.mask_export import export_coco

                out_file = (
                    Path(output_path)
                    if output_path
                    else input_path / "annotations.json"
                )
                if out_file.suffix.lower() != ".json":
                    out_file = out_file / "annotations.json"
                ordered_paths = [p for p in files if p.name in masks]
                ordered_masks = [masks[p.name] for p in ordered_paths]
                export_coco(
                    image_paths=ordered_paths,
                    masks=ordered_masks,
                    output_path=out_file,
                    categories=["face"],
                    min_area=min_area,
                )
                return f"Exported COCO annotations for {len(ordered_paths)} images to {out_file}"

            return f"Unsupported annotation format: {format_name}"
        except Exception as e:
            return f"Annotation export failed: {e}"

    def _import_annotations(
        self,
        annotation_path: str,
        output_dir: str,
        format_name: str,
    ) -> str:
        """Import LabelMe/COCO annotations and write masks."""
        path = Path(annotation_path)
        if not path.exists():
            return f"Annotation path not found: {annotation_path}"

        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if format_name == "labelme":
                from visagen.vision.mask_export import import_labelme

                json_files = [path] if path.is_file() else sorted(path.glob("*.json"))
                if not json_files:
                    return f"No LabelMe JSON files found in {path}"

                imported = 0
                failed = 0
                for json_file in json_files:
                    try:
                        mask, metadata = import_labelme(json_file)
                        image_name = (
                            metadata.get("image_path") or f"{json_file.stem}.jpg"
                        )
                        self._save_imported_mask(
                            image_name=image_name,
                            output_dir=out_dir,
                            mask=mask,
                        )
                        imported += 1
                    except Exception as e:
                        logger.warning("LabelMe import failed for %s: %s", json_file, e)
                        failed += 1

                if failed:
                    return f"Imported {imported} LabelMe masks to {out_dir} | Failed: {failed}"
                return f"Imported {imported} LabelMe masks to {out_dir}"

            if format_name == "coco":
                from visagen.vision.mask_export import import_coco

                if path.is_dir():
                    path = path / "annotations.json"
                if not path.exists():
                    return f"COCO annotation file not found: {path}"

                imported = 0
                for image_name, (mask, _labels) in import_coco(path).items():
                    self._save_imported_mask(
                        image_name=image_name,
                        output_dir=out_dir,
                        mask=mask,
                    )
                    imported += 1
                return f"Imported {imported} COCO masks to {out_dir}"

            return f"Unsupported annotation format: {format_name}"
        except Exception as e:
            return f"Annotation import failed: {e}"

    @staticmethod
    def _collect_batch_files(input_path: Path) -> list[Path]:
        """Collect supported image files for batch operations."""
        return sorted(
            list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
            + list(input_path.glob("*.png"))
        )

    @staticmethod
    def _build_batch_components(
        skin: bool,
        nose: bool,
        eyes: bool,
        mouth: bool,
        hair: bool,
    ) -> set[str]:
        """Build selected segmentation component set."""
        components: set[str] = set()
        if skin:
            components.add("skin")
        if nose:
            components.add("nose")
        if eyes:
            components.update({"left_eye", "right_eye", "left_brow", "right_brow"})
        if mouth:
            components.update({"mouth", "upper_lip", "lower_lip"})
        if hair:
            components.add("hair")
        return components

    def _build_batch_mask(
        self,
        image: np.ndarray,
        *,
        segmenter: Any,
        selected_components: set[str],
        erode: int,
        dilate: int,
        blur: int,
    ) -> np.ndarray:
        """Build refined mask from parsed segmentation output."""
        parsing_dict = segmenter.get_parsing(image)
        h, w = image.shape[:2]
        parsing = np.zeros((h, w), dtype=np.uint8)

        from visagen.vision.segmenter import LABEL_TO_ID

        for name, parsed_mask in parsing_dict.items():
            if name in LABEL_TO_ID:
                parsing[parsed_mask > 127] = LABEL_TO_ID[name]

        refine_config = MaskRefinementConfig(
            erode_size=erode,
            dilate_size=dilate,
            blur_size=blur,
        )
        mask = MaskOperations.combine_component_masks(parsing, selected_components)
        return cast(np.ndarray, MaskOperations.refine(mask, refine_config))

    @staticmethod
    def _build_overlay_preview(
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Create RGB overlay preview for mask visualization."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()
        overlay[mask > 127, 1] = np.clip(
            overlay[mask > 127, 1].astype(np.int32) + 80,
            0,
            255,
        ).astype(np.uint8)
        return cast(np.ndarray, overlay)

    def _resolve_batch_segmenter(
        self,
        *,
        use_lora: bool,
        lora_weights: str,
    ) -> Any:
        """Resolve default or LoRA-enabled segmenter."""
        if use_lora and lora_weights and Path(lora_weights).exists():
            from visagen.vision.segmenter_lora import FaceSegmenterLoRA

            return FaceSegmenterLoRA(lora_weights=lora_weights)
        return self.segmenter

    def _generate_batch_preview(
        self,
        files: list[Path],
        *,
        use_lora: bool,
        lora_weights: str,
        selected_components: set[str],
        erode: int,
        dilate: int,
        blur: int,
    ) -> tuple[list[tuple[np.ndarray, str]], int]:
        """Generate preview overlays for the first batch samples."""
        segmenter = self._resolve_batch_segmenter(
            use_lora=use_lora,
            lora_weights=lora_weights,
        )
        preview_items: list[tuple[np.ndarray, str]] = []
        errors = 0
        for file_path in files[:8]:
            try:
                image = cv2.imread(str(file_path))
                if image is None:
                    errors += 1
                    continue
                mask = self._build_batch_mask(
                    image,
                    segmenter=segmenter,
                    selected_components=selected_components,
                    erode=erode,
                    dilate=dilate,
                    blur=blur,
                )
                preview_items.append(
                    (self._build_overlay_preview(image, mask), file_path.stem)
                )
            except Exception as e:
                logger.warning("Batch preview failed for %s: %s", file_path, e)
                errors += 1
        return preview_items, errors

    def _save_batch_result(
        self,
        *,
        source_path: Path,
        output_dir: Path,
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Save batch output with face mask when possible, fallback to sidecar mask."""
        output_image = output_dir / source_path.name
        try:
            from visagen.vision.face_image import FaceImage

            if source_path.suffix.lower() in {".jpg", ".jpeg"}:
                src_image, metadata = FaceImage.load(source_path)
                if metadata is not None:
                    mask_float = mask.astype(np.float32) / 255.0
                    if mask_float.ndim == 2:
                        mask_float = mask_float[..., np.newaxis]
                    FaceImage.set_xseg_mask(metadata, mask_float)
                    FaceImage.save(output_image, src_image, metadata)
                    return
        except Exception as e:
            logger.debug("Falling back to sidecar mask for %s: %s", source_path, e)

        cv2.imwrite(str(output_image), image_bgr)
        mask_output = output_dir / f"{source_path.stem}_mask.png"
        cv2.imwrite(str(mask_output), mask)

    def _save_batch_request(self, request: dict[str, Any]) -> tuple[int, int]:
        """Run full batch processing and persist outputs."""
        files: list[Path] = request["files"]
        output_path: Path = request["output_path"]
        output_path.mkdir(parents=True, exist_ok=True)

        segmenter = self._resolve_batch_segmenter(
            use_lora=request["use_lora"],
            lora_weights=request["lora_weights"],
        )

        saved = 0
        failed = 0
        for file_path in files:
            try:
                image = cv2.imread(str(file_path))
                if image is None:
                    failed += 1
                    continue
                mask = self._build_batch_mask(
                    image,
                    segmenter=segmenter,
                    selected_components=request["components"],
                    erode=request["erode"],
                    dilate=request["dilate"],
                    blur=request["blur"],
                )
                self._save_batch_result(
                    source_path=file_path,
                    output_dir=output_path,
                    image_bgr=image,
                    mask=mask,
                )
                saved += 1
            except Exception as e:
                logger.warning("Batch apply failed for %s: %s", file_path, e)
                failed += 1

        return saved, failed
