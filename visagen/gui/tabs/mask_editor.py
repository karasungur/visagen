"""
Mask Editor Tab.

Main tab for face segmentation mask editing, LoRA training,
and batch application of trained models.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        self._samples_count: int = 0
        self._training_thread: threading.Thread | None = None
        self._training_active: bool = False

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
            outputs=[c["face_gallery"], c["face_info"], c["editor_status"]],
        )

        c["refresh_faceset_btn"].click(
            fn=self._load_faceset,
            inputs=[c["faceset_dir"]],
            outputs=[c["face_gallery"], c["face_info"], c["editor_status"]],
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
            outputs=[c["batch_preview_gallery"], c["batch_progress"]],
        )

    # ============ Editor Handlers ============

    def _load_faceset(
        self,
        directory: str,
    ) -> tuple[list[tuple[np.ndarray, str]], str, str]:
        """Load faceset from directory."""
        if not directory:
            return [], "", "Please specify a directory"

        dir_path = Path(directory)
        if not dir_path.exists():
            return [], "", f"Directory not found: {directory}"

        # Find face images
        self._face_files = sorted(
            list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")),
            key=lambda p: p.name,
        )

        if not self._face_files:
            return [], "", "No images found in directory"

        # Load gallery items
        gallery_items = []
        for path in self._face_files[:18]:  # First 18 for gallery
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gallery_items.append((img_rgb, path.stem))
            except Exception:
                continue

        self._current_face_idx = 0
        status = f"Loaded {len(self._face_files)} faces"
        face_info = self._face_files[0].name if self._face_files else ""

        return gallery_items, face_info, status

    def _prev_face(self) -> tuple[dict, np.ndarray, np.ndarray, str]:
        """Navigate to previous face."""
        if not self._face_files:
            return {}, None, None, ""

        self._current_face_idx = max(0, self._current_face_idx - 1)
        return self._load_current_face()

    def _next_face(self) -> tuple[dict, np.ndarray, np.ndarray, str]:
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
    ) -> tuple[dict, np.ndarray, np.ndarray, str]:
        """Handle gallery face selection."""
        if evt.index is not None and evt.index < len(self._face_files):
            self._current_face_idx = evt.index
        return self._load_current_face()

    def _load_current_face(self) -> tuple[dict, np.ndarray, np.ndarray, str]:
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
            from visagen.vision.dflimg import DFLImage

            image, metadata = DFLImage.load(face_path)
            if metadata is None:
                return "No DFL metadata in image"

            # Convert mask to float and set
            mask_float = mask.astype(np.float32) / 255.0
            if len(mask_float.shape) == 2:
                mask_float = mask_float[..., np.newaxis]

            DFLImage.set_xseg_mask(metadata, mask_float)
            DFLImage.save(face_path, image, metadata)

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

            self._training_thread = threading.Thread(target=train_fn)
            self._training_thread.start()

            return "Training started...", "\n".join(log_messages)

        except Exception as e:
            return f"Error starting training: {e}", ""

    def _stop_training(self) -> str:
        """Stop training."""
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
    ) -> tuple[list[tuple[np.ndarray, str]], str]:
        """Apply segmentation to batch of faces."""
        input_path = Path(input_dir)
        if not input_path.exists():
            return [], "Input directory not found"

        # Collect files
        files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
        if not files:
            return [], "No images found"

        # Build component set
        components = set()
        if skin:
            components.add("skin")
        if nose:
            components.add("nose")
        if eyes:
            components.update(["left_eye", "right_eye", "left_brow", "right_brow"])
        if mouth:
            components.update(["mouth", "upper_lip", "lower_lip"])
        if hair:
            components.add("hair")

        # Load segmenter (with optional LoRA)
        if use_lora and lora_weights and Path(lora_weights).exists():
            from visagen.vision.segmenter_lora import FaceSegmenterLoRA

            segmenter = FaceSegmenterLoRA(lora_weights=lora_weights)
        else:
            segmenter = self.segmenter

        # Refinement config
        refine_config = MaskRefinementConfig(
            erode_size=erode,
            dilate_size=dilate,
            blur_size=blur,
        )

        # Process files
        preview_items = []
        for file_path in files[:8]:  # Preview first 8
            try:
                image = cv2.imread(str(file_path))
                if image is None:
                    continue

                # Get parsing (segment call not needed, just get_parsing)
                parsing_dict = segmenter.get_parsing(image)

                # Build parsing map
                h, w = image.shape[:2]
                parsing = np.zeros((h, w), dtype=np.uint8)
                from visagen.vision.segmenter import LABEL_TO_ID

                for name, mask in parsing_dict.items():
                    if name in LABEL_TO_ID:
                        parsing[mask > 127] = LABEL_TO_ID[name]

                # Build mask
                mask = MaskOperations.combine_component_masks(parsing, components)
                mask = MaskOperations.refine(mask, refine_config)

                # Create preview overlay
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                overlay = image_rgb.copy()
                overlay[mask > 127, 1] = np.clip(
                    overlay[mask > 127, 1].astype(np.int32) + 80, 0, 255
                ).astype(np.uint8)

                preview_items.append((overlay, file_path.stem))

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

        return preview_items, f"Previewing {len(preview_items)}/{len(files)} images"
