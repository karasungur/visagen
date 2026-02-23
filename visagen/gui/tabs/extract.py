"""Extract tab for face extraction with real-time preview."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    LogOutput,
    LogOutputConfig,
    PathInput,
    PathInputConfig,
    ProcessControl,
    SliderConfig,
    SliderInput,
)
from visagen.gui.components.displays import (
    GalleryPreview,
    GalleryPreviewConfig,
    create_mask_overlay,
)
from visagen.gui.components.faceset_browser import FacesetBrowser, FacesetBrowserConfig
from visagen.gui.tabs.base import BaseTab
from visagen.tools.extract_v2 import ExtractionProgress, FaceExtractor
from visagen.vision.face_image import FaceImage
from visagen.vision.face_type import FaceType


class ExtractTab(BaseTab):
    """
    Face extraction tab with real-time preview.

    Allows users to:
    - Select input path (file or directory)
    - Configure output directory
    - Choose face type and output size
    - Set minimum detection confidence
    - Run extraction with real-time gallery preview
    - View extracted faces with mask overlays
    """

    MAX_PREVIEW_ITEMS = 16

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extractor: FaceExtractor | None = None
        self._extraction_active = False
        self._state_lock = threading.Lock()
        self._preview_buffer: deque[tuple[np.ndarray, str]] = deque(
            maxlen=self.MAX_PREVIEW_ITEMS
        )
        self._buffer_lock = threading.Lock()
        self._last_face_info: dict[str, Any] | None = None
        self._browser: FacesetBrowser | None = None

    @property
    def id(self) -> str:
        """Return unique tab identifier."""
        return "extract"

    def _build_content(self) -> dict[str, Any]:
        """Build extraction UI components with preview gallery."""
        components: dict[str, Any] = {}

        with gr.Column():
            # Description
            gr.Markdown(f"### {self.t('title')}")
            gr.Markdown(self.t("description"))

            # Input Section
            with gr.Row():
                with gr.Column():
                    # Input path (file or directory)
                    input_path = PathInput(
                        config=PathInputConfig(
                            key="extract.input_path",
                            path_type="file",
                            must_exist=False,
                        ),
                        i18n=self.i18n,
                    )
                    components["input_path"] = input_path.build()

                    # Output directory
                    output_dir = PathInput(
                        config=PathInputConfig(
                            key="extract.output_dir",
                            path_type="directory",
                            default="./workspace/extracted",
                        ),
                        i18n=self.i18n,
                    )
                    components["output_dir"] = output_dir.build()

                with gr.Column():
                    # Face Type dropdown
                    face_type = DropdownInput(
                        config=DropdownConfig(
                            key="extract.face_type",
                            choices=["whole_face", "full", "mid_full", "half", "head"],
                            default="whole_face",
                        ),
                        i18n=self.i18n,
                    )
                    components["face_type"] = face_type.build()

                    # Output Size slider
                    output_size = SliderInput(
                        config=SliderConfig(
                            key="extract.output_size",
                            default=512,
                            minimum=128,
                            maximum=1024,
                            step=64,
                        ),
                        i18n=self.i18n,
                    )
                    components["output_size"] = output_size.build()

                    # Min Confidence slider
                    min_confidence = SliderInput(
                        config=SliderConfig(
                            key="extract.min_confidence",
                            default=0.5,
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                        ),
                        i18n=self.i18n,
                    )
                    components["min_confidence"] = min_confidence.build()

            # Preview Options
            with gr.Row():
                components["show_mask"] = gr.Checkbox(
                    label=self.t("preview.show_mask"),
                    value=True,
                    info=self.t("preview.show_mask_info"),
                )

            # Process Control (Start/Stop buttons)
            with gr.Row():
                process_control = ProcessControl(
                    key="extract",
                    i18n=self.i18n,
                )
                start_btn, stop_btn = process_control.build()
                components["start_btn"] = start_btn
                components["stop_btn"] = stop_btn

            # Status
            components["status"] = gr.Textbox(
                label=self.t("status.label"),
                value="",
                interactive=False,
            )

            # Real-Time Preview Gallery
            gr.Markdown(f"### {self.t('preview.title')}")

            components["preview_gallery"] = GalleryPreview(
                GalleryPreviewConfig(
                    key="extract.preview_gallery",
                    columns=4,
                    rows=2,
                    height=400,
                ),
                self.i18n,
            ).build()

            # Last Face Detail
            with gr.Row():
                components["last_face"] = gr.Image(
                    label=self.t("preview.last_face"),
                    height=256,
                    interactive=False,
                )
                components["face_info"] = gr.JSON(
                    label=self.t("preview.face_info"),
                )

            # Timer for preview refresh
            components["preview_timer"] = gr.Timer(value=0.5, active=False)

            # Log Output
            log_output = LogOutput(
                config=LogOutputConfig(
                    key="extract.log",
                    lines=8,
                    max_lines=15,
                ),
                i18n=self.i18n,
            )
            components["log"] = log_output.build()

            # Faceset Browser (Accordion)
            gr.Markdown("---")
            self._browser = FacesetBrowser(
                FacesetBrowserConfig(
                    key="extract.browser",
                    columns=6,
                    rows=3,
                ),
                self.i18n,
            )
            browser_components = self._browser.build()
            components.update(
                {f"browser_{k}": v for k, v in browser_components.items()}
            )

        return components

    def _setup_events(self, components: dict[str, Any]) -> None:
        """Set up event handlers for extraction controls."""

        # Start button triggers extraction generator
        components["start_btn"].click(
            fn=self._start_extraction,
            inputs=[
                components["input_path"],
                components["output_dir"],
                components["face_type"],
                components["output_size"],
                components["min_confidence"],
                components["show_mask"],
            ],
            outputs=[
                components["log"],
                components["status"],
                components["preview_timer"],
            ],
        )

        # Stop button terminates process
        components["stop_btn"].click(
            fn=self._stop_extraction,
            outputs=[components["status"]],
        )

        # Timer refresh
        components["preview_timer"].tick(
            fn=self._refresh_preview,
            outputs=[
                components["preview_gallery"],
                components["last_face"],
                components["face_info"],
            ],
        )

        # Setup Faceset Browser events
        if self._browser:
            browser_comps = {
                k.replace("browser_", ""): v
                for k, v in components.items()
                if k.startswith("browser_")
            }
            self._browser.setup_events(browser_comps)

            # Update browser directory when output_dir changes
            components["output_dir"].change(
                fn=lambda d: d,
                inputs=[components["output_dir"]],
                outputs=[components["browser_dir_input"]],
            )

    def _start_extraction(
        self,
        input_path: str,
        output_dir: str,
        face_type: str,
        output_size: int,
        min_confidence: float,
        show_mask: bool,
    ) -> Generator[tuple[str, str, gr.Timer], None, None]:
        """
        In-process extraction with real-time preview.

        Args:
            input_path: Path to input file or directory.
            output_dir: Directory for extracted faces.
            face_type: Face type for alignment.
            output_size: Size of output face images.
            min_confidence: Minimum detection confidence threshold.
            show_mask: Whether to show mask overlay in preview.

        Yields:
            Tuple of (log_text, status_text, timer).
        """
        # Concurrent protection
        with self._state_lock:
            if self._extraction_active:
                yield (
                    self.i18n.t("extract.error.already_running"),
                    self.i18n.t("status.busy"),
                    gr.Timer(active=False),
                )
                return
            self._extraction_active = True

        # Validate input path
        if not input_path or not Path(input_path).exists():
            with self._state_lock:
                self._extraction_active = False
            yield (
                self.i18n.t("errors.path_not_found"),
                "Error",
                gr.Timer(active=False),
            )
            return

        # Clear buffer
        with self._buffer_lock:
            self._preview_buffer.clear()
        self._last_face_info = None

        # Create extractor
        self._extractor = FaceExtractor(
            output_size=int(output_size),
            face_type=FaceType.from_string(face_type),
            min_confidence=min_confidence,
        )

        self._extraction_active = True
        input_p = Path(input_path)
        output_p = Path(output_dir) if output_dir else Path("./workspace/extracted")
        output_p.mkdir(parents=True, exist_ok=True)

        log_buffer = f"Starting extraction from {input_p.name}...\n"
        yield (log_buffer, "Processing...", gr.Timer(active=True))

        try:
            if input_p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
                # Video extraction
                for face, progress in self._extractor.extract_streaming(
                    input_p, output_p, with_mask=True
                ):
                    if not self._extraction_active:
                        break

                    self._add_to_preview(face, progress, show_mask)

                    pct = progress.current_frame / max(progress.total_frames, 1) * 100
                    log_buffer += (
                        f"Frame {progress.current_frame}/{progress.total_frames} | "
                        f"Faces: {progress.faces_extracted}\n"
                    )
                    yield (
                        log_buffer,
                        f"Processing... {pct:.1f}%",
                        gr.Timer(active=True),
                    )

            else:
                # Single image or directory
                if input_p.is_dir():
                    image_files = list(input_p.glob("*.jpg")) + list(
                        input_p.glob("*.png")
                    )
                else:
                    image_files = [input_p]

                total_faces = 0
                for idx, img_path in enumerate(image_files):
                    if not self._extraction_active:
                        break

                    faces = self._extractor.extract_from_image(img_path, with_mask=True)
                    for face in faces:
                        total_faces += 1
                        progress = ExtractionProgress(
                            current_frame=idx + 1,
                            total_frames=len(image_files),
                            faces_extracted=total_faces,
                            current_face=face,
                            source_name=img_path.stem,
                        )
                        self._add_to_preview(face, progress, show_mask)

                        # Save
                        out_name = f"{img_path.stem}_{face.face_index}.jpg"
                        FaceImage.save(output_p / out_name, face.image, face.metadata)

                    log_buffer += f"Processed {img_path.name}: {len(faces)} face(s)\n"
                    yield (
                        log_buffer,
                        f"Processing... {idx + 1}/{len(image_files)}",
                        gr.Timer(active=True),
                    )

                log_buffer += f"\nExtracted {total_faces} face(s)\n"

            log_buffer += f"\n{self.i18n.t('status.extraction_completed')}\n"
            yield (
                log_buffer,
                self.i18n.t("status.extraction_completed"),
                gr.Timer(active=False),
            )

        except Exception as e:
            log_buffer += f"\nError: {e}\n"
            yield (log_buffer, "Error", gr.Timer(active=False))

        finally:
            with self._state_lock:
                self._extraction_active = False
            if self._extractor is not None:
                self._extractor.cleanup()
            self._extractor = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _add_to_preview(
        self,
        face: Any,
        progress: ExtractionProgress,
        show_mask: bool,
    ) -> None:
        """Add face to preview buffer."""
        with self._buffer_lock:
            image = face.image.copy()

            if show_mask and face.metadata.xseg_mask:
                mask = cv2.imdecode(
                    np.frombuffer(face.metadata.xseg_mask, np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                if mask is not None:
                    image = create_mask_overlay(image, mask, alpha=0.4)

            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            caption = f"{progress.source_name}_{face.face_index}"

            self._preview_buffer.append((image_rgb, caption))

            # Update last face info
            self._last_face_info = {
                "caption": caption,
                "confidence": round(face.confidence, 3),
                "face_type": face.metadata.face_type,
                "total_extracted": progress.faces_extracted,
            }

    def _refresh_preview(
        self,
    ) -> tuple[list[tuple[np.ndarray, str]], np.ndarray | None, dict[str, Any] | None]:
        """Timer callback to refresh gallery."""
        with self._buffer_lock:
            items = list(self._preview_buffer)

        if items:
            last_img, _ = items[-1]
            info = self._last_face_info
        else:
            last_img = None
            info = None

        return items, last_img, info

    def _stop_extraction(self) -> str:
        """
        Stop running extraction process.

        Returns:
            Status message indicating whether process was stopped.
        """
        with self._state_lock:
            self._extraction_active = False
            if self._extractor is not None:
                self._extractor.request_stop()
        return self.i18n.t("status.extraction_stopped")
