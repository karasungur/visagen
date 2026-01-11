"""Extract tab for face extraction."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

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
from visagen.gui.tabs.base import BaseTab

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class ExtractTab(BaseTab):
    """
    Face extraction tab for extracting faces from images/videos.

    Allows users to:
    - Select input path (file or directory)
    - Configure output directory
    - Choose face type and output size
    - Set minimum detection confidence
    - Run extraction with real-time log streaming
    """

    def __init__(
        self,
        app_state: AppState,
        i18n: I18n,
    ) -> None:
        """Initialize extract tab."""
        super().__init__(app_state, i18n)
        self._process: subprocess.Popen | None = None

    @property
    def id(self) -> str:
        """Return unique tab identifier."""
        return "extract"

    def _build_content(self) -> dict[str, Any]:
        """Build extraction UI components."""
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
                            path_type="file",  # Can be file or dir
                            must_exist=False,  # Validated at runtime
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

            # Process Control (Start/Stop buttons)
            with gr.Row():
                process_control = ProcessControl(
                    key="extract",
                    i18n=self.i18n,
                )
                start_btn, stop_btn = process_control.build()
                components["start_btn"] = start_btn
                components["stop_btn"] = stop_btn

            # Log Output
            log_output = LogOutput(
                config=LogOutputConfig(
                    key="extract.log",
                    lines=15,
                    max_lines=30,
                ),
                i18n=self.i18n,
            )
            components["log"] = log_output.build()

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
            ],
            outputs=[components["log"]],
        )

        # Stop button terminates process
        components["stop_btn"].click(
            fn=self._stop_extraction,
            outputs=[components["log"]],
        )

    def _start_extraction(
        self,
        input_path: str,
        output_dir: str,
        face_type: str,
        output_size: int,
        min_confidence: float,
    ) -> Generator[str, None, None]:
        """
        Start face extraction process.

        Generator method that yields log lines as extraction progresses.
        Calls visagen.tools.extract_v2 via subprocess and streams stdout.

        Args:
            input_path: Path to input file or directory.
            output_dir: Directory for extracted faces.
            face_type: Face type for alignment.
            output_size: Size of output face images.
            min_confidence: Minimum detection confidence threshold.

        Yields:
            Log lines from the extraction process.
        """
        # Validate input path
        if not input_path or not Path(input_path).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.extract_v2",
            str(input_path),
            str(output_dir) if output_dir else "./workspace/extracted",
            "--size",
            str(int(output_size)),
            "--face-type",
            face_type,
            "--min-confidence",
            str(min_confidence),
        ]

        yield f"Starting extraction...\n$ {' '.join(cmd)}\n"

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream stdout line by line
            for line in iter(self._process.stdout.readline, ""):
                if line:
                    yield line
                if self._process.poll() is not None:
                    break

            # Get any remaining output
            remaining, _ = self._process.communicate()
            if remaining:
                yield remaining

            exit_code = self._process.returncode
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.extraction_completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self._process = None

    def _stop_extraction(self) -> str:
        """
        Stop running extraction process.

        Returns:
            Status message indicating whether process was stopped.
        """
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            return self.i18n.t("status.extraction_stopped")
        return self.i18n.t("status.no_extraction")
