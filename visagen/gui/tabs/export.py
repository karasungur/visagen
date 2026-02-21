"""Export tab for model export to ONNX/TensorRT."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    LogOutput,
    LogOutputConfig,
    PathInput,
    PathInputConfig,
    ProcessControl,
)
from visagen.gui.tabs.base import BaseTab


class ExportTab(BaseTab):
    """
    Model export tab.

    Allows users to:
    - Export trained models to ONNX format
    - Convert ONNX to TensorRT for optimized inference
    - Configure precision (fp32, fp16, int8)
    - Validate exported models
    """

    @property
    def id(self) -> str:
        return "export"

    def _build_content(self) -> dict[str, Any]:
        """Build export tab UI components."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        with gr.Row():
            with gr.Column():
                components["input_path"] = PathInput(
                    PathInputConfig(
                        key="export.input_path",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["output_path"] = PathInput(
                    PathInputConfig(
                        key="export.output_path",
                        path_type="file",
                        default="./model.onnx",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["format"] = DropdownInput(
                    DropdownConfig(
                        key="export.format",
                        choices=["onnx", "tensorrt"],
                        default="onnx",
                    ),
                    self.i18n,
                ).build()

                components["precision"] = DropdownInput(
                    DropdownConfig(
                        key="export.precision",
                        choices=["fp32", "fp16", "int8"],
                        default="fp16",
                        interactive=False,
                    ),
                    self.i18n,
                ).build()

                components["validate"] = gr.Checkbox(
                    label=self.t("validate.label"),
                    value=True,
                    info=self.t("validate.info"),
                )

        # Process Control
        with gr.Row():
            process_ctrl = ProcessControl("export", self.i18n)
            components["start_btn"], components["stop_btn"] = process_ctrl.build()

        # Log Output
        components["log"] = LogOutput(
            LogOutputConfig(key="export.log", lines=15, max_lines=30),
            self.i18n,
        ).build()

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up export event handlers."""
        c["format"].change(
            fn=lambda export_format: gr.update(interactive=export_format == "tensorrt"),
            inputs=[c["format"]],
            outputs=[c["precision"]],
        )

        c["start_btn"].click(
            fn=self._start_export,
            inputs=[
                c["input_path"],
                c["output_path"],
                c["format"],
                c["precision"],
                c["validate"],
            ],
            outputs=c["log"],
        )

        c["stop_btn"].click(
            fn=self._stop_export,
            outputs=c["log"],
        )

    def _start_export(
        self,
        input_path: str,
        output_path: str,
        export_format: str,
        precision: str,
        validate: bool,
    ) -> Generator[str, None, None]:
        """Start export subprocess."""
        if not input_path or not Path(input_path).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.export",
            str(input_path),
            "-o",
            str(output_path) if output_path else "./model.onnx",
            "--format",
            export_format,
        ]
        if export_format == "tensorrt":
            cmd.extend(["--precision", precision])

        if validate:
            cmd.append("--validate")

        info_note = ""
        if export_format == "onnx":
            info_note = (
                "Note: Precision selection only applies to TensorRT builds; "
                "ONNX export uses model defaults.\n"
            )

        yield f"Starting export...\n{info_note}$ {' '.join(cmd)}\n"

        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                "export",
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process is None:
                yield "\n\nExport is already running. Stop the active job first."
                return

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        yield line
                    if process.poll() is not None:
                        break

            remaining, _ = process.communicate()
            if remaining:
                yield remaining

            exit_code = process.returncode
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            elif exit_code in {-15, -9, 143, 137}:
                yield f"\n\n{self.i18n.t('status.stopped')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.state.processes.clear_if("export", process)

    def _stop_export(self) -> str:
        """Stop export process."""
        if self.state.processes.terminate("export"):
            return self.i18n.t("status.stopped")
        return "No export in progress"
