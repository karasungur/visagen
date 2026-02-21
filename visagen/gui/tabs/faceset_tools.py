"""Faceset tools tab for dataset enhancement and resizing."""

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
    SliderConfig,
    SliderInput,
)
from visagen.gui.tabs.base import BaseTab


class FacesetToolsTab(BaseTab):
    """
    Faceset processing tools tab.

    Provides utilities for:
    - Enhancing face datasets (GFPGAN)
    - Resizing face datasets with metadata preservation
    """

    @property
    def id(self) -> str:
        return "faceset_tools"

    def _build_content(self) -> dict[str, Any]:
        """Build faceset tools tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # === Enhancement ===
        gr.Markdown(f"#### {self.t('enhance.title')}")
        gr.Markdown(self.t("enhance.description"))

        with gr.Row():
            with gr.Column():
                components["enhance_input"] = PathInput(
                    PathInputConfig(
                        key="faceset_tools.enhance.input",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["enhance_output"] = PathInput(
                    PathInputConfig(
                        key="faceset_tools.enhance.output",
                        path_type="directory",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["enhance_strength"] = SliderInput(
                    SliderConfig(
                        key="faceset_tools.enhance.strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        default=0.5,
                    ),
                    self.i18n,
                ).build()

                components["enhance_model"] = DropdownInput(
                    DropdownConfig(
                        key="faceset_tools.enhance.model",
                        choices=["1.2", "1.3", "1.4"],
                        default="1.4",
                    ),
                    self.i18n,
                ).build()

        components["enhance_btn"] = gr.Button(
            self.t("enhance.start"), variant="primary"
        )

        components["enhance_log"] = LogOutput(
            LogOutputConfig(key="faceset_tools.enhance.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # === Resizing ===
        gr.Markdown(f"#### {self.t('resize.title')}")
        gr.Markdown(self.t("resize.description"))

        with gr.Row():
            with gr.Column():
                components["resize_input"] = PathInput(
                    PathInputConfig(
                        key="faceset_tools.resize.input",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["resize_output"] = PathInput(
                    PathInputConfig(
                        key="faceset_tools.resize.output",
                        path_type="directory",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["resize_size"] = SliderInput(
                    SliderConfig(
                        key="faceset_tools.resize.size",
                        minimum=128,
                        maximum=1024,
                        step=64,
                        default=256,
                    ),
                    self.i18n,
                ).build()

                components["resize_face_type"] = DropdownInput(
                    DropdownConfig(
                        key="faceset_tools.resize.face_type",
                        choices=[
                            "keep",
                            "half_face",
                            "mid_face",
                            "full_face",
                            "whole_face",
                            "head",
                        ],
                        default="keep",
                    ),
                    self.i18n,
                ).build()

                components["resize_interp"] = DropdownInput(
                    DropdownConfig(
                        key="faceset_tools.resize.interp",
                        choices=["lanczos", "cubic", "linear", "nearest"],
                        default="lanczos",
                    ),
                    self.i18n,
                ).build()

        components["resize_btn"] = gr.Button(self.t("resize.start"), variant="primary")

        components["resize_log"] = LogOutput(
            LogOutputConfig(key="faceset_tools.resize.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # Global stop button
        components["stop_btn"] = gr.Button(
            self.t("stop_all"),
            variant="stop",
        )

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up faceset tools event handlers."""

        c["enhance_btn"].click(
            fn=self._run_enhance_faceset,
            inputs=[
                c["enhance_input"],
                c["enhance_output"],
                c["enhance_strength"],
                c["enhance_model"],
            ],
            outputs=c["enhance_log"],
        )

        c["resize_btn"].click(
            fn=self._run_resize_faceset,
            inputs=[
                c["resize_input"],
                c["resize_output"],
                c["resize_size"],
                c["resize_face_type"],
                c["resize_interp"],
            ],
            outputs=c["resize_log"],
        )

        # Global stop
        c["stop_btn"].click(
            fn=self._stop_current_process,
            outputs=[],
        )

    # =========================================================================
    # Operation Handlers
    # =========================================================================

    def _run_enhance_faceset(
        self,
        input_dir: str,
        output_dir: str,
        strength: float,
        model_version: float,
    ) -> Generator[str, None, None]:
        """Run faceset enhancement."""
        if not input_dir or not Path(input_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.faceset_enhancer",
            str(input_dir),
            "--strength",
            str(strength),
            "--model-version",
            str(model_version),
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        yield f"Starting faceset enhancement...\n$ {' '.join(cmd)}\n"

        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                "faceset_tools",
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if process is None:
                yield "\n\nFaceset tools process is already running. Stop it first."
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
                self.state.processes.clear_if("faceset_tools", process)

    def _run_resize_faceset(
        self,
        input_dir: str,
        output_dir: str,
        target_size: int,
        face_type: str | None,
        interpolation: str,
    ) -> Generator[str, None, None]:
        """Run faceset resizing."""
        if not input_dir or not Path(input_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.faceset_resizer",
            str(input_dir),
            "--size",
            str(int(target_size)),
            "--interpolation",
            interpolation,
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        if face_type and face_type != "keep":
            cmd.extend(["--face-type", face_type])

        yield f"Starting faceset resize...\n$ {' '.join(cmd)}\n"

        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                "faceset_tools",
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if process is None:
                yield "\n\nFaceset tools process is already running. Stop it first."
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
                self.state.processes.clear_if("faceset_tools", process)

    def _stop_current_process(self) -> None:
        """Stop currently running faceset tools process."""
        self.state.processes.terminate("faceset_tools")
