"""Merge tab for video face swap processing."""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from visagen.gui.command_builders import build_merge_command
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


class MergeTab(BaseTab):
    """
    Video face swap merge tab.

    Allows users to:
    - Select input/output video paths
    - Configure model checkpoint
    - Choose color transfer and blend modes
    - Enable face restoration (GFPGAN)
    - Configure video encoding options
    - Run merge with real-time log streaming
    """

    @property
    def id(self) -> str:
        return "merge"

    def _build_content(self) -> dict[str, Any]:
        """Build merge tab UI components."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # Input/Output Section
        with gr.Row():
            with gr.Column():
                components["input_video"] = PathInput(
                    PathInputConfig(
                        key="merge.input_video",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["output_video"] = PathInput(
                    PathInputConfig(
                        key="merge.output_video",
                        path_type="file",
                        default="./output.mp4",
                    ),
                    self.i18n,
                ).build()

                components["checkpoint"] = PathInput(
                    PathInputConfig(
                        key="merge.checkpoint",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["color_transfer"] = DropdownInput(
                    DropdownConfig(
                        key="merge.color_transfer",
                        choices=[
                            "none",
                            "rct",
                            "lct",
                            "sot",
                            "mkl",
                            "idt",
                            "mix",
                            "hist-match",
                        ],
                        default="rct",
                    ),
                    self.i18n,
                ).build()

                components["blend_mode"] = DropdownInput(
                    DropdownConfig(
                        key="merge.blend_mode",
                        choices=["laplacian", "poisson", "feather"],
                        default="laplacian",
                    ),
                    self.i18n,
                ).build()

        # Face Restoration Section
        gr.Markdown(f"#### {self.t('restoration.title')}")
        with gr.Row():
            components["restore_face"] = gr.Checkbox(
                label=self.t("restoration.enable.label"),
                value=False,
                info=self.t("restoration.enable.info"),
            )
            components["restore_strength"] = SliderInput(
                SliderConfig(
                    key="merge.restoration.strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    default=0.5,
                ),
                self.i18n,
            ).build()
            components["restore_version"] = DropdownInput(
                DropdownConfig(
                    key="merge.restoration.version",
                    choices=["1.2", "1.3", "1.4"],
                    default="1.4",
                ),
                self.i18n,
            ).build()

        # Video Encoding Section
        gr.Markdown(f"#### {self.t('encoding.title')}")
        with gr.Row():
            components["codec"] = DropdownInput(
                DropdownConfig(
                    key="merge.encoding.codec",
                    choices=["auto", "libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                    default="auto",
                ),
                self.i18n,
            ).build()
            components["crf"] = SliderInput(
                SliderConfig(
                    key="merge.encoding.crf",
                    minimum=0,
                    maximum=51,
                    step=1,
                    default=18,
                ),
                self.i18n,
            ).build()

        # Process Control
        with gr.Row():
            process_ctrl = ProcessControl("merge", self.i18n)
            components["start_btn"], components["stop_btn"] = process_ctrl.build()

        # Log Output
        components["log"] = LogOutput(
            LogOutputConfig(key="merge.log", lines=15, max_lines=30),
            self.i18n,
        ).build()

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up merge event handlers."""
        c["start_btn"].click(
            fn=self._start_merge,
            inputs=[
                c["input_video"],
                c["output_video"],
                c["checkpoint"],
                c["color_transfer"],
                c["blend_mode"],
                c["restore_face"],
                c["restore_strength"],
                c["restore_version"],
                c["codec"],
                c["crf"],
            ],
            outputs=c["log"],
        )

        c["stop_btn"].click(
            fn=self._stop_merge,
            outputs=c["log"],
        )

    def _start_merge(
        self,
        input_video: str,
        output_video: str,
        checkpoint: str,
        color_transfer: str,
        blend_mode: str,
        restore_face: bool,
        restore_strength: float,
        restore_version: str,
        codec: str,
        crf: int,
    ) -> Generator[str, None, None]:
        """Start merge subprocess."""
        # Validation
        if not input_video or not Path(input_video).exists():
            yield self.i18n.t("errors.path_not_found")
            return
        if not checkpoint or not Path(checkpoint).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = build_merge_command(
            input_video,
            output_video if output_video else "./output.mp4",
            checkpoint,
            color_transfer=color_transfer if color_transfer != "none" else "none",
            blend_mode=blend_mode,
            restore_face=restore_face,
            restore_strength=restore_strength,
            restore_model=restore_version,
            codec=codec,
            crf=int(crf),
        )

        yield f"Starting merge...\nResolved argv: {' '.join(cmd)}\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.merge = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if self.state.processes.merge.stdout:
                for line in iter(self.state.processes.merge.stdout.readline, ""):
                    if line:
                        yield line
                    if self.state.processes.merge.poll() is not None:
                        break

            remaining, _ = self.state.processes.merge.communicate()
            if remaining:
                yield remaining

            exit_code = self.state.processes.merge.returncode
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.merge = None

    def _stop_merge(self) -> str:
        """Stop merge process."""
        if self.state.processes.terminate("merge"):
            return self.i18n.t("status.stopped")
        return self.i18n.t("status.no_merge")
