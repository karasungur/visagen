"""Video tools tab for video processing utilities."""

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


class VideoToolsTab(BaseTab):
    """
    Video editing tools tab.

    Provides utilities for:
    - Extracting frames from video
    - Creating video from frames
    - Cutting video segments
    - Temporal denoising
    """

    @property
    def id(self) -> str:
        return "video_tools"

    def _build_content(self) -> dict[str, Any]:
        """Build video tools tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # === Extract Frames ===
        gr.Markdown(f"#### {self.t('extract.title')}")
        with gr.Row():
            with gr.Column():
                components["extract_input"] = PathInput(
                    PathInputConfig(
                        key="video_tools.extract.input",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["extract_output"] = PathInput(
                    PathInputConfig(
                        key="video_tools.extract.output",
                        path_type="directory",
                        default="./frames",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["extract_fps"] = gr.Number(
                    label=self.t("extract.fps.label"),
                    value=0,
                    info=self.t("extract.fps.info"),
                )

                components["extract_format"] = DropdownInput(
                    DropdownConfig(
                        key="video_tools.extract.format",
                        choices=["png", "jpg"],
                        default="png",
                    ),
                    self.i18n,
                ).build()

        components["extract_btn"] = gr.Button(
            self.t("extract.start"), variant="primary"
        )

        components["extract_log"] = LogOutput(
            LogOutputConfig(key="video_tools.extract.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # === Create Video ===
        gr.Markdown(f"#### {self.t('create.title')}")
        with gr.Row():
            with gr.Column():
                components["create_input"] = PathInput(
                    PathInputConfig(
                        key="video_tools.create.input",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["create_output"] = PathInput(
                    PathInputConfig(
                        key="video_tools.create.output",
                        path_type="file",
                        default="./output.mp4",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["create_fps"] = gr.Number(
                    label=self.t("create.fps.label"),
                    value=30,
                )

                components["create_codec"] = DropdownInput(
                    DropdownConfig(
                        key="video_tools.create.codec",
                        choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                        default="libx264",
                    ),
                    self.i18n,
                ).build()

                components["create_bitrate"] = gr.Textbox(
                    label=self.t("create.bitrate.label"),
                    value="16M",
                    info=self.t("create.bitrate.info"),
                )

        components["create_btn"] = gr.Button(self.t("create.start"), variant="primary")

        components["create_log"] = LogOutput(
            LogOutputConfig(key="video_tools.create.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # === Cut Video ===
        gr.Markdown(f"#### {self.t('cut.title')}")
        with gr.Row():
            with gr.Column():
                components["cut_input"] = PathInput(
                    PathInputConfig(
                        key="video_tools.cut.input",
                        path_type="file",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["cut_output"] = PathInput(
                    PathInputConfig(
                        key="video_tools.cut.output",
                        path_type="file",
                        default="./cut_output.mp4",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["cut_start"] = gr.Textbox(
                    label=self.t("cut.start_time.label"),
                    value="00:00:00",
                    info=self.t("cut.start_time.info"),
                )

                components["cut_end"] = gr.Textbox(
                    label=self.t("cut.end_time.label"),
                    value="00:00:10",
                    info=self.t("cut.end_time.info"),
                )

        components["cut_btn"] = gr.Button(self.t("cut.start"), variant="primary")

        components["cut_log"] = LogOutput(
            LogOutputConfig(key="video_tools.cut.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # === Denoise ===
        gr.Markdown(f"#### {self.t('denoise.title')}")
        gr.Markdown(self.t("denoise.description"))

        with gr.Row():
            with gr.Column():
                components["denoise_input"] = PathInput(
                    PathInputConfig(
                        key="video_tools.denoise.input",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["denoise_output"] = PathInput(
                    PathInputConfig(
                        key="video_tools.denoise.output",
                        path_type="directory",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["denoise_factor"] = SliderInput(
                    SliderConfig(
                        key="video_tools.denoise.factor",
                        minimum=3,
                        maximum=15,
                        step=2,
                        default=7,
                    ),
                    self.i18n,
                ).build()

        components["denoise_btn"] = gr.Button(
            self.t("denoise.start"), variant="primary"
        )

        components["denoise_log"] = LogOutput(
            LogOutputConfig(key="video_tools.denoise.log", lines=8, max_lines=15),
            self.i18n,
        ).build()

        gr.Markdown("---")

        # Global stop button for all video operations
        components["stop_btn"] = gr.Button(
            self.t("stop_all"),
            variant="stop",
        )

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up video tools event handlers."""

        # Extract Frames
        c["extract_btn"].click(
            fn=self._run_extract_frames,
            inputs=[
                c["extract_input"],
                c["extract_output"],
                c["extract_fps"],
                c["extract_format"],
            ],
            outputs=c["extract_log"],
        )

        # Create Video
        c["create_btn"].click(
            fn=self._run_create_video,
            inputs=[
                c["create_input"],
                c["create_output"],
                c["create_fps"],
                c["create_codec"],
                c["create_bitrate"],
            ],
            outputs=c["create_log"],
        )

        # Cut Video
        c["cut_btn"].click(
            fn=self._run_cut_video,
            inputs=[c["cut_input"], c["cut_output"], c["cut_start"], c["cut_end"]],
            outputs=c["cut_log"],
        )

        # Denoise
        c["denoise_btn"].click(
            fn=self._run_denoise_sequence,
            inputs=[c["denoise_input"], c["denoise_output"], c["denoise_factor"]],
            outputs=c["denoise_log"],
        )

        # Global stop
        c["stop_btn"].click(
            fn=self._stop_current_process,
            outputs=[],
        )

    # =========================================================================
    # Operation Handlers
    # =========================================================================

    def _run_extract_frames(
        self,
        input_video: str,
        output_dir: str,
        fps: float | None,
        output_format: str,
    ) -> Generator[str, None, None]:
        """Run frame extraction from video."""
        if not input_video or not Path(input_video).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "extract",
            str(input_video),
            "--output",
            output_dir or "./frames",
            "--format",
            output_format,
        ]

        if fps and fps > 0:
            cmd.extend(["--fps", str(fps)])

        yield f"Starting frame extraction...\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.video_tools = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(self.state.processes.video_tools.stdout.readline, ""):
                if line:
                    yield line

            exit_code = self.state.processes.video_tools.wait()
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.video_tools = None

    def _run_create_video(
        self,
        input_dir: str,
        output_video: str,
        fps: float,
        codec: str,
        bitrate: str,
    ) -> Generator[str, None, None]:
        """Run video creation from frames."""
        if not input_dir or not Path(input_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "create",
            str(input_dir),
            "--output",
            output_video or "./output.mp4",
            "--fps",
            str(fps),
            "--codec",
            codec,
            "--bitrate",
            bitrate,
        ]

        yield f"Starting video creation...\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.video_tools = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(self.state.processes.video_tools.stdout.readline, ""):
                if line:
                    yield line

            exit_code = self.state.processes.video_tools.wait()
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.video_tools = None

    def _run_cut_video(
        self,
        input_video: str,
        output_video: str,
        start_time: str,
        end_time: str,
    ) -> Generator[str, None, None]:
        """Run video cutting."""
        if not input_video or not Path(input_video).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "cut",
            str(input_video),
            "--output",
            output_video or "./cut_output.mp4",
            "--start",
            start_time,
            "--end",
            end_time,
        ]

        yield f"Starting video cut...\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.video_tools = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(self.state.processes.video_tools.stdout.readline, ""):
                if line:
                    yield line

            exit_code = self.state.processes.video_tools.wait()
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.video_tools = None

    def _run_denoise_sequence(
        self,
        input_dir: str,
        output_dir: str,
        factor: int,
    ) -> Generator[str, None, None]:
        """Run temporal denoising on frame sequence."""
        if not input_dir or not Path(input_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.video_ed",
            "denoise",
            str(input_dir),
            "--factor",
            str(factor),
        ]

        if output_dir:
            cmd.extend(["--output", output_dir])

        yield f"Starting temporal denoising...\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.video_tools = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(self.state.processes.video_tools.stdout.readline, ""):
                if line:
                    yield line

            exit_code = self.state.processes.video_tools.wait()
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.video_tools = None

    def _stop_current_process(self) -> None:
        """Stop currently running video tools process."""
        self.state.processes.terminate("video_tools")
