"""Video tools tab for video processing utilities."""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from visagen.gui.command_builders import (
    build_video_create_command,
    build_video_cut_command,
    build_video_denoise_command,
    build_video_extract_command,
)
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

                components["cut_codec"] = DropdownInput(
                    DropdownConfig(
                        key="video_tools.cut.codec",
                        choices=[
                            "copy",
                            "libx264",
                            "libx265",
                            "h264_nvenc",
                            "hevc_nvenc",
                        ],
                        default="copy",
                    ),
                    self.i18n,
                ).build()

                components["cut_audio_track"] = gr.Number(
                    label=self.t("cut.audio_track.label"),
                    value=0,
                    precision=0,
                    info=self.t("cut.audio_track.info"),
                )

                components["cut_bitrate"] = gr.Textbox(
                    label=self.t("cut.bitrate.label"),
                    value="",
                    info=self.t("cut.bitrate.info"),
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
            inputs=[
                c["cut_input"],
                c["cut_output"],
                c["cut_start"],
                c["cut_end"],
                c["cut_codec"],
                c["cut_audio_track"],
                c["cut_bitrate"],
            ],
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

    def _run_command_in_slot(
        self,
        slot: str,
        cmd: list[str],
        *,
        already_running_message: str,
    ) -> Generator[str, None, None]:
        """Run command in a managed process slot and stream stdout."""
        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                slot,
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process is None:
                yield f"\n\n{already_running_message}"
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
                self.state.processes.clear_if(slot, process)

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

        cmd = build_video_extract_command(
            input_video,
            output_dir or "./frames",
            fps=fps if fps and fps > 0 else None,
            output_format=output_format,
        )

        yield (
            f"Starting frame extraction...\n"
            f"Resolved argv: {' '.join(cmd)}\n"
            f"$ {' '.join(cmd)}\n"
        )
        yield from self._run_command_in_slot(
            "video_tools",
            cmd,
            already_running_message="Video tool is already running. Stop it first.",
        )

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

        cmd = build_video_create_command(
            input_dir,
            output_video or "./output.mp4",
            fps=fps,
            codec=codec,
            bitrate=bitrate,
        )

        yield (
            f"Starting video creation...\n"
            f"Resolved argv: {' '.join(cmd)}\n"
            f"$ {' '.join(cmd)}\n"
        )
        yield from self._run_command_in_slot(
            "video_tools",
            cmd,
            already_running_message="Video tool is already running. Stop it first.",
        )

    def _run_cut_video(
        self,
        input_video: str,
        output_video: str,
        start_time: str,
        end_time: str,
        codec: str,
        audio_track: float | None,
        bitrate: str,
    ) -> Generator[str, None, None]:
        """Run video cutting."""
        if not input_video or not Path(input_video).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        audio_track_id = 0 if audio_track is None else max(0, int(audio_track))
        cmd = build_video_cut_command(
            input_video,
            output_video or "./cut_output.mp4",
            start_time=start_time,
            end_time=end_time,
            codec=codec,
            audio_track_id=audio_track_id,
            bitrate=bitrate.strip() or None,
        )

        yield (
            f"Starting video cut...\n"
            f"Resolved argv: {' '.join(cmd)}\n"
            f"$ {' '.join(cmd)}\n"
        )
        yield from self._run_command_in_slot(
            "video_tools",
            cmd,
            already_running_message="Video tool is already running. Stop it first.",
        )

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

        cmd = build_video_denoise_command(
            input_dir,
            output_dir=output_dir if output_dir else None,
            factor=factor,
        )

        yield (
            f"Starting temporal denoising...\n"
            f"Resolved argv: {' '.join(cmd)}\n"
            f"$ {' '.join(cmd)}\n"
        )
        yield from self._run_command_in_slot(
            "video_tools",
            cmd,
            already_running_message="Video tool is already running. Stop it first.",
        )

    def _stop_current_process(self) -> None:
        """Stop currently running video tools process."""
        self.state.processes.terminate("video_tools")
