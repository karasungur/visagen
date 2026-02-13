"""Sort tab for dataset sorting and filtering."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

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


class SortTab(BaseTab):
    """
    Dataset sorting tab.

    Allows users to:
    - Select input directory of aligned faces
    - Choose sorting method (blur, face-yaw, final, etc.)
    - Configure target count for final methods
    - Preview with dry run mode
    - Run sorting with real-time log streaming
    """

    @property
    def id(self) -> str:
        return "sort"

    def _build_content(self) -> dict[str, Any]:
        """Build sort tab UI components."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        with gr.Row():
            with gr.Column():
                components["input_dir"] = PathInput(
                    PathInputConfig(
                        key="sort.input_dir",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                components["output_dir"] = PathInput(
                    PathInputConfig(
                        key="sort.output_dir",
                        path_type="directory",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["method"] = DropdownInput(
                    DropdownConfig(
                        key="sort.method",
                        choices=[
                            "blur",
                            "blur-fast",
                            "motion-blur",
                            "face-yaw",
                            "face-pitch",
                            "face-source-rect-size",
                            "hist",
                            "hist-dissim",
                            "absdiff",
                            "absdiff-dissim",
                            "id-sim",
                            "id-dissim",
                            "ssim",
                            "ssim-dissim",
                            "brightness",
                            "hue",
                            "black",
                            "origname",
                            "oneface",
                            "final",
                            "final-fast",
                        ],
                        default="blur",
                    ),
                    self.i18n,
                ).build()

                components["target_count"] = SliderInput(
                    SliderConfig(
                        key="sort.target_count",
                        minimum=100,
                        maximum=10000,
                        step=100,
                        default=2000,
                    ),
                    self.i18n,
                ).build()

                components["dry_run"] = gr.Checkbox(
                    label=self.t("dry_run.label"),
                    value=True,
                    info=self.t("dry_run.info"),
                )

                components["exec_mode"] = DropdownInput(
                    DropdownConfig(
                        key="sort.exec_mode",
                        choices=["auto", "process", "thread"],
                        default="auto",
                    ),
                    self.i18n,
                ).build()

                components["exact_limit"] = gr.Number(
                    label=self.t("exact_limit.label"),
                    value=3000,
                    precision=0,
                    info=self.t("exact_limit.info"),
                )

                components["jobs"] = gr.Number(
                    label=self.t("jobs.label"),
                    value=0,
                    precision=0,
                    info=self.t("jobs.info"),
                )

        # Process Control
        with gr.Row():
            process_ctrl = ProcessControl("sort", self.i18n)
            components["start_btn"], components["stop_btn"] = process_ctrl.build()

        # Log Output
        components["log"] = LogOutput(
            LogOutputConfig(key="sort.log", lines=15, max_lines=30),
            self.i18n,
        ).build()

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up sort event handlers."""
        c["start_btn"].click(
            fn=self._start_sort,
            inputs=[
                c["input_dir"],
                c["output_dir"],
                c["method"],
                c["target_count"],
                c["dry_run"],
                c["exec_mode"],
                c["exact_limit"],
                c["jobs"],
            ],
            outputs=c["log"],
        )

        c["stop_btn"].click(
            fn=self._stop_sort,
            outputs=c["log"],
        )

    def _start_sort(
        self,
        input_dir: str,
        output_dir: str,
        method: str,
        target_count: int,
        dry_run: bool,
        exec_mode: str,
        exact_limit: int,
        jobs: int,
    ) -> Generator[str, None, None]:
        """Start sort subprocess."""
        if not input_dir or not Path(input_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.sorter",
            str(input_dir),
            "--method",
            method,
            "--verbose",
        ]

        if output_dir:
            cmd.extend(["--output", str(output_dir)])

        if method in ("final", "final-fast"):
            cmd.extend(["--target", str(int(target_count))])

        if dry_run:
            cmd.append("--dry-run")

        cmd.extend(["--exec-mode", exec_mode])
        cmd.extend(["--exact-limit", str(int(exact_limit))])
        if int(jobs) > 0:
            cmd.extend(["--jobs", str(int(jobs))])

        yield f"Starting sorting...\n$ {' '.join(cmd)}\n"

        try:
            self.state.processes.sort = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if self.state.processes.sort.stdout:
                for line in iter(self.state.processes.sort.stdout.readline, ""):
                    if line:
                        yield line
                    if self.state.processes.sort.poll() is not None:
                        break

            remaining, _ = self.state.processes.sort.communicate()
            if remaining:
                yield remaining

            exit_code = self.state.processes.sort.returncode
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.state.processes.sort = None

    def _stop_sort(self) -> str:
        """Stop sort process."""
        if self.state.processes.terminate("sort"):
            return cast(str, self.i18n.t("status.stopped"))
        return cast(str, self.i18n.t("status.no_sorting"))
