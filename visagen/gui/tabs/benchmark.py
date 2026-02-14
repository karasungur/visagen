"""Benchmark tab for performance testing."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
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
    SliderConfig,
    SliderInput,
)
from visagen.gui.tabs.base import BaseTab


class BenchmarkTab(BaseTab):
    """
    Performance benchmark tab.

    Allows users to:
    - Run inference benchmarks on loaded models
    - Test training throughput
    - Analyze merger performance
    - View detailed performance reports
    """

    @property
    def id(self) -> str:
        return "benchmark"

    def _build_content(self) -> dict[str, Any]:
        """Build benchmark tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        with gr.Row():
            with gr.Column():
                # Configuration
                gr.Markdown(f"#### {self.t('config.title')}")

                components["mode"] = DropdownInput(
                    DropdownConfig(
                        key="benchmark.mode",
                        choices=["inference", "training", "merge", "all"],
                        default="inference",
                    ),
                    self.i18n,
                ).build()

                components["checkpoint"] = PathInput(
                    PathInputConfig(
                        key="benchmark.checkpoint",
                        path_type="file",
                        file_types=[".ckpt"],
                    ),
                    self.i18n,
                ).build()

                components["batch_sizes"] = gr.Textbox(
                    label=self.t("batch_sizes.label"),
                    value="1,2,4,8",
                    info=self.t("batch_sizes.info"),
                )

                components["resolutions"] = gr.Textbox(
                    label=self.t("resolutions.label"),
                    value="256",
                    info=self.t("resolutions.info"),
                )

            with gr.Column():
                # Advanced Settings
                gr.Markdown(f"#### {self.t('advanced.title')}")

                components["backends"] = gr.CheckboxGroup(
                    label=self.t("backends.label"),
                    choices=["pytorch", "onnx", "tensorrt"],
                    value=["pytorch"],
                )

                components["iterations"] = SliderInput(
                    SliderConfig(
                        key="benchmark.iterations",
                        minimum=10,
                        maximum=1000,
                        step=10,
                        default=100,
                    ),
                    self.i18n,
                ).build()

                components["warmup"] = SliderInput(
                    SliderConfig(
                        key="benchmark.warmup",
                        minimum=0,
                        maximum=50,
                        step=1,
                        default=10,
                    ),
                    self.i18n,
                ).build()

                components["device"] = DropdownInput(
                    DropdownConfig(
                        key="benchmark.device",
                        choices=["cuda", "cpu", "auto"],
                        default="cuda",
                    ),
                    self.i18n,
                ).build()

        # Process Control
        with gr.Row():
            process_ctrl = ProcessControl("benchmark", self.i18n)
            components["start_btn"], components["stop_btn"] = process_ctrl.build()

        # Log Output
        components["log"] = LogOutput(
            LogOutputConfig(key="benchmark.log", lines=20, max_lines=50),
            self.i18n,
        ).build()

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up benchmark event handlers."""
        c["start_btn"].click(
            fn=self._run_benchmark,
            inputs=[
                c["mode"],
                c["checkpoint"],
                c["batch_sizes"],
                c["resolutions"],
                c["backends"],
                c["iterations"],
                c["warmup"],
                c["device"],
            ],
            outputs=c["log"],
        )

        c["stop_btn"].click(
            fn=self._stop_benchmark,
            outputs=c["log"],
        )

    def _run_benchmark(
        self,
        mode: str,
        checkpoint: str,
        batch_sizes: str,
        resolutions: str,
        backends: list[str],
        iterations: float,
        warmup: float,
        device: str,
    ) -> Generator[str, None, None]:
        """Run benchmark process."""
        # Validation
        if mode in ("inference", "merge") and not checkpoint:
            yield self.i18n.t("errors.checkpoint_required")
            return

        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.benchmark",
            "--mode",
            mode,
            "--batch-sizes",
            batch_sizes,
            "--resolutions",
            resolutions,
            "--backends",
            ",".join(backends),
            "--iterations",
            str(int(iterations)),
            "--warmup",
            str(int(warmup)),
            "--device",
            device,
            "--output-format",
            "console",
        ]

        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])

        yield f"Starting benchmark...\n$ {' '.join(cmd)}\n"

        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                "benchmark",
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process is None:
                yield "\n\nBenchmark is already running. Stop the active job first."
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
                self.state.processes.clear_if("benchmark", process)

    def _stop_benchmark(self) -> str:
        """Stop benchmark process."""
        if self.state.processes.terminate("benchmark"):
            return self.i18n.t("status.stopped")
        return self.i18n.t("status.no_process")
