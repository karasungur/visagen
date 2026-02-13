"""Batch processing tab implementation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from visagen.gui.batch import BatchItem, BatchQueue, BatchStatus
from visagen.gui.command_builders import build_extract_command, build_merge_command
from visagen.gui.components import (
    PathInput,
    PathInputConfig,
)
from visagen.gui.tabs.base import BaseTab

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


logger = logging.getLogger(__name__)


class BatchTab(BaseTab):
    """
    Batch processing tab for multiple videos.

    Allows users to:
    - Queue multiple videos for processing
    - Monitor batch progress
    - Start/stop batch operations
    """

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        """Initialize batch tab with queue."""
        super().__init__(app_state, i18n)
        self.batch_queue = BatchQueue(max_parallel=1)

    @property
    def id(self) -> str:
        return "batch"

    def _build_content(self) -> dict[str, Any]:
        """Build batch processing tab UI."""
        components: dict[str, Any] = {}

        gr.Markdown(f"### {self.t('title')}")
        gr.Markdown(self.t("description"))

        # File upload section
        with gr.Row():
            with gr.Column(scale=2):
                components["file_input"] = gr.File(
                    label=self.t("files.label"),
                    file_count="multiple",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                )
            with gr.Column(scale=1):
                components["output_dir"] = PathInput(
                    PathInputConfig(
                        key="batch.output_dir",
                        path_type="directory",
                        default="./batch_output",
                    ),
                    self.i18n,
                ).build()

        # Model and operation selection
        with gr.Row():
            with gr.Column():
                components["checkpoint"] = PathInput(
                    PathInputConfig(
                        key="batch.checkpoint",
                        path_type="file",
                        file_types=[".ckpt"],
                    ),
                    self.i18n,
                ).build()
            with gr.Column():
                components["operation"] = gr.Dropdown(
                    label=self.t("operation.label"),
                    choices=[
                        ("Face Swap (Merge)", "merge"),
                        ("Extract Faces", "extract"),
                    ],
                    value="merge",
                    interactive=True,
                )

        # Add to queue button
        with gr.Row():
            components["add_btn"] = gr.Button(
                self.t("add_to_queue"),
                variant="secondary",
            )

        gr.Markdown("---")
        gr.Markdown(f"#### {self.t('queue.title')}")

        # Queue table
        components["queue_table"] = gr.Dataframe(
            headers=[
                "ID",
                self.t("queue.file"),
                self.t("queue.status"),
                self.t("queue.progress"),
            ],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=5,
        )

        # Progress section
        with gr.Row():
            components["progress_text"] = gr.Textbox(
                label=self.t("progress.label"),
                value="0 / 0 (0%)",
                interactive=False,
            )

        # Control buttons
        with gr.Row():
            components["start_btn"] = gr.Button(
                self.t("start_all"),
                variant="primary",
            )
            components["stop_btn"] = gr.Button(
                self.t("stop_all"),
                variant="stop",
            )
            components["clear_btn"] = gr.Button(
                self.t("clear_completed"),
            )
            components["refresh_btn"] = gr.Button(
                self.i18n.t("common.refresh"),
            )

        # Status message
        components["status"] = gr.Textbox(
            label=self.t("status.label"),
            interactive=False,
        )

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up batch processing event handlers."""

        def add_files(
            files: list | None,
            output_dir: str,
            operation: str,
        ) -> tuple[list[list[str]], str, str]:
            """Add files to batch queue."""
            if not files:
                return (
                    self.batch_queue.get_table_data(),
                    self._get_progress_text(),
                    self.t("status.no_files"),
                )

            output_path = Path(output_dir or "./batch_output")
            output_path.mkdir(parents=True, exist_ok=True)

            added_count = 0
            for file in files:
                # Handle both file objects and string paths
                input_path = file.name if hasattr(file, "name") else str(file)
                stem = Path(input_path).stem
                if operation == "extract":
                    out_path = str(output_path / stem)
                else:
                    out_path = str(output_path / f"{stem}_processed.mp4")
                self.batch_queue.add(input_path, out_path, operation)
                added_count += 1

            return (
                self.batch_queue.get_table_data(),
                self._get_progress_text(),
                self.t("status.added", count=added_count),
            )

        def start_processing(checkpoint: str) -> str:
            """Start batch processing."""
            if self.batch_queue.is_running():
                return self.t("status.already_running")

            pending_items = [
                item
                for item in self.batch_queue.items
                if item.status == BatchStatus.PENDING
            ]
            pending_count = len(pending_items)
            if pending_count == 0:
                return self.t("status.no_pending")

            if self._pending_merge_requires_checkpoint(pending_items):
                if not checkpoint or not Path(checkpoint).exists():
                    return self.i18n.t("errors.path_not_found")

            def process_item(item: BatchItem) -> None:
                """Process a single batch item."""
                if item.operation == "merge":
                    if not checkpoint:
                        raise ValueError("Checkpoint is required for merge operation")
                    cmd = build_merge_command(
                        item.input_path,
                        item.output_path,
                        checkpoint,
                    )
                elif item.operation == "extract":
                    cmd = build_extract_command(
                        item.input_path,
                        item.output_path,
                    )
                else:
                    raise ValueError(f"Unknown operation: {item.operation}")
                logger.debug("Resolved argv: %s", " ".join(cmd))

                item.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # Wait for completion
                if item.process.stdout:
                    for line in iter(item.process.stdout.readline, ""):
                        if not line:
                            break
                        # Could parse progress from output here
                        if item.process.poll() is not None:
                            break

                item.process.wait()

                if item.process.returncode != 0:
                    item.status = BatchStatus.FAILED
                    item.error = f"Exit code: {item.process.returncode}"

            self.batch_queue.start(process_item)
            return self.t("status.started", count=pending_count)

        def stop_processing() -> tuple[list[list[str]], str, str]:
            """Stop batch processing."""
            self.batch_queue.stop()
            return (
                self.batch_queue.get_table_data(),
                self._get_progress_text(),
                self.t("status.stopped"),
            )

        def clear_completed() -> tuple[list[list[str]], str, str]:
            """Clear completed items from queue."""
            count = self.batch_queue.clear_completed()
            return (
                self.batch_queue.get_table_data(),
                self._get_progress_text(),
                self.t("status.cleared", count=count),
            )

        def refresh_table() -> tuple[list[list[str]], str]:
            """Refresh queue table display."""
            return (
                self.batch_queue.get_table_data(),
                self._get_progress_text(),
            )

        # Wire events
        c["add_btn"].click(
            fn=add_files,
            inputs=[c["file_input"], c["output_dir"], c["operation"]],
            outputs=[c["queue_table"], c["progress_text"], c["status"]],
        )

        c["start_btn"].click(
            fn=start_processing,
            inputs=[c["checkpoint"]],
            outputs=[c["status"]],
        )

        c["stop_btn"].click(
            fn=stop_processing,
            outputs=[c["queue_table"], c["progress_text"], c["status"]],
        )

        c["clear_btn"].click(
            fn=clear_completed,
            outputs=[c["queue_table"], c["progress_text"], c["status"]],
        )

        c["refresh_btn"].click(
            fn=refresh_table,
            outputs=[c["queue_table"], c["progress_text"]],
        )

    @staticmethod
    def _pending_merge_requires_checkpoint(items: list[BatchItem]) -> bool:
        """Return True when pending queue contains at least one merge task."""
        return any(item.operation == "merge" for item in items)

    def _get_progress_text(self) -> str:
        """Get formatted progress text."""
        completed, total, percent = self.batch_queue.get_progress()
        return f"{completed} / {total} ({percent:.0f}%)"
