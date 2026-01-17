"""Batch processing queue management."""

from __future__ import annotations

import logging
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch item status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Single batch queue item."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_path: str = ""
    output_path: str = ""
    operation: str = "merge"  # merge, extract, etc.
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)

    def to_row(self) -> list[str]:
        """Convert to table row for display."""
        status_icons = {
            BatchStatus.PENDING: "⏳",
            BatchStatus.RUNNING: "🔄",
            BatchStatus.COMPLETED: "✅",
            BatchStatus.FAILED: "❌",
            BatchStatus.CANCELLED: "🚫",
        }
        return [
            self.id,
            Path(self.input_path).name,
            f"{status_icons.get(self.status, '?')} {self.status.value}",
            f"{self.progress:.0f}%",
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "operation": self.operation,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchItem:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            operation=data.get("operation", "merge"),
            status=BatchStatus(data.get("status", "pending")),
            progress=data.get("progress", 0.0),
            error=data.get("error"),
        )


class BatchQueue:
    """Thread-safe batch processing queue."""

    def __init__(self, max_parallel: int = 1) -> None:
        """
        Initialize batch queue.

        Args:
            max_parallel: Maximum number of parallel operations (currently only 1 supported).
        """
        self.items: list[BatchItem] = []
        self.max_parallel = max_parallel
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def add(
        self,
        input_path: str,
        output_path: str,
        operation: str = "merge",
    ) -> str:
        """
        Add item to queue.

        Args:
            input_path: Path to input file.
            output_path: Path for output file.
            operation: Operation type (merge, extract).

        Returns:
            Item ID.
        """
        with self._lock:
            item = BatchItem(
                input_path=input_path,
                output_path=output_path,
                operation=operation,
            )
            self.items.append(item)
            return item.id

    def add_multiple(
        self,
        paths: list[tuple[str, str]],
        operation: str = "merge",
    ) -> list[str]:
        """
        Add multiple items to queue.

        Args:
            paths: List of (input_path, output_path) tuples.
            operation: Operation type for all items.

        Returns:
            List of item IDs.
        """
        ids = []
        for input_path, output_path in paths:
            ids.append(self.add(input_path, output_path, operation))
        return ids

    def remove(self, item_id: str) -> bool:
        """
        Remove pending item from queue.

        Args:
            item_id: ID of item to remove.

        Returns:
            True if removed, False otherwise.
        """
        with self._lock:
            for i, item in enumerate(self.items):
                if item.id == item_id and item.status == BatchStatus.PENDING:
                    del self.items[i]
                    return True
            return False

    def clear_completed(self) -> int:
        """
        Remove completed/failed/cancelled items.

        Returns:
            Number of items removed.
        """
        with self._lock:
            before = len(self.items)
            self.items = [
                item
                for item in self.items
                if item.status in (BatchStatus.PENDING, BatchStatus.RUNNING)
            ]
            return before - len(self.items)

    def clear_all(self) -> int:
        """
        Clear all items (stops running items first).

        Returns:
            Number of items cleared.
        """
        self.stop()
        with self._lock:
            count = len(self.items)
            self.items.clear()
            return count

    def get_table_data(self) -> list[list[str]]:
        """
        Get data for Gradio Dataframe display.

        Returns:
            List of row data for table.
        """
        with self._lock:
            return [item.to_row() for item in self.items]

    def get_progress(self) -> tuple[int, int, float]:
        """
        Get overall progress.

        Returns:
            Tuple of (completed_count, total_count, percent_complete).
        """
        with self._lock:
            total = len(self.items)
            completed = sum(
                1 for item in self.items if item.status == BatchStatus.COMPLETED
            )
            if total == 0:
                return 0, 0, 0.0
            return completed, total, (completed / total) * 100

    def is_running(self) -> bool:
        """Check if queue is currently processing."""
        return self._running

    def start(
        self,
        process_fn: Callable[[BatchItem], None],
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """
        Start processing queue in background thread.

        Args:
            process_fn: Function to process each item.
            on_complete: Optional callback when all items complete.

        Returns:
            True if started, False if already running.
        """
        if self._running:
            return False

        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            args=(process_fn, on_complete),
            daemon=True,
        )
        self._worker_thread.start()
        return True

    def stop(self) -> None:
        """Stop processing and cancel running items."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        with self._lock:
            for item in self.items:
                if item.status == BatchStatus.RUNNING and item.process:
                    try:
                        item.process.terminate()
                        item.process.wait(timeout=5)
                    except Exception as e:
                        logger.debug(f"Process terminate failed, trying kill: {e}")
                        try:
                            item.process.kill()
                        except Exception as kill_err:
                            logger.warning(f"Failed to kill process: {kill_err}")
                    item.status = BatchStatus.CANCELLED

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)

    def _process_loop(
        self,
        process_fn: Callable[[BatchItem], None],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Main processing loop (runs in background thread)."""
        while self._running and not self._stop_event.is_set():
            item = self._get_next_pending()
            if item is None:
                # No more items, stop
                with self._lock:
                    self._running = False
                if on_complete:
                    on_complete()
                break

            try:
                with self._lock:
                    item.status = BatchStatus.RUNNING
                process_fn(item)
                if item.status == BatchStatus.RUNNING:
                    item.status = BatchStatus.COMPLETED
                    item.progress = 100.0
            except Exception as e:
                item.status = BatchStatus.FAILED
                item.error = str(e)

    def _get_next_pending(self) -> BatchItem | None:
        """Get next pending item from queue."""
        with self._lock:
            for item in self.items:
                if item.status == BatchStatus.PENDING:
                    return item
            return None

    def get_item(self, item_id: str) -> BatchItem | None:
        """Get item by ID."""
        with self._lock:
            for item in self.items:
                if item.id == item_id:
                    return item
            return None

    def update_progress(self, item_id: str, progress: float) -> None:
        """Update item progress."""
        with self._lock:
            for item in self.items:
                if item.id == item_id:
                    item.progress = min(100.0, max(0.0, progress))
                    break
