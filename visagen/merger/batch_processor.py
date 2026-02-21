"""
Parallel batch processing for Visagen merger.

Provides BatchProcessor for parallel frame processing using
multiprocessing with proper CUDA handling.

Features:
    - Producer-consumer pattern with queues
    - GPU distribution across workers
    - Graceful shutdown on errors
    - Progress tracking
"""

import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from visagen.merger.frame_processor import FrameProcessorConfig


@dataclass
class WorkItem:
    """
    Single work item for processing.

    Attributes:
        frame_idx: Frame index.
        frame_path: Path to frame file (if file-based).
        frame_data: Frame data (if in-memory).
    """

    frame_idx: int
    frame_path: Path | None = None
    frame_data: np.ndarray | None = None


@dataclass
class WorkResult:
    """
    Result from worker process.

    Attributes:
        frame_idx: Frame index.
        success: Whether processing succeeded.
        output_path: Path to output file.
        error: Error message if failed.
        processing_time: Time taken in seconds.
        faces_detected: Number of faces detected.
        faces_swapped: Number of faces swapped.
    """

    frame_idx: int
    success: bool
    output_path: Path | None = None
    error: str | None = None
    processing_time: float = 0.0
    faces_detected: int = 0
    faces_swapped: int = 0


@dataclass
class ProcessingStats:
    """
    Batch processing statistics.

    Attributes:
        total_items: Total items to process.
        completed_items: Successfully completed items.
        failed_items: Failed items.
        elapsed_time: Total elapsed time.
        items_per_second: Processing rate.
    """

    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    elapsed_time: float = 0.0
    items_per_second: float = 0.0


def _worker_process(
    worker_id: int,
    checkpoint_path: str,
    config_dict: dict,
    device: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    shutdown_event: Any,
) -> None:
    """
    Worker process function.

    Runs in a separate process, loading its own model copy
    and processing frames from the input queue.

    Args:
        worker_id: Unique worker identifier.
        checkpoint_path: Path to model checkpoint.
        config_dict: FrameProcessorConfig as dict.
        device: Torch device string.
        input_queue: Queue for receiving work items.
        output_queue: Queue for sending results.
        shutdown_event: Event signaling shutdown.
    """
    import cv2
    import torch

    from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

    # Set up device
    if "cuda" in device:
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.set_device(gpu_id)

    # Load model
    try:
        config = FrameProcessorConfig(**config_dict)
        processor = FrameProcessor(
            model=checkpoint_path,
            config=config,
            device=device,
        )
        logger.info(f"Worker {worker_id} initialized on {device}")
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to initialize: {e}")
        return

    # Process loop
    while not shutdown_event.is_set():
        try:
            # Get work item with timeout
            try:
                item_data = input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item_data is None:
                # Poison pill - shutdown
                break

            frame_idx, frame_path, output_path = item_data
            start_time = time.time()

            try:
                # Load frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise ValueError(f"Failed to load frame: {frame_path}")

                # Process
                result = processor.process_frame(frame, frame_idx=frame_idx)

                # Save output
                cv2.imwrite(str(output_path), result.output_image)

                # Send result
                output_queue.put(
                    WorkResult(
                        frame_idx=frame_idx,
                        success=True,
                        output_path=Path(output_path),
                        processing_time=time.time() - start_time,
                        faces_detected=result.faces_detected,
                        faces_swapped=result.faces_swapped,
                    )
                )

            except Exception as e:
                output_queue.put(
                    WorkResult(
                        frame_idx=frame_idx,
                        success=False,
                        error=str(e),
                        processing_time=time.time() - start_time,
                    )
                )

        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            continue

    logger.info(f"Worker {worker_id} shutting down")


class BatchProcessor:
    """
    Parallel frame processor using multiprocessing.

    Distributes work across multiple GPU/CPU workers
    for efficient batch processing.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: FrameProcessorConfig options.
        num_workers: Number of worker processes.
        gpu_ids: List of GPU IDs to distribute across.
        queue_size: Size of work queues.

    Example:
        >>> processor = BatchProcessor("model.ckpt", num_workers=4)
        >>> results = processor.process_batch(items, output_dir)
        >>> processor.shutdown()
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config: Optional["FrameProcessorConfig"] = None,
        num_workers: int = 4,
        gpu_ids: list[int] | None = None,
        queue_size: int = 32,
    ) -> None:
        from visagen.merger.frame_processor import FrameProcessorConfig

        self.checkpoint_path = Path(checkpoint_path)
        self.config = config or FrameProcessorConfig()
        self.num_workers = num_workers
        self.queue_size = queue_size

        # Determine devices
        self.devices = self._get_devices(gpu_ids)

        # Process management
        self._workers: list[Any] = []
        self._input_queue: Any | None = None
        self._output_queue: Any | None = None
        self._shutdown_event: Any | None = None
        self._started = False

    def _require_input_queue(self) -> Any:
        if self._input_queue is None:
            raise RuntimeError("Input queue is not initialized")
        return self._input_queue

    def _require_output_queue(self) -> Any:
        if self._output_queue is None:
            raise RuntimeError("Output queue is not initialized")
        return self._output_queue

    def _require_shutdown_event(self) -> Any:
        if self._shutdown_event is None:
            raise RuntimeError("Shutdown event is not initialized")
        return self._shutdown_event

    def _get_devices(self, gpu_ids: list[int] | None) -> list[str]:
        """Get list of device strings for workers."""
        import torch

        if not torch.cuda.is_available():
            return ["cpu"] * self.num_workers

        if gpu_ids is None:
            # Use all available GPUs
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))

        if not gpu_ids:
            return ["cpu"] * self.num_workers

        # Distribute workers across GPUs round-robin
        devices = []
        for i in range(self.num_workers):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            devices.append(f"cuda:{gpu_id}")

        return devices

    def _start_workers(self) -> None:
        """Start worker processes."""
        if self._started:
            return

        # Use spawn method for CUDA compatibility
        ctx = mp.get_context("spawn")

        self._input_queue = ctx.Queue(maxsize=self.queue_size)
        self._output_queue = ctx.Queue()
        self._shutdown_event = ctx.Event()
        input_queue = self._require_input_queue()
        output_queue = self._require_output_queue()
        shutdown_event = self._require_shutdown_event()

        # Convert config to dict for pickling
        from dataclasses import asdict

        config_dict = asdict(self.config)

        for worker_id in range(self.num_workers):
            device = self.devices[worker_id]

            worker = ctx.Process(
                target=_worker_process,
                args=(
                    worker_id,
                    str(self.checkpoint_path),
                    config_dict,
                    device,
                    input_queue,
                    output_queue,
                    shutdown_event,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        self._started = True
        logger.info(f"Started {self.num_workers} workers")

    def process_batch(
        self,
        items: list[WorkItem],
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[WorkResult]:
        """
        Process batch of frames in parallel.

        Args:
            items: List of work items to process.
            output_dir: Directory for output frames.
            progress_callback: Progress update callback.

        Returns:
            List of WorkResult for each item.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._start_workers()
        input_queue = self._require_input_queue()
        output_queue = self._require_output_queue()

        # Submit all work items
        for item in items:
            if item.frame_path is None:
                continue

            output_path = output_dir / f"{item.frame_idx:06d}.png"
            input_queue.put((item.frame_idx, str(item.frame_path), str(output_path)))

        # Collect results
        results = []
        pending = len(items)

        while pending > 0:
            try:
                result = output_queue.get(timeout=60.0)
                results.append(result)
                pending -= 1

                if progress_callback:
                    progress_callback(len(results), len(items))

            except queue.Empty:
                # Check if workers are still alive
                alive_workers = sum(1 for w in self._workers if w.is_alive())
                if alive_workers == 0:
                    logger.error("All workers died")
                    break

        return results

    def process_streaming(
        self,
        frame_generator: Generator[tuple[int, np.ndarray], None, None],
        output_dir: Path,
        total_frames: int | None = None,
    ) -> Generator[WorkResult, None, None]:
        """
        Process frames from generator with streaming output.

        Args:
            frame_generator: Generator yielding (idx, frame) tuples.
            output_dir: Directory for output frames.
            total_frames: Total frame count for progress.

        Yields:
            WorkResult as each frame completes.
        """
        import tempfile

        import cv2

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._start_workers()
        input_queue = self._require_input_queue()
        output_queue = self._require_output_queue()

        # Create temp dir for input frames
        temp_input = Path(tempfile.mkdtemp(prefix="visagen_input_"))

        try:
            submitted = 0
            received = 0

            # Submit frames as they arrive
            for frame_idx, frame in frame_generator:
                # Save frame to temp
                input_path = temp_input / f"{frame_idx:06d}.png"
                cv2.imwrite(str(input_path), frame)

                # Submit
                output_path = output_dir / f"{frame_idx:06d}.png"
                input_queue.put((frame_idx, str(input_path), str(output_path)))
                submitted += 1

                # Try to get results (non-blocking)
                while True:
                    try:
                        result = output_queue.get_nowait()
                        received += 1
                        yield result
                    except queue.Empty:
                        break

            # Wait for remaining results
            while received < submitted:
                try:
                    result = output_queue.get(timeout=60.0)
                    received += 1
                    yield result
                except queue.Empty:
                    break

        finally:
            # Clean up temp input
            import shutil

            shutil.rmtree(temp_input, ignore_errors=True)

    def shutdown(self) -> None:
        """Gracefully shutdown all workers."""
        if not self._started:
            return

        # Signal shutdown
        shutdown_event = self._require_shutdown_event()
        input_queue = self._require_input_queue()
        shutdown_event.set()

        # Send poison pills
        for _ in self._workers:
            try:
                input_queue.put(None, timeout=1.0)
            except Exception:
                pass

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

        self._workers.clear()
        self._started = False
        logger.info("All workers shut down")

    def __enter__(self) -> "BatchProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
