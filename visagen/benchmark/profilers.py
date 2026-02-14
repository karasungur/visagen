"""
Performance profiling utilities.

Provides:
    - CUDATimer: CUDA event-based timing for accurate GPU timing
    - MemoryTracker: GPU memory usage tracking
    - TorchProfilerContext: PyTorch profiler wrapper
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimingStats:
    """Statistics from timing measurements.

    Attributes:
        times: List of timing measurements in milliseconds
        mean: Mean time
        std: Standard deviation
        min: Minimum time
        max: Maximum time
        p50: 50th percentile
        p95: 95th percentile
        p99: 99th percentile
    """

    times: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def compute(self) -> None:
        """Compute statistics from times."""
        if not self.times:
            return

        arr = np.array(self.times)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.min = float(np.min(arr))
        self.max = float(np.max(arr))
        self.p50 = float(np.percentile(arr, 50))
        self.p95 = float(np.percentile(arr, 95))
        self.p99 = float(np.percentile(arr, 99))


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing.

    Uses CUDA events for synchronization-free timing measurement.
    Falls back to CPU timing if CUDA is not available.

    Example:
        >>> timer = CUDATimer()
        >>> with timer.time():
        ...     model(input)
        >>> print(f"Elapsed: {timer.elapsed_ms:.2f} ms")
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize CUDA timer.

        Args:
            device: Device to use for timing
        """
        self.device = device
        self.elapsed_ms: float = 0.0
        self._start_event = None
        self._end_event = None
        self._use_cuda = False

        try:
            import torch

            if torch.cuda.is_available() and "cuda" in device:
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                self._use_cuda = True
        except ImportError:
            pass

        # Fallback for CPU timing
        self._cpu_start: float = 0.0
        self._cpu_end: float = 0.0

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        """Context manager for timing a block of code.

        Yields:
            None
        """
        if self._use_cuda:
            import torch

            self._start_event.record()  # type: ignore
            yield
            self._end_event.record()  # type: ignore
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)  # type: ignore
        else:
            import time

            self._cpu_start = time.perf_counter()
            yield
            self._cpu_end = time.perf_counter()
            self.elapsed_ms = (self._cpu_end - self._cpu_start) * 1000.0

    def reset(self) -> None:
        """Reset timer."""
        self.elapsed_ms = 0.0


class MemoryTracker:
    """GPU memory usage tracker.

    Tracks peak and allocated memory during execution.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.reset()
        >>> with tracker.track():
        ...     model(input)
        >>> print(f"Peak: {tracker.peak_mb:.1f} MB")
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize memory tracker.

        Args:
            device: Device to track memory on
        """
        self.device = device
        self.peak_mb: float = 0.0
        self.allocated_mb: float = 0.0
        self.reserved_mb: float = 0.0
        self._use_cuda = False

        try:
            import torch

            if torch.cuda.is_available() and "cuda" in device:
                self._use_cuda = True
        except ImportError:
            pass

    def reset(self) -> None:
        """Reset peak memory statistics."""
        if self._use_cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        self.peak_mb = 0.0
        self.allocated_mb = 0.0
        self.reserved_mb = 0.0

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        """Context manager for tracking memory usage.

        Yields:
            None
        """
        self.reset()
        yield
        self._collect_stats()

    def _collect_stats(self) -> None:
        """Collect memory statistics."""
        if self._use_cuda:
            import torch

            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    def get_current(self) -> tuple[float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (allocated_mb, reserved_mb)
        """
        if self._use_cuda:
            import torch

            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return allocated, reserved
        return 0.0, 0.0


class TorchProfilerContext:
    """PyTorch profiler context wrapper.

    Wraps torch.profiler for easy profiling with TensorBoard support.

    Example:
        >>> profiler = TorchProfilerContext(tensorboard_dir="./logs")
        >>> with profiler:
        ...     for batch in dataloader:
        ...         model(batch)
        ...         profiler.step()
    """

    def __init__(
        self,
        tensorboard_dir: str | None = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> None:
        """Initialize profiler context.

        Args:
            tensorboard_dir: Directory for TensorBoard logs
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_stack: Include Python stack traces
        """
        self.tensorboard_dir = tensorboard_dir
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self._profiler: Any | None = None
        self._available = False

        try:
            import torch.profiler  # noqa: F401

            self._available = True
        except ImportError:
            pass

    def __enter__(self) -> "TorchProfilerContext":
        """Enter profiler context."""
        if not self._available:
            return self

        import torch.profiler as profiler

        activities = [profiler.ProfilerActivity.CPU]

        try:
            import torch

            if torch.cuda.is_available():
                activities.append(profiler.ProfilerActivity.CUDA)
        except Exception:
            pass

        on_trace_ready = None
        if self.tensorboard_dir:
            on_trace_ready = profiler.tensorboard_trace_handler(self.tensorboard_dir)

        self._profiler = profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            on_trace_ready=on_trace_ready,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit profiler context."""
        if self._profiler is not None:
            self._profiler.__exit__(*args)

    def step(self) -> None:
        """Mark a profiler step."""
        if self._profiler is not None:
            self._profiler.step()

    def key_averages(self) -> Any:
        """Get key averages from profiler.

        Returns:
            Key averages table or None if not available
        """
        if self._profiler is not None:
            return self._profiler.key_averages()
        return None


def warmup_cuda() -> None:
    """Warm up CUDA for accurate timing.

    Performs a dummy operation to initialize CUDA context.
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Create and destroy a small tensor to warm up CUDA
            x = torch.randn(1, device="cuda")
            _ = x * 2
            torch.cuda.synchronize()
            del x
    except Exception:
        pass
