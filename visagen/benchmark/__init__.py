"""
Visagen Benchmark Module.

Comprehensive performance benchmarking for Visagen models:
    - Inference benchmarking (PyTorch, ONNX, TensorRT)
    - Training throughput measurement
    - Video merge pipeline benchmarking
    - GPU memory profiling

Example:
    >>> from visagen.benchmark import InferenceBenchmark, BenchmarkConfig
    >>> config = BenchmarkConfig(
    ...     checkpoint_path=Path("model.ckpt"),
    ...     batch_sizes=[1, 4, 8],
    ...     backends=["pytorch", "onnx"],
    ... )
    >>> benchmark = InferenceBenchmark(config)
    >>> results = benchmark.run()

CLI Usage:
    visagen-benchmark --mode inference -c model.ckpt
    visagen-benchmark --mode all -c model.ckpt --batch-sizes 1,4,8
"""

from visagen.benchmark.config import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkResult,
    SystemInfo,
)
from visagen.benchmark.profilers import (
    CUDATimer,
    MemoryTracker,
    TimingStats,
    TorchProfilerContext,
    warmup_cuda,
)
from visagen.benchmark.reporters import (
    BaseReporter,
    ConsoleReporter,
    JSONReporter,
    MarkdownReporter,
    get_reporter,
)

# Conditional imports for benchmarks
try:
    from visagen.benchmark.benchmarks import (
        InferenceBenchmark,
        MergerBenchmark,
        TrainingBenchmark,
    )

    __all__ = [
        # Config
        "BenchmarkConfig",
        "BenchmarkResult",
        "BenchmarkReport",
        "SystemInfo",
        # Profilers
        "CUDATimer",
        "MemoryTracker",
        "TimingStats",
        "TorchProfilerContext",
        "warmup_cuda",
        # Benchmarks
        "InferenceBenchmark",
        "TrainingBenchmark",
        "MergerBenchmark",
        # Reporters
        "BaseReporter",
        "ConsoleReporter",
        "JSONReporter",
        "MarkdownReporter",
        "get_reporter",
    ]
except ImportError:
    # Some benchmarks may have additional dependencies
    __all__ = [
        # Config
        "BenchmarkConfig",
        "BenchmarkResult",
        "BenchmarkReport",
        "SystemInfo",
        # Profilers
        "CUDATimer",
        "MemoryTracker",
        "TimingStats",
        "TorchProfilerContext",
        "warmup_cuda",
        # Reporters
        "BaseReporter",
        "ConsoleReporter",
        "JSONReporter",
        "MarkdownReporter",
        "get_reporter",
    ]
