"""
Benchmark configuration and result dataclasses.

Provides:
    - BenchmarkConfig: Configuration for benchmark runs
    - BenchmarkResult: Individual benchmark result
    - BenchmarkReport: Collection of results with system info
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        mode: Benchmark mode ('inference', 'training', 'merge', 'all')
        checkpoint_path: Path to model checkpoint
        input_sizes: List of input resolutions to test
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        backends: List of backends to test ('pytorch', 'onnx', 'tensorrt')
        device: Device to run benchmarks on
        output_formats: Output format list ('console', 'json', 'markdown')
        output_dir: Directory to save reports
        profile: Enable PyTorch profiler
        tensorboard_dir: Directory for TensorBoard logs
    """

    mode: Literal["inference", "training", "merge", "all"] = "inference"
    checkpoint_path: Path | None = None
    input_sizes: list[int] = field(default_factory=lambda: [256])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_warmup: int = 10
    num_iterations: int = 100
    backends: list[str] = field(default_factory=lambda: ["pytorch"])
    device: str = "cuda"
    output_formats: list[str] = field(default_factory=lambda: ["console"])
    output_dir: Path | None = None
    profile: bool = False
    tensorboard_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.tensorboard_dir is not None:
            self.tensorboard_dir = Path(self.tensorboard_dir)


@dataclass
class BenchmarkResult:
    """Single benchmark result.

    Attributes:
        name: Benchmark name (e.g., 'inference_pytorch_bs4_256')
        backend: Backend used ('pytorch', 'onnx', 'tensorrt')
        batch_size: Batch size used
        input_size: Input resolution
        throughput: Images per second
        latency_mean: Mean latency in milliseconds
        latency_std: Standard deviation of latency
        latency_p50: 50th percentile latency
        latency_p95: 95th percentile latency
        latency_p99: 99th percentile latency
        gpu_memory_peak: Peak GPU memory usage in MB
        gpu_memory_allocated: Allocated GPU memory in MB
        metadata: Additional metadata
    """

    name: str
    backend: str
    batch_size: int
    input_size: int
    throughput: float  # images/sec
    latency_mean: float  # ms
    latency_std: float = 0.0  # ms
    latency_p50: float = 0.0  # ms
    latency_p95: float = 0.0  # ms
    latency_p99: float = 0.0  # ms
    gpu_memory_peak: float = 0.0  # MB
    gpu_memory_allocated: float = 0.0  # MB
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "backend": self.backend,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "throughput": self.throughput,
            "latency_mean": self.latency_mean,
            "latency_std": self.latency_std,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "gpu_memory_peak": self.gpu_memory_peak,
            "gpu_memory_allocated": self.gpu_memory_allocated,
            "metadata": self.metadata,
        }


@dataclass
class SystemInfo:
    """System information for benchmark report.

    Attributes:
        platform: Operating system
        python_version: Python version
        pytorch_version: PyTorch version
        cuda_version: CUDA version (if available)
        cudnn_version: cuDNN version (if available)
        gpu_name: GPU name (if available)
        gpu_count: Number of GPUs
        cpu_name: CPU name
        cpu_count: Number of CPU cores
    """

    platform: str = ""
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    cpu_name: str = ""
    cpu_count: int = 0

    @classmethod
    def collect(cls) -> "SystemInfo":
        """Collect system information."""
        import platform
        import sys

        info = cls(
            platform=platform.system(),
            python_version=sys.version.split()[0],
            cpu_name=platform.processor() or "Unknown",
            cpu_count=platform.os.cpu_count() or 0,  # type: ignore
        )

        # PyTorch info
        try:
            import torch

            info.pytorch_version = torch.__version__

            if torch.cuda.is_available():
                info.cuda_version = torch.version.cuda or ""
                info.cudnn_version = str(torch.backends.cudnn.version())
                info.gpu_count = torch.cuda.device_count()
                if info.gpu_count > 0:
                    info.gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return info

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "cpu_name": self.cpu_name,
            "cpu_count": self.cpu_count,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        timestamp: When the benchmark was run
        config: Benchmark configuration used
        system_info: System information
        results: List of benchmark results
    """

    timestamp: datetime = field(default_factory=datetime.now)
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    system_info: SystemInfo = field(default_factory=SystemInfo)
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
