"""
Inference benchmark runner.

Benchmarks model inference across different backends:
    - PyTorch (native)
    - ONNX Runtime
    - TensorRT

Measures throughput, latency percentiles, and GPU memory usage.
"""

from visagen.benchmark.config import BenchmarkConfig, BenchmarkResult
from visagen.benchmark.profilers import (
    CUDATimer,
    MemoryTracker,
    TimingStats,
    warmup_cuda,
)


class InferenceBenchmark:
    """Benchmark runner for model inference.

    Supports multiple backends and measures:
        - Throughput (images/second)
        - Latency (mean, p50, p95, p99)
        - GPU memory usage

    Example:
        >>> config = BenchmarkConfig(
        ...     checkpoint_path=Path("model.ckpt"),
        ...     batch_sizes=[1, 4, 8],
        ...     input_sizes=[256, 512],
        ...     backends=["pytorch", "onnx"],
        ... )
        >>> benchmark = InferenceBenchmark(config)
        >>> results = benchmark.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize inference benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._models: dict = {}  # Cached models per backend

    def run(self) -> list[BenchmarkResult]:
        """Run inference benchmarks for all configurations.

        Returns:
            List of benchmark results
        """
        results = []

        # Warm up CUDA
        warmup_cuda()

        for backend in self.config.backends:
            if not self._check_backend_available(backend):
                print(f"Warning: Backend '{backend}' not available, skipping")
                continue

            for input_size in self.config.input_sizes:
                for batch_size in self.config.batch_sizes:
                    try:
                        result = self._benchmark_config(
                            backend=backend,
                            input_size=input_size,
                            batch_size=batch_size,
                        )
                        results.append(result)
                    except Exception as e:
                        print(
                            f"Error benchmarking {backend} bs={batch_size} "
                            f"size={input_size}: {e}"
                        )

        return results

    def _check_backend_available(self, backend: str) -> bool:
        """Check if a backend is available.

        Args:
            backend: Backend name

        Returns:
            True if available
        """
        if backend == "pytorch":
            return True

        if backend == "onnx":
            try:
                import onnxruntime  # noqa: F401

                return True
            except ImportError:
                return False

        if backend == "tensorrt":
            try:
                import tensorrt  # noqa: F401

                return True
            except ImportError:
                return False

        return False

    def _get_model(self, backend: str, input_size: int) -> object:
        """Get or load model for a backend.

        Args:
            backend: Backend name
            input_size: Input resolution

        Returns:
            Model object
        """
        cache_key = f"{backend}_{input_size}"
        if cache_key in self._models:
            return self._models[cache_key]

        if backend == "pytorch":
            model = self._load_pytorch_model()
        elif backend == "onnx":
            model = self._load_onnx_model(input_size)
        elif backend == "tensorrt":
            model = self._load_tensorrt_model(input_size)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._models[cache_key] = model
        return model

    def _load_pytorch_model(self) -> object:
        """Load PyTorch model from checkpoint.

        Returns:
            PyTorch model
        """
        import torch

        from visagen.training.dfl_module import DFLModule

        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path required for PyTorch benchmark")

        model = DFLModule.load_from_checkpoint(
            self.config.checkpoint_path,
            map_location="cpu",
        )
        model.eval()

        if self.config.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        return model

    def _load_onnx_model(self, input_size: int) -> object:
        """Load ONNX model.

        Args:
            input_size: Input resolution

        Returns:
            ONNX Runtime session
        """
        from visagen.export import ONNXRunner

        # Check for exported ONNX model
        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path required")

        onnx_path = self.config.checkpoint_path.with_suffix(".onnx")
        if not onnx_path.exists():
            # Export the model first
            from visagen.export import ExportConfig, ONNXExporter

            exporter = ONNXExporter(
                checkpoint_path=self.config.checkpoint_path,
                output_path=onnx_path,
                config=ExportConfig(input_size=input_size),
            )
            exporter.export()

        device = "cuda" if self.config.device == "cuda" else "cpu"
        return ONNXRunner(onnx_path, device=device)

    def _load_tensorrt_model(self, input_size: int) -> object:
        """Load TensorRT model.

        Args:
            input_size: Input resolution

        Returns:
            TensorRT runner
        """
        from visagen.export import TensorRTRunner

        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path required")

        # Check for TensorRT engine
        engine_path = self.config.checkpoint_path.with_suffix(".engine")
        if not engine_path.exists():
            # Need ONNX first
            onnx_path = self.config.checkpoint_path.with_suffix(".onnx")
            if not onnx_path.exists():
                from visagen.export import ExportConfig, ONNXExporter

                exporter = ONNXExporter(
                    checkpoint_path=self.config.checkpoint_path,
                    output_path=onnx_path,
                    config=ExportConfig(input_size=input_size),
                )
                exporter.export()

            # Build TensorRT engine
            from visagen.export import TensorRTBuilder

            builder = TensorRTBuilder(
                onnx_path=onnx_path,
                engine_path=engine_path,
            )
            builder.build()

        return TensorRTRunner(engine_path)

    def _benchmark_config(
        self,
        backend: str,
        input_size: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark a specific configuration.

        Args:
            backend: Backend to use
            input_size: Input resolution
            batch_size: Batch size

        Returns:
            Benchmark result
        """
        import torch

        # Get model
        model = self._get_model(backend, input_size)

        # Create synthetic input
        device = self.config.device if self.config.device != "auto" else "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        dummy_input = torch.randn(
            batch_size,
            3,
            input_size,
            input_size,
            device=device,
            dtype=torch.float32,
        )

        # Warmup
        for _ in range(self.config.num_warmup):
            self._run_inference(model, dummy_input, backend)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed iterations
        timer = CUDATimer(device=device)
        memory_tracker = MemoryTracker(device=device)
        timing_stats = TimingStats()

        memory_tracker.reset()

        for _ in range(self.config.num_iterations):
            with timer.time():
                self._run_inference(model, dummy_input, backend)
            timing_stats.times.append(timer.elapsed_ms)

        # Collect memory stats
        memory_tracker._collect_stats()
        timing_stats.compute()

        # Calculate throughput
        total_images = batch_size * self.config.num_iterations
        total_time_sec = sum(timing_stats.times) / 1000.0
        throughput = total_images / total_time_sec if total_time_sec > 0 else 0

        name = f"inference_{backend}_bs{batch_size}_{input_size}"

        return BenchmarkResult(
            name=name,
            backend=backend,
            batch_size=batch_size,
            input_size=input_size,
            throughput=throughput,
            latency_mean=timing_stats.mean,
            latency_std=timing_stats.std,
            latency_p50=timing_stats.p50,
            latency_p95=timing_stats.p95,
            latency_p99=timing_stats.p99,
            gpu_memory_peak=memory_tracker.peak_mb,
            gpu_memory_allocated=memory_tracker.allocated_mb,
        )

    def _run_inference(self, model: object, input_tensor: object, backend: str) -> None:
        """Run inference on model.

        Args:
            model: Model object
            input_tensor: Input tensor
            backend: Backend type
        """
        import torch

        if backend == "pytorch":
            with torch.no_grad():
                _ = model(input_tensor)  # type: ignore
        elif backend == "onnx":
            _ = model(input_tensor)  # type: ignore
        elif backend == "tensorrt":
            _ = model(input_tensor)  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
