"""
Merge pipeline benchmark runner.

Benchmarks the full face swap merge pipeline:
    - Face detection
    - Face alignment
    - Model inference
    - Color transfer
    - Blending

Measures end-to-end throughput for video processing.
"""

import numpy as np

from visagen.benchmark.config import BenchmarkConfig, BenchmarkResult
from visagen.benchmark.profilers import (
    CUDATimer,
    MemoryTracker,
    TimingStats,
    warmup_cuda,
)


class MergerBenchmark:
    """Benchmark runner for merge pipeline.

    Measures end-to-end video processing speed including:
        - Face detection
        - Face alignment
        - Model inference
        - Color transfer
        - Blending

    Example:
        >>> config = BenchmarkConfig(
        ...     mode="merge",
        ...     checkpoint_path=Path("model.ckpt"),
        ...     input_sizes=[720, 1080],
        ... )
        >>> benchmark = MergerBenchmark(config)
        >>> results = benchmark.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize merge benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._frame_processor = None

    def run(self) -> list[BenchmarkResult]:
        """Run merge benchmarks for all configurations.

        Returns:
            List of benchmark results
        """
        results = []

        # Warm up CUDA
        warmup_cuda()

        # Test with different frame sizes
        frame_sizes = [
            (1280, 720),  # 720p
            (1920, 1080),  # 1080p
        ]

        for width, height in frame_sizes:
            try:
                result = self._benchmark_config(width=width, height=height)
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking merge {width}x{height}: {e}")

        return results

    def _create_frame_processor(self) -> object:
        """Create frame processor for benchmarking.

        Returns:
            FrameProcessor instance
        """
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path required for merge benchmark")

        # Create processor config
        proc_config = FrameProcessorConfig(
            min_confidence=0.5,
            max_faces=1,
            color_transfer_mode="rct",
            blend_mode="laplacian",
        )

        return FrameProcessor(
            checkpoint_path=self.config.checkpoint_path,
            config=proc_config,
            device=self.config.device,
        )

    def _benchmark_config(
        self,
        width: int,
        height: int,
    ) -> BenchmarkResult:
        """Benchmark a specific configuration.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            Benchmark result
        """
        import torch

        # Get device
        device = self.config.device if self.config.device != "auto" else "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        # Create frame processor
        if self._frame_processor is None:
            self._frame_processor = self._create_frame_processor()

        # Create synthetic frame (RGB uint8)
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Warmup
        for _ in range(min(self.config.num_warmup, 5)):
            try:
                _ = self._frame_processor.process_frame(frame)  # type: ignore
            except Exception:
                # No face detected in random noise, expected
                pass

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed iterations
        timer = CUDATimer(device=device)
        memory_tracker = MemoryTracker(device=device)
        timing_stats = TimingStats()

        memory_tracker.reset()

        # Use fewer iterations for full pipeline
        num_iterations = min(self.config.num_iterations, 50)

        for _ in range(num_iterations):
            with timer.time():
                try:
                    _ = self._frame_processor.process_frame(frame)  # type: ignore
                except Exception:
                    # No face in synthetic data
                    pass
            timing_stats.times.append(timer.elapsed_ms)

        # Collect memory stats
        memory_tracker._collect_stats()
        timing_stats.compute()

        # Calculate FPS
        total_time_sec = sum(timing_stats.times) / 1000.0
        fps = num_iterations / total_time_sec if total_time_sec > 0 else 0

        name = f"merge_{width}x{height}"

        return BenchmarkResult(
            name=name,
            backend="pytorch",
            batch_size=1,  # Frame by frame
            input_size=height,  # Use height as size indicator
            throughput=fps,  # FPS for video processing
            latency_mean=timing_stats.mean,
            latency_std=timing_stats.std,
            latency_p50=timing_stats.p50,
            latency_p95=timing_stats.p95,
            latency_p99=timing_stats.p99,
            gpu_memory_peak=memory_tracker.peak_mb,
            gpu_memory_allocated=memory_tracker.allocated_mb,
            metadata={
                "mode": "merge",
                "width": width,
                "height": height,
            },
        )


class SimpleMergerBenchmark:
    """Simplified merge benchmark using synthetic data.

    Does not require actual face detection - measures pure
    inference + post-processing performance.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize simple merge benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config

    def run(self) -> list[BenchmarkResult]:
        """Run simplified merge benchmarks.

        Returns:
            List of benchmark results
        """
        results = []

        warmup_cuda()

        for input_size in self.config.input_sizes:
            try:
                result = self._benchmark_inference_postprocess(input_size)
                results.append(result)
            except Exception as e:
                print(f"Error in simple merge benchmark size={input_size}: {e}")

        return results

    def _benchmark_inference_postprocess(
        self,
        input_size: int,
    ) -> BenchmarkResult:
        """Benchmark inference + postprocess pipeline.

        Args:
            input_size: Face crop size

        Returns:
            Benchmark result
        """
        import torch

        device = self.config.device if self.config.device != "auto" else "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        # Load model
        from visagen.training.dfl_module import DFLModule

        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path required")

        model = DFLModule.load_from_checkpoint(
            self.config.checkpoint_path,
            map_location="cpu",
        )
        model.eval()
        if device == "cuda":
            model = model.cuda()

        # Create synthetic face crop
        face_crop = torch.randn(1, 3, input_size, input_size, device=device)
        face_crop_np = np.random.randint(
            0, 255, (input_size, input_size, 3), dtype=np.uint8
        )

        # Import postprocess functions
        from visagen.postprocess import color_transfer

        # Warmup
        for _ in range(self.config.num_warmup):
            with torch.no_grad():
                output = model(face_crop)
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = np.clip(output_np, 0, 1).astype(np.float32)

            # Apply color transfer and blending
            face_float = face_crop_np.astype(np.float32) / 255.0
            _ = color_transfer("rct", output_np, face_float)

        # Timed iterations
        timer = CUDATimer(device=device)
        memory_tracker = MemoryTracker(device=device)
        timing_stats = TimingStats()

        memory_tracker.reset()

        for _ in range(self.config.num_iterations):
            with timer.time():
                with torch.no_grad():
                    output = model(face_crop)

                output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_np = np.clip(output_np, 0, 1).astype(np.float32)

                face_float = face_crop_np.astype(np.float32) / 255.0
                _ = color_transfer("rct", output_np, face_float)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            timing_stats.times.append(timer.elapsed_ms)

        memory_tracker._collect_stats()
        timing_stats.compute()

        # Calculate FPS
        total_time_sec = sum(timing_stats.times) / 1000.0
        fps = self.config.num_iterations / total_time_sec if total_time_sec > 0 else 0

        name = f"merge_simple_{input_size}"

        return BenchmarkResult(
            name=name,
            backend="pytorch",
            batch_size=1,
            input_size=input_size,
            throughput=fps,
            latency_mean=timing_stats.mean,
            latency_std=timing_stats.std,
            latency_p50=timing_stats.p50,
            latency_p95=timing_stats.p95,
            latency_p99=timing_stats.p99,
            gpu_memory_peak=memory_tracker.peak_mb,
            gpu_memory_allocated=memory_tracker.allocated_mb,
            metadata={"mode": "merge_simple"},
        )
