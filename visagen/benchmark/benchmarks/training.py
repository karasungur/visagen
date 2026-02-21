"""
Training benchmark runner.

Benchmarks training throughput with synthetic data:
    - Forward pass timing
    - Backward pass timing
    - Optimizer step timing
    - GPU memory usage

Measures images per second for different batch sizes and resolutions.
"""

from typing import Any

from visagen.benchmark.config import BenchmarkConfig, BenchmarkResult
from visagen.benchmark.profilers import (
    CUDATimer,
    MemoryTracker,
    TimingStats,
    warmup_cuda,
)


class TrainingBenchmark:
    """Benchmark runner for training throughput.

    Measures training speed using synthetic data to isolate
    model performance from data loading.

    Example:
        >>> config = BenchmarkConfig(
        ...     mode="training",
        ...     batch_sizes=[4, 8, 16],
        ...     input_sizes=[256, 512],
        ... )
        >>> benchmark = TrainingBenchmark(config)
        >>> results = benchmark.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize training benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._model = None
        self._optimizer = None

    def run(self) -> list[BenchmarkResult]:
        """Run training benchmarks for all configurations.

        Returns:
            List of benchmark results
        """
        results = []

        # Warm up CUDA
        warmup_cuda()

        for input_size in self.config.input_sizes:
            for batch_size in self.config.batch_sizes:
                try:
                    result = self._benchmark_config(
                        input_size=input_size,
                        batch_size=batch_size,
                    )
                    results.append(result)
                except Exception as e:
                    print(
                        f"Error benchmarking training bs={batch_size} "
                        f"size={input_size}: {e}"
                    )

        return results

    def _create_model(self, input_size: int) -> tuple:
        """Create model and optimizer for benchmarking.

        Args:
            input_size: Input resolution

        Returns:
            Tuple of (model, optimizer)
        """
        import torch

        from visagen.models import ConvNeXtEncoder, Decoder
        from visagen.training.losses import DSSIMLoss

        device = self.config.device if self.config.device != "auto" else "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        # Create encoder-decoder model
        encoder = ConvNeXtEncoder(
            in_channels=3,
            dims=[64, 128, 256, 512],
            depths=[2, 2, 4, 2],
        )
        decoder = Decoder(
            latent_channels=512,
            dims=[512, 256, 128, 64],
            skip_dims=[256, 128, 64, 64],
            out_channels=3,
        )

        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Combined parameters
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)

        # Loss function
        loss_fn = DSSIMLoss()

        return (encoder, decoder, optimizer, loss_fn, device)

    def _benchmark_config(
        self,
        input_size: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark a specific configuration.

        Args:
            input_size: Input resolution
            batch_size: Batch size

        Returns:
            Benchmark result
        """
        import torch

        # Create model
        encoder, decoder, optimizer, loss_fn, device = self._create_model(input_size)

        # Create synthetic data
        src_batch = torch.randn(
            batch_size,
            3,
            input_size,
            input_size,
            device=device,
            dtype=torch.float32,
        )
        dst_batch = torch.randn(
            batch_size,
            3,
            input_size,
            input_size,
            device=device,
            dtype=torch.float32,
        )

        # Warmup
        for _ in range(self.config.num_warmup):
            self._training_step(
                encoder,
                decoder,
                optimizer,
                loss_fn,
                src_batch,
                dst_batch,
            )

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
                self._training_step(
                    encoder,
                    decoder,
                    optimizer,
                    loss_fn,
                    src_batch,
                    dst_batch,
                )
            timing_stats.times.append(timer.elapsed_ms)

        # Collect memory stats
        memory_tracker._collect_stats()
        timing_stats.compute()

        # Calculate throughput (images per second)
        # Each step processes batch_size source + batch_size destination images
        total_images = batch_size * self.config.num_iterations
        total_time_sec = sum(timing_stats.times) / 1000.0
        throughput = total_images / total_time_sec if total_time_sec > 0 else 0

        name = f"training_bs{batch_size}_{input_size}"

        return BenchmarkResult(
            name=name,
            backend="pytorch",
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
            metadata={"mode": "training"},
        )

    def _training_step(
        self,
        encoder: Any,
        decoder: Any,
        optimizer: Any,
        loss_fn: Any,
        src_batch: Any,
        dst_batch: Any,
    ) -> None:
        """Perform a single training step.

        Args:
            encoder: Encoder model
            decoder: Decoder model
            optimizer: Optimizer
            loss_fn: Loss function
            src_batch: Source batch
            dst_batch: Destination batch
        """
        import torch

        # Forward pass
        src_features, src_latent = encoder(src_batch)  # type: ignore[misc]
        dst_features, dst_latent = encoder(dst_batch)  # type: ignore[misc]

        src_skips = src_features[:-1][::-1] + [src_features[0]]
        dst_skips = dst_features[:-1][::-1] + [dst_features[0]]
        src_dec = decoder(src_latent, src_skips)  # type: ignore[misc]
        dst_dec = decoder(dst_latent, dst_skips)  # type: ignore[misc]

        # Calculate loss
        loss = loss_fn(src_dec, src_batch) + loss_fn(dst_dec, dst_batch)  # type: ignore

        # Backward pass
        optimizer.zero_grad()  # type: ignore
        loss.backward()
        optimizer.step()  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
