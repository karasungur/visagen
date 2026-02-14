#!/usr/bin/env python3
"""
Visagen Benchmark CLI - Performance benchmarking tool.

Usage:
    visagen-benchmark --mode inference -c model.ckpt
    visagen-benchmark --mode all -c model.ckpt --batch-sizes 1,2,4,8

Examples:
    # Basic inference benchmark
    visagen-benchmark --mode inference -c model.ckpt

    # Full sweep with multiple backends
    visagen-benchmark --mode inference -c model.ckpt \\
        --batch-sizes 1,2,4,8,16 \\
        --resolutions 256,384,512 \\
        --backends pytorch,onnx

    # Training benchmark
    visagen-benchmark --mode training --batch-sizes 4,8,16

    # Export results to JSON
    visagen-benchmark --mode inference -c model.ckpt \\
        --output-format json --output-dir ./results
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from visagen.benchmark.config import (
    BenchmarkConfig,
    BenchmarkReport,
    SystemInfo,
)
from visagen.benchmark.reporters import get_reporter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visagen Performance Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference benchmark
  visagen-benchmark --mode inference -c model.ckpt

  # Full sweep with multiple backends and resolutions
  visagen-benchmark --mode inference -c model.ckpt \\
      --batch-sizes 1,2,4,8,16 \\
      --resolutions 256,384,512 \\
      --backends pytorch,onnx,tensorrt

  # Training throughput benchmark
  visagen-benchmark --mode training \\
      --batch-sizes 4,8,16,32 \\
      --resolutions 256,512

  # Export results to multiple formats
  visagen-benchmark --mode inference -c model.ckpt \\
      --output-format console,json,markdown \\
      --output-dir ./benchmark_results/
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="inference",
        choices=["inference", "training", "merge", "all"],
        help="Benchmark mode (default: inference)",
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to model checkpoint (.ckpt)",
    )

    # Benchmark configuration
    config = parser.add_argument_group("Benchmark Configuration")
    config.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes (default: 1,2,4,8)",
    )
    config.add_argument(
        "--resolutions",
        type=str,
        default="256",
        help="Comma-separated input resolutions (default: 256)",
    )
    config.add_argument(
        "--backends",
        type=str,
        default="pytorch",
        help="Comma-separated backends: pytorch,onnx,tensorrt (default: pytorch)",
    )
    config.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    config.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    config.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to run benchmarks on (default: cuda)",
    )

    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--output-format",
        type=str,
        default="console",
        help="Output formats: console,json,markdown (default: console)",
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save benchmark results",
    )

    # Profiling options
    profile = parser.add_argument_group("Profiling Options")
    profile.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler",
    )
    profile.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Directory for TensorBoard profiler logs",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode, minimal output",
    )

    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    """Parse comma-separated string to list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Build BenchmarkConfig from arguments.

    Args:
        args: Parsed arguments

    Returns:
        Benchmark configuration
    """
    return BenchmarkConfig(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        input_sizes=parse_int_list(args.resolutions),
        batch_sizes=parse_int_list(args.batch_sizes),
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        backends=parse_list(args.backends),
        device=args.device,
        output_formats=parse_list(args.output_format),
        output_dir=args.output_dir,
        profile=args.profile,
        tensorboard_dir=args.tensorboard_dir,
    )


def run_benchmarks(config: BenchmarkConfig, verbose: bool = False) -> BenchmarkReport:
    """Run benchmarks according to configuration.

    Args:
        config: Benchmark configuration
        verbose: Enable verbose output

    Returns:
        Benchmark report with all results
    """
    from visagen.benchmark.benchmarks import (
        InferenceBenchmark,
        MergerBenchmark,
        TrainingBenchmark,
    )

    # Create report
    report = BenchmarkReport(
        config=config,
        system_info=SystemInfo.collect(),
    )

    # Run benchmarks based on mode
    benchmark: Any
    if config.mode in ("inference", "all"):
        if verbose:
            print("Running inference benchmarks...")
        benchmark = InferenceBenchmark(config)
        results = benchmark.run()
        for result in results:
            report.add_result(result)

    if config.mode in ("training", "all"):
        if verbose:
            print("Running training benchmarks...")
        benchmark = TrainingBenchmark(config)
        results = benchmark.run()
        for result in results:
            report.add_result(result)

    if config.mode in ("merge", "all"):
        if config.checkpoint_path is not None:
            if verbose:
                print("Running merge benchmarks...")
            benchmark = MergerBenchmark(config)
            results = benchmark.run()
            for result in results:
                report.add_result(result)
        else:
            if verbose:
                print("Skipping merge benchmark (no checkpoint provided)")

    return report


def output_results(
    report: BenchmarkReport,
    output_formats: list[str],
    output_dir: Path | None = None,
    quiet: bool = False,
) -> None:
    """Output benchmark results in requested formats.

    Args:
        report: Benchmark report
        output_formats: List of output formats
        output_dir: Directory to save files
        quiet: Suppress console output
    """
    for format_name in output_formats:
        reporter = get_reporter(format_name)
        content = reporter.report(report)

        if format_name == "console":
            if not quiet:
                print(content)
        elif output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            ext = "json" if format_name == "json" else "md"
            filename = f"benchmark_results.{ext}"
            output_path = output_dir / filename

            output_path.write_text(content)
            if not quiet:
                print(f"Results saved to: {output_path}")


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Validate arguments
    if args.mode in ("inference", "merge", "all") and args.checkpoint is None:
        if args.mode != "all":
            print("Error: --checkpoint required for inference/merge benchmarks")
            return 1

    if args.checkpoint is not None and not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # Build configuration
    config = build_config(args)

    if not args.quiet:
        print("=" * 60)
        print("VISAGEN BENCHMARK")
        print("=" * 60)
        print(f"Mode: {config.mode}")
        print(f"Batch sizes: {config.batch_sizes}")
        print(f"Resolutions: {config.input_sizes}")
        print(f"Backends: {config.backends}")
        print(f"Device: {config.device}")
        print(f"Warmup: {config.num_warmup}")
        print(f"Iterations: {config.num_iterations}")
        if config.checkpoint_path:
            print(f"Checkpoint: {config.checkpoint_path}")
        print("=" * 60)
        print()

    try:
        # Run benchmarks
        report = run_benchmarks(config, verbose=args.verbose)

        # Output results
        output_results(
            report,
            config.output_formats,
            config.output_dir,
            quiet=args.quiet,
        )

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        return 130

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
