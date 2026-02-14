"""
Benchmark result reporters.

Provides multiple output formats for benchmark results:
    - ConsoleReporter: Human-readable table output
    - JSONReporter: Machine-readable JSON output
    - MarkdownReporter: Documentation-friendly markdown output
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TextIO

from visagen.benchmark.config import BenchmarkReport


class BaseReporter(ABC):
    """Base class for benchmark reporters."""

    @abstractmethod
    def report(self, benchmark_report: BenchmarkReport) -> str:
        """Generate report from benchmark results.

        Args:
            benchmark_report: Complete benchmark report

        Returns:
            Formatted report string
        """
        pass

    def write(self, benchmark_report: BenchmarkReport, output: Path | TextIO) -> None:
        """Write report to file or stream.

        Args:
            benchmark_report: Complete benchmark report
            output: Output path or file-like object
        """
        content = self.report(benchmark_report)

        if isinstance(output, Path):
            output.write_text(content)
        else:
            output.write(content)


class ConsoleReporter(BaseReporter):
    """Human-readable console output reporter.

    Produces formatted tables for terminal display.
    """

    def __init__(self, use_color: bool = True) -> None:
        """Initialize console reporter.

        Args:
            use_color: Use ANSI color codes
        """
        self.use_color = use_color

    def report(self, benchmark_report: BenchmarkReport) -> str:
        """Generate console report.

        Args:
            benchmark_report: Complete benchmark report

        Returns:
            Formatted console output
        """
        lines = []

        # Header
        sep = "=" * 80
        lines.append(sep)
        lines.append("VISAGEN BENCHMARK RESULTS".center(80))
        lines.append(sep)
        lines.append("")

        # System info
        info = benchmark_report.system_info
        lines.append("System Information:")
        lines.append(f"  Platform:        {info.platform}")
        lines.append(f"  Python:          {info.python_version}")
        lines.append(f"  PyTorch:         {info.pytorch_version}")
        if info.cuda_version:
            lines.append(f"  CUDA:            {info.cuda_version}")
        if info.gpu_name:
            lines.append(f"  GPU:             {info.gpu_name}")
            lines.append(f"  GPU Count:       {info.gpu_count}")
        lines.append(f"  CPU:             {info.cpu_name}")
        lines.append(f"  CPU Cores:       {info.cpu_count}")
        lines.append("")

        # Results table
        lines.append("-" * 80)
        header = (
            f"{'Benchmark':<35} {'Throughput':>12} {'Latency (ms)':>15} {'GPU Mem':>10}"
        )
        lines.append(header)
        subheader = f"{'':<35} {'(imgs/sec)':>12} {'mean/p95':>15} {'(MB)':>10}"
        lines.append(subheader)
        lines.append("-" * 80)

        for result in benchmark_report.results:
            latency_str = f"{result.latency_mean:>6.2f}/{result.latency_p95:>6.2f}"
            line = (
                f"{result.name:<35} "
                f"{result.throughput:>12.1f} "
                f"{latency_str:>15} "
                f"{result.gpu_memory_peak:>10.0f}"
            )
            lines.append(line)

        lines.append("-" * 80)
        lines.append("")

        # Timestamp
        lines.append(f"Benchmark completed: {benchmark_report.timestamp.isoformat()}")
        lines.append("")

        return "\n".join(lines)


class JSONReporter(BaseReporter):
    """JSON output reporter for automation and CI.

    Produces machine-readable JSON output.
    """

    def __init__(self, indent: int = 2) -> None:
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation level
        """
        self.indent = indent

    def report(self, benchmark_report: BenchmarkReport) -> str:
        """Generate JSON report.

        Args:
            benchmark_report: Complete benchmark report

        Returns:
            JSON string
        """
        data = benchmark_report.to_dict()
        return json.dumps(data, indent=self.indent)


class MarkdownReporter(BaseReporter):
    """Markdown output reporter for documentation.

    Produces markdown tables suitable for documentation.
    """

    def report(self, benchmark_report: BenchmarkReport) -> str:
        """Generate markdown report.

        Args:
            benchmark_report: Complete benchmark report

        Returns:
            Markdown string
        """
        lines = []

        # Title
        lines.append("# Visagen Benchmark Results")
        lines.append("")

        # Timestamp
        lines.append(
            f"**Date:** {benchmark_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

        # System info
        lines.append("## System Information")
        lines.append("")
        info = benchmark_report.system_info
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Platform | {info.platform} |")
        lines.append(f"| Python | {info.python_version} |")
        lines.append(f"| PyTorch | {info.pytorch_version} |")
        if info.cuda_version:
            lines.append(f"| CUDA | {info.cuda_version} |")
        if info.gpu_name:
            lines.append(f"| GPU | {info.gpu_name} |")
            lines.append(f"| GPU Count | {info.gpu_count} |")
        lines.append(f"| CPU | {info.cpu_name} |")
        lines.append(f"| CPU Cores | {info.cpu_count} |")
        lines.append("")

        # Results table
        lines.append("## Benchmark Results")
        lines.append("")
        lines.append(
            "| Benchmark | Backend | Batch | Size | Throughput | Latency (p95) | GPU Memory |"
        )
        lines.append(
            "|-----------|---------|-------|------|------------|---------------|------------|"
        )

        for result in benchmark_report.results:
            lines.append(
                f"| {result.name} | {result.backend} | {result.batch_size} | "
                f"{result.input_size} | {result.throughput:.1f} img/s | "
                f"{result.latency_p95:.2f} ms | {result.gpu_memory_peak:.0f} MB |"
            )

        lines.append("")

        # Summary statistics
        if benchmark_report.results:
            lines.append("## Summary")
            lines.append("")

            # Group by backend
            backends = {r.backend for r in benchmark_report.results}
            for backend in sorted(backends):
                backend_results = [
                    r for r in benchmark_report.results if r.backend == backend
                ]
                if backend_results:
                    max_throughput = max(r.throughput for r in backend_results)
                    min_latency = min(r.latency_p95 for r in backend_results)
                    lines.append(
                        f"- **{backend.upper()}**: Max throughput {max_throughput:.1f} img/s, "
                        f"Min latency {min_latency:.2f} ms (p95)"
                    )

            lines.append("")

        return "\n".join(lines)


def get_reporter(format_name: str) -> BaseReporter:
    """Get reporter by format name.

    Args:
        format_name: Format name ('console', 'json', 'markdown')

    Returns:
        Reporter instance
    """
    reporters: dict[str, type[BaseReporter]] = {
        "console": ConsoleReporter,
        "json": JSONReporter,
        "markdown": MarkdownReporter,
        "md": MarkdownReporter,
    }

    if format_name.lower() not in reporters:
        raise ValueError(
            f"Unknown format: {format_name}. Available: {', '.join(reporters.keys())}"
        )

    return reporters[format_name.lower()]()
