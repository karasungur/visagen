"""
Tests for Visagen benchmark module.

Tests cover:
- BenchmarkConfig, BenchmarkResult, SystemInfo dataclasses
- CUDATimer and MemoryTracker profilers
- ConsoleReporter, JSONReporter, MarkdownReporter
- CLI argument parsing
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

# =============================================================================
# BenchmarkConfig Tests
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from visagen.benchmark.config import BenchmarkConfig

        config = BenchmarkConfig()

        assert config.mode == "inference"
        assert config.checkpoint_path is None
        assert config.input_sizes == [256]
        assert config.batch_sizes == [1, 2, 4, 8]
        assert config.num_warmup == 10
        assert config.num_iterations == 100
        assert config.backends == ["pytorch"]
        assert config.device == "cuda"
        assert config.output_formats == ["console"]
        assert config.output_dir is None
        assert config.profile is False

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from visagen.benchmark.config import BenchmarkConfig

        config = BenchmarkConfig(
            mode="training",
            input_sizes=[256, 512],
            batch_sizes=[4, 8, 16],
            backends=["pytorch", "onnx"],
            device="cpu",
        )

        assert config.mode == "training"
        assert config.input_sizes == [256, 512]
        assert config.batch_sizes == [4, 8, 16]
        assert config.backends == ["pytorch", "onnx"]
        assert config.device == "cpu"

    def test_config_path_conversion(self):
        """Test path conversion in post_init."""
        from visagen.benchmark.config import BenchmarkConfig

        config = BenchmarkConfig(
            checkpoint_path="/path/to/model.ckpt",
            output_dir="/path/to/output",
        )

        assert isinstance(config.checkpoint_path, Path)
        assert isinstance(config.output_dir, Path)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        from visagen.benchmark.config import BenchmarkResult

        result = BenchmarkResult(
            name="inference_pytorch_bs4_256",
            backend="pytorch",
            batch_size=4,
            input_size=256,
            throughput=150.5,
            latency_mean=6.5,
            latency_p95=8.2,
            gpu_memory_peak=1024.0,
        )

        assert result.name == "inference_pytorch_bs4_256"
        assert result.backend == "pytorch"
        assert result.batch_size == 4
        assert result.throughput == 150.5
        assert result.latency_mean == 6.5

    def test_result_to_dict(self):
        """Test result to_dict method."""
        from visagen.benchmark.config import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            backend="pytorch",
            batch_size=1,
            input_size=256,
            throughput=100.0,
            latency_mean=10.0,
        )

        d = result.to_dict()

        assert d["name"] == "test"
        assert d["backend"] == "pytorch"
        assert d["throughput"] == 100.0


class TestSystemInfo:
    """Tests for SystemInfo dataclass."""

    def test_collect_system_info(self):
        """Test system info collection."""
        from visagen.benchmark.config import SystemInfo

        info = SystemInfo.collect()

        assert info.platform in ["Linux", "Windows", "Darwin"]
        assert len(info.python_version) > 0
        assert info.cpu_count > 0

    def test_to_dict(self):
        """Test to_dict method."""
        from visagen.benchmark.config import SystemInfo

        info = SystemInfo.collect()
        d = info.to_dict()

        assert "platform" in d
        assert "python_version" in d
        assert "pytorch_version" in d


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""

    def test_report_creation(self):
        """Test report creation."""
        from visagen.benchmark.config import BenchmarkReport

        report = BenchmarkReport()

        assert isinstance(report.timestamp, datetime)
        assert len(report.results) == 0

    def test_add_result(self):
        """Test adding results to report."""
        from visagen.benchmark.config import BenchmarkReport, BenchmarkResult

        report = BenchmarkReport()
        result = BenchmarkResult(
            name="test",
            backend="pytorch",
            batch_size=1,
            input_size=256,
            throughput=100.0,
            latency_mean=10.0,
        )

        report.add_result(result)

        assert len(report.results) == 1
        assert report.results[0].name == "test"

    def test_to_dict(self):
        """Test report to_dict method."""
        from visagen.benchmark.config import BenchmarkReport, BenchmarkResult

        report = BenchmarkReport()
        report.add_result(
            BenchmarkResult(
                name="test",
                backend="pytorch",
                batch_size=1,
                input_size=256,
                throughput=100.0,
                latency_mean=10.0,
            )
        )

        d = report.to_dict()

        assert "timestamp" in d
        assert "system_info" in d
        assert "results" in d
        assert len(d["results"]) == 1


# =============================================================================
# Profilers Tests
# =============================================================================


class TestTimingStats:
    """Tests for TimingStats."""

    def test_compute_stats(self):
        """Test statistics computation."""
        from visagen.benchmark.profilers import TimingStats

        stats = TimingStats(times=[1.0, 2.0, 3.0, 4.0, 5.0])
        stats.compute()

        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0

    def test_empty_stats(self):
        """Test empty statistics."""
        from visagen.benchmark.profilers import TimingStats

        stats = TimingStats()
        stats.compute()

        assert stats.mean == 0.0
        assert stats.std == 0.0


class TestCUDATimer:
    """Tests for CUDATimer."""

    def test_timer_creation(self):
        """Test timer creation."""
        from visagen.benchmark.profilers import CUDATimer

        timer = CUDATimer(device="cpu")
        assert timer.elapsed_ms == 0.0

    def test_timer_cpu_timing(self):
        """Test CPU timing."""
        from visagen.benchmark.profilers import CUDATimer

        timer = CUDATimer(device="cpu")

        with timer.time():
            # Do some work
            _ = [i**2 for i in range(1000)]

        assert timer.elapsed_ms > 0.0

    def test_timer_reset(self):
        """Test timer reset."""
        from visagen.benchmark.profilers import CUDATimer

        timer = CUDATimer(device="cpu")
        timer.elapsed_ms = 100.0
        timer.reset()

        assert timer.elapsed_ms == 0.0


class TestMemoryTracker:
    """Tests for MemoryTracker."""

    def test_tracker_creation(self):
        """Test tracker creation."""
        from visagen.benchmark.profilers import MemoryTracker

        tracker = MemoryTracker(device="cpu")
        assert tracker.peak_mb == 0.0

    def test_tracker_reset(self):
        """Test tracker reset."""
        from visagen.benchmark.profilers import MemoryTracker

        tracker = MemoryTracker(device="cpu")
        tracker.peak_mb = 100.0
        tracker.reset()

        assert tracker.peak_mb == 0.0

    def test_tracker_get_current(self):
        """Test get_current method."""
        from visagen.benchmark.profilers import MemoryTracker

        tracker = MemoryTracker(device="cpu")
        allocated, reserved = tracker.get_current()

        assert allocated == 0.0
        assert reserved == 0.0


class TestTorchProfilerContext:
    """Tests for TorchProfilerContext."""

    def test_profiler_creation(self):
        """Test profiler creation."""
        from visagen.benchmark.profilers import TorchProfilerContext

        profiler = TorchProfilerContext()
        assert profiler._profiler is None

    def test_profiler_context(self):
        """Test profiler as context manager."""
        from visagen.benchmark.profilers import TorchProfilerContext

        profiler = TorchProfilerContext()

        with profiler:
            pass  # Just test context manager works


class TestWarmupCUDA:
    """Tests for warmup_cuda function."""

    def test_warmup_no_error(self):
        """Test warmup doesn't raise errors."""
        from visagen.benchmark.profilers import warmup_cuda

        # Should not raise any errors
        warmup_cuda()


# =============================================================================
# Reporters Tests
# =============================================================================


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_report_generation(self):
        """Test console report generation."""
        from visagen.benchmark.config import (
            BenchmarkReport,
            BenchmarkResult,
            SystemInfo,
        )
        from visagen.benchmark.reporters import ConsoleReporter

        report = BenchmarkReport(system_info=SystemInfo.collect())
        report.add_result(
            BenchmarkResult(
                name="test_benchmark",
                backend="pytorch",
                batch_size=4,
                input_size=256,
                throughput=150.0,
                latency_mean=6.5,
                latency_p95=8.0,
                gpu_memory_peak=1024.0,
            )
        )

        reporter = ConsoleReporter()
        output = reporter.report(report)

        assert "VISAGEN BENCHMARK RESULTS" in output
        assert "test_benchmark" in output
        assert "150.0" in output


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_report_generation(self):
        """Test JSON report generation."""
        from visagen.benchmark.config import (
            BenchmarkReport,
            BenchmarkResult,
            SystemInfo,
        )
        from visagen.benchmark.reporters import JSONReporter

        report = BenchmarkReport(system_info=SystemInfo.collect())
        report.add_result(
            BenchmarkResult(
                name="test_benchmark",
                backend="pytorch",
                batch_size=4,
                input_size=256,
                throughput=150.0,
                latency_mean=6.5,
            )
        )

        reporter = JSONReporter()
        output = reporter.report(report)

        # Should be valid JSON
        data = json.loads(output)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "test_benchmark"


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_report_generation(self):
        """Test markdown report generation."""
        from visagen.benchmark.config import (
            BenchmarkReport,
            BenchmarkResult,
            SystemInfo,
        )
        from visagen.benchmark.reporters import MarkdownReporter

        report = BenchmarkReport(system_info=SystemInfo.collect())
        report.add_result(
            BenchmarkResult(
                name="test_benchmark",
                backend="pytorch",
                batch_size=4,
                input_size=256,
                throughput=150.0,
                latency_mean=6.5,
                latency_p95=8.0,
                gpu_memory_peak=1024.0,
            )
        )

        reporter = MarkdownReporter()
        output = reporter.report(report)

        assert "# Visagen Benchmark Results" in output
        assert "| test_benchmark |" in output


class TestGetReporter:
    """Tests for get_reporter function."""

    def test_get_console_reporter(self):
        """Test getting console reporter."""
        from visagen.benchmark.reporters import ConsoleReporter, get_reporter

        reporter = get_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_get_json_reporter(self):
        """Test getting JSON reporter."""
        from visagen.benchmark.reporters import JSONReporter, get_reporter

        reporter = get_reporter("json")
        assert isinstance(reporter, JSONReporter)

    def test_get_markdown_reporter(self):
        """Test getting markdown reporter."""
        from visagen.benchmark.reporters import MarkdownReporter, get_reporter

        reporter = get_reporter("markdown")
        assert isinstance(reporter, MarkdownReporter)

    def test_unknown_reporter(self):
        """Test error on unknown format."""
        from visagen.benchmark.reporters import get_reporter

        with pytest.raises(ValueError, match="Unknown format"):
            get_reporter("unknown")


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_parse_args_defaults(self, monkeypatch):
        """Test default argument parsing."""
        from visagen.tools.benchmark import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["visagen-benchmark"],
        )

        args = parse_args()

        assert args.mode == "inference"
        assert args.checkpoint is None
        assert args.batch_sizes == "1,2,4,8"
        assert args.resolutions == "256"
        assert args.backends == "pytorch"

    def test_parse_args_custom(self, monkeypatch, tmp_path):
        """Test custom argument parsing."""
        from visagen.tools.benchmark import parse_args

        checkpoint = tmp_path / "model.ckpt"
        checkpoint.touch()

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-benchmark",
                "--mode",
                "training",
                "--batch-sizes",
                "4,8,16",
                "--resolutions",
                "256,512",
                "--backends",
                "pytorch,onnx",
                "--warmup",
                "5",
                "--iterations",
                "50",
            ],
        )

        args = parse_args()

        assert args.mode == "training"
        assert args.batch_sizes == "4,8,16"
        assert args.resolutions == "256,512"
        assert args.backends == "pytorch,onnx"
        assert args.warmup == 5
        assert args.iterations == 50


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_parse_list(self):
        """Test parse_list function."""
        from visagen.tools.benchmark import parse_list

        result = parse_list("a,b,c")
        assert result == ["a", "b", "c"]

        result = parse_list("a, b, c")
        assert result == ["a", "b", "c"]

    def test_parse_int_list(self):
        """Test parse_int_list function."""
        from visagen.tools.benchmark import parse_int_list

        result = parse_int_list("1,2,4,8")
        assert result == [1, 2, 4, 8]

    def test_build_config(self, monkeypatch):
        """Test build_config function."""
        from visagen.tools.benchmark import build_config, parse_args

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-benchmark",
                "--mode",
                "inference",
                "--batch-sizes",
                "1,4",
                "--resolutions",
                "256",
            ],
        )

        args = parse_args()
        config = build_config(args)

        assert config.mode == "inference"
        assert config.batch_sizes == [1, 4]
        assert config.input_sizes == [256]


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_benchmark_exports(self):
        """Test benchmark module exports."""
        from visagen.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            BenchmarkResult,
            CUDATimer,
            MemoryTracker,
            SystemInfo,
            TimingStats,
            get_reporter,
        )

        assert BenchmarkConfig is not None
        assert BenchmarkResult is not None
        assert BenchmarkReport is not None
        assert SystemInfo is not None
        assert CUDATimer is not None
        assert MemoryTracker is not None
        assert TimingStats is not None
        assert callable(get_reporter)

    def test_benchmark_all_contains_essentials(self):
        """Test __all__ contains essential exports."""
        from visagen.benchmark import __all__

        assert "BenchmarkConfig" in __all__
        assert "BenchmarkResult" in __all__
        assert "CUDATimer" in __all__
        assert "MemoryTracker" in __all__


# =============================================================================
# Integration Tests
# =============================================================================


class TestReporterWrite:
    """Tests for reporter write functionality."""

    def test_write_to_file(self, tmp_path):
        """Test writing report to file."""
        from visagen.benchmark.config import (
            BenchmarkReport,
            BenchmarkResult,
            SystemInfo,
        )
        from visagen.benchmark.reporters import JSONReporter

        report = BenchmarkReport(system_info=SystemInfo.collect())
        report.add_result(
            BenchmarkResult(
                name="test",
                backend="pytorch",
                batch_size=1,
                input_size=256,
                throughput=100.0,
                latency_mean=10.0,
            )
        )

        output_file = tmp_path / "results.json"
        reporter = JSONReporter()
        reporter.write(report, output_file)

        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert "results" in content
