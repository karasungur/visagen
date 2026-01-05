"""
Benchmark runners package.

Provides specialized benchmark runners for different scenarios:
    - InferenceBenchmark: Model inference benchmarking
    - TrainingBenchmark: Training throughput benchmarking
    - MergerBenchmark: Video merge pipeline benchmarking
"""

from visagen.benchmark.benchmarks.inference import InferenceBenchmark
from visagen.benchmark.benchmarks.merger import MergerBenchmark
from visagen.benchmark.benchmarks.training import TrainingBenchmark

__all__ = [
    "InferenceBenchmark",
    "TrainingBenchmark",
    "MergerBenchmark",
]
