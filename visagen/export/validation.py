"""
Export validation utilities for Visagen.

Provides functions and classes for validating exported models
by comparing their outputs against the original PyTorch model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of PyTorch vs exported model validation.

    Attributes:
        passed: Whether validation passed within tolerance.
        max_diff: Maximum absolute difference between outputs.
        mean_diff: Mean absolute difference between outputs.
        rtol: Relative tolerance used.
        atol: Absolute tolerance used.
        details: Optional additional details.
    """

    passed: bool
    max_diff: float
    mean_diff: float
    rtol: float
    atol: float
    details: Optional[str] = None


def validate_export(
    pytorch_model: nn.Module,
    exported_path: Union[str, Path],
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> ValidationResult:
    """
    Compare PyTorch and ONNX model outputs.

    Runs the same input through both models and checks if outputs match
    within the specified tolerance.

    Args:
        pytorch_model: PyTorch model to compare against.
        exported_path: Path to exported ONNX file.
        test_input: Input tensor for testing.
        rtol: Relative tolerance. Default: 1e-3.
        atol: Absolute tolerance. Default: 1e-5.

    Returns:
        ValidationResult with comparison metrics.

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If ONNX file doesn't exist.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for validation. "
            "Install with: pip install onnxruntime-gpu"
        )

    exported_path = Path(exported_path)
    if not exported_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {exported_path}")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        str(exported_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    onnx_output = session.run(
        None,
        {input_name: test_input.numpy()},
    )[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    # Check tolerance
    passed = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

    return ValidationResult(
        passed=passed,
        max_diff=float(max_diff),
        mean_diff=float(mean_diff),
        rtol=rtol,
        atol=atol,
    )


def validate_tensorrt(
    pytorch_model: nn.Module,
    engine_path: Union[str, Path],
    test_input: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> ValidationResult:
    """
    Compare PyTorch and TensorRT outputs.

    Note: TensorRT uses FP16 by default, so tolerances are larger.

    Args:
        pytorch_model: PyTorch model to compare against.
        engine_path: Path to TensorRT engine file.
        test_input: Input tensor for testing.
        rtol: Relative tolerance. Default: 1e-2 (larger for FP16).
        atol: Absolute tolerance. Default: 1e-3.

    Returns:
        ValidationResult with comparison metrics.

    Raises:
        ImportError: If tensorrt is not installed.
        FileNotFoundError: If engine file doesn't exist.
    """
    try:
        from visagen.export.tensorrt_runner import TensorRTRunner
    except ImportError:
        raise ImportError(
            "TensorRT is required for validation. "
            "Install TensorRT from NVIDIA."
        )

    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).cpu().numpy()

    # TensorRT inference
    runner = TensorRTRunner(engine_path)
    trt_input = test_input.numpy()
    trt_output = runner(trt_input)

    # Compare outputs
    max_diff = np.abs(pytorch_output - trt_output).max()
    mean_diff = np.abs(pytorch_output - trt_output).mean()

    # Check tolerance (more lenient for FP16)
    passed = np.allclose(pytorch_output, trt_output, rtol=rtol, atol=atol)

    return ValidationResult(
        passed=passed,
        max_diff=float(max_diff),
        mean_diff=float(mean_diff),
        rtol=rtol,
        atol=atol,
        details="TensorRT validation (FP16 may have larger differences)",
    )


def compare_inference_speed(
    pytorch_model: nn.Module,
    onnx_path: Optional[Path] = None,
    trt_path: Optional[Path] = None,
    input_shape: tuple = (1, 3, 256, 256),
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Benchmark inference speed across different backends.

    Args:
        pytorch_model: PyTorch model.
        onnx_path: Path to ONNX model (optional).
        trt_path: Path to TensorRT engine (optional).
        input_shape: Input tensor shape.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of benchmark iterations.
        device: Device for PyTorch inference.

    Returns:
        Dictionary with timing results for each backend.
    """
    import time

    results = {}

    # Prepare input
    dummy_input = torch.randn(*input_shape)

    # PyTorch benchmark
    pytorch_model.eval()
    pytorch_model = pytorch_model.to(device)
    dummy_cuda = dummy_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = pytorch_model(dummy_cuda)
    torch.cuda.synchronize() if device == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_model(dummy_cuda)
    torch.cuda.synchronize() if device == "cuda" else None
    pytorch_time = (time.perf_counter() - start) / num_iterations * 1000

    results["pytorch"] = {
        "ms_per_inference": pytorch_time,
        "fps": 1000 / pytorch_time,
    }

    # ONNX benchmark
    if onnx_path and Path(onnx_path).exists():
        try:
            from visagen.export.onnx_runner import ONNXRunner

            runner = ONNXRunner(onnx_path, device=device)
            dummy_np = dummy_input.numpy()

            # Warmup
            for _ in range(num_warmup):
                _ = runner(dummy_np)

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = runner(dummy_np)
            onnx_time = (time.perf_counter() - start) / num_iterations * 1000

            results["onnx"] = {
                "ms_per_inference": onnx_time,
                "fps": 1000 / onnx_time,
            }
        except Exception as e:
            logger.warning(f"ONNX benchmark failed: {e}")

    # TensorRT benchmark
    if trt_path and Path(trt_path).exists():
        try:
            from visagen.export.tensorrt_runner import TensorRTRunner

            runner = TensorRTRunner(trt_path)
            dummy_np = dummy_input.numpy()

            # Warmup
            for _ in range(num_warmup):
                _ = runner(dummy_np)

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = runner(dummy_np)
            trt_time = (time.perf_counter() - start) / num_iterations * 1000

            results["tensorrt"] = {
                "ms_per_inference": trt_time,
                "fps": 1000 / trt_time,
            }
        except Exception as e:
            logger.warning(f"TensorRT benchmark failed: {e}")

    return results
