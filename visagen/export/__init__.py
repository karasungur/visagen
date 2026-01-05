"""
Visagen Export Module.

Provides ONNX and TensorRT export capabilities for trained models.

Features:
    - ONNX export with dynamic axes support
    - ONNX Runtime inference wrapper
    - TensorRT engine building (FP32/FP16/INT8)
    - TensorRT inference wrapper
    - Export validation utilities

Example:
    >>> from visagen.export import ONNXExporter
    >>> exporter = ONNXExporter("model.ckpt", "model.onnx")
    >>> exporter.export()
    >>> result = exporter.validate()
    >>> print(f"Validation passed: {result.passed}")

    >>> from visagen.export import ONNXRunner
    >>> runner = ONNXRunner("model.onnx", device="cuda")
    >>> output = runner(input_image)
"""

from visagen.export.model_wrapper import ExportableModel
from visagen.export.onnx_exporter import ExportConfig, ONNXExporter
from visagen.export.onnx_runner import ONNXRunner
from visagen.export.validation import ValidationResult, validate_export

# TensorRT exports (conditional)
try:
    from visagen.export.tensorrt_builder import TensorRTBuilder
    from visagen.export.tensorrt_runner import TensorRTRunner

    __all__ = [
        "ExportableModel",
        "ONNXExporter",
        "ExportConfig",
        "ONNXRunner",
        "TensorRTBuilder",
        "TensorRTRunner",
        "ValidationResult",
        "validate_export",
    ]
except ImportError:
    # TensorRT not available
    __all__ = [
        "ExportableModel",
        "ONNXExporter",
        "ExportConfig",
        "ONNXRunner",
        "ValidationResult",
        "validate_export",
    ]
