"""
TensorRT engine builder for Visagen.

Converts ONNX models to optimized TensorRT engines with support for:
- FP32, FP16, and INT8 precision
- Dynamic input shapes
- Workspace size configuration

Requires NVIDIA TensorRT and a CUDA-capable GPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """
    TensorRT build configuration.

    Attributes:
        precision: Precision mode ("fp32", "fp16", "int8"). Default: "fp16".
        max_batch_size: Maximum batch size. Default: 8.
        workspace_size: Workspace size in bytes. Default: 1GB.
        min_shape: Minimum input shape (B, C, H, W).
        opt_shape: Optimal input shape (most common).
        max_shape: Maximum input shape.
        strict_types: Enforce strict precision. Default: False.
    """

    precision: str = "fp16"
    max_batch_size: int = 8
    workspace_size: int = 1 << 30  # 1GB
    min_shape: tuple[int, ...] = (1, 3, 128, 128)
    opt_shape: tuple[int, ...] = (1, 3, 256, 256)
    max_shape: tuple[int, ...] = (8, 3, 512, 512)
    strict_types: bool = False


class TensorRTBuilder:
    """
    Build TensorRT engine from ONNX model.

    Creates optimized inference engines with FP16 or INT8 quantization
    for maximum GPU performance.

    Args:
        onnx_path: Path to input ONNX model.
        engine_path: Path for output TensorRT engine.
        config: Build configuration. Default: None (use defaults).

    Example:
        >>> builder = TensorRTBuilder("model.onnx", "model.engine")
        >>> engine_path = builder.build()
        >>> print(f"Engine saved to: {engine_path}")

    Note:
        Requires TensorRT installation from NVIDIA.
        Engine files are GPU-architecture specific and not portable.
    """

    def __init__(
        self,
        onnx_path: Path,
        engine_path: Path,
        config: BuildConfig | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.config = config or BuildConfig()

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        # Import TensorRT
        try:
            import tensorrt as trt

            self.trt = trt
        except ImportError:
            raise ImportError(
                "TensorRT is required for engine building. "
                "Install TensorRT from NVIDIA."
            )

    def build(self) -> Path:
        """
        Build TensorRT engine from ONNX model.

        Returns:
            Path to built engine file.

        Raises:
            RuntimeError: If build fails.
        """
        trt = self.trt

        logger.info(f"Building TensorRT engine from {self.onnx_path}")
        logger.info(f"Precision: {self.config.precision}")

        # Create logger
        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Create builder
        builder = trt.Builder(trt_logger)

        # Create network with explicit batch
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        # Create ONNX parser
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX model
        logger.info("Parsing ONNX model...")
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError("Failed to parse ONNX model:\n" + "\n".join(errors))

        logger.info(f"Network inputs: {network.num_inputs}")
        logger.info(f"Network outputs: {network.num_outputs}")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size,
        )
        logger.info(f"Workspace: {self.config.workspace_size / (1024**3):.1f} GB")

        # Create optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()

        # Get input tensor name
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # Set shape profiles
        profile.set_shape(
            input_name,
            self.config.min_shape,
            self.config.opt_shape,
            self.config.max_shape,
        )
        config.add_optimization_profile(profile)

        logger.info(
            f"Dynamic shapes: min={self.config.min_shape}, "
            f"opt={self.config.opt_shape}, max={self.config.max_shape}"
        )

        # Set precision
        if self.config.precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")

        elif self.config.precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled")
                # Note: INT8 requires calibration data
                logger.warning(
                    "INT8 calibration data not provided. Engine will use default ranges "
                    "which may result in significant accuracy loss. Consider using FP16 "
                    "or providing calibration data for best results."
                )
            else:
                logger.warning("INT8 not supported on this platform, using FP32")

        # Strict type constraints
        if self.config.strict_types:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)

        engine_size_mb = self.engine_path.stat().st_size / (1024 * 1024)
        logger.info(f"Engine saved to {self.engine_path} ({engine_size_mb:.1f} MB)")

        return self.engine_path

    def get_platform_info(self) -> dict:
        """
        Get platform capability information.

        Returns:
            Dictionary with platform capabilities.
        """
        trt = self.trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)

        return {
            "tensorrt_version": trt.__version__,
            "has_fast_fp16": builder.platform_has_fast_fp16,
            "has_fast_int8": builder.platform_has_fast_int8,
            "max_dla_batch_size": getattr(builder, "max_DLA_batch_size", None),
            "num_dla_cores": getattr(builder, "num_DLA_cores", None),
        }


class Int8Calibrator:
    """
    INT8 calibration data provider.

    Used for INT8 quantization to determine optimal scale factors.
    Requires a representative dataset.

    Note: This is a placeholder for future implementation.
    Full INT8 calibration requires:
    1. Representative calibration dataset
    2. Running inference on calibration data
    3. Collecting activation statistics
    """

    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: Path | None = None,
    ) -> None:
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_idx = 0

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return self.calibration_data.shape[0]

    def get_batch(self, names):
        """Get next calibration batch."""
        if self.batch_idx >= len(self.calibration_data):
            return None

        batch = self.calibration_data[self.batch_idx : self.batch_idx + 1]
        self.batch_idx += 1

        return [batch.ctypes.data]

    def read_calibration_cache(self):
        """Read calibration cache if exists."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache."""
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
