"""
TensorRT inference runner for Visagen.

Provides TensorRTRunner class for running inference on TensorRT engines
with CUDA memory management and dynamic shape support.

Requires NVIDIA TensorRT and CUDA.
"""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


class TensorRTRunner:
    """
    TensorRT engine inference wrapper.

    Manages CUDA memory allocation and provides a simple inference interface
    for TensorRT engines.

    Args:
        engine_path: Path to TensorRT engine file.
        device_id: CUDA device ID. Default: 0.

    Example:
        >>> runner = TensorRTRunner("model.engine")
        >>> runner.warmup()
        >>> image = np.random.randn(1, 3, 256, 256).astype(np.float32)
        >>> output = runner(image)
        >>> output.shape
        (1, 3, 256, 256)

    Note:
        Engine files are GPU-architecture specific.
        An engine built on one GPU may not work on another.
    """

    def __init__(
        self,
        engine_path: str | Path,
        device_id: int = 0,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device_id = device_id

        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        # Import TensorRT and CUDA
        try:
            import tensorrt as trt

            self.trt = trt
        except ImportError:
            raise ImportError(
                "TensorRT is required for inference. Install TensorRT from NVIDIA."
            )

        try:
            import pycuda.autoinit  # noqa: F401 - Initializes CUDA context
            import pycuda.driver as cuda

            self.cuda = cuda
        except ImportError:
            raise ImportError(
                "PyCUDA is required for TensorRT inference. "
                "Install with: pip install pycuda"
            )

        # Load engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # Get binding info
        self._setup_bindings()

        logger.info(f"TensorRTRunner initialized: {self.engine_path}")

    def _load_engine(self):
        """Load serialized TensorRT engine."""
        trt = self.trt

        logger.info(f"Loading TensorRT engine from {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        return engine

    def _setup_bindings(self) -> None:
        """Set up input/output bindings."""
        self.input_binding: dict[str, Any] | None = None
        self.output_binding: dict[str, Any] | None = None
        self.bindings: list[dict[str, Any]] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            binding = {
                "name": name,
                "shape": tuple(shape),
                "dtype": dtype,
                "is_input": mode == self.trt.TensorIOMode.INPUT,
            }

            if binding["is_input"]:
                self.input_binding = binding
            else:
                self.output_binding = binding

            self.bindings.append(binding)

        logger.info(
            f"Input: {self.input_binding['name']} {self.input_binding['shape']}"  # type: ignore
        )
        logger.info(
            f"Output: {self.output_binding['name']} {self.output_binding['shape']}"  # type: ignore
        )

    def _allocate_buffers(self, batch_size: int, height: int, width: int):
        """Allocate CUDA buffers for given dimensions."""
        cuda = self.cuda

        # Input shape
        input_shape = (batch_size, 3, height, width)
        input_size = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize

        # Output shape (same as input for face swap)
        output_shape = (batch_size, 3, height, width)
        output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

        # Allocate device memory
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        # Create output buffer
        h_output = np.empty(output_shape, dtype=np.float32)

        return d_input, d_output, h_output, input_shape, output_shape

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on input image.

        Args:
            image: Input array of shape (B, C, H, W) or (C, H, W), float32.

        Returns:
            Output array of shape (B, C, H, W).
        """
        cuda = self.cuda

        # Ensure batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        # Ensure float32 and contiguous
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image = np.ascontiguousarray(image)

        batch_size, channels, height, width = image.shape

        # Allocate buffers
        d_input, d_output, h_output, input_shape, output_shape = self._allocate_buffers(
            batch_size, height, width
        )

        try:
            # Set dynamic shapes
            if self.input_binding is None:
                raise RuntimeError("Input binding not found")
            if self.output_binding is None:
                raise RuntimeError("Output binding not found")

            self.context.set_input_shape(self.input_binding["name"], input_shape)

            # Copy input to device
            cuda.memcpy_htod(d_input, image)

            # Set tensor addresses
            self.context.set_tensor_address(self.input_binding["name"], int(d_input))
            self.context.set_tensor_address(self.output_binding["name"], int(d_output))

            # Execute inference
            self.context.execute_async_v3(stream_handle=0)

            # Copy output to host
            cuda.memcpy_dtoh(h_output, d_output)

        finally:
            # Free device memory
            d_input.free()
            d_output.free()

        return cast(np.ndarray, h_output)

    def warmup(
        self,
        shape: tuple[int, ...] | None = None,
        num_iterations: int = 3,
    ) -> None:
        """
        Warmup engine with dummy inference.

        Args:
            shape: Input shape for warmup. Default: (1, 3, 256, 256).
            num_iterations: Number of warmup iterations. Default: 3.
        """
        if shape is None:
            shape = (1, 3, 256, 256)

        logger.info(f"Warming up TensorRT engine with {num_iterations} iterations...")

        dummy = np.random.randn(*shape).astype(np.float32)

        for _ in range(num_iterations):
            _ = self(dummy)

        logger.info("Warmup completed")

    def get_binding_info(self) -> dict:
        """
        Get binding information.

        Returns:
            Dictionary with input/output binding details.
        """
        return {
            "input": self.input_binding,
            "output": self.output_binding,
            "all_bindings": self.bindings,
        }

    def __repr__(self) -> str:
        return (
            f"TensorRTRunner(engine_path='{self.engine_path}', "
            f"input_shape={self.input_binding['shape'] if self.input_binding else None})"
        )


class TensorRTRunnerV2:
    """
    Alternative TensorRT runner using cuda-python instead of PyCUDA.

    This version uses the cuda-python package which provides a more
    direct binding to the CUDA Driver API.

    Note: Requires cuda-python package.
    """

    def __init__(
        self,
        engine_path: str | Path,
        device_id: int = 0,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device_id = device_id

        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        # Try to import cuda-python
        try:
            from cuda import cuda, cudart

            self.cuda_driver = cuda
            self.cuda_runtime = cudart
        except ImportError:
            raise ImportError(
                "cuda-python is required for this runner. "
                "Install with: pip install cuda-python"
            )

        # Import TensorRT
        try:
            import tensorrt as trt

            self.trt = trt
        except ImportError:
            raise ImportError(
                "TensorRT is required for inference. Install TensorRT from NVIDIA."
            )

        # Initialize CUDA
        (err,) = self.cuda_driver.cuInit(0)
        if err != self.cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to initialize CUDA: {err}")

        # Set device
        err, self.cu_device = self.cuda_driver.cuDeviceGet(device_id)
        if err != self.cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to get device {device_id}: {err}")

        # Create context
        err, self.cu_context = self.cuda_driver.cuCtxCreate(0, self.cu_device)
        if err != self.cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create CUDA context: {err}")

        # Load engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # Setup bindings
        self._setup_bindings()

    def _load_engine(self):
        """Load TensorRT engine."""
        trt = self.trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def _setup_bindings(self) -> None:
        """Set up input/output bindings."""
        self.input_binding: dict[str, Any] | None = None
        self.output_binding: dict[str, Any] | None = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            binding = {
                "name": name,
                "shape": tuple(shape),
                "dtype": dtype,
            }

            if mode == self.trt.TensorIOMode.INPUT:
                self.input_binding = binding
            else:
                self.output_binding = binding

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run inference."""
        cuda_driver = self.cuda_driver

        # Ensure batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        # Ensure float32 and contiguous
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image = np.ascontiguousarray(image)

        batch_size, channels, height, width = image.shape

        # Allocate device memory
        input_size = image.nbytes
        output_shape = (batch_size, 3, height, width)
        output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

        err, d_input = cuda_driver.cuMemAlloc(input_size)
        err, d_output = cuda_driver.cuMemAlloc(output_size)

        h_output = np.empty(output_shape, dtype=np.float32)

        try:
            # Set dynamic shapes
            if self.input_binding is None:
                raise RuntimeError("Input binding not found")
            if self.output_binding is None:
                raise RuntimeError("Output binding not found")

            self.context.set_input_shape(
                self.input_binding["name"],
                (batch_size, channels, height, width),
            )

            # Copy input to device
            cuda_driver.cuMemcpyHtoD(d_input, image.ctypes.data, input_size)

            # Set tensor addresses
            self.context.set_tensor_address(self.input_binding["name"], int(d_input))
            self.context.set_tensor_address(self.output_binding["name"], int(d_output))

            # Execute
            self.context.execute_async_v3(stream_handle=0)

            # Synchronize
            cuda_driver.cuCtxSynchronize()

            # Copy output to host
            cuda_driver.cuMemcpyDtoH(h_output.ctypes.data, d_output, output_size)

        finally:
            # Free device memory
            cuda_driver.cuMemFree(d_input)
            cuda_driver.cuMemFree(d_output)

        return h_output

    def __del__(self):
        """Clean up CUDA context."""
        if hasattr(self, "cu_context"):
            self.cuda_driver.cuCtxDestroy(self.cu_context)
