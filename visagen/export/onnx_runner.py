"""
ONNX Runtime inference wrapper for Visagen.

Provides ONNXRunner class for running inference on exported ONNX models
with GPU/CPU support and optimized session configuration.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ONNXRunner:
    """
    ONNX Runtime inference wrapper.

    Provides a simple interface for running inference on ONNX models
    with automatic provider selection and session optimization.

    Args:
        model_path: Path to ONNX model file.
        device: Device for inference ("cuda" or "cpu"). Default: "cuda".
        device_id: GPU device ID when using CUDA. Default: 0.
        num_threads: Number of CPU threads for inference. Default: 4.

    Example:
        >>> runner = ONNXRunner("model.onnx", device="cuda")
        >>> runner.warmup()
        >>> image = np.random.randn(1, 3, 256, 256).astype(np.float32)
        >>> output = runner(image)
        >>> output.shape
        (1, 3, 256, 256)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        device_id: int = 0,
        num_threads: int = 4,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. "
                "Install with: pip install onnxruntime-gpu"
            )

        self.model_path = Path(model_path)
        self.device = device
        self.device_id = device_id
        self.num_threads = num_threads

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Create session
        self.session = self._create_session(ort)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape info
        input_info = self.session.get_inputs()[0]
        self.input_shape = input_info.shape
        self.input_dtype = input_info.type

        logger.info(
            f"ONNXRunner initialized: device={device}, "
            f"input_shape={self.input_shape}"
        )

    def _create_session(self, ort) -> "ort.InferenceSession":
        """Create optimized ONNX Runtime session."""
        # Configure providers
        providers = []

        if self.device == "cuda":
            # Check CUDA availability
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": self.device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))
                logger.info(f"Using CUDA provider on device {self.device_id}")
            else:
                logger.warning(
                    "CUDAExecutionProvider not available, falling back to CPU"
                )

        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 2
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # Enable memory-efficient execution
        sess_options.enable_mem_reuse = True

        return ort.InferenceSession(
            str(self.model_path),
            providers=providers,
            sess_options=sess_options,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on single image.

        Args:
            image: Input array of shape (C, H, W) or (B, C, H, W), float32.

        Returns:
            Output array of shape (B, C, H, W).
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image},
        )

        return outputs[0]

    def batch_inference(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
    ) -> List[np.ndarray]:
        """
        Run inference on batch of images.

        Args:
            images: List of input arrays, each (C, H, W) or (H, W, C).
            batch_size: Maximum batch size for inference. Default: 8.

        Returns:
            List of output arrays.
        """
        outputs = []
        num_images = len(images)

        for i in range(0, num_images, batch_size):
            batch = images[i : i + batch_size]

            # Stack into batch
            batch_array = np.stack(batch, axis=0)

            # Ensure float32
            if batch_array.dtype != np.float32:
                batch_array = batch_array.astype(np.float32)

            # Run inference
            batch_outputs = self.session.run(
                [self.output_name],
                {self.input_name: batch_array},
            )[0]

            # Split outputs
            for j in range(batch_outputs.shape[0]):
                outputs.append(batch_outputs[j])

        return outputs

    def warmup(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        num_iterations: int = 3,
    ) -> None:
        """
        Warmup session with dummy inference.

        Running a few inferences before actual use ensures optimal performance.

        Args:
            shape: Input shape for warmup. Default: (1, 3, 256, 256).
            num_iterations: Number of warmup iterations. Default: 3.
        """
        if shape is None:
            shape = (1, 3, 256, 256)

        logger.info(f"Warming up ONNX session with {num_iterations} iterations...")

        dummy = np.random.randn(*shape).astype(np.float32)

        for _ in range(num_iterations):
            _ = self(dummy)

        logger.info("Warmup completed")

    def get_io_info(self) -> dict:
        """
        Get input/output information.

        Returns:
            Dictionary with input and output metadata.
        """
        inputs = []
        for inp in self.session.get_inputs():
            inputs.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })

        outputs = []
        for out in self.session.get_outputs():
            outputs.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })

        return {
            "inputs": inputs,
            "outputs": outputs,
            "providers": self.session.get_providers(),
        }

    def __repr__(self) -> str:
        return (
            f"ONNXRunner(model_path='{self.model_path}', "
            f"device='{self.device}', input_shape={self.input_shape})"
        )
