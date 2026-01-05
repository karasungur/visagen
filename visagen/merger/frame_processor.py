"""
Single-frame face swap processor for Visagen.

Handles the complete pipeline for swapping faces in a single frame:
1. Face detection and alignment
2. Model inference
3. Color transfer
4. Seamless blending back to original frame

Features:
    - InsightFace-based face detection
    - Configurable color transfer (RCT, LCT, SOT)
    - Multiple blending modes (Laplacian, Poisson, Feather)
    - Mask erosion for seamless edges
    - Multiple inference backends (PyTorch, ONNX, TensorRT)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch

# Type alias for face metadata
FaceMetadata = dict[str, Any]


@dataclass
class FrameProcessorConfig:
    """
    Configuration for frame processing.

    Attributes:
        min_confidence: Minimum face detection confidence. Default: 0.5.
        max_faces: Maximum faces to process per frame. Default: 1.
        face_type: Face alignment type. Default: "whole_face".
        output_size: Model input/output size. Default: 256.
        color_transfer_mode: Color transfer algorithm. Default: "rct".
            Options: "rct", "lct", "sot", None (disabled).
        blend_mode: Blending algorithm. Default: "laplacian".
            Options: "laplacian", "poisson", "feather".
        blend_amount: Blending intensity (0-1). Default: 1.0.
        mask_erode: Mask erosion kernel size. Default: 5.
        mask_blur: Mask blur kernel size. Default: 5.
        sharpen: Apply sharpening to output. Default: False.
        sharpen_amount: Sharpening intensity. Default: 0.3.
        restore_face: Enable GFPGAN face restoration. Default: False.
        restore_strength: Restoration strength (0.0-1.0). Default: 0.5.
        restore_model_version: GFPGAN model version (1.2, 1.3, 1.4). Default: 1.4.
    """

    # Detection
    min_confidence: float = 0.5
    max_faces: int = 1
    face_type: str = "whole_face"
    output_size: int = 256

    # Color transfer
    color_transfer_mode: str | None = "rct"

    # Blending
    blend_mode: str = "laplacian"
    blend_amount: float = 1.0
    mask_erode: int = 5
    mask_blur: int = 5

    # Post-processing
    sharpen: bool = False
    sharpen_amount: float = 0.3

    # Face restoration (GFPGAN)
    restore_face: bool = False
    restore_strength: float = 0.5
    restore_model_version: float = 1.4


@dataclass
class ProcessedFrame:
    """
    Result of frame processing.

    Attributes:
        frame_idx: Frame index.
        output_image: Processed frame (H, W, 3) uint8.
        faces_detected: Number of faces detected.
        faces_swapped: Number of faces successfully swapped.
        processing_time: Processing time in seconds.
        metadata: Optional additional metadata.
    """

    frame_idx: int
    output_image: np.ndarray
    faces_detected: int
    faces_swapped: int
    processing_time: float
    metadata: dict[str, Any] | None = None


class FrameProcessor:
    """
    Single-frame face swap processor.

    Handles face detection, model inference, and seamless blending
    for swapping faces in individual video frames.

    Supports multiple inference backends:
    - "pytorch": Native PyTorch (default)
    - "onnx": ONNX Runtime for optimized CPU/GPU inference
    - "tensorrt": TensorRT for maximum GPU performance

    Args:
        model: Trained model (DFLModule, checkpoint path, ONNX path, or TensorRT engine).
        config: Processing configuration.
        device: Torch device for PyTorch inference.
        backend: Inference backend ("pytorch", "onnx", "tensorrt"). Default: "pytorch".

    Example:
        >>> # PyTorch backend
        >>> processor = FrameProcessor("model.ckpt")
        >>> result = processor.process_frame(frame, frame_idx=0)

        >>> # ONNX backend for faster inference
        >>> processor = FrameProcessor("model.onnx", backend="onnx")
        >>> result = processor.process_frame(frame)

        >>> # TensorRT backend for maximum performance
        >>> processor = FrameProcessor("model.engine", backend="tensorrt")
        >>> result = processor.process_frame(frame)
    """

    # Valid backends
    VALID_BACKENDS = ("pytorch", "onnx", "tensorrt")

    def __init__(
        self,
        model: Union[str, Path, "torch.nn.Module"],
        config: FrameProcessorConfig | None = None,
        device: str | None = None,
        backend: str = "pytorch",
    ) -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of: {self.VALID_BACKENDS}"
            )

        self.config = config or FrameProcessorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend

        # Load model based on backend
        self.model = self._load_model(model)
        if self.backend == "pytorch":
            self.model.eval()

        # Lazy-loaded components
        self._detector = None
        self._aligner = None
        self._segmenter = None
        self._restorer = None

    def _load_model(
        self, model: Union[str, Path, "torch.nn.Module"]
    ) -> Union["torch.nn.Module", "ONNXRunner", "TensorRTRunner"]:
        """Load model based on backend type."""
        if self.backend == "pytorch":
            return self._load_pytorch_model(model)
        elif self.backend == "onnx":
            return self._load_onnx_model(model)
        elif self.backend == "tensorrt":
            return self._load_tensorrt_model(model)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_pytorch_model(
        self, model: Union[str, Path, "torch.nn.Module"]
    ) -> "torch.nn.Module":
        """Load PyTorch model from checkpoint or return if already loaded."""
        if isinstance(model, (str, Path)):
            from visagen.training.dfl_module import DFLModule

            model_path = Path(model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

            loaded_model = DFLModule.load_from_checkpoint(
                str(model_path),
                map_location=self.device,
            )
            loaded_model = loaded_model.to(self.device)
            return loaded_model
        else:
            return model.to(self.device)

    def _load_onnx_model(self, model: str | Path) -> "ONNXRunner":
        """Load ONNX model for inference."""
        from visagen.export.onnx_runner import ONNXRunner

        model_path = Path(model) if isinstance(model, str) else model

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        return ONNXRunner(model_path, device=self.device)

    def _load_tensorrt_model(self, model: str | Path) -> "TensorRTRunner":
        """Load TensorRT engine for inference."""
        from visagen.export.tensorrt_runner import TensorRTRunner

        model_path = Path(model) if isinstance(model, str) else model

        if not model_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {model_path}")

        return TensorRTRunner(model_path)

    @property
    def detector(self):
        """Lazy-load face detector."""
        if self._detector is None:
            from visagen.vision.detector import FaceDetector

            self._detector = FaceDetector()
        return self._detector

    @property
    def aligner(self):
        """Lazy-load face aligner."""
        if self._aligner is None:
            from visagen.vision.aligner import FaceAligner

            self._aligner = FaceAligner(
                output_size=self.config.output_size,
                face_type=self.config.face_type,
            )
        return self._aligner

    @property
    def segmenter(self):
        """Lazy-load face segmenter."""
        if self._segmenter is None:
            from visagen.vision.segmenter import FaceSegmenter

            self._segmenter = FaceSegmenter()
        return self._segmenter

    @property
    def restorer(self):
        """Lazy-load face restorer (GFPGAN)."""
        if self._restorer is None and self.config.restore_face:
            from visagen.postprocess.restore import FaceRestorer, RestoreConfig

            restore_config = RestoreConfig(
                enabled=True,
                strength=self.config.restore_strength,
                model_version=self.config.restore_model_version,
            )
            self._restorer = FaceRestorer(restore_config, device=self.device)
        return self._restorer

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> ProcessedFrame:
        """
        Process a single frame for face swap.

        Args:
            frame: Input frame (H, W, 3) uint8 BGR.
            frame_idx: Frame index for tracking.

        Returns:
            ProcessedFrame with swapped result.
        """
        start_time = time.time()

        # Make a copy to avoid modifying original
        output = frame.copy()
        faces_detected = 0
        faces_swapped = 0

        try:
            # Detect and align faces
            aligned_faces = self._detect_and_align(frame)
            faces_detected = len(aligned_faces)

            # Process each face
            for aligned_face, face_meta, mask in aligned_faces[: self.config.max_faces]:
                try:
                    # Run inference
                    swapped_face = self._run_inference(aligned_face)

                    # Apply color transfer
                    if self.config.color_transfer_mode:
                        swapped_face = self._apply_color_transfer(
                            swapped_face, aligned_face
                        )

                    # Apply face restoration (GFPGAN)
                    if self.config.restore_face:
                        swapped_face = self._apply_restoration(swapped_face)

                    # Blend back to frame
                    output = self._blend_to_frame(output, swapped_face, face_meta, mask)

                    faces_swapped += 1
                except Exception:
                    # Skip this face on error
                    continue

        except Exception:
            # Return original frame on error
            pass

        processing_time = time.time() - start_time

        return ProcessedFrame(
            frame_idx=frame_idx,
            output_image=output,
            faces_detected=faces_detected,
            faces_swapped=faces_swapped,
            processing_time=processing_time,
        )

    def _detect_and_align(
        self, frame: np.ndarray
    ) -> list[tuple[np.ndarray, FaceMetadata, np.ndarray]]:
        """
        Detect faces and return aligned images with metadata.

        Args:
            frame: Input frame (H, W, 3) BGR.

        Returns:
            List of (aligned_face, metadata, mask) tuples.
        """
        # Detect faces
        faces = self.detector.detect(frame, threshold=self.config.min_confidence)

        if not faces:
            return []

        results = []
        for face in faces:
            # Align face
            aligned, matrix = self.aligner.align(
                frame,
                face.landmarks,
            )

            # Generate mask
            mask = self._generate_mask(aligned)

            # Store metadata for warp-back
            metadata = {
                "bbox": face.bbox,
                "landmarks": face.landmarks,
                "matrix": matrix,
                "original_shape": frame.shape[:2],
            }

            results.append((aligned, metadata, mask))

        return results

    def _run_inference(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Run model inference on aligned face.

        Supports multiple backends: PyTorch, ONNX, TensorRT.

        Args:
            aligned_face: Aligned face (H, W, 3) uint8 BGR.

        Returns:
            Swapped face (H, W, 3) uint8 BGR.
        """
        # Preprocess: normalize and convert to CHW
        img = aligned_face.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        if self.backend == "pytorch":
            # PyTorch inference
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)

            output = output.squeeze(0).cpu().numpy()

        elif self.backend in ("onnx", "tensorrt"):
            # ONNX / TensorRT inference
            img_batch = img[np.newaxis, ...].astype(np.float32)
            output = self.model(img_batch)

            # Remove batch dimension
            if output.ndim == 4:
                output = output[0]

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Postprocess: CHW -> HWC, scale to 0-255
        output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        return output

    def _apply_color_transfer(
        self,
        swapped_face: np.ndarray,
        target_face: np.ndarray,
    ) -> np.ndarray:
        """
        Apply color transfer from target to swapped face.

        Args:
            swapped_face: Swapped face (H, W, 3) BGR.
            target_face: Original target face (H, W, 3) BGR.

        Returns:
            Color-corrected face (H, W, 3) BGR.
        """
        from visagen.postprocess.color_transfer import (
            linear_color_transfer,
            reinhard_color_transfer,
        )

        mode = self.config.color_transfer_mode.lower()

        if mode == "rct":
            return reinhard_color_transfer(swapped_face, target_face)
        elif mode == "lct":
            return linear_color_transfer(swapped_face, target_face)
        else:
            # Unknown mode, return unchanged
            return swapped_face

    def _apply_restoration(self, swapped_face: np.ndarray) -> np.ndarray:
        """
        Apply GFPGAN face restoration.

        Args:
            swapped_face: Swapped face (H, W, 3) BGR uint8.

        Returns:
            Restored face (H, W, 3) BGR uint8.
        """
        if self.restorer is None:
            return swapped_face

        return self.restorer.restore(swapped_face)

    def _blend_to_frame(
        self,
        frame: np.ndarray,
        swapped_face: np.ndarray,
        metadata: FaceMetadata,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Blend swapped face back into original frame.

        Args:
            frame: Original frame (H, W, 3) BGR.
            swapped_face: Swapped face (H, W, 3) BGR.
            metadata: Face metadata with warp matrix.
            mask: Face mask (H, W) float32.

        Returns:
            Blended frame (H, W, 3) BGR.
        """
        # Get inverse matrix
        matrix = metadata["matrix"]
        orig_h, orig_w = metadata["original_shape"]

        # Warp face and mask back to frame coordinates
        warped_face = self._warp_back(swapped_face, (orig_h, orig_w), matrix)
        warped_mask = self._warp_back(
            (mask * 255).astype(np.uint8),
            (orig_h, orig_w),
            matrix,
            is_mask=True,
        )

        # Process mask
        warped_mask = self._process_mask(warped_mask)

        # Apply blending
        return self._apply_blend(frame, warped_face, warped_mask)

    def _warp_back(
        self,
        face: np.ndarray,
        frame_shape: tuple[int, int],
        matrix: np.ndarray,
        is_mask: bool = False,
    ) -> np.ndarray:
        """
        Warp aligned face back to frame coordinates.

        Args:
            face: Aligned face or mask.
            frame_shape: Original frame (H, W).
            matrix: Alignment matrix.
            is_mask: Whether input is a mask.

        Returns:
            Warped image in frame coordinates.
        """
        # Compute inverse matrix
        inv_matrix = cv2.invertAffineTransform(matrix)

        # Warp
        flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        warped = cv2.warpAffine(
            face,
            inv_matrix,
            (frame_shape[1], frame_shape[0]),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped

    def _generate_mask(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate segmentation mask for face region.

        Args:
            aligned_face: Aligned face (H, W, 3) BGR.

        Returns:
            Mask (H, W) float32 in [0, 1].
        """
        try:
            # Use segmenter if available
            mask = self.segmenter.segment(aligned_face)
            return mask.astype(np.float32)
        except Exception:
            # Fallback to ellipse mask
            h, w = aligned_face.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            axes = (int(w * 0.4), int(h * 0.45))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            return mask

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Process mask with erosion and blur.

        Args:
            mask: Input mask (H, W) uint8.

        Returns:
            Processed mask (H, W) float32 in [0, 1].
        """
        # Convert to float
        mask = mask.astype(np.float32) / 255.0

        # Erode
        if self.config.mask_erode > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.mask_erode, self.config.mask_erode),
            )
            mask = cv2.erode(mask, kernel)

        # Blur for smooth edges
        if self.config.mask_blur > 0:
            ksize = self.config.mask_blur | 1  # Ensure odd
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def _apply_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply blending based on configuration.

        Args:
            background: Original frame.
            foreground: Warped swapped face.
            mask: Blending mask.

        Returns:
            Blended result.
        """
        from visagen.postprocess.blending import (
            feather_blend,
            laplacian_blend,
        )

        mode = self.config.blend_mode.lower()

        # Ensure mask is 3-channel for blending
        if mask.ndim == 2:
            np.stack([mask] * 3, axis=-1)
        else:
            pass

        if mode == "laplacian":
            return laplacian_blend(foreground, background, mask)
        elif mode == "poisson":
            # Poisson blending via OpenCV
            try:
                center = self._get_mask_center(mask)
                result = cv2.seamlessClone(
                    foreground,
                    background,
                    (mask * 255).astype(np.uint8),
                    center,
                    cv2.NORMAL_CLONE,
                )
                return result
            except Exception:
                # Fallback to feather
                return feather_blend(foreground, background, mask)
        else:
            # Default: feather blend
            return feather_blend(foreground, background, mask)

    def _get_mask_center(self, mask: np.ndarray) -> tuple[int, int]:
        """Get center point of mask for Poisson blending."""
        # Find mask bounding box
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        coords = np.where(mask_2d > 0.5)
        if len(coords[0]) == 0:
            h, w = mask_2d.shape
            return (w // 2, h // 2)

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        return (int(center_x), int(center_y))

    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening to image."""
        if not self.config.sharpen:
            return image

        # Unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(
            image,
            1 + self.config.sharpen_amount,
            blurred,
            -self.config.sharpen_amount,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)
