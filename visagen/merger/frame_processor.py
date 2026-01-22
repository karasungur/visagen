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

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch

from visagen.postprocess.motion_blur import apply_motion_blur_to_face

# Type alias for face metadata
FaceMetadata = dict[str, Any]

logger = logging.getLogger(__name__)


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
    hist_match_threshold: int = 238  # 0-255, used for hist-match mode

    # Blending
    blend_mode: str = "laplacian"
    blend_amount: float = 1.0
    mask_erode: int = 5
    mask_dilate: int = 0
    mask_blur: int = 5

    # Post-processing
    sharpen: bool = False
    sharpen_amount: float = 0.3

    # Face restoration (GFPGAN)
    restore_face: bool = False
    restore_strength: float = 0.5
    restore_model_version: float = 1.4

    # Super resolution (4x upscale with GFPGAN)
    # 0 = disabled, 1-100 = blend power with 4x upscaled enhanced face
    super_resolution_power: int = 0

    # Motion blur (for temporal consistency)
    motion_blur_power: int = 0  # 0-100

    # Motion blur parameters
    motion_blur_magnitude: float = 10.0  # 0-50
    motion_blur_angle: float = 0.0  # 0-360 degrees

    # Face scale adjustment (-50 to 50, applied as 1.0 + 0.01*value)
    face_scale: float = 0.0

    # Image degradation effects
    image_denoise_power: int = 0  # 0-500
    bicubic_degrade_power: int = 0  # 0-100
    color_degrade_power: int = 0  # 0-100
    degrade_full_frame: bool = False  # Apply degradation to full frame

    # Motion blur auto-detection (optical flow based)
    motion_blur_auto: bool = False  # Auto-detect motion from optical flow


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

        # Optical flow state for motion blur auto-detection
        self._prev_frame_gray: np.ndarray | None = None

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

                    # Apply super resolution (4x upscale with blend)
                    if self.config.super_resolution_power > 0:
                        swapped_face = self._apply_super_resolution(swapped_face)

                    # Apply motion blur (for temporal consistency)
                    if self.config.motion_blur_power > 0:
                        # Auto-detect motion using optical flow
                        if self.config.motion_blur_auto:
                            motion_mag, motion_deg = self._analyze_motion(frame)
                        else:
                            motion_mag = self.config.motion_blur_magnitude
                            motion_deg = self.config.motion_blur_angle

                        swapped_face = apply_motion_blur_to_face(
                            swapped_face,
                            motion_power=motion_mag,
                            motion_deg=motion_deg,
                            blur_power=self.config.motion_blur_power,
                            super_resolution=self.config.super_resolution_power > 0,
                        )

                    # Apply degradation effects (denoise, bicubic, color degrade)
                    if (
                        self.config.image_denoise_power > 0
                        or self.config.bicubic_degrade_power > 0
                        or self.config.color_degrade_power > 0
                    ):
                        from visagen.postprocess.degrade import (
                            apply_degradation_pipeline,
                        )

                        # Normalize to float32 [0, 1]
                        swapped_face_f32 = swapped_face.astype(np.float32) / 255.0

                        swapped_face_f32 = apply_degradation_pipeline(
                            swapped_face_f32,
                            denoise_power=self.config.image_denoise_power,
                            bicubic_power=self.config.bicubic_degrade_power,
                            color_power=self.config.color_degrade_power,
                        )

                        swapped_face = np.clip(swapped_face_f32 * 255, 0, 255).astype(
                            np.uint8
                        )

                    # Blend back to frame
                    output = self._blend_to_frame(output, swapped_face, face_meta, mask)

                    faces_swapped += 1
                except Exception as e:
                    # Skip this face on error
                    logger.debug(f"Face processing failed: {e}")
                    continue

        except Exception as e:
            # Return original frame on error
            logger.debug(f"Frame processing failed: {e}")

        # Apply full-frame degradation if enabled
        if self.config.degrade_full_frame and (
            self.config.image_denoise_power > 0
            or self.config.bicubic_degrade_power > 0
            or self.config.color_degrade_power > 0
        ):
            from visagen.postprocess.degrade import apply_degradation_pipeline

            output_f32 = output.astype(np.float32) / 255.0
            output_f32 = apply_degradation_pipeline(
                output_f32,
                denoise_power=self.config.image_denoise_power,
                bicubic_power=self.config.bicubic_degrade_power,
                color_power=self.config.color_degrade_power,
            )
            output = np.clip(output_f32 * 255, 0, 255).astype(np.uint8)

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
            # 1. Input matrix - for model inference (scale=1.0 FIXED)
            aligned, _input_matrix = self.aligner.align(
                frame,
                face.landmarks,
                scale=1.0,
            )

            # 2. Output matrix - for warp-back (user setting)
            output_scale = 1.0 + 0.01 * self.config.face_scale
            output_matrix = self.aligner.get_transform_mat(
                face.landmarks,
                self.config.output_size,
                self.aligner.face_type,
                scale=output_scale,
            )

            # Generate mask
            mask = self._generate_mask(aligned)

            # Store metadata for warp-back (use OUTPUT matrix)
            metadata = {
                "bbox": face.bbox,
                "landmarks": face.landmarks,
                "matrix": output_matrix,
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
        elif mode == "hist-match":
            from visagen.postprocess.color_transfer import color_hist_match

            # Mask oluştur (face bölgesi için)
            face_mask = None
            if hasattr(self, "_current_mask"):
                face_mask = (self._current_mask > 0.5).astype(np.uint8)

            return color_hist_match(
                swapped_face,
                target_face,
                hist_match_threshold=self.config.hist_match_threshold,
                mask=face_mask,
            )
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

    def _apply_super_resolution(self, swapped_face: np.ndarray) -> np.ndarray:
        """
        Apply super resolution with 4x upscale and gradual blending.

        When enabled, upscales the face 4x using GFPGAN and blends with the
        original based on power setting (0-100).

        Args:
            swapped_face: Swapped face (H, W, 3) BGR uint8.

        Returns:
            Super-resolved face (H, W, 3) BGR uint8.
        """
        power = self.config.super_resolution_power
        if power == 0:
            return swapped_face

        # Ensure restorer is available
        if self._restorer is None:
            from visagen.postprocess.restore import FaceRestorer, RestoreConfig

            restore_config = RestoreConfig(
                enabled=True,
                strength=1.0,  # Full strength for super resolution
                model_version=self.config.restore_model_version,
            )
            self._restorer = FaceRestorer(restore_config, device=self.device)

        # Get original size
        h, w = swapped_face.shape[:2]
        output_size = w * 4  # 4x upscale

        # Apply GFPGAN restoration (returns 4x upscaled)
        enhanced = self._restorer.restore(swapped_face, upscale=4)

        # Ensure enhanced is the correct size
        if enhanced.shape[0] != output_size or enhanced.shape[1] != output_size:
            enhanced = cv2.resize(
                enhanced, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
            )

        # Upscale original for blending
        face_upscaled = cv2.resize(
            swapped_face, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
        )

        # Blend based on power (0-100)
        mod = power / 100.0
        result = (
            face_upscaled.astype(np.float32) * (1.0 - mod)
            + enhanced.astype(np.float32) * mod
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Resize back to original resolution
        if result.shape[0] != h or result.shape[1] != w:
            result = cv2.resize(
                result,
                (w, h),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return result

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
        result = self._apply_blend(frame, warped_face, warped_mask)

        # POST-SEAMLESS COLOR TRANSFER
        # After seamless clone, warp result back to aligned space for color transfer
        if (
            self.config.blend_mode.lower() == "poisson"
            and self.config.color_transfer_mode
        ):
            # 1. Warp result back to aligned space
            aligned_result = cv2.warpAffine(
                result,
                matrix,
                (self.config.output_size, self.config.output_size),
                flags=cv2.INTER_CUBIC,
            )

            # 2. Get aligned target face (original destination)
            aligned_target = cv2.warpAffine(
                frame,
                matrix,
                (self.config.output_size, self.config.output_size),
                flags=cv2.INTER_CUBIC,
            )

            # 3. Apply color transfer in aligned space
            aligned_result = self._apply_color_transfer(aligned_result, aligned_target)

            # 4. Warp corrected face back to frame
            corrected_face = self._warp_back(
                aligned_result,
                (orig_h, orig_w),
                matrix,
            )

            # 5. Final blend with mask
            warped_mask_3ch = (
                warped_mask[..., None] if warped_mask.ndim == 2 else warped_mask
            )
            if warped_mask_3ch.shape[-1] == 1:
                warped_mask_3ch = np.repeat(warped_mask_3ch, 3, axis=-1)

            result = frame * (1 - warped_mask_3ch) + corrected_face * warped_mask_3ch
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

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
            Mask (H, W) float32 with values 0 or 255.
            Note: Normalized to [0, 1] by _process_mask before blending.
        """
        try:
            # Use segmenter if available
            result = self.segmenter.segment(aligned_face)
            return result.mask.astype(np.float32)
        except Exception as e:
            # Fallback to ellipse mask
            logger.debug(f"Segmenter failed, using ellipse mask: {e}")
            h, w = aligned_face.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            axes = (int(w * 0.4), int(h * 0.45))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            return mask

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Process mask with padding, erosion, dilation, and blur.

        Adds padding before morphological operations to prevent boundary
        artifacts from boundary effects.

        Args:
            mask: Input mask (H, W) uint8.

        Returns:
            Processed mask (H, W) float32 in [0, 1].
        """
        # Convert to float
        mask = mask.astype(np.float32) / 255.0
        h, w = mask.shape[:2]

        # Calculate padding size based on max morphological operation
        # Minimum padding = output_size // 8 (32 pixels @ 256px) for boundary safety
        min_pad = self.config.output_size // 8
        pad_size = max(self.config.mask_erode, self.config.mask_dilate, min_pad)

        # Add padding to prevent boundary artifacts
        if pad_size > 0:
            mask = np.pad(mask, pad_size, mode="constant", constant_values=0)

        # Erode
        if self.config.mask_erode > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.mask_erode, self.config.mask_erode),
            )
            mask = cv2.erode(mask, kernel, iterations=1)

        # Dilate (mask expansion)
        if self.config.mask_dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.mask_dilate, self.config.mask_dilate),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Boundary clip (blur compensation) - zero out edges before blur
        if pad_size > 0:
            clip_size = pad_size + self.config.mask_blur // 2
            mask[:clip_size, :] = 0
            mask[-clip_size:, :] = 0
            mask[:, :clip_size] = 0
            mask[:, -clip_size:] = 0

        # Blur for smooth edges
        if self.config.mask_blur > 0:
            ksize = self.config.mask_blur | 1  # Ensure odd
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Remove padding
        if pad_size > 0:
            mask = mask[pad_size:-pad_size, pad_size:-pad_size]

        return np.clip(mask, 0, 1)

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
            mask = np.stack([mask] * 3, axis=-1)

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
            except Exception as e:
                # Fallback to feather
                logger.debug(f"Poisson blending failed, using feather: {e}")
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

    def _analyze_motion(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Analyze motion between current and previous frame using optical flow.

        Uses Farneback optical flow to compute motion magnitude and direction.
        This enables automatic motion blur parameters based on actual frame motion.

        Args:
            frame: Current frame (H, W, 3) BGR uint8.

        Returns:
            Tuple of (motion_magnitude, motion_angle_degrees).
            - motion_magnitude: Average motion power (0-50 range, clamped)
            - motion_angle_degrees: Average motion direction (0-360)
        """
        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If no previous frame, store and return defaults
        if self._prev_frame_gray is None:
            self._prev_frame_gray = curr_gray
            return (0.0, 0.0)

        try:
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_frame_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Convert to polar coordinates (magnitude and angle)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate average motion
            motion_power = float(np.mean(magnitude))
            motion_deg = float(np.degrees(np.mean(angle)))

            # Clamp motion power to reasonable range (0-50)
            motion_power = min(motion_power * 10, 50.0)  # Scale up for visibility

            # Normalize angle to 0-360
            motion_deg = motion_deg % 360

        except Exception as e:
            logger.debug(f"Optical flow analysis failed: {e}")
            motion_power = 0.0
            motion_deg = 0.0

        # Update previous frame
        self._prev_frame_gray = curr_gray

        return (motion_power, motion_deg)
