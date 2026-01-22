"""
Interactive Merger for Visagen.

Provides a Gradio-compatible interactive face merging system
with real-time preview and parameter adjustment.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from visagen.merger.interactive_config import (
    InteractiveMergerSession,
)

logger = logging.getLogger(__name__)


class InteractiveMerger:
    """
    Interactive face merger with real-time preview.

    Manages the interactive merging session including frame loading,
    processing with configurable parameters, and export.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        frames_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        device: Torch device for inference (default: auto).

    Example:
        >>> merger = InteractiveMerger("model.ckpt", "frames/", "output/")
        >>> merger.load_session()
        >>> preview = merger.process_current_frame()
        >>> merger.update_config(erode_mask=10, blur_mask=20)
        >>> preview = merger.process_current_frame()
        >>> merger.export_all()
    """

    # Supported image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        frames_dir: str | Path | None = None,
        output_dir: str | Path = "./output",
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.output_dir = Path(output_dir)

        self.device = device

        # Session state
        self.session = InteractiveMergerSession()
        self.frames: list[Path] = []

        # Lazy-loaded processor
        self._processor = None

        # Frame cache for performance
        self._cache: dict[int, np.ndarray] = {}
        self._cache_max_size = 10

    @property
    def processor(self):
        """Lazy-load frame processor."""
        if self._processor is None:
            if self.checkpoint_path is None:
                raise ValueError("Checkpoint path not set. Call load_session() first.")

            from visagen.merger.frame_processor import (
                FrameProcessor,
                FrameProcessorConfig,
            )

            config = FrameProcessorConfig(
                min_confidence=0.5,
                max_faces=1,
            )

            self._processor = FrameProcessor(
                model=self.checkpoint_path,
                config=config,
                device=self.device,
            )

        return self._processor

    def load_session(
        self,
        checkpoint_path: str | Path | None = None,
        frames_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        session_path: str | Path | None = None,
    ) -> tuple[bool, str]:
        """
        Load or initialize a merger session.

        Args:
            checkpoint_path: Path to model checkpoint.
            frames_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            session_path: Path to existing session JSON file.

        Returns:
            Tuple of (success, message).
        """
        try:
            # Load existing session if provided
            if session_path:
                session_path = Path(session_path)
                if session_path.exists():
                    self.session = InteractiveMergerSession.from_json(session_path)
                    self.checkpoint_path = Path(self.session.checkpoint_path)
                    self.frames_dir = Path(self.session.frames_dir)
                    self.output_dir = Path(self.session.output_dir)
                    logger.info(f"Loaded session from {session_path}")

            # Override with provided paths
            if checkpoint_path:
                self.checkpoint_path = Path(checkpoint_path)
            if frames_dir:
                self.frames_dir = Path(frames_dir)
            if output_dir:
                self.output_dir = Path(output_dir)

            # Validate paths
            if self.checkpoint_path is None or not self.checkpoint_path.exists():
                return False, f"Checkpoint not found: {self.checkpoint_path}"

            if self.frames_dir is None or not self.frames_dir.exists():
                return False, f"Frames directory not found: {self.frames_dir}"

            # Load frame list
            self.frames = self._get_frame_list(self.frames_dir)

            if not self.frames:
                return False, f"No image files found in: {self.frames_dir}"

            # Update session
            self.session.frames_dir = str(self.frames_dir)
            self.session.checkpoint_path = str(self.checkpoint_path)
            self.session.output_dir = str(self.output_dir)
            self.session.total_frames = len(self.frames)

            # Ensure current_idx is valid
            self.session.current_idx = max(
                0, min(self.session.current_idx, len(self.frames) - 1)
            )

            # Reset processor to reload model
            self._processor = None

            # Clear cache
            self._cache.clear()

            return True, f"Loaded {len(self.frames)} frames from {self.frames_dir}"

        except Exception as e:
            logger.exception("Failed to load session")
            return False, f"Error loading session: {e}"

    def _get_frame_list(self, frames_dir: Path) -> list[Path]:
        """Get sorted list of image files in directory."""
        frames = []

        for ext in self.IMAGE_EXTENSIONS:
            frames.extend(frames_dir.glob(f"*{ext}"))
            frames.extend(frames_dir.glob(f"*{ext.upper()}"))

        # Sort by name (natural sort for numbered frames)
        frames = sorted(set(frames), key=lambda p: p.name)

        return frames

    def process_current_frame(self) -> np.ndarray | None:
        """
        Process the current frame with current configuration.

        Returns:
            Processed frame as RGB numpy array, or None on error.
        """
        if not self.frames:
            return None

        idx = self.session.current_idx
        frame_path = self.frames[idx]

        # Check cache first (only if config hasn't changed)
        cache_key = self._get_cache_key(idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.error(f"Failed to read frame: {frame_path}")
                return None

            # Apply configuration to processor
            self._apply_config_to_processor()

            # Process frame
            result = self.processor.process_frame(frame, frame_idx=idx)

            # Convert BGR to RGB for Gradio display
            output_rgb = cv2.cvtColor(result.output_image, cv2.COLOR_BGR2RGB)

            # Cache result
            self._add_to_cache(cache_key, output_rgb)

            return output_rgb

        except Exception:
            logger.exception(f"Error processing frame {idx}")
            return None

    def _get_cache_key(self, idx: int) -> int:
        """Generate cache key from index and config hash."""
        # Simple hash of config for cache invalidation
        config_str = str(self.session.config.to_dict())
        return hash((idx, config_str))

    def _add_to_cache(self, key: int, value: np.ndarray) -> None:
        """Add to cache with size limit."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def invalidate_cache(self) -> None:
        """Clear the frame cache."""
        self._cache.clear()

    def _apply_config_to_processor(self) -> None:
        """Apply current interactive config to frame processor."""
        config = self.session.config

        # Map interactive config to processor config
        proc_config = self.processor.config

        # Color transfer
        if config.color_transfer == "none":
            proc_config.color_transfer_mode = None
        else:
            proc_config.color_transfer_mode = config.color_transfer

        # Mask processing - Pozitif değer = erode, negatif değer = dilate
        if config.erode_mask > 0:
            proc_config.mask_erode = config.erode_mask
            proc_config.mask_dilate = 0
        elif config.erode_mask < 0:
            proc_config.mask_erode = 0
            proc_config.mask_dilate = abs(config.erode_mask)
        else:
            proc_config.mask_erode = 0
            proc_config.mask_dilate = 0
        proc_config.mask_blur = config.blur_mask

        # Face restoration
        proc_config.restore_face = config.restore_face
        proc_config.restore_strength = config.restore_strength

        # Super resolution (legacy 4x upscale)
        proc_config.super_resolution_power = config.super_resolution_power

        # Motion blur (for temporal consistency)
        proc_config.motion_blur_power = config.motion_blur_power

        # Sharpening
        proc_config.sharpen = config.sharpen_mode != "none"
        proc_config.sharpen_amount = config.sharpen_amount / 100.0

    def update_config(self, **kwargs) -> np.ndarray | None:
        """
        Update configuration and return processed current frame.

        Args:
            **kwargs: Configuration parameters to update.

        Returns:
            Processed frame with new configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self.session.config, key):
                setattr(self.session.config, key, value)

        # Invalidate cache since config changed
        self.invalidate_cache()

        return self.process_current_frame()

    def navigate(self, delta: int) -> tuple[np.ndarray | None, int]:
        """
        Navigate to a different frame.

        Args:
            delta: Number of frames to move (positive = forward, negative = back).

        Returns:
            Tuple of (processed_frame, new_index).
        """
        if not self.frames:
            return None, 0

        new_idx = max(0, min(len(self.frames) - 1, self.session.current_idx + delta))
        self.session.current_idx = new_idx

        return self.process_current_frame(), new_idx

    def go_to_frame(self, idx: int) -> tuple[np.ndarray | None, int]:
        """
        Go to a specific frame.

        Args:
            idx: Target frame index.

        Returns:
            Tuple of (processed_frame, actual_index).
        """
        if not self.frames:
            return None, 0

        self.session.current_idx = max(0, min(len(self.frames) - 1, idx))

        return self.process_current_frame(), self.session.current_idx

    def save_session(self, path: str | Path | None = None) -> tuple[bool, str]:
        """
        Save current session to JSON file.

        Args:
            path: Output path (default: session.json in output_dir).

        Returns:
            Tuple of (success, message).
        """
        try:
            if path is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / "session.json"
            else:
                path = Path(path)

            self.session.to_json(path)
            return True, f"Session saved to {path}"

        except Exception as e:
            return False, f"Error saving session: {e}"

    def export_frame(self, idx: int) -> tuple[bool, str]:
        """
        Export a single processed frame.

        Args:
            idx: Frame index to export.

        Returns:
            Tuple of (success, message).
        """
        if not self.frames or idx < 0 or idx >= len(self.frames):
            return False, f"Invalid frame index: {idx}"

        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Read and process frame
            frame_path = self.frames[idx]
            frame = cv2.imread(str(frame_path))

            if frame is None:
                return False, f"Failed to read frame: {frame_path}"

            # Apply configuration
            self._apply_config_to_processor()

            # Process
            result = self.processor.process_frame(frame, frame_idx=idx)

            # Save output
            output_path = self.output_dir / f"{frame_path.stem}.png"
            cv2.imwrite(str(output_path), result.output_image)

            return True, str(output_path)

        except Exception as e:
            return False, f"Error exporting frame {idx}: {e}"

    def export_all(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[bool, str, int]:
        """
        Export all frames with current configuration.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Tuple of (success, message, num_exported).
        """
        if not self.frames:
            return False, "No frames loaded", 0

        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            total = len(self.frames)
            exported = 0

            for idx, _frame_path in enumerate(self.frames):
                if progress_callback:
                    progress_callback(idx, total)

                success, _ = self.export_frame(idx)
                if success:
                    exported += 1

            # Save session with export
            self.save_session()

            return (
                True,
                f"Exported {exported}/{total} frames to {self.output_dir}",
                exported,
            )

        except Exception as e:
            return False, f"Error during export: {e}", 0

    def get_current_frame_info(self) -> dict:
        """Get information about the current frame."""
        if not self.frames:
            return {"error": "No frames loaded"}

        idx = self.session.current_idx
        frame_path = self.frames[idx]

        return {
            "index": idx,
            "total": len(self.frames),
            "filename": frame_path.name,
            "path": str(frame_path),
        }

    def get_original_frame(self) -> np.ndarray | None:
        """Get the original (unprocessed) current frame as RGB."""
        if not self.frames:
            return None

        frame_path = self.frames[self.session.current_idx]
        frame = cv2.imread(str(frame_path))

        if frame is None:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
