"""
High-level face merger orchestration for Visagen.

Provides FaceMerger class that orchestrates the complete video
processing pipeline: frame extraction, face swap, and video encoding.

Features:
    - Video and frame directory support
    - Resume capability with checkpoint files
    - Progress tracking with callbacks
    - Audio preservation
    - YAML configuration export/import
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from visagen.merger.frame_processor import (
    FrameProcessor,
    FrameProcessorConfig,
)
from visagen.merger.video_io import (
    VideoInfo,
    VideoReader,
    probe_video,
    video_from_frames,
)

logger = logging.getLogger(__name__)


@dataclass
class MergerConfig:
    """
    Configuration for video face merger.

    Attributes:
        input_path: Input video file or frame directory.
        output_path: Output video file or directory.
        checkpoint_path: Path to trained model checkpoint.
        frame_processor_config: Frame processing configuration.
        num_workers: Number of parallel workers. Default: 1.
        codec: Video codec. "auto" selects best available (NVENC if GPU). Default: "auto".
        crf: Quality factor for software encoders (0-51, lower is better). Default: 18.
        preset: Encoding preset. Default: "medium".
        copy_audio: Copy audio from source. Default: True.
        resume: Resume from previous run. Default: True.
        temp_dir: Temporary directory for frames. Default: None (auto).
        device: Torch device. Default: None (auto).
    """

    input_path: Path
    output_path: Path
    checkpoint_path: Path
    frame_processor_config: FrameProcessorConfig | None = None
    num_workers: int = 1
    codec: str = "auto"
    crf: int = 18
    preset: str = "medium"
    copy_audio: bool = True
    resume: bool = True
    temp_dir: Path | None = None
    device: str | None = None

    def __post_init__(self):
        """Convert paths to Path objects."""
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        self.checkpoint_path = Path(self.checkpoint_path)
        if self.temp_dir:
            self.temp_dir = Path(self.temp_dir)
        if self.frame_processor_config is None:
            self.frame_processor_config = FrameProcessorConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "MergerConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            MergerConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested frame_processor_config
        if "frame_processor_config" in data and data["frame_processor_config"]:
            data["frame_processor_config"] = FrameProcessorConfig(
                **data["frame_processor_config"]
            )

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path for output YAML file.
        """
        data = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "checkpoint_path": str(self.checkpoint_path),
            "frame_processor_config": asdict(self.frame_processor_config)
            if self.frame_processor_config
            else None,
            "num_workers": self.num_workers,
            "codec": self.codec,
            "crf": self.crf,
            "preset": self.preset,
            "copy_audio": self.copy_audio,
            "resume": self.resume,
            "temp_dir": str(self.temp_dir) if self.temp_dir else None,
            "device": self.device,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class MergerStats:
    """
    Merger execution statistics.

    Attributes:
        total_frames: Total frames in input.
        processed_frames: Frames successfully processed.
        skipped_frames: Frames skipped (already processed).
        failed_frames: Frames that failed processing.
        faces_detected: Total faces detected.
        faces_swapped: Total faces swapped.
        total_time: Total processing time in seconds.
        avg_time_per_frame: Average time per frame.
        fps: Processing frames per second.
    """

    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    failed_frames: int = 0
    faces_detected: int = 0
    faces_swapped: int = 0
    total_time: float = 0.0
    avg_time_per_frame: float = 0.0
    fps: float = 0.0

    def update_averages(self) -> None:
        """Update calculated fields."""
        if self.processed_frames > 0:
            self.avg_time_per_frame = self.total_time / self.processed_frames
            self.fps = (
                self.processed_frames / self.total_time if self.total_time > 0 else 0.0
            )


class FaceMerger:
    """
    Video face swap merger.

    Orchestrates the complete pipeline for swapping faces in videos:
    1. Extract frames from video
    2. Process each frame with face swap
    3. Encode processed frames to output video
    4. Copy audio from source

    Args:
        config: MergerConfig with all settings.
        progress_callback: Optional callback for progress updates.
            Called with (current_frame, total_frames).

    Example:
        >>> config = MergerConfig(
        ...     input_path=Path("input.mp4"),
        ...     output_path=Path("output.mp4"),
        ...     checkpoint_path=Path("model.ckpt"),
        ... )
        >>> merger = FaceMerger(config)
        >>> stats = merger.run()
        >>> print(f"Processed {stats.total_frames} frames at {stats.fps:.1f} FPS")
    """

    def __init__(
        self,
        config: MergerConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self._processor: FrameProcessor | None = None

    @property
    def processor(self) -> FrameProcessor:
        """Lazy-load frame processor."""
        if self._processor is None:
            self._processor = FrameProcessor(
                model=self.config.checkpoint_path,
                config=self.config.frame_processor_config,
                device=self.config.device,
            )
        return self._processor

    def run(self) -> MergerStats:
        """
        Execute full merge pipeline.

        Returns:
            MergerStats with processing statistics.
        """
        # Validate inputs
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input not found: {self.config.input_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.config.checkpoint_path}"
            )

        # Determine input type
        if self.config.input_path.is_dir():
            return self._process_frames_dir()
        else:
            return self._process_video()

    def run_frames(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> MergerStats:
        """
        Process directory of frames.

        Args:
            input_dir: Directory with input frames.
            output_dir: Directory for output frames.

        Returns:
            MergerStats with processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get input frames
        frame_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError(f"No frames found in {input_dir}")

        stats = MergerStats(total_frames=len(frame_files))
        start_time = time.time()

        # Load resume state
        processed_indices = set()
        if self.config.resume:
            processed_indices = self._load_resume_state(output_dir)
            stats.skipped_frames = len(processed_indices)

        try:
            for idx, frame_path in enumerate(frame_files):
                # Check if already processed
                output_path = output_dir / frame_path.name
                if idx in processed_indices and output_path.exists():
                    continue

                # Load and process frame
                import cv2

                frame = cv2.imread(str(frame_path))
                if frame is None:
                    stats.failed_frames += 1
                    continue

                result = self.processor.process_frame(frame, frame_idx=idx)

                # Save result
                cv2.imwrite(str(output_path), result.output_image)

                # Update stats
                stats.processed_frames += 1
                stats.faces_detected += result.faces_detected
                stats.faces_swapped += result.faces_swapped
                processed_indices.add(idx)

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(idx + 1, stats.total_frames)

                # Save checkpoint periodically
                if (idx + 1) % 100 == 0:
                    self._save_resume_state(output_dir, processed_indices)

        finally:
            # Save final state
            self._save_resume_state(output_dir, processed_indices)

        stats.total_time = time.time() - start_time
        stats.update_averages()

        return stats

    def _process_video(self) -> MergerStats:
        """Process video file."""
        # Set up workspace
        temp_dir = self._prepare_workspace()
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            # Get video info
            video_info = probe_video(self.config.input_path)
            stats = MergerStats(total_frames=video_info.total_frames)
            start_time = time.time()

            # Load resume state
            processed_indices = set()
            if self.config.resume:
                processed_indices = self._load_resume_state(frames_dir)
                stats.skipped_frames = len(processed_indices)

            # Extract audio if needed
            audio_path = None
            if self.config.copy_audio and video_info.has_audio:
                audio_path = temp_dir / "audio.aac"
                with VideoReader(self.config.input_path) as reader:
                    reader.extract_audio(audio_path)

            # Process frames
            with VideoReader(self.config.input_path) as reader:
                for frame_idx, frame in reader.iter_frames():
                    # Check if already processed
                    output_path = frames_dir / f"{frame_idx:06d}.png"
                    if frame_idx in processed_indices and output_path.exists():
                        continue

                    # Process frame
                    result = self.processor.process_frame(frame, frame_idx=frame_idx)

                    # Save frame
                    import cv2

                    cv2.imwrite(str(output_path), result.output_image)

                    # Update stats
                    stats.processed_frames += 1
                    stats.faces_detected += result.faces_detected
                    stats.faces_swapped += result.faces_swapped
                    processed_indices.add(frame_idx)

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(frame_idx + 1, stats.total_frames)

                    # Save checkpoint periodically
                    if (frame_idx + 1) % 100 == 0:
                        self._save_resume_state(frames_dir, processed_indices)

            # Save final resume state
            self._save_resume_state(frames_dir, processed_indices)

            # Encode output video
            self._encode_output(
                frames_dir,
                video_info,
                audio_path,
            )

            stats.total_time = time.time() - start_time
            stats.update_averages()

            return stats

        finally:
            # Clean up temp directory (only if successful and not resumable)
            if not self.config.resume and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_frames_dir(self) -> MergerStats:
        """Process directory of frames."""
        if self.config.output_path.suffix:
            # Output is a video file
            temp_dir = self._prepare_workspace()
            output_frames = temp_dir / "output_frames"
        else:
            # Output is a directory
            output_frames = self.config.output_path

        stats = self.run_frames(self.config.input_path, output_frames)

        # If output is video, encode it
        if self.config.output_path.suffix:
            # Estimate FPS from filename pattern or default to 30
            fps = 30.0
            video_from_frames(
                output_frames,
                self.config.output_path,
                fps=fps,
                codec=self.config.codec,
                crf=self.config.crf,
            )

        return stats

    def _prepare_workspace(self) -> Path:
        """Set up temporary directory."""
        if self.config.temp_dir:
            temp_dir = self.config.temp_dir
        else:
            # Create temp dir based on output name
            temp_dir = (
                self.config.output_path.parent / f".{self.config.output_path.stem}_temp"
            )

        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _load_resume_state(self, frames_dir: Path) -> set[int]:
        """Load set of already processed frame indices."""
        checkpoint_file = frames_dir / ".checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    return set(data.get("processed_frames", []))
            except Exception as e:
                logger.warning(f"Failed to load resume state: {e}")
        return set()

    def _save_resume_state(self, frames_dir: Path, processed: set[int]) -> None:
        """Save resume checkpoint."""
        checkpoint_file = frames_dir / ".checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump({"processed_frames": list(processed)}, f)
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")

    def _encode_output(
        self,
        frames_dir: Path,
        video_info: VideoInfo,
        audio_path: Path | None,
    ) -> None:
        """Encode processed frames to video."""
        video_from_frames(
            frames_dir,
            self.config.output_path,
            fps=video_info.fps,
            audio_source=audio_path if audio_path and audio_path.exists() else None,
            codec=self.config.codec,
            crf=self.config.crf,
        )
