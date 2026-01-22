"""
FFmpeg-based video I/O for Visagen.

Provides video reading and writing using FFmpeg through ffmpeg-python,
with bundled FFmpeg binary from imageio-ffmpeg.

Features:
    - VideoReader: Stream frames from video files
    - VideoWriter: Write frames to video with audio support
    - probe_video: Get video metadata
    - extract_frames_to_dir: Extract all frames to directory
    - video_from_frames: Create video from frame sequence
    - NVENC hardware encoding support with automatic fallback
"""

import logging
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_ffmpeg = None
_ffmpeg_exe = None
_nvenc_available: bool | None = None  # Cached NVENC availability


def _get_ffmpeg():
    """Get ffmpeg module (lazy import)."""
    global _ffmpeg
    if _ffmpeg is None:
        try:
            import ffmpeg

            _ffmpeg = ffmpeg
        except ImportError:
            raise ImportError(
                "ffmpeg-python is required for video processing. "
                "Install with: pip install ffmpeg-python"
            )
    return _ffmpeg


def _get_ffmpeg_exe() -> str:
    """Get FFmpeg executable path from imageio-ffmpeg."""
    global _ffmpeg_exe
    if _ffmpeg_exe is None:
        try:
            import imageio_ffmpeg

            _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            # Fallback to system FFmpeg
            _ffmpeg_exe = "ffmpeg"
    return _ffmpeg_exe


def _get_ffprobe_exe() -> str:
    """Get FFprobe executable path."""
    ffmpeg_exe = _get_ffmpeg_exe()
    # imageio-ffmpeg provides ffmpeg, derive ffprobe path
    if "imageio_ffmpeg" in ffmpeg_exe:
        # Same directory, different name
        ffprobe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
        if Path(ffprobe).exists():
            return ffprobe
    return "ffprobe"


# =============================================================================
# Encoder Configuration
# =============================================================================

# Supported encoder types
EncoderType = Literal[
    "auto",  # Auto-select best available
    "libx264",  # Software H.264
    "libx265",  # Software H.265/HEVC
    "h264_nvenc",  # NVIDIA NVENC H.264
    "hevc_nvenc",  # NVIDIA NVENC H.265/HEVC
]

# NVENC presets (p1=fastest, p7=best quality)
NVENC_PRESETS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]

# Software encoder presets
SOFTWARE_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]


@dataclass
class EncoderConfig:
    """
    Video encoder configuration with hardware acceleration support.

    Supports both software (libx264/libx265) and hardware (NVENC) encoders
    with appropriate preset and quality settings for each.

    Attributes:
        codec: Encoder codec. "auto" selects best available.
        preset: Encoding preset. Software: ultrafast-veryslow, NVENC: p1-p7.
        crf: Constant rate factor for software encoders (0-51, lower=better).
        cq: Constant quality for NVENC (0-51, lower=better).
        rc: NVENC rate control mode (vbr, cbr, cq).
        bitrate: Optional target bitrate (e.g., "5M", "10M").
        gpu_index: GPU index for NVENC (multi-GPU support).

    Example:
        >>> config = EncoderConfig(codec="h264_nvenc", preset="p4", cq=23)
        >>> config.is_hardware()
        True
    """

    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    cq: int = 23
    rc: str = "vbr"
    bitrate: str | None = None
    gpu_index: int = 0

    def is_hardware(self) -> bool:
        """Check if using hardware encoder."""
        return self.codec in ("h264_nvenc", "hevc_nvenc")

    def get_ffmpeg_args(self) -> dict:
        """
        Get FFmpeg output arguments for this encoder configuration.

        Returns:
            Dictionary of FFmpeg output arguments.
        """
        if self.is_hardware():
            # NVENC encoder arguments
            args = {
                "vcodec": self.codec,
                "rc": self.rc,
                "cq": self.cq,
                "gpu": self.gpu_index,
            }
            # Map preset to NVENC format
            if self.preset in NVENC_PRESETS:
                args["preset"] = self.preset
            else:
                # Map software preset to NVENC equivalent
                preset_map = {
                    "ultrafast": "p1",
                    "superfast": "p2",
                    "veryfast": "p3",
                    "faster": "p3",
                    "fast": "p4",
                    "medium": "p4",
                    "slow": "p5",
                    "slower": "p6",
                    "veryslow": "p7",
                }
                args["preset"] = preset_map.get(self.preset, "p4")
        else:
            # Software encoder arguments
            args = {
                "vcodec": self.codec,
                "crf": self.crf,
                "preset": self.preset if self.preset in SOFTWARE_PRESETS else "medium",
            }

        # Add bitrate if specified
        if self.bitrate:
            args["b:v"] = self.bitrate

        return args


def check_nvenc_available() -> bool:
    """
    Check if NVENC hardware encoder is available.

    Checks FFmpeg encoders list for h264_nvenc support.
    Result is cached after first call.

    Returns:
        True if NVENC is available, False otherwise.

    Example:
        >>> if check_nvenc_available():
        ...     print("NVENC hardware encoding available!")
    """
    global _nvenc_available

    if _nvenc_available is not None:
        return _nvenc_available

    ffmpeg_exe = _get_ffmpeg_exe()
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _nvenc_available = "h264_nvenc" in result.stdout
    except Exception:
        _nvenc_available = False

    if _nvenc_available:
        logger.debug("NVENC hardware encoder is available")
    else:
        logger.debug("NVENC not available, using software encoder")

    return _nvenc_available


def get_available_encoders() -> dict[str, bool]:
    """
    Get dictionary of available video encoders.

    Returns:
        Dict mapping encoder names to availability status.

    Example:
        >>> encoders = get_available_encoders()
        >>> encoders["libx264"]
        True
        >>> encoders["h264_nvenc"]  # Depends on system
        True/False
    """
    nvenc = check_nvenc_available()
    return {
        "libx264": True,  # Always available
        "libx265": True,  # Always available
        "h264_nvenc": nvenc,
        "hevc_nvenc": nvenc,
    }


def select_best_encoder(prefer_hardware: bool = True) -> str:
    """
    Select the best available encoder.

    Args:
        prefer_hardware: Prefer hardware encoder if available.

    Returns:
        Encoder codec name.

    Example:
        >>> codec = select_best_encoder()
        >>> codec in ["h264_nvenc", "libx264"]
        True
    """
    if prefer_hardware and check_nvenc_available():
        return "h264_nvenc"
    return "libx264"


@dataclass
class VideoInfo:
    """
    Video metadata container.

    Attributes:
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames.
        duration: Duration in seconds.
        has_audio: Whether video has audio track.
        codec: Video codec name.
        audio_codec: Audio codec name (if has_audio).
    """

    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    has_audio: bool
    codec: str
    audio_codec: str | None = None


def probe_video(video_path: Path) -> VideoInfo:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        VideoInfo with video metadata.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = _get_ffmpeg()
    ffmpeg_exe = _get_ffmpeg_exe()

    try:
        probe = ffmpeg.probe(
            str(video_path), cmd=ffmpeg_exe.replace("ffmpeg", "ffprobe")
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}"
        )

    # Find video stream
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError("No video stream found")

    # Find audio stream
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"),
        None,
    )

    # Parse FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)

    # Parse duration
    duration = float(probe.get("format", {}).get("duration", 0))

    # Calculate total frames
    nb_frames = video_stream.get("nb_frames")
    if nb_frames:
        total_frames = int(nb_frames)
    else:
        total_frames = int(duration * fps)

    return VideoInfo(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        has_audio=audio_stream is not None,
        codec=video_stream.get("codec_name", "unknown"),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
    )


class VideoReader:
    """
    FFmpeg-based video frame reader.

    Streams frames from video files using FFmpeg subprocess,
    avoiding memory issues with large videos.

    Args:
        video_path: Path to video file.
        output_size: Optional (width, height) to resize frames.
        pixel_format: Output pixel format. Default: "bgr24" (OpenCV compatible).

    Example:
        >>> with VideoReader("video.mp4") as reader:
        ...     for idx, frame in reader.iter_frames():
        ...         process(frame)
    """

    def __init__(
        self,
        video_path: Path,
        output_size: tuple[int, int] | None = None,
        pixel_format: str = "bgr24",
    ) -> None:
        self.video_path = Path(video_path)
        self.output_size = output_size
        self.pixel_format = pixel_format
        self._info: VideoInfo | None = None
        self._process: subprocess.Popen | None = None

    def get_info(self) -> VideoInfo:
        """Get video metadata (cached)."""
        if self._info is None:
            self._info = probe_video(self.video_path)
        return self._info

    @property
    def width(self) -> int:
        """Output frame width."""
        if self.output_size:
            return self.output_size[0]
        return self.get_info().width

    @property
    def height(self) -> int:
        """Output frame height."""
        if self.output_size:
            return self.output_size[1]
        return self.get_info().height

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
        skip: int = 1,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Iterate over video frames.

        Args:
            start: Starting frame index.
            end: Ending frame index (exclusive). None for all frames.
            skip: Frame skip interval. Default: 1 (every frame).

        Yields:
            Tuple of (frame_index, frame_array).
            Frame array is (H, W, 3) uint8 in BGR format.
        """
        ffmpeg = _get_ffmpeg()
        ffmpeg_exe = _get_ffmpeg_exe()
        info = self.get_info()

        if end is None:
            end = info.total_frames

        # Build input
        input_kwargs = {}
        if start > 0:
            # Seek to start position
            input_kwargs["ss"] = start / info.fps

        stream = ffmpeg.input(str(self.video_path), **input_kwargs)

        # Apply filters
        if self.output_size:
            stream = stream.filter("scale", self.output_size[0], self.output_size[1])

        # Output to pipe
        stream = stream.output(
            "pipe:",
            format="rawvideo",
            pix_fmt=self.pixel_format,
            vframes=end - start,
        )

        # Run process
        process = stream.run_async(
            pipe_stdout=True,
            pipe_stderr=subprocess.PIPE,  # Capture errors for debugging
            cmd=ffmpeg_exe,
        )
        self._process = process

        frame_size = self.width * self.height * 3
        frame_idx = start

        try:
            while frame_idx < end:
                # Read one frame
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes or len(in_bytes) < frame_size:
                    break

                # Convert to numpy array
                frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                # Apply skip
                if (frame_idx - start) % skip == 0:
                    yield frame_idx, frame.copy()

                frame_idx += 1

        finally:
            process.stdout.close()
            process.wait()
            self._process = None

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """
        Read a single frame by index.

        Args:
            frame_idx: Frame index to read.

        Returns:
            Frame array or None if frame doesn't exist.
        """
        for _idx, frame in self.iter_frames(start=frame_idx, end=frame_idx + 1):
            return frame
        return None

    def extract_audio(self, output_path: Path) -> bool:
        """
        Extract audio track to file.

        Args:
            output_path: Path for output audio file.

        Returns:
            True if audio was extracted, False if no audio.
        """
        info = self.get_info()
        if not info.has_audio:
            return False

        ffmpeg = _get_ffmpeg()
        ffmpeg_exe = _get_ffmpeg_exe()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            (
                ffmpeg.input(str(self.video_path))
                .output(str(output_path), acodec="copy", vn=None)
                .overwrite_output()
                .run(quiet=True, cmd=ffmpeg_exe)
            )
            return True
        except ffmpeg.Error:
            return False

    def close(self) -> None:
        """Close any open subprocess."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VideoWriter:
    """
    FFmpeg-based video writer with hardware acceleration support.

    Writes frames to video file using FFmpeg subprocess,
    with optional audio track from source file.

    Supports both software (libx264/libx265) and hardware (NVENC) encoders
    with automatic fallback when hardware is unavailable.

    Args:
        output_path: Path for output video file.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        codec: Video codec. "auto" selects best available. Default: "auto".
        crf: Constant rate factor for software encoders (0-51). Default: 18.
        preset: Encoding preset. Default: "medium".
        pixel_format: Output pixel format. Default: "yuv420p".
        audio_source: Optional source file for audio track.
        encoder_config: Optional EncoderConfig for advanced settings.

    Example:
        >>> # Auto-select best encoder (NVENC if available)
        >>> with VideoWriter("output.mp4", 1920, 1080, 30) as writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)

        >>> # Force software encoder
        >>> with VideoWriter("output.mp4", 1920, 1080, 30, codec="libx264") as writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)

        >>> # Use NVENC with custom quality
        >>> config = EncoderConfig(codec="h264_nvenc", preset="p4", cq=20)
        >>> with VideoWriter("output.mp4", 1920, 1080, 30, encoder_config=config) as writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "auto",
        crf: int = 18,
        preset: str = "medium",
        pixel_format: str = "yuv420p",
        audio_source: Path | None = None,
        encoder_config: EncoderConfig | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.audio_source = Path(audio_source) if audio_source else None

        # Handle encoder configuration
        if encoder_config is not None:
            self.encoder_config = encoder_config
        else:
            # Auto-select or use specified codec
            if codec == "auto":
                codec = select_best_encoder(prefer_hardware=True)
                logger.info(f"Auto-selected encoder: {codec}")

            self.encoder_config = EncoderConfig(
                codec=codec,
                preset=preset,
                crf=crf,
            )

        # Store for backward compatibility
        self.codec = self.encoder_config.codec
        self.crf = crf
        self.preset = preset

        self._process: subprocess.Popen | None = None
        self._temp_video: Path | None = None
        self._frame_count = 0

        # Log encoder selection
        if self.encoder_config.is_hardware():
            logger.debug(
                f"Using hardware encoder: {self.encoder_config.codec} "
                f"(preset={self.encoder_config.get_ffmpeg_args().get('preset', 'default')})"
            )
        else:
            logger.debug(
                f"Using software encoder: {self.encoder_config.codec} "
                f"(preset={self.encoder_config.preset}, crf={self.encoder_config.crf})"
            )

    def _start_process(self) -> None:
        """Start FFmpeg encoding process."""
        if self._process is not None:
            return

        ffmpeg = _get_ffmpeg()
        ffmpeg_exe = _get_ffmpeg_exe()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # If we need to add audio later, write to temp file first
        if self.audio_source and self.audio_source.exists():
            self._temp_video = Path(tempfile.mktemp(suffix=".mp4"))
            output_file = str(self._temp_video)
        else:
            output_file = str(self.output_path)

        # Build FFmpeg command
        stream = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{self.width}x{self.height}",
            r=self.fps,
        )

        # Get encoder-specific arguments
        encoder_args = self.encoder_config.get_ffmpeg_args()
        encoder_args["pix_fmt"] = self.pixel_format

        stream = stream.output(
            output_file,
            **encoder_args,
        ).overwrite_output()

        self._process = stream.run_async(
            pipe_stdin=True,
            pipe_stderr=subprocess.PIPE,  # Capture errors for debugging
            cmd=ffmpeg_exe,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Frame array (H, W, 3) uint8 in BGR format.
        """
        if self._process is None:
            self._start_process()

        # Ensure correct shape
        if frame.shape[:2] != (self.height, self.width):
            import cv2

            frame = cv2.resize(frame, (self.width, self.height))

        # Write to stdin
        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def finalize(self) -> None:
        """Finalize video encoding and add audio if needed."""
        if self._process is None:
            return

        # Close stdin and wait for process
        self._process.stdin.close()
        self._process.wait()
        self._process = None

        # Add audio if needed
        if self._temp_video and self._temp_video.exists():
            self._merge_audio()

    def _merge_audio(self) -> None:
        """Merge audio from source into output."""
        ffmpeg = _get_ffmpeg()
        ffmpeg_exe = _get_ffmpeg_exe()

        try:
            # Combine video and audio
            video = ffmpeg.input(str(self._temp_video))
            audio = ffmpeg.input(str(self.audio_source))

            (
                ffmpeg.output(
                    video,
                    audio,
                    str(self.output_path),
                    vcodec="copy",
                    acodec="aac",
                    shortest=None,
                )
                .overwrite_output()
                .run(quiet=True, cmd=ffmpeg_exe)
            )
        except ffmpeg.Error as e:
            # If merge fails, just use video without audio
            import shutil

            logger.warning(f"Audio merge failed, video will have no audio: {e}")
            shutil.move(str(self._temp_video), str(self.output_path))
        finally:
            # Clean up temp file
            if self._temp_video and self._temp_video.exists():
                self._temp_video.unlink()

    def close(self) -> None:
        """Close writer and finalize video."""
        self.finalize()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        """Number of frames written."""
        return self._frame_count


def extract_frames_to_dir(
    video_path: Path,
    output_dir: Path,
    ext: str = "png",
    fps: float | None = None,
    quality: int = 95,
) -> int:
    """
    Extract all frames from video to directory.

    Args:
        video_path: Path to input video.
        output_dir: Directory for output frames.
        ext: Output image extension. Default: "png".
        fps: Output FPS (None for original). Default: None.
        quality: JPEG quality if ext is "jpg". Default: 95.

    Returns:
        Number of frames extracted.
    """
    ffmpeg = _get_ffmpeg()
    ffmpeg_exe = _get_ffmpeg_exe()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    stream = ffmpeg.input(str(video_path))

    if fps is not None:
        stream = stream.filter("fps", fps=fps)

    output_pattern = str(output_dir / f"%06d.{ext}")

    output_kwargs = {}
    if ext in ("jpg", "jpeg"):
        output_kwargs["qscale:v"] = int((100 - quality) / 100 * 31)

    stream = stream.output(output_pattern, **output_kwargs).overwrite_output()

    try:
        stream.run(quiet=True, cmd=ffmpeg_exe)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to extract frames: {e}")

    # Count extracted frames
    frame_files = list(output_dir.glob(f"*.{ext}"))
    return len(frame_files)


def video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: float,
    audio_source: Path | None = None,
    codec: str = "auto",
    crf: int = 18,
    pattern: str = "%06d.png",
) -> None:
    """
    Create video from frame sequence with hardware acceleration support.

    Args:
        frames_dir: Directory containing frames.
        output_path: Path for output video.
        fps: Frames per second.
        audio_source: Optional source for audio track.
        codec: Video codec. "auto" selects best available (NVENC if GPU). Default: "auto".
        crf: Quality factor. Default: 18.
        pattern: Frame filename pattern. Default: "%06d.png".
    """
    ffmpeg = _get_ffmpeg()
    ffmpeg_exe = _get_ffmpeg_exe()

    # NVENC auto-select
    if codec == "auto":
        codec = select_best_encoder(prefer_hardware=True)
        logger.info(f"Auto-selected encoder for frames: {codec}")

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_pattern = str(frames_dir / pattern)

    # Build command
    stream = ffmpeg.input(input_pattern, framerate=fps)

    # Encoder-specific output kwargs
    if codec in ("h264_nvenc", "hevc_nvenc"):
        # NVENC hardware encoder
        output_kwargs = {
            "vcodec": codec,
            "pix_fmt": "yuv420p",
            "rc": "vbr",
            "cq": crf,
            "preset": "p4",
        }
    else:
        # Software encoder
        output_kwargs = {
            "vcodec": codec,
            "pix_fmt": "yuv420p",
            "crf": crf,
        }

    # Add audio if provided
    if audio_source and Path(audio_source).exists():
        audio = ffmpeg.input(str(audio_source))
        stream = ffmpeg.output(
            stream,
            audio,
            str(output_path),
            acodec="aac",
            shortest=None,
            **output_kwargs,
        )
    else:
        stream = stream.output(str(output_path), **output_kwargs)

    stream = stream.overwrite_output()

    try:
        stream.run(quiet=True, cmd=ffmpeg_exe)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to create video: {e}")
