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
"""

import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Lazy imports for optional dependencies
_ffmpeg = None
_ffmpeg_exe = None


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
            pipe_stderr=subprocess.DEVNULL,
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
    FFmpeg-based video writer.

    Writes frames to video file using FFmpeg subprocess,
    with optional audio track from source file.

    Args:
        output_path: Path for output video file.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        codec: Video codec. Default: "libx264".
        crf: Constant rate factor (quality). Default: 18.
        preset: Encoding preset. Default: "medium".
        pixel_format: Output pixel format. Default: "yuv420p".
        audio_source: Optional source file for audio track.

    Example:
        >>> with VideoWriter("output.mp4", 1920, 1080, 30) as writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        pixel_format: str = "yuv420p",
        audio_source: Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.pixel_format = pixel_format
        self.audio_source = Path(audio_source) if audio_source else None

        self._process: subprocess.Popen | None = None
        self._temp_video: Path | None = None
        self._frame_count = 0

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

        stream = stream.output(
            output_file,
            vcodec=self.codec,
            pix_fmt=self.pixel_format,
            crf=self.crf,
            preset=self.preset,
        ).overwrite_output()

        self._process = stream.run_async(
            pipe_stdin=True,
            pipe_stderr=subprocess.DEVNULL,
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
        except ffmpeg.Error:
            # If merge fails, just use video without audio
            import shutil

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
    codec: str = "libx264",
    crf: int = 18,
    pattern: str = "%06d.png",
) -> None:
    """
    Create video from frame sequence.

    Args:
        frames_dir: Directory containing frames.
        output_path: Path for output video.
        fps: Frames per second.
        audio_source: Optional source for audio track.
        codec: Video codec. Default: "libx264".
        crf: Quality factor. Default: 18.
        pattern: Frame filename pattern. Default: "%06d.png".
    """
    ffmpeg = _get_ffmpeg()
    ffmpeg_exe = _get_ffmpeg_exe()

    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_pattern = str(frames_dir / pattern)

    # Build command
    stream = ffmpeg.input(input_pattern, framerate=fps)

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
