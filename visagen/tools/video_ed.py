"""Video editing utilities for frame extraction and video creation.

Provides tools for:
    - Extracting frames from video files
    - Creating videos from image sequences
    - Cutting video segments
    - Temporal denoising of image sequences

Uses FFmpeg via subprocess for reliable video processing.

References:
    - Legacy DFL: mainscripts/VideoEd.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class VideoInfo:
    """Video metadata information."""

    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str
    bitrate: str | None = None


def get_ffmpeg_path() -> Path:
    """Get FFmpeg executable path.

    Tries imageio_ffmpeg bundled binary first, then falls back to system PATH.

    Returns:
        Path to FFmpeg executable

    Raises:
        RuntimeError: If FFmpeg is not found
    """
    # Try imageio_ffmpeg bundled binary first
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg:
            return Path(ffmpeg)
    except ImportError:
        pass

    # Fallback to system PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Install via: pip install imageio-ffmpeg "
            "or install FFmpeg system-wide and ensure it's in PATH."
        )
    return Path(ffmpeg)


def get_ffprobe_path() -> Path:
    """Get FFprobe executable path.

    Tries imageio_ffmpeg bundled binary directory first, then falls back to system PATH.

    Returns:
        Path to FFprobe executable

    Raises:
        RuntimeError: If FFprobe is not found
    """
    # Try to find ffprobe next to imageio_ffmpeg's ffmpeg
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg:
            ffprobe = Path(ffmpeg).parent / "ffprobe"
            if ffprobe.exists():
                return ffprobe
            # Some platforms have ffprobe with same naming pattern
            ffprobe_pattern = Path(ffmpeg).with_name(
                Path(ffmpeg).name.replace("ffmpeg", "ffprobe")
            )
            if ffprobe_pattern.exists():
                return ffprobe_pattern
    except ImportError:
        pass

    # Fallback to system PATH
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(
            "FFprobe not found. Install FFmpeg system-wide for full functionality, "
            "or use imageio-ffmpeg (ffprobe may have limited availability)."
        )
    return Path(ffprobe)


def get_video_info(video_path: Path) -> VideoInfo:
    """Get video metadata using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with video metadata

    Raises:
        RuntimeError: If video cannot be probed
    """
    ffprobe = get_ffprobe_path()

    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,nb_frames,duration,bit_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(",")

        width = int(parts[0])
        height = int(parts[1])

        # Parse frame rate (e.g., "30000/1001" or "30/1")
        fps_parts = parts[2].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])

        codec = parts[3]

        # Duration and frame count may be N/A
        duration = float(parts[4]) if parts[4] != "N/A" else 0.0
        frame_count = int(parts[5]) if parts[5] != "N/A" else int(duration * fps)
        bitrate = parts[6] if len(parts) > 6 and parts[6] != "N/A" else None

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            codec=codec,
            bitrate=bitrate,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to probe video: {e.stderr}") from e
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse video info: {e}") from e


def extract_frames(
    input_video: Path,
    output_dir: Path,
    *,
    fps: float | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    output_format: Literal["png", "jpg"] = "png",
    quality: int = 95,
) -> int:
    """Extract frames from video using FFmpeg.

    Args:
        input_video: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Output frame rate (None = use source fps)
        start_time: Start time in HH:MM:SS.mmm format
        end_time: End time in HH:MM:SS.mmm format
        output_format: Output image format ('png' or 'jpg')
        quality: JPEG quality (1-100, only for jpg format)

    Returns:
        Number of frames extracted

    Raises:
        FileNotFoundError: If input video doesn't exist
        RuntimeError: If extraction fails
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()

    # Build command
    cmd = [str(ffmpeg), "-y"]

    # Input options
    if start_time:
        cmd.extend(["-ss", start_time])
    if end_time:
        cmd.extend(["-to", end_time])

    cmd.extend(["-i", str(input_video)])

    # Output options
    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    # Format-specific options
    if output_format == "jpg":
        cmd.extend(["-q:v", str(int((100 - quality) / 100 * 31))])

    # Output pattern
    output_pattern = output_dir / f"%07d.{output_format}"
    cmd.append(str(output_pattern))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Frame extraction failed: {e.stderr}") from e

    # Count extracted frames
    frames = list(output_dir.glob(f"*.{output_format}"))
    return len(frames)


def create_video(
    input_dir: Path,
    output_video: Path,
    *,
    fps: float = 30.0,
    input_pattern: str | None = None,
    reference_video: Path | None = None,
    include_audio: bool = True,
    codec: Literal["libx264", "libx265", "h264_nvenc", "hevc_nvenc"] = "libx264",
    preset: str = "medium",
    crf: int = 18,
    bitrate: str | None = None,
    pix_fmt: str = "yuv420p",
) -> bool:
    """Create video from image sequence using FFmpeg.

    Args:
        input_dir: Directory containing input frames
        output_video: Output video path
        fps: Output frame rate
        input_pattern: Glob pattern for input files (e.g., "*.png")
        reference_video: Reference video for audio track
        include_audio: Whether to copy audio from reference
        codec: Video codec to use
        preset: Encoding preset (ultrafast to veryslow)
        crf: Constant rate factor (0-51, lower = better quality)
        bitrate: Target bitrate (e.g., "16M") - overrides CRF if set
        pix_fmt: Pixel format

    Returns:
        True if successful

    Raises:
        FileNotFoundError: If input directory doesn't exist
        RuntimeError: If video creation fails
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find input files
    if input_pattern is None:
        # Auto-detect format
        for ext in ["png", "jpg", "jpeg"]:
            files = sorted(input_dir.glob(f"*.{ext}"))
            if files:
                input_pattern = f"*.{ext}"
                break

    if input_pattern is None:
        raise RuntimeError(f"No image files found in {input_dir}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()

    # Build input pattern for ffmpeg
    files = sorted(input_dir.glob(input_pattern))
    if not files:
        raise RuntimeError(f"No files matching {input_pattern} in {input_dir}")

    # Check if files are numbered sequentially
    first_file = files[0]
    ext = first_file.suffix

    # Create input pattern
    ffmpeg_pattern = input_dir / f"%07d{ext}"

    # Build command
    cmd = [
        str(ffmpeg),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(ffmpeg_pattern),
    ]

    # Add audio from reference video
    if reference_video and include_audio and reference_video.exists():
        cmd.extend(["-i", str(reference_video), "-map", "0:v", "-map", "1:a?"])

    # Video codec options
    cmd.extend(["-c:v", codec])

    if "nvenc" in codec:
        # NVENC options
        cmd.extend(["-preset", preset])
        if bitrate:
            cmd.extend(["-b:v", bitrate])
        else:
            cmd.extend(["-cq", str(crf)])
    else:
        # x264/x265 options
        cmd.extend(["-preset", preset])
        if bitrate:
            cmd.extend(["-b:v", bitrate])
        else:
            cmd.extend(["-crf", str(crf)])

    cmd.extend(["-pix_fmt", pix_fmt])

    # Audio codec
    if reference_video and include_audio:
        cmd.extend(["-c:a", "copy"])

    cmd.append(str(output_video))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video creation failed: {e.stderr}") from e


def cut_video(
    input_video: Path,
    output_video: Path,
    start_time: str,
    end_time: str,
    *,
    audio_track: int = 0,
    codec: str = "copy",
    bitrate: str | None = None,
) -> bool:
    """Cut a segment from video.

    Args:
        input_video: Input video path
        output_video: Output video path
        start_time: Start time in HH:MM:SS.mmm format
        end_time: End time in HH:MM:SS.mmm format
        audio_track: Audio track index to use
        codec: Video codec ('copy' for stream copy, or codec name)
        bitrate: Target bitrate if re-encoding

    Returns:
        True if successful

    Raises:
        FileNotFoundError: If input video doesn't exist
        RuntimeError: If cutting fails
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()

    cmd = [
        str(ffmpeg),
        "-y",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        str(input_video),
    ]

    # Video options
    if codec == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", codec])
        if bitrate:
            cmd.extend(["-b:v", bitrate])

    # Audio options
    cmd.extend(["-c:a", "copy", "-map", "0:v", "-map", f"0:a:{audio_track}?"])

    cmd.append(str(output_video))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video cutting failed: {e.stderr}") from e


def denoise_sequence(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    factor: int = 7,
    num_workers: int = 4,
) -> bool:
    """Apply temporal denoising to image sequence.

    Uses a temporal median filter to reduce noise while preserving
    motion. Processes each frame using neighboring frames.

    Args:
        input_dir: Directory containing input frames
        output_dir: Output directory (None = overwrite input)
        factor: Number of neighboring frames to use (odd number)
        num_workers: Number of parallel workers

    Returns:
        True if successful
    """
    if output_dir is None:
        output_dir = input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if not files:
        raise RuntimeError(f"No image files found in {input_dir}")

    # Ensure factor is odd
    factor = factor if factor % 2 == 1 else factor + 1
    half_factor = factor // 2

    def process_frame(idx: int) -> None:
        """Process a single frame with temporal median."""
        # Collect neighboring frames
        frames_to_load = []
        for offset in range(-half_factor, half_factor + 1):
            neighbor_idx = max(0, min(len(files) - 1, idx + offset))
            frames_to_load.append(files[neighbor_idx])

        # Load frames
        frames = [cv2.imread(str(f)) for f in frames_to_load]

        # Apply temporal median
        stacked = np.stack(frames, axis=0)
        result = np.median(stacked, axis=0).astype(np.uint8)

        # Save result
        output_path = output_dir / files[idx].name
        cv2.imwrite(str(output_path), result)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(process_frame, range(len(files))),
                total=len(files),
                desc="Denoising",
            )
        )

    return True


def main() -> None:
    """CLI entry point for video editing tools."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visagen Video Editor - Frame extraction and video creation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from video")
    extract_parser.add_argument("input", type=Path, help="Input video file")
    extract_parser.add_argument("output", type=Path, help="Output directory")
    extract_parser.add_argument(
        "--fps", type=float, default=None, help="Output frame rate"
    )
    extract_parser.add_argument(
        "--start", type=str, default=None, help="Start time (HH:MM:SS.mmm)"
    )
    extract_parser.add_argument(
        "--end", type=str, default=None, help="End time (HH:MM:SS.mmm)"
    )
    extract_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg"],
        default="png",
        help="Output format",
    )

    # Create command
    create_parser = subparsers.add_parser(
        "create", help="Create video from image sequence"
    )
    create_parser.add_argument("input", type=Path, help="Input directory")
    create_parser.add_argument("output", type=Path, help="Output video file")
    create_parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    create_parser.add_argument(
        "--reference", type=Path, default=None, help="Reference video for audio"
    )
    create_parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec",
    )
    create_parser.add_argument(
        "--bitrate", type=str, default=None, help="Target bitrate (e.g., 16M)"
    )
    create_parser.add_argument(
        "--crf", type=int, default=18, help="Constant rate factor (0-51)"
    )

    # Cut command
    cut_parser = subparsers.add_parser("cut", help="Cut video segment")
    cut_parser.add_argument("input", type=Path, help="Input video file")
    cut_parser.add_argument("output", type=Path, help="Output video file")
    cut_parser.add_argument("--start", type=str, required=True, help="Start time")
    cut_parser.add_argument("--end", type=str, required=True, help="End time")
    cut_parser.add_argument(
        "--codec", type=str, default="copy", help="Video codec (copy for stream copy)"
    )

    # Denoise command
    denoise_parser = subparsers.add_parser(
        "denoise", help="Temporal denoise image sequence"
    )
    denoise_parser.add_argument("input", type=Path, help="Input directory")
    denoise_parser.add_argument(
        "--output", type=Path, default=None, help="Output directory"
    )
    denoise_parser.add_argument(
        "--factor", type=int, default=7, help="Temporal filter size"
    )
    denoise_parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Get video information")
    info_parser.add_argument("input", type=Path, help="Input video file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "extract":
        count = extract_frames(
            args.input,
            args.output,
            fps=args.fps,
            start_time=args.start,
            end_time=args.end,
            output_format=args.format,
        )
        print(f"Extracted {count} frames to {args.output}")

    elif args.command == "create":
        create_video(
            args.input,
            args.output,
            fps=args.fps,
            reference_video=args.reference,
            codec=args.codec,
            bitrate=args.bitrate,
            crf=args.crf,
        )
        print(f"Created video: {args.output}")

    elif args.command == "cut":
        cut_video(
            args.input,
            args.output,
            args.start,
            args.end,
            codec=args.codec,
        )
        print(f"Cut video saved to: {args.output}")

    elif args.command == "denoise":
        denoise_sequence(
            args.input,
            args.output,
            factor=args.factor,
            num_workers=args.workers,
        )
        print(f"Denoised sequence saved to: {args.output or args.input}")

    elif args.command == "info":
        info = get_video_info(args.input)
        print(f"Resolution: {info.width}x{info.height}")
        print(f"FPS: {info.fps:.3f}")
        print(f"Duration: {info.duration:.2f}s")
        print(f"Frames: {info.frame_count}")
        print(f"Codec: {info.codec}")
        if info.bitrate:
            print(f"Bitrate: {info.bitrate}")


if __name__ == "__main__":
    main()
