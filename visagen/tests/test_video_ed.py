"""Tests for video editing tools."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from visagen.tools.video_ed import (
    VideoInfo,
    create_video,
    cut_video,
    denoise_sequence,
    extract_frames,
    get_ffmpeg_path,
    get_ffprobe_path,
)


class TestFFmpegPaths:
    """Tests for FFmpeg path detection.

    Note: video_ed.py prioritizes imageio_ffmpeg bundled binary,
    falling back to system PATH if unavailable.
    """

    def test_get_ffmpeg_path_imageio_priority(self) -> None:
        """Test imageio_ffmpeg is preferred when available."""
        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="/opt/imageio/ffmpeg"):
            path = get_ffmpeg_path()
            assert path == Path("/opt/imageio/ffmpeg")

    def test_get_ffmpeg_path_fallback_to_system(self) -> None:
        """Test fallback to system PATH when imageio_ffmpeg returns None."""
        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value=None):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                path = get_ffmpeg_path()
                assert path == Path("/usr/bin/ffmpeg")

    def test_get_ffmpeg_path_not_found(self) -> None:
        """Test RuntimeError when FFmpeg not available anywhere."""
        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value=None):
            with patch("shutil.which", return_value=None):
                with pytest.raises(RuntimeError, match="FFmpeg not found"):
                    get_ffmpeg_path()

    def test_get_ffprobe_path_system_fallback(self) -> None:
        """Test FFprobe detection via system PATH (imageio has no ffprobe)."""
        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value=None):
            with patch("shutil.which", return_value="/usr/bin/ffprobe"):
                path = get_ffprobe_path()
                assert path == Path("/usr/bin/ffprobe")

    def test_get_ffprobe_path_not_found(self) -> None:
        """Test RuntimeError when FFprobe not available."""
        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value=None):
            with patch("shutil.which", return_value=None):
                with pytest.raises(RuntimeError, match="FFprobe not found"):
                    get_ffprobe_path()


class TestVideoInfo:
    """Tests for video info dataclass."""

    def test_video_info_creation(self) -> None:
        """Test VideoInfo dataclass creation."""
        info = VideoInfo(
            width=1920,
            height=1080,
            fps=29.97,
            duration=120.5,
            frame_count=3612,
            codec="h264",
            bitrate="5000k",
        )
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 29.97
        assert info.duration == 120.5
        assert info.frame_count == 3612
        assert info.codec == "h264"
        assert info.bitrate == "5000k"


class TestDenoiseSequence:
    """Tests for temporal denoising."""

    def test_denoise_creates_output(self) -> None:
        """Test that denoise creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create test images
            for i in range(5):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"{i:07d}.png"), img)

            # Run denoise
            result = denoise_sequence(input_dir, output_dir, factor=3, num_workers=1)

            assert result is True
            assert output_dir.exists()
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 5

    def test_denoise_empty_directory(self) -> None:
        """Test denoise with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            with pytest.raises(RuntimeError, match="No image files"):
                denoise_sequence(input_dir)

    def test_denoise_factor_handling(self) -> None:
        """Test that factor is made odd if even."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create test images
            for i in range(3):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(input_dir / f"{i:07d}.png"), img)

            # Even factor should still work
            result = denoise_sequence(input_dir, factor=4, num_workers=1)
            assert result is True


class TestExtractFrames:
    """Tests for frame extraction."""

    def test_extract_missing_video(self) -> None:
        """Test extract with non-existent video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                extract_frames(
                    Path("/nonexistent/video.mp4"),
                    Path(tmpdir) / "output",
                )

    def test_extract_creates_output_dir(self) -> None:
        """Test that extract creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "frames"
            video_path = Path(tmpdir) / "test.mp4"

            # Create a minimal test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 64))
            for _ in range(10):
                frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            if video_path.exists():
                try:
                    _ = extract_frames(video_path, output_dir)
                    assert output_dir.exists()
                except RuntimeError:
                    # FFmpeg might not be available in test environment
                    pytest.skip("FFmpeg not available")


class TestCreateVideo:
    """Tests for video creation."""

    def test_create_missing_input(self) -> None:
        """Test create with non-existent input directory."""
        with pytest.raises(FileNotFoundError):
            create_video(
                Path("/nonexistent/frames"),
                Path("/tmp/output.mp4"),
            )

    def test_create_empty_directory(self) -> None:
        """Test create with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "frames"
            input_dir.mkdir()

            with pytest.raises(RuntimeError, match="No image files"):
                create_video(input_dir, Path(tmpdir) / "output.mp4")


class TestCutVideo:
    """Tests for video cutting."""

    def test_cut_missing_video(self) -> None:
        """Test cut with non-existent video."""
        with pytest.raises(FileNotFoundError):
            cut_video(
                Path("/nonexistent/video.mp4"),
                Path("/tmp/output.mp4"),
                "00:00:00",
                "00:00:10",
            )

    def test_cut_with_audio_track_and_bitrate_builds_expected_command(self) -> None:
        """Re-encode cut should forward audio track and bitrate to ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            output_video = Path(tmpdir) / "output.mp4"
            input_video.write_bytes(b"dummy")

            with (
                patch(
                    "visagen.tools.video_ed.get_ffmpeg_path",
                    return_value=Path("/ffmpeg"),
                ),
                patch("visagen.tools.video_ed.subprocess.run") as run_mock,
            ):
                cut_video(
                    input_video,
                    output_video,
                    "00:00:01",
                    "00:00:03",
                    audio_track=2,
                    codec="libx264",
                    bitrate="14M",
                )

            cmd = run_mock.call_args.args[0]
            assert "-c:v" in cmd
            assert "libx264" in cmd
            assert "-b:v" in cmd
            assert "14M" in cmd
            assert "-map" in cmd
            assert "0:a:2?" in cmd
