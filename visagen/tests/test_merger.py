"""
Tests for Visagen merger module.

Tests cover:
- Video I/O (VideoReader, VideoWriter, probe_video)
- Frame processing (FrameProcessor, FrameProcessorConfig)
- Merger orchestration (FaceMerger, MergerConfig)
- Batch processing (BatchProcessor)
- CLI (merge.py)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Skip tests if dependencies not available
cv2 = pytest.importorskip("cv2")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames(temp_dir):
    """Create sample frame files."""
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()

    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), frame)

    return frames_dir


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample video file."""
    video_path = temp_dir / "test_video.mp4"

    # Create video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for _ in range(30):  # 1 second of video
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def mock_model():
    """Create a mock DFLModule."""
    model = MagicMock()
    model.eval = Mock()
    model.to = Mock(return_value=model)

    # Mock forward pass
    def forward(x):
        # Return same shape tensor
        return x

    model.__call__ = forward
    model.forward = forward

    return model


# =============================================================================
# VideoInfo Tests
# =============================================================================


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_video_info_creation(self):
        """Test VideoInfo can be created."""
        from visagen.merger.video_io import VideoInfo

        info = VideoInfo(
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=300,
            duration=10.0,
            has_audio=True,
            codec="h264",
            audio_codec="aac",
        )

        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0
        assert info.total_frames == 300
        assert info.has_audio is True


# =============================================================================
# VideoReader Tests
# =============================================================================


class TestVideoReader:
    """Tests for VideoReader class."""

    def test_reader_init(self, sample_video):
        """Test VideoReader initialization."""
        from visagen.merger.video_io import VideoReader

        reader = VideoReader(sample_video)
        assert reader.video_path == sample_video

    @pytest.mark.skipif(
        not Path("/usr/bin/ffmpeg").exists()
        and not Path("/usr/local/bin/ffmpeg").exists(),
        reason="FFmpeg not found",
    )
    def test_reader_get_info(self, sample_video):
        """Test getting video info."""
        from visagen.merger.video_io import VideoReader

        with VideoReader(sample_video) as reader:
            info = reader.get_info()
            assert info.width == 640
            assert info.height == 480
            assert info.fps > 0

    def test_reader_context_manager(self, sample_video):
        """Test VideoReader as context manager."""
        from visagen.merger.video_io import VideoReader

        with VideoReader(sample_video) as reader:
            assert reader is not None


# =============================================================================
# VideoWriter Tests
# =============================================================================


class TestVideoWriter:
    """Tests for VideoWriter class."""

    def test_writer_init(self, temp_dir):
        """Test VideoWriter initialization."""
        from visagen.merger.video_io import VideoWriter

        output_path = temp_dir / "output.mp4"
        writer = VideoWriter(output_path, 640, 480, 30.0)

        assert writer.width == 640
        assert writer.height == 480
        assert writer.fps == 30.0

    def test_writer_context_manager(self, temp_dir, sample_frame):
        """Test VideoWriter as context manager."""
        from visagen.merger.video_io import VideoWriter

        output_path = temp_dir / "output.mp4"

        with VideoWriter(output_path, 640, 480, 30.0) as writer:
            # Write a few frames
            for _ in range(5):
                writer.write_frame(sample_frame)

        # Note: Actual file creation depends on FFmpeg


# =============================================================================
# FrameProcessorConfig Tests
# =============================================================================


class TestFrameProcessorConfig:
    """Tests for FrameProcessorConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from visagen.merger.frame_processor import FrameProcessorConfig

        config = FrameProcessorConfig()

        assert config.min_confidence == 0.5
        assert config.max_faces == 1
        assert config.face_type == "whole_face"
        assert config.output_size == 256
        assert config.color_transfer_mode == "rct"
        assert config.blend_mode == "laplacian"
        assert config.mask_erode == 5

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from visagen.merger.frame_processor import FrameProcessorConfig

        config = FrameProcessorConfig(
            min_confidence=0.8,
            max_faces=3,
            color_transfer_mode="lct",
            blend_mode="feather",
        )

        assert config.min_confidence == 0.8
        assert config.max_faces == 3
        assert config.color_transfer_mode == "lct"
        assert config.blend_mode == "feather"


# =============================================================================
# ProcessedFrame Tests
# =============================================================================


class TestProcessedFrame:
    """Tests for ProcessedFrame dataclass."""

    def test_processed_frame_creation(self, sample_frame):
        """Test ProcessedFrame can be created."""
        from visagen.merger.frame_processor import ProcessedFrame

        result = ProcessedFrame(
            frame_idx=0,
            output_image=sample_frame,
            faces_detected=1,
            faces_swapped=1,
            processing_time=0.05,
        )

        assert result.frame_idx == 0
        assert result.faces_detected == 1
        assert result.faces_swapped == 1
        assert result.processing_time == 0.05


# =============================================================================
# FrameProcessor Tests
# =============================================================================


class TestFrameProcessor:
    """Tests for FrameProcessor class."""

    def test_processor_with_mock_model(self, mock_model, sample_frame):
        """Test FrameProcessor with mock model."""
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        config = FrameProcessorConfig()

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_model"
        ) as load_mock:
            load_mock.return_value = mock_model
            processor = FrameProcessor(mock_model, config=config, device="cpu")

            assert processor.config == config
            assert processor.device == "cpu"

    def test_generate_ellipse_mask(self, mock_model):
        """Test ellipse mask generation fallback."""
        from visagen.merger.frame_processor import FrameProcessor

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_model"
        ) as load_mock:
            load_mock.return_value = mock_model
            processor = FrameProcessor(mock_model, device="cpu")

            # Test mask generation
            face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

            # Force fallback by making segmenter fail
            with patch.object(processor, "_segmenter", None):
                mask = processor._generate_mask(face)

            assert mask.shape == (256, 256)
            assert mask.dtype == np.float32
            assert mask.max() <= 1.0
            assert mask.min() >= 0.0


# =============================================================================
# MergerConfig Tests
# =============================================================================


class TestMergerConfig:
    """Tests for MergerConfig dataclass."""

    def test_config_creation(self, temp_dir):
        """Test MergerConfig creation."""
        from visagen.merger.merger import MergerConfig

        config = MergerConfig(
            input_path=temp_dir / "input.mp4",
            output_path=temp_dir / "output.mp4",
            checkpoint_path=temp_dir / "model.ckpt",
        )

        assert config.input_path == temp_dir / "input.mp4"
        assert config.output_path == temp_dir / "output.mp4"
        assert config.num_workers == 1
        assert config.codec == "libx264"
        assert config.crf == 18
        assert config.copy_audio is True
        assert config.resume is True

    def test_config_to_yaml(self, temp_dir):
        """Test saving config to YAML."""
        from visagen.merger.merger import MergerConfig

        config = MergerConfig(
            input_path=temp_dir / "input.mp4",
            output_path=temp_dir / "output.mp4",
            checkpoint_path=temp_dir / "model.ckpt",
        )

        yaml_path = temp_dir / "config.yaml"
        config.to_yaml(yaml_path)

        assert yaml_path.exists()

    def test_config_from_yaml(self, temp_dir):
        """Test loading config from YAML."""
        from visagen.merger.merger import MergerConfig

        # Create config and save
        config = MergerConfig(
            input_path=temp_dir / "input.mp4",
            output_path=temp_dir / "output.mp4",
            checkpoint_path=temp_dir / "model.ckpt",
            crf=23,
        )

        yaml_path = temp_dir / "config.yaml"
        config.to_yaml(yaml_path)

        # Load and verify
        loaded = MergerConfig.from_yaml(yaml_path)
        assert loaded.crf == 23
        assert loaded.codec == "libx264"


# =============================================================================
# MergerStats Tests
# =============================================================================


class TestMergerStats:
    """Tests for MergerStats dataclass."""

    def test_stats_defaults(self):
        """Test default stats values."""
        from visagen.merger.merger import MergerStats

        stats = MergerStats()

        assert stats.total_frames == 0
        assert stats.processed_frames == 0
        assert stats.faces_detected == 0
        assert stats.fps == 0.0

    def test_stats_update_averages(self):
        """Test stats average calculation."""
        from visagen.merger.merger import MergerStats

        stats = MergerStats(
            total_frames=100,
            processed_frames=100,
            total_time=10.0,
        )
        stats.update_averages()

        assert stats.fps == 10.0
        assert stats.avg_time_per_frame == 0.1


# =============================================================================
# FaceMerger Tests
# =============================================================================


class TestFaceMerger:
    """Tests for FaceMerger class."""

    def test_merger_init(self, temp_dir):
        """Test FaceMerger initialization."""
        from visagen.merger.merger import FaceMerger, MergerConfig

        config = MergerConfig(
            input_path=temp_dir / "input.mp4",
            output_path=temp_dir / "output.mp4",
            checkpoint_path=temp_dir / "model.ckpt",
        )

        merger = FaceMerger(config)
        assert merger.config == config

    def test_merger_resume_state(self, temp_dir):
        """Test resume state save/load."""
        from visagen.merger.merger import FaceMerger, MergerConfig

        config = MergerConfig(
            input_path=temp_dir / "input.mp4",
            output_path=temp_dir / "output.mp4",
            checkpoint_path=temp_dir / "model.ckpt",
        )

        merger = FaceMerger(config)

        # Save state
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        processed = {0, 1, 2, 5, 10}
        merger._save_resume_state(frames_dir, processed)

        # Load state
        loaded = merger._load_resume_state(frames_dir)
        assert loaded == processed


# =============================================================================
# WorkItem/WorkResult Tests
# =============================================================================


class TestBatchProcessorDataClasses:
    """Tests for batch processor data classes."""

    def test_work_item_creation(self, temp_dir):
        """Test WorkItem creation."""
        from visagen.merger.batch_processor import WorkItem

        item = WorkItem(
            frame_idx=5,
            frame_path=temp_dir / "frame.png",
        )

        assert item.frame_idx == 5
        assert item.frame_path == temp_dir / "frame.png"
        assert item.frame_data is None

    def test_work_result_creation(self, temp_dir):
        """Test WorkResult creation."""
        from visagen.merger.batch_processor import WorkResult

        result = WorkResult(
            frame_idx=5,
            success=True,
            output_path=temp_dir / "output.png",
            processing_time=0.1,
            faces_detected=1,
            faces_swapped=1,
        )

        assert result.frame_idx == 5
        assert result.success is True
        assert result.faces_swapped == 1


# =============================================================================
# CLI Tests
# =============================================================================


class TestMergeCLI:
    """Tests for merge CLI."""

    def test_parse_args_minimal(self, temp_dir, monkeypatch):
        """Test parsing minimal arguments."""
        from visagen.tools.merge import parse_args

        # Create dummy files
        input_file = temp_dir / "input.mp4"
        input_file.touch()
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.touch()

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-merge",
                str(input_file),
                str(temp_dir / "output.mp4"),
                "--checkpoint",
                str(checkpoint),
            ],
        )

        args = parse_args()

        assert args.input == input_file
        assert args.checkpoint == checkpoint
        assert args.color_transfer == "rct"
        assert args.blend_mode == "laplacian"

    def test_parse_args_full(self, temp_dir, monkeypatch):
        """Test parsing all arguments."""
        from visagen.tools.merge import parse_args

        input_file = temp_dir / "input.mp4"
        input_file.touch()
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.touch()

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-merge",
                str(input_file),
                str(temp_dir / "output.mp4"),
                "--checkpoint",
                str(checkpoint),
                "--color-transfer",
                "lct",
                "--blend-mode",
                "feather",
                "--mask-erode",
                "10",
                "--crf",
                "23",
                "--workers",
                "4",
                "--resume",
                "--verbose",
            ],
        )

        args = parse_args()

        assert args.color_transfer == "lct"
        assert args.blend_mode == "feather"
        assert args.mask_erode == 10
        assert args.crf == 23
        assert args.workers == 4
        assert args.resume is True
        assert args.verbose is True

    def test_build_config(self, temp_dir, monkeypatch):
        """Test building config from args."""
        from visagen.tools.merge import build_config, parse_args

        input_file = temp_dir / "input.mp4"
        input_file.touch()
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.touch()

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-merge",
                str(input_file),
                str(temp_dir / "output.mp4"),
                "--checkpoint",
                str(checkpoint),
                "--color-transfer",
                "none",
            ],
        )

        args = parse_args()
        config = build_config(args)

        assert config.input_path == input_file
        assert config.frame_processor_config.color_transfer_mode is None


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestIntegration:
    """Integration tests with mocked components."""

    def test_frames_processing_workflow(self, sample_frames, temp_dir, mock_model):
        """Test frame directory processing workflow."""
        from visagen.merger.merger import FaceMerger, MergerConfig

        output_dir = temp_dir / "output"
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.touch()

        config = MergerConfig(
            input_path=sample_frames,
            output_path=output_dir,
            checkpoint_path=checkpoint,
        )

        # Mock the processor
        with patch("visagen.merger.merger.FrameProcessor") as MockProcessor:
            mock_proc_instance = MagicMock()
            mock_proc_instance.process_frame.return_value = MagicMock(
                output_image=np.zeros((480, 640, 3), dtype=np.uint8),
                faces_detected=1,
                faces_swapped=1,
            )
            MockProcessor.return_value = mock_proc_instance

            merger = FaceMerger(config)
            merger._processor = mock_proc_instance

            stats = merger.run_frames(sample_frames, output_dir)

            assert stats.total_frames == 10
            assert stats.processed_frames == 10


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_mask_center(self, mock_model):
        """Test mask center calculation."""
        from visagen.merger.frame_processor import FrameProcessor

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_model"
        ) as load_mock:
            load_mock.return_value = mock_model
            processor = FrameProcessor(mock_model, device="cpu")

            # Create a simple mask
            mask = np.zeros((256, 256), dtype=np.float32)
            mask[100:150, 100:150] = 1.0

            center = processor._get_mask_center(mask)

            assert center[0] == 124  # x center
            assert center[1] == 124  # y center

    def test_process_mask(self, mock_model):
        """Test mask processing with erosion and blur."""
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        config = FrameProcessorConfig(mask_erode=5, mask_blur=5)

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_model"
        ) as load_mock:
            load_mock.return_value = mock_model
            processor = FrameProcessor(mock_model, config=config, device="cpu")

            # Create a mask with some structure
            mask = np.ones((256, 256), dtype=np.uint8) * 255

            processed = processor._process_mask(mask)

            assert processed.shape == (256, 256)
            assert processed.dtype == np.float32
            # After erosion and blur, mask should still have values in [0, 1]
            assert processed.max() <= 1.0
            assert processed.min() >= 0.0
