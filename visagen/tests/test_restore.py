"""
Tests for Visagen face restoration module (GFPGAN).

Tests cover:
- RestoreConfig dataclass
- GFPGAN availability checking
- FaceRestorer class
- Integration with FrameProcessor
- CLI arguments
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_face():
    """Create a sample face image (256x256 BGR)."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_float():
    """Create a sample face image (256x256 BGR) as float32."""
    return np.random.rand(256, 256, 3).astype(np.float32)


# =============================================================================
# RestoreConfig Tests
# =============================================================================


class TestRestoreConfig:
    """Tests for RestoreConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from visagen.postprocess.restore import RestoreConfig

        config = RestoreConfig()

        assert config.enabled is False
        assert config.mode == "gfpgan"
        assert config.strength == 0.5
        assert config.upscale == 1
        assert config.arch == "clean"
        assert config.model_version == 1.4
        assert config.bg_upsampler is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from visagen.postprocess.restore import RestoreConfig

        config = RestoreConfig(
            enabled=True,
            strength=0.8,
            model_version=1.3,
            upscale=2,
        )

        assert config.enabled is True
        assert config.strength == 0.8
        assert config.model_version == 1.3
        assert config.upscale == 2


# =============================================================================
# GFPGAN Availability Tests
# =============================================================================


class TestGFPGANAvailability:
    """Tests for GFPGAN availability checking."""

    def test_is_gfpgan_available_function_exists(self):
        """Test that is_gfpgan_available function exists."""
        from visagen.postprocess.restore import is_gfpgan_available

        assert callable(is_gfpgan_available)

    def test_is_gfpgan_available_returns_bool(self):
        """Test that is_gfpgan_available returns boolean."""
        from visagen.postprocess.restore import is_gfpgan_available

        result = is_gfpgan_available()
        assert isinstance(result, bool)

    def test_is_gfpgan_available_caching(self):
        """Test that availability check is cached."""
        import visagen.postprocess.restore as restore_module

        # Reset cache
        restore_module._GFPGAN_AVAILABLE = None

        # First call
        result1 = restore_module.is_gfpgan_available()

        # Second call should use cache
        result2 = restore_module.is_gfpgan_available()

        assert result1 == result2


# =============================================================================
# FaceRestorer Tests
# =============================================================================


class TestFaceRestorer:
    """Tests for FaceRestorer class."""

    def test_restorer_init_disabled(self):
        """Test FaceRestorer initialization when disabled."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=False)
        restorer = FaceRestorer(config)

        assert restorer.config == config
        assert restorer.gfpgan is None

    def test_restorer_init_default_config(self):
        """Test FaceRestorer with default config."""
        from visagen.postprocess.restore import FaceRestorer

        restorer = FaceRestorer()

        assert restorer.config.enabled is False
        assert restorer._gfpgan is None
        assert restorer._initialization_attempted is False

    def test_restorer_restore_disabled_returns_original(self, sample_face):
        """Test restore returns original when disabled."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=False)
        restorer = FaceRestorer(config)

        result = restorer.restore(sample_face)

        assert result is sample_face  # Same object

    def test_restorer_restore_zero_strength_returns_original(self, sample_face):
        """Test restore returns original when strength is 0."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, strength=0.0)
        restorer = FaceRestorer(config)

        # Mock GFPGAN loading to return None
        restorer._initialization_attempted = True
        restorer._gfpgan = None

        result = restorer.restore(sample_face, strength=0.0)

        assert np.array_equal(result, sample_face)

    def test_restorer_is_available(self):
        """Test is_available method."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        # Disabled config
        config = RestoreConfig(enabled=False)
        restorer = FaceRestorer(config)
        assert restorer.is_available() is False

        # Enabled config (depends on GFPGAN installation)
        config = RestoreConfig(enabled=True)
        restorer = FaceRestorer(config)
        # Result depends on actual GFPGAN availability
        assert isinstance(restorer.is_available(), bool)

    def test_restorer_model_urls(self):
        """Test model URLs are defined correctly."""
        from visagen.postprocess.restore import FaceRestorer

        assert 1.2 in FaceRestorer.MODEL_URLS
        assert 1.3 in FaceRestorer.MODEL_URLS
        assert 1.4 in FaceRestorer.MODEL_URLS

        for _version, url in FaceRestorer.MODEL_URLS.items():
            assert "github.com" in url
            assert "GFPGAN" in url

    def test_restorer_with_mock_gfpgan(self, sample_face):
        """Test FaceRestorer with mocked GFPGAN."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, strength=1.0)
        restorer = FaceRestorer(config)

        # Create mock GFPGAN
        mock_gfpgan = MagicMock()
        restored_face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mock_gfpgan.enhance.return_value = (None, None, restored_face)

        # Inject mock
        restorer._gfpgan = mock_gfpgan
        restorer._initialization_attempted = True

        result = restorer.restore(sample_face)

        # Should have called enhance
        mock_gfpgan.enhance.assert_called_once()

        # Result should be the restored face
        assert result.shape == sample_face.shape
        assert result.dtype == np.uint8

    def test_restorer_with_float_input(self, sample_face_float):
        """Test FaceRestorer handles float32 input correctly."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, strength=1.0)
        restorer = FaceRestorer(config)

        # Create mock GFPGAN
        mock_gfpgan = MagicMock()
        restored_face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mock_gfpgan.enhance.return_value = (None, None, restored_face)

        # Inject mock
        restorer._gfpgan = mock_gfpgan
        restorer._initialization_attempted = True

        result = restorer.restore(sample_face_float)

        # Result should be float32
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_restorer_strength_blending(self, sample_face):
        """Test strength parameter controls blending."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, strength=0.5)
        restorer = FaceRestorer(config)

        # Create mock with different restored face
        mock_gfpgan = MagicMock()
        restored_face = np.zeros((256, 256, 3), dtype=np.uint8) + 200
        mock_gfpgan.enhance.return_value = (None, None, restored_face)

        restorer._gfpgan = mock_gfpgan
        restorer._initialization_attempted = True

        # Restore with 50% strength
        result = restorer.restore(sample_face, strength=0.5)

        # Result should be a blend (not equal to original or restored)
        assert not np.array_equal(result, sample_face)
        assert not np.array_equal(result, restored_face)


# =============================================================================
# restore_face Function Tests
# =============================================================================


class TestRestoreFaceFunction:
    """Tests for restore_face convenience function."""

    def test_restore_face_exists(self):
        """Test restore_face function exists."""
        from visagen.postprocess.restore import restore_face

        assert callable(restore_face)

    def test_restore_face_creates_restorer(self, sample_face):
        """Test restore_face creates internal FaceRestorer."""
        from visagen.postprocess.restore import restore_face

        with patch("visagen.postprocess.restore.FaceRestorer") as MockRestorer:
            mock_instance = MagicMock()
            mock_instance.restore.return_value = sample_face
            MockRestorer.return_value = mock_instance

            restore_face(sample_face, strength=0.7)

            # Should create FaceRestorer with enabled=True
            MockRestorer.assert_called_once()
            call_kwargs = MockRestorer.call_args
            config = call_kwargs[0][0]  # First positional arg
            assert config.enabled is True
            assert config.strength == 0.7


# =============================================================================
# FrameProcessorConfig Integration Tests
# =============================================================================


class TestFrameProcessorConfigRestore:
    """Tests for restore fields in FrameProcessorConfig."""

    def test_config_has_restore_fields(self):
        """Test FrameProcessorConfig has restore fields."""
        from visagen.merger.frame_processor import FrameProcessorConfig

        config = FrameProcessorConfig()

        assert hasattr(config, "restore_face")
        assert hasattr(config, "restore_strength")
        assert hasattr(config, "restore_model_version")

    def test_config_restore_defaults(self):
        """Test restore field default values."""
        from visagen.merger.frame_processor import FrameProcessorConfig

        config = FrameProcessorConfig()

        assert config.restore_face is False
        assert config.restore_strength == 0.5
        assert config.restore_model_version == 1.4

    def test_config_restore_custom_values(self):
        """Test custom restore values."""
        from visagen.merger.frame_processor import FrameProcessorConfig

        config = FrameProcessorConfig(
            restore_face=True,
            restore_strength=0.8,
            restore_model_version=1.3,
        )

        assert config.restore_face is True
        assert config.restore_strength == 0.8
        assert config.restore_model_version == 1.3


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLIRestoreArgs:
    """Tests for restore CLI arguments."""

    def test_parse_args_restore_default(self, temp_dir, monkeypatch):
        """Test default restore arguments."""
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
            ],
        )

        args = parse_args()

        assert args.restore_face is False
        assert args.restore_strength == 0.5
        assert args.restore_model == 1.4

    def test_parse_args_restore_enabled(self, temp_dir, monkeypatch):
        """Test --restore-face argument."""
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
                "--restore-face",
            ],
        )

        args = parse_args()

        assert args.restore_face is True

    def test_parse_args_restore_strength(self, temp_dir, monkeypatch):
        """Test --restore-strength argument."""
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
                "--restore-face",
                "--restore-strength",
                "0.8",
            ],
        )

        args = parse_args()

        assert args.restore_strength == 0.8

    def test_parse_args_restore_model(self, temp_dir, monkeypatch):
        """Test --restore-model argument."""
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
                "--restore-face",
                "--restore-model",
                "1.3",
            ],
        )

        args = parse_args()

        assert args.restore_model == 1.3

    def test_build_config_with_restore(self, temp_dir, monkeypatch):
        """Test building config with restore options."""
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
                "--restore-face",
                "--restore-strength",
                "0.7",
                "--restore-model",
                "1.2",
            ],
        )

        args = parse_args()
        config = build_config(args)

        assert config.frame_processor_config.restore_face is True
        assert config.frame_processor_config.restore_strength == 0.7
        assert config.frame_processor_config.restore_model_version == 1.2


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_postprocess_exports(self):
        """Test postprocess module exports restore components."""
        from visagen.postprocess import (
            FaceRestorer,
            RestoreConfig,
            is_gfpgan_available,
            restore_face,
        )

        assert FaceRestorer is not None
        assert RestoreConfig is not None
        assert callable(is_gfpgan_available)
        assert callable(restore_face)

    def test_postprocess_all_contains_restore(self):
        """Test __all__ contains restore exports."""
        from visagen.postprocess import __all__

        assert "FaceRestorer" in __all__
        assert "RestoreConfig" in __all__
        assert "restore_face" in __all__
        assert "is_gfpgan_available" in __all__
