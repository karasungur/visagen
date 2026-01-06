"""
Tests for GPEN face restoration module.

These tests verify the GPEN restoration functionality
works correctly with various configurations.
"""

import numpy as np
import pytest

from visagen.postprocess.gpen import (
    GPENConfig,
    GPENRestorer,
    is_gpen_available,
    restore_face_gpen,
)


class TestGPENAvailability:
    """Tests for GPEN availability check."""

    def test_is_gpen_available_returns_bool(self):
        """is_gpen_available should return a boolean."""
        result = is_gpen_available()
        assert isinstance(result, bool)

    def test_is_gpen_available_cached(self):
        """Multiple calls should return same result (cached)."""
        result1 = is_gpen_available()
        result2 = is_gpen_available()
        assert result1 == result2


class TestGPENConfig:
    """Tests for GPENConfig dataclass."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = GPENConfig()

        assert config.enabled is False
        assert config.model_size == 512
        assert config.strength == 0.5
        assert config.enhance_background is False
        assert config.use_sr is False

    def test_custom_config(self):
        """Custom config values should be set correctly."""
        config = GPENConfig(
            enabled=True,
            model_size=1024,
            strength=0.8,
            enhance_background=True,
        )

        assert config.enabled is True
        assert config.model_size == 1024
        assert config.strength == 0.8
        assert config.enhance_background is True


class TestGPENRestorer:
    """Tests for GPENRestorer class."""

    @pytest.fixture
    def restorer(self):
        """Create a GPENRestorer with enabled config."""
        config = GPENConfig(enabled=True, strength=0.5)
        return GPENRestorer(config)

    @pytest.fixture
    def disabled_restorer(self):
        """Create a disabled GPENRestorer."""
        config = GPENConfig(enabled=False)
        return GPENRestorer(config)

    @pytest.fixture
    def sample_face_uint8(self):
        """Create sample face image (uint8)."""
        return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_face_float32(self):
        """Create sample face image (float32)."""
        return np.random.rand(128, 128, 3).astype(np.float32)

    def test_disabled_restorer_passthrough(self, disabled_restorer, sample_face_uint8):
        """Disabled restorer should return input unchanged."""
        result = disabled_restorer.restore(sample_face_uint8)

        np.testing.assert_array_equal(result, sample_face_uint8)

    def test_is_available_disabled(self, disabled_restorer):
        """is_available should return False when disabled."""
        assert disabled_restorer.is_available() is False

    def test_restore_zero_strength(self, restorer, sample_face_uint8):
        """Zero strength should return original image."""
        result = restorer.restore(sample_face_uint8, strength=0.0)

        np.testing.assert_array_equal(result, sample_face_uint8)

    def test_restore_preserves_dtype_uint8(self, restorer, sample_face_uint8):
        """Restore should preserve uint8 dtype."""
        result = restorer.restore(sample_face_uint8)

        assert result.dtype == np.uint8

    def test_restore_preserves_dtype_float32(self, restorer, sample_face_float32):
        """Restore should preserve float32 dtype."""
        result = restorer.restore(sample_face_float32)

        assert result.dtype == np.float32

    def test_restore_preserves_shape(self, restorer, sample_face_uint8):
        """Restore should preserve image shape."""
        result = restorer.restore(sample_face_uint8)

        assert result.shape == sample_face_uint8.shape

    def test_restore_output_valid_range_uint8(self, restorer, sample_face_uint8):
        """Restore output should be in valid uint8 range."""
        result = restorer.restore(sample_face_uint8)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_restore_output_valid_range_float32(self, restorer, sample_face_float32):
        """Restore output should be in valid float32 range [0, 1]."""
        result = restorer.restore(sample_face_float32)

        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestRestoreFaceGPEN:
    """Tests for restore_face_gpen convenience function."""

    @pytest.fixture
    def sample_face(self):
        """Create sample face image."""
        return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_returns_same_shape(self, sample_face):
        """Should return image with same shape."""
        result = restore_face_gpen(sample_face)

        assert result.shape == sample_face.shape

    def test_returns_same_dtype(self, sample_face):
        """Should return image with same dtype."""
        result = restore_face_gpen(sample_face)

        assert result.dtype == sample_face.dtype

    def test_strength_parameter(self, sample_face):
        """Strength parameter should be accepted."""
        result = restore_face_gpen(sample_face, strength=0.7)

        assert result.shape == sample_face.shape

    def test_model_size_parameter(self, sample_face):
        """Model size parameter should be accepted."""
        result = restore_face_gpen(sample_face, model_size=256)

        assert result.shape == sample_face.shape

    def test_zero_strength_passthrough(self, sample_face):
        """Zero strength should return original."""
        result = restore_face_gpen(sample_face, strength=0.0)

        np.testing.assert_array_equal(result, sample_face)


class TestFaceRestorerGPENMode:
    """Tests for FaceRestorer with GPEN mode."""

    def test_gpen_mode_config(self):
        """FaceRestorer should accept gpen mode."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, mode="gpen", strength=0.5)
        restorer = FaceRestorer(config)

        assert restorer.config.mode == "gpen"

    def test_gpen_mode_restore(self):
        """FaceRestorer with gpen mode should call GPEN."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(enabled=True, mode="gpen", strength=0.5)
        restorer = FaceRestorer(config)

        face = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = restorer.restore(face)

        assert result.shape == face.shape
        assert result.dtype == face.dtype

    def test_gpen_model_size_config(self):
        """FaceRestorer should accept gpen_model_size."""
        from visagen.postprocess.restore import FaceRestorer, RestoreConfig

        config = RestoreConfig(
            enabled=True,
            mode="gpen",
            gpen_model_size=256,
        )
        restorer = FaceRestorer(config)

        assert restorer.config.gpen_model_size == 256
