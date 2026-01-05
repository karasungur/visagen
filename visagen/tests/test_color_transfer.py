"""Tests for color transfer functions."""

import numpy as np
import pytest

from visagen.postprocess.color_transfer import (
    color_transfer,
    color_transfer_idt,
    color_transfer_mkl,
    color_transfer_sot,
    linear_color_transfer,
    reinhard_color_transfer,
)


class TestReinhardColorTransfer:
    """Tests for Reinhard color transfer (RCT)."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        target = np.random.rand(128, 128, 3).astype(np.float32)
        source = np.random.rand(128, 128, 3).astype(np.float32)
        return target, source

    def test_output_shape(self, random_images):
        """Output shape should match input shape."""
        target, source = random_images
        result = reinhard_color_transfer(target, source)

        assert result.shape == target.shape

    def test_output_dtype(self, random_images):
        """Output should be float32."""
        target, source = random_images
        result = reinhard_color_transfer(target, source)

        assert result.dtype == np.float32

    def test_output_range(self, random_images):
        """Output should be in [0, 1] range."""
        target, source = random_images
        result = reinhard_color_transfer(target, source)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_with_mask(self, random_images):
        """Color transfer should work with masks."""
        target, source = random_images
        mask = np.ones((128, 128, 1), dtype=np.float32)
        mask[:64, :, :] = 0  # Mask top half

        result = reinhard_color_transfer(target, source, target_mask=mask)

        assert result.shape == target.shape

    def test_identical_images(self):
        """Identical source and target should have minimal change."""
        np.random.seed(42)
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = reinhard_color_transfer(img.copy(), img.copy())

        # Result should be close to original
        assert np.allclose(result, img, atol=0.2)


class TestLinearColorTransfer:
    """Tests for linear color transfer (LCT)."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        target = np.random.rand(64, 64, 3).astype(np.float32)
        source = np.random.rand(64, 64, 3).astype(np.float32)
        return target, source

    @pytest.mark.parametrize("mode", ["pca", "chol", "sym"])
    def test_modes(self, random_images, mode):
        """All modes should work."""
        target, source = random_images
        result = linear_color_transfer(target, source, mode=mode)

        assert result.shape == target.shape
        assert result.dtype == np.float32

    def test_output_range(self, random_images):
        """Output should be clipped to [0, 1]."""
        target, source = random_images
        result = linear_color_transfer(target, source)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_invalid_mode(self, random_images):
        """Invalid mode should raise ValueError."""
        target, source = random_images
        with pytest.raises(ValueError):
            linear_color_transfer(target, source, mode="invalid")


class TestSOTColorTransfer:
    """Tests for Sliced Optimal Transfer (SOT)."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        src = np.random.rand(32, 32, 3).astype(np.float32)
        trg = np.random.rand(32, 32, 3).astype(np.float32)
        return src, trg

    def test_output_shape(self, random_images):
        """Output shape should match input shape."""
        src, trg = random_images
        result = color_transfer_sot(src, trg, steps=2, batch_size=2)

        assert result.shape == src.shape

    def test_output_dtype(self, random_images):
        """Output should be float32."""
        src, trg = random_images
        result = color_transfer_sot(src, trg, steps=2, batch_size=2)

        assert result.dtype == np.float32

    def test_output_range(self, random_images):
        """Output should be in [0, 1] range."""
        src, trg = random_images
        result = color_transfer_sot(src, trg, steps=2, batch_size=2)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        src = np.random.rand(32, 32, 3).astype(np.float32)
        trg = np.random.rand(64, 64, 3).astype(np.float32)

        with pytest.raises(ValueError):
            color_transfer_sot(src, trg)

    def test_without_regularization(self, random_images):
        """Should work without regularization."""
        src, trg = random_images
        result = color_transfer_sot(src, trg, steps=2, batch_size=2, reg_sigma_xy=0)

        assert result.shape == src.shape


class TestMKLColorTransfer:
    """Tests for Monge-Kantorovitch Linear transfer."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        target = np.random.rand(32, 32, 3).astype(np.float32)
        source = np.random.rand(32, 32, 3).astype(np.float32)
        return target, source

    def test_output_shape(self, random_images):
        """Output shape should match input shape."""
        target, source = random_images
        result = color_transfer_mkl(target, source)

        assert result.shape == target.shape

    def test_output_range(self, random_images):
        """Output should be in [0, 1] range."""
        target, source = random_images
        result = color_transfer_mkl(target, source)

        assert result.min() >= 0
        assert result.max() <= 1


class TestIDTColorTransfer:
    """Tests for Iterative Distribution Transfer."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        target = np.random.rand(32, 32, 3).astype(np.float32)
        source = np.random.rand(32, 32, 3).astype(np.float32)
        return target, source

    def test_output_shape(self, random_images):
        """Output shape should match input shape."""
        target, source = random_images
        result = color_transfer_idt(target, source, n_rot=2)

        assert result.shape == target.shape

    def test_output_range(self, random_images):
        """Output should be in [0, 1] range."""
        target, source = random_images
        result = color_transfer_idt(target, source, n_rot=2)

        assert result.min() >= 0
        assert result.max() <= 1


class TestUnifiedInterface:
    """Tests for unified color_transfer function."""

    @pytest.fixture
    def random_images(self):
        np.random.seed(42)
        target = np.random.rand(32, 32, 3).astype(np.float32)
        source = np.random.rand(32, 32, 3).astype(np.float32)
        return target, source

    @pytest.mark.parametrize("mode", ["rct", "lct", "sot", "mkl"])
    def test_modes(self, random_images, mode):
        """All modes should work through unified interface."""
        target, source = random_images

        if mode == "sot":
            result = color_transfer(mode, target, source, steps=2, batch_size=2)
        else:
            result = color_transfer(mode, target, source)

        assert result.shape == target.shape

    def test_invalid_mode(self, random_images):
        """Invalid mode should raise ValueError."""
        target, source = random_images
        with pytest.raises(ValueError):
            color_transfer("invalid", target, source)


class TestEdgeCases:
    """Edge case tests for color transfer."""

    def test_grayscale_like_input(self):
        """Should handle nearly grayscale images."""
        np.random.seed(42)
        gray_value = np.random.rand(32, 32, 1).astype(np.float32)
        target = np.repeat(gray_value, 3, axis=-1)
        source = np.random.rand(32, 32, 3).astype(np.float32)

        result = reinhard_color_transfer(target, source)

        assert result.shape == target.shape
        assert not np.isnan(result).any()

    def test_uniform_color(self):
        """Should handle uniform color images."""
        target = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        source = np.ones((32, 32, 3), dtype=np.float32) * 0.7

        result = linear_color_transfer(target, source)

        # Result should be close to source mean
        assert not np.isnan(result).any()

    def test_small_image(self):
        """Should handle small images."""
        np.random.seed(42)
        target = np.random.rand(8, 8, 3).astype(np.float32)
        source = np.random.rand(8, 8, 3).astype(np.float32)

        result = reinhard_color_transfer(target, source)

        assert result.shape == target.shape
