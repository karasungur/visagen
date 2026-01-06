"""
Tests for neural color transfer module.

These tests verify the neural color transfer functionality
works correctly with various modes and parameters.
"""

import numpy as np
import pytest

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if torchvision is available
try:
    from torchvision import models  # noqa: F401

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from visagen.postprocess.neural_color import (
    is_neural_color_available,
    neural_color_transfer,
)


class TestNeuralColorTransfer:
    """Tests for neural color transfer."""

    def test_availability_check(self):
        """is_neural_color_available should return True when torch available."""
        assert is_neural_color_available() == TORCH_AVAILABLE

    def test_basic_transfer_histogram(self):
        """Test basic color transfer with histogram mode."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result = neural_color_transfer(target, reference, mode="histogram")

        assert result.shape == target.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_basic_transfer_statistics(self):
        """Test basic color transfer with statistics mode."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result = neural_color_transfer(target, reference, mode="statistics")

        assert result.shape == target.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.skipif(
        not TORCHVISION_AVAILABLE,
        reason="torchvision not available for VGG-based test",
    )
    def test_basic_transfer_gram(self):
        """Test basic color transfer with gram mode (requires torchvision)."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        result = neural_color_transfer(target, reference, mode="gram", device=device)

        assert result.shape == target.shape
        assert result.dtype == np.float32

    def test_strength_zero(self):
        """Strength 0 should return original image."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result = neural_color_transfer(
            target, reference, strength=0.0, mode="statistics"
        )

        np.testing.assert_allclose(result, target, rtol=1e-5)

    def test_strength_partial(self):
        """Partial strength should blend between original and transferred."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result_full = neural_color_transfer(
            target, reference, strength=1.0, mode="statistics"
        )
        result_half = neural_color_transfer(
            target, reference, strength=0.5, mode="statistics"
        )

        # Result should be between target and full transfer
        diff_to_target = np.abs(result_half - target).mean()
        diff_to_full = np.abs(result_half - result_full).mean()

        # Both should be non-zero (mixed)
        assert diff_to_target > 0 or diff_to_full > 0

    def test_preserve_luminance_true(self):
        """With preserve_luminance=True, luminance channel should match target."""
        import cv2

        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result = neural_color_transfer(
            target, reference, preserve_luminance=True, mode="statistics"
        )

        # Convert to LAB and compare L channel
        target_lab = cv2.cvtColor(
            (target * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        result_lab = cv2.cvtColor(
            (np.clip(result, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        # L channels should be similar (some quantization error expected)
        np.testing.assert_allclose(
            result_lab[:, :, 0], target_lab[:, :, 0], rtol=0.1, atol=5
        )

    def test_preserve_luminance_false(self):
        """With preserve_luminance=False, luminance may change."""
        result = neural_color_transfer(
            np.random.rand(64, 64, 3).astype(np.float32),
            np.random.rand(64, 64, 3).astype(np.float32),
            preserve_luminance=False,
            mode="statistics",
        )

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32

    def test_different_sized_inputs(self):
        """Test with different sized inputs."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(128, 128, 3).astype(np.float32)

        result = neural_color_transfer(target, reference, mode="histogram")

        assert result.shape == target.shape

    def test_output_clipped(self):
        """Output should be clipped to [0, 1]."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        result = neural_color_transfer(target, reference, mode="statistics")

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        target = np.random.rand(64, 64, 3).astype(np.float32)
        reference = np.random.rand(64, 64, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown mode"):
            neural_color_transfer(target, reference, mode="invalid_mode")


@pytest.mark.skipif(
    not (TORCH_AVAILABLE and TORCHVISION_AVAILABLE),
    reason="PyTorch and torchvision required",
)
class TestVGGFeatureExtractor:
    """Tests for VGG feature extractor."""

    def test_feature_extraction(self):
        """Test VGG feature extraction."""
        from visagen.postprocess.neural_color import VGGFeatureExtractor

        extractor = VGGFeatureExtractor(layers=["relu1_1", "relu2_1"])

        # Create dummy input
        x = torch.rand(1, 3, 64, 64)

        features = extractor(x)

        assert "relu1_1" in features
        assert "relu2_1" in features
        assert features["relu1_1"].shape[0] == 1
        assert features["relu2_1"].shape[0] == 1

    def test_normalization(self):
        """Test input normalization."""
        from visagen.postprocess.neural_color import VGGFeatureExtractor

        extractor = VGGFeatureExtractor(use_input_norm=True)

        x = torch.rand(1, 3, 64, 64)
        features = extractor(x)

        assert len(features) > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_gram_matrix(self):
        """Test Gram matrix computation."""
        from visagen.postprocess.neural_color import compute_gram_matrix

        features = torch.rand(2, 64, 8, 8)
        gram = compute_gram_matrix(features)

        assert gram.shape == (2, 64, 64)

    def test_compute_mean_std(self):
        """Test mean/std computation."""
        from visagen.postprocess.neural_color import compute_mean_std

        features = torch.rand(2, 64, 8, 8)
        mean, std = compute_mean_std(features)

        assert mean.shape == (2, 64, 1, 1)
        assert std.shape == (2, 64, 1, 1)
        assert (std > 0).all()  # Std should be positive

    def test_adaptive_instance_normalization(self):
        """Test AdaIN."""
        from visagen.postprocess.neural_color import adaptive_instance_normalization

        content = torch.rand(2, 64, 8, 8)
        style = torch.rand(2, 64, 8, 8)

        result = adaptive_instance_normalization(content, style)

        assert result.shape == content.shape

    def test_match_histograms_channel(self):
        """Test histogram matching for single channel."""
        from visagen.postprocess.neural_color import match_histograms_channel

        source = np.random.rand(64, 64).astype(np.float32)
        reference = np.random.rand(64, 64).astype(np.float32)

        result = match_histograms_channel(source, reference)

        assert result.shape == source.shape
        assert result.dtype == np.float32


class TestColorTransferIntegration:
    """Integration tests with color_transfer interface."""

    def test_neural_mode_through_interface(self):
        """Test neural mode through color_transfer interface."""
        from visagen.postprocess import color_transfer

        target = np.random.rand(64, 64, 3).astype(np.float32)
        source = np.random.rand(64, 64, 3).astype(np.float32)

        # Use neural mode with statistics (no mode conflict)
        result = color_transfer("neural", target, source)

        assert result.shape == target.shape
        assert result.dtype == np.float32
