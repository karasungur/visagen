"""Tests for blur_out_mask augmentation."""

import pytest
import torch

from visagen.data.augmentations import blur_out_mask

# Check if kornia is available
try:
    import kornia  # noqa: F401

    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False


class TestBlurOutMask:
    """Test blur_out_mask function."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        image = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)
        result = blur_out_mask(image, mask)
        assert result.shape == image.shape

    def test_full_mask_preserves_image(self) -> None:
        """Test that full mask (all face) returns original image."""
        image = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)  # All face
        result = blur_out_mask(image, mask)
        # With full mask, result should be very close to original
        assert torch.allclose(result, image, atol=1e-5)

    def test_zero_mask_blurs_everything(self) -> None:
        """Test that zero mask blurs the entire image."""
        if not HAS_KORNIA:
            pytest.skip("kornia not installed")
        image = torch.randn(2, 3, 256, 256)
        mask = torch.zeros(2, 1, 256, 256)  # All background
        result = blur_out_mask(image, mask)
        # Result should be different from input (blurred)
        assert not torch.allclose(result, image)

    def test_partial_mask(self) -> None:
        """Test with partial mask (half face, half background)."""
        if not HAS_KORNIA:
            pytest.skip("kornia not installed")
        image = torch.randn(2, 3, 256, 256)
        mask = torch.zeros(2, 1, 256, 256)
        mask[:, :, :128, :] = 1.0  # Top half is face
        result = blur_out_mask(image, mask)

        # Top half (face) should be preserved
        top_diff = (result[:, :, :64, :] - image[:, :, :64, :]).abs().mean()
        # Bottom half (background) should be blurred (different)
        bottom_diff = (result[:, :, 192:, :] - image[:, :, 192:, :]).abs().mean()

        # Face region should have smaller difference than background
        assert top_diff < bottom_diff

    def test_different_resolutions(self) -> None:
        """Test with different image resolutions."""
        for res in [128, 192, 256, 512]:
            image = torch.randn(1, 3, res, res)
            mask = torch.ones(1, 1, res, res)
            mask[:, :, res // 2 :, :] = 0  # Bottom half background
            result = blur_out_mask(image, mask)
            assert result.shape == (1, 3, res, res)

    def test_explicit_resolution_parameter(self) -> None:
        """Test with explicit resolution parameter."""
        if not HAS_KORNIA:
            pytest.skip("kornia not installed")
        image = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)
        mask[:, :, 128:, :] = 0

        # Different resolution parameters should produce different results
        result_256 = blur_out_mask(image, mask, resolution=256)
        result_512 = blur_out_mask(image, mask, resolution=512)

        assert result_256.shape == result_512.shape
        # Different sigmas should produce different blur amounts
        assert not torch.allclose(result_256, result_512)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the function."""
        if not HAS_KORNIA:
            pytest.skip("kornia not installed")
        image = torch.randn(1, 3, 128, 128, requires_grad=True)
        mask = torch.ones(1, 1, 128, 128)
        mask[:, :, 64:, :] = 0

        result = blur_out_mask(image, mask)
        loss = result.sum()
        loss.backward()

        assert image.grad is not None
        assert image.grad.abs().sum() > 0

    def test_batch_processing(self) -> None:
        """Test that batch processing works correctly."""
        batch_size = 4
        image = torch.randn(batch_size, 3, 256, 256)
        mask = torch.rand(batch_size, 1, 256, 256) > 0.5
        mask = mask.float()

        result = blur_out_mask(image, mask)
        assert result.shape == (batch_size, 3, 256, 256)

    def test_soft_mask(self) -> None:
        """Test with soft (non-binary) mask values."""
        if not HAS_KORNIA:
            pytest.skip("kornia not installed")
        image = torch.randn(2, 3, 256, 256)
        # Gradient mask from 0 to 1
        mask = torch.linspace(0, 1, 256).view(1, 1, 256, 1).expand(2, 1, 256, 256)
        result = blur_out_mask(image, mask)
        assert result.shape == image.shape
        # Should not have NaN or Inf
        assert torch.isfinite(result).all()

    def test_single_channel_mask(self) -> None:
        """Test that single channel mask works."""
        image = torch.randn(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)
        result = blur_out_mask(image, mask)
        assert result.shape == (2, 3, 256, 256)

    def test_device_compatibility(self) -> None:
        """Test that function works on different devices."""
        image = torch.randn(1, 3, 128, 128)
        mask = torch.ones(1, 1, 128, 128)
        mask[:, :, 64:, :] = 0

        # CPU
        result_cpu = blur_out_mask(image, mask)
        assert result_cpu.device.type == "cpu"

        # GPU if available
        if torch.cuda.is_available():
            image_gpu = image.cuda()
            mask_gpu = mask.cuda()
            result_gpu = blur_out_mask(image_gpu, mask_gpu)
            assert result_gpu.device.type == "cuda"

    def test_numerical_stability(self) -> None:
        """Test numerical stability with edge cases."""
        image = torch.randn(1, 3, 128, 128)

        # Nearly zero mask
        mask_small = torch.full((1, 1, 128, 128), 1e-7)
        result = blur_out_mask(image, mask_small)
        assert torch.isfinite(result).all()

        # Nearly one mask
        mask_large = torch.full((1, 1, 128, 128), 1.0 - 1e-7)
        result = blur_out_mask(image, mask_large)
        assert torch.isfinite(result).all()

    def test_preserves_dtype(self) -> None:
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            image = torch.randn(1, 3, 128, 128, dtype=dtype)
            mask = torch.ones(1, 1, 128, 128, dtype=dtype)
            result = blur_out_mask(image, mask)
            assert result.dtype == dtype


class TestBlurOutMaskIntegration:
    """Integration tests for blur_out_mask with augmentation pipeline."""

    def test_with_face_augmentation_pipeline(self) -> None:
        """Test blur_out_mask can be used alongside FaceAugmentationPipeline."""
        from visagen.data.augmentations import FaceAugmentationPipeline

        pipeline = FaceAugmentationPipeline(target_size=256)
        image = torch.rand(3, 256, 256)
        mask = torch.ones(1, 256, 256)
        mask[:, 128:, :] = 0

        # Apply augmentation first
        aug_image, aug_mask = pipeline(image, mask)

        # Then apply blur_out_mask
        if aug_mask is not None:
            # Add batch dimension
            aug_image_b = aug_image.unsqueeze(0)
            aug_mask_b = aug_mask.unsqueeze(0)
            result = blur_out_mask(aug_image_b, aug_mask_b)
            assert result.shape == (1, 3, 256, 256)
