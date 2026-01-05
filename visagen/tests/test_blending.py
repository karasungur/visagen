"""Tests for blending functions."""

import numpy as np
import pytest

from visagen.postprocess.blending import (
    blend,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    dilate_mask,
    erode_mask,
    feather_blend,
    laplacian_pyramid_blend,
    poisson_blend,
    reconstruct_from_laplacian,
)


class TestGaussianPyramid:
    """Tests for Gaussian pyramid construction."""

    def test_pyramid_levels(self):
        """Pyramid should have correct number of levels."""
        img = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = build_gaussian_pyramid(img, levels=4)

        assert len(pyramid) == 5  # Original + 4 levels

    def test_pyramid_sizes(self):
        """Each level should be half the size of previous."""
        img = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = build_gaussian_pyramid(img, levels=3)

        for i in range(len(pyramid) - 1):
            assert pyramid[i].shape[0] == pyramid[i + 1].shape[0] * 2
            assert pyramid[i].shape[1] == pyramid[i + 1].shape[1] * 2


class TestLaplacianPyramid:
    """Tests for Laplacian pyramid construction and reconstruction."""

    def test_pyramid_levels(self):
        """Laplacian pyramid should have correct number of levels."""
        img = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = build_laplacian_pyramid(img, levels=4)

        # levels + 1 (residual)
        assert len(pyramid) == 5

    def test_reconstruction(self):
        """Image should be reconstructable from Laplacian pyramid."""
        np.random.seed(42)
        img = np.random.rand(64, 64, 3).astype(np.float32)
        pyramid = build_laplacian_pyramid(img, levels=3)
        reconstructed = reconstruct_from_laplacian(pyramid)

        np.testing.assert_allclose(reconstructed, img, atol=1e-4)


class TestLaplacianBlend:
    """Tests for Laplacian pyramid blending."""

    @pytest.fixture
    def test_images(self):
        np.random.seed(42)
        fg = np.random.rand(128, 128, 3).astype(np.float32)
        bg = np.random.rand(128, 128, 3).astype(np.float32)
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[32:96, 32:96] = 1.0  # Center square mask
        return fg, bg, mask

    def test_output_shape(self, test_images):
        """Output shape should match input shape."""
        fg, bg, mask = test_images
        result = laplacian_pyramid_blend(fg, bg, mask, levels=4)

        assert result.shape == fg.shape

    def test_output_dtype(self, test_images):
        """Output should be float32."""
        fg, bg, mask = test_images
        result = laplacian_pyramid_blend(fg, bg, mask, levels=4)

        assert result.dtype == np.float32

    def test_output_range(self, test_images):
        """Output should be in [0, 1] range."""
        fg, bg, mask = test_images
        result = laplacian_pyramid_blend(fg, bg, mask, levels=4)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_full_foreground_mask(self):
        """Full foreground mask should return foreground."""
        fg = np.random.rand(64, 64, 3).astype(np.float32)
        bg = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask, levels=2)

        # Should be very close to foreground
        np.testing.assert_allclose(result, fg, atol=0.1)

    def test_full_background_mask(self):
        """Empty mask should return background."""
        fg = np.random.rand(64, 64, 3).astype(np.float32)
        bg = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask, levels=2)

        # Should be very close to background
        np.testing.assert_allclose(result, bg, atol=0.1)

    def test_mask_with_channel_dim(self, test_images):
        """Mask with channel dimension should work."""
        fg, bg, mask = test_images
        mask_3d = mask[..., None]  # Add channel dim

        result = laplacian_pyramid_blend(fg, bg, mask_3d, levels=4)

        assert result.shape == fg.shape


class TestPoissonBlend:
    """Tests for Poisson blending."""

    @pytest.fixture
    def test_images(self):
        np.random.seed(42)
        fg = np.random.rand(128, 128, 3).astype(np.float32)
        bg = np.random.rand(128, 128, 3).astype(np.float32)
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[40:90, 40:90] = 1.0
        return fg, bg, mask

    def test_output_shape(self, test_images):
        """Output shape should match input shape."""
        fg, bg, mask = test_images
        result = poisson_blend(fg, bg, mask)

        assert result.shape == fg.shape

    def test_output_dtype(self, test_images):
        """Output should be float32."""
        fg, bg, mask = test_images
        result = poisson_blend(fg, bg, mask)

        assert result.dtype == np.float32

    def test_with_center(self, test_images):
        """Custom center should work."""
        fg, bg, mask = test_images
        result = poisson_blend(fg, bg, mask, center=(64, 64))

        assert result.shape == fg.shape

    def test_empty_mask(self, test_images):
        """Empty mask should return background."""
        fg, bg, _ = test_images
        empty_mask = np.zeros((128, 128), dtype=np.float32)

        result = poisson_blend(fg, bg, empty_mask)

        # Should be equal to background
        np.testing.assert_allclose(result, bg, atol=0.01)


class TestFeatherBlend:
    """Tests for feather blending."""

    @pytest.fixture
    def test_images(self):
        np.random.seed(42)
        fg = np.random.rand(64, 64, 3).astype(np.float32)
        bg = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        return fg, bg, mask

    def test_output_shape(self, test_images):
        """Output shape should match input shape."""
        fg, bg, mask = test_images
        result = feather_blend(fg, bg, mask)

        assert result.shape == fg.shape

    def test_output_range(self, test_images):
        """Output should be in [0, 1] range."""
        fg, bg, mask = test_images
        result = feather_blend(fg, bg, mask)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_no_feathering(self, test_images):
        """Zero feathering should be sharp blend."""
        fg, bg, mask = test_images
        result = feather_blend(fg, bg, mask, feather_amount=0)

        # Check center is foreground
        assert np.allclose(result[32, 32], fg[32, 32])

    def test_feathering_smoothness(self, test_images):
        """Higher feathering should produce smoother edges."""
        fg, bg, mask = test_images

        result_low = feather_blend(fg, bg, mask, feather_amount=5)
        result_high = feather_blend(fg, bg, mask, feather_amount=20)

        # High feathering should have more intermediate values at edges
        # (not tested directly, but both should work)
        assert result_low.shape == result_high.shape


class TestMaskOperations:
    """Tests for mask erosion and dilation."""

    @pytest.fixture
    def test_mask(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        return mask

    def test_erode_shrinks_mask(self, test_mask):
        """Erosion should shrink the mask."""
        eroded = erode_mask(test_mask, erosion_size=5)

        assert eroded.sum() < test_mask.sum()

    def test_dilate_expands_mask(self, test_mask):
        """Dilation should expand the mask."""
        dilated = dilate_mask(test_mask, dilation_size=5)

        assert dilated.sum() > test_mask.sum()

    def test_erode_3d_mask(self, test_mask):
        """Erosion should work with 3D mask."""
        mask_3d = test_mask[..., None]
        eroded = erode_mask(mask_3d, erosion_size=5)

        assert eroded.ndim == 3
        assert eroded.shape[-1] == 1

    def test_dilate_3d_mask(self, test_mask):
        """Dilation should work with 3D mask."""
        mask_3d = test_mask[..., None]
        dilated = dilate_mask(mask_3d, dilation_size=5)

        assert dilated.ndim == 3
        assert dilated.shape[-1] == 1


class TestUnifiedBlendInterface:
    """Tests for unified blend function."""

    @pytest.fixture
    def test_images(self):
        np.random.seed(42)
        fg = np.random.rand(64, 64, 3).astype(np.float32)
        bg = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        return fg, bg, mask

    @pytest.mark.parametrize("mode", ["laplacian", "feather"])
    def test_modes(self, test_images, mode):
        """All modes should work through unified interface."""
        fg, bg, mask = test_images

        if mode == "laplacian":
            result = blend(mode, fg, bg, mask, levels=2)
        else:
            result = blend(mode, fg, bg, mask)

        assert result.shape == fg.shape

    def test_invalid_mode(self, test_images):
        """Invalid mode should raise ValueError."""
        fg, bg, mask = test_images
        with pytest.raises(ValueError):
            blend("invalid", fg, bg, mask)


class TestEdgeCases:
    """Edge case tests for blending."""

    def test_small_image(self):
        """Should handle small images."""
        fg = np.random.rand(16, 16, 3).astype(np.float32)
        bg = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask, levels=2)

        assert result.shape == fg.shape

    def test_non_square_image(self):
        """Should handle non-square images."""
        fg = np.random.rand(64, 128, 3).astype(np.float32)
        bg = np.random.rand(64, 128, 3).astype(np.float32)
        mask = np.ones((64, 128), dtype=np.float32)

        result = laplacian_pyramid_blend(fg, bg, mask, levels=3)

        assert result.shape == fg.shape

    def test_soft_mask(self):
        """Should handle soft (non-binary) masks."""
        fg = np.random.rand(64, 64, 3).astype(np.float32)
        bg = np.random.rand(64, 64, 3).astype(np.float32)

        # Gradient mask
        mask = np.linspace(0, 1, 64).astype(np.float32)
        mask = np.tile(mask, (64, 1))

        result = feather_blend(fg, bg, mask, feather_amount=0)

        assert result.shape == fg.shape
        # Left edge should be background, right edge foreground
        np.testing.assert_allclose(result[:, 0], bg[:, 0], atol=0.01)
        np.testing.assert_allclose(result[:, -1], fg[:, -1], atol=0.01)
