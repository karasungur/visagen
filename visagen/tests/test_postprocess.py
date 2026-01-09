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


class TestColorTransfer:
    @pytest.fixture
    def sample_images(self):
        # Create synthetic images
        # Target: Blue-ish
        target = np.zeros((64, 64, 3), dtype=np.float32)
        target[..., 0] = 0.8  # Blue
        target[..., 1] = 0.2
        target[..., 2] = 0.2

        # Source: Red-ish
        source = np.zeros((64, 64, 3), dtype=np.float32)
        source[..., 0] = 0.2
        source[..., 1] = 0.2
        source[..., 2] = 0.8  # Red

        return target, source

    @pytest.fixture
    def masks(self):
        # Left half mask
        mask = np.zeros((64, 64, 1), dtype=np.float32)
        mask[:, :32] = 1.0
        return mask

    def test_reinhard_color_transfer(self, sample_images):
        target, source = sample_images
        result = reinhard_color_transfer(target, source)

        assert result.shape == target.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Result should move towards source color (more red)
        # Check mean color values
        assert result[..., 2].mean() > target[..., 2].mean()

    def test_linear_color_transfer_modes(self, sample_images):
        target, source = sample_images

        for mode in ["pca", "chol", "sym"]:
            result = linear_color_transfer(target, source, mode=mode)
            assert result.shape == target.shape
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_sot_transfer(self, sample_images):
        target, source = sample_images
        result = color_transfer_sot(
            source, target, steps=5
        )  # Note: SOT modifies src to match trg

        assert result.shape == source.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mkl_transfer(self, sample_images):
        target, source = sample_images
        result = color_transfer_mkl(target, source)

        assert result.shape == target.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_idt_transfer(self, sample_images):
        target, source = sample_images
        result = color_transfer_idt(target, source, n_rot=5)

        assert result.shape == target.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_masked_rct(self, sample_images, masks):
        target, source = sample_images
        mask = masks

        # Masked transfer
        result = reinhard_color_transfer(
            target, source, target_mask=mask, source_mask=mask
        )

        assert result.shape == target.shape
        # Just check it runs without error and produces valid output
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_masked_sot(self, sample_images, masks):
        target, source = sample_images
        mask = masks

        # In SOT, src is modified to match trg
        # If masked, only masked pixels in src should change ideally,
        # or at least the logic handles partial stats.
        # The implementation modifies 'new_src' using 'flat_src' and puts it back.

        # Let's ensure non-masked area is preserved if we only mask source
        source_masked_only = np.copy(source)
        # Make source right half different to verify it doesn't change
        source_masked_only[:, 32:] = 0.5

        result = color_transfer_sot(
            source_masked_only,
            target,
            source_mask=mask,
            steps=2,
            reg_sigma_xy=0,  # Disable regularization to strictly check pixel values
        )

        # Check that unmasked area (right half) is unchanged
        # mask is 1 on left half (0..31), 0 on right (32..63)
        assert np.allclose(result[:, 32:], source_masked_only[:, 32:])

        # Check that masked area changed
        assert not np.allclose(result[:, :32], source_masked_only[:, :32])

    def test_masked_mkl(self, sample_images, masks):
        target, source = sample_images
        mask = masks

        # MKL computes transform from masked pixels but applies to ALL pixels
        # in the current implementation.
        result = color_transfer_mkl(target, source, target_mask=mask, source_mask=mask)

        assert result.shape == target.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_unified_interface(self, sample_images, masks):
        target, source = sample_images
        mask = masks

        # Test standard mode
        res1 = color_transfer("rct", target, source)
        assert res1.shape == target.shape

        # Test masked mode string
        # SOT-masked implies we should pass masks
        # The function signature expects kwargs for masks
        res2 = color_transfer(
            "sot-masked",
            source,
            target,
            source_mask=mask,
            target_mask=mask,
            steps=2,
            reg_sigma_xy=0,
        )
        assert res2.shape == source.shape

        # Verify it called the masked logic (unchanged unmasked area for SOT)
        # SOT modifies first arg (src) to match second (trg)
        assert np.allclose(res2[:, 32:], source[:, 32:])

    def test_invalid_inputs(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown color transfer mode"):
            color_transfer("invalid_mode", img, img)
