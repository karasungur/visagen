"""
Tests for Temporal Discriminator and related components.

Tests cover:
- ResidualBlock3D
- TemporalDiscriminator
- TemporalPatchDiscriminator
- LightweightTemporalDiscriminator
- Temporal loss functions
- Sequence datasets
- DFLModule temporal integration
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visagen.models.discriminators import (
    LightweightTemporalDiscriminator,
    ResidualBlock3D,
    TemporalDiscriminator,
    TemporalPatchDiscriminator,
)
from visagen.training.losses import (
    TemporalConsistencyLoss,
    TemporalDiscriminatorLoss,
    TemporalGANLoss,
)
from visagen.vision.dflimg import DFLImage, FaceMetadata

# =============================================================================
# ResidualBlock3D Tests
# =============================================================================


class TestResidualBlock3D:
    """Tests for ResidualBlock3D."""

    def test_forward_shape(self):
        """Test 3D block preserves shape."""
        block = ResidualBlock3D(64)
        x = torch.randn(2, 64, 5, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Test residual adds correctly."""
        block = ResidualBlock3D(32)
        x = torch.randn(1, 32, 3, 16, 16)

        # Get output
        out = block(x)

        # Output should be different from input (due to convolutions)
        # but similar in magnitude (due to residual)
        assert not torch.allclose(out, x)
        assert out.shape == x.shape

    def test_different_kernel_sizes(self):
        """Test different kernel sizes work."""
        for kernel_size in [1, 3, 5]:
            block = ResidualBlock3D(32, kernel_size=kernel_size)
            x = torch.randn(1, 32, 5, 16, 16)
            out = block(x)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through block."""
        block = ResidualBlock3D(32)
        x = torch.randn(1, 32, 3, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# =============================================================================
# TemporalDiscriminator Tests
# =============================================================================


class TestTemporalDiscriminator:
    """Tests for TemporalDiscriminator."""

    def test_output_shape(self):
        """Test discriminator output shape."""
        D = TemporalDiscriminator(sequence_length=5)
        x = torch.randn(2, 3, 5, 64, 64)
        score = D(x)
        assert score.shape == (2, 1)

    def test_output_shape_larger_image(self):
        """Test with larger images."""
        D = TemporalDiscriminator(sequence_length=5)
        x = torch.randn(2, 3, 5, 128, 128)
        score = D(x)
        assert score.shape == (2, 1)

    def test_spectral_norm(self):
        """Test spectral normalization applies."""
        D = TemporalDiscriminator(use_spectral_norm=True)
        x = torch.randn(1, 3, 5, 64, 64)
        score = D(x)
        assert score.shape == (1, 1)

    def test_variable_sequence_length(self):
        """Test different sequence lengths work."""
        for seq_len in [3, 5, 7]:
            D = TemporalDiscriminator(sequence_length=seq_len)
            x = torch.randn(2, 3, seq_len, 64, 64)
            score = D(x)
            assert score.shape == (2, 1)

    def test_different_base_channels(self):
        """Test different base channel counts."""
        for base_ch in [16, 32, 64]:
            D = TemporalDiscriminator(base_ch=base_ch)
            x = torch.randn(1, 3, 5, 64, 64)
            score = D(x)
            assert score.shape == (1, 1)

    def test_gradient_flow(self):
        """Test gradients flow through discriminator."""
        D = TemporalDiscriminator()
        x = torch.randn(1, 3, 5, 64, 64, requires_grad=True)
        score = D(x)
        score.backward()
        assert x.grad is not None

    def test_output_range(self):
        """Test output values are reasonable (unbounded logits)."""
        D = TemporalDiscriminator()
        x = torch.randn(4, 3, 5, 64, 64)
        score = D(x)
        # Logits should be finite
        assert torch.isfinite(score).all()


# =============================================================================
# TemporalPatchDiscriminator Tests
# =============================================================================


class TestTemporalPatchDiscriminator:
    """Tests for TemporalPatchDiscriminator."""

    def test_dual_output(self):
        """Test both temporal and spatial outputs."""
        D = TemporalPatchDiscriminator()
        x = torch.randn(2, 3, 5, 64, 64)
        temporal, spatial = D(x)

        assert temporal.shape == (2, 1)
        assert spatial.shape[0] == 2
        assert spatial.shape[1] == 5  # T frames

    def test_spatial_output_shape(self):
        """Test spatial output has correct dimensions."""
        D = TemporalPatchDiscriminator()
        x = torch.randn(2, 3, 5, 64, 64)
        _, spatial = D(x)

        # Should be (B, T, 1, H', W') where H', W' are reduced
        assert len(spatial.shape) == 5
        assert spatial.shape[2] == 1  # Single channel output

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        for seq_len in [3, 5, 7]:
            D = TemporalPatchDiscriminator(sequence_length=seq_len)
            x = torch.randn(2, 3, seq_len, 64, 64)
            temporal, spatial = D(x)
            assert temporal.shape == (2, 1)
            assert spatial.shape[1] == seq_len


# =============================================================================
# LightweightTemporalDiscriminator Tests
# =============================================================================


class TestLightweightTemporalDiscriminator:
    """Tests for LightweightTemporalDiscriminator."""

    def test_output_shape(self):
        """Test lightweight discriminator output shape."""
        D = LightweightTemporalDiscriminator()
        x = torch.randn(2, 3, 5, 64, 64)
        score = D(x)
        assert score.shape == (2, 1)

    def test_fewer_parameters(self):
        """Test lightweight has fewer parameters than full."""
        light = LightweightTemporalDiscriminator()
        full = TemporalDiscriminator()

        light_params = sum(p.numel() for p in light.parameters())
        full_params = sum(p.numel() for p in full.parameters())

        assert light_params < full_params


# =============================================================================
# Temporal Loss Tests
# =============================================================================


class TestTemporalConsistencyLoss:
    """Tests for TemporalConsistencyLoss."""

    def test_identical_frames_zero_loss(self):
        """Identical frames should have near-zero loss."""
        loss_fn = TemporalConsistencyLoss()
        frame = torch.randn(2, 3, 64, 64)
        seq = frame.unsqueeze(2).repeat(1, 1, 5, 1, 1)
        loss = loss_fn(seq)
        assert loss < 1e-6

    def test_different_frames_nonzero_loss(self):
        """Different frames should have non-zero loss."""
        loss_fn = TemporalConsistencyLoss()
        seq = torch.randn(2, 3, 5, 64, 64)
        loss = loss_fn(seq)
        assert loss > 0

    def test_l1_mode(self):
        """Test L1 mode works."""
        loss_fn = TemporalConsistencyLoss(mode="l1")
        seq = torch.randn(2, 3, 5, 64, 64)
        loss = loss_fn(seq)
        assert loss.shape == ()
        assert loss >= 0

    def test_l2_mode(self):
        """Test L2 mode works."""
        loss_fn = TemporalConsistencyLoss(mode="l2")
        seq = torch.randn(2, 3, 5, 64, 64)
        loss = loss_fn(seq)
        assert loss.shape == ()
        assert loss >= 0

    def test_weight_parameter(self):
        """Test weight parameter scales loss."""
        seq = torch.randn(2, 3, 5, 64, 64)

        loss_fn_1 = TemporalConsistencyLoss(weight=1.0)
        loss_fn_2 = TemporalConsistencyLoss(weight=2.0)

        loss_1 = loss_fn_1(seq)
        loss_2 = loss_fn_2(seq)

        assert torch.isclose(loss_2, loss_1 * 2, rtol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through loss."""
        loss_fn = TemporalConsistencyLoss()
        seq = torch.randn(2, 3, 5, 64, 64, requires_grad=True)
        loss = loss_fn(seq)
        loss.backward()
        assert seq.grad is not None


class TestTemporalGANLoss:
    """Tests for TemporalGANLoss."""

    def test_vanilla_mode(self):
        """Test vanilla GAN loss mode."""
        loss_fn = TemporalGANLoss(mode="vanilla")
        score = torch.randn(2, 1)
        loss = loss_fn(score, target_is_real=True)
        assert loss.shape == ()

    def test_lsgan_mode(self):
        """Test LSGAN loss mode."""
        loss_fn = TemporalGANLoss(mode="lsgan")
        score = torch.randn(2, 1)
        loss = loss_fn(score, target_is_real=True)
        assert loss.shape == ()

    def test_hinge_mode(self):
        """Test hinge loss mode."""
        loss_fn = TemporalGANLoss(mode="hinge")
        score = torch.randn(2, 1)
        loss = loss_fn(score, target_is_real=True)
        assert loss.shape == ()

    def test_real_vs_fake_target(self):
        """Test real vs fake targets produce different losses."""
        loss_fn = TemporalGANLoss(mode="vanilla")
        score = torch.randn(2, 1)

        loss_real = loss_fn(score, target_is_real=True)
        loss_fake = loss_fn(score, target_is_real=False)

        # Should be different for same input
        assert not torch.isclose(loss_real, loss_fake)

    def test_all_modes_work(self):
        """Test all GAN loss modes work."""
        for mode in ["vanilla", "lsgan", "hinge"]:
            loss_fn = TemporalGANLoss(mode=mode)
            score = torch.randn(2, 1)
            loss = loss_fn(score, target_is_real=True)
            assert loss.shape == ()
            assert torch.isfinite(loss)


class TestTemporalDiscriminatorLoss:
    """Tests for TemporalDiscriminatorLoss."""

    def test_basic_loss(self):
        """Test basic discriminator loss computation."""
        loss_fn = TemporalDiscriminatorLoss()
        d_real = torch.randn(2, 1)
        d_fake = torch.randn(2, 1)
        loss = loss_fn(d_real, d_fake)
        assert loss.shape == ()

    def test_all_modes(self):
        """Test all loss modes work."""
        for mode in ["vanilla", "lsgan", "hinge"]:
            loss_fn = TemporalDiscriminatorLoss(mode=mode)
            d_real = torch.randn(2, 1)
            d_fake = torch.randn(2, 1)
            loss = loss_fn(d_real, d_fake)
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_perfect_discrimination(self):
        """Test loss with perfect discrimination."""
        loss_fn = TemporalDiscriminatorLoss(mode="lsgan")
        d_real = torch.ones(2, 1)  # Perfect real score
        d_fake = torch.zeros(2, 1)  # Perfect fake score
        loss = loss_fn(d_real, d_fake)
        # Loss should be low for perfect discrimination
        assert loss < 0.5


# =============================================================================
# Sequence Dataset Tests
# =============================================================================


class TestSequenceFaceDataset:
    """Tests for SequenceFaceDataset."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 10 dummy images
            for i in range(10):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(Path(tmpdir) / f"{i:05d}.jpg")
            yield tmpdir

    @staticmethod
    def _create_dfl_image(path: Path, source_filename: str) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        metadata = FaceMetadata(
            landmarks=np.zeros((68, 2), dtype=np.float32),
            source_landmarks=np.zeros((68, 2), dtype=np.float32),
            source_rect=(0, 0, 31, 31),
            source_filename=source_filename,
            face_type="whole_face",
            image_to_face_mat=np.eye(2, 3, dtype=np.float32),
        )
        DFLImage.save(path, image, metadata)

    def test_sequence_shape(self, temp_image_dir):
        """Test dataset returns correct sequence shape."""
        from visagen.data import SequenceFaceDataset

        dataset = SequenceFaceDataset(temp_image_dir, sequence_length=5, target_size=64)
        sample = dataset[0]
        assert sample["sequence"].shape == (3, 5, 64, 64)

    def test_sequence_length_parameter(self, temp_image_dir):
        """Test different sequence lengths."""
        from visagen.data import SequenceFaceDataset

        for seq_len in [3, 5]:
            dataset = SequenceFaceDataset(
                temp_image_dir, sequence_length=seq_len, target_size=64
            )
            sample = dataset[0]
            assert sample["sequence"].shape[1] == seq_len

    def test_stride_parameter(self, temp_image_dir):
        """Test stride parameter affects dataset length."""
        from visagen.data import SequenceFaceDataset

        dataset_stride1 = SequenceFaceDataset(
            temp_image_dir, sequence_length=5, stride=1, target_size=64
        )
        dataset_stride2 = SequenceFaceDataset(
            temp_image_dir, sequence_length=5, stride=2, target_size=64
        )

        # Stride 2 should have fewer samples
        assert len(dataset_stride2) < len(dataset_stride1)

    def test_value_range(self, temp_image_dir):
        """Test output values are in [-1, 1] range."""
        from visagen.data import SequenceFaceDataset

        dataset = SequenceFaceDataset(temp_image_dir, sequence_length=5, target_size=64)
        sample = dataset[0]
        seq = sample["sequence"]

        assert seq.min() >= -1.0
        assert seq.max() <= 1.0

    def test_not_enough_images_raises(self):
        """Test error when not enough images for sequence."""
        from visagen.data import SequenceFaceDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only 2 images
            for i in range(2):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(Path(tmpdir) / f"{i:05d}.jpg")

            with pytest.raises(ValueError, match="Not enough images"):
                SequenceFaceDataset(tmpdir, sequence_length=5)

    def test_source_filename_sort_mode_uses_dfl_metadata_order(self):
        """Legacy temporal order should prioritize source_filename metadata."""
        from visagen.data import SequenceFaceDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_dfl_image(root / "1.jpg", "frame_0003.png")
            self._create_dfl_image(root / "2.jpg", "frame_0001.png")
            self._create_dfl_image(root / "3.jpg", "frame_0002.png")

            dataset = SequenceFaceDataset(
                root,
                sequence_length=2,
                target_size=32,
                sort_mode="source_filename",
            )
            assert [p.name for p in dataset.image_paths] == ["2.jpg", "3.jpg", "1.jpg"]

    def test_source_filename_sort_mode_falls_back_to_stem(self):
        """When metadata is absent, source_filename mode should fall back to stem."""
        from visagen.data import SequenceFaceDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ["10.jpg", "2.jpg", "1.jpg"]:
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), np.uint8))
                img.save(root / name)

            dataset = SequenceFaceDataset(
                root,
                sequence_length=2,
                target_size=32,
                sort_mode="source_filename",
            )
            assert [p.name for p in dataset.image_paths] == ["1.jpg", "10.jpg", "2.jpg"]


class TestPairedSequenceDataset:
    """Tests for PairedSequenceDataset."""

    @pytest.fixture
    def temp_paired_dirs(self):
        """Create temporary src and dst directories with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            dst_dir = Path(tmpdir) / "dst"
            src_dir.mkdir()
            dst_dir.mkdir()

            # Create 10 dummy images in each
            for i in range(10):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(src_dir / f"{i:05d}.jpg")
                img.save(dst_dir / f"{i:05d}.jpg")

            yield str(src_dir), str(dst_dir)

    def test_paired_output(self, temp_paired_dirs):
        """Test dataset returns paired sequences."""
        from visagen.data import PairedSequenceDataset

        src_dir, dst_dir = temp_paired_dirs
        dataset = PairedSequenceDataset(
            src_dir, dst_dir, sequence_length=5, target_size=64
        )

        src_seq, dst_seq = dataset[0]
        assert src_seq.shape == (3, 5, 64, 64)
        assert dst_seq.shape == (3, 5, 64, 64)

    def test_length_uses_minimum(self, temp_paired_dirs):
        """Test dataset length uses minimum of src/dst."""
        from visagen.data import PairedSequenceDataset

        src_dir, dst_dir = temp_paired_dirs
        dataset = PairedSequenceDataset(
            src_dir, dst_dir, sequence_length=5, target_size=64
        )

        # Both have same number of images, so lengths should match
        assert len(dataset) > 0


class TestRandomSequenceDataset:
    """Tests for RandomSequenceDataset."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(Path(tmpdir) / f"{i:05d}.jpg")
            yield tmpdir

    def test_num_samples(self, temp_image_dir):
        """Test dataset has correct number of samples."""
        from visagen.data import RandomSequenceDataset

        dataset = RandomSequenceDataset(
            temp_image_dir, sequence_length=5, num_samples=100, target_size=64
        )
        assert len(dataset) == 100

    def test_reproducibility(self, temp_image_dir):
        """Test same index returns same sequence."""
        from visagen.data import RandomSequenceDataset

        dataset = RandomSequenceDataset(
            temp_image_dir, sequence_length=5, num_samples=10, target_size=64
        )

        sample1 = dataset[5]
        sample2 = dataset[5]

        assert torch.allclose(sample1["sequence"], sample2["sequence"])

    def test_random_stride_mode_can_skip_frames(self, temp_image_dir, monkeypatch):
        """Random stride mode should occasionally sample with temporal gaps > 1."""
        from visagen.data import RandomSequenceDataset

        dataset = RandomSequenceDataset(
            temp_image_dir,
            sequence_length=5,
            num_samples=50,
            target_size=64,
            stride_mode="random",
            max_stride=4,
        )

        def _fake_load_frame(path: Path) -> torch.Tensor:
            value = float(int(path.stem))
            return torch.full((3, 4, 4), value)

        monkeypatch.setattr(dataset, "_load_frame", _fake_load_frame)

        saw_gap = False
        for sample_idx in range(20):
            seq = dataset[sample_idx]["sequence"]
            frame_ids = [int(seq[0, t, 0, 0].item()) for t in range(seq.shape[1])]
            gaps = [b - a for a, b in zip(frame_ids[:-1], frame_ids[1:], strict=True)]
            if any(g > 1 for g in gaps):
                saw_gap = True
                break

        assert saw_gap

    def test_fixed_stride_mode_remains_contiguous(self, temp_image_dir, monkeypatch):
        """Fixed stride mode should keep per-frame gaps at exactly 1."""
        from visagen.data import RandomSequenceDataset

        dataset = RandomSequenceDataset(
            temp_image_dir,
            sequence_length=5,
            num_samples=10,
            target_size=64,
            stride_mode="fixed",
            max_stride=4,
        )

        def _fake_load_frame(path: Path) -> torch.Tensor:
            value = float(int(path.stem))
            return torch.full((3, 4, 4), value)

        monkeypatch.setattr(dataset, "_load_frame", _fake_load_frame)

        seq = dataset[0]["sequence"]
        frame_ids = [int(seq[0, t, 0, 0].item()) for t in range(seq.shape[1])]
        gaps = [b - a for a, b in zip(frame_ids[:-1], frame_ids[1:], strict=True)]
        assert all(g == 1 for g in gaps)


# =============================================================================
# DFLModule Temporal Integration Tests
# =============================================================================


class TestDFLModuleTemporalIntegration:
    """Tests for DFLModule temporal integration."""

    def test_temporal_init(self):
        """Test DFLModule initializes with temporal enabled."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True)

        assert module.temporal_enabled is True
        assert module.temporal_discriminator is not None
        assert module.temporal_gan_loss is not None
        assert module.temporal_d_loss_fn is not None
        assert module.temporal_consistency_loss is not None

    def test_temporal_disabled_by_default(self):
        """Test temporal is disabled by default."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule()

        assert module.temporal_enabled is False
        assert module.temporal_discriminator is None

    def test_forward_sequence(self):
        """Test forward_sequence method."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True)
        seq = torch.randn(2, 3, 5, 64, 64)
        out = module.forward_sequence(seq)

        assert out.shape == seq.shape

    def test_forward_sequence_gradient(self):
        """Test gradients flow through forward_sequence."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True)
        seq = torch.randn(1, 3, 5, 64, 64, requires_grad=True)
        out = module.forward_sequence(seq)
        loss = out.mean()
        loss.backward()

        assert seq.grad is not None

    def test_temporal_parameters_saved(self):
        """Test temporal parameters are in hparams."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(
            temporal_enabled=True,
            temporal_power=0.2,
            temporal_sequence_length=7,
            temporal_consistency_weight=2.0,
        )

        assert module.hparams.temporal_enabled is True
        assert module.hparams.temporal_power == 0.2
        assert module.hparams.temporal_sequence_length == 7
        assert module.hparams.temporal_consistency_weight == 2.0

    def test_combined_temporal_and_gan(self):
        """Test temporal and GAN can be combined."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True, gan_power=0.1)

        assert module.temporal_enabled is True
        assert module.gan_power > 0
        assert module.temporal_discriminator is not None
        assert module.discriminator is not None

    def test_configure_optimizers_temporal_only(self):
        """Test optimizer configuration with temporal only."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True)

        # Mock trainer
        class MockTrainer:
            max_epochs = 100

        module.trainer = MockTrainer()

        optimizers, schedulers = module.configure_optimizers()

        assert len(optimizers) == 2  # generator + temporal discriminator
        assert len(schedulers) == 2

    def test_configure_optimizers_temporal_and_gan(self):
        """Test optimizer configuration with temporal + GAN."""
        from visagen.training.dfl_module import DFLModule

        module = DFLModule(temporal_enabled=True, gan_power=0.1)

        class MockTrainer:
            max_epochs = 100

        module.trainer = MockTrainer()

        optimizers, schedulers = module.configure_optimizers()

        assert len(optimizers) == 3  # generator + spatial D + temporal D
        assert len(schedulers) == 3


class TestTemporalDataModule:
    """Tests for temporal datamodule integration contract."""

    @pytest.fixture
    def temp_paired_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            dst_dir = Path(tmpdir) / "dst"
            src_dir.mkdir()
            dst_dir.mkdir()

            for i in range(12):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(src_dir / f"{i:05d}.jpg")
                img.save(dst_dir / f"{i:05d}.jpg")

            yield src_dir, dst_dir

    def test_temporal_datamodule_returns_sequence_dict_contract(self, temp_paired_dirs):
        from visagen.data.datamodule import create_temporal_datamodule

        src_dir, dst_dir = temp_paired_dirs
        dm = create_temporal_datamodule(
            src_dir=src_dir,
            dst_dir=dst_dir,
            batch_size=2,
            num_workers=0,
            target_size=64,
            val_split=0.2,
            sequence_length=4,
        )
        dm.setup("fit")
        src_dict, dst_dict = next(iter(dm.train_dataloader()))

        assert isinstance(src_dict, dict)
        assert isinstance(dst_dict, dict)
        assert src_dict["image"].shape == (2, 3, 4, 64, 64)
        assert dst_dict["image"].shape == (2, 3, 4, 64, 64)
