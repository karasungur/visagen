"""
Tests for Visagen pretrain module.

Tests cover:
- PretrainDataset functionality
- PretrainDataModule setup and dataloaders
- PretrainModule training step
- Pretrained weights loading for fine-tuning
- CLI argument parsing
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample BGR image."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_images_dir(temp_dir, sample_image):
    """Create directory with sample images."""
    for i in range(10):
        path = temp_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(path), sample_image)
    return temp_dir


@pytest.fixture
def nested_images_dir(temp_dir, sample_image):
    """Create nested directory structure like FFHQ."""
    for folder in ["00000", "01000", "02000"]:
        folder_path = temp_dir / folder
        folder_path.mkdir()
        for i in range(3):
            path = folder_path / f"{folder}_{i:02d}.jpg"
            cv2.imwrite(str(path), sample_image)
    return temp_dir


# =============================================================================
# PretrainDataset Tests
# =============================================================================


class TestPretrainDataset:
    """Tests for PretrainDataset."""

    def test_create_dataset(self, sample_images_dir):
        """Test dataset creation."""
        from visagen.data.pretrain_datamodule import PretrainDataset

        image_paths = list(sample_images_dir.glob("*.jpg"))
        dataset = PretrainDataset(image_paths, target_size=256)

        assert len(dataset) == 10

    def test_getitem_returns_pair(self, sample_images_dir):
        """Test that __getitem__ returns (image, image) pair."""
        from visagen.data.pretrain_datamodule import PretrainDataset

        image_paths = list(sample_images_dir.glob("*.jpg"))
        dataset = PretrainDataset(image_paths, target_size=256)

        src, target = dataset[0]

        assert isinstance(src, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert src.shape == (3, 256, 256)
        assert target.shape == (3, 256, 256)

    def test_self_reconstruction(self, sample_images_dir):
        """Test that src and target are identical (self-reconstruction)."""
        from visagen.data.pretrain_datamodule import PretrainDataset

        image_paths = list(sample_images_dir.glob("*.jpg"))
        dataset = PretrainDataset(image_paths, target_size=256, transform=None)

        src, target = dataset[0]

        # Without augmentation, src and target should be identical
        assert torch.allclose(src, target)

    def test_different_target_size(self, sample_images_dir):
        """Test dataset with different target size."""
        from visagen.data.pretrain_datamodule import PretrainDataset

        image_paths = list(sample_images_dir.glob("*.jpg"))
        dataset = PretrainDataset(image_paths, target_size=128)

        src, target = dataset[0]

        assert src.shape == (3, 128, 128)
        assert target.shape == (3, 128, 128)

    def test_value_range(self, sample_images_dir):
        """Test that values are in [-1, 1] range."""
        from visagen.data.pretrain_datamodule import PretrainDataset

        image_paths = list(sample_images_dir.glob("*.jpg"))
        dataset = PretrainDataset(image_paths, target_size=256)

        src, _ = dataset[0]

        assert src.min() >= -1.0
        assert src.max() <= 1.0


# =============================================================================
# Image Scanning Tests
# =============================================================================


class TestImageScanning:
    """Tests for image scanning utilities."""

    def test_scan_images_flat(self, sample_images_dir):
        """Test flat directory scanning."""
        from visagen.data.pretrain_datamodule import scan_images_flat

        paths = scan_images_flat(sample_images_dir)

        assert len(paths) == 10
        assert all(p.suffix == ".jpg" for p in paths)

    def test_scan_images_recursive(self, nested_images_dir):
        """Test recursive directory scanning."""
        from visagen.data.pretrain_datamodule import scan_images_recursive

        paths = scan_images_recursive(nested_images_dir)

        # 3 folders × 3 images = 9 images
        assert len(paths) == 9

    def test_scan_images_recursive_on_flat(self, sample_images_dir):
        """Test recursive scanning on flat directory."""
        from visagen.data.pretrain_datamodule import scan_images_recursive

        paths = scan_images_recursive(sample_images_dir)

        assert len(paths) == 10

    def test_scan_empty_directory(self, temp_dir):
        """Test scanning empty directory."""
        from visagen.data.pretrain_datamodule import scan_images_flat

        paths = scan_images_flat(temp_dir)

        assert len(paths) == 0


# =============================================================================
# PretrainDataModule Tests
# =============================================================================


class TestPretrainDataModule:
    """Tests for PretrainDataModule."""

    def test_init(self, sample_images_dir):
        """Test DataModule initialization."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=sample_images_dir,
            batch_size=4,
            num_workers=0,
            target_size=256,
        )

        assert dm.batch_size == 4
        assert dm.target_size == 256

    def test_setup(self, sample_images_dir):
        """Test DataModule setup."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=sample_images_dir,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.num_total_images == 10

    def test_train_dataloader(self, sample_images_dir):
        """Test train dataloader creation."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=sample_images_dir,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        src, target = batch
        assert src.shape[0] == 2  # batch size
        assert src.shape[1:] == (3, 256, 256)

    def test_val_dataloader(self, sample_images_dir):
        """Test validation dataloader creation."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=sample_images_dir,
            batch_size=2,
            num_workers=0,
            val_split=0.3,
        )
        dm.setup()

        loader = dm.val_dataloader()

        assert loader is not None
        assert dm.num_val_samples > 0

    def test_recursive_mode(self, nested_images_dir):
        """Test recursive scanning mode."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=nested_images_dir,
            batch_size=2,
            num_workers=0,
            recursive=True,
        )
        dm.setup()

        assert dm.num_total_images == 9

    def test_non_recursive_mode(self, nested_images_dir):
        """Test non-recursive scanning mode."""
        from visagen.data.pretrain_datamodule import PretrainDataModule

        dm = PretrainDataModule(
            data_dir=nested_images_dir,
            batch_size=2,
            num_workers=0,
            recursive=False,
        )

        # No images in root directory - should raise ValueError
        with pytest.raises(ValueError, match="No images found"):
            dm.setup()


# =============================================================================
# PretrainModule Tests
# =============================================================================


class TestPretrainModule:
    """Tests for PretrainModule."""

    def test_init_defaults(self):
        """Test module initialization with defaults."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule()

        assert module.hparams.image_size == 256
        assert module.learning_rate == 1e-4

    def test_init_custom(self):
        """Test module initialization with custom params."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule(
            image_size=128,
            learning_rate=5e-5,
            dssim_weight=5.0,
        )

        assert module.hparams.image_size == 128
        assert module.learning_rate == 5e-5

    def test_forward(self):
        """Test forward pass."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule(image_size=64)
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            result = module(x)
        out = result[0] if isinstance(result, tuple) else result

        assert out.shape == (2, 3, 64, 64)

    def test_training_step(self):
        """Test training step."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule(image_size=64)
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))

        loss = module.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_validation_step(self):
        """Test validation step."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule(image_size=64)
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))

        loss = module.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        from visagen.training.pretrain_module import PretrainModule

        module = PretrainModule()
        # Mock trainer
        module.trainer = MagicMock()
        module.trainer.max_epochs = 100

        opt_config = module.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config


# =============================================================================
# Pretrained Weights Loading Tests
# =============================================================================


class TestPretrainedLoading:
    """Tests for loading pretrained weights."""

    def test_load_for_finetune(self, temp_dir):
        """Test loading pretrained checkpoint for fine-tuning."""
        from visagen.training.pretrain_module import PretrainModule
        from visagen.training.training_module import TrainingModule

        # Create and save a pretrained module
        pretrain_module = PretrainModule(image_size=64)
        checkpoint_path = temp_dir / "pretrain.ckpt"

        # Save checkpoint manually
        torch.save(
            {
                "state_dict": pretrain_module.state_dict(),
                "hyper_parameters": {
                    "image_size": 64,
                    "encoder_dims": [64, 128, 256, 512],
                    "encoder_depths": [2, 2, 4, 2],
                    "decoder_dims": [512, 256, 128, 64],
                    "learning_rate": 1e-4,
                },
            },
            checkpoint_path,
        )

        # Load for fine-tuning
        model = PretrainModule.load_for_finetune(
            checkpoint_path,
            learning_rate=5e-5,  # Override
        )

        assert isinstance(model, TrainingModule)
        assert model.learning_rate == 5e-5

    def test_extract_weights(self, temp_dir):
        """Test extracting weights from checkpoint."""
        from visagen.training.pretrain_module import PretrainModule

        # Create module and checkpoint
        module = PretrainModule(image_size=64)
        checkpoint_path = temp_dir / "full.ckpt"
        weights_path = temp_dir / "weights.pt"

        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": {"image_size": 64},
                "optimizer_states": [{"param_groups": []}],  # Simulate optimizer
            },
            checkpoint_path,
        )

        # Extract weights
        weights = PretrainModule.extract_weights(checkpoint_path, weights_path)

        assert "state_dict" in weights
        assert "hyper_parameters" in weights
        assert "optimizer_states" not in weights
        assert weights_path.exists()


# =============================================================================
# CLI Tests
# =============================================================================


class TestPretrainCLI:
    """Tests for pretrain CLI."""

    def test_parse_args_minimal(self, sample_images_dir, monkeypatch):
        """Test minimal argument parsing."""
        from visagen.tools.pretrain import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["visagen-pretrain", "--data-dir", str(sample_images_dir)],
        )

        args = parse_args()

        assert args.data_dir == sample_images_dir
        assert args.batch_size == 16
        assert args.max_epochs == 100

    def test_parse_args_full(self, sample_images_dir, temp_dir, monkeypatch):
        """Test full argument parsing."""
        from visagen.tools.pretrain import parse_args

        output_dir = temp_dir / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-pretrain",
                "--data-dir",
                str(sample_images_dir),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "32",
                "--max-epochs",
                "50",
                "--image-size",
                "128",
                "--learning-rate",
                "5e-5",
                "--precision",
                "16-mixed",
                "--devices",
                "2",
                "--enable-warp",
                "-v",
            ],
        )

        args = parse_args()

        assert args.output_dir == output_dir
        assert args.batch_size == 32
        assert args.max_epochs == 50
        assert args.image_size == 128
        assert args.learning_rate == 5e-5
        assert args.precision == "16-mixed"
        assert args.devices == 2
        assert args.enable_warp is True
        assert args.verbose is True

    def test_parse_args_recursive(self, sample_images_dir, monkeypatch):
        """Test recursive argument handling."""
        from visagen.tools.pretrain import parse_args

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-pretrain",
                "--data-dir",
                str(sample_images_dir),
                "--no-recursive",
            ],
        )

        args = parse_args()

        assert args.recursive is False


# =============================================================================
# Train CLI Pretrain-From Tests
# =============================================================================


class TestTrainPretrainFrom:
    """Tests for train CLI --pretrain-from argument."""

    def test_parse_pretrain_from_arg(self, sample_images_dir, temp_dir, monkeypatch):
        """Test --pretrain-from argument parsing."""
        from visagen.tools.train import parse_args

        pretrain_path = temp_dir / "pretrain.ckpt"
        pretrain_path.touch()

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-train",
                "--src-dir",
                str(sample_images_dir),
                "--dst-dir",
                str(sample_images_dir),
                "--pretrain-from",
                str(pretrain_path),
            ],
        )

        args = parse_args()

        assert args.pretrain_from == pretrain_path


# =============================================================================
# Integration Tests
# =============================================================================


class TestPretrainIntegration:
    """Integration tests for pretrain pipeline."""

    def test_full_pretrain_flow(self, sample_images_dir):
        """Test complete pretrain flow without actual training."""
        from visagen.data.pretrain_datamodule import PretrainDataModule
        from visagen.training.pretrain_module import PretrainModule

        # Create datamodule
        dm = PretrainDataModule(
            data_dir=sample_images_dir,
            batch_size=2,
            num_workers=0,
            target_size=64,
        )
        dm.setup()

        # Create model
        model = PretrainModule(image_size=64)

        # Get a batch and run forward
        loader = dm.train_dataloader()
        batch = next(iter(loader))

        with torch.no_grad():
            loss = model.training_step(batch, 0)

        assert loss.item() > 0

    def test_pretrain_augmentation_config(self):
        """Test pretrain-specific augmentation config."""
        from visagen.data.pretrain_datamodule import PRETRAIN_AUGMENTATION_CONFIG

        assert PRETRAIN_AUGMENTATION_CONFIG["random_warp"] is False
        assert PRETRAIN_AUGMENTATION_CONFIG["random_flip_prob"] == 0.5
