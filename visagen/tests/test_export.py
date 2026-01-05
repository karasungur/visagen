"""
Tests for Visagen export module.

Tests cover:
- ExportableModel wrapper
- ONNXExporter configuration and export
- ONNXRunner inference
- TensorRT builder and runner (mocked)
- Validation utilities
- CLI argument parsing
- FrameProcessor backend support
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockEncoder(nn.Module):
    """Mock encoder for testing."""

    def __init__(self):
        super().__init__()
        # Need at least one parameter to be a valid module
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        features = [
            torch.randn(batch_size, 64, 64, 64),
            torch.randn(batch_size, 128, 32, 32),
            torch.randn(batch_size, 256, 16, 16),
            torch.randn(batch_size, 512, 8, 8),
        ]
        latent = features[-1]
        return features, latent


class MockDecoder(nn.Module):
    """Mock decoder for testing."""

    def __init__(self):
        super().__init__()
        # Need at least one parameter to be a valid module
        self.dummy = nn.Linear(1, 1)

    def forward(self, latent, skip_features):
        batch_size = latent.shape[0]
        return torch.randn(batch_size, 3, 256, 256)


@pytest.fixture
def mock_encoder():
    """Create a mock encoder."""
    return MockEncoder()


@pytest.fixture
def mock_decoder():
    """Create a mock decoder."""
    return MockDecoder()


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return torch.tanh(self.conv(x))

    return SimpleModel()


# =============================================================================
# ExportableModel Tests
# =============================================================================


class TestExportableModel:
    """Tests for ExportableModel wrapper."""

    def test_model_init(self, mock_encoder, mock_decoder):
        """Test ExportableModel initialization."""
        from visagen.export.model_wrapper import ExportableModel

        model = ExportableModel(mock_encoder, mock_decoder)

        assert model.encoder is mock_encoder
        assert model.decoder is mock_decoder

    def test_model_forward(self, mock_encoder, mock_decoder):
        """Test ExportableModel forward pass."""
        from visagen.export.model_wrapper import ExportableModel

        model = ExportableModel(mock_encoder, mock_decoder)

        # Create input
        x = torch.randn(1, 3, 256, 256)

        # Forward pass
        output = model(x)

        assert output.shape == (1, 3, 256, 256)

    def test_get_input_shape(self, mock_encoder, mock_decoder):
        """Test get_input_shape method."""
        from visagen.export.model_wrapper import ExportableModel

        model = ExportableModel(mock_encoder, mock_decoder)

        assert model.get_input_shape() == (1, 3, 256, 256)

    def test_get_output_shape(self, mock_encoder, mock_decoder):
        """Test get_output_shape method."""
        from visagen.export.model_wrapper import ExportableModel

        model = ExportableModel(mock_encoder, mock_decoder)

        assert model.get_output_shape() == (1, 3, 256, 256)


# =============================================================================
# ExportConfig Tests
# =============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from visagen.export.onnx_exporter import ExportConfig

        config = ExportConfig()

        assert config.opset_version == 17
        assert config.dynamic_axes is True
        assert config.optimize is True
        assert config.input_names == ["input"]
        assert config.output_names == ["output"]

    def test_custom_config(self):
        """Test custom configuration values."""
        from visagen.export.onnx_exporter import ExportConfig

        config = ExportConfig(
            opset_version=14,
            dynamic_axes=False,
            optimize=False,
        )

        assert config.opset_version == 14
        assert config.dynamic_axes is False
        assert config.optimize is False


# =============================================================================
# ONNXExporter Tests
# =============================================================================


class TestONNXExporter:
    """Tests for ONNXExporter class."""

    def test_exporter_init(self, temp_dir):
        """Test ONNXExporter initialization."""
        from visagen.export.onnx_exporter import ONNXExporter

        checkpoint = temp_dir / "model.ckpt"
        output = temp_dir / "model.onnx"

        exporter = ONNXExporter(checkpoint, output)

        assert exporter.checkpoint_path == checkpoint
        assert exporter.output_path == output
        assert exporter.input_shape == (1, 3, 256, 256)

    def test_exporter_with_config(self, temp_dir):
        """Test ONNXExporter with custom config."""
        from visagen.export.onnx_exporter import ExportConfig, ONNXExporter

        config = ExportConfig(opset_version=14, dynamic_axes=False)

        exporter = ONNXExporter(
            temp_dir / "model.ckpt",
            temp_dir / "model.onnx",
            config=config,
        )

        assert exporter.config.opset_version == 14
        assert exporter.config.dynamic_axes is False

    def test_export_simple_model(self, temp_dir, simple_model):
        """Test exporting a simple model to ONNX."""
        from visagen.export.onnx_exporter import ExportConfig, ONNXExporter

        output_path = temp_dir / "simple.onnx"

        # Create exporter with mocked model loading
        with patch(
            "visagen.export.model_wrapper.ExportableModel.from_checkpoint"
        ) as mock_load:
            mock_load.return_value = simple_model

            config = ExportConfig(optimize=False)  # Skip optimization
            exporter = ONNXExporter(
                temp_dir / "dummy.ckpt",
                output_path,
                config=config,
            )

            # Export
            result_path = exporter.export()

            assert result_path == output_path
            assert output_path.exists()


# =============================================================================
# ONNXRunner Tests
# =============================================================================


class TestONNXRunner:
    """Tests for ONNXRunner class."""

    @pytest.fixture
    def onnx_model_path(self, temp_dir, simple_model):
        """Create a simple ONNX model for testing."""
        onnx_path = temp_dir / "test_model.onnx"

        # Export simple model
        dummy_input = torch.randn(1, 3, 64, 64)
        torch.onnx.export(
            simple_model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
        )

        return onnx_path

    def test_runner_init(self, onnx_model_path):
        """Test ONNXRunner initialization."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        runner = ONNXRunner(onnx_model_path, device="cpu")

        assert runner.model_path == onnx_model_path
        assert runner.device == "cpu"
        assert runner.input_name == "input"
        assert runner.output_name == "output"

    def test_runner_inference(self, onnx_model_path):
        """Test ONNXRunner inference."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        runner = ONNXRunner(onnx_model_path, device="cpu")

        # Run inference
        input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)
        output = runner(input_data)

        assert output.shape == (1, 3, 64, 64)
        assert output.dtype == np.float32

    def test_runner_warmup(self, onnx_model_path):
        """Test ONNXRunner warmup."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        runner = ONNXRunner(onnx_model_path, device="cpu")

        # Should not raise
        runner.warmup(shape=(1, 3, 64, 64))

    def test_runner_batch_inference(self, onnx_model_path):
        """Test ONNXRunner batch inference."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        runner = ONNXRunner(onnx_model_path, device="cpu")

        # Create batch of images
        images = [np.random.randn(3, 64, 64).astype(np.float32) for _ in range(5)]

        # Run batch inference
        outputs = runner.batch_inference(images, batch_size=2)

        assert len(outputs) == 5
        for out in outputs:
            assert out.shape == (3, 64, 64)

    def test_runner_get_io_info(self, onnx_model_path):
        """Test ONNXRunner get_io_info."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        runner = ONNXRunner(onnx_model_path, device="cpu")

        info = runner.get_io_info()

        assert "inputs" in info
        assert "outputs" in info
        assert "providers" in info
        assert len(info["inputs"]) == 1
        assert len(info["outputs"]) == 1


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self):
        """Test ValidationResult creation."""
        from visagen.export.validation import ValidationResult

        result = ValidationResult(
            passed=True,
            max_diff=0.001,
            mean_diff=0.0001,
            rtol=1e-3,
            atol=1e-5,
        )

        assert result.passed is True
        assert result.max_diff == 0.001
        assert result.mean_diff == 0.0001

    def test_result_with_details(self):
        """Test ValidationResult with details."""
        from visagen.export.validation import ValidationResult

        result = ValidationResult(
            passed=False,
            max_diff=0.1,
            mean_diff=0.05,
            rtol=1e-3,
            atol=1e-5,
            details="Validation failed due to precision issues",
        )

        assert result.passed is False
        assert result.details is not None


# =============================================================================
# validate_export Tests
# =============================================================================


class TestValidateExport:
    """Tests for validate_export function."""

    @pytest.fixture
    def simple_onnx(self, temp_dir, simple_model):
        """Create simple ONNX model for validation testing."""
        onnx_path = temp_dir / "validate_test.onnx"

        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(
            simple_model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
        )

        return onnx_path, simple_model

    def test_validate_export_passes(self, simple_onnx):
        """Test validation passes for consistent export."""
        pytest.importorskip("onnxruntime")
        from visagen.export.validation import validate_export

        onnx_path, model = simple_onnx

        test_input = torch.randn(1, 3, 32, 32)
        result = validate_export(model, onnx_path, test_input)

        assert result.passed is True
        assert result.max_diff < 1e-3


# =============================================================================
# TensorRT Builder Tests (Mocked)
# =============================================================================


class TestTensorRTBuilder:
    """Tests for TensorRTBuilder class (mocked)."""

    def test_build_config_defaults(self):
        """Test BuildConfig default values."""
        from visagen.export.tensorrt_builder import BuildConfig

        config = BuildConfig()

        assert config.precision == "fp16"
        assert config.max_batch_size == 8
        assert config.workspace_size == 1 << 30

    def test_build_config_custom(self):
        """Test BuildConfig custom values."""
        from visagen.export.tensorrt_builder import BuildConfig

        config = BuildConfig(
            precision="fp32",
            max_batch_size=16,
            workspace_size=2 << 30,
        )

        assert config.precision == "fp32"
        assert config.max_batch_size == 16

    def test_builder_init_file_not_found(self, temp_dir):
        """Test builder raises error for missing ONNX file."""
        pytest.importorskip("tensorrt")
        from visagen.export.tensorrt_builder import TensorRTBuilder

        with pytest.raises(FileNotFoundError):
            TensorRTBuilder(
                temp_dir / "nonexistent.onnx",
                temp_dir / "model.engine",
            )


# =============================================================================
# CLI Tests
# =============================================================================


class TestExportCLI:
    """Tests for export CLI."""

    def test_parse_args_onnx(self, temp_dir, monkeypatch):
        """Test parsing ONNX export arguments."""
        from visagen.tools.export import parse_args

        input_file = temp_dir / "model.ckpt"
        input_file.touch()
        output_file = temp_dir / "model.onnx"

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-export",
                str(input_file),
                "-o",
                str(output_file),
            ],
        )

        args = parse_args()

        assert args.input == input_file
        assert args.output == output_file
        assert args.opset == 17
        assert args.dynamic is False

    def test_parse_args_onnx_with_options(self, temp_dir, monkeypatch):
        """Test parsing ONNX export with all options."""
        from visagen.tools.export import parse_args

        input_file = temp_dir / "model.ckpt"
        input_file.touch()
        output_file = temp_dir / "model.onnx"

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-export",
                str(input_file),
                "-o",
                str(output_file),
                "--format",
                "onnx",
                "--opset",
                "14",
                "--dynamic",
                "--validate",
                "--no-optimize",
            ],
        )

        args = parse_args()

        assert args.format == "onnx"
        assert args.opset == 14
        assert args.dynamic is True
        assert args.validate is True
        assert args.no_optimize is True

    def test_parse_args_tensorrt(self, temp_dir, monkeypatch):
        """Test parsing TensorRT export arguments."""
        from visagen.tools.export import parse_args

        input_file = temp_dir / "model.onnx"
        input_file.touch()
        output_file = temp_dir / "model.engine"

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-export",
                str(input_file),
                "-o",
                str(output_file),
                "--format",
                "tensorrt",
                "--precision",
                "fp16",
                "--max-batch",
                "4",
            ],
        )

        args = parse_args()

        assert args.format == "tensorrt"
        assert args.precision == "fp16"
        assert args.max_batch == 4

    def test_detect_format_from_extension(self, temp_dir, monkeypatch):
        """Test format detection from file extension."""
        from visagen.tools.export import detect_format, parse_args

        input_file = temp_dir / "model.ckpt"
        input_file.touch()

        # ONNX detection
        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-export",
                str(input_file),
                "-o",
                str(temp_dir / "model.onnx"),
            ],
        )
        args = parse_args()
        assert detect_format(args) == "onnx"

        # TensorRT detection
        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-export",
                str(temp_dir / "model.onnx"),
                "-o",
                str(temp_dir / "model.engine"),
            ],
        )
        args = parse_args()
        assert detect_format(args) == "tensorrt"


# =============================================================================
# FrameProcessor Backend Tests
# =============================================================================


class TestFrameProcessorBackend:
    """Tests for FrameProcessor backend support."""

    def test_valid_backends(self):
        """Test valid backend names."""
        from visagen.merger.frame_processor import FrameProcessor

        assert "pytorch" in FrameProcessor.VALID_BACKENDS
        assert "onnx" in FrameProcessor.VALID_BACKENDS
        assert "tensorrt" in FrameProcessor.VALID_BACKENDS

    def test_invalid_backend_raises(self, temp_dir):
        """Test invalid backend raises error."""
        from visagen.merger.frame_processor import FrameProcessor

        with pytest.raises(ValueError, match="Invalid backend"):
            FrameProcessor(
                temp_dir / "model.ckpt",
                backend="invalid_backend",
            )

    def test_pytorch_backend_loads_model(self, temp_dir):
        """Test PyTorch backend with mocked model loading."""
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        config = FrameProcessorConfig()

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_pytorch_model"
        ) as mock_load:
            mock_model = MagicMock()
            mock_model.eval = Mock(return_value=mock_model)
            mock_load.return_value = mock_model

            processor = FrameProcessor(
                temp_dir / "model.ckpt",
                config=config,
                backend="pytorch",
            )

            assert processor.backend == "pytorch"
            mock_load.assert_called_once()

    def test_onnx_backend_loads_model(self, temp_dir):
        """Test ONNX backend with mocked model loading."""
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        config = FrameProcessorConfig()

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_onnx_model"
        ) as mock_load:
            mock_runner = MagicMock()
            mock_load.return_value = mock_runner

            processor = FrameProcessor(
                temp_dir / "model.onnx",
                config=config,
                backend="onnx",
            )

            assert processor.backend == "onnx"
            mock_load.assert_called_once()

    def test_tensorrt_backend_loads_model(self, temp_dir):
        """Test TensorRT backend with mocked model loading."""
        from visagen.merger.frame_processor import FrameProcessor, FrameProcessorConfig

        config = FrameProcessorConfig()

        with patch(
            "visagen.merger.frame_processor.FrameProcessor._load_tensorrt_model"
        ) as mock_load:
            mock_runner = MagicMock()
            mock_load.return_value = mock_runner

            processor = FrameProcessor(
                temp_dir / "model.engine",
                config=config,
                backend="tensorrt",
            )

            assert processor.backend == "tensorrt"
            mock_load.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestExportIntegration:
    """Integration tests for export pipeline."""

    def test_export_and_run_simple_model(self, temp_dir, simple_model):
        """Test full export and inference pipeline."""
        pytest.importorskip("onnxruntime")
        from visagen.export.onnx_runner import ONNXRunner

        # Export model
        onnx_path = temp_dir / "integration_test.onnx"

        dummy_input = torch.randn(1, 3, 64, 64)
        torch.onnx.export(
            simple_model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
        )

        # Run with ONNXRunner
        runner = ONNXRunner(onnx_path, device="cpu")
        runner.warmup(shape=(1, 3, 64, 64))

        # Compare outputs
        test_input = torch.randn(1, 3, 64, 64)

        # PyTorch inference
        simple_model.eval()
        with torch.no_grad():
            pytorch_output = simple_model(test_input).numpy()

        # ONNX inference
        onnx_output = runner(test_input.numpy())

        # Should be close
        assert np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
