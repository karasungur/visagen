"""Tests for Gradio web interface."""

from unittest.mock import Mock

import numpy as np
import pytest

# Skip all tests if gradio not installed
gr = pytest.importorskip("gradio")

# Import directly to avoid visagen.tools.__init__ importing extract_v2
from visagen.tools.gradio_app import GRADIO_AVAILABLE, GradioApp, create_app


class TestGradioApp:
    """Tests for GradioApp class."""

    def test_init(self):
        """Test app initializes correctly."""
        app = GradioApp()

        assert app.model is None
        assert app.model_path is None
        assert app.training_process is None
        assert "device" in app.settings
        assert "default_batch_size" in app.settings

    def test_load_model_invalid_path(self):
        """Test error handling for invalid checkpoint."""
        app = GradioApp()
        result = app.load_model("/nonexistent/path.ckpt")

        assert "Error" in result

    def test_load_model_empty_path(self):
        """Test error handling for empty path."""
        app = GradioApp()
        result = app.load_model("")

        assert "Error" in result

    def test_unload_model_no_model(self):
        """Test unload when no model loaded."""
        app = GradioApp()
        result = app.unload_model()

        assert result == "No model loaded"

    def test_unload_model_with_model(self):
        """Test unload with model loaded."""
        app = GradioApp()
        app.model = Mock()
        app.model_path = "/some/path.ckpt"

        result = app.unload_model()

        assert result == "Model unloaded"
        assert app.model is None
        assert app.model_path is None

    def test_stop_training_no_process(self):
        """Test stop when no training in progress."""
        app = GradioApp()
        result = app.stop_training()

        assert result == "No training in progress."

    def test_stop_training_with_process(self):
        """Test stop with training in progress."""
        app = GradioApp()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        app.training_process = mock_process

        result = app.stop_training()

        assert result == "Training stopped."
        assert app.training_process is None
        mock_process.terminate.assert_called_once()

    def test_save_settings(self):
        """Test settings save."""
        app = GradioApp()

        result = app.save_settings(
            device="cuda",
            gpu_id=1,
            default_batch_size=16,
            num_workers=8,
            workspace_dir="./my_workspace",
        )

        assert result == "Settings saved."
        assert app.settings["device"] == "cuda"
        assert app.settings["gpu_id"] == 1
        assert app.settings["default_batch_size"] == 16
        assert app.settings["num_workers"] == 8
        assert app.settings["workspace_dir"] == "./my_workspace"


class TestGradioAppColorTransfer:
    """Tests for color transfer in GradioApp."""

    def test_color_transfer_no_images(self):
        """Test error when images not provided."""
        app = GradioApp()

        with pytest.raises(gr.Error, match="provide both"):
            app.apply_color_transfer(None, None, "rct")

    def test_color_transfer_rct(self):
        """Test RCT color transfer."""
        app = GradioApp()

        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = app.apply_color_transfer(source, target, "rct")

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_color_transfer_lct(self):
        """Test LCT color transfer."""
        app = GradioApp()

        source = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result = app.apply_color_transfer(source, target, "lct")

        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8


class TestGradioAppBlending:
    """Tests for blending in GradioApp."""

    def test_blend_no_images(self):
        """Test error when images not provided."""
        app = GradioApp()

        with pytest.raises(gr.Error, match="provide foreground"):
            app.apply_blend(None, None, None, "feather")

    def test_blend_feather(self):
        """Test feather blending."""
        app = GradioApp()

        fg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 255

        result = app.apply_blend(fg, bg, mask, "feather")

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_blend_laplacian(self):
        """Test laplacian blending."""
        app = GradioApp()

        fg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.uint8) * 255

        result = app.apply_blend(fg, bg, mask, "laplacian")

        assert result.shape == (128, 128, 3)

    def test_blend_rgb_mask(self):
        """Test blending with RGB mask (should convert to grayscale)."""
        app = GradioApp()

        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64, 3), dtype=np.uint8) * 128  # RGB mask

        result = app.apply_blend(fg, bg, mask, "feather")

        assert result.shape == (64, 64, 3)


class TestCreateApp:
    """Tests for app creation."""

    def test_create_app(self):
        """Test app is created successfully."""
        app = create_app()

        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_gradio_available(self):
        """Test GRADIO_AVAILABLE flag."""
        assert GRADIO_AVAILABLE is True


class TestSwapFace:
    """Tests for face swap inference."""

    def test_swap_face_no_model(self):
        """Test error when no model loaded."""
        app = GradioApp()
        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with pytest.raises(gr.Error, match="No model loaded"):
            app.swap_face(source, None)

    def test_swap_face_no_source(self):
        """Test error when no source provided."""
        app = GradioApp()
        app.model = Mock()  # Fake model

        with pytest.raises(gr.Error, match="provide a source"):
            app.swap_face(None, None)


class TestGradioAppMerge:
    """Tests for merge functionality in GradioApp."""

    def test_init_has_merge_process(self):
        """Test app has merge_process attribute."""
        app = GradioApp()

        assert hasattr(app, "merge_process")
        assert app.merge_process is None

    def test_stop_merge_no_process(self):
        """Test stop when no merge in progress."""
        app = GradioApp()
        result = app.stop_merge()

        assert result == "No merge in progress."

    def test_stop_merge_with_process(self):
        """Test stop with merge in progress."""
        app = GradioApp()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        app.merge_process = mock_process

        result = app.stop_merge()

        assert result == "Merge stopped."
        assert app.merge_process is None
        mock_process.terminate.assert_called_once()

    def test_run_merge_invalid_input(self):
        """Test merge with invalid input path."""
        app = GradioApp()

        gen = app.run_merge(
            "/nonexistent/video.mp4",
            "./output.mp4",
            "./model.ckpt",
            "rct",
            "laplacian",
            False,
            0.5,
            1.4,
            "auto",
            18,
        )

        result = list(gen)
        assert any("Error" in line for line in result)


class TestGradioAppSort:
    """Tests for sort functionality in GradioApp."""

    def test_init_has_sort_process(self):
        """Test app has sort_process attribute."""
        app = GradioApp()

        assert hasattr(app, "sort_process")
        assert app.sort_process is None

    def test_stop_sort_no_process(self):
        """Test stop when no sort in progress."""
        app = GradioApp()
        result = app.stop_sort()

        assert result == "No sorting in progress."

    def test_stop_sort_with_process(self):
        """Test stop with sort in progress."""
        app = GradioApp()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        app.sort_process = mock_process

        result = app.stop_sort()

        assert result == "Sorting stopped."
        assert app.sort_process is None
        mock_process.terminate.assert_called_once()

    def test_run_sort_invalid_input(self):
        """Test sort with invalid input path."""
        app = GradioApp()

        gen = app.run_sort(
            "/nonexistent/dir",
            "",
            "blur",
            2000,
            True,
        )

        result = list(gen)
        assert any("Error" in line for line in result)


class TestGradioAppExport:
    """Tests for export functionality in GradioApp."""

    def test_init_has_export_process(self):
        """Test app has export_process attribute."""
        app = GradioApp()

        assert hasattr(app, "export_process")
        assert app.export_process is None

    def test_stop_export_no_process(self):
        """Test stop when no export in progress."""
        app = GradioApp()
        result = app.stop_export()

        assert result == "No export in progress."

    def test_stop_export_with_process(self):
        """Test stop with export in progress."""
        app = GradioApp()

        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        app.export_process = mock_process

        result = app.stop_export()

        assert result == "Export stopped."
        assert app.export_process is None
        mock_process.terminate.assert_called_once()

    def test_run_export_invalid_input(self):
        """Test export with invalid input path."""
        app = GradioApp()

        gen = app.run_export(
            "/nonexistent/model.ckpt",
            "./model.onnx",
            "onnx",
            "fp16",
            True,
        )

        result = list(gen)
        assert any("Error" in line for line in result)


class TestGradioAppFaceRestoration:
    """Tests for face restoration in GradioApp."""

    def test_init_has_restorer(self):
        """Test app has _restorer attribute."""
        app = GradioApp()

        assert hasattr(app, "_restorer")
        assert app._restorer is None

    def test_restore_no_image(self):
        """Test error when image not provided."""
        app = GradioApp()

        with pytest.raises(gr.Error, match="provide an image"):
            app.apply_face_restoration(None, 0.5, 1.4)
