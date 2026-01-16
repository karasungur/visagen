"""Tests for Gradio web interface."""

import pytest

# Skip all tests if gradio not installed
gr = pytest.importorskip("gradio")

from visagen.tools.gradio_app import GRADIO_AVAILABLE, create_app


class TestGradioApp:
    """Tests for main Gradio app entry point."""

    def test_gradio_available(self):
        """Test GRADIO_AVAILABLE flag."""
        assert GRADIO_AVAILABLE is True

    def test_create_app(self):
        """Test app is created successfully."""
        app = create_app()

        assert app is not None
        assert isinstance(app, gr.Blocks)
