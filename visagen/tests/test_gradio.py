"""Tests for Gradio web interface."""

import pytest
from unittest.mock import patch

# Skip all tests if gradio not installed
gr = pytest.importorskip("gradio")

from visagen.gui.app import create_app


class TestGradioApp:
    """Tests for main Gradio app entry point."""

    def test_create_app(self):
        """Test app is created successfully."""
        # Mask editor wiring is integration-heavy; patch it to keep this test
        # focused on the canonical app entrypoint contract.
        with patch("visagen.gui.tabs.mask_editor.MaskEditorTab.create", return_value=None):
            app = create_app()

        assert app is not None
        assert isinstance(app, gr.Blocks)
