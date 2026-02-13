"""Tests for Gradio web interface."""

from pathlib import Path
from unittest.mock import patch

import pytest

# Skip all tests if gradio not installed
gr = pytest.importorskip("gradio")

from visagen.gui.app import create_app
from visagen.gui.state.app_state import AppState
from visagen.gui.tabs.settings import SettingsTab


class TestGradioApp:
    """Tests for main Gradio app entry point."""

    def test_create_app(self):
        """Test app is created successfully."""
        # Mask editor wiring is integration-heavy; patch it to keep this test
        # focused on the canonical app entrypoint contract.
        with patch(
            "visagen.gui.tabs.mask_editor.MaskEditorTab.create", return_value=None
        ):
            app = create_app()

        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_create_app_forwards_settings_path_consistently(
        self, tmp_path: Path
    ) -> None:
        """Custom settings path should be passed to both state + settings tab."""
        settings_path = tmp_path / "custom_settings.json"
        captured: dict[str, Path | None] = {"settings_path": None}
        original_init = SettingsTab.__init__

        def _spy_init(self, app_state, i18n, settings_path=None):  # type: ignore[no-untyped-def]
            captured["settings_path"] = settings_path
            return original_init(self, app_state, i18n, settings_path=settings_path)

        with (
            patch(
                "visagen.gui.tabs.mask_editor.MaskEditorTab.create",
                return_value=None,
            ),
            patch.object(SettingsTab, "__init__", _spy_init),
            patch(
                "visagen.gui.app.AppState.create",
                wraps=AppState.create,
            ) as create_mock,
        ):
            app = create_app(settings_path=settings_path)

        assert isinstance(app, gr.Blocks)
        create_mock.assert_called_once_with(settings_path)
        assert captured["settings_path"] == settings_path
