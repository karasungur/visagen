"""Tests for AppState default settings autoload behavior."""

from __future__ import annotations

import json
from pathlib import Path


def test_app_state_create_autoloads_default_settings_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from visagen.gui.state.app_state import AppState

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "locale": "tr",
                "default_batch_size": 16,
            }
        )
    )

    monkeypatch.chdir(tmp_path)
    state = AppState.create()

    assert state.settings.locale == "tr"
    assert state.settings.default_batch_size == 16
