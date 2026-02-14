"""Contract tests for MaskEditor batch preview/confirm/cancel flow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")

from visagen.gui.i18n import I18n
from visagen.gui.state.app_state import AppState
from visagen.gui.tabs.mask_editor import MaskEditorTab


def _create_tab() -> MaskEditorTab:
    return MaskEditorTab(AppState(), I18n(locale="en"))


def _make_batch_files(tmp_path: Path) -> tuple[Path, list[Path]]:
    input_dir = tmp_path / "aligned"
    input_dir.mkdir()
    files = [input_dir / "a.jpg", input_dir / "b.jpg"]
    for p in files:
        p.write_bytes(b"x")
    return input_dir, files


def test_preview_then_confirm_saves_pending_request(
    monkeypatch, tmp_path: Path
) -> None:
    tab = _create_tab()
    input_dir, files = _make_batch_files(tmp_path)

    monkeypatch.setattr(tab, "_collect_batch_files", lambda _p: files)
    preview_items = [(np.zeros((8, 8, 3), dtype=np.uint8), "a")]
    monkeypatch.setattr(
        tab, "_generate_batch_preview", lambda *args, **kwargs: (preview_items, 0)
    )

    save_calls: list[dict] = []

    def _fake_save(request: dict) -> tuple[int, int]:
        save_calls.append(request)
        return (len(request["files"]), 0)

    monkeypatch.setattr(tab, "_save_batch_request", _fake_save)

    result = tab._apply_batch(
        str(input_dir),
        str(tmp_path / "out"),
        False,
        "",
        True,
        True,
        True,
        True,
        False,
        0,
        0,
        0,
        True,
    )

    assert "Preview ready" in result[1]
    assert tab._pending_batch_request is not None
    assert save_calls == []

    confirmed = tab._confirm_batch()
    assert "Saved 2/2 images" in confirmed[1]
    assert tab._pending_batch_request is None
    assert len(save_calls) == 1


def test_apply_without_preview_saves_immediately(monkeypatch, tmp_path: Path) -> None:
    tab = _create_tab()
    input_dir, files = _make_batch_files(tmp_path)

    monkeypatch.setattr(tab, "_collect_batch_files", lambda _p: files)
    monkeypatch.setattr(
        tab,
        "_generate_batch_preview",
        lambda *args, **kwargs: ([], 0),
    )

    save_calls: list[dict] = []

    def _fake_save(request: dict) -> tuple[int, int]:
        save_calls.append(request)
        return (1, 1)

    monkeypatch.setattr(tab, "_save_batch_request", _fake_save)

    result = tab._apply_batch(
        str(input_dir),
        str(tmp_path / "out"),
        False,
        "",
        True,
        True,
        True,
        True,
        False,
        0,
        0,
        0,
        False,
    )

    assert "Failed" in result[1]
    assert tab._pending_batch_request is None
    assert len(save_calls) == 1


def test_cancel_clears_pending_request() -> None:
    tab = _create_tab()
    tab._pending_batch_request = {"files": []}

    result = tab._cancel_batch()

    assert result[1] == "Batch preview cancelled"
    assert tab._pending_batch_request is None


def test_apply_rejects_when_pending_request_exists(tmp_path: Path) -> None:
    tab = _create_tab()
    input_dir, _files = _make_batch_files(tmp_path)
    tab._pending_batch_request = {"files": ["existing"]}

    result = tab._apply_batch(
        str(input_dir),
        str(tmp_path / "out"),
        False,
        "",
        True,
        True,
        True,
        True,
        False,
        0,
        0,
        0,
        True,
    )

    assert "Pending preview exists" in result[1]


def test_apply_is_guarded_by_batch_lock(tmp_path: Path) -> None:
    tab = _create_tab()
    input_dir, _files = _make_batch_files(tmp_path)

    acquired = tab._batch_lock.acquire(blocking=False)
    assert acquired is True
    try:
        result = tab._apply_batch(
            str(input_dir),
            str(tmp_path / "out"),
            False,
            "",
            True,
            True,
            True,
            True,
            False,
            0,
            0,
            0,
            True,
        )
    finally:
        tab._batch_lock.release()

    assert result[1] == "Batch operation already in progress"
