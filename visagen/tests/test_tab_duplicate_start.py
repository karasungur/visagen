"""Duplicate start protection tests for managed process slots."""

from __future__ import annotations

import pytest


class _DummyProcess:
    def __init__(self) -> None:
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def terminate(self) -> None:
        self._running = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._running = False
        return 0

    def kill(self) -> None:
        self._running = False


@pytest.mark.parametrize(
    "slot",
    [
        "merge",
        "sort",
        "export",
        "extract",
        "training",
        "video_tools",
        "faceset_tools",
        "benchmark",
        "batch",
    ],
)
def test_duplicate_launch_is_blocked_for_slot(monkeypatch, slot: str) -> None:
    from visagen.gui.state.app_state import ProcessState

    launches: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        launches.append(cmd)
        return _DummyProcess()

    monkeypatch.setattr("visagen.gui.state.app_state.subprocess.Popen", _fake_popen)

    state = ProcessState()
    first = state.launch(slot, ["python", "-m", "noop"])
    second = state.launch(slot, ["python", "-m", "noop"])

    assert first is not None
    assert second is None
    assert len(launches) == 1
