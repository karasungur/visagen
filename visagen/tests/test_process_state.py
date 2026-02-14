"""Tests for atomic process slot management in ProcessState."""

from __future__ import annotations


class _DummyProcess:
    """Minimal subprocess-like stub."""

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


def test_launch_prevents_duplicate_training_process(monkeypatch) -> None:
    """Launching same slot twice should keep only the first live process."""
    from visagen.gui.state.app_state import ProcessState

    launched: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        launched.append(cmd)
        return _DummyProcess()

    monkeypatch.setattr("visagen.gui.state.app_state.subprocess.Popen", _fake_popen)

    state = ProcessState()
    first = state.launch("training", ["python", "-m", "visagen.tools.train"])
    second = state.launch("training", ["python", "-m", "visagen.tools.train"])

    assert first is not None
    assert second is None
    assert state.is_running("training") is True
    assert len(launched) == 1


def test_clear_if_does_not_clear_newer_process() -> None:
    """clear_if should only clear exact expected handle."""
    from visagen.gui.state.app_state import ProcessState

    state = ProcessState()
    old = _DummyProcess()
    new = _DummyProcess()
    state.training = new

    assert state.clear_if("training", old) is False
    assert state.training is new

    assert state.clear_if("training", new) is True
    assert state.training is None
