"""Race-safety tests for tab process slots."""

from __future__ import annotations


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


def test_clear_if_does_not_erase_replaced_process(monkeypatch) -> None:
    from visagen.gui.state.app_state import ProcessState

    launched: list[_DummyProcess] = []

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del cmd, kwargs
        proc = _DummyProcess()
        launched.append(proc)
        return proc

    monkeypatch.setattr("visagen.gui.state.app_state.subprocess.Popen", _fake_popen)

    state = ProcessState()
    old = state.launch("merge", ["python", "-m", "visagen.tools.merge"])
    assert old is not None

    # Mark old process as exited, then launch replacement.
    old.terminate()
    new = state.launch("merge", ["python", "-m", "visagen.tools.merge"])
    assert new is not None
    assert new is not old

    assert state.clear_if("merge", old) is False
    assert state.get("merge") is new


def test_terminate_during_race_clears_slot(monkeypatch) -> None:
    from visagen.gui.state.app_state import ProcessState

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del cmd, kwargs
        return _DummyProcess()

    monkeypatch.setattr("visagen.gui.state.app_state.subprocess.Popen", _fake_popen)

    state = ProcessState()
    proc = state.launch("sort", ["python", "-m", "visagen.tools.sorter"])
    assert proc is not None
    assert state.is_running("sort") is True

    assert state.terminate("sort") is True
    assert state.get("sort") is None
