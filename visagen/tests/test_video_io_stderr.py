"""Regression tests for FFmpeg stderr pipe deadlock prevention."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


class _ReaderProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"")

    def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        return None


class _WriterProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()

    def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        return None


class _FakeStream:
    def __init__(self, captures: list[dict], process) -> None:
        self._captures = captures
        self._process = process

    def filter(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return self

    def output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return self

    def overwrite_output(self):
        return self

    def run_async(self, **kwargs):  # type: ignore[no-untyped-def]
        self._captures.append(kwargs)
        return self._process


class _FakeFFmpeg:
    def __init__(self, captures: list[dict], process) -> None:
        self._captures = captures
        self._process = process

    def input(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return _FakeStream(self._captures, self._process)


def test_video_reader_run_async_disables_stderr_pipe(
    monkeypatch, tmp_path: Path
) -> None:
    """Reader must avoid pipe_stderr buffering to prevent deadlocks."""
    from visagen.merger.video_io import VideoInfo, VideoReader

    captures: list[dict] = []
    fake_process = _ReaderProcess()
    monkeypatch.setattr(
        "visagen.merger.video_io._get_ffmpeg",
        lambda: _FakeFFmpeg(captures, fake_process),
    )
    monkeypatch.setattr("visagen.merger.video_io._get_ffmpeg_exe", lambda: "ffmpeg")

    reader = VideoReader(tmp_path / "input.mp4")
    reader.get_info = lambda: VideoInfo(  # type: ignore[method-assign]
        width=16,
        height=16,
        fps=25.0,
        total_frames=1,
        duration=0.04,
        has_audio=False,
        codec="h264",
    )

    list(reader.iter_frames(start=0, end=0))

    assert captures
    assert captures[0]["pipe_stderr"] is False


def test_video_writer_run_async_disables_stderr_pipe(
    monkeypatch, tmp_path: Path
) -> None:
    """Writer must avoid pipe_stderr buffering to prevent deadlocks."""
    from visagen.merger.video_io import VideoWriter

    captures: list[dict] = []
    fake_process = _WriterProcess()
    monkeypatch.setattr(
        "visagen.merger.video_io._get_ffmpeg",
        lambda: _FakeFFmpeg(captures, fake_process),
    )
    monkeypatch.setattr("visagen.merger.video_io._get_ffmpeg_exe", lambda: "ffmpeg")

    writer = VideoWriter(tmp_path / "out.mp4", 16, 16, 25.0, codec="libx264")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    writer.write_frame(frame)
    writer.finalize()

    assert captures
    assert captures[0]["pipe_stderr"] is False
