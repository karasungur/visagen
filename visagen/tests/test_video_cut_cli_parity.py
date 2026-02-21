"""CLI parity tests for video_ed cut command arguments."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from visagen.tools.video_ed import main


def test_cut_cli_forwards_audio_track_and_bitrate(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-video",
            "cut",
            "input.mp4",
            "output.mp4",
            "--start",
            "00:00:01",
            "--end",
            "00:00:03",
            "--audio-track-id",
            "3",
            "--codec",
            "libx264",
            "--bitrate",
            "18M",
        ],
    )

    with patch("visagen.tools.video_ed.cut_video") as cut_mock:
        main()

    cut_mock.assert_called_once_with(
        Path("input.mp4"),
        Path("output.mp4"),
        "00:00:01",
        "00:00:03",
        audio_track=3,
        codec="libx264",
        bitrate="18M",
    )


def test_cut_cli_uses_none_bitrate_when_not_provided(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visagen-video",
            "cut",
            "input.mp4",
            "output.mp4",
            "--start",
            "00:00:00",
            "--end",
            "00:00:02",
            "--audio-track-id",
            "0",
        ],
    )

    with patch("visagen.tools.video_ed.cut_video") as cut_mock:
        main()

    args, kwargs = cut_mock.call_args
    assert args[0] == Path("input.mp4")
    assert args[1] == Path("output.mp4")
    assert kwargs["audio_track"] == 0
    assert kwargs["bitrate"] is None
