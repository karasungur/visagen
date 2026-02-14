"""Restore default mode contract: GFPGAN default with GPEN fallback."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import numpy as np

from visagen.postprocess.restore import FaceRestorer, RestoreConfig


def test_gfpgan_mode_falls_back_to_gpen_when_gfpgan_unavailable() -> None:
    face = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = np.ones_like(face) * 7

    restorer = FaceRestorer(RestoreConfig(enabled=True, mode="gfpgan"))

    with (
        patch.object(
            FaceRestorer, "gfpgan", new_callable=PropertyMock, return_value=None
        ),
        patch("visagen.postprocess.gpen.is_gpen_available", return_value=True),
        patch.object(restorer, "_restore_gpen", return_value=expected) as gpen_mock,
    ):
        result = restorer._restore_gfpgan(face)

    gpen_mock.assert_called_once()
    assert np.array_equal(result, expected)


def test_gfpgan_mode_returns_original_when_no_fallback_available() -> None:
    face = np.zeros((8, 8, 3), dtype=np.uint8)
    restorer = FaceRestorer(RestoreConfig(enabled=True, mode="gfpgan"))

    with (
        patch.object(
            FaceRestorer, "gfpgan", new_callable=PropertyMock, return_value=None
        ),
        patch("visagen.postprocess.gpen.is_gpen_available", return_value=False),
    ):
        result = restorer._restore_gfpgan(face)

    assert result is face


def test_is_available_in_gfpgan_mode_accepts_gpen_fallback() -> None:
    restorer = FaceRestorer(RestoreConfig(enabled=True, mode="gfpgan"))

    with (
        patch("visagen.postprocess.restore.is_gfpgan_available", return_value=False),
        patch("visagen.postprocess.gpen.is_gpen_available", return_value=True),
    ):
        assert restorer.is_available() is True
