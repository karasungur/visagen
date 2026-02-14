"""Mask editor UI/backend contract tests for annotation import/export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("gradio")

from visagen.gui.i18n import I18n
from visagen.gui.state.app_state import AppState
from visagen.gui.tabs.mask_editor import MaskEditorTab


def _create_tab() -> MaskEditorTab:
    return MaskEditorTab(AppState(), I18n(locale="en"))


def _create_image_with_sidecar_mask(image_dir: Path, stem: str = "face") -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}.png"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    cv2.imwrite(str(image_path), image)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    cv2.imwrite(str(image_dir / f"{stem}_mask.png"), mask)


def test_annotation_controls_exist_in_batch_tab() -> None:
    import gradio as gr

    tab = _create_tab()
    with gr.Blocks():
        components = tab._build_batch_tab()

    required = {
        "annotation_input_dir",
        "annotation_output_path",
        "annotation_format",
        "annotation_export_btn",
        "annotation_import_path",
        "annotation_import_output_dir",
        "annotation_import_format",
        "annotation_import_btn",
        "annotation_status",
    }
    assert required.issubset(set(components.keys()))


def test_labelme_export_import_roundtrip(tmp_path: Path) -> None:
    tab = _create_tab()
    image_dir = tmp_path / "images"
    _create_image_with_sidecar_mask(image_dir, "face")

    export_dir = tmp_path / "labelme"
    status = tab._export_annotations(
        str(image_dir),
        str(export_dir),
        "labelme",
        False,
        10,
    )
    assert "Exported" in status
    assert (export_dir / "face.json").exists()

    import_dir = tmp_path / "imported"
    import_status = tab._import_annotations(str(export_dir), str(import_dir), "labelme")
    assert "Imported" in import_status
    assert (import_dir / "face_mask.png").exists()


def test_coco_export_import_roundtrip(tmp_path: Path) -> None:
    tab = _create_tab()
    image_dir = tmp_path / "images"
    _create_image_with_sidecar_mask(image_dir, "face")

    coco_path = tmp_path / "annotations.json"
    status = tab._export_annotations(
        str(image_dir),
        str(coco_path),
        "coco",
        False,
        10,
    )
    assert "COCO" in status
    assert coco_path.exists()

    import_dir = tmp_path / "imported"
    import_status = tab._import_annotations(str(coco_path), str(import_dir), "coco")
    assert "Imported" in import_status
    assert (import_dir / "face_mask.png").exists()
