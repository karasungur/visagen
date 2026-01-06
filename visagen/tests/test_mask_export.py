"""
Tests for Mask Export module.

These tests verify the LabelMe and COCO export/import
functionality for SegFormer mask outputs.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from visagen.vision.mask_export import (
    export_coco,
    export_labelme,
    export_masks_batch,
    import_coco,
    import_labelme,
    mask_to_polygons,
    polygons_to_mask,
)


class TestMaskToPolygons:
    """Tests for mask_to_polygons function."""

    def test_single_blob(self):
        """Single blob should produce one polygon."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 255, -1)

        polygons = mask_to_polygons(mask)

        assert len(polygons) == 1
        assert polygons[0].shape[1] == 2  # (N, 2) shape

    def test_multiple_blobs(self):
        """Multiple blobs should produce multiple polygons."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 30, 255, -1)
        cv2.circle(mask, (192, 192), 30, 255, -1)

        polygons = mask_to_polygons(mask)

        assert len(polygons) == 2

    def test_empty_mask(self):
        """Empty mask should produce no polygons."""
        mask = np.zeros((256, 256), dtype=np.uint8)

        polygons = mask_to_polygons(mask)

        assert len(polygons) == 0

    def test_small_blob_filtered(self):
        """Small blobs below min_area should be filtered."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 5, 255, -1)  # Small circle

        polygons = mask_to_polygons(mask, min_area=500)

        assert len(polygons) == 0

    def test_polygon_has_valid_points(self):
        """Polygon should have at least 3 points."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)

        polygons = mask_to_polygons(mask)

        assert len(polygons) == 1
        assert len(polygons[0]) >= 3


class TestPolygonsToMask:
    """Tests for polygons_to_mask function."""

    def test_single_polygon(self):
        """Single polygon should create filled region."""
        poly = np.array([[100, 100], [150, 100], [150, 150], [100, 150]])

        mask = polygons_to_mask([poly], 256, 256)

        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert mask[125, 125] == 255  # Inside polygon
        assert mask[50, 50] == 0  # Outside polygon

    def test_multiple_polygons(self):
        """Multiple polygons should all be filled."""
        poly1 = np.array([[50, 50], [100, 50], [100, 100], [50, 100]])
        poly2 = np.array([[150, 150], [200, 150], [200, 200], [150, 200]])

        mask = polygons_to_mask([poly1, poly2], 256, 256)

        assert mask[75, 75] == 255  # Inside poly1
        assert mask[175, 175] == 255  # Inside poly2
        assert mask[125, 125] == 0  # Between polygons

    def test_empty_list(self):
        """Empty polygon list should create empty mask."""
        mask = polygons_to_mask([], 256, 256)

        assert mask.shape == (256, 256)
        assert mask.max() == 0

    def test_roundtrip_mask_polygon_mask(self):
        """Mask -> polygon -> mask should preserve shape approximately."""
        original = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(original, (128, 128), 50, 255, -1)

        polygons = mask_to_polygons(original)
        reconstructed = polygons_to_mask(polygons, 256, 256)

        # Should have similar non-zero area (within 10%)
        original_area = (original > 0).sum()
        reconstructed_area = (reconstructed > 0).sum()

        assert abs(original_area - reconstructed_area) / original_area < 0.1


class TestExportLabelMe:
    """Tests for export_labelme function."""

    def test_creates_json_file(self):
        """Should create a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()

            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128, 128), 50, 255, -1)

            output_path = export_labelme(image_path, mask)

            assert output_path.exists()
            assert output_path.suffix == ".json"

    def test_json_structure(self):
        """Output JSON should have LabelMe structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()

            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128, 128), 50, 255, -1)

            output_path = export_labelme(image_path, mask, label="face")
            data = json.loads(output_path.read_text())

            assert "version" in data
            assert "shapes" in data
            assert "imagePath" in data
            assert "imageHeight" in data
            assert "imageWidth" in data
            assert data["imageHeight"] == 256
            assert data["imageWidth"] == 256

    def test_shape_structure(self):
        """Shapes should have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()

            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128, 128), 50, 255, -1)

            output_path = export_labelme(image_path, mask, label="face")
            data = json.loads(output_path.read_text())

            assert len(data["shapes"]) == 1
            shape = data["shapes"][0]
            assert shape["label"] == "face"
            assert shape["shape_type"] == "polygon"
            assert "points" in shape
            assert len(shape["points"]) >= 3

    def test_custom_output_path(self):
        """Should use custom output path when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()
            custom_output = tmpdir / "custom" / "output.json"

            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128, 128), 50, 255, -1)

            output_path = export_labelme(image_path, mask, output_path=custom_output)

            assert output_path == custom_output
            assert output_path.exists()


class TestImportLabelMe:
    """Tests for import_labelme function."""

    def test_roundtrip_export_import(self):
        """Export then import should reconstruct mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()

            original_mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(original_mask, (50, 50), (200, 200), 255, -1)

            json_path = export_labelme(image_path, original_mask)
            imported_mask, metadata = import_labelme(json_path)

            assert imported_mask.shape == original_mask.shape
            assert metadata["image_size"] == (256, 256)
            assert "face" in metadata["labels"]

    def test_metadata_contains_labels(self):
        """Metadata should contain label information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "test.jpg"
            image_path.touch()

            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128, 128), 50, 255, -1)

            json_path = export_labelme(image_path, mask, label="custom_label")
            _, metadata = import_labelme(json_path)

            assert "custom_label" in metadata["labels"]


class TestExportCOCO:
    """Tests for export_coco function."""

    def test_creates_json_file(self):
        """Should create a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_paths = [tmpdir / "img1.jpg", tmpdir / "img2.jpg"]
            for p in image_paths:
                p.touch()

            masks = [
                np.zeros((256, 256), dtype=np.uint8),
                np.zeros((256, 256), dtype=np.uint8),
            ]
            cv2.circle(masks[0], (128, 128), 50, 255, -1)
            cv2.circle(masks[1], (100, 100), 40, 255, -1)

            output_path = tmpdir / "annotations.json"
            result = export_coco(image_paths, masks, output_path)

            assert result.exists()
            assert result.suffix == ".json"

    def test_coco_structure(self):
        """Output should have COCO structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_paths = [tmpdir / "img1.jpg"]
            image_paths[0].touch()

            masks = [np.zeros((256, 256), dtype=np.uint8)]
            cv2.circle(masks[0], (128, 128), 50, 255, -1)

            output_path = tmpdir / "annotations.json"
            export_coco(image_paths, masks, output_path)
            data = json.loads(output_path.read_text())

            assert "images" in data
            assert "annotations" in data
            assert "categories" in data
            assert len(data["images"]) == 1
            assert len(data["annotations"]) == 1
            assert len(data["categories"]) == 1

    def test_annotation_structure(self):
        """Annotations should have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_paths = [tmpdir / "img1.jpg"]
            image_paths[0].touch()

            masks = [np.zeros((256, 256), dtype=np.uint8)]
            cv2.circle(masks[0], (128, 128), 50, 255, -1)

            output_path = tmpdir / "annotations.json"
            export_coco(image_paths, masks, output_path)
            data = json.loads(output_path.read_text())

            ann = data["annotations"][0]
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "segmentation" in ann
            assert "bbox" in ann
            assert "area" in ann
            assert len(ann["bbox"]) == 4

    def test_custom_categories(self):
        """Should use custom categories when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_paths = [tmpdir / "img1.jpg"]
            image_paths[0].touch()

            masks = [np.zeros((256, 256), dtype=np.uint8)]
            cv2.circle(masks[0], (128, 128), 50, 255, -1)

            output_path = tmpdir / "annotations.json"
            export_coco(
                image_paths, masks, output_path, categories=["face", "background"]
            )
            data = json.loads(output_path.read_text())

            assert len(data["categories"]) == 2
            cat_names = [c["name"] for c in data["categories"]]
            assert "face" in cat_names
            assert "background" in cat_names


class TestImportCOCO:
    """Tests for import_coco function."""

    def test_roundtrip_export_import(self):
        """Export then import should reconstruct masks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_paths = [tmpdir / "img1.jpg"]
            image_paths[0].touch()

            original_masks = [np.zeros((256, 256), dtype=np.uint8)]
            cv2.circle(original_masks[0], (128, 128), 50, 255, -1)

            output_path = tmpdir / "annotations.json"
            export_coco(image_paths, original_masks, output_path)

            result = import_coco(output_path, tmpdir)

            assert "img1.jpg" in result
            mask, cats = result["img1.jpg"]
            assert mask.shape == (256, 256)
            assert "face" in cats


class TestExportMasksBatch:
    """Tests for export_masks_batch function."""

    def test_labelme_batch(self):
        """Batch LabelMe export should create multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            output_dir = tmpdir / "annotations"
            image_dir.mkdir()

            # Create dummy images
            (image_dir / "face1.jpg").touch()
            (image_dir / "face2.jpg").touch()

            masks = {
                "face1.jpg": np.zeros((256, 256), dtype=np.uint8),
                "face2.jpg": np.zeros((256, 256), dtype=np.uint8),
            }
            cv2.circle(masks["face1.jpg"], (128, 128), 50, 255, -1)
            cv2.circle(masks["face2.jpg"], (100, 100), 40, 255, -1)

            paths = export_masks_batch(image_dir, masks, output_dir, format="labelme")

            assert len(paths) == 2
            for p in paths:
                assert p.exists()
                assert p.suffix == ".json"

    def test_coco_batch(self):
        """Batch COCO export should create single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            output_dir = tmpdir / "annotations"
            image_dir.mkdir()

            (image_dir / "face1.jpg").touch()
            (image_dir / "face2.jpg").touch()

            masks = {
                "face1.jpg": np.zeros((256, 256), dtype=np.uint8),
                "face2.jpg": np.zeros((256, 256), dtype=np.uint8),
            }
            cv2.circle(masks["face1.jpg"], (128, 128), 50, 255, -1)
            cv2.circle(masks["face2.jpg"], (100, 100), 40, 255, -1)

            paths = export_masks_batch(image_dir, masks, output_dir, format="coco")

            assert len(paths) == 1
            assert paths[0].name == "annotations.json"
            assert paths[0].exists()
