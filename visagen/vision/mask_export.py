"""Mask Export utilities for SegFormer results.

Export segmentation masks to LabelMe and COCO formats
for external editing with tools like LabelMe, CVAT, VIA, Label Studio.

SegFormer provides 90%+ accuracy for face segmentation, but for the
remaining cases that need manual correction, these functions enable
exporting to standard annotation formats.

Supported Formats:
    - LabelMe: Per-image JSON with polygon annotations
    - COCO: Dataset-wide JSON with segmentation annotations

Example:
    >>> from visagen.vision.mask_export import export_labelme, export_coco
    >>> # Single image to LabelMe
    >>> export_labelme(Path("face.jpg"), mask, label="face")
    >>> # Multiple images to COCO
    >>> export_coco(image_paths, masks, Path("annotations.json"))

Reference:
    - LabelMe: https://github.com/wkentaro/labelme
    - COCO format: https://cocodataset.org/#format-data
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 100,
    epsilon_ratio: float = 0.005,
) -> list[np.ndarray]:
    """Convert binary mask to polygon contours.

    Extracts contours from a binary mask and simplifies them using
    the Douglas-Peucker algorithm.

    Args:
        mask: Binary mask (H, W) uint8 with values 0 or 255.
        min_area: Minimum polygon area to keep. Default: 100.
        epsilon_ratio: Douglas-Peucker approximation ratio. Default: 0.005.
            Smaller values produce more detailed polygons.

    Returns:
        List of polygon arrays, each with shape (N, 2) containing
        (x, y) coordinates.

    Example:
        >>> mask = np.zeros((256, 256), dtype=np.uint8)
        >>> cv2.circle(mask, (128, 128), 50, 255, -1)
        >>> polygons = mask_to_polygons(mask)
        >>> len(polygons)
        1
    """
    # Ensure binary mask
    mask_binary = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour using Douglas-Peucker
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points for a valid polygon
        if len(approx) >= 3:
            # Squeeze from (N, 1, 2) to (N, 2)
            polygons.append(approx.squeeze())

    return polygons


def polygons_to_mask(
    polygons: list[np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert polygons to binary mask.

    Args:
        polygons: List of polygon arrays, each (N, 2).
        height: Output mask height.
        width: Output mask width.

    Returns:
        Binary mask (H, W) uint8 with values 0 or 255.

    Example:
        >>> poly = np.array([[100, 100], [150, 100], [150, 150], [100, 150]])
        >>> mask = polygons_to_mask([poly], 256, 256)
        >>> mask.shape
        (256, 256)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if len(pts.shape) == 1:
            pts = pts.reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def export_labelme(
    image_path: Path,
    mask: np.ndarray,
    output_path: Path | None = None,
    label: str = "face",
    include_image_data: bool = False,
    min_area: int = 100,
) -> Path:
    """Export mask to LabelMe JSON format.

    Creates a LabelMe-compatible JSON file with polygon annotations
    that can be opened and edited in LabelMe or compatible tools.

    Args:
        image_path: Source image path.
        mask: Binary mask (H, W) with values 0-255.
        output_path: Output JSON path. Default: image_path.with_suffix('.json').
        label: Polygon label name. Default: "face".
        include_image_data: Include base64-encoded image data. Default: False.
        min_area: Minimum polygon area to include. Default: 100.

    Returns:
        Path to saved JSON file.

    Example:
        >>> mask = create_face_mask(image)
        >>> json_path = export_labelme(Path("face.jpg"), mask)
        >>> # Open json_path in LabelMe for editing
    """
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.with_suffix(".json")
    output_path = Path(output_path)

    h, w = mask.shape[:2]
    polygons = mask_to_polygons(mask, min_area=min_area)

    shapes = []
    for poly in polygons:
        # Convert to list of [x, y] points
        points = poly.tolist()
        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
        )

    # Optionally include base64-encoded image
    image_data = None
    if include_image_data and image_path.exists():
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    labelme_data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(labelme_data, indent=2))

    logger.info(f"Exported LabelMe format: {output_path} ({len(shapes)} shapes)")
    return output_path


def import_labelme(json_path: Path) -> tuple[np.ndarray, dict]:
    """Import mask from LabelMe JSON.

    Reads a LabelMe JSON file and converts polygon annotations
    back to a binary mask.

    Args:
        json_path: LabelMe JSON file path.

    Returns:
        Tuple of (mask, metadata) where:
            - mask: Binary mask (H, W) uint8 with values 0 or 255
            - metadata: Dict with 'labels', 'image_path', 'image_size'

    Example:
        >>> mask, meta = import_labelme(Path("face.json"))
        >>> print(meta['labels'])
        ['face']
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    h = data["imageHeight"]
    w = data["imageWidth"]

    mask = np.zeros((h, w), dtype=np.uint8)
    labels = set()

    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "polygon":
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            labels.add(shape.get("label", "unknown"))

    metadata = {
        "labels": list(labels),
        "image_path": data.get("imagePath"),
        "image_size": (h, w),
    }

    return mask, metadata


def export_coco(
    image_paths: list[Path],
    masks: list[np.ndarray],
    output_path: Path,
    categories: list[str] | None = None,
    min_area: int = 100,
) -> Path:
    """Export masks to COCO JSON format.

    Creates a COCO-compatible annotation file for multiple images,
    suitable for use with COCO-compatible tools and training pipelines.

    Args:
        image_paths: List of image paths.
        masks: List of binary masks, one per image.
        output_path: Output JSON path.
        categories: Category names. Default: ["face"].
        min_area: Minimum polygon area to include. Default: 100.

    Returns:
        Path to saved JSON file.

    Example:
        >>> images = [Path("face1.jpg"), Path("face2.jpg")]
        >>> masks = [mask1, mask2]
        >>> export_coco(images, masks, Path("annotations.json"))
    """
    if categories is None:
        categories = ["face"]

    output_path = Path(output_path)

    coco_data: dict = {
        "info": {
            "description": "Visagen SegFormer Export",
            "version": "1.0",
            "year": 2026,
            "contributor": "Visagen",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(categories)
        ],
    }

    ann_id = 1
    for img_id, (img_path, mask) in enumerate(zip(image_paths, masks, strict=True), 1):
        img_path = Path(img_path)
        h, w = mask.shape[:2]

        coco_data["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "height": h,
                "width": w,
            }
        )

        polygons = mask_to_polygons(mask, min_area=min_area)
        for poly in polygons:
            # COCO expects flattened [x1, y1, x2, y2, ...] format
            flat_poly = poly.flatten().tolist()

            # Compute bounding box and area
            x, y, pw, ph = cv2.boundingRect(poly.astype(np.int32))
            area = cv2.contourArea(poly.astype(np.float32))

            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # Default to first category
                    "segmentation": [flat_poly],
                    "area": float(area),
                    "bbox": [int(x), int(y), int(pw), int(ph)],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco_data, indent=2))

    n_images = len(coco_data["images"])
    n_anns = len(coco_data["annotations"])
    logger.info(
        f"Exported COCO format: {output_path} ({n_images} images, {n_anns} annotations)"
    )
    return output_path


def import_coco(
    json_path: Path,
    image_dir: Path | None = None,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Import masks from COCO JSON.

    Reads a COCO annotation file and converts segmentation annotations
    to binary masks.

    Args:
        json_path: COCO JSON file path.
        image_dir: Directory containing images. Default: json_path.parent.

    Returns:
        Dict mapping image filenames to (mask, categories) tuples.

    Example:
        >>> masks = import_coco(Path("annotations.json"))
        >>> for filename, (mask, cats) in masks.items():
        ...     print(f"{filename}: {mask.shape}, categories: {cats}")
    """
    json_path = Path(json_path)
    if image_dir is None:
        image_dir = json_path.parent

    data = json.loads(json_path.read_text())

    # Build category lookup
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Build image lookup
    images = {img["id"]: img for img in data.get("images", [])}

    # Group annotations by image
    image_annotations: dict[int, list] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    result = {}
    for img_id, img_info in images.items():
        h = img_info["height"]
        w = img_info["width"]
        filename = img_info["file_name"]

        mask = np.zeros((h, w), dtype=np.uint8)
        cats = set()

        for ann in image_annotations.get(img_id, []):
            cat_id = ann.get("category_id", 1)
            cats.add(categories.get(cat_id, "unknown"))

            for seg in ann.get("segmentation", []):
                if isinstance(seg, list):
                    # Polygon format: [x1, y1, x2, y2, ...]
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 255)

        result[filename] = (mask, list(cats))

    return result


def export_masks_batch(
    image_dir: Path,
    masks: dict[str, np.ndarray],
    output_dir: Path,
    format: str = "labelme",
    label: str = "face",
) -> list[Path]:
    """Export multiple masks in batch.

    Convenience function for exporting masks for multiple images.

    Args:
        image_dir: Directory containing source images.
        masks: Dict mapping image filename to mask.
        output_dir: Output directory for annotations.
        format: Export format ('labelme' or 'coco'). Default: 'labelme'.
        label: Label for annotations. Default: 'face'.

    Returns:
        List of created annotation file paths.

    Example:
        >>> masks = {"face1.jpg": mask1, "face2.jpg": mask2}
        >>> paths = export_masks_batch(
        ...     Path("images"), masks, Path("annotations"), format="labelme"
        ... )
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "coco":
        # Export all to single COCO file
        image_paths = [image_dir / name for name in masks.keys()]
        mask_list = list(masks.values())
        output_path = output_dir / "annotations.json"
        export_coco(image_paths, mask_list, output_path, categories=[label])
        return [output_path]

    else:  # labelme
        output_paths = []
        for filename, mask in masks.items():
            image_path = image_dir / filename
            output_path = output_dir / Path(filename).with_suffix(".json")
            export_labelme(image_path, mask, output_path, label=label)
            output_paths.append(output_path)
        return output_paths
