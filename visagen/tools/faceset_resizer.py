"""Faceset resizing tool with metadata preservation.

Batch resizes faceset images while preserving DFL metadata such as
landmarks, face type, and segmentation masks.

Features:
    - Maintains DFL metadata (landmarks, masks, etc.)
    - Parallel batch processing
    - Multiple interpolation methods
    - Face type conversion support

References:
    - Legacy DFL: mainscripts/FacesetResizer.py
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from visagen.vision.dflimg import DFLImage, FaceMetadata

logger = logging.getLogger(__name__)

# Face type definitions matching DFL conventions
FACE_TYPES = Literal["half_face", "mid_face", "full_face", "whole_face", "head"]

# Face type resolution multipliers (relative to half_face)
FACE_TYPE_MULTS = {
    "half_face": 1.0,
    "mid_face": 1.2,
    "full_face": 1.5,
    "whole_face": 2.0,
    "head": 2.5,
}


@dataclass
class ResizeResult:
    """Result of faceset resize operation."""

    total_images: int
    resized_count: int
    skipped_count: int
    error_count: int
    output_dir: Path


def calculate_target_resolution(
    source_size: int,
    target_size: int,
    source_face_type: str | None,
    target_face_type: str | None,
) -> int:
    """Calculate target resolution considering face type conversion.

    Args:
        source_size: Source image size
        target_size: Target image size
        source_face_type: Source face type
        target_face_type: Target face type (None = keep same)

    Returns:
        Adjusted target resolution
    """
    if target_face_type is None or source_face_type is None:
        return target_size

    if source_face_type == target_face_type:
        return target_size

    # Get multipliers
    source_mult = FACE_TYPE_MULTS.get(source_face_type, 1.0)
    target_mult = FACE_TYPE_MULTS.get(target_face_type, 1.0)

    # Adjust resolution
    ratio = target_mult / source_mult
    return int(target_size * ratio)


def resize_single_face(
    input_path: Path,
    output_path: Path,
    target_size: int,
    *,
    target_face_type: str | None = None,
    interpolation: int = cv2.INTER_LANCZOS4,
    preserve_metadata: bool = True,
    jpeg_quality: int = 95,
) -> bool:
    """Resize a single face image with metadata preservation.

    Args:
        input_path: Path to input face image
        output_path: Path to save resized image
        target_size: Target image size (width = height)
        target_face_type: Target face type (None = keep same)
        interpolation: OpenCV interpolation method
        preserve_metadata: Whether to preserve DFL metadata
        jpeg_quality: JPEG quality for output

    Returns:
        True if successful, False otherwise
    """
    try:
        # Try to load with DFL metadata
        source_face_type = None
        metadata = None

        if preserve_metadata and input_path.suffix.lower() in [".jpg", ".jpeg"]:
            try:
                _, dfl_metadata = DFLImage.load(input_path)
                if dfl_metadata is not None:
                    source_face_type = dfl_metadata.face_type
                    metadata = dfl_metadata
            except Exception:
                pass

        # Read image with cv2 (faster than DFLImage.load for non-metadata cases)
        image = cv2.imread(str(input_path))
        if image is None:
            logger.warning(f"Failed to read image: {input_path}")
            return False

        source_size = image.shape[0]  # Assuming square images

        # Calculate actual target size considering face type
        actual_target = calculate_target_resolution(
            source_size, target_size, source_face_type, target_face_type
        )

        # Resize image
        if source_size != actual_target:
            resized = cv2.resize(
                image,
                (actual_target, actual_target),
                interpolation=interpolation,
            )
        else:
            resized = image

        # Scale metadata if needed
        if metadata is not None and source_size != actual_target:
            scale = actual_target / source_size

            # Scale landmarks
            if metadata.landmarks is not None:
                landmarks = np.array(metadata.landmarks) * scale
                metadata.landmarks = landmarks

            # Scale image_to_face_mat translation component
            if metadata.image_to_face_mat is not None:
                mat = np.array(metadata.image_to_face_mat)
                if mat.shape == (2, 3):
                    mat[:, 2] = mat[:, 2] * scale
                    metadata.image_to_face_mat = mat

            # Update face type if requested
            if target_face_type is not None:
                metadata.face_type = target_face_type

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image with or without metadata
        if metadata is not None and output_path.suffix.lower() in [".jpg", ".jpeg"]:
            DFLImage.save(output_path, resized, metadata, quality=jpeg_quality)
        elif output_path.suffix.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(
                str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
        else:
            cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 4])

        return True

    except Exception as e:
        logger.error(f"Error resizing {input_path}: {e}")
        return False


def resize_faceset(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    target_size: int = 256,
    target_face_type: str | None = None,
    interpolation: Literal["lanczos", "cubic", "linear", "nearest"] = "lanczos",
    preserve_metadata: bool = True,
    num_workers: int = 4,
    jpeg_quality: int = 95,
) -> ResizeResult:
    """Resize all faces in a faceset with metadata preservation.

    Batch processes all face images in a directory, resizing them to
    the target size while preserving DFL metadata.

    Args:
        input_dir: Directory containing face images
        output_dir: Output directory (None = create _resized suffix)
        target_size: Target image size (width = height)
        target_face_type: Target face type (None = keep same)
        interpolation: Interpolation method
        preserve_metadata: Preserve DFL metadata
        num_workers: Number of parallel workers
        jpeg_quality: JPEG quality for output

    Returns:
        ResizeResult with processing statistics

    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Determine output directory
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_{target_size}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Map interpolation string to OpenCV constant
    interp_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    cv_interpolation = interp_map.get(interpolation, cv2.INTER_LANCZOS4)

    # Find all image files
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return ResizeResult(
            total_images=0,
            resized_count=0,
            skipped_count=0,
            error_count=0,
            output_dir=output_dir,
        )

    resized_count = 0
    skipped_count = 0
    error_count = 0

    def process_file(image_path: Path) -> tuple[bool, bool]:
        """Process a single file. Returns (success, skipped)."""
        output_path = output_dir / image_path.name

        # Check if already correct size
        img = cv2.imread(str(image_path))
        if img is not None and img.shape[0] == target_size and target_face_type is None:
            # Just copy if already correct size and no face type change
            if input_dir != output_dir:
                import shutil

                shutil.copy2(image_path, output_path)
            return True, True

        success = resize_single_face(
            image_path,
            output_path,
            target_size,
            target_face_type=target_face_type,
            interpolation=cv_interpolation,
            preserve_metadata=preserve_metadata,
            jpeg_quality=jpeg_quality,
        )
        return success, False

    # Process files with parallel workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, img): img for img in image_files}

        for future in tqdm(
            as_completed(futures),
            total=len(image_files),
            desc=f"Resizing to {target_size}",
        ):
            try:
                success, skipped = future.result()
                if success:
                    if skipped:
                        skipped_count += 1
                    else:
                        resized_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                error_count += 1

    return ResizeResult(
        total_images=len(image_files),
        resized_count=resized_count,
        skipped_count=skipped_count,
        error_count=error_count,
        output_dir=output_dir,
    )


def main() -> None:
    """CLI entry point for faceset resizing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Resize faceset images with metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input faceset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: input_<size>)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Target size (default: 256)",
    )
    parser.add_argument(
        "--face-type",
        type=str,
        choices=["half_face", "mid_face", "full_face", "whole_face", "head"],
        default=None,
        help="Target face type (default: keep same)",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["lanczos", "cubic", "linear", "nearest"],
        default="lanczos",
        help="Interpolation method",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't preserve DFL metadata",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100)",
    )

    args = parser.parse_args()

    print(f"Resizing faceset: {args.input}")
    print(f"Target size: {args.size}")
    if args.face_type:
        print(f"Target face type: {args.face_type}")
    print(f"Interpolation: {args.interpolation}")

    result = resize_faceset(
        args.input,
        args.output,
        target_size=args.size,
        target_face_type=args.face_type,
        interpolation=args.interpolation,
        preserve_metadata=not args.no_metadata,
        num_workers=args.workers,
        jpeg_quality=args.quality,
    )

    print("\nResize complete:")
    print(f"  Total images: {result.total_images}")
    print(f"  Resized: {result.resized_count}")
    print(f"  Skipped (already correct size): {result.skipped_count}")
    print(f"  Errors: {result.error_count}")
    print(f"  Output: {result.output_dir}")


if __name__ == "__main__":
    main()
