"""Faceset enhancement tool using GFPGAN.

Batch processes facesets to enhance quality using GFPGAN face restoration.
Useful for improving low-quality face extractions before training.

Features:
    - Parallel batch processing
    - DFL metadata preservation
    - Multiple restoration backends (GFPGAN)
    - Progress tracking

References:
    - Legacy DFL: mainscripts/FacesetEnhancer.py
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from visagen.postprocess.restore import FaceRestorer, RestoreConfig, is_gfpgan_available
from visagen.vision.dflimg import DFLImage

logger = logging.getLogger(__name__)


@dataclass
class EnhanceResult:
    """Result of faceset enhancement operation."""

    total_images: int
    enhanced_count: int
    skipped_count: int
    error_count: int
    output_dir: Path


def enhance_single_face(
    input_path: Path,
    output_path: Path,
    restorer: FaceRestorer,
    *,
    strength: float = 0.5,
    preserve_metadata: bool = True,
) -> bool:
    """Enhance a single face image.

    Args:
        input_path: Path to input face image
        output_path: Path to save enhanced image
        restorer: FaceRestorer instance
        strength: Restoration strength
        preserve_metadata: Whether to preserve DFL metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            logger.warning(f"Failed to read image: {input_path}")
            return False

        # Load DFL metadata if present
        metadata = None
        if preserve_metadata and input_path.suffix.lower() in [".jpg", ".jpeg"]:
            try:
                _, dfl_metadata = DFLImage.load(input_path)
                metadata = dfl_metadata
            except Exception as e:
                logger.debug(f"Failed to load metadata from {input_path}: {e}")

        # Enhance face
        enhanced = restorer.restore(image, strength=strength)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata if applicable
        if metadata is not None and output_path.suffix.lower() in [".jpg", ".jpeg"]:
            DFLImage.save(output_path, enhanced, metadata, quality=95)
        else:
            # Save without metadata
            if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(output_path), enhanced)

        return True

    except Exception as e:
        logger.error(f"Error enhancing {input_path}: {e}")
        return False


def enhance_faceset(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    backend: Literal["gfpgan"] = "gfpgan",
    strength: float = 0.5,
    model_version: float = 1.4,
    preserve_metadata: bool = True,
    num_workers: int = 1,
    device: str | None = None,
) -> EnhanceResult:
    """Enhance all faces in a faceset using GFPGAN.

    Batch processes all face images in a directory, applying GFPGAN
    face restoration to improve quality.

    Args:
        input_dir: Directory containing face images
        output_dir: Output directory (None = create _enhanced suffix)
        backend: Restoration backend ('gfpgan')
        strength: Restoration strength (0.0-1.0)
        model_version: GFPGAN model version (1.2, 1.3, 1.4)
        preserve_metadata: Preserve DFL metadata in output images
        num_workers: Number of parallel workers (1 recommended for GFPGAN)
        device: Device for inference ('cuda', 'cpu', None for auto)

    Returns:
        EnhanceResult with processing statistics

    Raises:
        FileNotFoundError: If input directory doesn't exist
        RuntimeError: If GFPGAN is not available
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not is_gfpgan_available():
        raise RuntimeError("GFPGAN is not installed. Install with: pip install gfpgan")

    # Determine output directory
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_enhanced"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return EnhanceResult(
            total_images=0,
            enhanced_count=0,
            skipped_count=0,
            error_count=0,
            output_dir=output_dir,
        )

    # Create restorer
    config = RestoreConfig(
        enabled=True,
        mode=backend,
        strength=strength,
        model_version=model_version,
    )
    restorer = FaceRestorer(config, device=device)

    # Force model loading before parallel processing
    _ = restorer.gfpgan

    enhanced_count = 0
    error_count = 0

    # Process images
    # Note: Using single worker is recommended for GFPGAN as it's GPU-bound
    # and multiple workers don't provide speedup
    for image_path in tqdm(image_files, desc="Enhancing faces"):
        output_path = output_dir / image_path.name
        success = enhance_single_face(
            image_path,
            output_path,
            restorer,
            strength=strength,
            preserve_metadata=preserve_metadata,
        )
        if success:
            enhanced_count += 1
        else:
            error_count += 1

    return EnhanceResult(
        total_images=len(image_files),
        enhanced_count=enhanced_count,
        skipped_count=0,
        error_count=error_count,
        output_dir=output_dir,
    )


def main() -> None:
    """CLI entry point for faceset enhancement."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhance faceset using GFPGAN face restoration",
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
        help="Output directory (default: input_enhanced)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gfpgan"],
        default="gfpgan",
        help="Restoration backend",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="Restoration strength (0.0-1.0)",
    )
    parser.add_argument(
        "--model-version",
        type=float,
        choices=[1.2, 1.3, 1.4],
        default=1.4,
        help="GFPGAN model version",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't preserve DFL metadata",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda, cpu)",
    )

    args = parser.parse_args()

    # Check GFPGAN availability
    if not is_gfpgan_available():
        print("Error: GFPGAN is not installed.")
        print("Install with: pip install gfpgan")
        return

    print(f"Enhancing faceset: {args.input}")
    print(f"Backend: {args.backend}")
    print(f"Strength: {args.strength}")
    print(f"Model version: {args.model_version}")

    result = enhance_faceset(
        args.input,
        args.output,
        backend=args.backend,
        strength=args.strength,
        model_version=args.model_version,
        preserve_metadata=not args.no_metadata,
        device=args.device,
    )

    print("\nEnhancement complete:")
    print(f"  Total images: {result.total_images}")
    print(f"  Enhanced: {result.enhanced_count}")
    print(f"  Errors: {result.error_count}")
    print(f"  Output: {result.output_dir}")


if __name__ == "__main__":
    main()
