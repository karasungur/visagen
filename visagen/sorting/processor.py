"""
Parallel processing wrapper for sorting operations.

Provides efficient multi-core processing for image scoring.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visagen.sorting.base import SortResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Result from processing a single image."""

    filepath: Path
    image: np.ndarray | None
    sharpness: float
    yaw: float
    pitch: float
    histogram: np.ndarray | None
    error: str | None = None


def _load_and_process_single(
    filepath: Path,
    compute_sharpness: bool = True,
    compute_pose: bool = True,
    compute_histogram: bool = True,
) -> ProcessedImage:
    """
    Load and process a single image.

    This function is designed to be called in a subprocess.

    Args:
        filepath: Path to image file.
        compute_sharpness: Whether to compute sharpness score.
        compute_pose: Whether to compute pose angles.
        compute_histogram: Whether to compute histogram.

    Returns:
        ProcessedImage with computed values.
    """
    try:
        from visagen.sorting.blur import estimate_sharpness, get_face_hull_mask
        from visagen.sorting.histogram import compute_grayscale_histogram
        from visagen.vision.aligner import FaceAligner
        from visagen.vision.dflimg import DFLImage

        # Load image with metadata
        image, metadata = DFLImage.load(filepath)

        if image is None:
            return ProcessedImage(
                filepath=filepath,
                image=None,
                sharpness=0.0,
                yaw=0.0,
                pitch=0.0,
                histogram=None,
                error="Failed to load image",
            )

        # Apply face mask if available
        masked_image = image
        if metadata is not None and metadata.landmarks is not None:
            try:
                mask = get_face_hull_mask(image.shape, metadata.landmarks)
                masked_image = (image * mask).astype(np.uint8)
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        # Compute sharpness
        sharpness = 0.0
        if compute_sharpness:
            sharpness = estimate_sharpness(masked_image)

        # Compute pose
        yaw, pitch = 0.0, 0.0
        if compute_pose and metadata is not None and metadata.landmarks is not None:
            try:
                aligner = FaceAligner()
                size = image.shape[0]
                p, y, r = aligner.estimate_pitch_yaw_roll(metadata.landmarks, size=size)
                pitch, yaw = p, y
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        # Compute histogram
        histogram = None
        if compute_histogram:
            histogram = compute_grayscale_histogram(masked_image)

        return ProcessedImage(
            filepath=filepath,
            image=image,
            sharpness=sharpness,
            yaw=yaw,
            pitch=pitch,
            histogram=histogram,
        )

    except Exception as e:
        return ProcessedImage(
            filepath=filepath,
            image=None,
            sharpness=0.0,
            yaw=0.0,
            pitch=0.0,
            histogram=None,
            error=str(e),
        )


class ParallelSortProcessor:
    """
    Parallel image processing wrapper for sorting operations.

    Uses ThreadPoolExecutor for I/O-bound operations (loading images)
    to avoid pickling issues with ProcessPoolExecutor.

    Args:
        max_workers: Number of parallel workers. Default: CPU count.
        use_threads: Use threads instead of processes. Default: True.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        use_threads: bool = True,
    ) -> None:
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)
        self.use_threads = use_threads

    def process_images(
        self,
        image_paths: list[Path],
        compute_fn: Callable[[Path], SortResult],
        desc: str = "Processing",
        show_progress: bool = True,
    ) -> list[SortResult]:
        """
        Process images in parallel.

        Args:
            image_paths: List of image paths to process.
            compute_fn: Function to compute score for each image.
            desc: Description for progress bar.
            show_progress: Whether to show progress bar.

        Returns:
            List of SortResult objects.
        """
        results: list[SortResult] = []

        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        with ExecutorClass(max_workers=self.max_workers) as executor:
            futures = {executor.submit(compute_fn, p): p for p in image_paths}

            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=desc,
                    )
                except ImportError:
                    iterator = as_completed(futures)
            else:
                iterator = as_completed(futures)

            for future in iterator:
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    results.append(SortResult(path, 0.0, {"error": str(e)}))

        return results

    def load_and_process_all(
        self,
        image_paths: list[Path],
        compute_sharpness: bool = True,
        compute_pose: bool = True,
        compute_histogram: bool = True,
        show_progress: bool = True,
    ) -> list[ProcessedImage]:
        """
        Load and process all images with full metrics.

        Used by FinalSorter for comprehensive analysis.

        Args:
            image_paths: List of image paths.
            compute_sharpness: Compute sharpness scores.
            compute_pose: Compute pose angles.
            compute_histogram: Compute histograms.
            show_progress: Show progress bar.

        Returns:
            List of ProcessedImage objects.
        """
        results: list[ProcessedImage] = []

        def process_single(path: Path) -> ProcessedImage:
            return _load_and_process_single(
                path,
                compute_sharpness=compute_sharpness,
                compute_pose=compute_pose,
                compute_histogram=compute_histogram,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, p): p for p in image_paths}

            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Loading",
                    )
                except ImportError:
                    iterator = as_completed(futures)
            else:
                iterator = as_completed(futures)

            for future in iterator:
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    results.append(
                        ProcessedImage(
                            filepath=path,
                            image=None,
                            sharpness=0.0,
                            yaw=0.0,
                            pitch=0.0,
                            histogram=None,
                            error=str(e),
                        )
                    )

        return results
