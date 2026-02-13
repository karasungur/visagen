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
    source_rect_area: float
    error: str | None = None


def _load_and_process_single(
    filepath: Path,
    compute_sharpness: bool = True,
    compute_pose: bool = True,
    compute_histogram: bool = True,
    histogram_mode: str = "gray",
    compute_source_rect: bool = False,
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
        from visagen.sorting.histogram import (
            compute_grayscale_histogram,
        )
        from visagen.sorting.histogram import (
            compute_histogram as compute_color_histogram,
        )
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
                source_rect_area=0.0,
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
            if histogram_mode == "color":
                histogram = compute_color_histogram(masked_image)
            else:
                histogram = compute_grayscale_histogram(masked_image)

        source_rect_area = 0.0
        if (
            compute_source_rect
            and metadata is not None
            and metadata.source_rect is not None
        ):
            try:
                x1, y1, x2, y2 = metadata.source_rect
                source_rect_area = float(abs(x2 - x1) * abs(y2 - y1))
            except Exception as e:
                logger.debug(f"Optional operation failed: {e}")

        return ProcessedImage(
            filepath=filepath,
            image=image,
            sharpness=sharpness,
            yaw=yaw,
            pitch=pitch,
            histogram=histogram,
            source_rect_area=source_rect_area,
        )

    except Exception as e:
        return ProcessedImage(
            filepath=filepath,
            image=None,
            sharpness=0.0,
            yaw=0.0,
            pitch=0.0,
            histogram=None,
            source_rect_area=0.0,
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
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.use_threads = use_threads

    def process_images(
        self,
        image_paths: list[Path],
        compute_fn: Callable[[Path], SortResult],
        desc: str = "Processing",
        show_progress: bool = True,
        use_threads: bool | None = None,
    ) -> list[SortResult]:
        """
        Process images in parallel.

        Args:
            image_paths: List of image paths to process.
            compute_fn: Function to compute score for each image.
            desc: Description for progress bar.
            show_progress: Whether to show progress bar.
            use_threads: Override processor mode for this call.

        Returns:
            List of SortResult objects.
        """
        results_by_index: list[SortResult | None] = [None] * len(image_paths)
        run_on_threads = self.use_threads if use_threads is None else use_threads
        executor_class = ThreadPoolExecutor if run_on_threads else ProcessPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compute_fn, path): idx
                for idx, path in enumerate(image_paths)
            }

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
                idx = futures[future]
                path = image_paths[idx]
                try:
                    result = future.result()
                    results_by_index[idx] = result
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    results_by_index[idx] = SortResult(path, 0.0, {"error": str(e)})

        return [
            result
            if result is not None
            else SortResult(path, 0.0, {"error": "Unknown processing failure"})
            for path, result in zip(image_paths, results_by_index, strict=True)
        ]

    def load_and_process_all(
        self,
        image_paths: list[Path],
        compute_sharpness: bool = True,
        compute_pose: bool = True,
        compute_histogram: bool = True,
        histogram_mode: str = "gray",
        compute_source_rect: bool = False,
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
            histogram_mode: Histogram mode ("gray" or "color").
            show_progress: Show progress bar.

        Returns:
            List of ProcessedImage objects.
        """
        results_by_index: list[ProcessedImage | None] = [None] * len(image_paths)

        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        with ExecutorClass(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _load_and_process_single,
                    p,
                    compute_sharpness,
                    compute_pose,
                    compute_histogram,
                    histogram_mode,
                    compute_source_rect,
                ): idx
                for idx, p in enumerate(image_paths)
            }

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
                idx = futures[future]
                path = image_paths[idx]
                try:
                    result = future.result()
                    results_by_index[idx] = result
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    results_by_index[idx] = ProcessedImage(
                        filepath=path,
                        image=None,
                        sharpness=0.0,
                        yaw=0.0,
                        pitch=0.0,
                        histogram=None,
                        source_rect_area=0.0,
                        error=str(e),
                    )

        return [
            result
            if result is not None
            else ProcessedImage(
                filepath=path,
                image=None,
                sharpness=0.0,
                yaw=0.0,
                pitch=0.0,
                histogram=None,
                source_rect_area=0.0,
                error="Unknown processing failure",
            )
            for path, result in zip(image_paths, results_by_index, strict=True)
        ]
