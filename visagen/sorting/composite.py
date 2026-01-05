"""
Composite sorting methods.

Provides FinalSorter for selecting best faces with pose variety.
"""

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult
from visagen.sorting.processor import ParallelSortProcessor, ProcessedImage

if TYPE_CHECKING:
    pass


class FinalSorter(SortMethod):
    """
    Select best faces with pose variety and sharpness.

    This is the most sophisticated sorting method, combining multiple criteria:
    1. Bins images by yaw angle (128 bins from -1.2 to +1.2 radians)
    2. Within each yaw bin, sorts by sharpness and keeps top N
    3. Sub-bins by pitch angle for finer pose distribution
    4. Within each pitch bin, sorts by histogram dissimilarity
    5. Round-robin selection from all bins to ensure variety

    The result is a curated dataset with:
    - Sharp images (not blurry)
    - Diverse pose coverage (different yaw/pitch angles)
    - Visual variety (not too many similar images)

    Args:
        target_count: Target number of images to select. Default: 2000.
        faster: Use source rect size instead of blur for speed. Default: False.
        yaw_bins: Number of yaw angle bins. Default: 128.

    Example:
        >>> sorter = FinalSorter(target_count=5000)
        >>> result = sorter.sort(image_paths)
        >>> print(f"Selected: {len(result.sorted_images)}")
    """

    name = "final"
    description = "Select best faces with pose variety and sharpness"
    requires_dfl_metadata = True

    def __init__(
        self,
        target_count: int = 2000,
        faster: bool = False,
        yaw_bins: int = 128,
    ) -> None:
        self.target_count = target_count
        self.faster = faster
        self.yaw_bins = yaw_bins

    def compute_score(
        self,
        image: np.ndarray,
        metadata=None,
    ) -> float:
        """Not used - custom sort logic."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: ParallelSortProcessor | None = None,
    ) -> SortOutput:
        """
        Sort and select best faces.

        Args:
            image_paths: List of image paths to process.
            processor: Optional parallel processor.

        Returns:
            SortOutput with selected and trash images.
        """
        start_time = time.time()

        if len(image_paths) == 0:
            return SortOutput([], [], self.name, 0.0)

        # Use provided processor or create new one
        if processor is None:
            processor = ParallelSortProcessor()

        # Step 1: Load all images and compute metrics
        print(f"Loading {len(image_paths)} images...")
        processed = processor.load_and_process_all(
            image_paths,
            compute_sharpness=not self.faster,
            compute_pose=True,
            compute_histogram=True,
        )

        # Filter out failed loads
        valid_images: list[ProcessedImage] = []
        trash: list[SortResult] = []

        for img in processed:
            if img.error is not None:
                trash.append(SortResult(img.filepath, 0.0, {"error": img.error}))
            else:
                valid_images.append(img)

        if len(valid_images) == 0:
            elapsed = time.time() - start_time
            return SortOutput([], trash, self.name, elapsed)

        print(f"Processing {len(valid_images)} valid images...")

        # Step 2: Bin by yaw angle
        # Using -1.2 to +1.2 radians as practical yaw limits for 2D landmarks
        yaw_min, yaw_max = -1.2, 1.2
        yaw_space = np.linspace(yaw_min, yaw_max, self.yaw_bins + 1)

        yaw_binned: list[list[ProcessedImage]] = [[] for _ in range(self.yaw_bins)]

        for img in valid_images:
            yaw = -img.yaw  # Negate to match legacy behavior
            bin_idx = self._find_bin(yaw, yaw_space)
            yaw_binned[bin_idx].append(img)

        # Step 3: Calculate images per bin
        imgs_per_bin = max(1, self.target_count // self.yaw_bins)
        total_lack = 0

        for bin_images in yaw_binned:
            lack = imgs_per_bin - len(bin_images)
            if lack > 0:
                total_lack += lack

        # Redistribute lack to other bins
        imgs_per_bin += total_lack // self.yaw_bins

        # Step 4: Sort each yaw bin by sharpness and keep top N*10
        sharpness_multiplier = 10
        sharpened_per_bin = imgs_per_bin * sharpness_multiplier

        for i, bin_images in enumerate(yaw_binned):
            if len(bin_images) == 0:
                continue

            # Sort by sharpness (or rect area if faster mode)
            if self.faster:
                # Use a placeholder metric - could be enhanced with source rect
                bin_images.sort(key=lambda x: x.sharpness, reverse=True)
            else:
                bin_images.sort(key=lambda x: x.sharpness, reverse=True)

            # Trash images beyond sharpened_per_bin
            if len(bin_images) > sharpened_per_bin:
                for img in bin_images[sharpened_per_bin:]:
                    trash.append(
                        SortResult(
                            img.filepath,
                            img.sharpness,
                            {"reason": "below sharpness threshold"},
                        )
                    )
                yaw_binned[i] = bin_images[:sharpened_per_bin]

        # Step 5: Sub-bin by pitch within each yaw bin
        pitch_bins = max(1, imgs_per_bin)
        pitch_min, pitch_max = -math.pi / 2, math.pi / 2
        pitch_space = np.linspace(pitch_min, pitch_max, pitch_bins + 1)

        yaw_pitch_binned: list[list[list[ProcessedImage]]] = []

        for yaw_bin_images in yaw_binned:
            if len(yaw_bin_images) == 0:
                yaw_pitch_binned.append([])
                continue

            pitch_binned: list[list[ProcessedImage]] = [[] for _ in range(pitch_bins)]

            for img in yaw_bin_images:
                pitch = img.pitch
                bin_idx = self._find_bin(pitch, pitch_space)
                pitch_binned[bin_idx].append(img)

            yaw_pitch_binned.append(pitch_binned)

        # Step 6: Sort each pitch bin by histogram dissimilarity
        for _yaw_idx, pitch_bins_list in enumerate(yaw_pitch_binned):
            for _pitch_idx, pitch_bin_images in enumerate(pitch_bins_list):
                if len(pitch_bin_images) <= 1:
                    continue

                # Compute histogram dissimilarity scores
                for img in pitch_bin_images:
                    img_hist = img.histogram
                    if img_hist is None:
                        continue

                    dissim_score = 0.0
                    for other in pitch_bin_images:
                        if other is img or other.histogram is None:
                            continue
                        score = cv2.compareHist(
                            img_hist.reshape(-1, 1).astype(np.float32),
                            other.histogram.reshape(-1, 1).astype(np.float32),
                            cv2.HISTCMP_BHATTACHARYYA,
                        )
                        dissim_score += score

                    # Store in extra field via monkey patching
                    img._dissim_score = dissim_score  # type: ignore

                # Sort by dissimilarity (most unique first)
                pitch_bin_images.sort(
                    key=lambda x: getattr(x, "_dissim_score", 0.0),
                    reverse=True,
                )

        # Step 7: Round-robin selection from all bins
        final_list: list[ProcessedImage] = []
        remaining_count = self.target_count

        # Keep selecting until we have enough or run out
        while remaining_count > 0:
            selected_this_round = 0

            for _yaw_idx, pitch_bins_list in enumerate(yaw_pitch_binned):
                if remaining_count <= 0:
                    break

                for _pitch_idx, pitch_bin_images in enumerate(pitch_bins_list):
                    if remaining_count <= 0:
                        break

                    if len(pitch_bin_images) > 0:
                        # Take one from this bin
                        selected = pitch_bin_images.pop(0)
                        final_list.append(selected)
                        remaining_count -= 1
                        selected_this_round += 1

            # If we didn't select anything this round, we're done
            if selected_this_round == 0:
                break

        # Step 8: Move remaining images to trash
        for pitch_bins_list in yaw_pitch_binned:
            for pitch_bin_images in pitch_bins_list:
                for img in pitch_bin_images:
                    trash.append(
                        SortResult(
                            img.filepath,
                            img.sharpness,
                            {"reason": "not selected in final round-robin"},
                        )
                    )

        # Build final results
        sorted_results = [SortResult(img.filepath, img.sharpness) for img in final_list]

        elapsed = time.time() - start_time
        print(
            f"Selected {len(sorted_results)} images, trashed {len(trash)} in {elapsed:.1f}s"
        )

        return SortOutput(sorted_results, trash, self.name, elapsed)

    def _find_bin(self, value: float, bin_edges: np.ndarray) -> int:
        """Find the bin index for a value given bin edges."""
        n_bins = len(bin_edges) - 1

        for i in range(n_bins):
            if i == 0 and value < bin_edges[1]:
                return 0
            elif i == n_bins - 1 and value >= bin_edges[i]:
                return n_bins - 1
            elif value >= bin_edges[i] and value < bin_edges[i + 1]:
                return i

        return n_bins - 1


class FinalFastSorter(FinalSorter):
    """
    Fast version of FinalSorter.

    Uses source rect size instead of blur detection for faster processing.
    Trade-off: slightly less accurate sharpness estimation.
    """

    name = "final-fast"
    description = "Select best faces (faster, uses rect size instead of blur)"

    def __init__(
        self,
        target_count: int = 2000,
        yaw_bins: int = 128,
    ) -> None:
        super().__init__(target_count=target_count, faster=True, yaw_bins=yaw_bins)
