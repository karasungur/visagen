"""
Base classes for sorting methods.

Provides abstract base class for sort methods and data classes for results.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from visagen.vision.dflimg import FaceMetadata


@dataclass
class SortResult:
    """
    Result for a single sorted image.

    Attributes:
        filepath: Path to the image file.
        score: Computed score for sorting.
        metadata: Optional additional metadata dict.
    """

    filepath: Path
    score: float
    metadata: dict | None = None

    def __lt__(self, other: "SortResult") -> bool:
        """Compare by score for sorting."""
        return self.score < other.score


@dataclass
class SortOutput:
    """
    Output from a sorting operation.

    Attributes:
        sorted_images: List of images in sorted order.
        trash_images: List of images marked for trash/removal.
        method: Name of the sorting method used.
        elapsed_seconds: Time taken for sorting.
    """

    sorted_images: list[SortResult]
    trash_images: list[SortResult]
    method: str
    elapsed_seconds: float


@dataclass
class ImageData:
    """
    Internal data structure for processing images.

    Holds image path, loaded image, metadata, and computed values.
    """

    filepath: Path
    image: np.ndarray | None = None
    metadata: "FaceMetadata | None" = None
    score: float = 0.0
    sharpness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    histogram: np.ndarray | None = None
    hist_dissim_score: float = 0.0
    extra: dict = field(default_factory=dict)


def _compute_sort_result(sorter: "SortMethod", filepath: Path) -> SortResult:
    """
    Compute a single sort result in a process-safe way.

    This is a top-level function so it can be pickled when using
    ProcessPoolExecutor.
    """
    import cv2

    from visagen.vision.dflimg import DFLImage

    try:
        # Load image and metadata
        image: np.ndarray | None
        if sorter.requires_dfl_metadata:
            image, metadata = DFLImage.load(filepath)
            if metadata is None:
                return SortResult(filepath, 0.0, {"error": "No DFL metadata"})
        else:
            image = cv2.imread(str(filepath))
            metadata = None

        if image is None:
            return SortResult(filepath, 0.0, {"error": "Failed to load image"})

        score = sorter.compute_score(image, metadata)
        return SortResult(filepath, score)
    except Exception as e:
        return SortResult(filepath, 0.0, {"error": str(e)})


class SortMethod(ABC):
    """
    Abstract base class for sorting methods.

    Subclasses must implement:
    - name: Method name for CLI
    - description: Human-readable description
    - compute_score: Score computation for a single image

    Optional overrides:
    - reverse_sort: Whether higher scores are better (default: True)
    - requires_dfl_metadata: Whether DFL metadata is required
    - sort: Custom sorting logic (for methods needing all images)
    """

    name: str = "base"
    description: str = "Base sorting method"
    requires_dfl_metadata: bool = False

    @abstractmethod
    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """
        Compute score for a single image.

        Args:
            image: BGR image as numpy array.
            metadata: Optional DFL face metadata.

        Returns:
            Score value (higher = better by default).
        """
        pass

    @property
    def reverse_sort(self) -> bool:
        """
        Whether to sort in descending order (higher scores first).

        Returns:
            True for descending (default), False for ascending.
        """
        return True

    def sort(
        self,
        image_paths: list[Path],
        processor: "ParallelSortProcessor | None" = None,
    ) -> SortOutput:
        """
        Sort images using this method.

        Default implementation loads images, computes scores in parallel,
        and sorts by score. Override for custom logic.

        Args:
            image_paths: List of paths to sort.
            processor: Optional parallel processor.

        Returns:
            SortOutput with sorted and trash images.
        """
        import time

        start_time = time.time()

        results: list[SortResult] = []
        trash: list[SortResult] = []

        if processor is None:
            computed = [
                _compute_sort_result(self, filepath) for filepath in image_paths
            ]
        elif not processor.use_threads:
            computed = []
            with ProcessPoolExecutor(max_workers=processor.max_workers) as executor:
                futures = {
                    executor.submit(_compute_sort_result, self, filepath): filepath
                    for filepath in image_paths
                }
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        computed.append(future.result())
                    except Exception as e:
                        computed.append(SortResult(path, 0.0, {"error": str(e)}))
        else:
            computed = processor.process_images(
                image_paths=image_paths,
                compute_fn=lambda p: _compute_sort_result(self, p),
                desc=f"Sort:{self.name}",
                show_progress=True,
            )

        for item in computed:
            if item.metadata is not None and "error" in item.metadata:
                trash.append(item)
            else:
                results.append(item)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=self.reverse_sort)

        elapsed = time.time() - start_time
        return SortOutput(results, trash, self.name, elapsed)


# Forward reference for type hints
if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
