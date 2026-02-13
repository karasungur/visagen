"""
Metadata-based sorting methods.

Provides sorting by DFL metadata fields like original filename,
source rect size, and face count filtering.
"""

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visagen.sorting.base import SortMethod, SortOutput, SortResult

if TYPE_CHECKING:
    from visagen.sorting.processor import ParallelSortProcessor
    from visagen.vision.dflimg import FaceMetadata


class OrigNameSorter(SortMethod):
    """
    Sort by original source filename.

    Uses the source_filename field from DFL metadata to
    restore original ordering from video frames.
    """

    name = "origname"
    description = "Sort by original filename"
    requires_dfl_metadata = True
    execution_profile = "io_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Not used - custom sort extracts filename."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: "ParallelSortProcessor | None" = None,
    ) -> SortOutput:
        """Sort by original source filename from metadata."""
        from visagen.vision.dflimg import DFLImage

        start_time = time.time()

        results: list[tuple[Path, str]] = []
        trash: list[SortResult] = []

        for filepath in image_paths:
            try:
                _, metadata = DFLImage.load(filepath)
                if metadata is None:
                    trash.append(
                        SortResult(filepath, 0.0, {"error": "No DFL metadata"})
                    )
                    continue

                source_name = metadata.source_filename or filepath.stem
                results.append((filepath, source_name))
            except Exception as e:
                trash.append(SortResult(filepath, 0.0, {"error": str(e)}))

        # Sort alphabetically by source filename
        results.sort(key=lambda x: x[1])

        sorted_results = [
            SortResult(path, float(i)) for i, (path, _) in enumerate(results)
        ]

        elapsed = time.time() - start_time
        return SortOutput(sorted_results, trash, self.name, elapsed)


class SourceRectSorter(SortMethod):
    """
    Sort by face size in source image.

    Uses source_rect area from DFL metadata.
    Larger faces (closer to camera) rank first.
    """

    name = "face-source-rect-size"
    description = "Sort by face rect size in source image"
    requires_dfl_metadata = True
    execution_profile = "io_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute source rect area."""
        if metadata is None or metadata.source_rect is None:
            return 0.0

        x1, y1, x2, y2 = metadata.source_rect
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return float(width * height)


class OneFaceSorter(SortMethod):
    """
    Filter images to keep only those from single-face frames.

    Identifies multi-face frames by filename pattern (e.g., "00001_0", "00001_1")
    and moves all faces from multi-face frames to trash.

    Useful for cleaning datasets where multiple faces were detected
    in the same source frame.
    """

    name = "oneface"
    description = "Filter to keep only one face per source image"
    requires_dfl_metadata = False
    execution_profile = "io_bound"

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Not used - custom sort logic."""
        return 0.0

    def sort(
        self,
        image_paths: list[Path],
        processor: "ParallelSortProcessor | None" = None,
    ) -> SortOutput:
        """
        Filter to images from single-face source frames.

        Filename pattern: "00001_0.jpg" indicates frame 1, face 0.
        If any frame has face index > 0, all faces from that frame go to trash.
        """
        start_time = time.time()

        # Pattern: digits_digits (frame_faceindex)
        pattern = re.compile(r"^(\d+)_(\d+)$")

        # Parse filenames
        frame_faces: dict[str, list[Path]] = {}
        other_files: list[Path] = []

        for filepath in image_paths:
            match = pattern.match(filepath.stem)
            if match:
                frame_id = match.group(1)
                if frame_id not in frame_faces:
                    frame_faces[frame_id] = []
                frame_faces[frame_id].append(filepath)
            else:
                other_files.append(filepath)

        # Separate single-face and multi-face frames
        sorted_images: list[SortResult] = []
        trash_images: list[SortResult] = []

        for _frame_id, faces in frame_faces.items():
            if len(faces) == 1:
                # Single face - keep
                sorted_images.append(SortResult(faces[0], 0.0))
            else:
                # Multiple faces - trash all
                for face_path in faces:
                    trash_images.append(
                        SortResult(face_path, 0.0, {"reason": "multi-face frame"})
                    )

        # Add files that don't match pattern (assume single face)
        for filepath in other_files:
            sorted_images.append(SortResult(filepath, 0.0))

        # Sort by filename
        sorted_images.sort(key=lambda x: x.filepath.stem)

        elapsed = time.time() - start_time
        return SortOutput(sorted_images, trash_images, self.name, elapsed)
