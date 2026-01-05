"""
Visagen Sorting Module - Face image sorting and filtering.

This module provides various sorting methods for face datasets:
- Blur detection (sharpness estimation)
- Face pose sorting (yaw, pitch)
- Histogram-based similarity/dissimilarity
- Color-based sorting (brightness, hue)
- Composite "final" selection for best faces

Example:
    >>> from visagen.sorting import BlurSorter, ParallelSortProcessor
    >>> sorter = BlurSorter()
    >>> processor = ParallelSortProcessor()
    >>> result = sorter.sort(image_paths, processor)
"""

from visagen.sorting.base import SortMethod, SortOutput, SortResult
from visagen.sorting.blur import BlurSorter, MotionBlurSorter
from visagen.sorting.color import BlackPixelSorter, BrightnessSorter, HueSorter
from visagen.sorting.composite import FinalSorter
from visagen.sorting.histogram import (
    HistogramDissimilaritySorter,
    HistogramSimilaritySorter,
)
from visagen.sorting.metadata import OneFaceSorter, OrigNameSorter, SourceRectSorter
from visagen.sorting.pose import PitchSorter, YawSorter
from visagen.sorting.processor import ParallelSortProcessor

__all__ = [
    # Base
    "SortMethod",
    "SortResult",
    "SortOutput",
    # Blur
    "BlurSorter",
    "MotionBlurSorter",
    # Pose
    "YawSorter",
    "PitchSorter",
    # Histogram
    "HistogramSimilaritySorter",
    "HistogramDissimilaritySorter",
    # Color
    "BrightnessSorter",
    "HueSorter",
    "BlackPixelSorter",
    # Metadata
    "OrigNameSorter",
    "SourceRectSorter",
    "OneFaceSorter",
    # Composite
    "FinalSorter",
    # Processor
    "ParallelSortProcessor",
]
