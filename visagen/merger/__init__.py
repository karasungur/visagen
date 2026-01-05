"""
Visagen Merger - Video face swap pipeline.

This module provides tools for merging face swaps into videos:
- VideoReader/VideoWriter: FFmpeg-based video I/O
- FrameProcessor: Single-frame face swap processing
- FaceMerger: High-level video processing orchestration
- BatchProcessor: Parallel frame processing

Example:
    >>> from visagen.merger import FaceMerger, MergerConfig
    >>> config = MergerConfig(
    ...     input_path=Path("input.mp4"),
    ...     output_path=Path("output.mp4"),
    ...     checkpoint_path=Path("model.ckpt"),
    ... )
    >>> merger = FaceMerger(config)
    >>> stats = merger.run()
"""

from visagen.merger.video_io import (
    VideoInfo,
    VideoReader,
    VideoWriter,
    probe_video,
    extract_frames_to_dir,
    video_from_frames,
)
from visagen.merger.frame_processor import (
    FrameProcessorConfig,
    ProcessedFrame,
    FrameProcessor,
)
from visagen.merger.merger import (
    MergerConfig,
    MergerStats,
    FaceMerger,
)
from visagen.merger.batch_processor import (
    WorkItem,
    WorkResult,
    BatchProcessor,
)

__all__ = [
    # Video I/O
    "VideoInfo",
    "VideoReader",
    "VideoWriter",
    "probe_video",
    "extract_frames_to_dir",
    "video_from_frames",
    # Frame processing
    "FrameProcessorConfig",
    "ProcessedFrame",
    "FrameProcessor",
    # Merger
    "MergerConfig",
    "MergerStats",
    "FaceMerger",
    # Batch processing
    "WorkItem",
    "WorkResult",
    "BatchProcessor",
]
