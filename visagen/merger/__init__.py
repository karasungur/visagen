"""
Visagen Merger - Video face swap pipeline.

This module provides tools for merging face swaps into videos:
- VideoReader/VideoWriter: FFmpeg-based video I/O
- FrameProcessor: Single-frame face swap processing
- FaceMerger: High-level video processing orchestration
- BatchProcessor: Parallel frame processing
- InteractiveMerger: Real-time preview with parameter adjustment

Example:
    >>> from visagen.merger import FaceMerger, MergerConfig
    >>> config = MergerConfig(
    ...     input_path=Path("input.mp4"),
    ...     output_path=Path("output.mp4"),
    ...     checkpoint_path=Path("model.ckpt"),
    ... )
    >>> merger = FaceMerger(config)
    >>> stats = merger.run()

Interactive Example:
    >>> from visagen.merger import InteractiveMerger
    >>> merger = InteractiveMerger("model.ckpt", "frames/", "output/")
    >>> merger.load_session()
    >>> preview = merger.process_current_frame()
    >>> merger.update_config(erode_mask=10, blur_mask=20)
"""

from visagen.merger.batch_processor import (
    BatchProcessor,
    WorkItem,
    WorkResult,
)
from visagen.merger.frame_processor import (
    FrameProcessor,
    FrameProcessorConfig,
    ProcessedFrame,
)
from visagen.merger.interactive import InteractiveMerger
from visagen.merger.interactive_config import (
    COLOR_TRANSFER_MODES,
    MASK_MODES,
    MERGE_MODES,
    SHARPEN_MODES,
    InteractiveMergerConfig,
    InteractiveMergerSession,
)
from visagen.merger.merger import (
    FaceMerger,
    MergerConfig,
    MergerStats,
)
from visagen.merger.video_io import (
    EncoderConfig,
    VideoInfo,
    VideoReader,
    VideoWriter,
    check_nvenc_available,
    extract_frames_to_dir,
    get_available_encoders,
    probe_video,
    select_best_encoder,
    video_from_frames,
)

__all__ = [
    # Video I/O
    "VideoInfo",
    "VideoReader",
    "VideoWriter",
    "probe_video",
    "extract_frames_to_dir",
    "video_from_frames",
    # Encoder configuration
    "EncoderConfig",
    "check_nvenc_available",
    "get_available_encoders",
    "select_best_encoder",
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
    # Interactive merger
    "InteractiveMerger",
    "InteractiveMergerConfig",
    "InteractiveMergerSession",
    "MERGE_MODES",
    "MASK_MODES",
    "COLOR_TRANSFER_MODES",
    "SHARPEN_MODES",
]
