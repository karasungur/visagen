"""
Visagen Vision - Face Detection, Alignment, and Segmentation Pipeline.

This module provides modern computer vision capabilities:
- Face detection via InsightFace SCRFD
- 3D landmark extraction via AntelopeV2
- Face alignment using Umeyama transform
- Face segmentation via SegFormer
"""

from visagen.vision.face_type import FaceType
from visagen.vision.aligner import FaceAligner
from visagen.vision.dflimg import DFLImage, FaceMetadata

# Lazy imports for optional dependencies
_FaceDetector = None
_FaceSegmenter = None


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    global _FaceDetector, _FaceSegmenter

    if name == "FaceDetector":
        if _FaceDetector is None:
            from visagen.vision.detector import FaceDetector as _FD
            _FaceDetector = _FD
        return _FaceDetector

    if name == "FaceSegmenter":
        if _FaceSegmenter is None:
            from visagen.vision.segmenter import FaceSegmenter as _FS
            _FaceSegmenter = _FS
        return _FaceSegmenter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FaceType",
    "FaceDetector",
    "FaceAligner",
    "FaceSegmenter",
    "DFLImage",
    "FaceMetadata",
]
