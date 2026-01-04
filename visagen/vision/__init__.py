"""
Visagen Vision - Face Detection, Alignment, and Segmentation Pipeline.

This module provides modern computer vision capabilities:
- Face detection via InsightFace SCRFD
- 3D landmark extraction via AntelopeV2
- Face alignment using Umeyama transform
- Face segmentation via SegFormer
"""

from visagen.vision.face_type import FaceType
from visagen.vision.detector import FaceDetector
from visagen.vision.aligner import FaceAligner
from visagen.vision.segmenter import FaceSegmenter
from visagen.vision.dflimg import DFLImage, FaceMetadata

__all__ = [
    "FaceType",
    "FaceDetector",
    "FaceAligner",
    "FaceSegmenter",
    "DFLImage",
    "FaceMetadata",
]
