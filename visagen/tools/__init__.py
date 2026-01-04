"""
Visagen Tools - Command-line utilities for face processing.

This module provides CLI tools for:
- Face extraction from images and videos
- Dataset preparation and management
- Model training utilities
- Hyperparameter tuning
- Web interface
"""

# Lazy imports to avoid loading heavy dependencies at module import time
def __getattr__(name: str):
    """Lazy import for FaceExtractor and ExtractedFace."""
    if name == "FaceExtractor":
        from visagen.tools.extract_v2 import FaceExtractor
        return FaceExtractor
    elif name == "ExtractedFace":
        from visagen.tools.extract_v2 import ExtractedFace
        return ExtractedFace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FaceExtractor",
    "ExtractedFace",
]
