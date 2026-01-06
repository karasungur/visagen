"""
Experimental Models - Research-stage architectures.

This module contains experimental model architectures that are under
active development and may change significantly between versions.

Available Models:
- DiffusionAutoEncoder: Hybrid encoder combining ConvNeXt structure
  preservation with Stable Diffusion texture generation.
- EG3DGenerator: 3D geometry-aware generator using tri-plane
  representation for view-consistent face generation.

Note:
    These models require additional dependencies. Install with:
    pip install 'visagen[experimental]'

Example:
    >>> from visagen.models.experimental import DiffusionAutoEncoder
    >>> model = DiffusionAutoEncoder(image_size=256)
    >>> output = model(input_image)
"""

from __future__ import annotations

# Lazy imports to avoid loading heavy dependencies unless needed
__all__ = [
    # Diffusion AutoEncoder
    "DiffusionAutoEncoder",
    "TextureEncoder",
    "CrossAttentionFusion",
    "DiffusionDecoder",
    # EG3D
    "EG3DGenerator",
    "EG3DEncoder",
    "TriplaneGenerator",
    "NeRFDecoder",
    "VolumeRenderer",
    "SuperResolutionModule",
    "CameraParams",
]


def __getattr__(name: str):
    """Lazy import experimental modules."""
    if name in (
        "DiffusionAutoEncoder",
        "TextureEncoder",
        "CrossAttentionFusion",
        "DiffusionDecoder",
    ):
        from visagen.models.experimental.diffusion import (
            CrossAttentionFusion,
            DiffusionAutoEncoder,
            DiffusionDecoder,
            TextureEncoder,
        )

        return {
            "DiffusionAutoEncoder": DiffusionAutoEncoder,
            "TextureEncoder": TextureEncoder,
            "CrossAttentionFusion": CrossAttentionFusion,
            "DiffusionDecoder": DiffusionDecoder,
        }[name]

    if name in (
        "EG3DGenerator",
        "EG3DEncoder",
        "TriplaneGenerator",
        "NeRFDecoder",
        "VolumeRenderer",
        "SuperResolutionModule",
        "CameraParams",
    ):
        from visagen.models.experimental.eg3d import (
            CameraParams,
            EG3DEncoder,
            EG3DGenerator,
            NeRFDecoder,
            SuperResolutionModule,
            TriplaneGenerator,
            VolumeRenderer,
        )

        return {
            "EG3DGenerator": EG3DGenerator,
            "EG3DEncoder": EG3DEncoder,
            "TriplaneGenerator": TriplaneGenerator,
            "NeRFDecoder": NeRFDecoder,
            "VolumeRenderer": VolumeRenderer,
            "SuperResolutionModule": SuperResolutionModule,
            "CameraParams": CameraParams,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
