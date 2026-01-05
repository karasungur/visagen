"""
Post-processing modules for Visagen.

Includes color transfer, blending algorithms, and face restoration
for face merging and post-production workflows.

Color Transfer:
    - reinhard_color_transfer (RCT): LAB space statistics matching
    - linear_color_transfer (LCT): PCA/Cholesky covariance matching
    - color_transfer_sot (SOT): Sliced Optimal Transfer
    - color_transfer_mkl (MKL): Monge-Kantorovitch Linear mapping
    - color_transfer_idt (IDT): Iterative Distribution Transfer

Blending:
    - laplacian_pyramid_blend: Multi-band frequency-based fusion
    - poisson_blend: Gradient-domain seamless cloning
    - feather_blend: Simple alpha blending with feathered edges

Face Restoration:
    - FaceRestorer: GFPGAN-based face enhancement
    - restore_face: One-shot face restoration function
    - is_gfpgan_available: Check GFPGAN availability
"""

from visagen.postprocess.blending import (
    BlendMode,
    blend,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    dilate_mask,
    erode_mask,
    feather_blend,
    laplacian_pyramid_blend,
    poisson_blend,
    reconstruct_from_laplacian,
)
from visagen.postprocess.color_transfer import (
    ColorTransferMode,
    color_transfer,
    color_transfer_idt,
    color_transfer_mkl,
    color_transfer_sot,
    linear_color_transfer,
    reinhard_color_transfer,
)
from visagen.postprocess.restore import (
    FaceRestorer,
    RestoreConfig,
    is_gfpgan_available,
    restore_face,
)

__all__ = [
    # Color transfer
    "reinhard_color_transfer",
    "linear_color_transfer",
    "color_transfer_sot",
    "color_transfer_mkl",
    "color_transfer_idt",
    "color_transfer",
    "ColorTransferMode",
    # Blending
    "laplacian_pyramid_blend",
    "poisson_blend",
    "feather_blend",
    "blend",
    "erode_mask",
    "dilate_mask",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "reconstruct_from_laplacian",
    "BlendMode",
    # Face restoration
    "FaceRestorer",
    "RestoreConfig",
    "restore_face",
    "is_gfpgan_available",
]
