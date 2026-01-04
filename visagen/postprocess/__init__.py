"""
Post-processing modules for Visagen.

Includes color transfer and blending algorithms for face merging
and post-production workflows.

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
"""

from visagen.postprocess.color_transfer import (
    reinhard_color_transfer,
    linear_color_transfer,
    color_transfer_sot,
    color_transfer_mkl,
    color_transfer_idt,
    color_transfer,
    ColorTransferMode,
)

from visagen.postprocess.blending import (
    laplacian_pyramid_blend,
    poisson_blend,
    feather_blend,
    blend,
    erode_mask,
    dilate_mask,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian,
    BlendMode,
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
]
