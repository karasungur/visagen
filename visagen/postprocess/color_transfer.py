"""
Color Transfer functions for post-processing.

Implements multiple color transfer algorithms for matching the color
distribution between source and target images:

- RCT (Reinhard Color Transfer): LAB space statistics matching
- LCT (Linear Color Transfer): PCA/Cholesky covariance matching
- SOT (Sliced Optimal Transfer): Iterative optimal transport
- MKL (Monge-Kantorovitch Linear): Covariance matrix matching

All functions expect float32 images in [0, 1] range with BGR channel order.

Reference:
    - Reinhard et al., "Color Transfer between Images" (2001)
    - https://github.com/dcoeurjo/OTColorTransfer
"""

from typing import Literal

import cv2
import numpy as np
from numpy import linalg as npla

ColorTransferMode = Literal[
    "none",
    "rct",
    "lct",
    "sot",
    "mkl",
    "idt",
    "mix",
    "neural",
    "mkl-masked",
    "idt-masked",
    "sot-masked",
]


def reinhard_color_transfer(
    target: np.ndarray,
    source: np.ndarray,
    target_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    mask_cutoff: float = 0.5,
) -> np.ndarray:
    """
    Transfer color using Reinhard Color Transfer (RCT) method.

    Matches the color distribution of target image to source image
    using LAB color space statistics (mean and standard deviation).

    Args:
        target: Target image to modify (H, W, 3) BGR float32 [0, 1].
        source: Source image to match colors from (H, W, 3) BGR float32 [0, 1].
        target_mask: Optional mask for target statistics (H, W, 1) [0, 1].
        source_mask: Optional mask for source statistics (H, W, 1) [0, 1].
        mask_cutoff: Threshold for mask application. Default: 0.5.

    Returns:
        Color-transferred target image (H, W, 3) BGR float32 [0, 1].
    """
    # Convert to LAB color space
    source_uint8 = np.clip(source * 255, 0, 255).astype(np.uint8)
    target_uint8 = np.clip(target * 255, 0, 255).astype(np.uint8)

    source_lab = cv2.cvtColor(source_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Get pixels for statistics
    if source_mask is not None:
        s_mask = source_mask.squeeze() > mask_cutoff
        if s_mask.any():
            s_pixels = source_lab[s_mask]
        else:
            s_pixels = source_lab.reshape(-1, 3)
    else:
        s_pixels = source_lab.reshape(-1, 3)

    if target_mask is not None:
        t_mask = target_mask.squeeze() > mask_cutoff
        if t_mask.any():
            t_pixels = target_lab[t_mask]
        else:
            t_pixels = target_lab.reshape(-1, 3)
    else:
        t_pixels = target_lab.reshape(-1, 3)

    # Compute statistics for each LAB channel
    eps = 1e-6

    src_l_mean, src_l_std = s_pixels[..., 0].mean(), s_pixels[..., 0].std()
    src_a_mean, src_a_std = s_pixels[..., 1].mean(), s_pixels[..., 1].std()
    src_b_mean, src_b_std = s_pixels[..., 2].mean(), s_pixels[..., 2].std()

    tgt_l_mean, tgt_l_std = t_pixels[..., 0].mean(), t_pixels[..., 0].std()
    tgt_a_mean, tgt_a_std = t_pixels[..., 1].mean(), t_pixels[..., 1].std()
    tgt_b_mean, tgt_b_std = t_pixels[..., 2].mean(), t_pixels[..., 2].std()

    # Transfer: scale by standard deviations, shift by means
    result_l = (target_lab[..., 0] - tgt_l_mean) * (
        src_l_std / (tgt_l_std + eps)
    ) + src_l_mean
    result_a = (target_lab[..., 1] - tgt_a_mean) * (
        src_a_std / (tgt_a_std + eps)
    ) + src_a_mean
    result_b = (target_lab[..., 2] - tgt_b_mean) * (
        src_b_std / (tgt_b_std + eps)
    ) + src_b_mean

    # Clip to valid LAB ranges (OpenCV LAB: L∈[0,100], A/B∈[-127,127] mapped to [0,255])
    # OpenCV uses L: 0-255 (scaled from 0-100), A/B: 0-255 (shifted from -128 to 127)
    # After cv2.cvtColor with uint8, the ranges are already [0,255]
    # Statistics are computed on these uint8 values, so clipping matches the format
    result_l = np.clip(result_l, 0, 255)
    result_a = np.clip(result_a, 0, 255)
    result_b = np.clip(result_b, 0, 255)

    # Convert back to BGR
    result_lab = np.stack([result_l, result_a, result_b], axis=-1).astype(np.uint8)
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    return np.clip(result, 0, 1)


def linear_color_transfer(
    target: np.ndarray,
    source: np.ndarray,
    mode: Literal["pca", "chol", "sym"] = "pca",
    target_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    mask_cutoff: float = 0.5,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Linear color transfer using covariance matching.

    Matches the color distribution using linear transformation
    based on covariance matrices of source and target.

    Args:
        target: Target image (H, W, 3) float32 [0, 1].
        source: Source image (H, W, 3) float32 [0, 1].
        mode: Transfer mode - 'pca', 'chol', or 'sym'. Default: 'pca'.
        target_mask: Optional mask for target statistics.
        source_mask: Optional mask for source statistics.
        mask_cutoff: Mask threshold. Default: 0.5.
        eps: Regularization epsilon. Default: 1e-5.

    Returns:
        Color-transferred image (H, W, 3) float32 [0, 1].
    """
    h, w, c = target.shape

    # Get pixels for statistics
    if source_mask is not None:
        s_mask = source_mask.squeeze() > mask_cutoff
        if s_mask.any():
            s_pixels = source[s_mask]
        else:
            s_pixels = source.reshape(-1, c)
    else:
        s_pixels = source.reshape(-1, c)

    if target_mask is not None:
        t_mask = target_mask.squeeze() > mask_cutoff
        if t_mask.any():
            t_pixels = target[t_mask]
        else:
            t_pixels = target.reshape(-1, c)
    else:
        t_pixels = target.reshape(-1, c)

    # Compute means
    mu_t = t_pixels.mean(axis=0)
    mu_s = s_pixels.mean(axis=0)

    # Center data
    t = (t_pixels - mu_t).T
    s = (s_pixels - mu_s).T

    # Compute covariance matrices
    Ct = np.cov(t) + eps * np.eye(c)
    Cs = np.cov(s) + eps * np.eye(c)

    # Compute transform for all target pixels
    # We apply the transform derived from masked pixels to the whole image
    # to avoid seams.
    target_flat = target.reshape(-1, c).T
    centered_target = target_flat - mu_t[:, None]

    if mode == "chol":
        # Cholesky decomposition based transfer
        chol_t = npla.cholesky(Ct)
        chol_s = npla.cholesky(Cs)
        ts = chol_s @ npla.inv(chol_t) @ centered_target

    elif mode == "pca":
        # PCA-based transfer (eigenvalue decomposition)
        eva_t, eve_t = npla.eigh(Ct)
        Qt = eve_t @ np.sqrt(np.diag(eva_t)) @ eve_t.T
        eva_s, eve_s = npla.eigh(Cs)
        Qs = eve_s @ np.sqrt(np.diag(eva_s)) @ eve_s.T
        ts = Qs @ npla.inv(Qt) @ centered_target

    elif mode == "sym":
        # Symmetric transfer
        eva_t, eve_t = npla.eigh(Ct)
        Qt = eve_t @ np.sqrt(np.diag(eva_t)) @ eve_t.T
        Qt_Cs_Qt = Qt @ Cs @ Qt
        eva_QtCsQt, eve_QtCsQt = npla.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt @ np.sqrt(np.diag(eva_QtCsQt)) @ eve_QtCsQt.T
        ts = npla.inv(Qt) @ QtCsQt @ npla.inv(Qt) @ centered_target

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pca', 'chol', or 'sym'.")

    # Reshape and add source mean
    result = ts.T.reshape(h, w, c) + mu_s

    return np.clip(result.astype(np.float32), 0, 1)


def color_transfer_sot(
    src: np.ndarray,
    trg: np.ndarray,
    steps: int = 10,
    batch_size: int = 5,
    reg_sigma_xy: float = 16.0,
    reg_sigma_v: float = 5.0,
    target_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    mask_cutoff: float = 0.5,
) -> np.ndarray:
    """
    Color Transfer via Sliced Optimal Transfer.

    Iteratively matches color distribution using random projections
    and optimal transport in 1D slices.

    Args:
        src: Source image (H, W, C) float32 [0, 1].
        trg: Target image (H, W, C) float32 [0, 1], same shape as src.
        steps: Number of solver iterations. Default: 10.
        batch_size: Number of random projections per step. Default: 5.
        reg_sigma_xy: Bilateral filter spatial sigma (0 to disable). Default: 16.0.
        reg_sigma_v: Bilateral filter value sigma. Default: 5.0.
        target_mask: Optional mask for target.
        source_mask: Optional mask for source.
        mask_cutoff: Mask threshold.

    Returns:
        Color-transferred source image (H, W, C) float32.
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src must be float type")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg must be float type")

    if src.shape != trg.shape:
        raise ValueError(
            f"src and trg must have same shape, got {src.shape} vs {trg.shape}"
        )

    h, w, c = src.shape
    new_src = src.copy()

    # Determine working pixels
    # For SOT, we modify pixels in-place. If masks are provided, we only
    # modify masked pixels and use masked target pixels for distribution matching.
    if source_mask is not None:
        s_mask = source_mask.squeeze() > mask_cutoff
        if not s_mask.any():
            return src  # No source pixels to modify
        flat_src = new_src[s_mask]  # (N_s, C)
    else:
        flat_src = new_src.reshape(-1, c)

    if target_mask is not None:
        t_mask = target_mask.squeeze() > mask_cutoff
        if t_mask.any():
            flat_trg = trg[t_mask]
        else:
            flat_trg = trg.reshape(-1, c)
    else:
        flat_trg = trg.reshape(-1, c)

    # If shapes mismatch (due to different mask sizes), we can still run SOT
    # because it matches distributions via 1D projections (independent of N).

    n_src = flat_src.shape[0]
    n_trg = flat_trg.shape[0]

    advect = np.empty((n_src, c), dtype=src.dtype)

    for _ in range(steps):
        advect.fill(0)

        for _ in range(batch_size):
            # Random projection direction
            direction = np.random.normal(size=c).astype(src.dtype)
            direction /= npla.norm(direction)

            # Project onto random direction
            proj_source = (flat_src * direction).sum(axis=-1)
            proj_target = (flat_trg * direction).sum(axis=-1)

            # Sort projections
            id_source = np.argsort(proj_source)
            id_target = np.argsort(proj_target)

            # Optimal transport in 1D: match sorted values
            # Handle different number of points by interpolation if needed
            # But standard SOT assumes matching counts or uses resampling?
            # Legacy implementation:
            # a = projtarget[idTarget]-projsource[idSource]
            # This assumes same size.

            # If sizes differ, we must resample target distribution to match source count
            if n_src != n_trg:
                # Interpolate target values to match source count
                # We want n_src quantiles from target distribution
                # Sorted proj_target represents the CDF
                x_trg = np.linspace(0, 1, n_trg)
                x_src = np.linspace(0, 1, n_src)
                sorted_trg = proj_target[id_target]
                matched_trg = np.interp(x_src, x_trg, sorted_trg)

                # Now we have n_src values sorted
                # proj_source[id_source] is sorted source
                a = matched_trg - proj_source[id_source]
            else:
                a = proj_target[id_target] - proj_source[id_source]

            # Accumulate advection
            for i_c in range(c):
                advect[id_source, i_c] += a * direction[i_c]

        # Update source
        flat_src += advect / batch_size

    # Put modified pixels back
    if source_mask is not None:
        new_src[s_mask] = flat_src
    else:
        new_src = flat_src.reshape(h, w, c)

    # Optional bilateral filtering for regularization
    if reg_sigma_xy > 0:
        src_diff = (new_src - src).astype(np.float32)
        src_diff_filt = cv2.bilateralFilter(src_diff, 0, reg_sigma_v, reg_sigma_xy)
        if src_diff_filt.ndim == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt

    return np.clip(new_src, 0, 1).astype(np.float32)


def color_transfer_mkl(
    target: np.ndarray,
    source: np.ndarray,
    target_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    mask_cutoff: float = 0.5,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Color Transfer using Monge-Kantorovitch Linear mapping.

    Computes optimal linear transformation between color distributions
    based on covariance matrix matching.

    Args:
        target: Target image (H, W, 3) float32 [0, 1].
        source: Source image (H, W, 3) float32 [0, 1].
        target_mask: Optional mask for target statistics.
        source_mask: Optional mask for source statistics.
        mask_cutoff: Mask threshold.
        eps: Numerical stability epsilon. Default: 1e-10.

    Returns:
        Color-transferred target image (H, W, 3) float32 [0, 1].
    """
    h, w, c = target.shape

    # Get pixels for statistics
    if source_mask is not None:
        s_mask = source_mask.squeeze() > mask_cutoff
        if s_mask.any():
            s_pixels = source[s_mask]
        else:
            s_pixels = source.reshape(-1, c)
    else:
        s_pixels = source.reshape(-1, c)

    if target_mask is not None:
        t_mask = target_mask.squeeze() > mask_cutoff
        if t_mask.any():
            t_pixels = target[t_mask]
        else:
            t_pixels = target.reshape(-1, c)
    else:
        t_pixels = target.reshape(-1, c)

    x0 = t_pixels
    x1 = s_pixels

    # Compute covariance matrices
    a = np.cov(x0.T)
    b = np.cov(x1.T)

    # Eigendecomposition of target covariance
    Da2, Ua = npla.eig(a)
    Da = np.diag(np.sqrt(np.clip(Da2.real, eps, None)))

    # Compute intermediate matrix C
    C = Da @ Ua.T @ b @ Ua @ Da

    # Eigendecomposition of C
    Dc2, Uc = npla.eig(C)
    Dc = np.diag(np.sqrt(np.clip(Dc2.real, eps, None)))

    # Inverse of Da
    Da_inv = np.diag(1.0 / np.diag(Da))

    # Compute transformation matrix
    T = Ua @ Da_inv @ Uc @ Dc @ Uc.T @ Da_inv @ Ua.T

    # Apply transformation to ALL pixels
    # Calculate means from MASKED pixels
    mx0 = x0.mean(axis=0)
    mx1 = x1.mean(axis=0)

    # Apply to full target image
    target_flat = target.reshape(-1, c)
    result = (target_flat - mx0) @ T.real + mx1

    return np.clip(result.reshape(h, w, c).astype(np.float32), 0, 1)


def color_transfer_idt(
    target: np.ndarray,
    source: np.ndarray,
    bins: int = 256,
    n_rot: int = 20,
    target_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    mask_cutoff: float = 0.5,
) -> np.ndarray:
    """
    Iterative Distribution Transfer for color matching.

    Uses random rotations and histogram matching in rotated spaces
    for iterative color distribution matching.

    Args:
        target: Target image (H, W, 3) float32 [0, 1].
        source: Source image (H, W, 3) float32 [0, 1].
        bins: Number of histogram bins. Default: 256.
        n_rot: Number of random rotations. Default: 20.
        target_mask: Optional mask for target.
        source_mask: Optional mask for source.
        mask_cutoff: Mask threshold.

    Returns:
        Color-transferred target image (H, W, 3) float32 [0, 1].
    """
    import scipy.stats

    relaxation = 1.0 / n_rot
    h, w, c = target.shape

    # Get pixels to modify
    new_target = target.copy()

    if target_mask is not None:
        t_mask = target_mask.squeeze() > mask_cutoff
        if not t_mask.any():
            return target
        d0 = new_target[t_mask].T  # (C, N_t)
    else:
        d0 = new_target.reshape(-1, c).T  # (C, N_t)

    # Get source distribution
    if source_mask is not None:
        s_mask = source_mask.squeeze() > mask_cutoff
        if s_mask.any():
            d1 = source[s_mask].T  # (C, N_s)
        else:
            d1 = source.reshape(-1, c).T
    else:
        d1 = source.reshape(-1, c).T

    for _ in range(n_rot):
        # Random orthogonal rotation matrix
        r = scipy.stats.special_ortho_group.rvs(c).astype(np.float32)

        # Rotate both distributions
        d0r = r @ d0
        d1r = r @ d1
        d_r = np.empty_like(d0)

        # Match histograms in each rotated dimension
        for j in range(c):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            # Cumulative histograms
            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1] + 1e-10

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1] + 1e-10

            # Transfer function
            f = np.interp(cp0r, cp1r, edges[1:])

            # Apply transfer
            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        # Inverse rotation and relaxation update
        d0 = relaxation * npla.solve(r, d_r - d0r) + d0

    # Put back modified pixels
    if target_mask is not None:
        new_target[t_mask] = d0.T
    else:
        new_target = d0.T.reshape(h, w, c)

    return np.clip(new_target.astype(np.float32), 0, 1)


def color_transfer_mix(
    target: np.ndarray,
    source: np.ndarray,
    sot_steps: int = 10,
    sot_batch_size: int = 30,
) -> np.ndarray:
    """
    Mixed color transfer combining LCT (lightness) and SOT (chrominance).

    This method:
    1. Applies Linear Color Transfer to the L (lightness) channel in LAB space
    2. Applies Sliced Optimal Transport to the A and B (color) channels
    3. Combines the results for natural-looking color matching

    This is the "mix-m" mode from DeepFaceLab, providing the best of both:
    - LCT for stable luminance matching
    - SOT for accurate color distribution matching

    Args:
        target: Target image (H, W, 3) float32 [0, 1] BGR.
        source: Source image (H, W, 3) float32 [0, 1] BGR.
        sot_steps: Number of SOT iterations. Default: 10.
        sot_batch_size: SOT batch size. Default: 30.

    Returns:
        Color-transferred target image (H, W, 3) float32 [0, 1] BGR.
    """
    # Convert to uint8 for LAB conversion
    target_uint8 = np.clip(target * 255, 0, 255).astype(np.uint8)
    source_uint8 = np.clip(source * 255, 0, 255).astype(np.uint8)

    # Convert to LAB
    target_lab = cv2.cvtColor(target_uint8, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(source_uint8, cv2.COLOR_BGR2LAB)

    # Step 1: Apply LCT to lightness channel only
    target_l = target_lab[..., 0:1].astype(np.float32) / 255.0
    source_l = source_lab[..., 0:1].astype(np.float32) / 255.0

    # Linear color transfer on L channel (1D case)
    transferred_l = linear_color_transfer(target_l, source_l)
    transferred_l = np.clip(transferred_l[..., 0] * 255, 0, 255).astype(np.uint8)

    # Step 2: Apply SOT to color channels (AB)
    # Set L to constant (100) for both images to isolate color
    target_lab_neutral = target_lab.copy()
    source_lab_neutral = source_lab.copy()
    target_lab_neutral[..., 0] = 100
    source_lab_neutral[..., 0] = 100

    # Convert back to BGR with neutral L for SOT
    target_bgr_neutral = cv2.cvtColor(target_lab_neutral, cv2.COLOR_LAB2BGR)
    source_bgr_neutral = cv2.cvtColor(source_lab_neutral, cv2.COLOR_LAB2BGR)

    # Apply SOT to get color-matched result
    sot_result = color_transfer_sot(
        target_bgr_neutral.astype(np.float32),
        source_bgr_neutral.astype(np.float32),
        steps=sot_steps,
        batch_size=sot_batch_size,
        reg_sigma_xy=0,  # Disable regularization for cleaner result
    )
    sot_result_uint8 = np.clip(sot_result * 255, 0, 255).astype(np.uint8)

    # Step 3: Combine LCT lightness with SOT color
    result_lab = cv2.cvtColor(sot_result_uint8, cv2.COLOR_BGR2LAB)
    result_lab[..., 0] = transferred_l
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return (result_bgr / 255.0).astype(np.float32)


def color_transfer(
    mode: ColorTransferMode,
    target: np.ndarray,
    source: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Unified color transfer interface.

    Args:
        mode: Transfer mode ('rct', 'lct', 'sot', 'mkl', 'idt', 'neural').
              Also supports masked variants ('mkl-masked', etc.).
        target: Target image to modify (H, W, 3) float32 [0, 1].
        source: Source image to match colors from (H, W, 3) float32 [0, 1].
        **kwargs: Additional arguments passed to specific function.

    Returns:
        Color-transferred image (H, W, 3) float32 [0, 1].
    """
    # Handle masked modes
    if mode.endswith("-masked"):
        base_mode = mode.replace("-masked", "")
        # The caller is expected to provide target_mask and source_mask in kwargs
        # if they want masking. If not provided, it falls back to full image.
    else:
        base_mode = mode
        # If not masked mode, we should suppress masks if they were passed
        # OR we just let them pass through but set them to None?
        # Actually, if the user explicitly requests 'rct' but passes masks,
        # they probably want masking. 'rct' supports masking natively.
        # But 'mkl' vs 'mkl-masked': 'mkl-masked' forces intention.
        # Let's use base_mode for dispatch but keep kwargs.
        pass

    if base_mode == "rct":
        return reinhard_color_transfer(target, source, **kwargs)
    elif base_mode == "lct":
        return linear_color_transfer(target, source, **kwargs)
    elif base_mode == "sot":
        return color_transfer_sot(target, source, **kwargs)
    elif base_mode == "mkl":
        return color_transfer_mkl(target, source, **kwargs)
    elif base_mode == "idt":
        return color_transfer_idt(target, source, **kwargs)
    elif base_mode == "mix":
        return color_transfer_mix(target, source, **kwargs)
    elif base_mode == "neural":
        from visagen.postprocess.neural_color import neural_color_transfer

        return neural_color_transfer(target, source, **kwargs)
    else:
        raise ValueError(f"Unknown color transfer mode: {mode}")
