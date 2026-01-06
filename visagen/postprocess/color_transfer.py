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

ColorTransferMode = Literal["rct", "lct", "sot", "mkl", "idt", "neural"]


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

    Reference:
        "Color Transfer between Images" (Reinhard et al., 2001)
        https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    # Convert to LAB color space
    source_uint8 = np.clip(source * 255, 0, 255).astype(np.uint8)
    target_uint8 = np.clip(target * 255, 0, 255).astype(np.uint8)

    source_lab = cv2.cvtColor(source_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Apply masks for statistics computation
    source_input = source_lab.copy()
    target_input = target_lab.copy()

    if source_mask is not None:
        source_mask_2d = source_mask[..., 0] if source_mask.ndim == 3 else source_mask
        source_input[source_mask_2d < mask_cutoff] = 0

    if target_mask is not None:
        target_mask_2d = target_mask[..., 0] if target_mask.ndim == 3 else target_mask
        target_input[target_mask_2d < mask_cutoff] = 0

    # Compute statistics for each LAB channel
    eps = 1e-6

    src_l_mean, src_l_std = source_input[..., 0].mean(), source_input[..., 0].std()
    src_a_mean, src_a_std = source_input[..., 1].mean(), source_input[..., 1].std()
    src_b_mean, src_b_std = source_input[..., 2].mean(), source_input[..., 2].std()

    tgt_l_mean, tgt_l_std = target_input[..., 0].mean(), target_input[..., 0].std()
    tgt_a_mean, tgt_a_std = target_input[..., 1].mean(), target_input[..., 1].std()
    tgt_b_mean, tgt_b_std = target_input[..., 2].mean(), target_input[..., 2].std()

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

    # Clip to valid LAB ranges
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
        eps: Regularization epsilon. Default: 1e-5.

    Returns:
        Color-transferred image (H, W, 3) float32 [0, 1].
    """
    h, w, c = target.shape

    # Compute means
    mu_t = target.mean(axis=(0, 1))
    mu_s = source.mean(axis=(0, 1))

    # Flatten and center
    t = (target - mu_t).reshape(-1, c).T
    s = (source - mu_s).reshape(-1, c).T

    # Compute covariance matrices
    Ct = np.cov(t) + eps * np.eye(c)
    Cs = np.cov(s) + eps * np.eye(c)

    if mode == "chol":
        # Cholesky decomposition based transfer
        chol_t = npla.cholesky(Ct)
        chol_s = npla.cholesky(Cs)
        ts = chol_s @ npla.inv(chol_t) @ t

    elif mode == "pca":
        # PCA-based transfer (eigenvalue decomposition)
        eva_t, eve_t = npla.eigh(Ct)
        Qt = eve_t @ np.sqrt(np.diag(eva_t)) @ eve_t.T
        eva_s, eve_s = npla.eigh(Cs)
        Qs = eve_s @ np.sqrt(np.diag(eva_s)) @ eve_s.T
        ts = Qs @ npla.inv(Qt) @ t

    elif mode == "sym":
        # Symmetric transfer
        eva_t, eve_t = npla.eigh(Ct)
        Qt = eve_t @ np.sqrt(np.diag(eva_t)) @ eve_t.T
        Qt_Cs_Qt = Qt @ Cs @ Qt
        eva_QtCsQt, eve_QtCsQt = npla.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt @ np.sqrt(np.diag(eva_QtCsQt)) @ eve_QtCsQt.T
        ts = npla.inv(Qt) @ QtCsQt @ npla.inv(Qt) @ t

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

    Returns:
        Color-transferred source image (H, W, C) float32.

    Reference:
        https://github.com/dcoeurjo/OTColorTransfer
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
    advect = np.empty((h * w, c), dtype=src.dtype)

    for _ in range(steps):
        advect.fill(0)

        for _ in range(batch_size):
            # Random projection direction
            direction = np.random.normal(size=c).astype(src.dtype)
            direction /= npla.norm(direction)

            # Project onto random direction
            proj_source = (new_src * direction).sum(axis=-1).reshape(-1)
            proj_target = (trg * direction).sum(axis=-1).reshape(-1)

            # Sort projections
            id_source = np.argsort(proj_source)
            id_target = np.argsort(proj_target)

            # Optimal transport in 1D: match sorted values
            a = proj_target[id_target] - proj_source[id_source]

            # Accumulate advection
            for i_c in range(c):
                advect[id_source, i_c] += a * direction[i_c]

        # Update source
        new_src += advect.reshape(h, w, c) / batch_size

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
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Color Transfer using Monge-Kantorovitch Linear mapping.

    Computes optimal linear transformation between color distributions
    based on covariance matrix matching.

    Args:
        target: Target image (H, W, 3) float32 [0, 1].
        source: Source image (H, W, 3) float32 [0, 1].
        eps: Numerical stability epsilon. Default: 1e-10.

    Returns:
        Color-transferred target image (H, W, 3) float32 [0, 1].
    """
    h, w, c = target.shape
    h1, w1, c1 = source.shape

    # Flatten
    x0 = target.reshape(h * w, c)
    x1 = source.reshape(h1 * w1, c1)

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

    # Apply transformation
    mx0 = x0.mean(axis=0)
    mx1 = x1.mean(axis=0)

    result = (x0 - mx0) @ T.real + mx1

    return np.clip(result.reshape(h, w, c).astype(np.float32), 0, 1)


def color_transfer_idt(
    target: np.ndarray,
    source: np.ndarray,
    bins: int = 256,
    n_rot: int = 20,
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

    Returns:
        Color-transferred target image (H, W, 3) float32 [0, 1].
    """
    import scipy.stats

    relaxation = 1.0 / n_rot
    h, w, c = target.shape
    h1, w1, c1 = source.shape

    # Flatten
    i0 = target.reshape(h * w, c)
    i1 = source.reshape(h1 * w1, c1)

    d0 = i0.T
    d1 = i1.T

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

    return np.clip(d0.T.reshape(h, w, c).astype(np.float32), 0, 1)


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
        target: Target image to modify (H, W, 3) float32 [0, 1].
        source: Source image to match colors from (H, W, 3) float32 [0, 1].
        **kwargs: Additional arguments passed to specific function.

    Returns:
        Color-transferred image (H, W, 3) float32 [0, 1].

    Raises:
        ValueError: If unknown mode is specified.

    Example:
        >>> result = color_transfer('rct', target_img, source_img)
        >>> result = color_transfer('lct', target_img, source_img, mode='pca')
        >>> result = color_transfer('sot', src_img, trg_img, steps=20)
        >>> result = color_transfer('neural', target_img, source_img, strength=0.8)
    """
    if mode == "rct":
        return reinhard_color_transfer(target, source, **kwargs)
    elif mode == "lct":
        return linear_color_transfer(target, source, **kwargs)
    elif mode == "sot":
        return color_transfer_sot(target, source, **kwargs)
    elif mode == "mkl":
        return color_transfer_mkl(target, source, **kwargs)
    elif mode == "idt":
        return color_transfer_idt(target, source, **kwargs)
    elif mode == "neural":
        from visagen.postprocess.neural_color import neural_color_transfer

        return neural_color_transfer(target, source, **kwargs)
    else:
        raise ValueError(f"Unknown color transfer mode: {mode}")
