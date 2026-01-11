"""
GPU-Accelerated Color Transfer Functions.

Uses Kornia and PyTorch for GPU-accelerated color transfer operations.
Falls back to CPU implementation if CUDA/Kornia not available.

Supported modes:
- RCT (Reinhard Color Transfer): LAB space statistics matching
- LCT (Linear Color Transfer): PCA/Cholesky covariance matching
- MKL (Monge-Kantorovitch Linear): Optimal transport linear mapping

All functions expect float32 tensors in [0, 1] range with RGB channel order
and shape (B, 3, H, W) or (3, H, W).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Check Kornia availability
try:
    import kornia

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    kornia = None  # type: ignore[assignment]
    logger.info("kornia not available, GPU color transfer will use CPU fallback")


GPUColorTransferMode = Literal["rct", "lct", "mkl"]


def _ensure_4d(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure tensor is 4D (B, C, H, W). Returns tensor and whether it was 3D."""
    if tensor.ndim == 3:
        return tensor.unsqueeze(0), True
    return tensor, False


def _restore_dims(tensor: torch.Tensor, was_3d: bool) -> torch.Tensor:
    """Restore original dimensions if needed."""
    if was_3d:
        return tensor.squeeze(0)
    return tensor


def reinhard_color_transfer_gpu(
    target: torch.Tensor,
    source: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    mask_cutoff: float = 0.5,
) -> torch.Tensor:
    """
    GPU-accelerated Reinhard Color Transfer using Kornia.

    Matches the color distribution of target image to source image
    using LAB color space statistics (mean and standard deviation).

    Args:
        target: Target image to modify (B, 3, H, W) or (3, H, W) RGB float32 [0, 1].
        source: Source image to match colors from, same shape as target.
        target_mask: Optional mask for target statistics (B, 1, H, W) or (1, H, W).
        source_mask: Optional mask for source statistics (B, 1, H, W) or (1, H, W).
        mask_cutoff: Threshold for mask application. Default: 0.5.

    Returns:
        Color-transferred target image, same shape as input.
    """
    if not KORNIA_AVAILABLE:
        raise ImportError("kornia is required for GPU color transfer")

    # Ensure 4D tensors
    target, target_was_3d = _ensure_4d(target)
    source, _ = _ensure_4d(source)

    if target_mask is not None:
        target_mask, _ = _ensure_4d(target_mask)
        target_mask = (target_mask > mask_cutoff).float()

    if source_mask is not None:
        source_mask, _ = _ensure_4d(source_mask)
        source_mask = (source_mask > mask_cutoff).float()

    eps = 1e-6

    # RGB to LAB (Kornia uses RGB, not BGR like OpenCV)
    source_lab = kornia.color.rgb_to_lab(source)  # (B, 3, H, W)
    target_lab = kornia.color.rgb_to_lab(target)

    # Compute source statistics per channel
    if source_mask is not None and source_mask.any():
        # Expand mask to 3 channels for broadcasting
        s_mask = source_mask.expand_as(source_lab)
        mask_sum = source_mask.sum(dim=[-2, -1], keepdim=True).clamp(min=1)
        src_mean = (source_lab * s_mask).sum(dim=[-2, -1], keepdim=True) / mask_sum
        src_var = ((source_lab - src_mean) ** 2 * s_mask).sum(
            dim=[-2, -1], keepdim=True
        ) / mask_sum
        src_std = torch.sqrt(src_var + eps)
    else:
        src_mean = source_lab.mean(dim=[-2, -1], keepdim=True)
        src_std = source_lab.std(dim=[-2, -1], keepdim=True)

    # Compute target statistics per channel
    if target_mask is not None and target_mask.any():
        t_mask = target_mask.expand_as(target_lab)
        mask_sum = target_mask.sum(dim=[-2, -1], keepdim=True).clamp(min=1)
        tgt_mean = (target_lab * t_mask).sum(dim=[-2, -1], keepdim=True) / mask_sum
        tgt_var = ((target_lab - tgt_mean) ** 2 * t_mask).sum(
            dim=[-2, -1], keepdim=True
        ) / mask_sum
        tgt_std = torch.sqrt(tgt_var + eps)
    else:
        tgt_mean = target_lab.mean(dim=[-2, -1], keepdim=True)
        tgt_std = target_lab.std(dim=[-2, -1], keepdim=True)

    # Transfer: normalize by target stats, denormalize by source stats
    result_lab = (target_lab - tgt_mean) * (src_std / (tgt_std + eps)) + src_mean

    # LAB to RGB
    result = kornia.color.lab_to_rgb(result_lab)

    return _restore_dims(result.clamp(0, 1), target_was_3d)


def linear_color_transfer_gpu(
    target: torch.Tensor,
    source: torch.Tensor,
    mode: Literal["pca", "chol", "sym"] = "pca",
    target_mask: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    mask_cutoff: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    GPU-accelerated Linear Color Transfer using covariance matching.

    Matches the color distribution using linear transformation
    based on covariance matrices of source and target.

    Args:
        target: Target image (B, 3, H, W) or (3, H, W) float32 [0, 1].
        source: Source image, same shape as target.
        mode: Transfer mode - 'pca', 'chol', or 'sym'. Default: 'pca'.
        target_mask: Optional mask for target statistics.
        source_mask: Optional mask for source statistics.
        mask_cutoff: Mask threshold. Default: 0.5.
        eps: Regularization epsilon. Default: 1e-5.

    Returns:
        Color-transferred image, same shape as input.
    """
    # Ensure 4D tensors
    target, target_was_3d = _ensure_4d(target)
    source, _ = _ensure_4d(source)

    b, c, h, w = target.shape
    device = target.device
    dtype = target.dtype

    # Flatten to (B, C, N) where N = H * W
    t_flat = target.view(b, c, -1)
    s_flat = source.view(b, c, -1)

    # Handle masks for statistics computation
    if source_mask is not None:
        source_mask, _ = _ensure_4d(source_mask)
        s_mask_flat = (source_mask > mask_cutoff).float().view(b, 1, -1)
        s_mask_sum = s_mask_flat.sum(dim=-1, keepdim=True).clamp(min=1)
        # Masked mean
        mu_s = (s_flat * s_mask_flat).sum(dim=-1, keepdim=True) / s_mask_sum
        # Masked covariance
        s_centered = (s_flat - mu_s) * s_mask_flat
        n_s = s_mask_sum.squeeze(-1)
    else:
        mu_s = s_flat.mean(dim=-1, keepdim=True)
        s_centered = s_flat - mu_s
        n_s = torch.tensor(h * w, device=device, dtype=dtype)

    if target_mask is not None:
        target_mask, _ = _ensure_4d(target_mask)
        t_mask_flat = (target_mask > mask_cutoff).float().view(b, 1, -1)
        t_mask_sum = t_mask_flat.sum(dim=-1, keepdim=True).clamp(min=1)
        mu_t = (t_flat * t_mask_flat).sum(dim=-1, keepdim=True) / t_mask_sum
        t_centered_masked = (t_flat - mu_t) * t_mask_flat
        n_t = t_mask_sum.squeeze(-1)
    else:
        mu_t = t_flat.mean(dim=-1, keepdim=True)
        t_centered_masked = t_flat - mu_t
        n_t = torch.tensor(h * w, device=device, dtype=dtype)

    # Covariance matrices (B, C, C)
    # For masked case, we need to normalize by mask sum
    if target_mask is not None:
        Ct = torch.bmm(t_centered_masked, t_centered_masked.transpose(-2, -1)) / (
            n_t.view(b, 1, 1) - 1
        ).clamp(min=1)
    else:
        Ct = torch.bmm(t_centered_masked, t_centered_masked.transpose(-2, -1)) / (
            n_t - 1
        )

    if source_mask is not None:
        Cs = torch.bmm(s_centered, s_centered.transpose(-2, -1)) / (
            n_s.view(b, 1, 1) - 1
        ).clamp(min=1)
    else:
        Cs = torch.bmm(s_centered, s_centered.transpose(-2, -1)) / (n_s - 1)

    # Add regularization
    eye = torch.eye(c, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
    Ct = Ct + eps * eye
    Cs = Cs + eps * eye

    # Compute transform based on mode
    if mode == "chol":
        # Cholesky decomposition based transfer
        chol_t = torch.linalg.cholesky(Ct)
        chol_s = torch.linalg.cholesky(Cs)
        transform = chol_s @ torch.linalg.inv(chol_t)

    elif mode == "pca":
        # PCA-based transfer (eigenvalue decomposition)
        eva_t, eve_t = torch.linalg.eigh(Ct)
        eva_s, eve_s = torch.linalg.eigh(Cs)
        Qt = (
            eve_t
            @ torch.diag_embed(torch.sqrt(eva_t.clamp(min=eps)))
            @ eve_t.transpose(-2, -1)
        )
        Qs = (
            eve_s
            @ torch.diag_embed(torch.sqrt(eva_s.clamp(min=eps)))
            @ eve_s.transpose(-2, -1)
        )
        transform = Qs @ torch.linalg.inv(Qt)

    elif mode == "sym":
        # Symmetric transfer
        eva_t, eve_t = torch.linalg.eigh(Ct)
        Qt = (
            eve_t
            @ torch.diag_embed(torch.sqrt(eva_t.clamp(min=eps)))
            @ eve_t.transpose(-2, -1)
        )
        Qt_Cs_Qt = Qt @ Cs @ Qt
        eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = (
            eve_QtCsQt
            @ torch.diag_embed(torch.sqrt(eva_QtCsQt.clamp(min=eps)))
            @ eve_QtCsQt.transpose(-2, -1)
        )
        Qt_inv = torch.linalg.inv(Qt)
        transform = Qt_inv @ QtCsQt @ Qt_inv

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pca', 'chol', or 'sym'.")

    # Apply transform to ALL target pixels (not just masked)
    t_centered_all = t_flat - mu_t
    result = torch.bmm(transform, t_centered_all) + mu_s

    return _restore_dims(result.view(b, c, h, w).clamp(0, 1), target_was_3d)


def color_transfer_mkl_gpu(
    target: torch.Tensor,
    source: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    mask_cutoff: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    GPU-accelerated Monge-Kantorovitch Linear Transfer.

    Computes optimal linear transformation between color distributions
    based on covariance matrix matching using eigendecomposition.

    Args:
        target: Target image (B, 3, H, W) or (3, H, W) float32 [0, 1].
        source: Source image, same shape as target.
        target_mask: Optional mask for target statistics.
        source_mask: Optional mask for source statistics.
        mask_cutoff: Mask threshold.
        eps: Numerical stability epsilon. Default: 1e-10.

    Returns:
        Color-transferred target image, same shape as input.
    """
    # Ensure 4D tensors
    target, target_was_3d = _ensure_4d(target)
    source, _ = _ensure_4d(source)

    b, c, h, w = target.shape
    device = target.device
    dtype = target.dtype

    # Flatten to (B, C, N)
    t_flat = target.view(b, c, -1)
    s_flat = source.view(b, c, -1)

    # Handle masks
    if source_mask is not None:
        source_mask, _ = _ensure_4d(source_mask)
        s_mask_flat = (source_mask > mask_cutoff).float().view(b, 1, -1)
        s_mask_sum = s_mask_flat.sum(dim=-1, keepdim=True).clamp(min=1)
        mu_s = (s_flat * s_mask_flat).sum(dim=-1, keepdim=True) / s_mask_sum
        s_centered = (s_flat - mu_s) * s_mask_flat
        n_s = s_mask_sum.squeeze(-1)
    else:
        mu_s = s_flat.mean(dim=-1, keepdim=True)
        s_centered = s_flat - mu_s
        n_s = torch.tensor(h * w, device=device, dtype=dtype)

    if target_mask is not None:
        target_mask, _ = _ensure_4d(target_mask)
        t_mask_flat = (target_mask > mask_cutoff).float().view(b, 1, -1)
        t_mask_sum = t_mask_flat.sum(dim=-1, keepdim=True).clamp(min=1)
        mu_t = (t_flat * t_mask_flat).sum(dim=-1, keepdim=True) / t_mask_sum
        t_centered = (t_flat - mu_t) * t_mask_flat
        n_t = t_mask_sum.squeeze(-1)
    else:
        mu_t = t_flat.mean(dim=-1, keepdim=True)
        t_centered = t_flat - mu_t
        n_t = torch.tensor(h * w, device=device, dtype=dtype)

    # Covariance matrices
    if target_mask is not None:
        Ct = torch.bmm(t_centered, t_centered.transpose(-2, -1)) / (
            n_t.view(b, 1, 1) - 1
        ).clamp(min=1)
    else:
        Ct = torch.bmm(t_centered, t_centered.transpose(-2, -1)) / (n_t - 1)

    if source_mask is not None:
        Cs = torch.bmm(s_centered, s_centered.transpose(-2, -1)) / (
            n_s.view(b, 1, 1) - 1
        ).clamp(min=1)
    else:
        Cs = torch.bmm(s_centered, s_centered.transpose(-2, -1)) / (n_s - 1)

    # Eigendecomposition of target covariance
    Da2, Ua = torch.linalg.eig(Ct)
    Da = torch.diag_embed(torch.sqrt(Da2.real.clamp(min=eps)))
    Ua_real = Ua.real

    # Compute intermediate matrix C
    C = Da @ Ua_real.transpose(-2, -1) @ Cs @ Ua_real @ Da

    # Eigendecomposition of C
    Dc2, Uc = torch.linalg.eig(C)
    Dc = torch.diag_embed(torch.sqrt(Dc2.real.clamp(min=eps)))
    Uc_real = Uc.real

    # Inverse of Da
    Da_diag = torch.diagonal(Da, dim1=-2, dim2=-1)
    Da_inv = torch.diag_embed(1.0 / Da_diag.clamp(min=eps))

    # Compute transformation matrix
    T = (
        Ua_real
        @ Da_inv
        @ Uc_real
        @ Dc
        @ Uc_real.transpose(-2, -1)
        @ Da_inv
        @ Ua_real.transpose(-2, -1)
    )

    # Apply transformation to ALL target pixels
    t_centered_all = t_flat - mu_t
    result = torch.bmm(T, t_centered_all) + mu_s

    return _restore_dims(result.view(b, c, h, w).clamp(0, 1), target_was_3d)


def color_transfer_gpu(
    mode: str,
    target: torch.Tensor,
    source: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Unified GPU color transfer interface.

    Args:
        mode: Transfer mode ('rct', 'lct', 'mkl').
        target: Target image (B, 3, H, W) or (3, H, W) float32 [0, 1].
        source: Source image, same shape as target.
        **kwargs: Additional arguments passed to specific function.

    Returns:
        Color-transferred image, same shape as input.

    Raises:
        ImportError: If kornia is not available (for RCT mode).
        ValueError: If mode is not supported.
    """
    if mode == "rct":
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for GPU RCT color transfer")
        return reinhard_color_transfer_gpu(target, source, **kwargs)
    elif mode == "lct":
        return linear_color_transfer_gpu(target, source, **kwargs)
    elif mode == "mkl":
        return color_transfer_mkl_gpu(target, source, **kwargs)
    else:
        raise ValueError(f"GPU mode not supported: {mode}. Use 'rct', 'lct', or 'mkl'.")


def is_gpu_mode_available(mode: str) -> bool:
    """
    Check if GPU mode is available for the given color transfer mode.

    Args:
        mode: Color transfer mode to check.

    Returns:
        True if GPU acceleration is available for this mode.
    """
    if mode == "rct":
        return KORNIA_AVAILABLE
    elif mode in ("lct", "mkl"):
        return True  # Pure PyTorch, always available
    return False
