"""
Grid-based Face Warping.

Port of legacy DeepFaceLab warping to PyTorch.
Uses random displacement grids for data augmentation.
"""

import random
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gen_warp_params(
    size: int,
    rng: torch.Generator | None = None,
) -> dict[str, Any]:
    """
    Generate random warp displacement maps.

    Creates grid-based displacement maps for face warping augmentation.
    Matches legacy DeepFaceLab behavior.

    Args:
        size: Image size (assumes square images).
        rng: Random number generator for reproducibility.

    Returns:
        Dict with 'grid' tensor of shape (1, H, W, 2) for grid_sample.

    Example:
        >>> params = gen_warp_params(256)
        >>> params['grid'].shape
        torch.Size([1, 256, 256, 2])
    """
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(random.randint(0, 2**31))

    # Choose random cell size from [size//8, size//4, size//2]
    cell_power = int(torch.randint(1, 4, (1,), generator=rng).item())
    cell_size = max(size // (2**cell_power), 4)
    cell_count = size // cell_size + 1

    # Create base grid points
    grid_points = torch.linspace(0, size - 1, cell_count)
    mapx = grid_points.unsqueeze(0).expand(cell_count, -1).clone()
    mapy = grid_points.unsqueeze(1).expand(-1, cell_count).clone()

    # Add random displacement to interior points
    # Displacement magnitude: cell_size * 0.24 (legacy value)
    displacement = cell_size * 0.24
    interior_h = cell_count - 2
    interior_w = cell_count - 2

    if interior_h > 0 and interior_w > 0:
        noise_x = torch.randn(interior_h, interior_w, generator=rng) * displacement
        noise_y = torch.randn(interior_h, interior_w, generator=rng) * displacement

        mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + noise_x
        mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + noise_y

    # Upscale to full resolution using bilinear interpolation
    mapx = mapx.unsqueeze(0).unsqueeze(0)  # (1, 1, cell_count, cell_count)
    mapy = mapy.unsqueeze(0).unsqueeze(0)

    mapx = F.interpolate(mapx, size=(size, size), mode="bilinear", align_corners=True)
    mapy = F.interpolate(mapy, size=(size, size), mode="bilinear", align_corners=True)

    mapx = mapx.squeeze(0).squeeze(0)  # (size, size)
    mapy = mapy.squeeze(0).squeeze(0)

    # Normalize to [-1, 1] for grid_sample
    grid_x = (mapx / (size - 1)) * 2 - 1
    grid_y = (mapy / (size - 1)) * 2 - 1

    # Stack to create grid (H, W, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return {
        "grid": grid.unsqueeze(0),  # (1, H, W, 2)
        "size": size,
    }


def warp_by_params(
    image: torch.Tensor,
    params: dict[str, Any],
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Apply warping to image using displacement grid.

    Uses torch.nn.functional.grid_sample for GPU-accelerated warping.

    Args:
        image: Image tensor of shape (B, C, H, W) or (C, H, W).
        params: Warp parameters from gen_warp_params().
        mode: Interpolation mode ('bilinear' or 'nearest').
        padding_mode: Padding mode ('zeros', 'border', or 'reflection').

    Returns:
        Warped image tensor with same shape as input.

    Example:
        >>> image = torch.randn(1, 3, 256, 256)
        >>> params = gen_warp_params(256)
        >>> warped = warp_by_params(image, params)
        >>> warped.shape
        torch.Size([1, 3, 256, 256])
    """
    # Handle unbatched input
    squeeze_output = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True

    grid = cast(torch.Tensor, params["grid"])

    # Expand grid for batch size if needed
    if grid.shape[0] != image.shape[0]:
        grid = grid.expand(image.shape[0], -1, -1, -1)

    # Move grid to same device as image
    grid = grid.to(image.device, image.dtype)

    # Apply warping
    warped = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )

    if squeeze_output:
        warped = warped.squeeze(0)

    return warped


def gen_legacy_warp_params(
    size: int,
    *,
    flip: bool = False,
    flip_prob: float = 0.4,
    rotation_range: tuple[float, float] = (-10.0, 10.0),
    scale_range: tuple[float, float] = (-0.05, 0.05),
    tx_range: tuple[float, float] = (-0.05, 0.05),
    ty_range: tuple[float, float] = (-0.05, 0.05),
    rng: np.random.Generator | None = None,
    warp_rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Generate legacy DeepFaceLab warp parameters.

    This mirrors `legacy/core/imagelib/warp.py::gen_warp_params` and returns
    remap matrices + affine matrix + flip flag for strict parity mode.
    """
    if rng is None:
        rng = np.random.default_rng()
    if warp_rng is None:
        warp_rng = rng

    w = size
    rw: int | None = None
    if w < 64:
        rw = w
        w = 64

    rotation = float(rng.uniform(rotation_range[0], rotation_range[1]))
    scale = float(rng.uniform(1.0 / (1.0 - scale_range[0]), 1.0 + scale_range[1]))
    tx = float(rng.uniform(tx_range[0], tx_range[1]))
    ty = float(rng.uniform(ty_range[0], ty_range[1]))
    p_flip = bool(flip and rng.random() < flip_prob)

    cell_size = [w // (2**i) for i in range(1, 4)][int(warp_rng.integers(0, 3))]
    cell_size = max(int(cell_size), 1)
    cell_count = w // cell_size + 1

    grid_points = np.linspace(0, w, cell_count, dtype=np.float32)
    mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
    mapy = mapx.T.copy()
    if cell_count > 2:
        shape = (cell_count - 2, cell_count - 2)
        disp_scale = cell_size * 0.24
        mapx[1:-1, 1:-1] += (
            warp_rng.standard_normal(shape).astype(np.float32) * disp_scale
        )
        mapy[1:-1, 1:-1] += (
            warp_rng.standard_normal(shape).astype(np.float32) * disp_scale
        )

    half_cell_size = cell_size // 2
    resize_shape = (w + cell_size, w + cell_size)
    mapx = cv2.resize(mapx, resize_shape, interpolation=cv2.INTER_LINEAR)[
        half_cell_size : half_cell_size + w,
        half_cell_size : half_cell_size + w,
    ].astype(np.float32)
    mapy = cv2.resize(mapy, resize_shape, interpolation=cv2.INTER_LINEAR)[
        half_cell_size : half_cell_size + w,
        half_cell_size : half_cell_size + w,
    ].astype(np.float32)

    random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
    random_transform_mat[0, 2] += tx * w
    random_transform_mat[1, 2] += ty * w
    random_transform_mat = random_transform_mat.astype(np.float32)

    u_mat = random_transform_mat.copy()
    u_mat[:, 2] = u_mat[:, 2] / float(w)

    return {
        "mapx": mapx,
        "mapy": mapy,
        "rmat": random_transform_mat,
        "umat": u_mat,
        "w": w,
        "rw": rw,
        "flip": p_flip,
    }


def warp_legacy_by_params(
    image: torch.Tensor,
    params: dict[str, Any],
    *,
    can_warp: bool = True,
    can_transform: bool = True,
    can_flip: bool = True,
    border_replicate: bool = True,
    cv2_inter: int = cv2.INTER_CUBIC,
) -> torch.Tensor:
    """
    Apply legacy DeepFaceLab remap+affine+flip warp on torch tensors.

    Args:
        image: Tensor of shape (C, H, W) or (B, C, H, W).
        params: Params returned by `gen_legacy_warp_params`.
        can_warp: Apply remap stage.
        can_transform: Apply affine stage.
        can_flip: Apply flip stage based on params["flip"].
        border_replicate: Use BORDER_REPLICATE for affine border handling.
        cv2_inter: OpenCV interpolation flag.
    """
    squeeze_output = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True

    rw = cast(int | None, params["rw"])
    mapx = cast(np.ndarray, params["mapx"])
    mapy = cast(np.ndarray, params["mapy"])
    rmat = cast(np.ndarray, params["rmat"])
    w = cast(int, params["w"])
    do_flip = bool(params.get("flip", False))

    input_device = image.device
    input_dtype = image.dtype

    image_np = image.detach().cpu().numpy()
    warped_np: list[np.ndarray] = []
    for sample in image_np:
        sample_hwc = np.transpose(sample, (1, 2, 0))

        if (can_warp or can_transform) and rw is not None:
            sample_hwc = cv2.resize(sample_hwc, (64, 64), interpolation=cv2_inter)

        if can_warp:
            sample_hwc = cv2.remap(sample_hwc, mapx, mapy, cv2_inter)

        if can_transform:
            sample_hwc = cv2.warpAffine(
                sample_hwc,
                rmat,
                (w, w),
                borderMode=(
                    cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
                ),
                flags=cv2_inter,
            )

        if (can_warp or can_transform) and rw is not None:
            sample_hwc = cv2.resize(sample_hwc, (rw, rw), interpolation=cv2_inter)

        if sample_hwc.ndim == 2:
            sample_hwc = sample_hwc[..., np.newaxis]

        if can_flip and do_flip:
            sample_hwc = sample_hwc[:, ::-1, ...]

        warped_np.append(np.transpose(sample_hwc, (2, 0, 1)))

    output = torch.from_numpy(np.stack(warped_np, axis=0)).to(
        device=input_device, dtype=input_dtype
    )

    if squeeze_output:
        output = output.squeeze(0)
    return output


def gen_affine_params(
    size: int,
    rotation_range: tuple[float, float] = (-10, 10),
    scale_range: tuple[float, float] = (-0.05, 0.05),
    translation_range: tuple[float, float] = (-0.05, 0.05),
    rng: torch.Generator | None = None,
) -> dict[str, Any]:
    """
    Generate random affine transformation parameters.

    Args:
        size: Image size.
        rotation_range: Min/max rotation in degrees.
        scale_range: Min/max scale deviation (relative).
        translation_range: Min/max translation (relative to size).
        rng: Random number generator.

    Returns:
        Dict with 'matrix' (2x3 affine matrix) and individual params.
    """
    import math

    if rng is None:
        rng = torch.Generator()

    # Generate random parameters
    rotation = (
        torch.rand(1, generator=rng).item() * (rotation_range[1] - rotation_range[0])
        + rotation_range[0]
    )
    scale = (
        1.0
        + torch.rand(1, generator=rng).item() * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    tx = (
        torch.rand(1, generator=rng).item()
        * (translation_range[1] - translation_range[0])
        + translation_range[0]
    ) * size
    ty = (
        torch.rand(1, generator=rng).item()
        * (translation_range[1] - translation_range[0])
        + translation_range[0]
    ) * size

    # Build rotation matrix around center
    angle_rad = math.radians(rotation)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    center = size / 2.0

    # Affine matrix: rotate around center, then scale, then translate
    # M = T_center @ R @ S @ T_neg_center @ T_offset
    matrix = torch.tensor(
        [
            [
                cos_a * scale,
                -sin_a * scale,
                (1 - cos_a * scale) * center + sin_a * scale * center + tx,
            ],
            [
                sin_a * scale,
                cos_a * scale,
                (1 - cos_a * scale) * center - sin_a * scale * center + ty,
            ],
        ],
        dtype=torch.float32,
    )

    return {
        "matrix": matrix,
        "rotation": rotation,
        "scale": scale,
        "tx": tx,
        "ty": ty,
    }


def apply_affine(
    image: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Apply affine transformation to image.

    Args:
        image: Image tensor of shape (B, C, H, W) or (C, H, W).
        matrix: Affine matrix of shape (2, 3).
        mode: Interpolation mode.
        padding_mode: Padding mode.

    Returns:
        Transformed image tensor.
    """
    squeeze_output = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True

    b, c, h, w = image.shape

    # Convert matrix to normalized coordinates for affine_grid
    # Need to convert from pixel to [-1, 1] space
    matrix_norm = matrix.clone().to(image.device, image.dtype)

    # Normalize translation
    matrix_norm[0, 2] = (
        matrix_norm[0, 2] / (w / 2) - 1 + matrix_norm[0, 0] + matrix_norm[0, 1]
    )
    matrix_norm[1, 2] = (
        matrix_norm[1, 2] / (h / 2) - 1 + matrix_norm[1, 0] + matrix_norm[1, 1]
    )

    # Create inverse transformation for grid_sample
    # affine_grid expects theta that maps from output to input
    theta = matrix_norm.unsqueeze(0).expand(b, -1, -1)

    grid = F.affine_grid(theta, [b, c, h, w], align_corners=True)

    output = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )

    if squeeze_output:
        output = output.squeeze(0)

    return output
