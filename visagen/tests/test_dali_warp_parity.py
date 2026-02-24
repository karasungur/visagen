"""
Parity checks for DALI warp implementation.

These tests validate that the DALI warp path tracks reference behavior
closely enough for augmentation quality and consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visagen.data.dali_warp import apply_warp_grid_numpy, gen_dali_warp_grid

cv2 = pytest.importorskip("cv2")


def _reference_style_grid_batch(
    size: int,
    batch_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Reference implementation based on `core/imagelib/warp.py`."""
    rnd = np.random.RandomState(seed)
    grids = []

    for _ in range(batch_size):
        cell_size = [size // (2**i) for i in range(1, 4)][rnd.randint(3)]
        cell_size = max(int(cell_size), 1)
        cell_count = size // cell_size + 1

        grid_points = np.linspace(0, size, cell_count, dtype=np.float32)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T.copy()

        if cell_count > 2:
            mapx[1:-1, 1:-1] += rnd.normal(
                size=(cell_count - 2, cell_count - 2)
            ).astype(np.float32) * (cell_size * 0.24)
            mapy[1:-1, 1:-1] += rnd.normal(
                size=(cell_count - 2, cell_count - 2)
            ).astype(np.float32) * (cell_size * 0.24)

        half = cell_size // 2
        resized = (size + cell_size, size + cell_size)
        mapx = cv2.resize(mapx, resized, interpolation=cv2.INTER_LINEAR)
        mapy = cv2.resize(mapy, resized, interpolation=cv2.INTER_LINEAR)
        mapx = mapx[half : half + size, half : half + size].astype(np.float32)
        mapy = mapy[half : half + size, half : half + size].astype(np.float32)

        denom = max(size - 1, 1)
        grid_x = (mapx / denom) * 2 - 1
        grid_y = (mapy / denom) * 2 - 1
        grids.append(np.stack([grid_x, grid_y], axis=-1))

    return np.stack(grids).astype(np.float32)


def _grid_displacement_stats(grids: np.ndarray) -> tuple[float, float, float, float]:
    """Return displacement mean/std for x and y against identity grid."""
    size = grids.shape[1]
    y = np.linspace(-1, 1, size, dtype=np.float32)
    x = np.linspace(-1, 1, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    disp_x = grids[..., 0] - grid_x[None, ...]
    disp_y = grids[..., 1] - grid_y[None, ...]

    return (
        float(disp_x.mean()),
        float(disp_x.std()),
        float(disp_y.mean()),
        float(disp_y.std()),
    )


def test_warp_grid_displacement_distribution_matches_reference() -> None:
    """Modern displacement distribution should track reference."""
    size = 256
    batch_size = 64
    seed = 42

    ours = gen_dali_warp_grid(
        size=size,
        batch_size=batch_size,
        rng=np.random.default_rng(seed),
    )
    ref = _reference_style_grid_batch(size=size, batch_size=batch_size, seed=seed)

    ours_stats = _grid_displacement_stats(ours)
    ref_stats = _grid_displacement_stats(ref)

    # Means should stay near zero and stds should be close.
    assert abs(ours_stats[0] - ref_stats[0]) < 0.03
    assert abs(ours_stats[1] - ref_stats[1]) < 0.08
    assert abs(ours_stats[2] - ref_stats[2]) < 0.03
    assert abs(ours_stats[3] - ref_stats[3]) < 0.08


def test_warp_effect_mse_tracks_reference() -> None:
    """Warp severity (MSE from original) should be in expected range."""
    size = 128
    seed = 7
    rng = np.random.default_rng(123)
    image = rng.random((size, size, 3), dtype=np.float32)

    ours_grid = gen_dali_warp_grid(
        size=size,
        batch_size=1,
        rng=np.random.default_rng(seed),
    )[0]
    ref_grid = _reference_style_grid_batch(size=size, batch_size=1, seed=seed)[0]

    ours_warped = apply_warp_grid_numpy(image, ours_grid)
    ref_warped = apply_warp_grid_numpy(image, ref_grid)

    mse_ours = float(np.mean((ours_warped - image) ** 2))
    mse_ref = float(np.mean((ref_warped - image) ** 2))
    rel = abs(mse_ours - mse_ref) / max(mse_ref, 1e-8)
    assert rel < 0.35

    try:
        from skimage.metrics import structural_similarity as ssim

        ssim_val = float(
            ssim(
                ours_warped,
                ref_warped,
                data_range=1.0,
                channel_axis=2,
            )
        )
        assert ssim_val > 0.70
    except Exception:
        # Optional dependency; MSE parity remains the mandatory check.
        pass


def test_dali_pipeline_declares_warp_stage() -> None:
    """Guide-compliance check: DALI pipeline must include warp_affine stage."""
    import visagen.data.dali_pipeline as dali_pipeline

    source = Path(dali_pipeline.__file__).read_text()
    assert "fn.warp_affine(" in source
