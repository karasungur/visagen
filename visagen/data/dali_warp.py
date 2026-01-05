"""
DFL-style Warp Grid Generator for NVIDIA DALI.

Provides GPU-compatible random displacement grids for face warping augmentation.
Used with DALI's external_source operator for custom augmentation.

The algorithm matches legacy DeepFaceLab behavior:
    1. Create a coarse grid with random cell sizes
    2. Add random displacement to interior points
    3. Upscale to full resolution via bilinear interpolation
    4. Normalize to [-1, 1] for grid_sample
"""

from typing import Optional, Tuple

import numpy as np


def gen_dali_warp_grid(
    size: int,
    batch_size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate DFL-style warp grids for DALI external_source.

    Creates random displacement grids matching legacy DeepFaceLab behavior.
    Returns numpy arrays suitable for DALI consumption.

    Args:
        size: Image size (assumes square images).
        batch_size: Number of grids to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        Warp grids of shape (batch_size, size, size, 2) in [-1, 1] range.
        The last dimension is (x, y) displacement for grid_sample.

    Example:
        >>> grids = gen_dali_warp_grid(256, batch_size=4)
        >>> grids.shape
        (4, 256, 256, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    grids = []

    for _ in range(batch_size):
        # Choose random cell size from [size//8, size//4, size//2]
        cell_power = rng.integers(1, 4)
        cell_size = max(size // (2 ** cell_power), 4)
        cell_count = size // cell_size + 1

        # Create base grid points
        grid_points = np.linspace(0, size - 1, cell_count, dtype=np.float32)
        mapx = np.tile(grid_points, (cell_count, 1))
        mapy = np.tile(grid_points.reshape(-1, 1), (1, cell_count))

        # Add random displacement to interior points
        # Displacement magnitude: cell_size * 0.24 (legacy value)
        displacement = cell_size * 0.24
        interior_h = cell_count - 2
        interior_w = cell_count - 2

        if interior_h > 0 and interior_w > 0:
            noise_x = rng.standard_normal((interior_h, interior_w)).astype(np.float32) * displacement
            noise_y = rng.standard_normal((interior_h, interior_w)).astype(np.float32) * displacement

            mapx[1:-1, 1:-1] += noise_x
            mapy[1:-1, 1:-1] += noise_y

        # Upscale to full resolution using scipy zoom (bilinear)
        from scipy.ndimage import zoom

        scale_factor = size / cell_count
        mapx_full = zoom(mapx, scale_factor, order=1)  # order=1 is bilinear
        mapy_full = zoom(mapy, scale_factor, order=1)

        # Ensure exact size (zoom might have slight variations)
        mapx_full = mapx_full[:size, :size]
        mapy_full = mapy_full[:size, :size]

        # Pad if needed
        if mapx_full.shape[0] < size or mapx_full.shape[1] < size:
            pad_h = size - mapx_full.shape[0]
            pad_w = size - mapx_full.shape[1]
            mapx_full = np.pad(mapx_full, ((0, pad_h), (0, pad_w)), mode='edge')
            mapy_full = np.pad(mapy_full, ((0, pad_h), (0, pad_w)), mode='edge')

        # Normalize to [-1, 1] for grid_sample
        grid_x = (mapx_full / (size - 1)) * 2 - 1
        grid_y = (mapy_full / (size - 1)) * 2 - 1

        # Stack to create grid (H, W, 2)
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        grids.append(grid)

    return np.stack(grids, axis=0)


class DALIWarpGridGenerator:
    """
    DALI external source callback for generating warp grids.

    Provides an iterator interface for DALI's external_source operator.
    Generates DFL-style random displacement grids on each call.

    Args:
        size: Image size for warp grids.
        seed: Random seed for reproducibility.

    Example:
        >>> generator = DALIWarpGridGenerator(256, seed=42)
        >>> grid = generator(sample_info)
        >>> grid.shape
        (256, 256, 2)
    """

    def __init__(self, size: int, seed: int = 42) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample_info) -> np.ndarray:
        """
        Generate a single warp grid for DALI external_source.

        Args:
            sample_info: DALI sample info (contains idx).

        Returns:
            Warp grid of shape (size, size, 2) as float32.
        """
        grid = gen_dali_warp_grid(
            size=self.size,
            batch_size=1,
            rng=self.rng,
        )
        return grid[0]  # Return single grid (H, W, 2)

    def reset(self) -> None:
        """Reset the generator (called at epoch boundaries)."""
        # Optionally reseed for different augmentation each epoch
        pass


def gen_dali_affine_matrix(
    size: int,
    batch_size: int = 1,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translation_range: Tuple[float, float] = (-0.05, 0.05),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate random affine transformation matrices for DALI.

    Args:
        size: Image size.
        batch_size: Number of matrices to generate.
        rotation_range: Min/max rotation in degrees.
        scale_range: Min/max scale factors.
        translation_range: Min/max translation (relative to size).
        rng: NumPy random generator.

    Returns:
        Affine matrices of shape (batch_size, 2, 3) for DALI warp_affine.
    """
    if rng is None:
        rng = np.random.default_rng()

    matrices = []

    for _ in range(batch_size):
        # Generate random parameters
        rotation = rng.uniform(rotation_range[0], rotation_range[1])
        scale = rng.uniform(scale_range[0], scale_range[1])
        tx = rng.uniform(translation_range[0], translation_range[1]) * size
        ty = rng.uniform(translation_range[0], translation_range[1]) * size

        # Build rotation matrix around center
        angle_rad = np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        center = size / 2.0

        # Affine matrix: rotate around center, then scale, then translate
        matrix = np.array([
            [cos_a * scale, -sin_a * scale, (1 - cos_a * scale) * center + sin_a * scale * center + tx],
            [sin_a * scale, cos_a * scale, (1 - cos_a * scale) * center - sin_a * scale * center + ty],
        ], dtype=np.float32)

        matrices.append(matrix)

    return np.stack(matrices, axis=0)


class DALIAffineGenerator:
    """
    DALI external source callback for generating affine matrices.

    Args:
        size: Image size.
        rotation_range: Min/max rotation in degrees.
        scale_range: Min/max scale factors.
        translation_range: Min/max translation (relative to size).
        seed: Random seed.
    """

    def __init__(
        self,
        size: int,
        rotation_range: Tuple[float, float] = (-10.0, 10.0),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translation_range: Tuple[float, float] = (-0.05, 0.05),
        seed: int = 42,
    ) -> None:
        self.size = size
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample_info) -> np.ndarray:
        """Generate a single affine matrix."""
        matrix = gen_dali_affine_matrix(
            size=self.size,
            batch_size=1,
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            translation_range=self.translation_range,
            rng=self.rng,
        )
        return matrix[0]  # Return single matrix (2, 3)

    def reset(self) -> None:
        """Reset the generator."""
        pass


def apply_warp_grid_numpy(
    image: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Apply warp grid to image using numpy (CPU fallback).

    Uses scipy map_coordinates for warping. This is a CPU fallback
    for testing or when DALI is not available.

    Args:
        image: Image array of shape (H, W, C) or (H, W).
        grid: Warp grid of shape (H, W, 2) in [-1, 1] range.

    Returns:
        Warped image with same shape as input.
    """
    from scipy.ndimage import map_coordinates

    h, w = image.shape[:2]
    is_color = image.ndim == 3

    # Convert grid from [-1, 1] to pixel coordinates
    grid_x = (grid[..., 0] + 1) * (w - 1) / 2
    grid_y = (grid[..., 1] + 1) * (h - 1) / 2

    if is_color:
        channels = []
        for c in range(image.shape[2]):
            warped_c = map_coordinates(
                image[..., c],
                [grid_y, grid_x],
                order=1,
                mode='nearest',
            )
            channels.append(warped_c)
        return np.stack(channels, axis=-1)
    else:
        return map_coordinates(
            image,
            [grid_y, grid_x],
            order=1,
            mode='nearest',
        )
