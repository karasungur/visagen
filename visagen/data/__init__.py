"""Visagen Data - Dataset and DataLoader Utilities."""

from visagen.data.noise_dataset import RandomNoiseDataset
from visagen.data.face_sample import FaceSample
from visagen.data.face_dataset import FaceDataset, SimpleFaceDataset
from visagen.data.augmentations import FaceAugmentationPipeline, SimpleAugmentation
from visagen.data.warp import gen_warp_params, warp_by_params
from visagen.data.datamodule import PairedFaceDataset, FaceDataModule

# DALI imports (with graceful fallback)
from visagen.data.dali_pipeline import (
    DALI_AVAILABLE,
    check_dali_available,
    create_dali_iterator,
)
from visagen.data.dali_warp import (
    gen_dali_warp_grid,
    DALIWarpGridGenerator,
    gen_dali_affine_matrix,
    DALIAffineGenerator,
)
from visagen.data.dali_loader import (
    DALIFaceDataModule,
    create_dali_datamodule,
    benchmark_dataloaders,
)

__all__ = [
    # Core datasets
    "RandomNoiseDataset",
    "FaceSample",
    "FaceDataset",
    "SimpleFaceDataset",
    # Augmentations
    "FaceAugmentationPipeline",
    "SimpleAugmentation",
    # Warp utilities
    "gen_warp_params",
    "warp_by_params",
    # PyTorch DataModule
    "PairedFaceDataset",
    "FaceDataModule",
    # DALI pipeline
    "DALI_AVAILABLE",
    "check_dali_available",
    "create_dali_iterator",
    # DALI warp
    "gen_dali_warp_grid",
    "DALIWarpGridGenerator",
    "gen_dali_affine_matrix",
    "DALIAffineGenerator",
    # DALI DataModule
    "DALIFaceDataModule",
    "create_dali_datamodule",
    "benchmark_dataloaders",
]
