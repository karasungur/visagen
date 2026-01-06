"""Visagen Data - Dataset and DataLoader Utilities."""

from visagen.data.augmentations import FaceAugmentationPipeline, SimpleAugmentation
from visagen.data.dali_loader import (
    DALIFaceDataModule,
    benchmark_dataloaders,
    create_dali_datamodule,
)

# DALI imports (with graceful fallback)
from visagen.data.dali_pipeline import (
    DALI_AVAILABLE,
    check_dali_available,
    create_dali_iterator,
)
from visagen.data.dali_warp import (
    DALIAffineGenerator,
    DALIWarpGridGenerator,
    gen_dali_affine_matrix,
    gen_dali_warp_grid,
)
from visagen.data.datamodule import FaceDataModule, PairedFaceDataset
from visagen.data.face_dataset import FaceDataset, SimpleFaceDataset
from visagen.data.face_sample import FaceSample
from visagen.data.noise_dataset import RandomNoiseDataset
from visagen.data.pretrain_datamodule import (
    PRETRAIN_AUGMENTATION_CONFIG,
    PretrainDataModule,
    PretrainDataset,
    scan_images_flat,
    scan_images_recursive,
)
from visagen.data.sequence_dataset import (
    PairedSequenceDataset,
    RandomSequenceDataset,
    SequenceFaceDataset,
)
from visagen.data.warp import gen_warp_params, warp_by_params

__all__ = [
    # Core datasets
    "RandomNoiseDataset",
    "FaceSample",
    "FaceDataset",
    "SimpleFaceDataset",
    # Sequence datasets (temporal)
    "SequenceFaceDataset",
    "PairedSequenceDataset",
    "RandomSequenceDataset",
    # Augmentations
    "FaceAugmentationPipeline",
    "SimpleAugmentation",
    # Warp utilities
    "gen_warp_params",
    "warp_by_params",
    # PyTorch DataModule
    "PairedFaceDataset",
    "FaceDataModule",
    # Pretrain DataModule
    "PretrainDataset",
    "PretrainDataModule",
    "PRETRAIN_AUGMENTATION_CONFIG",
    "scan_images_recursive",
    "scan_images_flat",
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
