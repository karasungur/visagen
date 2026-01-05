"""
Pose-based sorting methods.

Provides sorting by face yaw (left-right) and pitch (up-down) angles.
"""

from typing import TYPE_CHECKING

import numpy as np

from visagen.sorting.base import SortMethod
from visagen.vision.aligner import FaceAligner

if TYPE_CHECKING:
    from visagen.vision.dflimg import FaceMetadata


class YawSorter(SortMethod):
    """
    Sort by face yaw angle (left-right rotation).

    Uses 3D pose estimation from 68-point landmarks.
    Yaw ranges from -π/2 (looking right) to +π/2 (looking left).

    Sorted from most left-facing to most right-facing.
    """

    name = "face-yaw"
    description = "Sort by face yaw angle (left-right rotation)"
    requires_dfl_metadata = True

    def __init__(self) -> None:
        self._aligner = FaceAligner()

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute yaw angle."""
        if metadata is None or metadata.landmarks is None:
            return 0.0

        try:
            size = image.shape[0]  # Aligned image is square
            pitch, yaw, roll = self._aligner.estimate_pitch_yaw_roll(
                metadata.landmarks, size=size
            )
            return float(yaw)
        except Exception:
            return 0.0


class PitchSorter(SortMethod):
    """
    Sort by face pitch angle (up-down tilt).

    Uses 3D pose estimation from 68-point landmarks.
    Pitch ranges from -π/2 (looking up) to +π/2 (looking down).

    Sorted from most upward-facing to most downward-facing.
    """

    name = "face-pitch"
    description = "Sort by face pitch angle (up-down tilt)"
    requires_dfl_metadata = True

    def __init__(self) -> None:
        self._aligner = FaceAligner()

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute pitch angle."""
        if metadata is None or metadata.landmarks is None:
            return 0.0

        try:
            size = image.shape[0]
            pitch, yaw, roll = self._aligner.estimate_pitch_yaw_roll(
                metadata.landmarks, size=size
            )
            return float(pitch)
        except Exception:
            return 0.0


class RollSorter(SortMethod):
    """
    Sort by face roll angle (head tilt).

    Uses 3D pose estimation from 68-point landmarks.
    Roll ranges from -π/2 to +π/2.
    """

    name = "face-roll"
    description = "Sort by face roll angle (head tilt)"
    requires_dfl_metadata = True

    def __init__(self) -> None:
        self._aligner = FaceAligner()

    def compute_score(
        self,
        image: np.ndarray,
        metadata: "FaceMetadata | None" = None,
    ) -> float:
        """Compute roll angle."""
        if metadata is None or metadata.landmarks is None:
            return 0.0

        try:
            size = image.shape[0]
            pitch, yaw, roll = self._aligner.estimate_pitch_yaw_roll(
                metadata.landmarks, size=size
            )
            return float(roll)
        except Exception:
            return 0.0
