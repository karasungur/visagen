"""
Face Restoration using GFPGAN.

Provides face restoration and enhancement using GFPGAN (Generative Facial Prior GAN).
GFPGAN leverages pretrained GAN priors to restore facial details and reduce artifacts.

Features:
    - Lazy model loading for performance
    - Graceful fallback when GFPGAN unavailable
    - Configurable restoration strength
    - Support for multiple GFPGAN versions (1.2, 1.3, 1.4)
    - Automatic model weight download

Reference:
    "GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior"
    https://github.com/TencentARC/GFPGAN

Example:
    >>> from visagen.postprocess.restore import restore_face, FaceRestorer
    >>> # One-shot restoration
    >>> restored = restore_face(face_image, strength=0.6)
    >>> # Batch processing with reusable restorer
    >>> config = RestoreConfig(enabled=True, strength=0.7)
    >>> restorer = FaceRestorer(config)
    >>> for face in faces:
    ...     restored = restorer.restore(face)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from gfpgan import GFPGANer

    from visagen.postprocess.gpen import GPENRestorer

logger = logging.getLogger(__name__)

# GFPGAN availability flag (cached)
_GFPGAN_AVAILABLE: bool | None = None


def is_gfpgan_available() -> bool:
    """
    Check if GFPGAN and its dependencies are available.

    Returns:
        True if GFPGAN can be imported, False otherwise.

    Example:
        >>> if is_gfpgan_available():
        ...     print("GFPGAN is available for face restoration")
    """
    global _GFPGAN_AVAILABLE

    if _GFPGAN_AVAILABLE is None:
        try:
            from gfpgan import GFPGANer  # noqa: F401

            _GFPGAN_AVAILABLE = True
            logger.debug("GFPGAN is available")
        except ImportError:
            _GFPGAN_AVAILABLE = False
            logger.debug("GFPGAN not available. Install with: pip install gfpgan")

    return _GFPGAN_AVAILABLE


# Restore mode type
RestoreMode = Literal["gfpgan", "gpen", "none"]


@dataclass
class RestoreConfig:
    """
    Configuration for face restoration.

    Attributes:
        enabled: Whether restoration is enabled. Default: False.
        mode: Restoration mode ('gfpgan', 'gpen', 'none'). Default: 'gfpgan'.
        strength: Restoration strength (0.0-1.0). Default: 0.5.
            0.0 = original face, 1.0 = fully restored.
        upscale: Upscale factor for GFPGAN. Default: 1 (no upscale).
        arch: GFPGAN architecture version ('clean', 'RestoreFormer'). Default: 'clean'.
        model_version: GFPGAN model version (1.2, 1.3, 1.4). Default: 1.4.
        bg_upsampler: Background upsampler ('realesrgan', None). Default: None.
        gpen_model_size: GPEN model size (256, 512, 1024). Default: 512.

    Example:
        >>> config = RestoreConfig(enabled=True, strength=0.7)
        >>> config.strength
        0.7
        >>> # Use GPEN instead of GFPGAN
        >>> config = RestoreConfig(enabled=True, mode='gpen', strength=0.6)
    """

    enabled: bool = False
    mode: RestoreMode = "gfpgan"
    strength: float = 0.5
    upscale: int = 1
    arch: str = "clean"
    model_version: float = 1.4
    bg_upsampler: str | None = None
    gpen_model_size: int = 512


class FaceRestorer:
    """
    Face restoration using GFPGAN or GPEN.

    Restores and enhances face images using GFPGAN's or GPEN's generative priors.
    Supports lazy model loading and graceful fallback when models are unavailable.

    Args:
        config: Restoration configuration.
        device: Device for inference ('cuda', 'cpu', None for auto).
        model_path: Custom model path. Default: None (auto-download).

    Example:
        >>> config = RestoreConfig(enabled=True, strength=0.7)
        >>> restorer = FaceRestorer(config)
        >>> restored_face = restorer.restore(swapped_face)

        >>> # Use GPEN instead of GFPGAN
        >>> config = RestoreConfig(enabled=True, mode='gpen', strength=0.6)
        >>> restorer = FaceRestorer(config)
        >>> restored_face = restorer.restore(swapped_face)
    """

    # Model download URLs (from GFPGAN repo)
    MODEL_URLS = {
        1.2: "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
        1.3: "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        1.4: "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    }

    def __init__(
        self,
        config: RestoreConfig | None = None,
        device: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.config = config or RestoreConfig()
        self.device = device
        self.model_path = Path(model_path) if model_path else None

        # Lazy-loaded instances
        self._gfpgan: GFPGANer | None = None
        self._gpen: GPENRestorer | None = None
        self._initialization_attempted = False

    @property
    def gfpgan(self) -> GFPGANer | None:
        """Lazy-load GFPGAN model."""
        if not self.config.enabled:
            return None

        if self._gfpgan is None and not self._initialization_attempted:
            self._initialization_attempted = True
            self._gfpgan = self._load_gfpgan()

        return self._gfpgan

    def _load_gfpgan(self) -> GFPGANer | None:
        """
        Load GFPGAN model.

        Returns:
            GFPGANer instance or None if unavailable.
        """
        if not is_gfpgan_available():
            logger.warning(
                "GFPGAN not installed. Face restoration disabled. "
                "Install with: pip install gfpgan"
            )
            return None

        try:
            from gfpgan import GFPGANer

            # Determine model path
            if self.model_path and self.model_path.exists():
                model_path = str(self.model_path)
            else:
                # Use GFPGAN's auto-download mechanism
                model_path = f"GFPGANv{self.config.model_version}.pth"

            # Setup background upsampler if requested
            bg_upsampler = None
            if self.config.bg_upsampler == "realesrgan":
                bg_upsampler = self._setup_bg_upsampler()

            # Initialize GFPGAN
            gfpgan = GFPGANer(
                model_path=model_path,
                upscale=self.config.upscale,
                arch=self.config.arch,
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device=self.device,
            )

            logger.info(f"GFPGAN v{self.config.model_version} loaded successfully")
            return gfpgan

        except Exception as e:
            logger.error(f"Failed to load GFPGAN: {e}")
            return None

    def _setup_bg_upsampler(self):
        """Setup RealESRGAN background upsampler if available."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                device=self.device,
            )
            logger.debug("RealESRGAN background upsampler loaded")
            return bg_upsampler
        except ImportError:
            logger.warning("RealESRGAN not available for background upsampling")
            return None

    def restore(
        self,
        face_image: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """
        Restore face image using GFPGAN or GPEN.

        Args:
            face_image: Input face image (H, W, 3) BGR uint8 or float32 [0, 1].
            strength: Override restoration strength. Default: use config.

        Returns:
            Restored face image, same shape and dtype as input.

        Example:
            >>> restorer = FaceRestorer(RestoreConfig(enabled=True))
            >>> restored = restorer.restore(face_image)
            >>> restored.shape == face_image.shape
            True
        """
        if not self.config.enabled:
            return face_image

        # Route to appropriate restoration method
        if self.config.mode == "gpen":
            return self._restore_gpen(face_image, strength)
        elif self.config.mode == "gfpgan":
            return self._restore_gfpgan(face_image, strength)
        else:
            # mode == "none"
            return face_image

    def _restore_gpen(
        self,
        face_image: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """Restore using GPEN."""
        # Lazy load GPEN
        if self._gpen is None:
            from visagen.postprocess.gpen import GPENConfig, GPENRestorer

            model_size = self.config.gpen_model_size
            if model_size not in (256, 512, 1024):
                model_size = 512

            gpen_config = GPENConfig(
                enabled=True,
                model_size=cast(Literal[256, 512, 1024], model_size),
                strength=self.config.strength,
            )
            self._gpen = GPENRestorer(gpen_config, device=self.device)

        if self._gpen is None:
            return face_image
        return cast(np.ndarray, self._gpen.restore(face_image, strength))

    def _restore_gfpgan(
        self,
        face_image: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """Restore using GFPGAN."""
        if self.gfpgan is None:
            try:
                from visagen.postprocess.gpen import is_gpen_available

                if is_gpen_available():
                    logger.warning(
                        "GFPGAN unavailable in gfpgan mode. Falling back to GPEN."
                    )
                    return self._restore_gpen(face_image, strength)
            except Exception as e:
                logger.debug("GPEN fallback availability check failed: %s", e)

            logger.warning(
                "GFPGAN unavailable in gfpgan mode and GPEN fallback unavailable. "
                "Returning original face."
            )
            return face_image

        strength = strength if strength is not None else self.config.strength

        # Skip if strength is 0
        if strength <= 0:
            return face_image

        # Convert to uint8 if needed
        input_dtype = face_image.dtype
        input_float = input_dtype in (np.float32, np.float64)

        if input_float:
            face_uint8 = np.clip(face_image * 255, 0, 255).astype(np.uint8)
        else:
            face_uint8 = face_image.copy()

        try:
            # Run GFPGAN restoration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, restored = self.gfpgan.enhance(
                    face_uint8,
                    has_aligned=True,  # Face is already aligned
                    only_center_face=False,
                    paste_back=False,
                )

            if restored is None:
                logger.debug("GFPGAN returned None, using original")
                return face_image

            # Resize if upscaled
            if restored.shape[:2] != face_uint8.shape[:2]:
                restored = cv2.resize(
                    restored,
                    (face_uint8.shape[1], face_uint8.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Blend with original based on strength
            if strength < 1.0:
                restored = cv2.addWeighted(
                    face_uint8,
                    1.0 - strength,
                    restored,
                    strength,
                    0,
                )

            # Convert back to original dtype
            if input_float:
                return cast(np.ndarray, restored.astype(np.float32) / 255.0)
            else:
                return cast(np.ndarray, restored)

        except Exception as e:
            logger.warning(f"GFPGAN restoration failed: {e}")
            return face_image

    def is_available(self) -> bool:
        """
        Check if restoration is available and enabled.

        Returns:
            True if restoration can be performed, False otherwise.
        """
        if not self.config.enabled:
            return False

        if self.config.mode == "gpen":
            from visagen.postprocess.gpen import is_gpen_available

            return cast(bool, is_gpen_available())
        elif self.config.mode == "gfpgan":
            if is_gfpgan_available():
                return True
            try:
                from visagen.postprocess.gpen import is_gpen_available

                return cast(bool, is_gpen_available())
            except Exception:
                return False

        return False


def restore_face(
    face_image: np.ndarray,
    strength: float = 0.5,
    model_version: float = 1.4,
    device: str | None = None,
) -> np.ndarray:
    """
    Convenience function for one-shot face restoration.

    Creates a temporary FaceRestorer and restores the given face image.
    For batch processing, use FaceRestorer class directly for efficiency.

    Args:
        face_image: Input face image (H, W, 3) BGR uint8 or float32.
        strength: Restoration strength (0.0-1.0). Default: 0.5.
        model_version: GFPGAN version (1.2, 1.3, 1.4). Default: 1.4.
        device: Device for inference. Default: auto-detect.

    Returns:
        Restored face image, same shape and dtype as input.

    Example:
        >>> restored = restore_face(swapped_face, strength=0.7)
        >>> restored.shape == swapped_face.shape
        True
    """
    config = RestoreConfig(
        enabled=True,
        strength=strength,
        model_version=model_version,
    )
    restorer = FaceRestorer(config, device=device)
    return cast(np.ndarray, restorer.restore(face_image))
