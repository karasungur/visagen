"""Display components (logs, status, images)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import cv2
import gradio as gr
import numpy as np

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


@dataclass
class LogOutputConfig(ComponentConfig):
    """Configuration for log output."""

    lines: int = 15
    max_lines: int = 30


class LogOutput(BaseComponent):
    """Log output display."""

    def __init__(
        self,
        config: LogOutputConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.log_config = config

    def build(self) -> gr.Textbox:
        """Build log output textbox."""
        return gr.Textbox(
            label=self.label,
            lines=self.log_config.lines,
            max_lines=self.log_config.max_lines,
            interactive=False,
            elem_id=self.config.get_elem_id(),
            elem_classes=["log-output", *self.config.elem_classes],
        )


class StatusDisplay(BaseComponent):
    """Status message display."""

    def build(self) -> gr.Textbox:
        """Build status display."""
        return gr.Textbox(
            label=self.label,
            value=self.config.default or "",
            interactive=False,
            elem_id=self.config.get_elem_id(),
            elem_classes=["status-display", *self.config.elem_classes],
        )


@dataclass
class ImagePreviewConfig(ComponentConfig):
    """Configuration for image preview."""

    height: int = 400
    show_label: bool = True


class ImagePreview(BaseComponent):
    """Image preview component."""

    def __init__(
        self,
        config: ImagePreviewConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.image_config = config

    def build(self) -> gr.Image:
        """Build image preview."""
        return gr.Image(
            label=self.label if self.image_config.show_label else None,
            type="numpy",
            height=self.image_config.height,
            interactive=self.config.interactive,
            visible=self.config.visible,
            elem_id=self.config.get_elem_id(),
            elem_classes=self.config.elem_classes,
        )


@dataclass
class GalleryPreviewConfig(ComponentConfig):
    """Configuration for gallery preview."""

    columns: int = 4
    rows: int = 2
    height: int = 400
    object_fit: Literal["contain", "cover", "fill"] = "contain"
    allow_preview: bool = True


class GalleryPreview(BaseComponent):
    """Gallery component for displaying multiple images."""

    def __init__(
        self,
        config: GalleryPreviewConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.gallery_config = config

    def build(self) -> gr.Gallery:
        """Build gallery preview."""
        return gr.Gallery(
            label=self.label,
            columns=self.gallery_config.columns,
            rows=self.gallery_config.rows,
            height=self.gallery_config.height,
            object_fit=self.gallery_config.object_fit,
            allow_preview=self.gallery_config.allow_preview,
            interactive=False,
            elem_id=self.config.get_elem_id(),
            elem_classes=["gallery-preview", *self.config.elem_classes],
        )


def create_mask_overlay(
    face: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Create side-by-side face + mask overlay image.

    Args:
        face: BGR face image (H, W, 3).
        mask: Binary mask (H, W) values 0-255.
        alpha: Mask overlay transparency.
        color: BGR color for mask overlay.

    Returns:
        Combined image (H, W*2, 3) with face left, overlay right.
    """
    # Normalize mask to 0-1
    if mask.max() > 1:
        mask_norm = mask.astype(np.float32) / 255.0
    else:
        mask_norm = mask.astype(np.float32)

    if len(mask_norm.shape) == 3:
        mask_norm = mask_norm[:, :, 0]

    # Resize mask if dimensions don't match
    if mask_norm.shape[:2] != face.shape[:2]:
        mask_norm = cast(
            np.ndarray,
            cv2.resize(mask_norm, (face.shape[1], face.shape[0])),
        )

    # Create colored overlay
    overlay = face.copy()
    for c in range(3):
        overlay[:, :, c] = (
            face[:, :, c] * (1 - alpha * mask_norm) + color[c] * alpha * mask_norm
        ).astype(np.uint8)

    # Side by side
    return np.hstack([face, overlay])
