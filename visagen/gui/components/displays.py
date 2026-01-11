"""Display components (logs, status, images)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr

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
