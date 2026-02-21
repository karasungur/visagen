"""Control components (buttons, process controls)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


@dataclass
class ButtonConfig(ComponentConfig):
    """Configuration for button."""

    variant: Literal["primary", "secondary", "stop"] = "secondary"
    size: Literal["sm", "md", "lg"] = "md"


class ActionButton(BaseComponent):
    """Single action button with i18n support."""

    def __init__(
        self,
        config: ButtonConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.button_config = config

    def build(self) -> gr.Button:
        """Build button component."""
        return gr.Button(
            value=self.label,
            variant=self.button_config.variant,
            size=self.button_config.size,
            interactive=self.config.interactive,
            visible=self.config.visible,
            elem_id=self.config.get_elem_id(),
            elem_classes=self.config.elem_classes,
        )


class ProcessControl:
    """Start/Stop button pair for long-running processes."""

    def __init__(
        self,
        key: str,
        i18n: I18n,
    ) -> None:
        self.key = key
        self.i18n = i18n
        self._start_btn: gr.Button | None = None
        self._stop_btn: gr.Button | None = None

    def build(self) -> tuple[gr.Button, gr.Button]:
        """Build start and stop buttons."""
        self._start_btn = gr.Button(
            value=self.i18n.t(f"{self.key}.start"),
            variant="primary",
            elem_id=f"{self.key}-start",
        )
        self._stop_btn = gr.Button(
            value=self.i18n.t(f"{self.key}.stop"),
            variant="stop",
            elem_id=f"{self.key}-stop",
        )
        return self._start_btn, self._stop_btn

    @property
    def start_button(self) -> gr.Button:
        if self._start_btn is None:
            self.build()
        assert self._start_btn is not None
        return self._start_btn

    @property
    def stop_button(self) -> gr.Button:
        if self._stop_btn is None:
            self.build()
        assert self._stop_btn is not None
        return self._stop_btn
