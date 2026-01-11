"""Base component with i18n support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gradio as gr

    from visagen.gui.i18n import I18n


@dataclass
class ComponentConfig:
    """Configuration for a GUI component."""

    key: str  # Translation key (e.g., "training.src_dir")
    default: Any = None  # Default value
    interactive: bool = True  # User can modify
    visible: bool = True  # Component visibility
    elem_id: str | None = None  # CSS element ID
    elem_classes: list[str] = field(default_factory=list)

    def get_elem_id(self) -> str:
        """Generate element ID from key if not provided."""
        return self.elem_id or self.key.replace(".", "-")


class BaseComponent:
    """Base class for i18n-aware Gradio components."""

    def __init__(
        self,
        config: ComponentConfig,
        i18n: I18n,
    ) -> None:
        self.config = config
        self.i18n = i18n
        self._component: gr.Component | None = None

    @property
    def label(self) -> str:
        """Get localized label."""
        return self.i18n.t(f"{self.config.key}.label")

    @property
    def info(self) -> str | None:
        """Get localized info/help text."""
        info_key = f"{self.config.key}.info"
        translated = self.i18n.t(info_key)
        return translated if translated != info_key else None

    @property
    def placeholder(self) -> str | None:
        """Get localized placeholder."""
        ph_key = f"{self.config.key}.placeholder"
        translated = self.i18n.t(ph_key)
        return translated if translated != ph_key else None

    def build(self) -> gr.Component:
        """Build and return the Gradio component. Override in subclasses."""
        raise NotImplementedError

    @property
    def component(self) -> gr.Component:
        """Get the built component, building if necessary."""
        if self._component is None:
            self._component = self.build()
        return self._component
