"""Base tab interface and abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import gradio as gr

    from visagen.gui.i18n import I18n
    from visagen.gui.i18n.locales import I18nSection
    from visagen.gui.state.app_state import AppState


class TabProtocol(Protocol):
    """Protocol defining the tab interface."""

    @property
    def id(self) -> str:
        """Unique tab identifier."""
        ...

    @property
    def title(self) -> str:
        """Localized tab title."""
        ...

    def create(self) -> gr.Tab:
        """Create and return the Gradio Tab."""
        ...


class BaseTab(ABC):
    """
    Abstract base class for all GUI tabs.

    Provides:
    - Consistent interface for tab creation
    - Access to app state and i18n
    - Common utility methods

    Subclasses must implement:
    - id: Unique identifier
    - _build_content(): Build tab UI elements
    - _setup_events(): Connect event handlers

    Example:
        class TrainingTab(BaseTab):
            @property
            def id(self) -> str:
                return "training"

            def _build_content(self) -> dict[str, gr.Component]:
                # Build UI components
                ...

            def _setup_events(self, components: dict[str, gr.Component]) -> None:
                # Wire up event handlers
                ...
    """

    def __init__(
        self,
        app_state: AppState,
        i18n: I18n,
    ) -> None:
        """
        Initialize tab with dependencies.

        Args:
            app_state: Central application state
            i18n: Internationalization instance
        """
        self.state = app_state
        self.i18n = i18n
        self._components: dict[str, Any] = {}

    @property
    @abstractmethod
    def id(self) -> str:
        """Return unique tab identifier (e.g., 'training', 'inference')."""
        pass

    @property
    def title(self) -> str:
        """Return localized tab title."""
        return self.i18n.t(f"{self.id}.title")

    @property
    def i18n_section(self) -> I18nSection:
        """Get scoped i18n for this tab."""
        return self.i18n.section(self.id)

    def t(self, key: str, **kwargs: Any) -> str:
        """Shorthand for tab-scoped translation."""
        return self.i18n.t(f"{self.id}.{key}", **kwargs)

    @abstractmethod
    def _build_content(self) -> dict[str, Any]:
        """
        Build tab UI content.

        Returns:
            Dictionary mapping component names to Gradio components.
        """
        pass

    @abstractmethod
    def _setup_events(self, components: dict[str, Any]) -> None:
        """
        Set up event handlers for components.

        Args:
            components: Dictionary of named components from _build_content()
        """
        pass

    def create(self) -> gr.Tab:
        """
        Create the complete Gradio Tab.

        This is the main entry point called by the app factory.

        Returns:
            Configured gr.Tab instance.
        """
        import gradio as gr

        with gr.Tab(self.title, id=self.id) as tab:
            self._components = self._build_content()
            self._setup_events(self._components)

        return tab

    @property
    def components(self) -> dict[str, Any]:
        """Access components after tab creation."""
        return self._components
