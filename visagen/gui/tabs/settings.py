"""Settings tab for application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    SliderConfig,
    SliderInput,
)
from visagen.gui.tabs.base import BaseTab

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class SettingsTab(BaseTab):
    """
    Settings tab for configuring application preferences.

    Allows users to configure:
    - Device selection (auto, cuda, cpu)
    - Default batch size
    - Number of workers
    - Locale (language)

    Changes are persisted to disk automatically.
    """

    # Default path for settings file
    DEFAULT_SETTINGS_PATH = Path("./settings.json")

    def __init__(
        self,
        app_state: AppState,
        i18n: I18n,
        settings_path: Path | str | None = None,
    ) -> None:
        """
        Initialize settings tab.

        Args:
            app_state: Central application state
            i18n: Internationalization instance
            settings_path: Optional custom path for settings file
        """
        super().__init__(app_state, i18n)
        self._settings_path = (
            Path(settings_path) if settings_path else self.DEFAULT_SETTINGS_PATH
        )

    @property
    def id(self) -> str:
        """Return unique tab identifier."""
        return "settings"

    def _build_content(self) -> dict[str, Any]:
        """Build settings UI components."""
        components: dict[str, Any] = {}

        with gr.Column():
            # Section: Device Settings
            gr.Markdown(f"### {self.t('device_section')}")

            # Device selection dropdown
            device_dropdown = DropdownInput(
                config=DropdownConfig(
                    key="settings.device",
                    choices=["auto", "cuda", "cpu"],
                    default=self.state.settings.device,
                ),
                i18n=self.i18n,
            )
            components["device"] = device_dropdown.build()

            # Section: Performance Settings
            gr.Markdown(f"### {self.t('performance_section')}")

            # Batch size slider
            batch_size_slider = SliderInput(
                config=SliderConfig(
                    key="settings.batch_size",
                    default=self.state.settings.default_batch_size,
                    minimum=1,
                    maximum=64,
                    step=1,
                ),
                i18n=self.i18n,
            )
            components["batch_size"] = batch_size_slider.build()

            # Number of workers slider
            num_workers_slider = SliderInput(
                config=SliderConfig(
                    key="settings.num_workers",
                    default=self.state.settings.num_workers,
                    minimum=0,
                    maximum=16,
                    step=1,
                ),
                i18n=self.i18n,
            )
            components["num_workers"] = num_workers_slider.build()

            # Section: Language Settings
            gr.Markdown(f"### {self.t('language_section')}")

            # Locale selection dropdown
            locale_dropdown = DropdownInput(
                config=DropdownConfig(
                    key="settings.locale",
                    choices=["en", "tr"],  # Add more locales as available
                    default=self.state.settings.locale,
                ),
                i18n=self.i18n,
            )
            components["locale"] = locale_dropdown.build()

            # Status message
            components["status"] = gr.Textbox(
                label=self.t("status.label"),
                value="",
                interactive=False,
                visible=True,
            )

        return components

    def _setup_events(self, components: dict[str, Any]) -> None:
        """Set up event handlers for settings changes."""

        def update_device(value: str) -> str:
            """Update device setting."""
            self.state.settings.device = value
            self._save_settings()
            return self.t("status.saved")

        def update_batch_size(value: int) -> str:
            """Update batch size setting."""
            self.state.settings.default_batch_size = int(value)
            self._save_settings()
            return self.t("status.saved")

        def update_num_workers(value: int) -> str:
            """Update number of workers setting."""
            self.state.settings.num_workers = int(value)
            self._save_settings()
            return self.t("status.saved")

        def update_locale(value: str) -> str:
            """Update locale setting."""
            self.state.settings.locale = value
            self.i18n.locale = value  # Update i18n instance
            self._save_settings()
            return self.t("status.saved_reload")

        # Wire up change events
        components["device"].change(
            fn=update_device,
            inputs=[components["device"]],
            outputs=[components["status"]],
        )

        components["batch_size"].change(
            fn=update_batch_size,
            inputs=[components["batch_size"]],
            outputs=[components["status"]],
        )

        components["num_workers"].change(
            fn=update_num_workers,
            inputs=[components["num_workers"]],
            outputs=[components["status"]],
        )

        components["locale"].change(
            fn=update_locale,
            inputs=[components["locale"]],
            outputs=[components["status"]],
        )

    def _save_settings(self) -> None:
        """Persist settings to disk."""
        self.state.settings.save(self._settings_path)
