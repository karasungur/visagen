"""Progress indicator components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


@dataclass
class ProgressConfig(ComponentConfig):
    """Configuration for progress indicator."""

    show_percentage: bool = True
    show_eta: bool = False
    color: str = "blue"
    height: int = 24


class ProgressIndicator(BaseComponent):
    """
    Progress bar with percentage and optional ETA display.

    Uses HTML/CSS for custom styling since Gradio's built-in
    progress is tied to function execution.
    """

    def __init__(
        self,
        config: ProgressConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.progress_config = config

    def build(self) -> gr.HTML:
        """Build progress indicator HTML component."""
        return gr.HTML(
            value=self._render(0, 100),
            elem_id=self.config.get_elem_id(),
            elem_classes=["progress-indicator", *self.config.elem_classes],
        )

    def _render(
        self,
        current: int,
        total: int,
        eta: str = "",
        status: str = "",
    ) -> str:
        """
        Render progress bar HTML.

        Args:
            current: Current progress value.
            total: Total/maximum value.
            eta: Optional ETA string.
            status: Optional status text.

        Returns:
            HTML string for progress bar.
        """
        if total <= 0:
            total = 1
        percent = min(100, max(0, (current / total) * 100))

        # Color mapping
        colors = {
            "blue": "#3b82f6",
            "green": "#22c55e",
            "red": "#ef4444",
            "yellow": "#eab308",
            "purple": "#a855f7",
        }
        bar_color = colors.get(self.progress_config.color, "#3b82f6")

        # Build percentage text
        percent_text = f"{percent:.1f}%" if self.progress_config.show_percentage else ""

        # Build ETA text
        eta_text = f" • ETA: {eta}" if eta and self.progress_config.show_eta else ""

        # Build status text
        status_text = f" • {status}" if status else ""

        height = self.progress_config.height

        return f"""
        <div style="width: 100%; margin: 8px 0;">
            <div style="
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
                font-size: 12px;
                color: #64748b;
            ">
                <span>{percent_text}{status_text}</span>
                <span>{eta_text}</span>
            </div>
            <div style="
                width: 100%;
                height: {height}px;
                background: #e2e8f0;
                border-radius: {height // 2}px;
                overflow: hidden;
            ">
                <div style="
                    width: {percent}%;
                    height: 100%;
                    background: {bar_color};
                    border-radius: {height // 2}px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """

    @staticmethod
    def update(
        current: int,
        total: int,
        eta: str = "",
        status: str = "",
        color: str = "blue",
    ) -> str:
        """
        Static method to generate progress HTML for updates.

        Args:
            current: Current progress value.
            total: Total/maximum value.
            eta: Optional ETA string.
            status: Optional status text.
            color: Progress bar color.

        Returns:
            HTML string for progress bar.
        """
        if total <= 0:
            total = 1
        percent = min(100, max(0, (current / total) * 100))

        colors = {
            "blue": "#3b82f6",
            "green": "#22c55e",
            "red": "#ef4444",
            "yellow": "#eab308",
            "purple": "#a855f7",
        }
        bar_color = colors.get(color, "#3b82f6")

        percent_text = f"{percent:.1f}%"
        eta_text = f" • ETA: {eta}" if eta else ""
        status_text = f" • {status}" if status else ""

        return f"""
        <div style="width: 100%; margin: 8px 0;">
            <div style="
                display: flex;
                justify-content: space-between;
                margin-bottom: 4px;
                font-size: 12px;
                color: #64748b;
            ">
                <span>{percent_text}{status_text}</span>
                <span>{eta_text}</span>
            </div>
            <div style="
                width: 100%;
                height: 24px;
                background: #e2e8f0;
                border-radius: 12px;
                overflow: hidden;
            ">
                <div style="
                    width: {percent}%;
                    height: 100%;
                    background: {bar_color};
                    border-radius: 12px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """


def create_indeterminate_progress(message: str = "Processing...") -> str:
    """
    Create an indeterminate (animated) progress indicator.

    Args:
        message: Status message to display.

    Returns:
        HTML string for animated progress bar.
    """
    return f"""
    <div style="width: 100%; margin: 8px 0;">
        <div style="
            margin-bottom: 4px;
            font-size: 12px;
            color: #64748b;
        ">{message}</div>
        <div style="
            width: 100%;
            height: 24px;
            background: #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                position: absolute;
                width: 30%;
                height: 100%;
                background: linear-gradient(90deg, transparent, #3b82f6, transparent);
                border-radius: 12px;
                animation: indeterminate 1.5s infinite ease-in-out;
            "></div>
        </div>
        <style>
            @keyframes indeterminate {{
                0% {{ left: -30%; }}
                100% {{ left: 100%; }}
            }}
        </style>
    </div>
    """
