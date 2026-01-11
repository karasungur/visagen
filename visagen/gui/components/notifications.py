"""Notification and toast components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


class NotificationType(Enum):
    """Notification severity levels."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class NotificationConfig(ComponentConfig):
    """Configuration for notification component."""

    auto_dismiss: bool = True
    dismiss_after_ms: int = 5000
    position: str = "top-right"  # top-right, top-left, bottom-right, bottom-left


class NotificationManager:
    """
    Toast notification manager.

    Provides methods to show success, error, warning, and info toasts.
    Uses HTML/JS for animation and auto-dismiss functionality.
    """

    def __init__(self, position: str = "top-right") -> None:
        """
        Initialize notification manager.

        Args:
            position: Toast position on screen.
        """
        self.position = position
        self._toast_id = 0

    def success(self, message: str, duration: int = 5000) -> str:
        """
        Create success toast HTML.

        Args:
            message: Toast message.
            duration: Auto-dismiss duration in ms.

        Returns:
            HTML string for toast.
        """
        return self._create_toast(message, NotificationType.SUCCESS, duration)

    def error(self, message: str, duration: int = 8000) -> str:
        """Create error toast HTML."""
        return self._create_toast(message, NotificationType.ERROR, duration)

    def warning(self, message: str, duration: int = 6000) -> str:
        """Create warning toast HTML."""
        return self._create_toast(message, NotificationType.WARNING, duration)

    def info(self, message: str, duration: int = 5000) -> str:
        """Create info toast HTML."""
        return self._create_toast(message, NotificationType.INFO, duration)

    def _create_toast(
        self,
        message: str,
        toast_type: NotificationType,
        duration: int,
    ) -> str:
        """
        Create toast HTML with styling and auto-dismiss.

        Args:
            message: Toast message.
            toast_type: Type of notification.
            duration: Auto-dismiss duration in ms.

        Returns:
            HTML string for toast.
        """
        self._toast_id += 1
        toast_id = f"toast-{self._toast_id}"

        # Icon mapping
        icons = {
            NotificationType.SUCCESS: "✓",
            NotificationType.ERROR: "✕",
            NotificationType.WARNING: "⚠",
            NotificationType.INFO: "ℹ",
        }

        # Color mapping
        colors = {
            NotificationType.SUCCESS: "#22c55e",
            NotificationType.ERROR: "#ef4444",
            NotificationType.WARNING: "#f59e0b",
            NotificationType.INFO: "#3b82f6",
        }

        icon = icons.get(toast_type, "ℹ")
        color = colors.get(toast_type, "#3b82f6")

        # Position styles
        position_styles = {
            "top-right": "top: 20px; right: 20px;",
            "top-left": "top: 20px; left: 20px;",
            "bottom-right": "bottom: 20px; right: 20px;",
            "bottom-left": "bottom: 20px; left: 20px;",
        }
        pos_style = position_styles.get(self.position, "top: 20px; right: 20px;")

        return f"""
        <div id="{toast_id}" style="
            position: fixed;
            {pos_style}
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 20px;
            background: {color};
            color: white;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            animation: toastSlideIn 0.3s ease;
            cursor: pointer;
        " onclick="this.remove()">
            <span style="font-size: 18px;">{icon}</span>
            <span>{message}</span>
        </div>
        <style>
            @keyframes toastSlideIn {{
                from {{
                    transform: translateX(100%);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}
            @keyframes toastSlideOut {{
                from {{
                    transform: translateX(0);
                    opacity: 1;
                }}
                to {{
                    transform: translateX(100%);
                    opacity: 0;
                }}
            }}
        </style>
        <script>
            setTimeout(function() {{
                var toast = document.getElementById('{toast_id}');
                if (toast) {{
                    toast.style.animation = 'toastSlideOut 0.3s ease forwards';
                    setTimeout(function() {{
                        toast.remove();
                    }}, 300);
                }}
            }}, {duration});
        </script>
        """


class NotificationDisplay(BaseComponent):
    """
    Notification display component.

    Renders in a fixed position and shows toast notifications.
    """

    def __init__(
        self,
        config: NotificationConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.notification_config = config
        self.manager = NotificationManager(position=config.position)

    def build(self) -> gr.HTML:
        """Build notification container."""
        return gr.HTML(
            value="",
            elem_id=self.config.get_elem_id(),
            elem_classes=["notification-container", *self.config.elem_classes],
        )

    def show_success(self, message: str) -> str:
        """Show success notification."""
        return self.manager.success(message, self.notification_config.dismiss_after_ms)

    def show_error(self, message: str) -> str:
        """Show error notification."""
        return self.manager.error(
            message, self.notification_config.dismiss_after_ms + 3000
        )

    def show_warning(self, message: str) -> str:
        """Show warning notification."""
        return self.manager.warning(
            message, self.notification_config.dismiss_after_ms + 1000
        )

    def show_info(self, message: str) -> str:
        """Show info notification."""
        return self.manager.info(message, self.notification_config.dismiss_after_ms)


# Convenience functions for quick notifications
def toast_success(message: str, duration: int = 5000) -> str:
    """Create a success toast HTML."""
    return NotificationManager().success(message, duration)


def toast_error(message: str, duration: int = 8000) -> str:
    """Create an error toast HTML."""
    return NotificationManager().error(message, duration)


def toast_warning(message: str, duration: int = 6000) -> str:
    """Create a warning toast HTML."""
    return NotificationManager().warning(message, duration)


def toast_info(message: str, duration: int = 5000) -> str:
    """Create an info toast HTML."""
    return NotificationManager().info(message, duration)


# Status banner for inline notifications
def create_status_banner(
    message: str,
    status: str = "info",
    dismissible: bool = True,
) -> str:
    """
    Create an inline status banner.

    Args:
        message: Banner message.
        status: Status type (success, error, warning, info).
        dismissible: Whether banner can be dismissed.

    Returns:
        HTML string for banner.
    """
    colors = {
        "success": ("#dcfce7", "#166534", "#22c55e"),
        "error": ("#fee2e2", "#991b1b", "#ef4444"),
        "warning": ("#fef3c7", "#92400e", "#f59e0b"),
        "info": ("#dbeafe", "#1e40af", "#3b82f6"),
    }

    bg, text, border = colors.get(status, colors["info"])

    icons = {
        "success": "✓",
        "error": "✕",
        "warning": "⚠",
        "info": "ℹ",
    }
    icon = icons.get(status, "ℹ")

    dismiss_btn = ""
    if dismissible:
        dismiss_btn = """
        <button onclick="this.parentElement.remove()" style="
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: inherit;
            opacity: 0.7;
        ">✕</button>
        """

    return f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 16px;
        background: {bg};
        color: {text};
        border-left: 4px solid {border};
        border-radius: 4px;
        font-size: 14px;
        margin: 8px 0;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 16px;">{icon}</span>
            <span>{message}</span>
        </div>
        {dismiss_btn}
    </div>
    """
