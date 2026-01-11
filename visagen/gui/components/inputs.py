"""Input components with i18n support and validation feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


# Validation result type
ValidationResult = tuple[bool, str]  # (is_valid, error_message)


def create_validation_feedback(
    is_valid: bool,
    message: str = "",
    show_success: bool = False,
) -> str:
    """
    Create HTML validation feedback.

    Args:
        is_valid: Whether the input is valid.
        message: Error or success message.
        show_success: Whether to show success indicator.

    Returns:
        HTML string for validation feedback.
    """
    if is_valid:
        if show_success and message:
            return f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 6px 10px;
                background: #dcfce7;
                color: #166534;
                border-radius: 6px;
                font-size: 12px;
                margin-top: 4px;
            ">
                <span>✓</span>
                <span>{message}</span>
            </div>
            """
        return ""

    if not message:
        return ""

    return f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        background: #fee2e2;
        color: #991b1b;
        border-radius: 6px;
        font-size: 12px;
        margin-top: 4px;
    ">
        <span>✕</span>
        <span>{message}</span>
    </div>
    """


@dataclass
class PathInputConfig(ComponentConfig):
    """Configuration for path input."""

    path_type: Literal["file", "directory"] = "file"
    file_types: list[str] = field(default_factory=list)  # e.g., [".ckpt", ".pt"]
    must_exist: bool = False
    show_validation_feedback: bool = True


class PathInput(BaseComponent):
    """Path input with validation and i18n support."""

    def __init__(
        self,
        config: PathInputConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.path_config = config

    def build(self) -> gr.Textbox:
        """Build path input textbox."""
        return gr.Textbox(
            label=self.label,
            placeholder=self.placeholder,
            info=self.info,
            value=self.config.default or "",
            interactive=self.config.interactive,
            visible=self.config.visible,
            elem_id=self.config.get_elem_id(),
            elem_classes=self.config.elem_classes,
        )

    def validate(self, value: str) -> ValidationResult:
        """Validate path value."""
        if not value:
            if self.path_config.must_exist:
                return False, self.i18n.t("errors.path_required")
            return True, ""

        path = Path(value)

        if self.path_config.must_exist and not path.exists():
            return False, self.i18n.t("errors.path_not_found")

        if (
            self.path_config.path_type == "directory"
            and path.exists()
            and not path.is_dir()
        ):
            return False, self.i18n.t("errors.not_a_directory")

        if (
            self.path_config.file_types
            and path.suffix not in self.path_config.file_types
        ):
            return False, self.i18n.t(
                "errors.invalid_file_type", types=", ".join(self.path_config.file_types)
            )

        return True, ""

    def validate_with_feedback(self, value: str) -> str:
        """
        Validate path and return HTML feedback.

        Args:
            value: Path string to validate.

        Returns:
            HTML string with validation feedback.
        """
        if not self.path_config.show_validation_feedback:
            return ""

        is_valid, error_message = self.validate(value)
        return create_validation_feedback(is_valid, error_message)


@dataclass
class SliderConfig(ComponentConfig):
    """Configuration for slider input."""

    minimum: float = 0
    maximum: float = 100
    step: float = 1


class SliderInput(BaseComponent):
    """Slider input with i18n support."""

    def __init__(
        self,
        config: SliderConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.slider_config = config

    def build(self) -> gr.Slider:
        """Build slider component."""
        return gr.Slider(
            label=self.label,
            info=self.info,
            minimum=self.slider_config.minimum,
            maximum=self.slider_config.maximum,
            step=self.slider_config.step,
            value=self.config.default or self.slider_config.minimum,
            interactive=self.config.interactive,
            visible=self.config.visible,
            elem_id=self.config.get_elem_id(),
            elem_classes=self.config.elem_classes,
        )


@dataclass
class DropdownConfig(ComponentConfig):
    """Configuration for dropdown input."""

    choices: list[str] = field(default_factory=list)
    multiselect: bool = False


class DropdownInput(BaseComponent):
    """Dropdown with i18n-aware choices."""

    def __init__(
        self,
        config: DropdownConfig,
        i18n: I18n,
    ) -> None:
        super().__init__(config, i18n)
        self.dropdown_config = config

    def build(self) -> gr.Dropdown:
        """Build dropdown with localized choice labels."""
        # Choices can be localized via i18n keys
        choices = []
        for choice in self.dropdown_config.choices:
            # Try to get localized label, fall back to raw value
            label_key = f"{self.config.key}.choices.{choice}"
            label = self.i18n.t(label_key)
            if label == label_key:
                label = choice
            choices.append((label, choice))  # (display, value) tuple

        return gr.Dropdown(
            label=self.label,
            info=self.info,
            choices=choices,
            value=self.config.default,
            multiselect=self.dropdown_config.multiselect,
            interactive=self.config.interactive,
            visible=self.config.visible,
            elem_id=self.config.get_elem_id(),
            elem_classes=self.config.elem_classes,
        )


# Convenience validation functions
def validate_path_exists(path: str) -> ValidationResult:
    """Quick validation for path existence."""
    if not path:
        return False, "Path is required"
    if not Path(path).exists():
        return False, "Path not found"
    return True, ""


def validate_directory(path: str) -> ValidationResult:
    """Quick validation for directory path."""
    if not path:
        return False, "Directory path is required"
    p = Path(path)
    if not p.exists():
        return False, "Directory not found"
    if not p.is_dir():
        return False, "Path is not a directory"
    return True, ""


def validate_file_type(path: str, allowed_types: list[str]) -> ValidationResult:
    """Quick validation for file type."""
    if not path:
        return True, ""
    if Path(path).suffix not in allowed_types:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_types)}"
    return True, ""
