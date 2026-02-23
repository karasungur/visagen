"""Workflow indicator components for step-by-step guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import gradio as gr

from .base import BaseComponent, ComponentConfig

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n


class WorkflowStep(Enum):
    """Visagen workflow steps."""

    EXTRACT = "extract"
    SORT = "sort"
    TRAIN = "train"
    MERGE = "merge"
    EXPORT = "export"


@dataclass
class StepInfo:
    """Information about a workflow step."""

    key: WorkflowStep
    icon: str
    label_key: str  # i18n key for label
    description_key: str  # i18n key for description


# Default workflow steps
DEFAULT_STEPS: list[StepInfo] = [
    StepInfo(
        WorkflowStep.EXTRACT,
        "1️⃣",
        "workflow.steps.extract",
        "workflow.descriptions.extract",
    ),
    StepInfo(
        WorkflowStep.SORT, "2️⃣", "workflow.steps.sort", "workflow.descriptions.sort"
    ),
    StepInfo(
        WorkflowStep.TRAIN, "3️⃣", "workflow.steps.train", "workflow.descriptions.train"
    ),
    StepInfo(
        WorkflowStep.MERGE, "4️⃣", "workflow.steps.merge", "workflow.descriptions.merge"
    ),
    StepInfo(
        WorkflowStep.EXPORT,
        "5️⃣",
        "workflow.steps.export",
        "workflow.descriptions.export",
    ),
]


@dataclass
class WorkflowConfig(ComponentConfig):
    """Configuration for workflow indicator."""

    steps: list[StepInfo] = field(default_factory=lambda: DEFAULT_STEPS.copy())
    show_descriptions: bool = False
    compact: bool = True


class WorkflowIndicator(BaseComponent):
    """
    Visual workflow step indicator.

    Shows the current step in the Visagen workflow:
    Extract → Sort → Train → Merge → Export

    Provides visual feedback on:
    - Current active step
    - Completed steps
    - Pending steps
    """

    def __init__(
        self,
        config: WorkflowConfig,
        i18n: I18n,
        current_step: WorkflowStep = WorkflowStep.EXTRACT,
    ) -> None:
        super().__init__(config, i18n)
        self.workflow_config = config
        self.current_step = current_step
        self.completed_steps: set[WorkflowStep] = set()

    def build(self) -> gr.HTML:
        """Build workflow indicator HTML component."""
        return gr.HTML(
            value=self._render(),
            elem_id=self.config.get_elem_id(),
            elem_classes=["workflow-indicator", *self.config.elem_classes],
        )

    def set_current_step(self, step: WorkflowStep) -> None:
        """Set the current active step."""
        self.current_step = step

    def mark_completed(self, step: WorkflowStep) -> None:
        """Mark a step as completed."""
        self.completed_steps.add(step)

    def reset(self) -> None:
        """Reset workflow to initial state."""
        self.current_step = WorkflowStep.EXTRACT
        self.completed_steps.clear()

    def _render(self) -> str:
        """
        Render workflow indicator HTML.

        Returns:
            HTML string for workflow visualization.
        """
        if self.workflow_config.compact:
            return self._render_compact()
        return self._render_full()

    def _render_compact(self) -> str:
        """Render compact horizontal workflow indicator."""
        steps_html = []

        for i, step_info in enumerate(self.workflow_config.steps):
            is_current = step_info.key == self.current_step
            is_completed = step_info.key in self.completed_steps

            # Determine step styling
            if is_completed:
                bg_color = "#dcfce7"
                text_color = "#166534"
                border_color = "#22c55e"
            elif is_current:
                bg_color = "#3b82f6"
                text_color = "white"
                border_color = "#3b82f6"
            else:
                bg_color = "#f1f5f9"
                text_color = "#64748b"
                border_color = "#e2e8f0"

            # Get translated label
            label = self.i18n.t(step_info.label_key)

            step_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 8px 14px;
                background: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 20px;
                font-size: 13px;
                font-weight: {"600" if is_current else "500"};
                transition: all 0.2s ease;
            ">
                <span>{step_info.icon}</span>
                <span>{label}</span>
                {'<span style="margin-left: 4px;">✓</span>' if is_completed else ""}
            </div>
            """
            steps_html.append(step_html)

            # Add connector arrow (except for last step)
            if i < len(self.workflow_config.steps) - 1:
                steps_html.append("""
                <div style="
                    color: #94a3b8;
                    font-size: 16px;
                ">→</div>
                """)

        return f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 16px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            flex-wrap: wrap;
        ">
            {"".join(steps_html)}
        </div>
        """

    def _render_full(self) -> str:
        """Render full workflow indicator with descriptions."""
        steps_html = []

        for step_info in self.workflow_config.steps:
            is_current = step_info.key == self.current_step
            is_completed = step_info.key in self.completed_steps

            # Determine step styling
            if is_completed:
                bg_color = "#dcfce7"
                text_color = "#166534"
                icon_bg = "#22c55e"
            elif is_current:
                bg_color = "#eff6ff"
                text_color = "#1e40af"
                icon_bg = "#3b82f6"
            else:
                bg_color = "#f8fafc"
                text_color = "#64748b"
                icon_bg = "#94a3b8"

            # Get translated label and description
            label = self.i18n.t(step_info.label_key)
            description = (
                self.i18n.t(step_info.description_key)
                if self.workflow_config.show_descriptions
                else ""
            )

            description_html = ""
            if description:
                description_html = f"""
                <div style="
                    font-size: 12px;
                    color: #64748b;
                    margin-top: 4px;
                ">{description}</div>
                """

            step_html = f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 12px;
                padding: 12px 16px;
                background: {bg_color};
                border-radius: 8px;
                margin-bottom: 8px;
            ">
                <div style="
                    width: 32px;
                    height: 32px;
                    background: {icon_bg};
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    flex-shrink: 0;
                ">
                    {"✓" if is_completed else step_info.icon.replace("️⃣", "")}
                </div>
                <div style="flex: 1;">
                    <div style="
                        font-weight: 600;
                        color: {text_color};
                    ">{label}</div>
                    {description_html}
                </div>
            </div>
            """
            steps_html.append(step_html)

        return f"""
        <div style="
            padding: 16px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        ">
            <div style="
                font-weight: 600;
                color: #334155;
                margin-bottom: 12px;
                font-size: 14px;
            ">📋 {self.i18n.t("workflow.title")}</div>
            {"".join(steps_html)}
        </div>
        """

    @staticmethod
    def update(
        current_step: str,
        completed_steps: list[str] | None = None,
        compact: bool = True,
    ) -> str:
        """
        Static method to generate workflow HTML for updates.

        Args:
            current_step: Current active step key (extract, sort, train, merge, export).
            completed_steps: List of completed step keys.
            compact: Whether to use compact mode.

        Returns:
            HTML string for workflow indicator.
        """
        completed = set(completed_steps or [])
        steps = [
            ("extract", "1️⃣", "Extract"),
            ("sort", "2️⃣", "Sort"),
            ("train", "3️⃣", "Train"),
            ("merge", "4️⃣", "Merge"),
            ("export", "5️⃣", "Export"),
        ]

        steps_html = []

        for i, (key, icon, label) in enumerate(steps):
            is_current = key == current_step
            is_completed = key in completed

            if is_completed:
                bg_color = "#dcfce7"
                text_color = "#166534"
                border_color = "#22c55e"
            elif is_current:
                bg_color = "#3b82f6"
                text_color = "white"
                border_color = "#3b82f6"
            else:
                bg_color = "#f1f5f9"
                text_color = "#64748b"
                border_color = "#e2e8f0"

            step_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 8px 14px;
                background: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 20px;
                font-size: 13px;
                font-weight: {"600" if is_current else "500"};
            ">
                <span>{icon}</span>
                <span>{label}</span>
                {'<span style="margin-left: 4px;">✓</span>' if is_completed else ""}
            </div>
            """
            steps_html.append(step_html)

            if i < len(steps) - 1:
                steps_html.append(
                    '<div style="color: #94a3b8; font-size: 16px;">→</div>'
                )

        return f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 16px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            flex-wrap: wrap;
        ">
            {"".join(steps_html)}
        </div>
        """


def create_simple_workflow_header(current_step: str = "extract") -> str:
    """
    Create a simple workflow header for quick use.

    Args:
        current_step: The current active step.

    Returns:
        HTML string for workflow header.
    """
    return WorkflowIndicator.update(current_step)
