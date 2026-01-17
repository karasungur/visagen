"""
Workflow Step Manager for Mask Editing.

Provides step-based workflow management for progressive
mask refinement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MaskWorkflowStep(Enum):
    """Steps in the mask editing workflow."""

    AUTO_SEGMENT = "auto"  # Step 1: Automatic segmentation
    COMPONENTS = "components"  # Step 2: Component selection
    POLYGONS = "polygons"  # Step 3: Polygon editing (optional)
    BRUSH = "brush"  # Step 4: Brush refinement
    REFINE = "refine"  # Step 5: Morphological refinement
    PREVIEW = "preview"  # Step 6: Preview and save


# Step order for navigation
STEP_ORDER = [
    MaskWorkflowStep.AUTO_SEGMENT,
    MaskWorkflowStep.COMPONENTS,
    MaskWorkflowStep.POLYGONS,
    MaskWorkflowStep.BRUSH,
    MaskWorkflowStep.REFINE,
    MaskWorkflowStep.PREVIEW,
]

# Step display names
STEP_NAMES = {
    MaskWorkflowStep.AUTO_SEGMENT: "Auto Segment",
    MaskWorkflowStep.COMPONENTS: "Components",
    MaskWorkflowStep.POLYGONS: "Polygons",
    MaskWorkflowStep.BRUSH: "Brush",
    MaskWorkflowStep.REFINE: "Refine",
    MaskWorkflowStep.PREVIEW: "Preview",
}

# Step descriptions
STEP_DESCRIPTIONS = {
    MaskWorkflowStep.AUTO_SEGMENT: "Generate initial mask using SegFormer",
    MaskWorkflowStep.COMPONENTS: "Select face components to include",
    MaskWorkflowStep.POLYGONS: "Add/remove regions with polygons",
    MaskWorkflowStep.BRUSH: "Fine-tune mask with brush tools",
    MaskWorkflowStep.REFINE: "Apply morphological operations",
    MaskWorkflowStep.PREVIEW: "Preview result and save",
}


@dataclass
class WorkflowState:
    """
    State for mask editing workflow.

    Attributes:
        current_step: Currently active step.
        steps_completed: Set of completed step names.
        skip_polygons: Whether to skip polygon step.
    """

    current_step: MaskWorkflowStep = MaskWorkflowStep.AUTO_SEGMENT
    steps_completed: set[MaskWorkflowStep] = field(default_factory=set)
    skip_polygons: bool = False

    def get_step_index(self) -> int:
        """Get current step index (1-based for display)."""
        return STEP_ORDER.index(self.current_step) + 1

    def get_total_steps(self) -> int:
        """Get total number of steps."""
        if self.skip_polygons:
            return len(STEP_ORDER) - 1
        return len(STEP_ORDER)

    def get_available_steps(self) -> list[MaskWorkflowStep]:
        """Get list of available steps (respecting skip_polygons)."""
        if self.skip_polygons:
            return [s for s in STEP_ORDER if s != MaskWorkflowStep.POLYGONS]
        return STEP_ORDER.copy()

    def next_step(self) -> MaskWorkflowStep:
        """
        Move to next step.

        Returns:
            The new current step.
        """
        available = self.get_available_steps()
        current_idx = available.index(self.current_step)

        if current_idx < len(available) - 1:
            self.steps_completed.add(self.current_step)
            self.current_step = available[current_idx + 1]

        return self.current_step

    def prev_step(self) -> MaskWorkflowStep:
        """
        Move to previous step.

        Returns:
            The new current step.
        """
        available = self.get_available_steps()
        current_idx = available.index(self.current_step)

        if current_idx > 0:
            self.current_step = available[current_idx - 1]

        return self.current_step

    def go_to_step(self, step: MaskWorkflowStep) -> bool:
        """
        Jump to specific step if allowed.

        Args:
            step: Step to jump to.

        Returns:
            True if jump was successful.
        """
        available = self.get_available_steps()
        if step not in available:
            return False

        target_idx = available.index(step)
        current_idx = available.index(self.current_step)

        # Can always go back, or forward to completed steps
        if target_idx <= current_idx or step in self.steps_completed:
            self.current_step = step
            return True

        return False

    def mark_completed(self, step: MaskWorkflowStep | None = None) -> None:
        """Mark a step as completed."""
        if step is None:
            step = self.current_step
        self.steps_completed.add(step)

    def is_completed(self, step: MaskWorkflowStep) -> bool:
        """Check if step is completed."""
        return step in self.steps_completed

    def reset(self) -> None:
        """Reset workflow to beginning."""
        self.current_step = MaskWorkflowStep.AUTO_SEGMENT
        self.steps_completed.clear()

    def get_step_indicator(self) -> str:
        """Get step indicator string for display."""
        available = self.get_available_steps()
        idx = available.index(self.current_step) + 1
        total = len(available)
        name = STEP_NAMES[self.current_step]
        return f"Step {idx}/{total}: {name}"


class WorkflowManager:
    """
    Manager for mask editing workflow UI.

    Coordinates visibility and state of workflow step groups
    in the Gradio interface.

    Example:
        >>> manager = WorkflowManager()
        >>> visibility = manager.get_step_visibility()
        >>> # Use visibility dict to update Gradio component visibility
    """

    def __init__(self) -> None:
        self.state = WorkflowState()

    def get_step_visibility(self) -> dict[MaskWorkflowStep, bool]:
        """
        Get visibility state for each step group.

        Returns:
            Dictionary mapping steps to visibility boolean.
        """
        return {step: step == self.state.current_step for step in STEP_ORDER}

    def get_button_states(self) -> dict[str, bool]:
        """
        Get enabled/disabled states for navigation buttons.

        Returns:
            Dictionary with 'prev_enabled' and 'next_enabled'.
        """
        available = self.state.get_available_steps()
        current_idx = available.index(self.state.current_step)

        return {
            "prev_enabled": current_idx > 0,
            "next_enabled": current_idx < len(available) - 1,
        }

    def handle_next(self) -> tuple[str, dict[MaskWorkflowStep, bool], dict[str, bool]]:
        """
        Handle next button click.

        Returns:
            Tuple of (indicator_text, visibility_dict, button_states).
        """
        self.state.next_step()
        return (
            self.state.get_step_indicator(),
            self.get_step_visibility(),
            self.get_button_states(),
        )

    def handle_prev(self) -> tuple[str, dict[MaskWorkflowStep, bool], dict[str, bool]]:
        """
        Handle previous button click.

        Returns:
            Tuple of (indicator_text, visibility_dict, button_states).
        """
        self.state.prev_step()
        return (
            self.state.get_step_indicator(),
            self.get_step_visibility(),
            self.get_button_states(),
        )

    def reset(self) -> tuple[str, dict[MaskWorkflowStep, bool], dict[str, bool]]:
        """
        Reset workflow to beginning.

        Returns:
            Tuple of (indicator_text, visibility_dict, button_states).
        """
        self.state.reset()
        return (
            self.state.get_step_indicator(),
            self.get_step_visibility(),
            self.get_button_states(),
        )

    def set_skip_polygons(self, skip: bool) -> None:
        """Set whether to skip polygon step."""
        self.state.skip_polygons = skip


def create_step_groups_visibility_update(
    visibility: dict[MaskWorkflowStep, bool],
) -> list[Any]:
    """
    Create Gradio update values for step group visibility.

    Args:
        visibility: Dictionary mapping steps to visibility.

    Returns:
        List of gr.update() values for each step group.
    """
    import gradio as gr

    return [gr.update(visible=visibility.get(step, False)) for step in STEP_ORDER]
