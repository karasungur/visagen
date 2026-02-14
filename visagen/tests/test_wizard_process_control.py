"""Wizard process-control contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from visagen.gui.i18n import I18n
from visagen.gui.state.app_state import AppState
from visagen.gui.tabs.wizard import WizardTab


def test_wizard_build_contains_step_stop_controls() -> None:
    import gradio as gr

    tab = WizardTab(AppState(), I18n(locale="en"))
    with gr.Blocks():
        components = tab._build_content()

    assert "step2_stop" in components
    assert "step3_stop" in components
    assert "step4_stop" in components
    assert "step2_extract" in components
    assert "step3_train" in components
    assert "step4_apply" in components


def test_wizard_uses_managed_slots_for_start_and_stop() -> None:
    source = Path("visagen/gui/tabs/wizard.py").read_text(encoding="utf-8")

    for slot in ("extract", "training", "merge"):
        launch_pattern = rf'processes\.launch\(\s*"{slot}"'
        terminate_pattern = rf'processes\.terminate\("{slot}"\)'

        assert re.search(launch_pattern, source) is not None
        assert re.search(terminate_pattern, source) is not None
