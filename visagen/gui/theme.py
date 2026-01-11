"""Visagen custom Gradio theme."""

from __future__ import annotations

import gradio as gr


def create_visagen_theme() -> gr.themes.Base:
    """
    Create custom Visagen theme for Gradio.

    Features:
    - Modern blue primary color
    - Clean slate secondary
    - Inter font family
    - Refined spacing and borders

    Returns:
        Configured Gradio theme.
    """
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    ).set(
        # === Colors ===
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_color="*neutral_200",
        block_label_background_fill="*primary_50",
        block_label_text_color="*primary_700",
        block_title_text_color="*neutral_800",
        # === Buttons ===
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200",
        button_secondary_text_color="*neutral_700",
        # === Inputs ===
        input_background_fill="white",
        input_border_color="*neutral_300",
        input_border_color_focus="*primary_500",
        input_placeholder_color="*neutral_400",
        # === Sliders ===
        slider_color="*primary_600",
        # === Checkboxes ===
        checkbox_background_color="white",
        checkbox_background_color_selected="*primary_600",
        checkbox_border_color="*neutral_300",
        checkbox_border_color_selected="*primary_600",
        # === Borders & Shadows ===
        block_border_width="1px",
        block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
        input_shadow="0 1px 2px 0 rgb(0 0 0 / 0.05)",
        # === Spacing ===
        block_padding="16px",
        block_radius="8px",
        input_radius="6px",
        button_large_radius="8px",
        button_small_radius="6px",
        # === Typography ===
        block_title_text_weight="600",
        block_label_text_weight="500",
    )


def create_dark_theme() -> gr.themes.Base:
    """
    Create dark variant of Visagen theme.

    Returns:
        Configured dark Gradio theme.
    """
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    ).set(
        # === Dark Mode Colors ===
        body_background_fill="*neutral_900",
        block_background_fill="*neutral_800",
        block_border_color="*neutral_700",
        block_label_background_fill="*primary_900",
        block_label_text_color="*primary_300",
        block_title_text_color="*neutral_100",
        # === Buttons ===
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_700",
        button_secondary_background_fill_hover="*neutral_600",
        button_secondary_text_color="*neutral_200",
        # === Inputs ===
        input_background_fill="*neutral_800",
        input_border_color="*neutral_600",
        input_border_color_focus="*primary_500",
        input_placeholder_color="*neutral_500",
        # === Checkboxes ===
        checkbox_background_color="*neutral_700",
        checkbox_background_color_selected="*primary_600",
        checkbox_border_color="*neutral_600",
        # === Spacing (same as light) ===
        block_padding="16px",
        block_radius="8px",
        input_radius="6px",
    )


# Custom CSS for additional styling
CUSTOM_CSS = """
/* Visagen Custom Styles */

/* Log output styling */
.log-output textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Status display */
.status-display input {
    font-weight: 500 !important;
}

/* Progress indicator */
.progress-indicator {
    margin: 8px 0;
}

/* System monitor */
.system-monitor {
    font-size: 13px;
}

/* Tab styling */
.tab-nav button {
    font-weight: 500 !important;
    padding: 12px 20px !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid var(--primary-500) !important;
}

/* Image preview */
.image-preview img {
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Workflow indicator */
.workflow-indicator {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding: 16px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid #e2e8f0;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 13px;
    color: #64748b;
    background: #f1f5f9;
    transition: all 0.2s ease;
}

.workflow-step.active {
    color: white;
    background: #3b82f6;
    font-weight: 500;
}

.workflow-step.completed {
    color: #22c55e;
    background: #dcfce7;
}

/* Toast notifications */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.toast {
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    animation: slideIn 0.3s ease;
}

.toast.success {
    background: #22c55e;
    color: white;
}

.toast.error {
    background: #ef4444;
    color: white;
}

.toast.warning {
    background: #f59e0b;
    color: white;
}

.toast.info {
    background: #3b82f6;
    color: white;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* Footer styling */
.footer {
    text-align: center;
    padding: 16px;
    color: #64748b;
    font-size: 13px;
    border-top: 1px solid #e2e8f0;
    margin-top: 24px;
}
"""


def get_theme_css() -> str:
    """Get custom CSS for the theme."""
    return CUSTOM_CSS
