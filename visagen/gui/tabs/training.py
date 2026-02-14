"""Training tab implementation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr
import numpy as np

from visagen.gui.components import (
    DropdownConfig,
    DropdownInput,
    ImagePreview,
    ImagePreviewConfig,
    LogOutput,
    LogOutputConfig,
    PathInput,
    PathInputConfig,
    ProcessControl,
    SliderConfig,
    SliderInput,
)
from visagen.gui.presets import PresetManager, TrainingPreset
from visagen.gui.tabs.base import BaseTab
from visagen.utils.io import read_json_locked, write_json_locked

if TYPE_CHECKING:
    from visagen.gui.i18n import I18n
    from visagen.gui.state.app_state import AppState


class TrainingTab(BaseTab):
    """Training configuration and execution tab."""

    def __init__(self, app_state: AppState, i18n: I18n) -> None:
        """Initialize training tab with preset manager."""
        super().__init__(app_state, i18n)
        self.preset_manager = PresetManager()

    @property
    def id(self) -> str:
        return "training"

    def _build_content(self) -> dict[str, Any]:
        """Build training tab UI."""
        components = {}

        gr.Markdown(f"### {self.t('title')}")

        # Preset section
        with gr.Row():
            with gr.Column(scale=3):
                components["preset_dropdown"] = gr.Dropdown(
                    label=self.t("preset.label"),
                    choices=self.preset_manager.list_presets(),
                    value=None,
                    interactive=True,
                )
            with gr.Column(scale=1):
                components["load_preset_btn"] = gr.Button(
                    self.t("preset.load"),
                    size="sm",
                )
            with gr.Column(scale=1):
                components["save_preset_btn"] = gr.Button(
                    self.t("preset.save"),
                    size="sm",
                    variant="secondary",
                )

        with gr.Row(visible=False) as save_preset_row:
            components["preset_name_input"] = gr.Textbox(
                label=self.t("preset.name_input"),
                placeholder="My Custom Preset",
            )
            components["confirm_save_btn"] = gr.Button(
                self.t("preset.confirm_save"),
                variant="primary",
            )
            components["cancel_save_btn"] = gr.Button(self.i18n.t("common.cancel"))
        components["save_preset_row"] = save_preset_row

        with gr.Row():
            with gr.Column():
                # Source directory input
                components["src_dir"] = PathInput(
                    PathInputConfig(
                        key="training.src_dir",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                # Destination directory input
                components["dst_dir"] = PathInput(
                    PathInputConfig(
                        key="training.dst_dir",
                        path_type="directory",
                        must_exist=True,
                    ),
                    self.i18n,
                ).build()

                # Output directory
                components["output_dir"] = PathInput(
                    PathInputConfig(
                        key="training.output_dir",
                        default="./workspace/model",
                        path_type="directory",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                # Batch size slider
                components["batch_size"] = SliderInput(
                    SliderConfig(
                        key="training.batch_size",
                        minimum=1,
                        maximum=32,
                        step=1,
                        default=8,
                    ),
                    self.i18n,
                ).build()

                # Max epochs slider
                components["max_epochs"] = SliderInput(
                    SliderConfig(
                        key="training.max_epochs",
                        minimum=10,
                        maximum=2000,
                        step=10,
                        default=500,
                    ),
                    self.i18n,
                ).build()

                # Learning rate
                components["learning_rate"] = gr.Number(
                    value=1e-4,
                    label=self.t("learning_rate.label"),
                )

        # Loss weights section
        with gr.Row():
            with gr.Column():
                components["dssim_weight"] = SliderInput(
                    SliderConfig(
                        key="training.dssim_weight", minimum=0, maximum=30, default=10.0
                    ),
                    self.i18n,
                ).build()
                components["l1_weight"] = SliderInput(
                    SliderConfig(
                        key="training.l1_weight", minimum=0, maximum=30, default=10.0
                    ),
                    self.i18n,
                ).build()
                components["lpips_weight"] = SliderInput(
                    SliderConfig(
                        key="training.lpips_weight", minimum=0, maximum=10, default=0.0
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["gan_power"] = SliderInput(
                    SliderConfig(
                        key="training.gan_power", minimum=0, maximum=1.0, default=0.0
                    ),
                    self.i18n,
                ).build()
                components["precision"] = DropdownInput(
                    DropdownConfig(
                        key="training.precision",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        default="16-mixed",
                    ),
                    self.i18n,
                ).build()

        # Advanced loss weights section
        gr.Markdown("### Advanced Loss Weights")
        with gr.Row():
            with gr.Column():
                components["eyes_mouth_weight"] = SliderInput(
                    SliderConfig(
                        key="training.eyes_mouth_weight",
                        minimum=0,
                        maximum=300,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()
                components["gaze_weight"] = SliderInput(
                    SliderConfig(
                        key="training.gaze_weight",
                        minimum=0,
                        maximum=10,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()
                components["true_face_power"] = SliderInput(
                    SliderConfig(
                        key="training.true_face_power",
                        minimum=0,
                        maximum=1.0,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["face_style_weight"] = SliderInput(
                    SliderConfig(
                        key="training.face_style_weight",
                        minimum=0,
                        maximum=100,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()
                components["bg_style_weight"] = SliderInput(
                    SliderConfig(
                        key="training.bg_style_weight",
                        minimum=0,
                        maximum=100,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()
                components["id_weight"] = SliderInput(
                    SliderConfig(
                        key="training.id_weight",
                        minimum=0,
                        maximum=1.0,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()

        gr.Markdown("### Temporal Training")
        components["temporal_enabled"] = gr.Checkbox(
            label="Enable temporal consistency training",
            value=False,
            info="Enable 3D temporal discriminator and consistency losses.",
        )
        with gr.Row():
            with gr.Column():
                components["temporal_power"] = SliderInput(
                    SliderConfig(
                        key="training.temporal_power",
                        minimum=0,
                        maximum=1.0,
                        default=0.1,
                        interactive=False,
                    ),
                    self.i18n,
                ).build()
            with gr.Column():
                components["temporal_consistency_weight"] = SliderInput(
                    SliderConfig(
                        key="training.temporal_consistency_weight",
                        minimum=0,
                        maximum=5.0,
                        default=1.0,
                        interactive=False,
                    ),
                    self.i18n,
                ).build()

        gr.Markdown("### Experimental Models")
        with gr.Row():
            with gr.Column():
                components["model_type"] = DropdownInput(
                    DropdownConfig(
                        key="training.model_type",
                        choices=["standard", "diffusion", "eg3d"],
                        default="standard",
                    ),
                    self.i18n,
                ).build()

            with gr.Column():
                components["texture_weight"] = SliderInput(
                    SliderConfig(
                        key="training.texture_weight",
                        minimum=0,
                        maximum=10,
                        default=0.0,
                    ),
                    self.i18n,
                ).build()
                components["use_pretrained_vae"] = gr.Checkbox(
                    label=self.t("use_pretrained_vae.label"),
                    value=True,
                    info=self.t("use_pretrained_vae.info"),
                )

        gr.Markdown("### Advanced Settings")
        with gr.Row():
            components["uniform_yaw"] = gr.Checkbox(
                label=self.t("uniform_yaw.label"),
                value=False,
                info=self.t("uniform_yaw.info"),
            )
            components["masked_training"] = gr.Checkbox(
                label=self.t("masked_training.label"),
                value=False,
                info=self.t("masked_training.info"),
            )

        # Resume checkpoint
        components["resume_ckpt"] = PathInput(
            PathInputConfig(
                key="training.resume_ckpt",
                path_type="file",
            ),
            self.i18n,
        ).build()

        # Start/Stop buttons
        with gr.Row():
            process_ctrl = ProcessControl("training", self.i18n)
            components["train_btn"], components["stop_btn"] = process_ctrl.build()

        gr.Markdown("---")
        gr.Markdown(f"### {self.t('preview.title')}")

        with gr.Row():
            components["preview_status"] = gr.Textbox(
                label=self.t("preview.status.label"),
                value=self.i18n.t("status.no_training"),
                interactive=False,
            )
            components["tensorboard_path"] = gr.Textbox(
                label="TensorBoard log directory",
                value="",
                interactive=False,
            )

        components["preview_image"] = ImagePreview(
            ImagePreviewConfig(key="training.preview.image", height=400),
            self.i18n,
        ).build()

        # Timer for auto-refresh preview
        components["preview_timer"] = gr.Timer(value=5)

        with gr.Row():
            components["refresh_btn"] = gr.Button(self.t("refresh_preview"))

        gr.Markdown("---")
        # Log output
        components["log"] = LogOutput(
            LogOutputConfig(key="training.log", lines=15, max_lines=30),
            self.i18n,
        ).build()

        return components

    def _setup_events(self, c: dict[str, Any]) -> None:
        """Wire up training event handlers."""
        # Debouncing state for live parameter updates
        self._param_update_times: dict[str, float] = {}
        DEBOUNCE_DELAY = 0.3  # seconds

        # Live control helper
        def send_live_param(key: str, value: Any, output_dir: str) -> None:
            """Send parameter update to running training process."""
            if not output_dir or not self.state.processes.is_running("training"):
                return

            # Debounce rapid updates
            current_time = time.time()
            last_update = self._param_update_times.get(key, 0)

            if current_time - last_update < DEBOUNCE_DELAY:
                return  # Skip rapid updates

            self._param_update_times[key] = current_time

            try:
                cmd_file = Path(output_dir) / "cmd_training.json"

                # Read existing securely to preserve other pending updates
                data = read_json_locked(cmd_file) or {"params": {}}

                # Update param
                if "params" not in data:
                    data["params"] = {}

                data["params"][key] = value
                data["timestamp"] = time.time()

                # Write securely
                write_json_locked(cmd_file, data)
            except Exception as e:
                print(f"Failed to send live command: {e}")

        # Bind live controls
        live_controls = [
            ("learning_rate", c["learning_rate"]),
            ("gan_power", c["gan_power"]),
            ("true_face_power", c["true_face_power"]),
            ("face_style_weight", c["face_style_weight"]),
            ("bg_style_weight", c["bg_style_weight"]),
            ("dssim_weight", c["dssim_weight"]),
            ("l1_weight", c["l1_weight"]),
            ("lpips_weight", c["lpips_weight"]),
            ("eyes_mouth_weight", c["eyes_mouth_weight"]),
            ("gaze_weight", c["gaze_weight"]),
            ("texture_weight", c["texture_weight"]),
            ("id_weight", c["id_weight"]),
        ]

        for key, component in live_controls:
            # Create a closure for each key
            def handler(val, out, k=key):
                send_live_param(k, val, out)

            component.change(
                fn=handler,
                inputs=[component, c["output_dir"]],
                outputs=None,
                show_progress=False,
            )

        def send_temporal_live_param(
            value: Any,
            output_dir: str,
            temporal_enabled: bool,
            key: str,
        ) -> None:
            if not temporal_enabled:
                return
            send_live_param(key, value, output_dir)

        c["temporal_power"].change(
            fn=lambda val, out, enabled: send_temporal_live_param(
                val, out, enabled, "temporal_power"
            ),
            inputs=[c["temporal_power"], c["output_dir"], c["temporal_enabled"]],
            outputs=None,
            show_progress=False,
        )
        c["temporal_consistency_weight"].change(
            fn=lambda val, out, enabled: send_temporal_live_param(
                val, out, enabled, "temporal_consistency_weight"
            ),
            inputs=[
                c["temporal_consistency_weight"],
                c["output_dir"],
                c["temporal_enabled"],
            ],
            outputs=None,
            show_progress=False,
        )

        c["temporal_enabled"].change(
            fn=lambda enabled: (  # noqa: E731
                gr.update(interactive=bool(enabled)),
                gr.update(interactive=bool(enabled)),
            ),
            inputs=[c["temporal_enabled"]],
            outputs=[c["temporal_power"], c["temporal_consistency_weight"]],
        )

        # Start training
        c["train_btn"].click(
            fn=self._start_training,
            inputs=[
                c["src_dir"],
                c["dst_dir"],
                c["output_dir"],
                c["batch_size"],
                c["max_epochs"],
                c["learning_rate"],
                c["dssim_weight"],
                c["l1_weight"],
                c["lpips_weight"],
                c["gan_power"],
                c["precision"],
                c["eyes_mouth_weight"],
                c["gaze_weight"],
                c["true_face_power"],
                c["face_style_weight"],
                c["bg_style_weight"],
                c["resume_ckpt"],
                c["model_type"],
                c["texture_weight"],
                c["use_pretrained_vae"],
                c["uniform_yaw"],
                c["masked_training"],
                c["id_weight"],
                c["temporal_enabled"],
                c["temporal_power"],
                c["temporal_consistency_weight"],
            ],
            outputs=c["log"],
        )

        # Stop training
        c["stop_btn"].click(
            fn=self._stop_training,
            outputs=c["log"],
        )

        # Refresh preview - manual button
        c["refresh_btn"].click(
            fn=self._refresh_preview,
            inputs=[c["output_dir"]],
            outputs=[c["preview_status"], c["preview_image"]],
        )

        # Auto-refresh preview every 5 seconds
        c["preview_timer"].tick(
            fn=self._refresh_preview,
            inputs=[c["output_dir"]],
            outputs=[c["preview_status"], c["preview_image"]],
        )

        c["output_dir"].change(
            fn=lambda out: str(Path(out) / "logs") if out else "",
            inputs=[c["output_dir"]],
            outputs=[c["tensorboard_path"]],
            show_progress=False,
        )

        # Preset events
        def load_preset(key: str) -> tuple:
            """Load preset and return all parameter values."""
            if not key:
                return tuple([gr.update()] * 22)
            preset = self.preset_manager.load_preset(key)
            if not preset:
                return tuple([gr.update()] * 22)
            return (
                preset.batch_size,
                preset.max_epochs,
                preset.learning_rate,
                preset.dssim_weight,
                preset.l1_weight,
                preset.lpips_weight,
                preset.gan_power,
                preset.precision,
                preset.model_type,
                preset.texture_weight,
                preset.use_pretrained_vae,
                getattr(preset, "uniform_yaw", False),
                getattr(preset, "masked_training", False),
                getattr(preset, "eyes_mouth_weight", 0.0),
                getattr(preset, "gaze_weight", 0.0),
                getattr(preset, "true_face_power", 0.0),
                getattr(preset, "face_style_weight", 0.0),
                getattr(preset, "bg_style_weight", 0.0),
                getattr(preset, "id_weight", 0.0),
                getattr(preset, "temporal_enabled", False),
                getattr(preset, "temporal_power", 0.1),
                getattr(preset, "temporal_consistency_weight", 1.0),
            )

        def show_save_dialog() -> dict:
            return gr.update(visible=True)

        def hide_save_dialog() -> dict:
            return gr.update(visible=False)

        def save_preset(
            name: str,
            batch_size: int,
            max_epochs: int,
            learning_rate: float,
            dssim_weight: float,
            l1_weight: float,
            lpips_weight: float,
            gan_power: float,
            precision: str,
            model_type: str,
            texture_weight: float,
            use_pretrained_vae: bool,
            uniform_yaw: bool,
            masked_training: bool,
            eyes_mouth_weight: float,
            gaze_weight: float,
            true_face_power: float,
            face_style_weight: float,
            bg_style_weight: float,
            id_weight: float,
            temporal_enabled: bool,
            temporal_power: float,
            temporal_consistency_weight: float,
        ) -> tuple:
            """Save current parameters as preset."""
            if not name:
                return gr.update(), gr.update(visible=True)
            preset = TrainingPreset(
                name=name,
                batch_size=int(batch_size),
                max_epochs=int(max_epochs),
                learning_rate=float(learning_rate),
                dssim_weight=float(dssim_weight),
                l1_weight=float(l1_weight),
                lpips_weight=float(lpips_weight),
                gan_power=float(gan_power),
                precision=precision,
                model_type=model_type,
                texture_weight=float(texture_weight),
                use_pretrained_vae=bool(use_pretrained_vae),
                uniform_yaw=bool(uniform_yaw),
                masked_training=bool(masked_training),
                eyes_mouth_weight=float(eyes_mouth_weight),
                gaze_weight=float(gaze_weight),
                true_face_power=float(true_face_power),
                face_style_weight=float(face_style_weight),
                bg_style_weight=float(bg_style_weight),
                id_weight=float(id_weight),
                temporal_enabled=bool(temporal_enabled),
                temporal_power=float(temporal_power),
                temporal_consistency_weight=float(temporal_consistency_weight),
            )
            self.preset_manager.save_preset(preset)
            new_choices = self.preset_manager.list_presets()
            return gr.update(choices=new_choices), gr.update(visible=False)

        # Wire preset events
        c["load_preset_btn"].click(
            fn=load_preset,
            inputs=[c["preset_dropdown"]],
            outputs=[
                c["batch_size"],
                c["max_epochs"],
                c["learning_rate"],
                c["dssim_weight"],
                c["l1_weight"],
                c["lpips_weight"],
                c["gan_power"],
                c["precision"],
                c["model_type"],
                c["texture_weight"],
                c["use_pretrained_vae"],
                c["uniform_yaw"],
                c["masked_training"],
                c["eyes_mouth_weight"],
                c["gaze_weight"],
                c["true_face_power"],
                c["face_style_weight"],
                c["bg_style_weight"],
                c["id_weight"],
                c["temporal_enabled"],
                c["temporal_power"],
                c["temporal_consistency_weight"],
            ],
        )

        c["save_preset_btn"].click(
            fn=show_save_dialog,
            outputs=[c["save_preset_row"]],
        )

        c["cancel_save_btn"].click(
            fn=hide_save_dialog,
            outputs=[c["save_preset_row"]],
        )

        c["confirm_save_btn"].click(
            fn=save_preset,
            inputs=[
                c["preset_name_input"],
                c["batch_size"],
                c["max_epochs"],
                c["learning_rate"],
                c["dssim_weight"],
                c["l1_weight"],
                c["lpips_weight"],
                c["gan_power"],
                c["precision"],
                c["model_type"],
                c["texture_weight"],
                c["use_pretrained_vae"],
                c["uniform_yaw"],
                c["masked_training"],
                c["eyes_mouth_weight"],
                c["gaze_weight"],
                c["true_face_power"],
                c["face_style_weight"],
                c["bg_style_weight"],
                c["id_weight"],
                c["temporal_enabled"],
                c["temporal_power"],
                c["temporal_consistency_weight"],
            ],
            outputs=[c["preset_dropdown"], c["save_preset_row"]],
        )

    def _start_training(
        self,
        src_dir: str,
        dst_dir: str,
        output_dir: str,
        batch_size: int,
        max_epochs: int,
        learning_rate: float,
        dssim_weight: float,
        l1_weight: float,
        lpips_weight: float,
        gan_power: float,
        precision: str,
        eyes_mouth_weight: float,
        gaze_weight: float,
        true_face_power: float,
        face_style_weight: float,
        bg_style_weight: float,
        resume_ckpt: str,
        model_type: str,
        texture_weight: float,
        use_pretrained_vae: bool,
        uniform_yaw: bool,
        masked_training: bool,
        id_weight: float,
        temporal_enabled: bool,
        temporal_power: float,
        temporal_consistency_weight: float,
    ) -> Generator[str, None, None]:
        """Start training subprocess."""
        # Validation
        if not src_dir or not Path(src_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return
        if not dst_dir or not Path(dst_dir).exists():
            yield self.i18n.t("errors.path_not_found")
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "visagen.tools.train",
            "--src-dir",
            src_dir,
            "--dst-dir",
            dst_dir,
            "--output-dir",
            output_dir or "./workspace/model",
            "--batch-size",
            str(int(batch_size)),
            "--max-epochs",
            str(int(max_epochs)),
            "--learning-rate",
            str(learning_rate),
            "--dssim-weight",
            str(dssim_weight),
            "--l1-weight",
            str(l1_weight),
            "--lpips-weight",
            str(lpips_weight),
            "--precision",
            precision,
            "--model-type",
            model_type,
            "--texture-weight",
            str(texture_weight),
            "--gan-power",
            str(gan_power),
            "--eyes-mouth-weight",
            str(eyes_mouth_weight),
            "--gaze-weight",
            str(gaze_weight),
        ]

        # Advanced loss weights (only add if non-zero to avoid clutter)
        if face_style_weight > 0:
            cmd.extend(["--face-style-weight", str(face_style_weight)])
        if bg_style_weight > 0:
            cmd.extend(["--bg-style-weight", str(bg_style_weight)])
        if true_face_power > 0:
            cmd.extend(["--true-face-power", str(true_face_power)])

        if use_pretrained_vae:
            cmd.append("--use-pretrained-vae")
        else:
            cmd.append("--no-pretrained-vae")

        if uniform_yaw:
            cmd.append("--uniform-yaw")

        if masked_training:
            cmd.append("--masked-training")

        if resume_ckpt and Path(resume_ckpt).exists():
            cmd.extend(["--resume", resume_ckpt])

        # id_weight
        if id_weight > 0:
            cmd.extend(["--id-weight", str(id_weight)])

        # Temporal training parameters
        if temporal_enabled:
            cmd.append("--temporal-enabled")
            cmd.extend(["--temporal-power", str(temporal_power)])
            cmd.extend(
                ["--temporal-consistency-weight", str(temporal_consistency_weight)]
            )

        yield f"Starting training...\n$ {' '.join(cmd)}\n"

        process: subprocess.Popen | None = None
        try:
            process = self.state.processes.launch(
                "training",
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process is None:
                yield "\n\nTraining is already running. Stop the active job first."
                return

            # Stream output
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        yield line
                    if process.poll() is not None:
                        break

            remaining, _ = process.communicate()
            if remaining:
                yield remaining

            exit_code = process.returncode
            if exit_code == 0:
                yield f"\n\n{self.i18n.t('status.completed')}"
            elif exit_code in {-15, -9, 143, 137}:
                yield f"\n\n{self.i18n.t('status.stopped')}"
            else:
                yield f"\n\n{self.i18n.t('errors.process_failed', code=exit_code)}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            # Proper process cleanup with timeout
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.state.processes.clear_if("training", process)

    def _stop_training(self) -> str:
        """Stop training process."""
        if self.state.processes.terminate("training"):
            return self.i18n.t("status.stopped")
        return self.i18n.t("status.no_training")

    def _refresh_preview(self, output_dir: str) -> tuple[str, np.ndarray | None]:
        """Get current training preview status."""
        if not output_dir:
            return self.i18n.t("errors.no_output_dir"), None

        preview_path = Path(output_dir) / "previews"
        latest_img = preview_path / "latest.png"
        latest_json = preview_path / "latest.json"

        if not latest_img.exists():
            return (
                "No preview available yet. Training may not have started or reached first interval.",
                None,
            )

        try:
            import cv2

            # Retry logic for file reading (file might be in-flight)
            MAX_RETRIES = 3
            img = None
            for _attempt in range(MAX_RETRIES):
                try:
                    # Check if file is empty (still being written)
                    if latest_img.stat().st_size == 0:
                        time.sleep(0.1)
                        continue
                    img = cv2.imread(str(latest_img))
                    if img is not None:
                        break
                except Exception:
                    time.sleep(0.1)

            if img is None:
                return "Error: Could not read preview image", None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            status = self.i18n.t("status.preview_available")
            if latest_json.exists():
                with open(latest_json) as f:
                    meta = json.load(f)
                loss_val = meta.get("loss")
                loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
                status = (
                    f"Step: {meta.get('step', '?')} | "
                    f"Epoch: {meta.get('epoch', '?')} | "
                    f"Loss: {loss_str}"
                )

            return status, img
        except Exception as e:
            return f"Error loading preview: {e}", None
