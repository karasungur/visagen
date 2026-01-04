"""
Gradio Web Interface for Visagen.

Provides a user-friendly web UI for:
- Training model configuration and execution
- Single image face swap inference
- Face extraction from images/videos
- Color transfer and blending demos
- Application settings

Usage:
    visagen-gui --port 7860 --share
"""

import argparse
import sys
import subprocess
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple
import numpy as np

# Check gradio availability
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class GradioApp:
    """
    Gradio application state and handlers.

    Manages model loading, inference, and training processes.
    """

    def __init__(self) -> None:
        self.model = None
        self.model_path: Optional[str] = None
        self.training_process: Optional[subprocess.Popen] = None
        self.training_queue: queue.Queue = queue.Queue()
        self.device = "auto"
        self.settings = {
            "device": "auto",
            "default_batch_size": 8,
            "num_workers": 4,
            "workspace_dir": "./workspace",
        }

    def load_model(self, checkpoint_path: str) -> str:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.

        Returns:
            Status message.
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return "Error: Checkpoint file not found"

        try:
            import torch
            from visagen.training.dfl_module import DFLModule

            self.model = DFLModule.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            self.model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model_path = checkpoint_path
            return f"Model loaded: {Path(checkpoint_path).name}"

        except Exception as e:
            return f"Error loading model: {e}"

    def unload_model(self) -> str:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_path = None

            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            return "Model unloaded"
        return "No model loaded"

    def swap_face(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
    ) -> np.ndarray:
        """
        Perform face swap inference.

        Args:
            source_img: Source face image (H, W, C) uint8.
            target_img: Target face image (H, W, C) uint8 (unused for now).

        Returns:
            Swapped face image (H, W, C) uint8.
        """
        if self.model is None:
            raise gr.Error("No model loaded. Please load a checkpoint first.")

        if source_img is None:
            raise gr.Error("Please provide a source image.")

        try:
            import torch
            import cv2

            # Preprocess: resize and normalize
            img = cv2.resize(source_img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img).unsqueeze(0)  # Add batch dim

            if next(self.model.parameters()).is_cuda:
                img = img.cuda()

            # Inference
            with torch.no_grad():
                output = self.model(img)

            # Postprocess
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
            output = np.clip(output * 255, 0, 255).astype(np.uint8)

            # Resize back to original size if needed
            if source_img.shape[:2] != (256, 256):
                output = cv2.resize(output, (source_img.shape[1], source_img.shape[0]))

            return output

        except Exception as e:
            raise gr.Error(f"Inference failed: {e}")

    def start_training(
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
        resume_ckpt: str,
    ) -> Generator[str, None, None]:
        """
        Start training process with progress updates.

        Yields log lines as training progresses.
        """
        # Validate inputs
        if not src_dir or not Path(src_dir).exists():
            yield "Error: Source directory not found"
            return
        if not dst_dir or not Path(dst_dir).exists():
            yield "Error: Destination directory not found"
            return

        # Build command
        cmd = [
            sys.executable, "-m", "visagen.tools.train",
            "--src-dir", src_dir,
            "--dst-dir", dst_dir,
            "--output-dir", output_dir or "./workspace/model",
            "--batch-size", str(int(batch_size)),
            "--max-epochs", str(int(max_epochs)),
            "--learning-rate", str(learning_rate),
            "--dssim-weight", str(dssim_weight),
            "--l1-weight", str(l1_weight),
            "--lpips-weight", str(lpips_weight),
            "--precision", precision,
        ]

        if resume_ckpt and Path(resume_ckpt).exists():
            cmd.extend(["--resume", resume_ckpt])

        yield f"Starting training...\n$ {' '.join(cmd)}\n"

        # Launch subprocess
        try:
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in iter(self.training_process.stdout.readline, ""):
                if line:
                    yield line
                if self.training_process.poll() is not None:
                    break

            # Get remaining output
            remaining, _ = self.training_process.communicate()
            if remaining:
                yield remaining

            exit_code = self.training_process.returncode
            if exit_code == 0:
                yield "\n\nTraining completed successfully!"
            else:
                yield f"\n\nTraining exited with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

        finally:
            self.training_process = None

    def stop_training(self) -> str:
        """Stop running training process."""
        if self.training_process is not None:
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
            self.training_process = None
            return "Training stopped."
        return "No training in progress."

    def apply_color_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """
        Apply color transfer.

        Args:
            source: Source image (color reference).
            target: Target image (to modify).
            mode: Color transfer mode.

        Returns:
            Color-transferred image.
        """
        if source is None or target is None:
            raise gr.Error("Please provide both source and target images.")

        try:
            from visagen.postprocess import color_transfer

            # Ensure float32 [0, 1] and BGR
            if source.dtype == np.uint8:
                source = source.astype(np.float32) / 255.0
            if target.dtype == np.uint8:
                target = target.astype(np.float32) / 255.0

            # Gradio provides RGB, convert to BGR for OpenCV-based functions
            import cv2
            source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            result_bgr = color_transfer(mode, target_bgr, source_bgr)

            # Convert back to RGB
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            return (result_rgb * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Color transfer failed: {e}")

    def apply_blend(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """
        Apply image blending.

        Args:
            foreground: Foreground image.
            background: Background image.
            mask: Blending mask.
            mode: Blend mode.

        Returns:
            Blended image.
        """
        if foreground is None or background is None or mask is None:
            raise gr.Error("Please provide foreground, background, and mask images.")

        try:
            from visagen.postprocess import blend
            import cv2

            # Ensure float32 [0, 1]
            if foreground.dtype == np.uint8:
                foreground = foreground.astype(np.float32) / 255.0
            if background.dtype == np.uint8:
                background = background.astype(np.float32) / 255.0
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0

            # Handle mask dimensions
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Resize mask to match images if needed
            if mask.shape[:2] != foreground.shape[:2]:
                mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

            result = blend(mode, foreground, background, mask)

            return (result * 255).astype(np.uint8)

        except Exception as e:
            raise gr.Error(f"Blending failed: {e}")

    def run_extraction(
        self,
        input_path: str,
        output_dir: str,
        face_type: str,
        output_size: int,
        min_confidence: float,
    ) -> Generator[str, None, None]:
        """Run face extraction."""
        if not input_path or not Path(input_path).exists():
            yield "Error: Input path not found"
            return

        cmd = [
            sys.executable, "-m", "visagen.tools.extract_v2",
            "--input", input_path,
            "--output", output_dir or "./workspace/extracted",
            "--face-type", face_type,
            "--output-size", str(int(output_size)),
            "--min-confidence", str(min_confidence),
        ]

        yield f"Starting extraction...\n$ {' '.join(cmd)}\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                if line:
                    yield line

            exit_code = process.wait()
            if exit_code == 0:
                yield "\n\nExtraction completed!"
            else:
                yield f"\n\nExtraction failed with code {exit_code}"

        except Exception as e:
            yield f"\n\nError: {e}"

    def save_settings(
        self,
        device: str,
        gpu_id: int,
        default_batch_size: int,
        num_workers: int,
        workspace_dir: str,
    ) -> str:
        """Save application settings."""
        self.settings.update({
            "device": device,
            "gpu_id": int(gpu_id),
            "default_batch_size": int(default_batch_size),
            "num_workers": int(num_workers),
            "workspace_dir": workspace_dir,
        })
        return "Settings saved."


def create_training_tab(app: GradioApp) -> Dict[str, Any]:
    """Create training configuration and execution tab."""
    with gr.Tab("Training"):
        gr.Markdown("### Model Training")

        with gr.Row():
            with gr.Column():
                src_dir = gr.Textbox(
                    label="Source Directory",
                    placeholder="./workspace/data_src/aligned",
                    info="Directory containing source face images",
                )
                dst_dir = gr.Textbox(
                    label="Destination Directory",
                    placeholder="./workspace/data_dst/aligned",
                    info="Directory containing destination face images",
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./workspace/model",
                    info="Directory for checkpoints and logs",
                )

            with gr.Column():
                batch_size = gr.Slider(
                    1, 32, value=8, step=1,
                    label="Batch Size",
                )
                max_epochs = gr.Slider(
                    10, 2000, value=500, step=10,
                    label="Max Epochs",
                )
                learning_rate = gr.Number(
                    value=1e-4,
                    label="Learning Rate",
                )

        with gr.Row():
            with gr.Column():
                dssim_weight = gr.Slider(
                    0, 30, value=10.0,
                    label="DSSIM Weight",
                )
                l1_weight = gr.Slider(
                    0, 30, value=10.0,
                    label="L1 Weight",
                )
                lpips_weight = gr.Slider(
                    0, 10, value=0.0,
                    label="LPIPS Weight",
                    info="Requires lpips package",
                )

            with gr.Column():
                gan_power = gr.Slider(
                    0, 1.0, value=0.0,
                    label="GAN Power",
                    info="0 = disabled, > 0 = adversarial training",
                )
                precision = gr.Dropdown(
                    ["32", "16-mixed", "bf16-mixed"],
                    value="32",
                    label="Precision",
                )

        with gr.Row():
            resume_ckpt = gr.Textbox(
                label="Resume from Checkpoint (optional)",
                placeholder="./workspace/model/checkpoints/last.ckpt",
            )

        with gr.Row():
            train_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Training", variant="stop")

        training_log = gr.Textbox(
            label="Training Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )

        # Event handlers
        train_btn.click(
            fn=app.start_training,
            inputs=[
                src_dir, dst_dir, output_dir,
                batch_size, max_epochs, learning_rate,
                dssim_weight, l1_weight, lpips_weight,
                gan_power, precision, resume_ckpt,
            ],
            outputs=training_log,
        )

        stop_btn.click(
            fn=app.stop_training,
            outputs=training_log,
        )

    return {
        "src_dir": src_dir,
        "dst_dir": dst_dir,
        "output_dir": output_dir,
    }


def create_inference_tab(app: GradioApp) -> Dict[str, Any]:
    """Create single image face swap inference tab."""
    with gr.Tab("Inference"):
        gr.Markdown("### Face Swap Inference")

        with gr.Row():
            checkpoint_path = gr.Textbox(
                label="Model Checkpoint",
                placeholder="./workspace/model/checkpoints/last.ckpt",
            )
            load_btn = gr.Button("Load Model")
            unload_btn = gr.Button("Unload Model")

        model_status = gr.Textbox(
            label="Model Status",
            value="No model loaded",
            interactive=False,
        )

        with gr.Row():
            source_image = gr.Image(
                label="Source Face",
                type="numpy",
            )
            target_image = gr.Image(
                label="Target Face (reference)",
                type="numpy",
            )
            output_image = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            swap_btn = gr.Button("Swap Face", variant="primary")

        # Event handlers
        load_btn.click(
            fn=app.load_model,
            inputs=checkpoint_path,
            outputs=model_status,
        )

        unload_btn.click(
            fn=app.unload_model,
            outputs=model_status,
        )

        swap_btn.click(
            fn=app.swap_face,
            inputs=[source_image, target_image],
            outputs=output_image,
        )

    return {}


def create_extract_tab(app: GradioApp) -> Dict[str, Any]:
    """Create face extraction tab."""
    with gr.Tab("Extract"):
        gr.Markdown("### Face Extraction")
        gr.Markdown("Extract faces from images or videos for training.")

        with gr.Row():
            input_path = gr.Textbox(
                label="Input (image, video, or directory)",
                placeholder="./input_video.mp4",
            )
            output_dir = gr.Textbox(
                label="Output Directory",
                value="./workspace/extracted",
            )

        with gr.Row():
            face_type = gr.Dropdown(
                ["whole_face", "full", "mid_full", "half", "head"],
                value="whole_face",
                label="Face Type",
            )
            output_size = gr.Slider(
                128, 1024, value=512, step=64,
                label="Output Size",
            )
            min_confidence = gr.Slider(
                0.1, 1.0, value=0.5, step=0.05,
                label="Min Confidence",
            )

        with gr.Row():
            extract_btn = gr.Button("Extract Faces", variant="primary")

        extract_log = gr.Textbox(
            label="Extraction Log",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        extract_btn.click(
            fn=app.run_extraction,
            inputs=[input_path, output_dir, face_type, output_size, min_confidence],
            outputs=extract_log,
        )

    return {}


def create_postprocess_tab(app: GradioApp) -> Dict[str, Any]:
    """Create color transfer and blending demo tab."""
    with gr.Tab("Postprocess"):
        gr.Markdown("### Color Transfer Demo")

        with gr.Row():
            ct_source = gr.Image(
                label="Source (color reference)",
                type="numpy",
            )
            ct_target = gr.Image(
                label="Target (to modify)",
                type="numpy",
            )
            ct_result = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            ct_mode = gr.Dropdown(
                ["rct", "lct", "sot", "mkl", "idt"],
                value="rct",
                label="Color Transfer Mode",
                info="RCT=Reinhard, LCT=Linear, SOT=Sliced OT, MKL=Monge-Kantorovitch, IDT=Iterative",
            )
            ct_btn = gr.Button("Apply Color Transfer")

        ct_btn.click(
            fn=app.apply_color_transfer,
            inputs=[ct_source, ct_target, ct_mode],
            outputs=ct_result,
        )

        gr.Markdown("---")
        gr.Markdown("### Blending Demo")

        with gr.Row():
            bl_fg = gr.Image(
                label="Foreground",
                type="numpy",
            )
            bl_bg = gr.Image(
                label="Background",
                type="numpy",
            )
            bl_mask = gr.Image(
                label="Mask",
                type="numpy",
            )
            bl_result = gr.Image(
                label="Result",
                type="numpy",
            )

        with gr.Row():
            bl_mode = gr.Dropdown(
                ["laplacian", "poisson", "feather"],
                value="laplacian",
                label="Blend Mode",
                info="Laplacian=multi-band pyramid, Poisson=seamless clone, Feather=alpha blend",
            )
            bl_btn = gr.Button("Blend Images")

        bl_btn.click(
            fn=app.apply_blend,
            inputs=[bl_fg, bl_bg, bl_mask, bl_mode],
            outputs=bl_result,
        )

    return {}


def create_settings_tab(app: GradioApp) -> Dict[str, Any]:
    """Create application settings tab."""
    with gr.Tab("Settings"):
        gr.Markdown("### Application Settings")

        with gr.Row():
            device = gr.Dropdown(
                ["auto", "cuda", "cpu", "mps"],
                value="auto",
                label="Device",
            )
            gpu_id = gr.Number(
                value=0,
                label="GPU ID (if multiple GPUs)",
            )

        with gr.Row():
            default_batch_size = gr.Slider(
                1, 32, value=8, step=1,
                label="Default Batch Size",
            )
            num_workers = gr.Slider(
                0, 16, value=4, step=1,
                label="DataLoader Workers",
            )

        with gr.Row():
            workspace_dir = gr.Textbox(
                value="./workspace",
                label="Default Workspace Directory",
            )

        save_settings_btn = gr.Button("Save Settings")
        settings_status = gr.Textbox(
            value="",
            label="Status",
            interactive=False,
        )

        save_settings_btn.click(
            fn=app.save_settings,
            inputs=[device, gpu_id, default_batch_size, num_workers, workspace_dir],
            outputs=settings_status,
        )

    return {}


def create_app() -> "gr.Blocks":
    """Create Gradio application."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required. Install with: pip install 'visagen[gui]'")

    app_state = GradioApp()

    with gr.Blocks(
        title="Visagen - Face Swapping Framework",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Visagen")
        gr.Markdown("Modern Face Swapping Framework with PyTorch Lightning")

        # Create tabs
        create_training_tab(app_state)
        create_inference_tab(app_state)
        create_extract_tab(app_state)
        create_postprocess_tab(app_state)
        create_settings_tab(app_state)

        gr.Markdown("---")
        gr.Markdown(
            "Made with PyTorch Lightning | "
            "[GitHub](https://github.com/karasungur/visagen)"
        )

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visagen Gradio Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Username:password for basic auth",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point for Gradio app."""
    if not GRADIO_AVAILABLE:
        print("Error: Gradio is required for the GUI.")
        print("Install with: pip install 'visagen[gui]'")
        return 1

    args = parse_args()

    print("=" * 50)
    print("VISAGEN WEB INTERFACE")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 50)

    app = create_app()

    # Parse auth if provided
    auth = None
    if args.auth:
        parts = args.auth.split(":", 1)
        if len(parts) == 2:
            auth = (parts[0], parts[1])

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
