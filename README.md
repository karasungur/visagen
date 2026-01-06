<p align="center">
  <img src="https://raw.githubusercontent.com/karasungur/visagen/main/docs/assets/logo.png" alt="Visagen Logo" width="200"/>
</p>

<h1 align="center">Visagen</h1>

<p align="center">
  <strong>Modern Face Swapping Framework with PyTorch Lightning</strong>
</p>

<p align="center">
  <a href="https://github.com/karasungur/visagen/actions"><img src="https://img.shields.io/github/actions/workflow/status/karasungur/visagen/test.yml?branch=main&style=flat-square&logo=github" alt="Build Status"/></a>
  <a href="https://pypi.org/project/visagen/"><img src="https://img.shields.io/pypi/v/visagen?style=flat-square&logo=pypi" alt="PyPI Version"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch 2.0+"/></a>
  <a href="https://github.com/karasungur/visagen/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"/></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#cli-tools">CLI Tools</a> •
  <a href="#documentation">Docs</a>
</p>

---

## Overview

**Visagen** is a next-generation face swapping framework built from the ground up with modern deep learning practices. Inspired by DeepFaceLab, Visagen reimagines the entire pipeline using **PyTorch Lightning**, offering cleaner code, better performance, and easier extensibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                         VISAGEN                                  │
├─────────────────────────────────────────────────────────────────┤
│  Extract → Train → Swap → Postprocess                           │
│     │        │       │         │                                 │
│     ▼        ▼       ▼         ▼                                 │
│  InsightFace  DFL   CBAM    Color Transfer                      │
│  SegFormer   Module  Attention  Blending                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Architecture
| Component | Technology | Description |
|-----------|------------|-------------|
| **Encoder** | ConvNeXt V2 | State-of-the-art vision backbone with GRN layers |
| **Decoder** | Swish Activation | Smooth gradients for better convergence |
| **Attention** | CBAM | Channel & Spatial attention for face details |
| **Face Detection** | InsightFace | High-accuracy face detection & recognition |
| **Segmentation** | SegFormer | Semantic face parsing for masks |

### Training Features
- **Multi-Loss Training**: DSSIM, L1, LPIPS, ID Loss, GAN Loss
- **GAN Support**: UNet Patch Discriminator with progressive training
- **Mixed Precision**: FP16/BF16 training with automatic scaling
- **Gradient Clipping**: Stable training with configurable clipping
- **LR Scheduling**: Cosine annealing with warm restarts

### Advanced Features
- **Optuna HPO**: Automated hyperparameter optimization
- **Gradio UI**: Web interface for training & inference (10 tabs)
- **Color Transfer**: RCT, LCT, SOT, MKL, IDT, and Neural (VGG-based) algorithms
- **Blending**: Feather, Laplacian, Poisson blending
- **Face Restoration**: GFPGAN and GPEN for enhanced quality
- **Hardware Encoding**: NVENC support for fast video encoding
- **Model Export**: ONNX and TensorRT for production deployment
- **Dataset Sorting**: 14 methods including blur, face-yaw, histogram
- **Mask Export**: LabelMe and COCO format export for external editing
- **Gaze Loss**: Eye region consistency for realistic results

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/karasungur/visagen.git
cd visagen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install base package
pip install -e .
```

### Full Installation (All Features)

```bash
pip install -e ".[full]"
```

### Optional Dependencies

```bash
# Vision (InsightFace, SegFormer)
pip install -e ".[vision]"

# Training (LPIPS)
pip install -e ".[training]"

# Hyperparameter Tuning (Optuna)
pip install -e ".[tuning]"

# Web Interface (Gradio)
pip install -e ".[gui]"

# Postprocessing (Color Transfer, Blending)
pip install -e ".[postprocess]"

# Video Merger (FFmpeg)
pip install -e ".[merger]"

# Model Export (ONNX, TensorRT)
pip install -e ".[export]"

# Face Restoration (GFPGAN)
pip install -e ".[restore]"

# GPU Data Loading (NVIDIA DALI)
pip install -e ".[dali]"
```

---

## Quick Start

### 1. Extract Faces

```bash
# Extract from video
visagen-extract \
    --input video.mp4 \
    --output-dir ./workspace/data_src/aligned \
    --face-size 512

# Extract from images
visagen-extract \
    --input ./photos/ \
    --output-dir ./workspace/data_dst/aligned \
    --face-size 512
```

### 2. Train Model

```bash
visagen-train \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/model \
    --epochs 500 \
    --batch-size 8 \
    --resolution 512
```

### 3. Hyperparameter Tuning (Optional)

```bash
visagen-tune \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/optuna \
    --n-trials 20 \
    --epochs-per-trial 50
```

### 4. Merge Faces into Video

```bash
# Basic merge with trained model
visagen-merge input.mp4 output.mp4 -c ./workspace/model/model.ckpt

# With face restoration and hardware encoding
visagen-merge input.mp4 output.mp4 -c model.ckpt \
    --restore-face --restore-strength 0.7 \
    --codec h264_nvenc --color-transfer rct
```

### 5. Export Model for Production

```bash
# Export to ONNX
visagen-export model.ckpt -o model.onnx --validate

# Export to TensorRT (FP16)
visagen-export model.onnx -o model.engine --format tensorrt --precision fp16
```

### 6. Launch Web Interface

```bash
visagen-gui --port 7860
```

---

## Architecture

### Model Architecture

```
Input (512x512x3)
       │
       ▼
┌──────────────┐
│   ConvNeXt   │  ← Encoder (pretrained)
│   Encoder    │
└──────────────┘
       │
       ▼
┌──────────────┐
│    CBAM      │  ← Channel & Spatial Attention
│  Attention   │
└──────────────┘
       │
       ▼
┌──────────────┐
│    Swish     │  ← Decoder with skip connections
│   Decoder    │
└──────────────┘
       │
       ▼
Output (512x512x3)
```

### Training Pipeline

```python
from visagen.training import DFLModule
from visagen.data import FaceDataModule
import pytorch_lightning as pl

# Initialize
datamodule = FaceDataModule(
    src_dir="./data_src/aligned",
    dst_dir="./data_dst/aligned",
    batch_size=8,
)

model = DFLModule(
    resolution=512,
    learning_rate=5e-5,
    dssim_weight=10.0,
    lpips_weight=1.0,
)

# Train
trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu",
    precision="16-mixed",
)
trainer.fit(model, datamodule)
```

---

## CLI Tools

| Command | Description |
|---------|-------------|
| `visagen-extract` | Extract and align faces from images/videos |
| `visagen-train` | Train face swap model |
| `visagen-pretrain` | Pretrain encoder on FFHQ/CelebA datasets |
| `visagen-tune` | Hyperparameter optimization with Optuna |
| `visagen-merge` | Merge face swaps into video with NVENC support |
| `visagen-export` | Export model to ONNX/TensorRT formats |
| `visagen-sort` | Sort and filter face datasets (14 methods) |
| `visagen-gui` | Launch Gradio web interface |
| `visagen-video` | Extract frames from video / create video from frames |
| `visagen-enhance` | Batch face enhancement using GFPGAN/GPEN |
| `visagen-resize` | Resize faceset images preserving DFL metadata |
| `visagen-benchmark` | Run performance benchmarks |

---

## Project Structure

```
visagen/
├── data/               # Data loading & augmentation
│   ├── dataset.py      # FaceDataset
│   ├── datamodule.py   # FaceDataModule
│   ├── dali_pipeline.py # NVIDIA DALI GPU pipeline
│   └── augmentations.py
├── models/             # Neural network architectures
│   ├── encoder.py      # ConvNeXt encoder
│   ├── decoder.py      # Swish decoder
│   ├── attention.py    # CBAM attention
│   └── discriminator.py
├── training/           # Training logic
│   ├── dfl_module.py   # PyTorch Lightning module
│   ├── pretrain_module.py # Pretraining module
│   └── losses.py       # Loss functions
├── merger/             # Video processing pipeline
│   ├── video_io.py     # FFmpeg video I/O with NVENC
│   ├── frame_processor.py # Single-frame processing
│   ├── batch_processor.py # Parallel processing
│   └── merger.py       # High-level orchestration
├── postprocess/        # Post-processing
│   ├── color_transfer.py # RCT, LCT, SOT, MKL, IDT algorithms
│   ├── neural_color.py # VGG-based neural color transfer
│   ├── blending.py     # Laplacian, Poisson, Feather
│   ├── restore.py      # GFPGAN face restoration
│   └── gpen.py         # GPEN face restoration
├── export/             # Model export
│   ├── onnx_exporter.py # ONNX export
│   ├── tensorrt_builder.py # TensorRT engine builder
│   └── validation.py   # Export validation
├── sorting/            # Dataset sorting
│   └── sorter.py       # 14 sorting methods
├── tuning/             # Hyperparameter optimization
│   └── optuna_tuner.py
├── tools/              # CLI tools
│   ├── extract_v2.py   # Face extraction
│   ├── train.py        # Training
│   ├── pretrain.py     # Pretraining
│   ├── merge.py        # Video merging
│   ├── export.py       # Model export
│   ├── sorter.py       # Dataset sorting
│   ├── tune.py         # HPO
│   ├── video_ed.py     # Video frame tools
│   ├── faceset_enhancer.py # Batch face enhancement
│   ├── faceset_resizer.py  # Faceset resizing
│   ├── benchmark.py    # Performance benchmarks
│   └── gradio_app.py   # Web UI (10 tabs)
├── vision/             # Computer vision
│   ├── detector.py     # InsightFace SCRFD detection
│   ├── aligner.py      # Face alignment (Umeyama)
│   ├── segmenter.py    # SegFormer segmentation
│   ├── dflimg.py       # DFL image metadata
│   └── mask_export.py  # LabelMe/COCO export
└── tests/              # Unit tests (580+)
```

---

## Performance

| Metric | Value |
|--------|-------|
| Training Speed | ~50 img/s (RTX 3090) |
| Memory Usage | ~8GB (512x512, batch=8) |
| Inference | ~30 FPS (512x512) |
| Unit Tests | 580+ |

---

## Roadmap

- [x] **Phase 0-1**: Core architecture (ConvNeXt, Swish, CBAM)
- [x] **Phase 2**: Face detection & segmentation integration
- [x] **Phase 3**: Multi-loss training system
- [x] **Phase 4**: Data pipeline with augmentations
- [x] **Phase 5**: GAN training, color transfer, blending
- [x] **Phase 6**: Optuna HPO & Gradio UI
- [x] **Phase 7**: Video pipeline & batch processing
- [x] **Phase 8**: Model export (ONNX, TensorRT)
- [x] **Phase 9**: Face restoration (GFPGAN)
- [x] **Phase 10**: Hardware encoding (NVENC)
- [x] **Phase 11**: Advanced restoration & mask export (GPEN, LabelMe/COCO, Gaze Loss)

---

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR.

```bash
# Development installation
pip install -e ".[dev]"

# Run tests
pytest visagen/tests/ -v

# Code formatting
ruff check visagen/
ruff format visagen/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [DeepFaceLab](https://github.com/iperov/DeepFaceLab) - Original inspiration
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection
- [Optuna](https://optuna.org/) - Hyperparameter optimization

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/karasungur">Mustafa Karasungur</a>
</p>
