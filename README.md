<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=120&section=header" width="100%"/>

<!-- Logo -->
<img src="https://raw.githubusercontent.com/karasungur/visagen/main/assets/logo.png" alt="Visagen Logo" width="200"/>

<h1>
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&pause=1000&color=667EEA&center=true&vCenter=true&width=435&lines=Visagen;Modern+Face+Swapping;Built+with+PyTorch+Lightning" alt="Typing SVG" />
</h1>

<p><strong>Next-Generation Face Swapping Framework</strong></p>
<p><em>Powered by ConvNeXt, CBAM Attention & PyTorch Lightning</em></p>

<!-- Language Selector -->
<p>
  <a href="README.md">🇺🇸 English</a> |
  <a href="README_TR.md">🇹🇷 Türkçe</a>
</p>

<!-- Badges Row 1 -->
<p>
  <a href="https://github.com/karasungur/visagen/actions"><img src="https://img.shields.io/github/actions/workflow/status/karasungur/visagen/test.yml?branch=main&style=for-the-badge&logo=github&logoColor=white&label=CI" alt="Build Status"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Lightning-2.0%2B-792EE5?style=for-the-badge&logo=lightning&logoColor=white" alt="Lightning"/>
</p>

<!-- Badges Row 2 -->
<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-00C853?style=for-the-badge" alt="License: MIT"/></a>
  <a href="https://github.com/karasungur/visagen/stargazers"><img src="https://img.shields.io/github/stars/karasungur/visagen?style=for-the-badge&logo=github&color=FFD700" alt="Stars"/></a>
  <a href="https://github.com/karasungur/visagen/network/members"><img src="https://img.shields.io/github/forks/karasungur/visagen?style=for-the-badge&logo=github&color=1E90FF" alt="Forks"/></a>
  <a href="https://github.com/karasungur/visagen/issues"><img src="https://img.shields.io/github/issues/karasungur/visagen?style=for-the-badge&logo=github&color=FF6B6B" alt="Issues"/></a>
</p>

<!-- Quick Navigation -->
<p>
  <a href="#-features">✨ Features</a> •
  <a href="#-installation">📦 Installation</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#%EF%B8%8F-cli-tools">🛠️ CLI Tools</a> •
  <a href="#%EF%B8%8F-architecture">🏗️ Architecture</a> •
  <a href="#-contributing">🤝 Contributing</a>
</p>

<br/>

</div>

---

## 📖 Overview

**Visagen** is a next-generation face swapping framework built from the ground up with modern deep learning practices. Inspired by DeepFaceLab, Visagen reimagines the entire pipeline using **PyTorch Lightning**, offering cleaner code, better performance, and easier extensibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                         VISAGEN PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  📥 Extract  →  🏋️ Train  →  🎬 Swap  →  ✨ Postprocess         │
│      │            │           │             │                   │
│      ▼            ▼           ▼             ▼                   │
│  InsightFace   DFLModule    CBAM      Color Transfer            │
│  SegFormer     Lightning   Attention    Blending                │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Modern Architecture
- **ConvNeXt V2** encoder with GRN layers
- **CBAM** attention (Channel & Spatial)
- **Swish** activation for smooth gradients
- Skip connections for detail preservation

</td>
<td width="50%">

### 🎯 Advanced Training
- Multi-loss: DSSIM, L1, LPIPS, ID, GAN
- Mixed precision (FP16/BF16)
- Gradient clipping & LR scheduling
- Eyes/Mouth & Gaze consistency loss

</td>
</tr>
<tr>
<td width="50%">

### 🎨 Post-Processing
- 6 color transfer algorithms (RCT, LCT, SOT...)
- Neural color transfer (VGG-based)
- Laplacian, Poisson, Feather blending
- GFPGAN & GPEN face restoration

</td>
<td width="50%">

### ⚡ Production Ready
- ONNX & TensorRT export
- NVENC hardware encoding
- 30+ FPS inference
- 12 CLI tools

</td>
</tr>
</table>

---

## 📦 Installation

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

<details>
<summary><b>📋 Optional Dependencies</b></summary>

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

</details>

---

## 🚀 Quick Start

<details open>
<summary><b>📥 Step 1: Extract Faces</b></summary>

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

</details>

<details open>
<summary><b>🏋️ Step 2: Train Model</b></summary>

```bash
visagen-train \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/model \
    --epochs 500 \
    --batch-size 8 \
    --resolution 512
```

</details>

<details>
<summary><b>🔧 Step 3: Hyperparameter Tuning (Optional)</b></summary>

```bash
visagen-tune \
    --src-dir ./workspace/data_src/aligned \
    --dst-dir ./workspace/data_dst/aligned \
    --output-dir ./workspace/optuna \
    --n-trials 20 \
    --epochs-per-trial 50
```

</details>

<details>
<summary><b>🎬 Step 4: Merge Faces into Video</b></summary>

```bash
# Basic merge with trained model
visagen-merge input.mp4 output.mp4 -c ./workspace/model/model.ckpt

# With face restoration and hardware encoding
visagen-merge input.mp4 output.mp4 -c model.ckpt \
    --restore-face --restore-strength 0.7 \
    --codec h264_nvenc --color-transfer rct
```

</details>

<details>
<summary><b>📦 Step 5: Export Model for Production</b></summary>

```bash
# Export to ONNX
visagen-export model.ckpt -o model.onnx --validate

# Export to TensorRT (FP16)
visagen-export model.onnx -o model.engine --format tensorrt --precision fp16
```

</details>

<details>
<summary><b>🌐 Step 6: Launch Web Interface</b></summary>

```bash
visagen-gui --port 7860
```

</details>

---

## 🏗️ Architecture

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

---

## 🛠️ CLI Tools

| Command | Description |
|:--------|:------------|
| 📥 `visagen-extract` | Extract and align faces from images/videos |
| 🏋️ `visagen-train` | Train face swap model |
| 🎯 `visagen-pretrain` | Pretrain encoder on FFHQ/CelebA |
| 🔧 `visagen-tune` | Hyperparameter optimization (Optuna) |
| 🎬 `visagen-merge` | Merge face swaps with NVENC encoding |
| 📦 `visagen-export` | Export to ONNX/TensorRT |
| 📊 `visagen-sort` | Sort datasets (14 methods) |
| 🌐 `visagen-gui` | Launch Gradio web interface |
| 🎞️ `visagen-video` | Video frame extraction/creation |
| ✨ `visagen-enhance` | Batch face enhancement (GFPGAN/GPEN) |
| 📐 `visagen-resize` | Resize faceset with metadata |
| ⚡ `visagen-benchmark` | Performance benchmarks |

---

<details>
<summary><b>📁 Project Structure</b></summary>

```
visagen/
├── 📂 data/               # Data loading & augmentation
│   ├── dataset.py         # FaceDataset
│   ├── datamodule.py      # FaceDataModule
│   ├── dali_pipeline.py   # NVIDIA DALI GPU pipeline
│   └── augmentations.py
├── 📂 models/             # Neural network architectures
│   ├── encoder.py         # ConvNeXt encoder
│   ├── decoder.py         # Swish decoder
│   ├── attention.py       # CBAM attention
│   └── discriminator.py
├── 📂 training/           # Training logic
│   ├── dfl_module.py      # PyTorch Lightning module
│   ├── pretrain_module.py # Pretraining module
│   └── losses.py          # Loss functions
├── 📂 merger/             # Video processing pipeline
│   ├── video_io.py        # FFmpeg video I/O with NVENC
│   ├── frame_processor.py # Single-frame processing
│   ├── batch_processor.py # Parallel processing
│   └── merger.py          # High-level orchestration
├── 📂 postprocess/        # Post-processing
│   ├── color_transfer.py  # RCT, LCT, SOT, MKL, IDT algorithms
│   ├── neural_color.py    # VGG-based neural color transfer
│   ├── blending.py        # Laplacian, Poisson, Feather
│   ├── restore.py         # GFPGAN face restoration
│   └── gpen.py            # GPEN face restoration
├── 📂 export/             # Model export
│   ├── onnx_exporter.py   # ONNX export
│   ├── tensorrt_builder.py# TensorRT engine builder
│   └── validation.py      # Export validation
├── 📂 sorting/            # Dataset sorting
│   └── sorter.py          # 14 sorting methods
├── 📂 tuning/             # Hyperparameter optimization
│   └── optuna_tuner.py
├── 📂 tools/              # CLI tools
│   ├── extract_v2.py      # Face extraction
│   ├── train.py           # Training
│   ├── pretrain.py        # Pretraining
│   ├── merge.py           # Video merging
│   ├── export.py          # Model export
│   ├── sorter.py          # Dataset sorting
│   ├── tune.py            # HPO
│   ├── video_ed.py        # Video frame tools
│   ├── faceset_enhancer.py# Batch face enhancement
│   ├── faceset_resizer.py # Faceset resizing
│   ├── benchmark.py       # Performance benchmarks
│   └── gradio_app.py      # Web UI (10 tabs)
├── 📂 vision/             # Computer vision
│   ├── detector.py        # InsightFace SCRFD detection
│   ├── aligner.py         # Face alignment (Umeyama)
│   ├── segmenter.py       # SegFormer segmentation
│   ├── dflimg.py          # DFL image metadata
│   └── mask_export.py     # LabelMe/COCO export
└── 📂 tests/              # Unit tests (636+)
```

</details>

---

## 📊 Performance

<table>
<tr>
<td align="center">
<h3>🚄 50 img/s</h3>
<sub>Training Speed (RTX 3090)</sub>
</td>
<td align="center">
<h3>💾 8 GB</h3>
<sub>VRAM (512×512, batch=8)</sub>
</td>
<td align="center">
<h3>⚡ 30 FPS</h3>
<sub>Inference Speed</sub>
</td>
<td align="center">
<h3>✅ 636+</h3>
<sub>Unit Tests</sub>
</td>
</tr>
</table>

---

## 👥 Contributors

<a href="https://github.com/karasungur/visagen/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=karasungur/visagen" />
</a>

### Core Team

<table>
<tr>
<td align="center">
  <a href="https://github.com/karasungur">
    <img src="https://github.com/karasungur.png" width="100px;" alt="Mustafa Karasungur"/><br />
    <sub><b>Mustafa Karasungur</b></sub>
  </a><br />
  <sub>🏗️ Project Lead & Core Developer</sub>
</td>
</tr>
</table>

<p align="center">
  <i>Contributions are welcome! See the section below.</i>
</p>

---

## 🤝 Contributing

We love contributions! Whether you're fixing bugs, improving documentation, or proposing new features, your help is welcome.

<details>
<summary><b>📋 Quick Start for Contributors</b></summary>

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/visagen.git
cd visagen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Development installation
pip install -e ".[dev]"

# Create a new branch
git checkout -b feature/your-feature-name
```

</details>

<details>
<summary><b>🧪 Running Tests</b></summary>

```bash
# Run all tests
pytest visagen/tests/ -v

# Run with coverage
pytest visagen/tests/ --cov=visagen --cov-report=html

# Run specific test file
pytest visagen/tests/test_forward_pass.py -v
```

</details>

<details>
<summary><b>🎨 Code Style</b></summary>

We use **Ruff** for linting and formatting:

```bash
# Check code style
ruff check visagen/

# Auto-format code
ruff format visagen/

# Fix auto-fixable issues
ruff check visagen/ --fix
```

</details>

For detailed guidelines, see our [**Contributing Guide**](CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [DeepFaceLab](https://github.com/iperov/DeepFaceLab) - Original inspiration
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection
- [Optuna](https://optuna.org/) - Hyperparameter optimization

---

<div align="center">

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=100&section=footer" width="100%"/>

<br/>

**Made with ❤️ by [Mustafa Karasungur](https://github.com/karasungur)**

<sub>If you find this project useful, please consider giving it a ⭐</sub>

<br/>

<a href="https://github.com/karasungur/visagen/stargazers">
  <img src="https://img.shields.io/github/stars/karasungur/visagen?style=social" alt="GitHub Stars"/>
</a>

</div>
