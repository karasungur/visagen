# =============================================================================
# Visagen Dockerfile - Multi-stage build for GPU-accelerated face swapping
# =============================================================================
# Base: NVIDIA CUDA 12.1 + cuDNN 8 + TensorRT
# Features: Python 3.10, PyTorch 2.x, ONNX Runtime GPU, FFmpeg with NVENC
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA and system dependencies
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg with NVENC support
    ffmpeg \
    # Git for pip installs from repos
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY pyproject.toml ./

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Install the package with all optional dependencies except tensorrt
# (TensorRT requires separate installation from NVIDIA)
RUN pip install --no-cache-dir \
    pytorch-lightning>=2.0.0 \
    numpy>=1.24.0 \
    pyyaml>=6.0 \
    # Vision dependencies
    insightface>=0.7.3 \
    onnxruntime-gpu>=1.16.0 \
    transformers>=4.35.0 \
    opencv-python>=4.8.0 \
    tqdm>=4.65.0 \
    # Training dependencies
    lpips>=0.1.4 \
    # Postprocess dependencies
    scipy>=1.10.0 \
    # Tuning dependencies
    optuna>=3.0.0 \
    "optuna-integration[pytorch_lightning]>=3.0.0" \
    # GUI dependencies
    gradio>=4.0.0 \
    # Merger dependencies
    ffmpeg-python>=0.2.0 \
    imageio-ffmpeg>=0.4.8 \
    # Export dependencies
    onnx>=1.14.0 \
    onnx-simplifier>=0.4.0 \
    # Dev dependencies
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    ruff>=0.1.0

# -----------------------------------------------------------------------------
# Stage 3: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY visagen/ ./visagen/
COPY pyproject.toml ./

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create workspace directory for user data
RUN mkdir -p /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Expose port for Gradio UI
EXPOSE 7860

# Default command - start Gradio UI
CMD ["python", "-m", "visagen.tools.gradio_app"]

# -----------------------------------------------------------------------------
# Stage 4: Development image with additional tools
# -----------------------------------------------------------------------------
FROM runtime AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    tensorboard \
    mypy

# Install the package with dev extras
RUN pip install --no-cache-dir -e ".[dev]"

# Expose additional ports
EXPOSE 6006  # TensorBoard
EXPOSE 8888  # Jupyter

# Default command for development
CMD ["/bin/bash"]
