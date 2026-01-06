# Contributing to Visagen

<div align="center">

<img src="https://raw.githubusercontent.com/karasungur/visagen/main/assets/logo.png" alt="Visagen Logo" width="120"/>

**Thank you for your interest in contributing to Visagen!**

*Every contribution, no matter how small, makes a difference.*

</div>

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Making Changes](#-making-changes)
- [Commit Guidelines](#-commit-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Issue Guidelines](#-issue-guidelines)
- [Recognition](#-recognition)

---

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. Please:

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Focus on helpful feedback and solutions
- **Be inclusive** - Welcome newcomers and help them get started
- **Be patient** - Remember that everyone was a beginner once

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA 11.8+ for GPU acceleration

### Fork & Clone

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/visagen.git
cd visagen
```

3. **Add upstream** remote:

```bash
git remote add upstream https://github.com/karasungur/visagen.git
```

---

## 🔧 Development Setup

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# For full development (all optional dependencies)
pip install -e ".[full,dev]"
```

### Verify Installation

```bash
# Run tests to verify everything works
pytest visagen/tests/ -v --tb=short

# Check code style
ruff check visagen/
```

---

## ✏️ Making Changes

### 1. Create a Branch

Always create a new branch for your changes:

```bash
# Update main branch first
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description

# Or for documentation
git checkout -b docs/what-you-changed
```

### 2. Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/add-tensorrt-export` |
| Bug Fix | `fix/description` | `fix/memory-leak-decoder` |
| Documentation | `docs/description` | `docs/update-api-reference` |
| Refactor | `refactor/description` | `refactor/simplify-losses` |
| Test | `test/description` | `test/add-integration-tests` |

### 3. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation if needed

---

## 📝 Commit Guidelines

We use **Gitmoji + Conventional Commits** for clear, semantic commit history.

### Commit Message Format

```
<emoji> <type>(<scope>): <description>

[optional body]

[optional footer]
```

### Common Commit Types

| Emoji | Type | Description |
|:-----:|------|-------------|
| ✨ | `feat` | New feature |
| 🐛 | `fix` | Bug fix |
| 📝 | `docs` | Documentation |
| 🎨 | `style` | Code formatting (no logic change) |
| ♻️ | `refactor` | Code refactoring |
| ⚡ | `perf` | Performance improvement |
| ✅ | `test` | Adding/updating tests |
| 🔧 | `chore` | Maintenance tasks |
| 🏗️ | `build` | Build system changes |
| 👷 | `ci` | CI/CD changes |

### Examples

```bash
# New feature
git commit -m "✨ feat(export): add TensorRT INT8 quantization support"

# Bug fix
git commit -m "🐛 fix(decoder): resolve memory leak in skip connections"

# Documentation
git commit -m "📝 docs(readme): update installation instructions"

# Tests
git commit -m "✅ test(losses): add unit tests for gaze loss"
```

---

## 🔀 Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest visagen/tests/ -v
   ```

3. **Check code style**:
   ```bash
   ruff check visagen/
   ruff format visagen/
   ```

### Submitting a PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill in the PR template with:
   - **Description** of changes
   - **Related issues** (if any)
   - **Testing done**
   - **Screenshots** (for UI changes)

### PR Review Process

- All PRs require at least one review
- CI must pass (tests, linting)
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized

---

## 🎨 Code Style

We use **Ruff** for linting and formatting, configured in `pyproject.toml`.

### Quick Commands

```bash
# Check for issues
ruff check visagen/

# Auto-fix issues
ruff check visagen/ --fix

# Format code
ruff format visagen/
```

### Style Guidelines

- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort rules
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

### Example Code Style

```python
"""Module docstring describing the module purpose."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class MyModule(nn.Module):
    """Short description of the class.

    Longer description if needed, explaining the purpose
    and usage of this class.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layers.

    Example:
        >>> module = MyModule(input_dim=64, hidden_dim=128)
        >>> output = module(torch.randn(1, 64))
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input tensor.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, hidden_dim).
        """
        return self.layer(x)
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest visagen/tests/ -v

# Run with coverage
pytest visagen/tests/ --cov=visagen --cov-report=html

# Run specific test file
pytest visagen/tests/test_forward_pass.py -v

# Run tests matching pattern
pytest visagen/tests/ -k "test_encoder" -v

# Run only fast tests (skip slow ones)
pytest visagen/tests/ -m "not slow" -v
```

### Writing Tests

- Place tests in `visagen/tests/`
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

```python
"""Tests for the encoder module."""

import pytest
import torch

from visagen.models.encoders.convnext import ConvNeXtEncoder


class TestConvNeXtEncoder:
    """Test suite for ConvNeXtEncoder."""

    def test_output_shape(self) -> None:
        """Test that encoder produces correct output shape."""
        encoder = ConvNeXtEncoder(in_channels=3)
        x = torch.randn(2, 3, 256, 256)

        features, latent = encoder(x)

        assert len(features) == 4
        assert latent.shape[0] == 2

    @pytest.mark.slow
    def test_large_input(self) -> None:
        """Test encoder with large input (slow test)."""
        encoder = ConvNeXtEncoder(in_channels=3)
        x = torch.randn(1, 3, 1024, 1024)

        features, latent = encoder(x)

        assert latent is not None
```

---

## 📚 Documentation

### Docstrings

All public functions, classes, and modules should have docstrings:

```python
def process_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: int = 512,
) -> np.ndarray:
    """Align and crop face from image using landmarks.

    Args:
        image: Input image as BGR numpy array (H, W, 3).
        landmarks: Facial landmarks array of shape (68, 2).
        target_size: Output image size (default: 512).

    Returns:
        Aligned face image as BGR numpy array (target_size, target_size, 3).

    Raises:
        ValueError: If landmarks shape is invalid.

    Example:
        >>> face = process_face(image, landmarks, target_size=256)
        >>> assert face.shape == (256, 256, 3)
    """
```

### README Updates

When adding new features:
1. Update the feature list if applicable
2. Add usage examples to Quick Start
3. Update CLI tools table if adding new commands

---

## 🐛 Issue Guidelines

### Reporting Bugs

Please include:
- **Clear title** describing the issue
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment info** (OS, Python version, GPU)
- **Error messages** and tracebacks
- **Screenshots** if applicable

### Requesting Features

Please include:
- **Clear description** of the feature
- **Use case** - why is this needed?
- **Proposed solution** (optional)
- **Alternatives considered** (optional)

---

## 🏆 Recognition

All contributors are recognized in our README! Your contributions, whether code, documentation, bug reports, or feature requests, are valued and appreciated.

### Ways to Contribute

| Type | Examples |
|------|----------|
| 💻 Code | Features, bug fixes, optimizations |
| 📝 Documentation | README, docstrings, tutorials |
| 🐛 Bug Reports | Finding and reporting issues |
| 💡 Ideas | Feature requests, suggestions |
| 🧪 Testing | Writing tests, testing PRs |
| 🎨 Design | UI/UX improvements |
| 🌍 Translation | Internationalization |

---

<div align="center">

## Questions?

If you have any questions, feel free to open an [Issue](https://github.com/karasungur/visagen/issues).

<br/>

**Happy Contributing!** 🎉

<br/>

<sub>Made with ❤️ by the Visagen community</sub>

</div>
