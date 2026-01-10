#!/usr/bin/env python3
"""
Visagen Cross-Platform Installation Script.

Features:
- Language selection (Turkish/English)
- Platform auto-detection (Windows/Linux/macOS)
- Python version check (3.10-3.12)
- CUDA auto-detection via nvidia-smi
- CUDA version selection (12.4/12.1/11.8/CPU)
- Installation profiles (minimal/full/dev/all)
- Real-time progress indicators
- Post-install verification

Usage:
    python install.py
"""

from __future__ import annotations

import itertools
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import venv
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)
MIN_DISK_GB = 5

# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGES (TR/EN)
# ═══════════════════════════════════════════════════════════════════════════════

MESSAGES = {
    "tr": {
        "welcome": "Visagen Kurulum Sihirbazi",
        "select_lang": "Dil secin / Select language:",
        "detecting": "Sistem algilaniyor...",
        "python_version": "Python surumu",
        "platform": "Platform",
        "disk_space": "Disk alani",
        "cuda_detected": "CUDA algilandi",
        "cuda_select": "CUDA surumunuzu secin:",
        "cuda_124": "CUDA 12.4 (RTX 40xx, en yeni)",
        "cuda_121": "CUDA 12.1 (RTX 40xx, yeni kartlar)",
        "cuda_118": "CUDA 11.8 (RTX 30xx, eski kartlar)",
        "cuda_cpu": "CPU only (GPU yok)",
        "profile_select": "Kurulum profilini secin:",
        "profile_minimal": "Minimal - Sadece temel egitim",
        "profile_full": "Full - Tum ozellikler",
        "profile_dev": "Developer - Gelistirici araclari dahil",
        "profile_all": "All - Her sey (restore dahil)",
        "installing": "Kuruluyor",
        "success": "Kurulum basarili!",
        "error": "Hata",
        "verify": "Kurulum dogrulaniyor...",
        "venv_create": "Virtual environment olusturuluyor...",
        "venv_exists": "Mevcut .venv bulundu. Uzerine yazilsin mi? (e/h)",
        "pip_upgrade": "pip guncelleniyor...",
        "pytorch_install": "PyTorch kuruluyor...",
        "visagen_install": "Visagen kuruluyor...",
        "done": "Tamamlandi!",
        "activate_hint": "Aktive etmek icin:",
        "test_hint": "Test etmek icin:",
        "train_hint": "Egitime baslamak icin:",
        "cancelled": "Kurulum iptal edildi.",
        "steps_total": "Kurulum: {0} adim",
        "step_system": "Sistem Kontrolleri",
        "step_cuda": "CUDA Secimi",
        "step_profile": "Kurulum Profili",
        "step_venv": "Virtual Environment",
        "step_packages": "Paket Kurulumu",
        "step_verify": "Dogrulama",
        "recommended": "Onerilen",
        "detected": "Algilandi",
        "yes": "e",
        "no": "h",
        "invalid_choice": "Gecersiz secim, tekrar deneyin.",
        "python_error": "Python {0}.{1}+ gerekli. Mevcut: {2}.{3}",
        "not_in_repo": "Bu script visagen repo kok dizininde calistirilmali.",
        "disk_error": "Yetersiz disk alani. Gerekli: {0} GB, Mevcut: {1:.1f} GB",
        "estimate": "~{0}",
        "minutes": "dakika",
        "seconds": "saniye",
    },
    "en": {
        "welcome": "Visagen Installation Wizard",
        "select_lang": "Dil secin / Select language:",
        "detecting": "Detecting system...",
        "python_version": "Python version",
        "platform": "Platform",
        "disk_space": "Disk space",
        "cuda_detected": "CUDA detected",
        "cuda_select": "Select your CUDA version:",
        "cuda_124": "CUDA 12.4 (RTX 40xx, newest)",
        "cuda_121": "CUDA 12.1 (RTX 40xx, newer cards)",
        "cuda_118": "CUDA 11.8 (RTX 30xx, older cards)",
        "cuda_cpu": "CPU only (no GPU)",
        "profile_select": "Select installation profile:",
        "profile_minimal": "Minimal - Basic training only",
        "profile_full": "Full - All features",
        "profile_dev": "Developer - With dev tools",
        "profile_all": "All - Everything (incl. restore)",
        "installing": "Installing",
        "success": "Installation successful!",
        "error": "Error",
        "verify": "Verifying installation...",
        "venv_create": "Creating virtual environment...",
        "venv_exists": "Existing .venv found. Overwrite? (y/n)",
        "pip_upgrade": "Upgrading pip...",
        "pytorch_install": "Installing PyTorch...",
        "visagen_install": "Installing Visagen...",
        "done": "Done!",
        "activate_hint": "To activate:",
        "test_hint": "To test:",
        "train_hint": "To start training:",
        "cancelled": "Installation cancelled.",
        "steps_total": "Installation: {0} steps",
        "step_system": "System Checks",
        "step_cuda": "CUDA Selection",
        "step_profile": "Installation Profile",
        "step_venv": "Virtual Environment",
        "step_packages": "Package Installation",
        "step_verify": "Verification",
        "recommended": "Recommended",
        "detected": "Detected",
        "yes": "y",
        "no": "n",
        "invalid_choice": "Invalid choice, please try again.",
        "python_error": "Python {0}.{1}+ required. Current: {2}.{3}",
        "not_in_repo": "This script must be run from visagen repo root.",
        "disk_error": "Insufficient disk space. Required: {0} GB, Available: {1:.1f} GB",
        "estimate": "~{0}",
        "minutes": "min",
        "seconds": "sec",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

PROFILES = {
    "minimal": "",
    "full": "[full]",
    "dev": "[full,dev]",
    "all": "[full,dev,restore]",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CUDA INDEX URLS
# ═══════════════════════════════════════════════════════════════════════════════

CUDA_URLS = {
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP TIME ESTIMATES (seconds)
# ═══════════════════════════════════════════════════════════════════════════════

STEP_ESTIMATES = {
    "venv": 5,
    "pip_upgrade": 10,
    "pytorch": 180,
    "visagen": 60,
    "verify": 10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════════

# Enable ANSI colors on Windows
if sys.platform == "win32":
    os.system("")


class Colors:
    """Cross-platform ANSI color codes."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors for non-TTY or NO_COLOR mode."""
        for attr in dir(cls):
            if not attr.startswith("_") and attr.isupper() and attr != "RESET":
                setattr(cls, attr, "")
        cls.RESET = ""


# Check color support
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    Colors.disable()


def supports_emoji() -> bool:
    """Check if terminal supports emoji."""
    if sys.platform == "win32":
        return os.environ.get("WT_SESSION") is not None
    return True


def get_symbols() -> dict:
    """Get display symbols with fallback."""
    if supports_emoji():
        return {
            "success": "[OK]",
            "error": "[X]",
            "warning": "[!]",
            "info": "[i]",
            "check": "+",
            "cross": "x",
            "arrow": "->",
            "spinner": ["|", "/", "-", "\\"],
            "box_h": "=",
            "box_v": "|",
            "star": "*",
            "rocket": ">>>",
            "package": "[#]",
            "party": "!!!",
        }
    return {
        "success": "[OK]",
        "error": "[X]",
        "warning": "[!]",
        "info": "[i]",
        "check": "+",
        "cross": "x",
        "arrow": "->",
        "spinner": ["|", "/", "-", "\\"],
        "box_h": "=",
        "box_v": "|",
        "star": "*",
        "rocket": ">>>",
        "package": "[#]",
        "party": "!!!",
    }


SYM = get_symbols()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Setup logging to file."""
    logger = logging.getLogger("visagen_install")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler("install.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# SPINNER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class Spinner:
    """Animated spinner for long-running operations."""

    def __init__(self, message: str = "Processing") -> None:
        self.frames = SYM["spinner"]
        self.message = message
        self.running = False
        self.thread: threading.Thread | None = None
        self._cycle = itertools.cycle(self.frames)

    def _spin(self) -> None:
        while self.running:
            frame = next(self._cycle)
            sys.stdout.write(f"\r  {frame} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, success: bool = True) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        symbol = SYM["check"] if success else SYM["cross"]
        color = Colors.GREEN if success else Colors.RED
        sys.stdout.write(f"\r  {color}{symbol}{Colors.RESET} {self.message}   \n")
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def print_banner(lang: str) -> None:
    """Print welcome banner."""
    msg = MESSAGES[lang]
    width = 62
    print()
    print(f"{Colors.CYAN}{SYM['box_h'] * width}{Colors.RESET}")
    title = f"  {SYM['rocket']} {msg['welcome']}  "
    print(f"{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.CYAN}{SYM['box_h'] * width}{Colors.RESET}")
    print()


def print_step_header(step_num: int, total: int, name: str, estimate: str = "") -> None:
    """Print step header."""
    est_str = f" ({estimate})" if estimate else ""
    print(
        f"\n{Colors.BOLD}[{step_num}/{total}] {name}{Colors.DIM}{est_str}{Colors.RESET}"
    )
    print("-" * 40)


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}{SYM['check']} {msg}{Colors.RESET}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}{SYM['cross']} {msg}{Colors.RESET}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}{SYM['warning']} {msg}{Colors.RESET}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"  {Colors.CYAN}{SYM['check']}{Colors.RESET} {msg}")


def print_step_done(msg: str) -> None:
    """Print step completion."""
    print(f"{Colors.GREEN}{SYM['success']} {msg}{Colors.RESET}\n")


def format_estimate(seconds: int, lang: str) -> str:
    """Format time estimate."""
    msg = MESSAGES[lang]
    if seconds >= 60:
        return f"{msg['estimate'].format(seconds // 60)} {msg['minutes']}"
    return f"{msg['estimate'].format(seconds)} {msg['seconds']}"


# ═══════════════════════════════════════════════════════════════════════════════
# USER INPUT
# ═══════════════════════════════════════════════════════════════════════════════


def prompt_choice(
    question: str,
    options: list[tuple[str, str, bool]],
    lang: str,
) -> int:
    """
    Prompt user for choice.

    Args:
        question: Question to ask
        options: List of (label, description, is_recommended)
        lang: Language code

    Returns:
        Selected option index (0-based)
    """
    msg = MESSAGES[lang]
    print(f"\n  {question}")

    for i, (label, desc, recommended) in enumerate(options, 1):
        rec_str = f" {Colors.YELLOW}{SYM['star']}{Colors.RESET}" if recommended else ""
        if desc:
            print(f"    [{i}] {label} - {desc}{rec_str}")
        else:
            print(f"    [{i}] {label}{rec_str}")

    while True:
        try:
            choice = input("  > ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except (ValueError, EOFError):
            pass
        print(f"  {Colors.YELLOW}{msg['invalid_choice']}{Colors.RESET}")


def prompt_yes_no(question: str, lang: str, default: bool = True) -> bool:
    """Prompt yes/no question."""
    msg = MESSAGES[lang]
    yes_key = msg["yes"]
    no_key = msg["no"]

    while True:
        try:
            choice = input(f"  {question} ").strip().lower()
            if not choice:
                return default
            if choice == yes_key:
                return True
            if choice == no_key:
                return False
        except EOFError:
            return default
        print(f"  {Colors.YELLOW}{msg['invalid_choice']}{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════════════════════


def check_python_version(lang: str, logger: logging.Logger) -> bool:
    """Check Python version is 3.10-3.12."""
    msg = MESSAGES[lang]
    major, minor = sys.version_info[:2]
    logger.info(f"Python version: {major}.{minor}")

    if (major, minor) < MIN_PYTHON:
        print_error(
            msg["python_error"].format(MIN_PYTHON[0], MIN_PYTHON[1], major, minor)
        )
        return False

    print_info(f"{msg['python_version']}: {major}.{minor}")
    return True


def check_in_repo(lang: str, logger: logging.Logger) -> bool:
    """Check if running from visagen repo root."""
    msg = MESSAGES[lang]
    pyproject = Path("pyproject.toml")

    if not pyproject.exists():
        print_error(msg["not_in_repo"])
        logger.error("Not in repo root - pyproject.toml not found")
        return False

    # Verify it's visagen project
    content = pyproject.read_text()
    if 'name = "visagen"' not in content:
        print_error(msg["not_in_repo"])
        logger.error("Not visagen repo")
        return False

    return True


def check_disk_space(lang: str, logger: logging.Logger) -> bool:
    """Check available disk space."""
    msg = MESSAGES[lang]

    try:
        if sys.platform == "win32":
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p("."), None, None, ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(".")
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        logger.info(f"Disk space: {free_gb:.1f} GB")

        if free_gb < MIN_DISK_GB:
            print_error(msg["disk_error"].format(MIN_DISK_GB, free_gb))
            return False

        print_info(f"{msg['disk_space']}: {free_gb:.1f} GB")
        return True

    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        print_warning(f"{msg['disk_space']}: unknown")
        return True


def detect_platform(lang: str, logger: logging.Logger) -> dict:
    """Detect OS and architecture."""
    msg = MESSAGES[lang]

    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    logger.info(f"Platform: {info}")
    print_info(f"{msg['platform']}: {info['system']} ({info['machine']})")

    return info


def detect_cuda_version(logger: logging.Logger) -> str | None:
    """Auto-detect CUDA version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            driver = result.stdout.strip().split("\n")[0]
            driver_major = int(driver.split(".")[0])
            logger.info(f"NVIDIA driver: {driver}")

            # Driver to CUDA mapping
            if driver_major >= 550:
                return "12.4"
            elif driver_major >= 525:
                return "12.1"
            elif driver_major >= 450:
                return "11.8"

        return None

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        logger.debug(f"nvidia-smi failed: {e}")
        return None


def check_system_dependencies(logger: logging.Logger) -> list[str]:
    """Check for system dependencies. Returns list of missing deps."""
    missing = []

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
        logger.warning("ffmpeg not found")

    # Check for git
    if not shutil.which("git"):
        missing.append("git")
        logger.warning("git not found")

    return missing


# ═══════════════════════════════════════════════════════════════════════════════
# USER SELECTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def select_language() -> str:
    """Select language (tr/en)."""
    print("\nDil secin / Select language:")
    print("  [1] Turkce")
    print("  [2] English")

    while True:
        try:
            choice = input("  > ").strip()
            if choice == "1":
                return "tr"
            if choice == "2":
                return "en"
        except EOFError:
            return "en"
        print("  Invalid / Gecersiz")


def select_cuda(lang: str, detected: str | None) -> str:
    """Select CUDA version."""
    msg = MESSAGES[lang]

    options = []

    # CUDA 12.4
    is_detected = detected == "12.4"
    label = msg["cuda_124"]
    if is_detected:
        label += f" ({msg['detected']})"
    options.append((label, "", is_detected))

    # CUDA 12.1
    is_detected = detected == "12.1"
    label = msg["cuda_121"]
    if is_detected:
        label += f" ({msg['detected']})"
    options.append((label, "", is_detected or detected is None))

    # CUDA 11.8
    is_detected = detected == "11.8"
    label = msg["cuda_118"]
    if is_detected:
        label += f" ({msg['detected']})"
    options.append((label, "", is_detected))

    # CPU
    options.append((msg["cuda_cpu"], "", False))

    idx = prompt_choice(msg["cuda_select"], options, lang)

    cuda_map = {0: "12.4", 1: "12.1", 2: "11.8", 3: "cpu"}
    return cuda_map[idx]


def select_profile(lang: str) -> str:
    """Select installation profile."""
    msg = MESSAGES[lang]

    options = [
        (msg["profile_minimal"], "", False),
        (msg["profile_full"], msg["recommended"], True),
        (msg["profile_dev"], "", False),
        (msg["profile_all"], "", False),
    ]

    idx = prompt_choice(msg["profile_select"], options, lang)

    profile_map = {0: "minimal", 1: "full", 2: "dev", 3: "all"}
    return profile_map[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_pip_path(venv_path: Path) -> Path:
    """Get pip executable path."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_python_path(venv_path: Path) -> Path:
    """Get python executable path."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def create_venv(
    venv_path: Path,
    lang: str,
    logger: logging.Logger,
) -> bool:
    """Create virtual environment."""
    msg = MESSAGES[lang]

    # Check if exists
    if venv_path.exists():
        if not prompt_yes_no(msg["venv_exists"], lang, default=False):
            return True  # Use existing

        # Remove existing
        spinner = Spinner("Removing old .venv")
        spinner.start()
        try:
            shutil.rmtree(venv_path)
            spinner.stop(True)
        except Exception as e:
            spinner.stop(False)
            logger.error(f"Failed to remove venv: {e}")
            return False

    # Create new venv
    spinner = Spinner(msg["venv_create"])
    spinner.start()

    try:
        venv.create(venv_path, with_pip=True)
        spinner.stop(True)
        logger.info("Virtual environment created")
        return True

    except Exception as e:
        spinner.stop(False)
        print_error(f"{msg['error']}: {e}")
        logger.error(f"venv creation failed: {e}")
        return False


def upgrade_pip(
    venv_path: Path,
    lang: str,
    logger: logging.Logger,
) -> bool:
    """Upgrade pip, setuptools, wheel."""
    msg = MESSAGES[lang]
    pip = get_pip_path(venv_path)

    spinner = Spinner(msg["pip_upgrade"])
    spinner.start()

    try:
        result = subprocess.run(
            [str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            spinner.stop(True)
            logger.info("pip upgraded successfully")
            return True
        else:
            spinner.stop(False)
            logger.error(f"pip upgrade failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        spinner.stop(False)
        print_error("Timeout")
        logger.error("pip upgrade timeout")
        return False
    except Exception as e:
        spinner.stop(False)
        print_error(str(e))
        logger.error(f"pip upgrade error: {e}")
        return False


def run_pip_install(
    pip: Path,
    args: list[str],
    message: str,
    logger: logging.Logger,
) -> bool:
    """Run pip install with real-time output."""
    print(f"\n{SYM['package']} {message}")
    print("-" * 50)

    cmd = [str(pip), "install"] + args

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            line = line.strip()
            if any(
                kw in line
                for kw in ["Collecting", "Downloading", "Installing", "Building"]
            ):
                print(f"  {line}")
            elif "Successfully" in line:
                print(f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} {line}")

            logger.debug(line)

        process.wait()
        print("-" * 50)

        if process.returncode == 0:
            logger.info("pip install successful")
            return True
        else:
            logger.error(f"pip install failed with code {process.returncode}")
            return False

    except Exception as e:
        print_error(str(e))
        logger.error(f"pip install error: {e}")
        return False


def install_pytorch(
    venv_path: Path,
    cuda_version: str,
    lang: str,
    logger: logging.Logger,
) -> bool:
    """Install PyTorch with CUDA support."""
    msg = MESSAGES[lang]
    pip = get_pip_path(venv_path)

    cuda_url = CUDA_URLS[cuda_version]
    cuda_label = f"CUDA {cuda_version}" if cuda_version != "cpu" else "CPU"

    message = f"{msg['pytorch_install']} ({cuda_label})"

    args = [
        "torch",
        "torchvision",
        "--index-url",
        cuda_url,
    ]

    return run_pip_install(pip, args, message, logger)


def install_visagen(
    venv_path: Path,
    profile: str,
    lang: str,
    logger: logging.Logger,
) -> bool:
    """Install visagen with selected profile."""
    msg = MESSAGES[lang]
    pip = get_pip_path(venv_path)

    profile_str = PROFILES[profile]
    message = f"{msg['visagen_install']} [{profile}]"

    args = ["-e", f".{profile_str}"]

    return run_pip_install(pip, args, message, logger)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def verify_installation(
    venv_path: Path,
    lang: str,
    logger: logging.Logger,
) -> bool:
    """Verify installation."""
    msg = MESSAGES[lang]
    python = get_python_path(venv_path)

    spinner = Spinner(msg["verify"])
    spinner.start()

    try:
        # Check torch
        result = subprocess.run(
            [str(python), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            spinner.stop(False)
            print_error("torch import failed")
            return False

        torch_version = result.stdout.strip()
        logger.info(f"torch version: {torch_version}")

        # Check CUDA
        result = subprocess.run(
            [str(python), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        cuda_available = result.stdout.strip() == "True"
        logger.info(f"CUDA available: {cuda_available}")

        # Check visagen
        result = subprocess.run(
            [str(python), "-c", "import visagen; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            spinner.stop(False)
            print_error("visagen import failed")
            logger.error(f"visagen import failed: {result.stderr}")
            return False

        spinner.stop(True)

        # Print details
        print_info(f"torch: {torch_version}")
        print_info(f"CUDA available: {cuda_available}")
        print_info("visagen: OK")

        return True

    except subprocess.TimeoutExpired:
        spinner.stop(False)
        print_error("Verification timeout")
        return False
    except Exception as e:
        spinner.stop(False)
        print_error(str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# POST-INSTALL
# ═══════════════════════════════════════════════════════════════════════════════


def show_system_deps_warning(missing: list[str], lang: str) -> None:
    """Show warning about missing system dependencies."""
    if not missing:
        return

    print(f"\n{Colors.YELLOW}{SYM['warning']} System dependencies:{Colors.RESET}")

    if "ffmpeg" in missing:
        if sys.platform == "win32":
            print("  - FFmpeg: https://ffmpeg.org/download.html")
            print("    Download and add to PATH")
        else:
            print("  - FFmpeg: sudo apt install ffmpeg")


def show_next_steps(venv_path: Path, lang: str) -> None:
    """Show activation and usage instructions."""
    msg = MESSAGES[lang]

    print(f"\n{'=' * 60}")
    print(f"\n{Colors.GREEN}{SYM['party']} {msg['done']}{Colors.RESET}\n")

    # Activation
    print(f"{msg['activate_hint']}")
    if sys.platform == "win32":
        print(f"  {Colors.CYAN}.venv\\Scripts\\activate{Colors.RESET}")
    else:
        print(f"  {Colors.CYAN}source .venv/bin/activate{Colors.RESET}")

    # Training
    print(f"\n{msg['train_hint']}")
    print(f"  {Colors.CYAN}visagen-train --help{Colors.RESET}")

    # Testing
    print(f"\n{msg['test_hint']}")
    print(f"  {Colors.CYAN}pytest visagen/tests/ -v{Colors.RESET}")

    print(f"\n{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Main installation flow."""
    logger = setup_logging()
    logger.info("Installation started")

    try:
        # 1. Language selection
        lang = select_language()
        msg = MESSAGES[lang]

        # 2. Print banner
        print_banner(lang)

        # 3. System checks
        print(f"\n{Colors.BOLD}{msg['step_system']}{Colors.RESET}")
        print("-" * 40)

        if not check_python_version(lang, logger):
            return 1

        if not check_in_repo(lang, logger):
            return 1

        if not check_disk_space(lang, logger):
            return 1

        detect_platform(lang, logger)

        detected_cuda = detect_cuda_version(logger)
        if detected_cuda:
            print_info(f"{msg['cuda_detected']}: {detected_cuda}")

        missing_deps = check_system_dependencies(logger)

        print_step_done(msg["step_system"])

        # 4. CUDA selection
        cuda = select_cuda(lang, detected_cuda)
        cuda_label = f"CUDA {cuda}" if cuda != "cpu" else "CPU"
        print_step_done(f"{cuda_label}")

        # 5. Profile selection
        profile = select_profile(lang)
        print_step_done(f"{profile}")

        # 6. Create venv
        print_step_header(
            4,
            6,
            msg["step_venv"],
            format_estimate(STEP_ESTIMATES["venv"], lang),
        )

        venv_path = Path(".venv")

        if not create_venv(venv_path, lang, logger):
            return 1

        if not upgrade_pip(venv_path, lang, logger):
            return 1

        print_step_done(msg["step_venv"])

        # 7. Install packages
        total_time = STEP_ESTIMATES["pytorch"] + STEP_ESTIMATES["visagen"]
        print_step_header(
            5,
            6,
            msg["step_packages"],
            format_estimate(total_time, lang),
        )

        if not install_pytorch(venv_path, cuda, lang, logger):
            return 1

        if not install_visagen(venv_path, profile, lang, logger):
            return 1

        print_step_done(msg["step_packages"])

        # 8. Verification
        print_step_header(
            6,
            6,
            msg["step_verify"],
            format_estimate(STEP_ESTIMATES["verify"], lang),
        )

        if not verify_installation(venv_path, lang, logger):
            return 1

        print_step_done(msg["step_verify"])

        # 9. Post-install
        if missing_deps:
            show_system_deps_warning(missing_deps, lang)

        show_next_steps(venv_path, lang)

        logger.info("Installation completed successfully")
        return 0

    except KeyboardInterrupt:
        print(
            f"\n\n{Colors.YELLOW}{MESSAGES.get(lang, MESSAGES['en'])['cancelled']}{Colors.RESET}"
        )
        logger.info("Installation cancelled by user")
        return 130

    except Exception as e:
        logger.exception("Installation failed")
        print_error(f"Installation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
