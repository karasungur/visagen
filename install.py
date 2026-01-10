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
- Non-interactive mode for CI/CD
- Automatic retry on network failures
- GPU detection with recommendations

Usage:
    python install.py                                    # Interactive mode
    python install.py -y --cuda 12.4 --profile full     # Non-interactive mode
"""

from __future__ import annotations

import argparse
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)
MIN_DISK_GB = 5
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

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
        "retry": "Yeniden deneniyor ({0}/{1})...",
        "gpu_name": "GPU",
        "gpu_memory": "GPU bellegi",
        "recommended_settings": "Onerilen egitim ayarlari",
        "resolution": "Cozunurluk",
        "batch_size": "Batch boyutu",
        "summary": "Kurulum Ozeti",
        "non_interactive": "Non-interactive mod",
        "using_defaults": "Varsayilan ayarlar kullaniliyor",
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
        "retry": "Retrying ({0}/{1})...",
        "gpu_name": "GPU",
        "gpu_memory": "GPU memory",
        "recommended_settings": "Recommended training settings",
        "resolution": "Resolution",
        "batch_size": "Batch size",
        "summary": "Installation Summary",
        "non_interactive": "Non-interactive mode",
        "using_defaults": "Using default settings",
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
# SPINNER CLASSES
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


class LiveSpinner:
    """Spinner with elapsed time display for long operations."""

    def __init__(self, message: str = "Processing") -> None:
        self.frames = SYM["spinner"]
        self.message = message
        self.running = False
        self.thread: threading.Thread | None = None
        self._cycle = itertools.cycle(self.frames)
        self.start_time: float = 0

    def _format_elapsed(self) -> str:
        """Format elapsed time as MM:SS or SS s."""
        elapsed = int(time.time() - self.start_time)
        if elapsed >= 60:
            mins, secs = divmod(elapsed, 60)
            return f"{mins}m {secs:02d}s"
        return f"{elapsed}s"

    def _spin(self) -> None:
        while self.running:
            frame = next(self._cycle)
            elapsed = self._format_elapsed()
            # Overwrite line with spinner + message + elapsed time
            sys.stdout.write(f"\r  {frame} {self.message}... [{elapsed}]   ")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self) -> None:
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def update_message(self, message: str) -> None:
        """Update spinner message without stopping."""
        self.message = message

    def stop(self, success: bool = True) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

        elapsed = self._format_elapsed()
        symbol = SYM["check"] if success else SYM["cross"]
        color = Colors.GREEN if success else Colors.RED

        # Clear line and print final status
        sys.stdout.write(
            f"\r  {color}{symbol}{Colors.RESET} {self.message} [{elapsed}]   \n"
        )
        sys.stdout.flush()


class InstallProgress:
    """Track overall installation progress with elapsed time."""

    def __init__(self, total_steps: int, lang: str = "en") -> None:
        self.total = total_steps
        self.current = 0
        self.lang = lang
        self.start_time = time.time()

    def _format_elapsed(self) -> str:
        """Format elapsed time."""
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        return f"{mins}m {secs:02d}s"

    def next_step(self, name: str, estimate_seconds: int = 0) -> None:
        """Move to next step and display header."""
        self.current += 1
        elapsed = self._format_elapsed()

        # Format estimate
        if estimate_seconds >= 60:
            est_str = f" (~{estimate_seconds // 60}m)"
        elif estimate_seconds > 0:
            est_str = f" (~{estimate_seconds}s)"
        else:
            est_str = ""

        print(
            f"\n{Colors.BOLD}[{self.current}/{self.total}] {name}{est_str}{Colors.RESET}"
        )
        print(f"{Colors.DIM}Elapsed: {elapsed}{Colors.RESET}")
        print("-" * 50)

    def step_done(self, name: str) -> None:
        """Mark current step as done."""
        print(f"{Colors.GREEN}{SYM['success']} {name}{Colors.RESET}")

    def complete(self) -> None:
        """Show completion message with total time."""
        elapsed = self._format_elapsed()
        msg = MESSAGES[self.lang]
        print(f"\n{'=' * 60}")
        print(
            f"\n{Colors.GREEN}{SYM['party']} {msg['done']} (Total: {elapsed}){Colors.RESET}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InstallSummary:
    """Track and display installation summary."""

    cuda_version: str = ""
    profile: str = ""
    torch_version: str = ""
    cuda_available: bool = False
    gpu_name: str = ""
    gpu_memory_gb: int = 0
    total_time: float = 0
    warnings: list[str] = field(default_factory=list)

    def show(self, lang: str) -> None:
        """Display summary report."""
        msg = MESSAGES[lang]

        print(f"\n{'=' * 60}")
        print(f"{Colors.BOLD}  {msg['summary'].upper()}{Colors.RESET}")
        print(f"{'=' * 60}\n")

        # Configuration
        print(f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} Profile: {self.profile}")
        cuda_label = (
            f"CUDA {self.cuda_version}" if self.cuda_version != "cpu" else "CPU"
        )
        print(f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} CUDA: {cuda_label}")

        if self.torch_version:
            print(
                f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} PyTorch: {self.torch_version}"
            )

        gpu_status = "Yes" if self.cuda_available else "No"
        print(
            f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} GPU Available: {gpu_status}"
        )

        # GPU info
        if self.gpu_name:
            print(
                f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} {msg['gpu_name']}: {self.gpu_name} ({self.gpu_memory_gb} GB)"
            )

            # Recommendations
            settings = get_recommended_settings(self.gpu_memory_gb)
            print(
                f"\n  {Colors.CYAN}{SYM['info']}{Colors.RESET} {msg['recommended_settings']}:"
            )
            print(f"      {msg['resolution']}: {settings['resolution']}")
            print(f"      {msg['batch_size']}: {settings['batch_size']}")

        # Warnings
        if self.warnings:
            print(f"\n  {Colors.YELLOW}{SYM['warning']} Warnings:{Colors.RESET}")
            for w in self.warnings:
                print(f"    - {w}")

        # Time
        mins, secs = divmod(int(self.total_time), 60)
        print(f"\n  Total time: {mins}m {secs:02d}s")
        print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visagen Cross-Platform Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                                    # Interactive mode
  python install.py -y --cuda 12.4 --profile full     # Non-interactive
  python install.py -y --cuda cpu --profile minimal   # CPU-only minimal
        """,
    )

    parser.add_argument(
        "-y",
        "--non-interactive",
        action="store_true",
        help="Run without prompts (use defaults or specified options)",
    )
    parser.add_argument(
        "--cuda",
        choices=["12.4", "12.1", "11.8", "cpu"],
        default=None,
        help="CUDA version (auto-detect if not specified)",
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "full", "dev", "all"],
        default="full",
        help="Installation profile (default: full)",
    )
    parser.add_argument(
        "--lang",
        choices=["tr", "en"],
        default="en",
        help="Language (default: en)",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Skip virtual environment creation (install to current env)",
    )

    return parser.parse_args()


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


def detect_gpu_info(logger: logging.Logger) -> tuple[str, int]:
    """Detect GPU name and memory in GB."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) >= 2:
                name = parts[0].strip()
                memory_mb = int(parts[1].strip())
                memory_gb = memory_mb // 1024

                logger.info(f"GPU: {name}, Memory: {memory_gb} GB")
                return name, memory_gb

        return "", 0

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        logger.debug(f"GPU detection failed: {e}")
        return "", 0


def get_recommended_settings(gpu_memory_gb: int) -> dict:
    """Get recommended training settings based on GPU memory."""
    if gpu_memory_gb >= 24:  # RTX 3090, 4090
        return {"resolution": 512, "batch_size": 8, "note": "High-end GPU"}
    elif gpu_memory_gb >= 12:  # RTX 3080, 4070
        return {"resolution": 512, "batch_size": 4, "note": "Mid-range GPU"}
    elif gpu_memory_gb >= 8:  # RTX 3060, 4060
        return {"resolution": 256, "batch_size": 4, "note": "Entry-level GPU"}
    else:
        return {"resolution": 128, "batch_size": 2, "note": "Limited VRAM"}


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
            [
                str(pip),
                "install",
                "--upgrade",
                "--no-cache-dir",
                "pip",
                "setuptools",
                "wheel",
            ],
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
    """Run pip install with real-time output and live spinner for builds."""
    print(f"\n{SYM['package']} {message}")
    print("-" * 50)

    cmd = [str(pip), "install", "--no-cache-dir"] + args

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        packages_collected = 0
        current_package = ""
        spinner: LiveSpinner | None = None

        def stop_spinner(success: bool = True) -> None:
            nonlocal spinner
            if spinner and spinner.running:
                spinner.stop(success)
                spinner = None

        for line in process.stdout:
            line = line.strip()
            logger.debug(line)

            # Skip empty lines
            if not line:
                continue

            # Package collection - show package name with count
            if line.startswith("Collecting "):
                stop_spinner(True)
                packages_collected += 1
                current_package = (
                    line.split()[1]
                    .split("[")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .split("=")[0]
                )
                print(f"  {Colors.CYAN}>{Colors.RESET} [{packages_collected}] {line}")

            # Downloading with size info
            elif "Downloading" in line and (
                "MB" in line or "kB" in line or "KB" in line or "GB" in line
            ):
                print(f"  {Colors.DIM}    {line}{Colors.RESET}")

            # Building/compiling - start live spinner
            elif any(
                kw in line
                for kw in [
                    "Building wheel",
                    "building wheel",
                    "Preparing metadata",
                    "Running setup.py",
                    "running build",
                ]
            ):
                if spinner is None or not spinner.running:
                    pkg = current_package or "package"
                    spinner = LiveSpinner(f"Building {pkg}")
                    spinner.start()

            # Backend dependencies - start spinner if not already running
            elif "backend dependencies" in line.lower():
                if "still running" in line.lower():
                    # Don't print, spinner is already showing progress
                    pass
                elif spinner is None or not spinner.running:
                    pkg = current_package or "package"
                    spinner = LiveSpinner(f"Building {pkg}")
                    spinner.start()

            # Installation success
            elif "Successfully installed" in line:
                stop_spinner(True)
                # Count installed packages
                parts = line.replace("Successfully installed ", "").split()
                count = len(parts)
                print(
                    f"  {Colors.GREEN}{SYM['check']}{Colors.RESET} {count} paket kuruldu"
                )

            # Using cached
            elif "Using cached" in line:
                pass  # Skip - too verbose

            # Requirement already satisfied
            elif "Requirement already satisfied" in line:
                pass  # Skip - too verbose

            # Error messages
            elif "error" in line.lower() and "warning" not in line.lower():
                stop_spinner(False)
                print(f"  {Colors.RED}{SYM['cross']}{Colors.RESET} {line}")

        # Ensure spinner is stopped
        stop_spinner(True)

        process.wait()
        print("-" * 50)

        if process.returncode == 0:
            logger.info("pip install successful")
            return True
        else:
            logger.error(f"pip install failed with code {process.returncode}")
            return False

    except Exception as e:
        if spinner and spinner.running:
            spinner.stop(False)
        print_error(str(e))
        logger.error(f"pip install error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════


def run_with_retry(
    func: Callable[[], bool],
    max_retries: int = MAX_RETRIES,
    delay: int = RETRY_DELAY,
    lang: str = "en",
    logger: logging.Logger | None = None,
) -> bool:
    """Run function with retry on failure.

    Args:
        func: Function to run (should return bool)
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        lang: Language code for messages
        logger: Logger instance

    Returns:
        True if successful, False if all retries failed
    """
    msg = MESSAGES[lang]

    for attempt in range(1, max_retries + 1):
        try:
            if func():
                return True
        except Exception as e:
            if logger:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")

        if attempt < max_retries:
            print(
                f"  {Colors.YELLOW}{SYM['warning']} {msg['retry'].format(attempt, max_retries)}{Colors.RESET}"
            )
            time.sleep(delay)

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

    # Use retry mechanism for PyTorch installation
    return run_with_retry(
        lambda: run_pip_install(pip, args, message, logger),
        max_retries=MAX_RETRIES,
        delay=RETRY_DELAY,
        lang=lang,
        logger=logger,
    )


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

    # Use retry mechanism for Visagen installation
    return run_with_retry(
        lambda: run_pip_install(pip, args, message, logger),
        max_retries=MAX_RETRIES,
        delay=RETRY_DELAY,
        lang=lang,
        logger=logger,
    )


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
    """Main installation flow with progress tracking."""
    # Parse command line arguments
    args = parse_args()

    logger = setup_logging()
    logger.info("Installation started")
    logger.info(
        f"Args: non_interactive={args.non_interactive}, cuda={args.cuda}, profile={args.profile}, lang={args.lang}"
    )

    # Initialize summary tracker
    summary = InstallSummary()
    start_time = time.time()

    # Initialize progress tracker (6 steps total)
    progress: InstallProgress | None = None

    try:
        # 1. Language selection
        if args.non_interactive:
            lang = args.lang
            print(
                f"\n{Colors.CYAN}{SYM['info']}{Colors.RESET} {MESSAGES[lang]['non_interactive']}"
            )
        else:
            lang = select_language()

        msg = MESSAGES[lang]

        # 2. Print banner
        print_banner(lang)

        # Initialize progress tracker after language selection
        progress = InstallProgress(total_steps=6, lang=lang)

        # 3. System checks
        progress.next_step(msg["step_system"], estimate_seconds=5)

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

        # GPU detection
        gpu_name, gpu_memory = detect_gpu_info(logger)
        if gpu_name:
            print_info(f"{msg['gpu_name']}: {gpu_name} ({gpu_memory} GB)")
            summary.gpu_name = gpu_name
            summary.gpu_memory_gb = gpu_memory

            # Show recommendations
            settings = get_recommended_settings(gpu_memory)
            print_info(
                f"{msg['recommended_settings']}: {settings['resolution']}px, batch={settings['batch_size']}"
            )

        missing_deps = check_system_dependencies(logger)
        if missing_deps:
            summary.warnings.extend([f"Missing: {dep}" for dep in missing_deps])

        progress.step_done(msg["step_system"])

        # 4. CUDA selection
        progress.next_step(msg["step_cuda"])

        if args.non_interactive:
            if args.cuda:
                cuda = args.cuda
            elif detected_cuda:
                cuda = detected_cuda
            else:
                cuda = "cpu"
            print_info(f"{msg['using_defaults']}: {cuda}")
        else:
            cuda = select_cuda(lang, detected_cuda)

        cuda_label = f"CUDA {cuda}" if cuda != "cpu" else "CPU"
        summary.cuda_version = cuda
        progress.step_done(cuda_label)

        # 5. Profile selection
        progress.next_step(msg["step_profile"])

        if args.non_interactive:
            profile = args.profile
            print_info(f"{msg['using_defaults']}: {profile}")
        else:
            profile = select_profile(lang)

        summary.profile = profile
        progress.step_done(profile)

        # 6. Create venv
        progress.next_step(msg["step_venv"], estimate_seconds=STEP_ESTIMATES["venv"])

        venv_path = Path(".venv")

        if not args.no_venv:
            if args.non_interactive and venv_path.exists():
                # In non-interactive mode, reuse existing venv
                print_info("Using existing .venv")
            elif not create_venv(venv_path, lang, logger):
                return 1

            if not upgrade_pip(venv_path, lang, logger):
                return 1
        else:
            print_info("Skipping venv creation (--no-venv)")
            venv_path = Path(sys.prefix)

        progress.step_done(msg["step_venv"])

        # 7. Install packages
        total_time = STEP_ESTIMATES["pytorch"] + STEP_ESTIMATES["visagen"]
        progress.next_step(msg["step_packages"], estimate_seconds=total_time)

        if not install_pytorch(venv_path, cuda, lang, logger):
            return 1

        if not install_visagen(venv_path, profile, lang, logger):
            return 1

        progress.step_done(msg["step_packages"])

        # 8. Verification
        progress.next_step(
            msg["step_verify"], estimate_seconds=STEP_ESTIMATES["verify"]
        )

        if not verify_installation(venv_path, lang, logger):
            return 1

        # Get torch version for summary
        python = get_python_path(venv_path)
        try:
            result = subprocess.run(
                [str(python), "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                summary.torch_version = result.stdout.strip()

            result = subprocess.run(
                [str(python), "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            summary.cuda_available = result.stdout.strip() == "True"
        except Exception:
            pass

        progress.step_done(msg["step_verify"])

        # 9. Post-install
        if missing_deps:
            show_system_deps_warning(missing_deps, lang)

        # Calculate total time
        summary.total_time = time.time() - start_time

        # Show summary
        summary.show(lang)

        # Show next steps
        print(f"\n{msg['activate_hint']}")
        if sys.platform == "win32":
            print(f"  {Colors.CYAN}.venv\\Scripts\\activate{Colors.RESET}")
        else:
            print(f"  {Colors.CYAN}source .venv/bin/activate{Colors.RESET}")

        print(f"\n{msg['train_hint']}")
        print(f"  {Colors.CYAN}visagen-train --help{Colors.RESET}")

        print(f"\n{msg['test_hint']}")
        print(f"  {Colors.CYAN}pytest visagen/tests/ -v{Colors.RESET}")

        print(f"\n{'=' * 60}\n")

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
