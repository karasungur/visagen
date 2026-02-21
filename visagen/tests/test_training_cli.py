"""Tests for training CLI arguments."""

import subprocess
import sys


def _run_train_help() -> subprocess.CompletedProcess[str]:
    """Run `visagen.tools.train --help` with current interpreter."""
    return subprocess.run(
        [sys.executable, "-m", "visagen.tools.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_accepts_id_weight():
    """Verify --id-weight argument is accepted."""
    result = _run_train_help()
    assert "--id-weight" in result.stdout


def test_cli_accepts_temporal_args():
    """Verify temporal arguments are accepted."""
    result = _run_train_help()
    assert "--temporal-power" in result.stdout
    assert "--temporal-enabled" in result.stdout
    assert "--temporal-consistency-weight" in result.stdout
    assert "--temporal-sequence-length" in result.stdout


def test_cli_accepts_color_transfer_mode():
    """Verify --color-transfer-mode argument is accepted."""
    result = _run_train_help()
    assert "--color-transfer-mode" in result.stdout


def test_cli_shows_color_transfer_choices():
    """Verify color transfer mode shows available choices."""
    result = _run_train_help()
    # Check that at least some choices are shown
    assert "rct" in result.stdout or "{rct,lct,sot,mkl,idt,mix}" in result.stdout


def test_cli_accepts_data_backend():
    """Verify --data-backend argument is accepted."""
    result = _run_train_help()
    assert "--data-backend" in result.stdout
    assert "{auto,dali,pytorch}" in result.stdout


def test_cli_default_precision_is_16_mixed():
    """Help output should advertise the modern mixed precision default."""
    result = _run_train_help()
    assert "default: 16-mixed" in result.stdout
