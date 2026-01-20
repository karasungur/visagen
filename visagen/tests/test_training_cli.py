"""Tests for training CLI arguments."""

import subprocess


def test_cli_accepts_id_weight():
    """Verify --id-weight argument is accepted."""
    result = subprocess.run(
        ["python", "-m", "visagen.tools.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--id-weight" in result.stdout


def test_cli_accepts_temporal_args():
    """Verify temporal arguments are accepted."""
    result = subprocess.run(
        ["python", "-m", "visagen.tools.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--temporal-power" in result.stdout
    assert "--temporal-enabled" in result.stdout
    assert "--temporal-consistency-weight" in result.stdout
    assert "--temporal-sequence-length" in result.stdout


def test_cli_accepts_color_transfer_mode():
    """Verify --color-transfer-mode argument is accepted."""
    result = subprocess.run(
        ["python", "-m", "visagen.tools.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--color-transfer-mode" in result.stdout


def test_cli_shows_color_transfer_choices():
    """Verify color transfer mode shows available choices."""
    result = subprocess.run(
        ["python", "-m", "visagen.tools.train", "--help"],
        capture_output=True,
        text=True,
    )
    # Check that at least some choices are shown
    assert "rct" in result.stdout or "{rct,lct,sot,mkl,idt,mix}" in result.stdout
