"""Run wemake-python-styleguide (flake8) against library source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_wemake_styleguide() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flake8", str(SRC)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "wemake-python-styleguide reported violations:\n"
        f"{completed.stdout}{completed.stderr}"
    )
