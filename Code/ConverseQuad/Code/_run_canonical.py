"""Helper for bundle-level ``Code/*.py`` compatibility wrappers."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run(script_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    code_dir = root / "Code mode 2" / "Code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    runpy.run_path(str(code_dir / script_name), run_name="__main__")
