#!/usr/bin/env python3
"""One-shot wrapper for building the LOCOMO v2 test set end-to-end.

Usage:
    PYTHONPATH=src uv run python scripts/build_locomo_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scholar_mind.eval.locomo_build.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
