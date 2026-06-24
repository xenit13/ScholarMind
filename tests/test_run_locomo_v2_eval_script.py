from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_locomo_v2_eval.py"
    spec = importlib.util.spec_from_file_location("run_locomo_v2_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_locomo_v2_eval_requires_api_url():
    parser = _load_script_module().build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    assert exc_info.value.code == 2


def test_run_locomo_v2_eval_accepts_api_url():
    parser = _load_script_module().build_parser()

    args = parser.parse_args(["--api-url", "http://127.0.0.1:8000"])

    assert args.api_url == "http://127.0.0.1:8000"
