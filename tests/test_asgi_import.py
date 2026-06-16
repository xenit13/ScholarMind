from __future__ import annotations

import importlib


def test_asgi_import_does_not_require_optional_ragas_metric_stack():
    module = importlib.import_module("scholar_mind.asgi")

    assert module.app is not None
