from __future__ import annotations

import runpy
from pathlib import Path


def test_validate_rag_e2e_script_uses_current_embedding_service_contract():
    script_path = Path("scripts/validate_rag_e2e.py")

    runpy.run_path(str(script_path), run_name="scholarmind_validate_rag_e2e")

    source = script_path.read_text(encoding="utf-8")
    assert "ResilientEmbeddingService" not in source
