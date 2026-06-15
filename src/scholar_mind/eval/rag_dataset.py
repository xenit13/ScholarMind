"""JSONL dataset loader for fixed RAG evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from scholar_mind.models.rag_eval_models import RagEvalCase


class RagEvalDatasetValidationError(ValueError):
    pass


class RagEvalDatasetLoader:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def load_cases(self, *, limit: int | None = None) -> list[RagEvalCase]:
        if not self.dataset_path.exists():
            raise RagEvalDatasetValidationError(
                f"RAG eval dataset file not found: {self.dataset_path}"
            )
        cases: list[RagEvalCase] = []
        seen: set[str] = set()
        for line_number, raw_line in enumerate(
            self.dataset_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                case = RagEvalCase.model_validate(payload)
            except Exception as exc:
                raise RagEvalDatasetValidationError(
                    f"Invalid RAG eval case at line {line_number}: {exc}"
                ) from exc
            if case.case_id in seen:
                raise RagEvalDatasetValidationError(f"duplicate case_id: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
        if not cases:
            raise RagEvalDatasetValidationError(f"No RAG eval cases found in {self.dataset_path}")
        return cases
