from __future__ import annotations

from pathlib import Path

import httpx

from scholar_mind.models.domain import RetrievedChunk
from scholar_mind.utils.text import overlap_score

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional at runtime
    CrossEncoder = None


class LexicalReranker:
    """Small local reranker used in the offline MVP path."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        reranked = sorted(
            candidates,
            key=lambda item: (overlap_score(query, item.content) * 0.7) + (item.score * 0.3),
            reverse=True,
        )
        return reranked[:top_k]


class SentenceTransformerReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @property
    def is_ready(self) -> bool:
        return self._ensure_model() is not None

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        model = self._ensure_model()
        if model is None:
            return LexicalReranker().rerank(query, candidates, top_k=top_k)
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [
            candidate.model_copy(update={"score": float(score)})
            for score, candidate in ranked[:top_k]
        ]

    @staticmethod
    def _load_model(model_name: str):
        if CrossEncoder is None:
            return None
        try:
            model_path = Path(model_name)
            target = str(model_path) if model_path.exists() else model_name
            return CrossEncoder(target)
        except Exception:
            return None

    def _ensure_model(self):
        if self.model is None:
            self.model = self._load_model(self.model_name)
        return self.model


class RemoteReranker:
    def __init__(
        self,
        model_name: str,
        base_url: str | None,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.model_name = model_name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.fallback = LexicalReranker()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if not self.base_url:
            return self.fallback.rerank(query, candidates, top_k=top_k)
        payload = {
            "pairs": [
                {
                    "query": query,
                    "passage": candidate.content,
                }
                for candidate in candidates
            ],
        }
        if self.model_name:
            payload["model"] = self.model_name
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                self._endpoint(),
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return self._rank_candidates(candidates, response.json(), top_k=top_k)
        except Exception:
            return self.fallback.rerank(query, candidates, top_k=top_k)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/rerank"):
            return self.base_url
        return f"{self.base_url}/rerank"

    def _rank_candidates(
        self,
        candidates: list[RetrievedChunk],
        payload: dict,
        top_k: int,
    ) -> list[RetrievedChunk]:
        score_map = self._score_map(payload)
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                score_map.get(item[0], float("-inf")),
                item[1].score,
            ),
            reverse=True,
        )
        return [
            candidate.model_copy(update={"score": score_map.get(index, candidate.score)})
            for index, candidate in ranked[:top_k]
        ]

    @staticmethod
    def _score_map(payload: dict) -> dict[int, float]:
        rows = payload.get("results")
        if isinstance(rows, list):
            ordered_scores = [
                float(row["score"])
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("score"), (int, float))
            ]
            if ordered_scores:
                return {index: score for index, score in enumerate(ordered_scores)}

        scores = payload.get("scores")
        if isinstance(scores, list):
            return {
                index: float(score)
                for index, score in enumerate(scores)
                if isinstance(score, (int, float))
            }
        for key in ("results", "data"):
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            score_map: dict[int, float] = {}
            for offset, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                score = row.get("score", row.get("relevance_score"))
                if not isinstance(score, (int, float)):
                    continue
                index = row.get("index", offset)
                if isinstance(index, int):
                    score_map[index] = float(score)
            if score_map:
                return score_map
        raise ValueError("Unsupported reranker response payload")
