from __future__ import annotations

import re

from scholar_mind.utils.text import top_keywords


class QueryTransformer:
    def normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    def expand_for_idea_novelty(self, idea: str, limit: int = 4) -> list[str]:
        normalized = self.normalize(idea)
        keywords = top_keywords(normalized, limit=max(limit + 2, 6))
        expansions = [normalized]
        if keywords:
            expansions.append(" ".join(keywords[:3]))
        if len(keywords) >= 4:
            expansions.append(" ".join(keywords[1:4]))
        if len(keywords) >= 5:
            expansions.append(f"{keywords[0]} {keywords[3]} {keywords[4]}")
        deduped: list[str] = []
        for candidate in expansions:
            candidate = candidate.strip()
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped[:limit] or [normalized]

    def decompose(self, query: str, limit: int = 4) -> list[str]:
        normalized = self.normalize(query)
        pieces = re.split(r"\b(?:and|vs\.?|versus|compare|with|以及|并且)\b|[?;；。]", normalized)
        parts = [piece.strip(" ,") for piece in pieces if piece.strip(" ,")]
        return parts[:limit] or [normalized]

    def hyde_passage(self, query: str) -> str:
        normalized = self.normalize(query)
        keywords = ", ".join(top_keywords(normalized, limit=4))
        return (
            "A relevant scientific passage would likely explain "
            f"{normalized}. Key concepts include {keywords or normalized}, "
            "with evidence grounded in methods, experiments, and cited findings."
        )
