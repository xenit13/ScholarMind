from __future__ import annotations

FINAL_CITATION_TOP_K = 4
IDEA_EVIDENCE_TOP_K = 10
CROSS_DOMAIN_CANDIDATE_TOP_K = 10
HYBRID_CANDIDATE_MULTIPLIER = 4
RAG_EVAL_CONTEXT_TOP_K = 5


def hybrid_candidate_limit(final_top_k: int) -> int:
    """Candidate pool size for hybrid-style retrieval."""
    return max(int(final_top_k), 0) * HYBRID_CANDIDATE_MULTIPLIER
