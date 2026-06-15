from __future__ import annotations

from typing import Any

from scholar_mind.rag.top_k import IDEA_EVIDENCE_TOP_K
from scholar_mind.utils.text import top_keywords, truncate

NO_DIRECT_EVIDENCE = "当前索引语料中未发现直接证据"


def build_evidence_cards(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk["paper_id"], []).append(chunk)

    cards: list[dict[str, Any]] = []
    for paper_id, paper_chunks in grouped.items():
        ordered = sorted(paper_chunks, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        text = " ".join(chunk.get("content", "") for chunk in ordered[:2]).strip()
        matched_aspects = [
            keyword
            for keyword in top_keywords(query, limit=6)
            if keyword in text.lower()
        ] or top_keywords(text, limit=3)
        cards.append(
            {
                "paper_id": paper_id,
                "title": ordered[0].get("title", paper_id),
                "text": text,
                "claim": truncate(text, 180),
                "matched_aspects": matched_aspects[:3],
                "evidence_sections": [
                    {
                        "section": chunk.get("section", "unknown"),
                        "snippet": truncate(chunk.get("content", ""), 140),
                    }
                    for chunk in ordered[:2]
                ],
                "possible_gap": _possible_gap(query, text),
                "year": _extract_year(ordered[0]),
            }
        )
    cards.sort(key=lambda item: len(item.get("matched_aspects", [])), reverse=True)
    return cards


def build_novelty_payload(
    query: str,
    evidence_cards: list[dict[str, Any]],
    evidence_top_k: int = IDEA_EVIDENCE_TOP_K,
) -> dict[str, Any]:
    query_keywords = top_keywords(query, limit=6)
    covered = {aspect for card in evidence_cards for aspect in card.get("matched_aspects", [])}
    overlapping_papers = [
        {
            "paper_id": card["paper_id"],
            "title": card["title"],
            "overlap_aspects": card.get("matched_aspects", [])[:3],
            "evidence": card.get("evidence_sections", [])[:2],
        }
        for card in evidence_cards[:evidence_top_k]
    ]
    differences = [
        {
            "aspect": card.get("possible_gap") or "应用边界",
            "description": (
                f"{card['title']} 主要覆盖 {', '.join(card.get('matched_aspects', [])[:2]) or '相邻问题'}，"
                "与用户构想仍存在场景或组合方式差异。"
            ),
        }
        for card in evidence_cards[:3]
    ]
    uncovered = [keyword for keyword in query_keywords if keyword not in covered]
    unexplored_aspects = [
        {"aspect": keyword, "reason": NO_DIRECT_EVIDENCE}
        for keyword in uncovered[:3]
    ]
    if not unexplored_aspects and not overlapping_papers:
        unexplored_aspects = [{"aspect": query, "reason": NO_DIRECT_EVIDENCE}]

    overlap_count = len(overlapping_papers)
    uncovered_count = len(unexplored_aspects)
    if overlap_count >= 4 and uncovered_count <= 1:
        overall_judgement = "high_overlap"
    elif overlap_count == 0:
        overall_judgement = "no_direct_evidence"
    else:
        overall_judgement = "partial_overlap"

    summary = _summary_text(query, overlapping_papers, unexplored_aspects, overall_judgement)
    references = [
        {
            "paper_id": card["paper_id"],
            "title": card["title"],
            "year": card.get("year"),
        }
        for card in evidence_cards[:evidence_top_k]
    ]
    return {
        "idea_summary": truncate(query, 180),
        "overlapping_papers": overlapping_papers,
        "differences": differences,
        "unexplored_aspects": unexplored_aspects,
        "novelty_report": {
            "summary": summary,
            "overall_judgement": overall_judgement,
        },
        "references": references,
    }


def _possible_gap(query: str, text: str) -> str:
    query_keywords = top_keywords(query, limit=5)
    content_keywords = set(top_keywords(text, limit=5))
    for keyword in query_keywords:
        if keyword not in content_keywords:
            return keyword
    return "组合方式"


def _summary_text(
    query: str,
    overlapping_papers: list[dict[str, Any]],
    unexplored_aspects: list[dict[str, str]],
    overall_judgement: str,
) -> str:
    if overall_judgement == "high_overlap":
        return (
            f"当前语料显示该构想与 {len(overlapping_papers)} 篇相关工作存在较强重叠，"
            "更适合作为已知路线的定向变体。"
        )
    if overall_judgement == "no_direct_evidence":
        return f"围绕“{truncate(query, 60)}”在当前索引语料中未发现直接对齐工作。"
    return (
        f"当前语料中存在 {len(overlapping_papers)} 篇部分重叠工作，"
        f"但仍有 {len(unexplored_aspects)} 个方面仅在当前索引语料中缺少直接证据。"
    )


def _extract_year(chunk: dict[str, Any]) -> int | None:
    publish_date = chunk.get("publish_date")
    if hasattr(publish_date, "year"):
        return int(publish_date.year)
    if isinstance(publish_date, str) and len(publish_date) >= 4 and publish_date[:4].isdigit():
        return int(publish_date[:4])
    return None
