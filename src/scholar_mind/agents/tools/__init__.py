from __future__ import annotations

from scholar_mind.agents.tools.analytics import build_analytics_tools
from scholar_mind.agents.tools.papers import build_paper_tools
from scholar_mind.agents.tools.retrieval import build_retrieval_tools

AGENT_TOOLSETS = {
    "researcher": ["rag_retrieve", "paper_search", "paper_get", "related_papers"],
    "trend": ["paper_search", "paper_count_stats", "keyword_trend_stats"],
    "writer": ["citation_lookup"],
    "reviewer": ["paper_get", "paper_read_section"],
    "crossdomain": [
        "rag_top10_similar_papers",
    ],
    "hypothesis": [
        "paper_methodology_lookup",
    ],
    "paper_reader": [
        "paper_get",
        "paper_outline",
        "paper_read_section",
        "paper_read_passage",
        "paper_section_assets",
    ],
}


def build_tool_registry(*, paper_repository, rag_engine):
    registry = {}
    registry.update(build_retrieval_tools(rag_engine))
    registry.update(build_paper_tools(paper_repository))
    registry.update(build_analytics_tools(paper_repository))
    return registry


def get_tools_for(agent_name: str, registry: dict[str, object]) -> list[object]:
    return [registry[name] for name in AGENT_TOOLSETS.get(agent_name, [])]
