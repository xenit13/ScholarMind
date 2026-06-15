from __future__ import annotations
import json

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def _tool_command(name: str, payload, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                    name=name,
                )
            ]
        }
    )


def build_analytics_tools(paper_repository):
    @tool
    async def paper_count_stats(
        runtime: ToolRuntime,
        topic: str = "",
        categories: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        granularity: str = "quarterly",
    ) -> Command:
        """Count papers over time for a topic."""
        payload = paper_repository.paper_count_stats(
            topic=topic,
            categories=categories or [],
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
            granularity=granularity,
        )
        return _tool_command(
            "paper_count_stats",
            payload,
            runtime,
        )

    @tool
    async def keyword_trend_stats(
        runtime: ToolRuntime,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        granularity: str = "quarterly",
    ) -> Command:
        """Track keyword frequency over time."""
        payload = paper_repository.keyword_trend_stats(
            keywords=keywords,
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
            granularity=granularity,
        )
        return _tool_command(
            "keyword_trend_stats",
            payload,
            runtime,
        )

    return {
        "paper_count_stats": paper_count_stats,
        "keyword_trend_stats": keyword_trend_stats,
    }


def _parse_date(value: str | None):
    if not value:
        return None
    from datetime import date

    return date.fromisoformat(value)
