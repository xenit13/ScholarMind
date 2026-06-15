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


def build_paper_tools(paper_repository):
    @tool
    async def paper_search(
        query: str,
        runtime: ToolRuntime,
        categories: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 10,
    ) -> Command:
        """Search papers by title and abstract."""
        papers, total = paper_repository.search_papers(
            query=query,
            categories=categories or [],
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
            sort_by=sort_by,
            page=page,
            page_size=page_size,
        )
        return _tool_command("paper_search", {"papers": papers, "total": total}, runtime)

    @tool
    async def paper_get(paper_id: str, runtime: ToolRuntime) -> Command:
        """Get a full structured paper by paper id."""
        paper = paper_repository.get_paper(paper_id)
        return _tool_command(
            "paper_get",
            paper.model_dump(mode="json") if paper else {},
            runtime,
        )

    @tool
    async def paper_read_section(paper_id: str, section: str, runtime: ToolRuntime) -> Command:
        """Read one section from a paper."""
        paper = paper_repository.get_paper(paper_id)
        if paper is None:
            return _tool_command("paper_read_section", {}, runtime)
        for paper_section in paper.sections:
            if paper_section.title.lower() == section.lower():
                return _tool_command(
                    "paper_read_section",
                    paper_section.model_dump(mode="json"),
                    runtime,
                )
        return _tool_command("paper_read_section", {}, runtime)

    @tool
    async def paper_outline(paper_id: str, runtime: ToolRuntime) -> Command:
        """Get a paper outline with paragraph counts."""
        payload = paper_repository.paper_outline(paper_id)
        return _tool_command(
            "paper_outline",
            payload,
            runtime,
        )

    @tool
    async def paper_read_passage(
        paper_id: str,
        section: str,
        paragraph_index: int,
        runtime: ToolRuntime,
        window: int = 1,
    ) -> Command:
        """Read one paragraph passage from a paper section."""
        payload = paper_repository.paper_read_passage(
            paper_id=paper_id,
            section=section,
            paragraph_index=paragraph_index,
            window=window,
        )
        return _tool_command("paper_read_passage", payload or {}, runtime)

    @tool
    async def paper_section_assets(
        paper_id: str,
        section: str,
        runtime: ToolRuntime,
        chunk_types: list[str] | None = None,
    ) -> Command:
        """Read structured assets such as formula, algorithm, table, or figure descriptions."""
        payload = paper_repository.paper_section_assets(
            paper_id=paper_id,
            section=section,
            chunk_types=chunk_types or [],
        )
        return _tool_command(
            "paper_section_assets",
            payload,
            runtime,
        )

    @tool
    async def citation_lookup(paper_ids: list[str], runtime: ToolRuntime) -> Command:
        """Look up precise citation metadata for a list of papers."""
        payload = _citation_payload(paper_repository, paper_ids)
        return _tool_command("citation_lookup", payload, runtime)

    @tool
    async def related_papers(paper_id: str, runtime: ToolRuntime, limit: int = 5) -> Command:
        """Retrieve related papers for a source paper."""
        payload = paper_repository.related_papers(paper_id, limit=limit)
        return _tool_command(
            "related_papers",
            payload,
            runtime,
        )

    @tool
    async def paper_methodology_lookup(paper_id: str, runtime: ToolRuntime) -> Command:
        """Read the full methodology details of a paper."""
        payload = paper_repository.paper_methodology_details(paper_id)
        return _tool_command(
            "paper_methodology_lookup",
            payload,
            runtime,
        )

    return {
        "paper_search": paper_search,
        "paper_get": paper_get,
        "paper_outline": paper_outline,
        "paper_read_section": paper_read_section,
        "paper_read_passage": paper_read_passage,
        "paper_section_assets": paper_section_assets,
        "citation_lookup": citation_lookup,
        "related_papers": related_papers,
        "paper_methodology_lookup": paper_methodology_lookup,
    }


def _parse_date(value: str | None):
    if not value:
        return None
    from datetime import date

    return date.fromisoformat(value)


def _citation_payload(paper_repository, paper_ids: list[str]) -> list[dict]:
    payload = []
    for paper_id in paper_ids:
        paper = paper_repository.get_paper(paper_id)
        if paper is None:
            continue
        author_text = ", ".join(paper.authors[:4])
        if len(paper.authors) > 4:
            author_text = f"{author_text}, et al."
        payload.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.publish_date.year,
                "publish_date": paper.publish_date.isoformat(),
                "categories": list(paper.categories),
                "citation_count": paper.citation_count,
                "formatted_reference": (
                    f"{author_text} ({paper.publish_date.year}). "
                    f"{paper.title}. {paper.paper_id}."
                ),
                "sources": [
                    {
                        "kind": "citation_lookup",
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                    }
                ],
            }
        )
    return payload
