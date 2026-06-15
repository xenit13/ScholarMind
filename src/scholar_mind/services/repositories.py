from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime
from statistics import quantiles
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, exists, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.db.models import (
    ConversationMetricModel,
    EvalReportModel,
    MemoryEvalRunV2Model,
    MemoryExtractionEventV2Model,
    MemoryMetricModel,
    MemoryRetrievalEventV2Model,
    PaperChunkModel,
    PaperModel,
    PaperSectionModel,
    RagRetrievalEventV2Model,
    RequestRagEvalAnnotationModel,
    RequestRunModel,
    SessionModel,
)
from scholar_mind.eval.answer_quality import compute_answer_quality_score
from scholar_mind.models.domain import (
    PaperChunk,
    PaperSection,
    ReportSummary,
    SessionInfo,
    StructuredPaper,
)
from scholar_mind.pipeline.chunker import StructureAwareChunker
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K
from scholar_mind.utils.text import SparseCorpusStats, overlap_score, tokenize, top_keywords


class PaperRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory
        self.chunker = StructureAwareChunker()

    def _paper_statement(
        self,
        *,
        paper_ids: list[str] | None = None,
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        query: str = "",
    ):
        stmt = select(PaperModel)
        if paper_ids:
            stmt = stmt.where(PaperModel.paper_id.in_(paper_ids))
        if date_from:
            stmt = stmt.where(PaperModel.publish_date >= date_from)
        if date_to:
            stmt = stmt.where(PaperModel.publish_date <= date_to)
        if categories:
            stmt = stmt.where(self._category_clause(categories))
        query_tokens = tokenize(query)
        if query_tokens:
            stmt = stmt.where(
                or_(
                    *[
                        or_(
                            PaperModel.title.ilike(f"%{token}%"),
                            PaperModel.abstract.ilike(f"%{token}%"),
                        )
                        for token in query_tokens[:8]
                    ]
                )
            )
        return stmt.execution_options(yield_per=128)

    @staticmethod
    def _category_clause(categories: list[str]):
        category_values = func.json_each(PaperModel.categories_json).table_valued("value").alias(
            "category_values"
        )
        return exists(
            select(1)
            .select_from(category_values)
            .where(category_values.c.value.in_(categories))
        )

    @staticmethod
    def _paper_preview(paper: PaperModel) -> dict[str, Any]:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": json.loads(paper.authors_json),
            "abstract": paper.abstract,
            "categories": json.loads(paper.categories_json),
            "publish_date": paper.publish_date.isoformat(),
            "citation_count": paper.citation_count,
        }

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
        """Resolve chunk_ids to their content. Returns {chunk_id: content}."""
        if not chunk_ids:
            return {}
        with self.session_factory() as session:
            rows = session.scalars(
                select(PaperChunkModel).where(PaperChunkModel.chunk_id.in_(chunk_ids))
            ).all()
            return {row.chunk_id: row.content for row in rows}

    def list_chunk_models(self) -> list[PaperChunk]:
        return list(self.iter_chunk_models())

    def iter_chunk_models(self, batch_size: int = 128) -> Iterator[PaperChunk]:
        with self.session_factory() as session:
            stmt = select(PaperChunkModel).execution_options(yield_per=batch_size)
            for chunk in session.scalars(stmt):
                yield PaperChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    chunk_type=chunk.chunk_type,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    metadata=json.loads(chunk.metadata_json),
                )

    def paper_payloads(self, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not paper_ids:
            return {}
        with self.session_factory() as session:
            payloads: dict[str, dict[str, Any]] = {}
            stmt = (
                select(PaperModel)
                .where(PaperModel.paper_id.in_(paper_ids))
                .execution_options(yield_per=128)
            )
            for paper in session.scalars(stmt):
                payloads[paper.paper_id] = {
                    "title": paper.title,
                    "categories": json.loads(paper.categories_json),
                    "publish_date": paper.publish_date.isoformat(),
                }
            return payloads

    def list_chunks(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        with self.session_factory() as session:
            paper_ids = set(filters.get("paper_ids") or [])
            categories = set(filters.get("categories") or [])
            date_from = filters.get("date_from")
            date_to = filters.get("date_to")
            stmt = (
                select(
                    PaperChunkModel,
                    PaperModel.title,
                    PaperModel.categories_json,
                    PaperModel.publish_date,
                )
                .join(PaperModel, PaperChunkModel.paper_id == PaperModel.paper_id)
                .execution_options(yield_per=128)
            )
            if paper_ids:
                stmt = stmt.where(PaperChunkModel.paper_id.in_(paper_ids))
            if date_from:
                stmt = stmt.where(PaperModel.publish_date >= date_from)
            if date_to:
                stmt = stmt.where(PaperModel.publish_date <= date_to)
            if categories:
                stmt = stmt.where(self._category_clause(list(categories)))
            results: list[dict[str, Any]] = []
            for row, title, categories_json, publish_date in session.execute(stmt):
                paper_categories = json.loads(categories_json)
                if categories and not categories.intersection(set(paper_categories)):
                    continue
                results.append(
                    {
                        "chunk_id": row.chunk_id,
                        "paper_id": row.paper_id,
                        "title": title,
                        "section": row.section,
                        "content": row.content,
                        "chunk_type": row.chunk_type,
                        "token_count": row.token_count,
                        "categories": paper_categories,
                        "publish_date": publish_date,
                    }
                )
            return results

    def build_sparse_stats(self, batch_size: int = 128) -> SparseCorpusStats:
        document_frequencies: Counter[str] = Counter()
        total_length = 0
        document_count = 0
        with self.session_factory() as session:
            stmt = select(PaperChunkModel.content).execution_options(yield_per=batch_size)
            for content in session.scalars(stmt):
                tokens = tokenize(content)
                total_length += len(tokens)
                document_count += 1
                for token in set(tokens):
                    document_frequencies[token] += 1
        if document_count == 0:
            document_count = 1
        average_length = total_length / document_count if total_length else 1.0
        return SparseCorpusStats(
            document_count=document_count,
            average_length=average_length,
            document_frequencies=dict(document_frequencies),
        )

    def get_paper(self, paper_id: str) -> StructuredPaper | None:
        with self.session_factory() as session:
            paper = session.get(PaperModel, paper_id)
            if paper is None:
                return None
            sections = list(
                session.scalars(
                    select(PaperSectionModel).where(PaperSectionModel.paper_id == paper_id)
                ).all()
            )
            return StructuredPaper(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=json.loads(paper.authors_json),
                abstract=paper.abstract,
                categories=json.loads(paper.categories_json),
                publish_date=paper.publish_date,
                citation_count=paper.citation_count,
                sections=[
                    PaperSection(
                        section_id=section.section_id,
                        title=section.title,
                        content=section.content,
                        level=section.level,
                    )
                    for section in sections
                ],
                references=[],
                metadata={"has_source": paper.has_source},
            )

    def upsert_structured_paper(
        self, paper: StructuredPaper, chunks: list[PaperChunk] | None = None
    ) -> list[PaperChunk]:
        chunk_models = chunks or self.chunker.chunk(paper)
        with self.session_factory() as session:
            row = session.get(PaperModel, paper.paper_id)
            if row is None:
                row = PaperModel(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors_json=json.dumps(paper.authors),
                    abstract=paper.abstract,
                    categories_json=json.dumps(paper.categories),
                    publish_date=paper.publish_date,
                    citation_count=paper.citation_count,
                    has_source=bool(paper.metadata.get("has_source", False)),
                )
                session.add(row)
            else:
                row.title = paper.title
                row.authors_json = json.dumps(paper.authors)
                row.abstract = paper.abstract
                row.categories_json = json.dumps(paper.categories)
                row.publish_date = paper.publish_date
                row.citation_count = paper.citation_count
                row.has_source = bool(paper.metadata.get("has_source", False))

            session.execute(
                delete(PaperSectionModel).where(PaperSectionModel.paper_id == paper.paper_id)
            )
            session.execute(
                delete(PaperChunkModel).where(PaperChunkModel.paper_id == paper.paper_id)
            )

            for section in paper.sections:
                session.add(
                    PaperSectionModel(
                        paper_id=paper.paper_id,
                        section_id=section.section_id,
                        title=section.title,
                        content=section.content,
                        level=section.level,
                    )
                )
            for chunk in chunk_models:
                session.add(
                    PaperChunkModel(
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        chunk_type=chunk.chunk_type.value,
                        section=chunk.section,
                        subsection=chunk.subsection,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        metadata_json=json.dumps(chunk.metadata),
                    )
                )
            session.commit()
        return chunk_models

    def resolve_paper_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for raw_query in queries:
            query = raw_query.strip()
            if not query:
                continue
            direct = self.get_paper(query)
            if direct is not None:
                resolved.append(self._resolved_paper_payload(query, direct, resolved=True))
                continue
            matches, _ = self.search_papers(query=query, page=1, page_size=5)
            best_match = None
            best_score = 0.0
            for match in matches:
                score = overlap_score(
                    query,
                    f"{match['paper_id']} {match['title']} {match['abstract']}",
                )
                if score > best_score:
                    best_score = score
                    best_match = match
            if best_match is None or best_score <= 0:
                resolved.append(
                    {
                        "requested_paper": query,
                        "resolved": False,
                        "paper_id": None,
                        "title": query,
                        "categories": [],
                        "summary": "",
                        "methodology_summary": "",
                        "sources": [
                            {
                                "kind": "planner_request",
                                "label": query,
                                "metadata": {"resolution": "not_found"},
                            }
                        ],
                    }
                )
                continue
            paper = self.get_paper(best_match["paper_id"])
            if paper is None:
                continue
            resolved.append(self._resolved_paper_payload(query, paper, resolved=True))
        return resolved

    def paper_outline(self, paper_id: str) -> list[dict[str, Any]]:
        paper = self.get_paper(paper_id)
        if paper is None:
            return []
        return [
            {
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "paragraph_count": len(self._split_paragraphs(section.content)),
            }
            for section in paper.sections
        ]

    def paper_read_passage(
        self, paper_id: str, section: str, paragraph_index: int, window: int = 1
    ) -> dict[str, Any] | None:
        paper = self.get_paper(paper_id)
        if paper is None:
            return None
        for paper_section in paper.sections:
            if paper_section.title.lower() != section.lower():
                continue
            paragraphs = self._split_paragraphs(paper_section.content)
            if not paragraphs:
                return None
            index = min(max(paragraph_index, 0), len(paragraphs) - 1)
            start = max(0, index - max(window, 0))
            end = min(len(paragraphs), index + max(window, 0) + 1)
            return {
                "section": paper_section.title,
                "paragraph_index": index,
                "text": "\n\n".join(paragraphs[start:end]),
                "paragraphs": paragraphs,
                "section_paragraph_count": len(paragraphs),
            }
        return None

    def paper_methodology_details(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if paper is None:
            return {}
        methodology_sections = [
            {
                "section_id": section.section_id,
                "title": section.title,
                "content": section.content,
            }
            for section in paper.sections
            if self._is_methodology_section(section.title)
        ]
        if not methodology_sections and paper.sections:
            methodology_sections = [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "content": section.content,
                }
                for section in paper.sections[:2]
            ]
        methodology_text = "\n\n".join(
            section["content"] for section in methodology_sections
        ).strip()
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "categories": list(paper.categories),
            "summary": self._paper_summary(paper),
            "methodology_summary": self._paper_methodology_summary(paper),
            "methodology_sections": methodology_sections,
            "sources": [
                {
                    "kind": "paper_repository",
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "section": section["title"],
                }
                for section in methodology_sections
            ],
            "methodology_text": methodology_text,
        }

    def paper_section_assets(
        self,
        paper_id: str,
        section: str,
        chunk_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        requested = {item.lower() for item in (chunk_types or [])}
        with self.session_factory() as session:
            stmt = (
                select(PaperChunkModel)
                .where(PaperChunkModel.paper_id == paper_id)
                .execution_options(yield_per=64)
            )
            assets = []
            for chunk in session.scalars(stmt):
                if chunk.section.lower() != section.lower():
                    continue
                if requested and chunk.chunk_type.lower() not in requested:
                    continue
                assets.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "content": chunk.content,
                        "metadata": json.loads(chunk.metadata_json),
                    }
                )
            return assets

    def search_papers(
        self,
        query: str,
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        categories = categories or []
        start = max(page - 1, 0) * page_size
        end = start + page_size
        with self.session_factory() as session:
            stmt = self._paper_statement(
                categories=categories,
                date_from=date_from,
                date_to=date_to,
                query=query,
            )
            if sort_by == "date":
                stmt = stmt.order_by(PaperModel.publish_date.desc(), PaperModel.paper_id.asc())
            elif sort_by == "citations":
                stmt = stmt.order_by(
                    PaperModel.citation_count.desc().nullslast(), PaperModel.paper_id.asc()
                )

            total = 0
            if sort_by in {"date", "citations"}:
                items: list[dict[str, Any]] = []
                for paper in session.scalars(stmt):
                    score = (
                        overlap_score(query, f"{paper.title} {paper.abstract}") if query else 1.0
                    )
                    if query and score == 0:
                        continue
                    if total >= start and len(items) < page_size:
                        items.append(self._paper_preview(paper))
                    total += 1
                return items, total

            ranked: list[tuple[float, int, dict[str, Any]]] = []
            for paper in session.scalars(stmt):
                score = overlap_score(query, f"{paper.title} {paper.abstract}") if query else 1.0
                if query and score == 0:
                    continue
                preview = self._paper_preview(paper)
                if len(ranked) < end:
                    heapq.heappush(ranked, (score, total, preview))
                elif score > ranked[0][0]:
                    heapq.heapreplace(ranked, (score, total, preview))
                total += 1

            ranked_items = [
                item
                for _, _, item in sorted(
                    ranked, key=lambda value: (value[0], -value[1]), reverse=True
                )
            ]
            return ranked_items[start:end], total

    def related_papers(self, paper_id: str, limit: int = 10) -> list[dict[str, Any]]:
        source = self.get_paper(paper_id)
        if source is None:
            return []
        query = f"{source.title} {source.abstract}"
        candidates, _ = self.search_papers(
            query=query,
            categories=source.categories,
            sort_by="relevance",
            page=1,
            page_size=max(limit * 5, 20),
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        for preview in candidates:
            if preview["paper_id"] == paper_id:
                continue
            category_overlap = len(set(source.categories) & set(preview["categories"]))
            score = category_overlap + overlap_score(
                query,
                f"{preview['title']} {preview['abstract']}",
            )
            if score <= 0:
                continue
            scored.append((score, preview))
        ordered = sorted(scored, key=lambda value: value[0], reverse=True)[:limit]
        return [
            {"paper_id": item["paper_id"], "title": item["title"], "score": round(score, 4)}
            for score, item in ordered
        ]

    def papers_matching_topic(
        self,
        topic: str,
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[StructuredPaper]:
        items, _ = self.search_papers(
            topic,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            sort_by="relevance",
            page=1,
            page_size=1000,
        )
        return [
            paper
            for paper_id in [item["paper_id"] for item in items]
            if (paper := self.get_paper(paper_id))
        ]

    def paper_count_stats(
        self,
        topic: str = "",
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        granularity: str = "quarterly",
    ) -> list[dict[str, Any]]:
        buckets: dict[str, int] = defaultdict(int)
        with self.session_factory() as session:
            for paper in session.scalars(
                self._paper_statement(
                    categories=categories,
                    date_from=date_from,
                    date_to=date_to,
                    query=topic,
                )
            ):
                if topic and overlap_score(topic, f"{paper.title} {paper.abstract}") == 0:
                    continue
                buckets[self._bucket_label(paper.publish_date, granularity)] += 1
        return [{"period": period, "count": buckets[period]} for period in sorted(buckets)]

    def keyword_trend_stats(
        self,
        keywords: list[str],
        date_from: date | None = None,
        date_to: date | None = None,
        granularity: str = "quarterly",
    ) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        periods: set[str] = set()
        with self.session_factory() as session:
            for paper in session.scalars(
                self._paper_statement(
                    date_from=date_from,
                    date_to=date_to,
                    query=" ".join(keywords),
                )
            ):
                corpus = f"{paper.title} {paper.abstract}".lower()
                period = self._bucket_label(paper.publish_date, granularity)
                matched = False
                for keyword in keywords:
                    if keyword.lower() in corpus:
                        counts[(keyword, period)] += 1
                        matched = True
                if matched:
                    periods.add(period)
        ordered_periods = sorted(periods)
        results: list[dict[str, Any]] = []
        for keyword in keywords:
            previous = 0
            for period in ordered_periods:
                current = counts.get((keyword, period), 0)
                growth = ((current - previous) / previous) if previous else float(current > 0)
                results.append(
                    {
                        "keyword": keyword,
                        "period": period,
                        "count": current,
                        "growth_rate": round(growth, 4),
                    }
                )
                previous = current
        return results

    def all_papers(self) -> list[StructuredPaper]:
        with self.session_factory() as session:
            ids = [paper.paper_id for paper in session.scalars(self._paper_statement())]
        return [paper for paper_id in ids if (paper := self.get_paper(paper_id))]

    def cross_domain_candidates(self, paper_id: str, limit: int = 10) -> list[dict[str, Any]]:
        source = self.get_paper(paper_id)
        if source is None:
            return []
        source_category = source.categories[0] if source.categories else ""
        query = f"{source.title} {source.abstract}"
        previews, _ = self.search_papers(
            query=query,
            sort_by="relevance",
            page=1,
            page_size=max(limit * 8, 32),
        )
        candidates: list[tuple[float, dict[str, Any]]] = []
        for preview in previews:
            if preview["paper_id"] == paper_id:
                continue
            preview_categories = preview["categories"]
            if preview_categories and preview_categories[0] == source_category:
                continue
            similarity = overlap_score(query, f"{preview['title']} {preview['abstract']}")
            if similarity <= 0:
                continue
            candidates.append(
                (
                    similarity,
                    {
                        "paper_id": preview["paper_id"],
                        "title": preview["title"],
                        "category": preview_categories[0] if preview_categories else "unknown",
                    },
                )
            )
        ordered = sorted(candidates, key=lambda value: value[0], reverse=True)[:limit]
        return [
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "category": item["category"],
                "similarity": round(score, 4),
            }
            for score, item in ordered
        ]

    def representative_papers(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        papers, _ = self.search_papers(topic, sort_by="citations", page=1, page_size=limit)
        return [
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "citation_count": paper["citation_count"] or 0,
            }
            for paper in papers
        ]

    def top_keywords_for_topic(self, topic: str, limit: int = 5) -> list[str]:
        counts: Counter[str] = Counter()
        with self.session_factory() as session:
            for paper in session.scalars(self._paper_statement(query=topic)):
                if overlap_score(topic, f"{paper.title} {paper.abstract}") == 0:
                    continue
                counts.update(tokenize(f"{paper.title} {paper.abstract}"))
        ranked = [token for token, _ in counts.most_common(limit * 3)]
        return [token for token in ranked if token not in {"the", "and", "for"}][:limit]

    def _paper_in_filters(
        self,
        paper: StructuredPaper,
        categories: list[str],
        date_from: date | None,
        date_to: date | None,
    ) -> bool:
        if categories and not set(categories).intersection(set(paper.categories)):
            return False
        if date_from and paper.publish_date < date_from:
            return False
        if date_to and paper.publish_date > date_to:
            return False
        return True

    @staticmethod
    def _bucket_label(published: date, granularity: str) -> str:
        if granularity == "monthly":
            return f"{published.year}-{published.month:02d}"
        if granularity == "quarterly":
            quarter = ((published.month - 1) // 3) + 1
            return f"{published.year}-Q{quarter}"
        return str(published.year)

    @staticmethod
    def _split_paragraphs(content: str) -> list[str]:
        normalized = content.strip()
        if not normalized:
            return []
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", normalized) if item.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?。！？])\s+", normalized)
            if item.strip()
        ]
        if len(sentences) <= 2:
            return [normalized]
        grouped = []
        for index in range(0, len(sentences), 2):
            grouped.append(" ".join(sentences[index : index + 2]).strip())
        return grouped

    @staticmethod
    def _is_methodology_section(title: str) -> bool:
        lowered = title.lower()
        return lowered in {"method", "methods", "approach", "methodology"} or any(
            token in lowered for token in ["method", "approach"]
        )

    def _paper_summary(self, paper: StructuredPaper) -> str:
        return paper.abstract.strip()[:300]

    def _paper_methodology_summary(self, paper: StructuredPaper) -> str:
        for section in paper.sections:
            if self._is_methodology_section(section.title):
                return section.content.strip()[:400]
        return paper.abstract.strip()[:400]

    def _resolved_paper_payload(
        self, requested_paper: str, paper: StructuredPaper, *, resolved: bool
    ) -> dict[str, Any]:
        return {
            "requested_paper": requested_paper,
            "resolved": resolved,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "categories": list(paper.categories),
            "summary": self._paper_summary(paper),
            "methodology_summary": self._paper_methodology_summary(paper),
            "sources": [
                {
                    "kind": "paper_repository",
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "metadata": {"requested_paper": requested_paper},
                }
            ],
        }


class SessionRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def create_or_get(self, user_id: str, session_id: str) -> SessionInfo:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                row = SessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(UTC),
                    message_count=0,
                    topics_json="[]",
                    papers_json="[]",
                    memory_context_loaded=False,
                    last_state_json="{}",
                )
                session.add(row)
                session.commit()
                session.refresh(row)
            return self._to_model(row)

    def get(self, session_id: str) -> SessionInfo | None:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            return None if row is None else self._to_model(row)

    def get_last_state(self, session_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None or not row.last_state_json:
                return {}
            state = json.loads(row.last_state_json)
            if state.get("messages"):
                from scholar_mind.utils.messages import deserialize_messages

                state["messages"] = deserialize_messages(state["messages"])
            return state

    def update_from_state(
        self, user_id: str, session_id: str, state: dict[str, Any]
    ) -> SessionInfo:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                row = SessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(UTC),
                    topics_json="[]",
                    papers_json="[]",
                )
                session.add(row)

            row.user_id = user_id
            row.message_count = len(state.get("messages", []))
            row.topics_json = json.dumps(top_keywords(state.get("query", ""), limit=4))
            paper_ids = sorted(
                {chunk["paper_id"] for chunk in state.get("retrieved_chunks", [])[:10]}
            )
            row.papers_json = json.dumps(paper_ids)
            row.memory_context_loaded = bool(state.get("memory_context"))
            row.last_state_json = json.dumps(state, default=str)
            session.commit()
            session.refresh(row)
            return self._to_model(row)

    def close(self, session_id: str) -> SessionInfo | None:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                return None
            row.closed_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            return self._to_model(row)

    @staticmethod
    def _to_model(row: SessionModel) -> SessionInfo:
        return SessionInfo(
            session_id=row.session_id,
            user_id=row.user_id,
            created_at=row.created_at,
            closed_at=row.closed_at,
            message_count=row.message_count,
            topics_discussed=json.loads(row.topics_json or "[]"),
            papers_mentioned=json.loads(row.papers_json or "[]"),
            memory_context_loaded=row.memory_context_loaded,
        )


class EvalRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def save_report(
        self, report_id: str, report_type: str, config: dict[str, Any], results: dict[str, Any]
    ) -> None:
        with self.session_factory() as session:
            session.add(
                EvalReportModel(
                    report_id=report_id,
                    type=report_type,
                    created_at=datetime.now(UTC),
                    config_json=json.dumps(config),
                    results_json=json.dumps(results),
                )
            )
            session.commit()

    def get_report(self, report_id: str) -> ReportSummary | None:
        with self.session_factory() as session:
            row = session.get(EvalReportModel, report_id)
            if row is None:
                return None
            return ReportSummary(
                report_id=row.report_id,
                type=row.type,
                created_at=row.created_at,
                config=json.loads(row.config_json),
                results=json.loads(row.results_json),
            )

    def list_reports_by_type(
        self, report_type: str, limit: int = 20
    ) -> list[ReportSummary]:
        """List recent reports of a given type, newest first."""
        with self.session_factory() as session:
            rows = session.scalars(
                select(EvalReportModel)
                .where(EvalReportModel.type == report_type)
                .order_by(EvalReportModel.created_at.desc())
                .limit(limit)
            ).all()
            return [
                ReportSummary(
                    report_id=row.report_id,
                    type=row.type,
                    created_at=row.created_at,
                    config=json.loads(row.config_json),
                    results=json.loads(row.results_json),
                )
                for row in rows
            ]


class MetricsRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def record_round(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        query_type: str,
        success: bool,
        retrieval_latency_ms: int,
        latency_ms: int,
        citations_count: int,
        retrieved_chunks_count: int,
        output_length: int,
        agent_path: list[str],
        error_summary: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self.session_factory() as session:
            session.merge(
                ConversationMetricModel(
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    query_type=query_type,
                    success=success,
                    retrieval_latency_ms=retrieval_latency_ms,
                    latency_ms=latency_ms,
                    citations_count=citations_count,
                    retrieved_chunks_count=retrieved_chunks_count,
                    output_length=output_length,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    agent_path_json=json.dumps(agent_path),
                    error_summary=error_summary,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def record_memory_run(
        self,
        *,
        user_id: str,
        success: bool,
        extracted_count: int,
        latency_ms: int,
        error_summary: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self.session_factory() as session:
            session.add(
                MemoryMetricModel(
                    metric_id=f"mem_metric_{uuid4().hex}",
                    user_id=user_id,
                    success=success,
                    extracted_count=extracted_count,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error_summary=error_summary,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def health_stats(self) -> dict[str, Any]:
        with self.session_factory() as session:
            papers = session.query(PaperModel).count()
            chunks = session.query(PaperChunkModel).count()
            active_sessions = (
                session.query(SessionModel).filter(SessionModel.closed_at.is_(None)).count()
            )
            return {
                "papers_indexed": papers,
                "chunks_indexed": chunks,
                "active_sessions": active_sessions,
            }

    def latency_p95(self, report_type: str) -> float:
        with self.session_factory() as session:
            latencies = [
                row.latency_ms
                for row in session.scalars(
                    select(ConversationMetricModel).where(
                        ConversationMetricModel.query_type == report_type
                    )
                ).all()
            ]
        if len(latencies) < 2:
            return float(latencies[0]) if latencies else 0.0
        return float(quantiles(latencies, n=20)[-1])


# ---------------------------------------------------------------------------
# Request audit repository (Document 23)
# ---------------------------------------------------------------------------


class OnlineEvalRepository:
    """Repository for neutral request audit data and RAG retrieval events."""

    HEALTH_SCORE_PENALTIES = {
        "has_error": 0.45,
        "timeout": 0.25,
        "has_fallback": 0.20,
        "has_retry": 0.10,
    }

    OVERALL_SCORE_WEIGHTS = {
        "answer_quality_score": 0.50,
        "rag_score": 0.30,
        "memory_score": 0.20,
    }

    RAG_METRIC_FIELDS = (
        "rag_score",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "noise_sensitivity",
        "semantic_similarity",
        "redundancy",
        "completeness",
    )

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def save_request_run(self, payload: dict[str, Any]) -> None:
        execution_health = dict(payload.get("execution_health", {}))
        execution_health_score = self._compute_execution_health_score(
            has_error=execution_health.get("has_error", False),
            timeout=execution_health.get("timeout", False),
            has_fallback=payload.get(
                "has_fallback",
                execution_health.get("has_fallback", False),
            ),
            has_retry=payload.get("has_retry", execution_health.get("has_retry", False)),
        )
        execution_health["execution_health_score"] = execution_health_score
        with self.session_factory() as session:
            session.merge(
                RequestRunModel(
                    request_id=payload["request_id"],
                    session_id=payload.get("session_id", ""),
                    user_id=payload.get("user_id", ""),
                    query=payload.get("query", ""),
                    query_type=payload.get("query_type", ""),
                    final_answer=payload.get("final_answer", ""),
                    rag_score=payload.get("rag_score"),
                    faithfulness=payload.get("faithfulness"),
                    answer_relevancy=payload.get("answer_relevancy"),
                    context_precision=payload.get("context_precision"),
                    context_recall=payload.get("context_recall"),
                    noise_sensitivity=payload.get("noise_sensitivity"),
                    semantic_similarity=payload.get("semantic_similarity"),
                    redundancy=payload.get("redundancy"),
                    completeness=payload.get("completeness"),
                    memory_score=payload.get("memory_score"),
                    execution_health_score=execution_health_score,
                    has_retry=payload.get("has_retry", execution_health.get("has_retry", False)),
                    has_fallback=payload.get(
                        "has_fallback",
                        execution_health.get("has_fallback", False),
                    ),
                    execution_health_json=json.dumps(execution_health),
                    runtime_metrics_json=json.dumps(payload.get("runtime_metrics", {})),
                    agent_trace_json=json.dumps(payload.get("agent_trace", [])),
                    agent_events_json=json.dumps(payload.get("agent_events", [])),
                    answer_event_json=json.dumps(payload.get("answer_event", {})),
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def save_rag_retrieval_event(self, event: dict[str, Any]) -> None:
        contexts = event.get("returned_contexts")
        if contexts is None:
            contexts = [
                item.get("content", "")
                for item in event.get("returned_chunks", [])
                if isinstance(item, dict)
            ]
        with self.session_factory() as session:
            session.add(
                RagRetrievalEventV2Model(
                    event_id=event["event_id"],
                    request_id=event.get("request_id", ""),
                    query=event.get("query", ""),
                    normalized_query=event.get("normalized_query"),
                    strategy=event.get("strategy", "hybrid"),
                    top_k=event.get("top_k", FINAL_CITATION_TOP_K),
                    filters_json=json.dumps(event.get("filters", {})),
                    latency_ms=event.get("latency_ms", 0),
                    returned_contexts_json=json.dumps(contexts[:10], ensure_ascii=False),
                    returned_chunk_ids_json=json.dumps(event.get("returned_chunk_ids", [])),
                    returned_paper_ids_json=json.dumps(event.get("returned_paper_ids", [])),
                    rag_score=event.get("rag_score"),
                    faithfulness=event.get("faithfulness"),
                    answer_relevancy=event.get("answer_relevancy"),
                    context_precision=event.get("context_precision"),
                    context_recall=event.get("context_recall"),
                    noise_sensitivity=event.get("noise_sensitivity"),
                    semantic_similarity=event.get("semantic_similarity"),
                    redundancy=event.get("redundancy"),
                    completeness=event.get("completeness"),
                    caller_agent=event.get("caller_agent"),
                    tool_name=event.get("tool_name", "rag_retrieve"),
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def get_request_rag_eval_annotation(self, request_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.get(RequestRagEvalAnnotationModel, request_id)
            if row is None:
                return None
            return {
                "request_id": row.request_id,
                "reference": row.reference,
                "required_points": json.loads(row.required_points_json or "[]"),
            }

    def get_online_rag_eval_source(self, request_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            request = session.get(RequestRunModel, request_id)
            if request is None:
                return None
            event = self._latest_rag_event(session, request_id)
            contexts = json.loads(event.returned_contexts_json or "[]") if event else []
            chunk_ids = json.loads(event.returned_chunk_ids_json or "[]") if event else []
            return {
                "request_id": request.request_id,
                "user_input": request.query,
                "response": request.final_answer,
                "retrieved_contexts": contexts,
                "retrieved_chunk_ids": chunk_ids,
                "retrieval_latency_ms": event.latency_ms if event else 0,
                "strategy": event.strategy if event else "",
                "event_id": event.event_id if event else None,
                "metrics_complete": self._rag_metrics_are_complete(request),
            }

    def save_online_rag_metrics(self, request_id: str, metrics: dict[str, Any]) -> None:
        with self.session_factory() as session:
            request = session.get(RequestRunModel, request_id)
            if request is not None:
                for field_name in self.RAG_METRIC_FIELDS:
                    if field_name in metrics:
                        setattr(request, field_name, metrics[field_name])
                request.rag_eval_status = "scored"
                request.rag_scored_at = datetime.now(UTC)

            event = (
                session.get(RagRetrievalEventV2Model, metrics.get("event_id"))
                if metrics.get("event_id")
                else self._latest_rag_event(session, request_id)
            )
            if event is not None:
                for field_name in self.RAG_METRIC_FIELDS:
                    if field_name in metrics:
                        setattr(event, field_name, metrics[field_name])
            session.commit()

    def _rag_metrics_are_complete(self, request: RequestRunModel) -> bool:
        if request.rag_eval_status == "scored":
            return True
        if request.rag_eval_status in {"pending", "failed"}:
            return False
        return any(
            getattr(request, field_name) is not None
            for field_name in self.RAG_METRIC_FIELDS
        )

    def get_request_eval(self, request_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.get(RequestRunModel, request_id)
            if row is None:
                return None
            return self._request_to_dict(row, self._latest_rag_event(session, request_id))

    def get_request_diagnosis(self, request_id: str) -> dict[str, Any] | None:
        request = self.get_request_eval(request_id)
        if request is None:
            return None
        return self._request_diagnosis_from_scores(request)

    def get_session_evals(self, session_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .where(RequestRunModel.session_id == session_id)
                .order_by(RequestRunModel.created_at.desc())
            ).all()
            return [
                self._request_to_dict(row, self._latest_rag_event(session, row.request_id))
                for row in rows
            ]

    def get_request_events(self, request_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            request = session.get(RequestRunModel, request_id)
            events = session.scalars(
                select(RagRetrievalEventV2Model)
                .where(RagRetrievalEventV2Model.request_id == request_id)
                .order_by(RagRetrievalEventV2Model.created_at)
            ).all()
            return {
                "request": (
                    self._request_to_dict(request, events[-1] if events else None)
                    if request
                    else None
                ),
                "event_summary": {
                    "request_id": request_id,
                    "rag_event_count": len(events),
                    "rag_chunk_count": sum(
                        len(json.loads(item.returned_chunk_ids_json or "[]"))
                        for item in events
                    ),
                },
                "rag_events": [self._rag_event_to_dict(item) for item in events],
                "memory_events": [],
            }

    def get_recent_complete_requests(
        self,
        *,
        hours: int = 168,
        query_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            stmt = (
                select(RequestRunModel)
                .where(RequestRunModel.created_at >= since, RequestRunModel.final_answer != "")
                .order_by(RequestRunModel.created_at.desc())
                .limit(limit)
            )
            if query_type:
                stmt = stmt.where(RequestRunModel.query_type == query_type)
            rows = list(session.scalars(stmt).all())
            return [
                self._request_to_dict(row, self._latest_rag_event(session, row.request_id))
                for row in rows
            ]

    def get_dashboard_stats(
        self,
        *,
        hours: int = 24,
        query_type: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(
                session, since=since, query_type=query_type, user_id=user_id
            )
            if not rows:
                return {
                    "total_requests": 0,
                    "avg_rag_score": 0.0,
                    "avg_memory_score": 0.0,
                    "avg_overall_score": None,
                    "avg_answer_quality_score": 0.0,
                    "avg_latency_ms": 0,
                    "avg_total_tokens": 0,
                    "empty_retrieval_count": 0,
                    "has_error_count": 0,
                    "timeout_count": 0,
                    "has_retry_count": 0,
                    "has_fallback_count": 0,
                    "low_score_count": 0,
                    "by_query_type": {},
                    "by_strategy": {},
                    "recent_scores": [],
                }
            details = [
                self._request_to_dict(row, self._latest_rag_event(session, row.request_id))
                for row in rows
            ]
        memory_scores = [
            item["memory_score"] for item in details if item.get("memory_score") is not None
        ]
        rag_scores = [item["rag_score"] for item in details if item.get("rag_score") is not None]
        answer_quality_scores = [
            item["answer_quality_score"]
            for item in details
            if item.get("answer_quality_score") is not None
        ]
        overall_scores = [
            item["overall_score"] for item in details if item.get("overall_score") is not None
        ]
        latencies = [item["runtime_metrics"].get("latency_ms", 0) for item in details]
        token_counts = [item["runtime_metrics"].get("total_tokens", 0) for item in details]
        by_query_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in details:
            by_query_type[item.get("query_type", "unknown")].append(item)
            by_strategy[item.get("rag_metrics", {}).get("strategy", "unknown")].append(item)
        return {
            "total_requests": len(details),
            "avg_rag_score": round(sum(rag_scores) / len(rag_scores), 4)
            if rag_scores
            else 0.0,
            "avg_memory_score": (
                round(sum(memory_scores) / len(memory_scores), 4) if memory_scores else 0.0
            ),
            "avg_overall_score": (
                round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0
            ),
            "avg_answer_quality_score": (
                round(sum(answer_quality_scores) / len(answer_quality_scores), 4)
                if answer_quality_scores
                else 0.0
            ),
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "avg_total_tokens": int(sum(token_counts) / len(token_counts)) if token_counts else 0,
            "empty_retrieval_count": sum(1 for item in details if not item.get("rag_events")),
            "has_error_count": sum(
                1 for item in details if item["execution_health"].get("has_error")
            ),
            "timeout_count": sum(1 for item in details if item["execution_health"].get("timeout")),
            "has_retry_count": sum(1 for item in details if item.get("has_retry")),
            "has_fallback_count": sum(1 for item in details if item.get("has_fallback")),
            "low_score_count": sum(
                1
                for item in details
                if item.get("overall_score") is not None and item["overall_score"] <= 0.4
            ),
            "by_query_type": {
                key: {"count": len(items)} for key, items in by_query_type.items()
            },
            "by_strategy": {
                key: {"count": len(items)} for key, items in by_strategy.items()
            },
            "recent_scores": [],
        }

    def get_low_score_requests(
        self, threshold: float = 0.4, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .order_by(RequestRunModel.created_at.desc())
            ).all()
            requests = [
                self._request_to_dict(row, self._latest_rag_event(session, row.request_id))
                for row in rows
            ]
            filtered = [
                item
                for item in requests
                if item.get("overall_score") is not None and item["overall_score"] <= threshold
            ]
            return filtered[offset : offset + limit]

    def get_all_requests(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .order_by(RequestRunModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [
                self._request_to_dict(row, self._latest_rag_event(session, row.request_id))
                for row in rows
            ]

    def get_score_trend(
        self,
        hours: int = 168,
        granularity: str = "hourly",
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(session, since=since, user_id=user_id)
        buckets: dict[str, list[RequestRunModel]] = defaultdict(list)
        for row in rows:
            buckets[self._time_bucket_label(row.created_at, granularity)].append(row)
        return [
            {
                "period": period,
                "count": len(items),
                "avg_overall_score": self._avg(
                    [
                        self._compute_overall_score(
                            answer_quality_score=compute_answer_quality_score(
                                query=item.query,
                                query_type=item.query_type,
                                final_answer=item.final_answer,
                            ),
                            rag_score=item.rag_score,
                            memory_score=item.memory_score,
                        )
                        for item in items
                    ]
                ),
                "avg_rag_score": self._avg([item.rag_score for item in items]),
                "avg_memory_score": self._avg([item.memory_score for item in items]),
                "avg_answer_quality_score": self._avg(
                    [
                        compute_answer_quality_score(
                            query=item.query,
                            query_type=item.query_type,
                            final_answer=item.final_answer,
                        )
                        for item in items
                    ]
                ),
            }
            for period, items in sorted(buckets.items())
        ]

    def get_distinct_users(self) -> list[str]:
        with self.session_factory() as session:
            rows = session.scalars(select(RequestRunModel.user_id).distinct()).all()
            return sorted(set(rows))

    def get_eval_rows_for_export(
        self, *, hours: int = 168, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(session, since=since, user_id=user_id)
            return [
                self._flatten_export_row(
                    row,
                    self._latest_rag_event(session, row.request_id),
                    self._latest_memory_eval_run(session, row.request_id),
                    self._latest_memory_retrieval_event(session, row.request_id),
                    self._memory_extraction_event(session, row.request_id),
                )
                for row in rows
            ]

    @staticmethod
    def _filtered_request_rows(
        session: Session,
        *,
        since: datetime,
        query_type: str | None = None,
        user_id: str | None = None,
    ) -> list[RequestRunModel]:
        stmt = select(RequestRunModel).where(RequestRunModel.created_at >= since)
        if query_type:
            stmt = stmt.where(RequestRunModel.query_type == query_type)
        if user_id:
            stmt = stmt.where(RequestRunModel.user_id == user_id)
        return list(session.scalars(stmt.order_by(RequestRunModel.created_at.desc())).all())

    @staticmethod
    def _latest_rag_event(session: Session, request_id: str) -> RagRetrievalEventV2Model | None:
        return session.scalars(
            select(RagRetrievalEventV2Model)
            .where(RagRetrievalEventV2Model.request_id == request_id)
            .order_by(RagRetrievalEventV2Model.created_at.desc())
        ).first()

    @staticmethod
    def _latest_memory_eval_run(session: Session, request_id: str) -> MemoryEvalRunV2Model | None:
        return session.scalars(
            select(MemoryEvalRunV2Model)
            .where(MemoryEvalRunV2Model.request_id == request_id)
            .order_by(MemoryEvalRunV2Model.created_at.desc())
        ).first()

    @staticmethod
    def _latest_memory_retrieval_event(
        session: Session, request_id: str
    ) -> MemoryRetrievalEventV2Model | None:
        return session.scalars(
            select(MemoryRetrievalEventV2Model)
            .where(MemoryRetrievalEventV2Model.request_id == request_id)
            .order_by(MemoryRetrievalEventV2Model.created_at.desc())
        ).first()

    @staticmethod
    def _memory_extraction_event(
        session: Session, request_id: str
    ) -> MemoryExtractionEventV2Model | None:
        return session.scalars(
            select(MemoryExtractionEventV2Model).where(
                MemoryExtractionEventV2Model.request_id == request_id
            )
        ).first()

    @classmethod
    def _request_to_dict(
        cls,
        row: RequestRunModel,
        latest_event: RagRetrievalEventV2Model | None,
    ) -> dict[str, Any]:
        runtime = json.loads(row.runtime_metrics_json or "{}")
        execution_health = json.loads(row.execution_health_json or "{}")
        health_score = cls._compute_execution_health_score(
            has_error=execution_health.get("has_error", False),
            timeout=execution_health.get("timeout", False),
            has_fallback=row.has_fallback,
            has_retry=row.has_retry,
        )
        execution_health["execution_health_score"] = health_score
        rag_metrics = cls._rag_metrics_from_event(latest_event, request_row=row)
        rag_score = row.rag_score if row.rag_score is not None else rag_metrics.get("rag_score")
        faithfulness = (
            row.faithfulness
            if row.faithfulness is not None
            else rag_metrics.get("faithfulness")
        )
        answer_quality_score = compute_answer_quality_score(
            query=row.query,
            query_type=row.query_type,
            final_answer=row.final_answer,
        )
        rag_used = latest_event is not None or rag_score is not None
        memory_used = row.memory_score is not None
        used_modules = {
            "answer": answer_quality_score is not None,
            "rag": rag_used,
            "memory": memory_used,
        }
        overall_score = cls._compute_overall_score(
            answer_quality_score=answer_quality_score,
            rag_score=rag_score if rag_used else None,
            memory_score=row.memory_score if memory_used else None,
        )
        return {
            "request_id": row.request_id,
            "session_id": row.session_id,
            "user_id": row.user_id,
            "query": row.query,
            "query_type": row.query_type,
            "final_answer": row.final_answer,
            "rag_score": rag_score,
            "memory_score": row.memory_score,
            "overall_score": overall_score,
            "answer_quality_score": answer_quality_score,
            "faithfulness_score": faithfulness,
            "rag_eval_status": row.rag_eval_status,
            "rag_scored_at": cls._datetime_to_local_iso(row.rag_scored_at),
            "used_modules": used_modules,
            "has_retry": row.has_retry,
            "has_fallback": row.has_fallback,
            "execution_health_score": health_score,
            "rag_metrics": rag_metrics,
            "memory_metrics": {},
            "execution_health": execution_health,
            "runtime_metrics": runtime,
            "agent_trace": json.loads(row.agent_trace_json or "[]"),
            "agent_events": json.loads(row.agent_events_json or "[]"),
            "answer_event": json.loads(row.answer_event_json or "{}"),
            "rag_events": [cls._rag_event_to_dict(latest_event)] if latest_event else [],
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _request_diagnosis_from_scores(cls, request: dict[str, Any]) -> dict[str, Any]:
        scores = {
            "overall_score": request.get("overall_score"),
            "rag_score": request.get("rag_score"),
            "memory_score": request.get("memory_score"),
            "answer_quality_score": request.get("answer_quality_score"),
        }
        used_modules = request.get("used_modules") or {}
        issues: list[str] = []
        strengths: list[str] = []
        recommendations: list[str] = []

        if used_modules.get("rag"):
            cls._add_score_diagnosis(
                label="RAG",
                score=scores["rag_score"],
                issues=issues,
                strengths=strengths,
                recommendations=recommendations,
                missing_recommendation=(
                    "Add reference and required_points, then open the request detail to run "
                    "RAG evaluation."
                ),
                low_issue="check retrieved contexts, reference coverage, and noisy evidence.",
                low_recommendation=(
                    "Inspect retrieved contexts and required_points before changing retrieval "
                    "settings."
                ),
                strong_detail="retrieval evidence is currently supporting the answer well.",
            )
        if used_modules.get("memory"):
            cls._add_score_diagnosis(
                label="Memory",
                score=scores["memory_score"],
                issues=issues,
                strengths=strengths,
                recommendations=recommendations,
                missing_recommendation="Run Memory evaluation if this request should use memory.",
                low_issue="memory retrieval or memory use quality needs review.",
                low_recommendation=(
                    "Check whether relevant memories were retrieved and used correctly."
                ),
                strong_detail="memory behavior is currently healthy.",
            )
        if used_modules.get("answer"):
            cls._add_score_diagnosis(
                label="Answer",
                score=scores["answer_quality_score"],
                issues=issues,
                strengths=strengths,
                recommendations=recommendations,
                missing_recommendation=(
                    "Check whether the final answer is empty or failed during generation."
                ),
                low_issue="the final answer may miss user intent, coverage, structure, or clarity.",
                low_recommendation=(
                    "Review the final answer against the query for coverage, specificity, "
                    "and format."
                ),
                strong_detail="the final answer is aligned, specific, and clear.",
            )

        if not issues and not strengths and not recommendations:
            recommendations.append("No score-based diagnosis is available for this request.")

        return {
            "request_id": request.get("request_id"),
            "scores": scores,
            "used_modules": used_modules,
            "issues": issues,
            "strengths": strengths,
            "recommendations": recommendations,
        }

    @classmethod
    def _compute_overall_score(
        cls,
        *,
        answer_quality_score: float | None,
        rag_score: float | None,
        memory_score: float | None,
    ) -> float | None:
        scores = {
            "answer_quality_score": answer_quality_score,
            "rag_score": rag_score,
            "memory_score": memory_score,
        }
        weighted = [
            (cls.OVERALL_SCORE_WEIGHTS[name], float(score))
            for name, score in scores.items()
            if score is not None
        ]
        if not weighted:
            return None
        total_weight = sum(weight for weight, _ in weighted)
        return round(sum(weight * score for weight, score in weighted) / total_weight, 4)

    @classmethod
    def _compute_execution_health_score(
        cls,
        *,
        has_error: bool,
        timeout: bool,
        has_fallback: bool,
        has_retry: bool,
    ) -> float:
        score = 1.0
        score -= cls.HEALTH_SCORE_PENALTIES["has_error"] * float(bool(has_error))
        score -= cls.HEALTH_SCORE_PENALTIES["timeout"] * float(bool(timeout))
        score -= cls.HEALTH_SCORE_PENALTIES["has_fallback"] * float(bool(has_fallback))
        score -= cls.HEALTH_SCORE_PENALTIES["has_retry"] * float(bool(has_retry))
        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def _add_score_diagnosis(
        *,
        label: str,
        score: float | None,
        issues: list[str],
        strengths: list[str],
        recommendations: list[str],
        missing_recommendation: str,
        low_issue: str,
        low_recommendation: str,
        strong_detail: str,
    ) -> None:
        if score is None:
            issues.append(f"{label} score is not available for this request.")
            recommendations.append(missing_recommendation)
            return

        value = float(score)
        if value < 0.60:
            issues.append(f"{label} score is low ({value:.2f}); {low_issue}")
            recommendations.append(low_recommendation)
        elif value >= 0.75:
            strengths.append(f"{label} score is strong ({value:.2f}); {strong_detail}")
        else:
            recommendations.append(
                f"{label} score is moderate ({value:.2f}); monitor this request if it is important."
            )

    @classmethod
    def _rag_metrics_from_event(
        cls,
        row: RagRetrievalEventV2Model | None,
        *,
        request_row: RequestRunModel | None = None,
    ) -> dict[str, Any]:
        contexts = json.loads(row.returned_contexts_json or "[]") if row is not None else []
        chunk_ids = json.loads(row.returned_chunk_ids_json or "[]") if row is not None else []
        return {
            "rag_score": cls._metric_value("rag_score", row, request_row),
            "faithfulness": cls._metric_value("faithfulness", row, request_row),
            "answer_relevancy": cls._metric_value("answer_relevancy", row, request_row),
            "context_precision": cls._metric_value("context_precision", row, request_row),
            "context_recall": cls._metric_value("context_recall", row, request_row),
            "noise_sensitivity": cls._metric_value("noise_sensitivity", row, request_row),
            "semantic_similarity": cls._metric_value("semantic_similarity", row, request_row),
            "retrieval_latency_ms": row.latency_ms if row is not None else 0,
            "strategy": row.strategy if row is not None else "",
            "caller_agent": cls._caller_agent_value(row),
            "redundancy": cls._metric_value("redundancy", row, request_row),
            "completeness": cls._metric_value("completeness", row, request_row),
            "returned_chunks_count": len(chunk_ids),
            "retrieved_contexts": contexts,
            "retrieved_chunk_ids": chunk_ids,
        }

    @staticmethod
    def _metric_value(
        field_name: str,
        event: RagRetrievalEventV2Model | None,
        request_row: RequestRunModel | None,
    ) -> float | None:
        if event is not None:
            value = getattr(event, field_name)
            if value is not None:
                return value
        if request_row is None:
            return None
        return getattr(request_row, field_name)

    @classmethod
    def _rag_event_to_dict(cls, row: RagRetrievalEventV2Model | None) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "event_id": row.event_id,
            "request_id": row.request_id,
            "query": row.query,
            "normalized_query": row.normalized_query,
            "strategy": row.strategy,
            "top_k": row.top_k,
            "filters": json.loads(row.filters_json or "{}"),
            "latency_ms": row.latency_ms,
            "returned_contexts": json.loads(row.returned_contexts_json or "[]"),
            "returned_chunk_ids": json.loads(row.returned_chunk_ids_json or "[]"),
            "returned_paper_ids": json.loads(row.returned_paper_ids_json or "[]"),
            "rag_score": row.rag_score,
            "faithfulness": row.faithfulness,
            "answer_relevancy": row.answer_relevancy,
            "context_precision": row.context_precision,
            "context_recall": row.context_recall,
            "noise_sensitivity": row.noise_sensitivity,
            "semantic_similarity": row.semantic_similarity,
            "redundancy": row.redundancy,
            "completeness": row.completeness,
            "caller_agent": cls._caller_agent_value(row),
            "tool_name": row.tool_name,
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @staticmethod
    def _caller_agent_value(row: RagRetrievalEventV2Model | None) -> str:
        if row is None:
            return ""
        if row.caller_agent:
            return row.caller_agent
        if row.tool_name == "rag_retrieve":
            return "researcher"
        if row.tool_name == "rag_top10_similar_papers":
            return "crossdomain"
        return ""

    @classmethod
    def _memory_eval_run_to_dict(cls, row: MemoryEvalRunV2Model | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "run_id": row.run_id,
            "batch_id": row.batch_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "session_id": row.session_id,
            "memory_score": round(row.memory_score, 4),
            "memory_injected_count": row.memory_injected_count,
            "memory_injected_latency_ms": row.memory_injected_latency_ms,
            "memory_injected_tokens": row.memory_injected_tokens,
            "memory_hit_at_k": cls._round_or_none(row.memory_hit_at_k),
            "memory_relevant_recall": cls._round_or_none(row.memory_relevant_recall),
            "memory_relevant_precision": cls._round_or_none(row.memory_relevant_precision),
            "first_relevant_rank": row.first_relevant_rank,
            "memory_stale_retrieval_rate": cls._round_or_none(row.memory_stale_retrieval_rate),
            "memory_answer_relevance": cls._round_or_none(row.memory_answer_relevance),
            "memory_extraction_precision": cls._round_or_none(row.memory_extraction_precision),
            "memory_extraction_latency_ms": row.memory_extraction_latency_ms,
            "memory_extraction_tokens": row.memory_extraction_tokens,
            "score_breakdown": json.loads(row.score_breakdown_json or "{}"),
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _memory_retrieval_event_to_dict(
        cls, row: MemoryRetrievalEventV2Model | None
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "event_id": row.event_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "query": row.query,
            "embedding_latency_ms": row.embedding_latency_ms,
            "vector_search_latency_ms": row.vector_search_latency_ms,
            "retrieved_memory_ids": json.loads(row.retrieved_memory_ids_json or "[]"),
            "retrieved_scores": json.loads(row.retrieved_scores_json or "[]"),
            "retrieved_count": row.retrieved_count,
            "injected_memory_ids": json.loads(row.injected_memory_ids_json or "[]"),
            "injected_count": row.injected_count,
            "injected_text": row.injected_text,
            "injected_tokens": row.injected_tokens,
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _memory_extraction_event_to_dict(
        cls, row: MemoryExtractionEventV2Model | None
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "event_id": row.event_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "dispatch_latency_ms": row.dispatch_latency_ms,
            "dispatch_success": row.dispatch_success,
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "written_memory_ids": json.loads(row.written_memory_ids_json or "[]"),
            "written_memory_texts": json.loads(row.written_memory_texts_json or "[]"),
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @staticmethod
    def _round_or_none(value: float | None) -> float | None:
        return round(float(value), 4) if value is not None else None

    @classmethod
    def _flatten_export_row(
        cls,
        row: RequestRunModel,
        latest_event: RagRetrievalEventV2Model | None,
        memory_run: MemoryEvalRunV2Model | None = None,
        memory_retrieval: MemoryRetrievalEventV2Model | None = None,
        memory_extraction: MemoryExtractionEventV2Model | None = None,
    ) -> dict[str, Any]:
        payload = cls._request_to_dict(row, latest_event)
        runtime = payload["runtime_metrics"]
        health = payload["execution_health"]
        rag_metrics = payload["rag_metrics"]
        total_latency_ms = health.get("total_latency_ms", runtime.get("latency_ms", 0))
        request_overview = {
            "request_id": payload["request_id"],
            "user_id": payload["user_id"],
            "session_id": payload["session_id"],
            "query_type": payload["query_type"],
            "query": payload["query"],
            "final_answer": payload["final_answer"],
            "total_latency_ms": total_latency_ms,
            "execution_health_score": payload["execution_health_score"],
            "prompt_tokens": runtime.get("prompt_tokens", 0),
            "completion_tokens": runtime.get("completion_tokens", 0),
            "total_tokens": runtime.get("total_tokens", 0),
            "overall_score": payload["overall_score"],
            "answer_quality_score": payload["answer_quality_score"],
            "faithfulness_score": payload["faithfulness_score"],
            "has_error": health.get("has_error", False),
            "has_retry": payload["has_retry"],
            "has_fallback": payload["has_fallback"],
            "timeout": health.get("timeout", False),
            "created_at": payload["created_at"],
        }
        memory_data = {
            "run": cls._memory_eval_run_to_dict(memory_run),
            "retrieval_event": cls._memory_retrieval_event_to_dict(memory_retrieval),
            "extraction_event": cls._memory_extraction_event_to_dict(memory_extraction),
        }
        rag_data = {
            "metrics": rag_metrics,
            "events": [cls._rag_event_to_dict(latest_event)] if latest_event else [],
            "empty_retrieval": bool(
                (latest_event is not None and rag_metrics.get("returned_chunks_count") == 0)
                or (
                    rag_metrics.get("returned_chunks_count") == 0
                    and rag_metrics.get("rag_score") is not None
                )
            ),
        }
        return {
            "request_overview": request_overview,
            "memory_data": memory_data,
            "rag_data": rag_data,
            "request_overview_json": json.dumps(request_overview, ensure_ascii=False),
            "memory_data_json": json.dumps(memory_data, ensure_ascii=False),
            "rag_data_json": json.dumps(rag_data, ensure_ascii=False),
        }

    @classmethod
    def _time_bucket_label(cls, dt: datetime, granularity: str) -> str:
        dt = cls._to_local_datetime(dt)
        if granularity == "daily":
            return dt.strftime("%Y-%m-%d")
        if granularity == "weekly":
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        return dt.strftime("%Y-%m-%dT%H:00")

    @staticmethod
    def _local_timezone():
        return datetime.now().astimezone().tzinfo or UTC

    @classmethod
    def _to_local_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(cls._local_timezone())

    @classmethod
    def _datetime_to_local_iso(cls, value: datetime | None) -> str | None:
        localized = cls._to_local_datetime(value)
        return localized.isoformat() if localized else None

    @staticmethod
    def _avg(values: list[float | None]) -> float:
        scored = [float(value) for value in values if value is not None]
        return round(sum(scored) / len(scored), 4) if scored else 0.0
