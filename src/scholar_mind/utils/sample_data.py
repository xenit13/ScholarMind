from __future__ import annotations

import json

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from scholar_mind.config.settings import Settings
from scholar_mind.db.models import PaperChunkModel, PaperModel, PaperSectionModel
from scholar_mind.models.domain import StructuredPaper
from scholar_mind.pipeline.chunker import StructureAwareChunker


def load_seed_papers(settings: Settings) -> list[StructuredPaper]:
    path = settings.resolve_path(settings.papers_seed_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [StructuredPaper.model_validate(item) for item in payload]


def _has_existing_paper_data(session: Session) -> bool:
    for model in (PaperModel, PaperSectionModel, PaperChunkModel):
        if session.execute(select(model).limit(1)).first() is not None:
            return True
    return False


def seed_sample_data(session: Session, settings: Settings) -> bool:
    if _has_existing_paper_data(session):
        return False

    seed_papers = load_seed_papers(settings)
    if not seed_papers:
        return False

    session.execute(delete(PaperChunkModel))
    session.execute(delete(PaperSectionModel))
    session.execute(delete(PaperModel))

    chunker = StructureAwareChunker()
    for paper in seed_papers:
        session.add(
            PaperModel(
                paper_id=paper.paper_id,
                title=paper.title,
                authors_json=json.dumps(paper.authors),
                abstract=paper.abstract,
                categories_json=json.dumps(paper.categories),
                publish_date=paper.publish_date,
                citation_count=paper.citation_count,
                has_source=bool(paper.metadata.get("has_source", True)),
            )
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
        for chunk in chunker.chunk(paper):
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
    return True
