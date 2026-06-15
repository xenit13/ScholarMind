from __future__ import annotations

import json
from datetime import date

from scholar_mind.config.settings import Settings
from scholar_mind.db.models import PaperChunkModel, PaperModel, PaperSectionModel
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.models.domain import PaperSection, StructuredPaper
from scholar_mind.services.repositories import PaperRepository
from scholar_mind.utils.sample_data import seed_sample_data


def _settings(tmp_path, seed_payload: list[dict]) -> Settings:
    seed_path = tmp_path / "sample_papers.json"
    seed_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'scholar_mind.db'}",
        checkpoint_database_url=f"sqlite:///{tmp_path / 'checkpoints.db'}",
        papers_seed_path=str(seed_path),
        qdrant_location=":memory:",
    )


def _paper(paper_id: str) -> StructuredPaper:
    return StructuredPaper(
        paper_id=paper_id,
        title="Persisted Paper",
        authors=["Ada Lovelace"],
        abstract="Existing paper data must not be deleted by sample bootstrapping.",
        categories=["cs.AI"],
        publish_date=date(2026, 1, 1),
        sections=[
            PaperSection(
                section_id="section-1",
                title="Introduction",
                content="Existing section content.",
            )
        ],
    )


def test_empty_sample_seed_does_not_delete_existing_paper_data(tmp_path):
    settings = _settings(tmp_path, [])
    init_database(settings)
    session_factory = build_session_factory(settings)
    repository = PaperRepository(session_factory)
    repository.upsert_structured_paper(_paper("2604.00001"))

    with session_factory() as session:
        seeded = seed_sample_data(session, settings)
        paper_count = session.query(PaperModel).count()
        section_count = session.query(PaperSectionModel).count()
        chunk_count = session.query(PaperChunkModel).count()

    assert seeded is False
    assert paper_count == 1
    assert section_count == 1
    assert chunk_count > 0


def test_sample_seed_does_not_replace_existing_paper_data(tmp_path):
    seed_paper = _paper("seed-paper").model_dump(mode="json")
    settings = _settings(tmp_path, [seed_paper])
    init_database(settings)
    session_factory = build_session_factory(settings)
    repository = PaperRepository(session_factory)
    repository.upsert_structured_paper(_paper("2604.00001"))

    with session_factory() as session:
        seeded = seed_sample_data(session, settings)
        paper_ids = [row.paper_id for row in session.query(PaperModel).all()]

    assert seeded is False
    assert paper_ids == ["2604.00001"]


def test_sample_seed_populates_empty_paper_tables(tmp_path):
    seed_paper = _paper("seed-paper").model_dump(mode="json")
    settings = _settings(tmp_path, [seed_paper])
    init_database(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        seeded = seed_sample_data(session, settings)
        session.commit()
        paper_ids = [row.paper_id for row in session.query(PaperModel).all()]
        section_count = session.query(PaperSectionModel).count()
        chunk_count = session.query(PaperChunkModel).count()

    assert seeded is True
    assert paper_ids == ["seed-paper"]
    assert section_count == 1
    assert chunk_count > 0
