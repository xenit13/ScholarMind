from __future__ import annotations

import json
import tarfile
from datetime import date
from pathlib import Path

import fitz
import pytest

from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.models.domain import PaperSection, StructuredPaper
from scholar_mind.pipeline.arxiv_storage import list_category_paper_ids, preferred_category_name
from scholar_mind.pipeline.chunker import StructureAwareChunker
from scholar_mind.pipeline.downloader import ArxivMetadataDownloader, ArxivPdfDownloader
from scholar_mind.pipeline.ingestor import ArxivPaperIngestor
from scholar_mind.pipeline.parser import LaTeXParser, PDFParser
from scholar_mind.services.repositories import PaperRepository


def test_latex_parser_extracts_basic_metadata(tmp_path):
    tex_file = tmp_path / "sample.tex"
    tex_file.write_text(
        r"""
        \title{Sample Retrieval Paper}
        \author{Ada Lovelace \and Alan Turing}
        \begin{abstract}
        This paper studies retrieval quality.
        \end{abstract}
        \section{Introduction}
        Retrieval quality matters.
        \section{Method}
        We combine dense and sparse retrieval.
        """,
        encoding="utf-8",
    )

    paper = LaTeXParser().parse(tex_file)

    assert paper.title == "Sample Retrieval Paper"
    assert paper.abstract == "This paper studies retrieval quality."
    assert [section.title for section in paper.sections] == ["Introduction", "Method"]


def test_latex_parser_expands_input_and_include_from_archive(tmp_path):
    source_root = tmp_path / "archive"
    sections_dir = source_root / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    (source_root / "arxiv-main.tex").write_text(
        r"""
        \documentclass{article}
        \title{Included Sections Paper}
        \author{Ada Lovelace}
        \begin{document}
        \maketitle
        \begin{abstract}
        This paper uses input files.
        \end{abstract}
        \input{sections/intro}
        \include{sections/method}
        \end{document}
        """,
        encoding="utf-8",
    )
    (sections_dir / "intro.tex").write_text(
        r"""
        \section{Introduction}
        Retrieval quality matters for grounded reasoning.
        """,
        encoding="utf-8",
    )
    (sections_dir / "method.tex").write_text(
        r"""
        \section{Method}
        We combine symbolic decomposition with neural retrieval.
        """,
        encoding="utf-8",
    )

    archive_path = tmp_path / "included-paper.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_root / "arxiv-main.tex", arcname="arxiv-main.tex")
        archive.add(sections_dir / "intro.tex", arcname="sections/intro.tex")
        archive.add(sections_dir / "method.tex", arcname="sections/method.tex")

    paper = LaTeXParser().parse(archive_path)

    assert paper.title == "Included Sections Paper"
    assert paper.abstract == "This paper uses input files."
    assert [section.title for section in paper.sections] == ["Introduction", "Method"]


def test_pdf_parser_extracts_sections(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Sample PDF Paper\nAbstract\nThis is the abstract.\nIntroduction\nThe body starts here.",
    )
    document.save(pdf_file)
    document.close()

    paper = PDFParser().parse(pdf_file)

    assert paper.title == "Sample PDF Paper"
    assert "abstract" in paper.abstract.lower()
    assert paper.sections


def test_structure_aware_chunker_emits_table_formula_and_algorithm_chunks():
    paper = StructuredPaper(
        paper_id="paper-1",
        title="Chunk Test",
        authors=["Tester"],
        abstract="A paper with structured content.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="s1",
                title="Method",
                content=(
                    "Algorithm: retrieve then rerank.\n\n"
                    "score = $q \\times d$\n\n"
                    "col1 | col2\n"
                    "a | b\n"
                ),
            )
        ],
    )

    chunk_types = {chunk.chunk_type for chunk in StructureAwareChunker().chunk(paper)}

    assert {"metadata", "section", "algorithm", "formula", "table"} <= {
        chunk_type.value for chunk_type in chunk_types
    }


def test_structure_aware_chunker_does_not_treat_absolute_value_as_table():
    paper = StructuredPaper(
        paper_id="paper-math",
        title="Math Test",
        authors=["Tester"],
        abstract="A paper with math prose.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="s1",
                title="Method",
                content=(
                    "We bound the error by |x| and continue the derivation.\n"
                    "This remains ordinary prose without a real table.\n"
                ),
            )
        ],
    )

    chunk_types = {chunk.chunk_type.value for chunk in StructureAwareChunker().chunk(paper)}

    assert "table" not in chunk_types


def test_structure_aware_chunker_uses_section_id_in_chunk_ids_for_duplicate_titles():
    paper = StructuredPaper(
        paper_id="paper-dup",
        title="Duplicate Sections",
        authors=["Tester"],
        abstract="A paper with repeated section titles.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="section-1",
                title="Evaluation",
                content="First evaluation content " * 60,
            ),
            PaperSection(
                section_id="section-2",
                title="Evaluation",
                content="Second evaluation content " * 60,
            ),
        ],
    )

    chunks = StructureAwareChunker().chunk(paper)
    section_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_type.value == "section"]

    assert len(section_chunk_ids) == len(set(section_chunk_ids))
    assert any(chunk_id.startswith("paper-dup::section-1::") for chunk_id in section_chunk_ids)
    assert any(chunk_id.startswith("paper-dup::section-2::") for chunk_id in section_chunk_ids)


def test_arxiv_metadata_downloader_parses_oai_response():
    xml_text = """
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
             xmlns:arxiv="http://arxiv.org/OAI/arXiv/">
      <ListRecords>
        <record>
          <metadata>
            <arxiv:arXiv>
              <arxiv:id>2405.00005</arxiv:id>
              <arxiv:created>2024-05-01</arxiv:created>
              <arxiv:updated>2024-05-02</arxiv:updated>
              <arxiv:title> Retrieval Planning </arxiv:title>
              <arxiv:abstract> Agentic retrieval system. </arxiv:abstract>
              <arxiv:categories>cs.AI cs.CL</arxiv:categories>
              <arxiv:authors>
                <arxiv:author>
                  <arxiv:keyname>Lovelace</arxiv:keyname>
                  <arxiv:forenames>Ada</arxiv:forenames>
                </arxiv:author>
              </arxiv:authors>
            </arxiv:arXiv>
          </metadata>
        </record>
        <resumptionToken>next-token</resumptionToken>
      </ListRecords>
    </OAI-PMH>
    """

    records, token = ArxivMetadataDownloader()._parse_response(xml_text)

    assert token == "next-token"
    assert records == [
        {
            "paper_id": "2405.00005",
            "title": "Retrieval Planning",
            "abstract": "Agentic retrieval system.",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.AI", "cs.CL"],
            "created": "2024-05-01",
            "updated": "2024-05-02",
        }
    ]


@pytest.mark.asyncio
async def test_arxiv_metadata_downloader_fetch_page_uses_current_endpoint_and_redirects(monkeypatch):
    calls = {}

    class DummyResponse:
        text = "<xml/>"

        def raise_for_status(self):
            return None

    class DummyAsyncClient:
        def __init__(self, *, timeout, follow_redirects):
            calls["timeout"] = timeout
            calls["follow_redirects"] = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params):
            calls["url"] = url
            calls["params"] = params
            return DummyResponse()

    monkeypatch.setattr("scholar_mind.pipeline.downloader.httpx.AsyncClient", DummyAsyncClient)

    xml_text = await ArxivMetadataDownloader()._fetch_page(
        {"verb": "GetRecord", "identifier": "oai:arXiv.org:2405.00005", "metadataPrefix": "arXiv"}
    )

    assert xml_text == "<xml/>"
    assert calls["url"] == "https://oaipmh.arxiv.org/oai"
    assert calls["follow_redirects"] is True


@pytest.mark.asyncio
async def test_arxiv_pdf_downloader_uses_pdf_url_and_redirects(monkeypatch, tmp_path):
    calls = {}

    class DummyResponse:
        content = b"%PDF-1.7"

        def raise_for_status(self):
            return None

    class DummyAsyncClient:
        def __init__(self, *, timeout, follow_redirects):
            calls["timeout"] = timeout
            calls["follow_redirects"] = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            calls["url"] = url
            return DummyResponse()

    monkeypatch.setattr("scholar_mind.pipeline.downloader.httpx.AsyncClient", DummyAsyncClient)

    path = await ArxivPdfDownloader().download_single("2405.00005", tmp_path)

    assert path.read_bytes() == b"%PDF-1.7"
    assert calls["url"] == "https://arxiv.org/pdf/2405.00005.pdf"
    assert calls["follow_redirects"] is True


@pytest.mark.asyncio
async def test_arxiv_paper_ingestor_falls_back_to_pdf_and_persists_chunks(tmp_path):
    class DummyMetadataDownloader:
        async def download_record(self, paper_id: str):
            assert paper_id == "2405.00005"
            return {
                "paper_id": paper_id,
                "title": "Scheduling of Distributed Applications on the Computing Continuum: A Survey",
                "abstract": "A survey of scheduling on the computing continuum.",
                "authors": ["Ada Lovelace"],
                "categories": ["cs.DC"],
                "created": "2024-05-01",
            }

    class FailingSourceDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            raise RuntimeError(f"source unavailable for {paper_id}")

    class DummyPdfDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{paper_id}.pdf"
            path.write_bytes(b"%PDF-1.7")
            return path

    class DummyPdfParser:
        def parse(self, pdf_path: Path):
            return StructuredPaper(
                paper_id=pdf_path.stem,
                title="Parsed From PDF",
                authors=["Unknown Author"],
                abstract="Parsed abstract from the fallback PDF.",
                categories=["cs.AI"],
                publish_date=date(2024, 1, 1),
                sections=[
                    PaperSection(
                        section_id="section-1",
                        title="Introduction",
                        content="The fallback parser extracted this introduction section.",
                    )
                ],
                metadata={"source_format": "pdf"},
            )

    class RecordingRagEngine:
        def __init__(self):
            self.paper = None
            self.chunks = None

        def upsert_paper(self, paper, chunks=None):
            self.paper = paper
            self.chunks = chunks or []

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ingest.db'}",
        checkpoint_database_url=f"sqlite:///{tmp_path / 'checkpoints.db'}",
        bootstrap_sample_data=False,
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = PaperRepository(build_session_factory(settings))
    rag_engine = RecordingRagEngine()
    ingestor = ArxivPaperIngestor(
        settings,
        repository,
        rag_engine,
        metadata_downloader=DummyMetadataDownloader(),
        source_downloader=FailingSourceDownloader(),
        pdf_downloader=DummyPdfDownloader(),
        pdf_parser=DummyPdfParser(),
    )

    result = await ingestor.ingest_paper("2405.00005")
    stored = repository.get_paper("2405.00005")
    stored_chunks = [chunk for chunk in repository.list_chunk_models() if chunk.paper_id == "2405.00005"]

    assert result["paper_id"] == "2405.00005"
    assert result["source_format"] == "pdf"
    assert result["has_source"] is False
    assert result["chunk_count"] == len(stored_chunks)
    assert stored is not None
    assert stored.title == "Scheduling of Distributed Applications on the Computing Continuum: A Survey"
    assert stored.authors == ["Ada Lovelace"]
    assert stored.categories == ["cs.DC"]
    assert stored.publish_date == date(2024, 5, 1)
    assert stored.metadata["has_source"] is False
    assert stored_chunks
    assert rag_engine.paper is not None
    assert rag_engine.paper.metadata["source_format"] == "pdf"


@pytest.mark.asyncio
async def test_arxiv_paper_ingestor_falls_back_to_pdf_when_source_parse_fails(tmp_path):
    class DummyMetadataDownloader:
        async def download_record(self, paper_id: str):
            return {
                "paper_id": paper_id,
                "title": "Title From Metadata",
                "abstract": "Abstract From Metadata",
                "authors": ["Ada Lovelace"],
                "categories": ["cs.AI"],
                "created": "2024-05-03",
            }

    class DummySourceDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{paper_id}.tar.gz"
            path.write_text("not a gzip archive", encoding="utf-8")
            return path

    class DummyPdfDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{paper_id}.pdf"
            path.write_bytes(b"%PDF-1.7")
            return path

    class FailingLatexParser:
        def parse(self, source_path: Path):
            raise ValueError(f"bad source: {source_path}")

    class DummyPdfParser:
        def parse(self, pdf_path: Path):
            return StructuredPaper(
                paper_id=pdf_path.stem,
                title="Parsed From PDF",
                authors=["Grace Hopper"],
                abstract="Parsed abstract from the fallback PDF.",
                categories=["cs.AI"],
                publish_date=date(2024, 1, 1),
                sections=[
                    PaperSection(
                        section_id="section-1",
                        title="Introduction",
                        content="The fallback parser extracted this introduction section.",
                    )
                ],
                metadata={"source_format": "pdf"},
            )

    class RecordingRagEngine:
        def __init__(self):
            self.paper = None
            self.chunks = None

        def upsert_paper(self, paper, chunks=None):
            self.paper = paper
            self.chunks = chunks or []

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ingest-parse-fallback.db'}",
        checkpoint_database_url=f"sqlite:///{tmp_path / 'checkpoints-parse-fallback.db'}",
        bootstrap_sample_data=False,
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = PaperRepository(build_session_factory(settings))
    rag_engine = RecordingRagEngine()
    ingestor = ArxivPaperIngestor(
        settings,
        repository,
        rag_engine,
        metadata_downloader=DummyMetadataDownloader(),
        source_downloader=DummySourceDownloader(),
        pdf_downloader=DummyPdfDownloader(),
        latex_parser=FailingLatexParser(),
        pdf_parser=DummyPdfParser(),
    )

    result = await ingestor.ingest_paper("2405.00008")
    stored = repository.get_paper("2405.00008")

    assert result["source_format"] == "pdf"
    assert result["has_source"] is False
    assert stored is not None
    assert stored.title == "Title From Metadata"
    assert stored.authors == ["Ada Lovelace"]
    assert stored.metadata["has_source"] is False
    assert rag_engine.paper is not None
    assert rag_engine.paper.metadata["source_format"] == "pdf"
    assert rag_engine.chunks


@pytest.mark.asyncio
async def test_arxiv_paper_ingestor_ingests_from_local_artifacts_without_remote_download(tmp_path):
    class UnexpectedRemoteMetadataDownloader:
        async def download_record(self, paper_id: str):
            raise AssertionError(f"remote metadata download should not be used for {paper_id}")

    class UnexpectedRemoteSourceDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            raise AssertionError(f"remote source download should not be used for {paper_id}")

    class UnexpectedRemotePdfDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            raise AssertionError(f"remote pdf download should not be used for {paper_id}")

    class RecordingRagEngine:
        def __init__(self):
            self.paper = None
            self.chunks = None

        def upsert_paper(self, paper, chunks=None):
            self.paper = paper
            self.chunks = chunks or []

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ingest-local.db'}",
        checkpoint_database_url=f"sqlite:///{tmp_path / 'checkpoints-local.db'}",
        bootstrap_sample_data=False,
        qdrant_location=":memory:",
        raw_data_dir=str(tmp_path / "raw"),
    )
    init_database(settings)
    repository = PaperRepository(build_session_factory(settings))
    rag_engine = RecordingRagEngine()
    ingestor = ArxivPaperIngestor(
        settings,
        repository,
        rag_engine,
        metadata_downloader=UnexpectedRemoteMetadataDownloader(),
        source_downloader=UnexpectedRemoteSourceDownloader(),
        pdf_downloader=UnexpectedRemotePdfDownloader(),
    )

    source_dir = tmp_path / "raw" / "arxiv" / "source" / "cs.AI"
    metadata_dir = tmp_path / "raw" / "arxiv" / "metadata" / "cs.AI"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    tex_path = tmp_path / "main.tex"
    tex_path.write_text(
        r"""
        \title{Local Source Title}
        \author{Grace Hopper}
        \begin{abstract}
        Local abstract from source.
        \end{abstract}
        \section{Introduction}
        Local source content.
        """,
        encoding="utf-8",
    )
    archive_path = source_dir / "2405.00006.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tex_path, arcname="main.tex")

    (metadata_dir / "2405.00006.json").write_text(
        json.dumps(
            {
                "paper_id": "2405.00006",
                "title": "Title From Metadata",
                "abstract": "Abstract From Metadata",
                "authors": ["Ada Lovelace"],
                "categories": ["cs.AI"],
                "created": "2024-05-03",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = await ingestor.ingest_local_paper("2405.00006", category="cs.AI")
    stored = repository.get_paper("2405.00006")
    stored_chunks = [chunk for chunk in repository.list_chunk_models() if chunk.paper_id == "2405.00006"]

    assert result["paper_id"] == "2405.00006"
    assert result["source_format"] == "latex"
    assert result["has_source"] is True
    assert result["artifact_path"] == str(archive_path)
    assert stored is not None
    assert stored.title == "Title From Metadata"
    assert stored.abstract == "Abstract From Metadata"
    assert stored.authors == ["Ada Lovelace"]
    assert stored.categories == ["cs.AI"]
    assert stored.publish_date == date(2024, 5, 3)
    assert stored.metadata["has_source"] is True
    assert stored_chunks
    assert rag_engine.paper is not None
    assert rag_engine.paper.metadata["source_format"] == "latex"


@pytest.mark.asyncio
async def test_arxiv_paper_ingestor_falls_back_to_local_pdf_when_source_archive_is_invalid(tmp_path):
    class UnexpectedRemoteMetadataDownloader:
        async def download_record(self, paper_id: str):
            raise AssertionError(f"remote metadata download should not be used for {paper_id}")

    class UnexpectedRemoteSourceDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            raise AssertionError(f"remote source download should not be used for {paper_id}")

    class UnexpectedRemotePdfDownloader:
        async def download_single(self, paper_id: str, output_dir: Path):
            raise AssertionError(f"remote pdf download should not be used for {paper_id}")

    class RecordingRagEngine:
        def __init__(self):
            self.paper = None
            self.chunks = None

        def upsert_paper(self, paper, chunks=None):
            self.paper = paper
            self.chunks = chunks or []

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ingest-local-fallback.db'}",
        checkpoint_database_url=f"sqlite:///{tmp_path / 'checkpoints-local-fallback.db'}",
        bootstrap_sample_data=False,
        qdrant_location=":memory:",
        raw_data_dir=str(tmp_path / "raw"),
    )
    init_database(settings)
    repository = PaperRepository(build_session_factory(settings))
    rag_engine = RecordingRagEngine()
    ingestor = ArxivPaperIngestor(
        settings,
        repository,
        rag_engine,
        metadata_downloader=UnexpectedRemoteMetadataDownloader(),
        source_downloader=UnexpectedRemoteSourceDownloader(),
        pdf_downloader=UnexpectedRemotePdfDownloader(),
    )

    source_dir = tmp_path / "raw" / "arxiv" / "source" / "cs.AI"
    pdf_dir = tmp_path / "raw" / "arxiv" / "pdf" / "cs.AI"
    metadata_dir = tmp_path / "raw" / "arxiv" / "metadata" / "cs.AI"
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "2405.00007.tar.gz"
    source_path.write_text("not a gzip archive", encoding="utf-8")
    pdf_path = pdf_dir / "2405.00007.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fallback pdf")

    (metadata_dir / "2405.00007.json").write_text(
        json.dumps(
            {
                "paper_id": "2405.00007",
                "title": "Metadata Title",
                "abstract": "Metadata abstract",
                "authors": ["Ada Lovelace"],
                "categories": ["cs.AI"],
                "created": "2024-05-04",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class FallbackPdfParser:
        def parse(self, path: Path):
            assert path == pdf_path
            return StructuredPaper(
                paper_id="2405.00007",
                title="PDF fallback title",
                authors=["Grace Hopper"],
                abstract="PDF fallback abstract",
                categories=["cs.AI"],
                publish_date=date(2024, 5, 4),
                sections=[
                    PaperSection(
                        section_id="section-1",
                        title="Body",
                        content="Fallback PDF body.",
                    )
                ],
                metadata={"source_format": "pdf"},
            )

    ingestor.pdf_parser = FallbackPdfParser()

    result = await ingestor.ingest_local_paper("2405.00007", category="cs.AI")
    stored = repository.get_paper("2405.00007")

    assert result["source_format"] == "pdf"
    assert result["artifact_path"] == str(pdf_path)
    assert result["has_source"] is False
    assert stored is not None
    assert stored.title == "Metadata Title"
    assert stored.authors == ["Ada Lovelace"]
    assert stored.publish_date == date(2024, 5, 4)
    assert rag_engine.paper is not None
    assert rag_engine.paper.metadata["source_format"] == "pdf"


def test_preferred_category_name_prefers_single_requested_category():
    category = preferred_category_name(
        {"paper_id": "2405.00007", "categories": ["cs.LG", "cs.AI"]},
        requested_categories=["cs.AI"],
    )

    assert category == "cs.AI"


def test_list_category_paper_ids_reads_category_subdirectories(tmp_path):
    raw_root = tmp_path / "raw" / "arxiv"
    (raw_root / "metadata" / "cs.AI").mkdir(parents=True, exist_ok=True)
    (raw_root / "source" / "cs.AI").mkdir(parents=True, exist_ok=True)
    (raw_root / "metadata" / "cs.AI" / "2604.16205.json").write_text("{}", encoding="utf-8")
    (raw_root / "source" / "cs.AI" / "2604.16206.tar.gz").write_bytes(b"x")

    paper_ids = list_category_paper_ids(raw_root, "cs.AI")

    assert paper_ids == ["2604.16205", "2604.16206"]
