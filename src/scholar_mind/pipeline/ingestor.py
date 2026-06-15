from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from scholar_mind.models.domain import StructuredPaper
from scholar_mind.pipeline.arxiv_storage import (
    DEFAULT_CATEGORY,
    find_local_artifact,
    find_metadata_path,
    metadata_path,
    pdf_dir,
    preferred_category_name,
    normalize_category_name,
    source_dir,
)
from scholar_mind.pipeline.chunker import StructureAwareChunker
from scholar_mind.pipeline.downloader import (
    ArxivMetadataDownloader,
    ArxivPdfDownloader,
    ArxivSourceDownloader,
)
from scholar_mind.pipeline.parser import LaTeXParser, PDFParser


class ArxivPaperIngestor:
    def __init__(
        self,
        settings,
        paper_repository,
        rag_engine,
        *,
        metadata_downloader: ArxivMetadataDownloader | None = None,
        source_downloader: ArxivSourceDownloader | None = None,
        pdf_downloader: ArxivPdfDownloader | None = None,
        latex_parser: LaTeXParser | None = None,
        pdf_parser: PDFParser | None = None,
        chunker: StructureAwareChunker | None = None,
        proxy: str | None = None,
    ):
        self.settings = settings
        self.paper_repository = paper_repository
        self.rag_engine = rag_engine
        self.proxy = proxy
        self.metadata_downloader = metadata_downloader or ArxivMetadataDownloader(proxy=proxy)
        self.source_downloader = source_downloader or ArxivSourceDownloader(proxy=proxy)
        self.pdf_downloader = pdf_downloader or ArxivPdfDownloader(proxy=proxy)
        self.latex_parser = latex_parser or LaTeXParser()
        self.pdf_parser = pdf_parser or PDFParser()
        self.chunker = chunker or StructureAwareChunker()

    async def ingest_paper(self, paper_id: str, *, category: str | None = None) -> dict[str, Any]:
        metadata = await self._download_metadata(paper_id)
        category_name = preferred_category_name(metadata, category_override=category)
        artifact_path, source_format = await self._download_artifact(paper_id, category=category_name)
        if metadata is not None:
            self._write_local_metadata(paper_id, metadata, category=category_name)
        try:
            parsed = self._parse_artifact(artifact_path, source_format)
        except Exception:
            if source_format != "latex":
                raise
            artifact_path = await self.pdf_downloader.download_single(paper_id, self._pdf_dir(category_name))
            source_format = "pdf"
            parsed = self._parse_artifact(artifact_path, source_format)
        return self._persist_ingested_paper(
            paper_id=paper_id,
            parsed=parsed,
            metadata=metadata,
            source_format=source_format,
            artifact_path=artifact_path,
        )

    async def download_paper_assets(self, paper_id: str, *, category: str | None = None) -> dict[str, Any]:
        metadata = await self._download_metadata(paper_id)
        category_name = preferred_category_name(metadata, category_override=category)
        artifact_path, source_format = await self._download_artifact(paper_id, category=category_name)
        if metadata is not None:
            self._write_local_metadata(paper_id, metadata, category=category_name)
        return {
            "paper_id": paper_id,
            "category": category_name,
            "source_format": source_format,
            "artifact_path": str(artifact_path),
            "metadata_path": str(self._metadata_path(paper_id, category_name)) if metadata is not None else None,
            "has_metadata": metadata is not None,
        }

    async def ingest_local_paper(self, paper_id: str, *, category: str | None = None) -> dict[str, Any]:
        metadata = self._read_local_metadata(paper_id, category=category)
        artifact_path, source_format = self._local_artifact_path(paper_id, category=category)
        parsed, artifact_path, source_format = self._parse_local_artifact(
            paper_id,
            artifact_path=artifact_path,
            source_format=source_format,
            category=category,
        )
        return self._persist_ingested_paper(
            paper_id=paper_id,
            parsed=parsed,
            metadata=metadata,
            source_format=source_format,
            artifact_path=artifact_path,
        )

    async def _download_metadata(self, paper_id: str) -> dict[str, Any] | None:
        try:
            return await self.metadata_downloader.download_record(paper_id)
        except Exception:
            return None

    async def _download_artifact(self, paper_id: str, *, category: str) -> tuple[Path, str]:
        try:
            source_path = await self.source_downloader.download_single(paper_id, self._source_dir(category))
            return source_path, "latex"
        except Exception:
            pdf_path = await self.pdf_downloader.download_single(paper_id, self._pdf_dir(category))
            return pdf_path, "pdf"

    def _local_artifact_path(self, paper_id: str, *, category: str | None = None) -> tuple[Path, str]:
        artifact = find_local_artifact(self._raw_root(), paper_id, category=category)
        if artifact is not None:
            return artifact
        raise FileNotFoundError(f"No local arXiv artifact found for {paper_id}")

    def _parse_artifact(self, artifact_path: Path, source_format: str) -> StructuredPaper:
        if source_format == "latex":
            return self.latex_parser.parse(artifact_path)
        return self.pdf_parser.parse(artifact_path)

    def _parse_local_artifact(
        self,
        paper_id: str,
        *,
        artifact_path: Path,
        source_format: str,
        category: str | None = None,
    ) -> tuple[StructuredPaper, Path, str]:
        try:
            return self._parse_artifact(artifact_path, source_format), artifact_path, source_format
        except Exception:
            if source_format != "latex":
                raise
            fallback_path = self._local_pdf_artifact_path(paper_id, category=category)
            if fallback_path is None:
                raise
            return self._parse_artifact(fallback_path, "pdf"), fallback_path, "pdf"

    def _persist_ingested_paper(
        self,
        *,
        paper_id: str,
        parsed: StructuredPaper,
        metadata: dict[str, Any] | None,
        source_format: str,
        artifact_path: Path,
    ) -> dict[str, Any]:
        merged = self._merge_paper_metadata(
            paper_id=paper_id,
            parsed=parsed,
            metadata=metadata,
            source_format=source_format,
        )
        chunks = self.chunker.chunk(merged)
        self.paper_repository.upsert_structured_paper(merged, chunks=chunks)
        self.rag_engine.upsert_paper(merged, chunks=chunks)
        return {
            "paper_id": merged.paper_id,
            "title": merged.title,
            "source_format": source_format,
            "artifact_path": str(artifact_path),
            "section_count": len(merged.sections),
            "chunk_count": len(chunks),
            "has_source": bool(merged.metadata.get("has_source", False)),
        }

    def _raw_root(self) -> Path:
        return self.settings.resolve_path(self.settings.raw_data_dir) / "arxiv"

    def _source_dir(self, category: str = DEFAULT_CATEGORY) -> Path:
        return source_dir(self._raw_root(), category)

    def _pdf_dir(self, category: str = DEFAULT_CATEGORY) -> Path:
        return pdf_dir(self._raw_root(), category)

    def _metadata_path(self, paper_id: str, category: str = DEFAULT_CATEGORY) -> Path:
        return metadata_path(self._raw_root(), paper_id, category)

    def _write_local_metadata(self, paper_id: str, metadata: dict[str, Any], *, category: str) -> None:
        path = self._metadata_path(paper_id, category)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _local_pdf_artifact_path(self, paper_id: str, *, category: str | None = None) -> Path | None:
        raw_root = self._raw_root()
        candidates: list[Path] = []
        if category:
            normalized = normalize_category_name(category)
            candidates.append(pdf_dir(raw_root, normalized) / f"{paper_id}.pdf")
        candidates.append(raw_root / "pdf" / f"{paper_id}.pdf")
        candidates.extend(sorted((raw_root / "pdf").glob(f"*/{paper_id}.pdf")))
        for path in candidates:
            if path.exists():
                return path
        return None

    def _read_local_metadata(self, paper_id: str, *, category: str | None = None) -> dict[str, Any] | None:
        path = find_metadata_path(self._raw_root(), paper_id, category=category)
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _merge_paper_metadata(
        *, paper_id: str, parsed: StructuredPaper, metadata: dict[str, Any] | None, source_format: str
    ) -> StructuredPaper:
        payload = metadata or {}
        publish_date = parsed.publish_date
        created = payload.get("created")
        if created:
            publish_date = date.fromisoformat(created)
        merged_metadata = dict(parsed.metadata)
        merged_metadata.update(
            {
                "source_format": source_format,
                "has_source": source_format == "latex",
                "ingested_from": "arxiv",
            }
        )
        return StructuredPaper(
            paper_id=payload.get("paper_id") or paper_id or parsed.paper_id,
            title=payload.get("title") or parsed.title,
            authors=payload.get("authors") or parsed.authors,
            abstract=payload.get("abstract") or parsed.abstract,
            categories=payload.get("categories") or parsed.categories,
            publish_date=publish_date,
            citation_count=parsed.citation_count,
            sections=parsed.sections,
            references=parsed.references,
            metadata=merged_metadata,
        )
