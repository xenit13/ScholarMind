from __future__ import annotations

import json
import posixpath
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import fitz

from scholar_mind.models.domain import PaperSection, StructuredPaper

TITLE_RE = re.compile(r"\\title\{(?P<value>.*?)\}", re.DOTALL)
AUTHOR_RE = re.compile(r"\\author\{(?P<value>.*?)\}", re.DOTALL)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(?P<value>.*?)\\end\{abstract\}", re.DOTALL)
SECTION_RE = re.compile(r"\\section\{(?P<title>.*?)\}")


class StructuredPaperParser:
    """Small parser for local MVP data files."""

    def parse_json(self, path: Path) -> StructuredPaper:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return StructuredPaper.model_validate(payload)


class LaTeXParser:
    """Parse basic arXiv source bundles into a StructuredPaper."""

    INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{(?P<path>[^}]+)\}")

    def parse(self, source_path: Path) -> StructuredPaper:
        if source_path.suffix == ".json":
            return StructuredPaperParser().parse_json(source_path)
        tex = self._load_tex(source_path)
        return self._parse_tex(tex, paper_id=source_path.stem.replace(".tar", ""))

    def _load_tex(self, source_path: Path) -> str:
        if source_path.suffix == ".tex":
            return source_path.read_text(encoding="utf-8", errors="ignore")
        if source_path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(source_path, "r:gz") as archive:
                tex_files = self._read_archive_tex_files(archive)
                if not tex_files:
                    raise ValueError(f"No .tex file found in {source_path}")
                root_name = self._select_root_tex_file(tex_files)
                return self._expand_tex_includes(root_name, tex_files, active=())
        raise ValueError(f"Unsupported LaTeX source format: {source_path.suffix}")

    def _read_archive_tex_files(self, archive: tarfile.TarFile) -> dict[str, str]:
        tex_files: dict[str, str] = {}
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".tex"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            tex_files[member.name] = handle.read().decode("utf-8", errors="ignore")
        return tex_files

    def _select_root_tex_file(self, tex_files: dict[str, str]) -> str:
        ranked = sorted(
            tex_files.items(),
            key=lambda item: self._root_file_sort_key(item[0], item[1]),
            reverse=True,
        )
        return ranked[0][0]

    def _root_file_sort_key(self, name: str, content: str) -> tuple[int, int, int, int]:
        basename = posixpath.basename(name).lower()
        lower_content = content.lower()
        has_documentclass = "\\documentclass" in lower_content
        has_begin_document = "\\begin{document}" in lower_content
        has_main_name = basename == "main.tex"
        looks_like_root = basename.endswith("main.tex") or basename in {"paper.tex", "ms.tex"}
        depth = name.count("/")
        return (
            int(has_documentclass),
            int(has_begin_document),
            2 if has_main_name else int(looks_like_root),
            -depth,
        )

    def _expand_tex_includes(
        self,
        tex_name: str,
        tex_files: dict[str, str],
        *,
        active: tuple[str, ...],
    ) -> str:
        if tex_name in active:
            return ""
        text = self._strip_latex_comments(tex_files[tex_name])
        base_dir = posixpath.dirname(tex_name)

        def replace_include(match: re.Match[str]) -> str:
            raw_path = match.group("path").strip()
            for candidate in self._resolve_include_candidates(base_dir, raw_path):
                if candidate in tex_files:
                    return self._expand_tex_includes(candidate, tex_files, active=(*active, tex_name))
            return ""

        return self.INCLUDE_RE.sub(replace_include, text)

    def _resolve_include_candidates(self, base_dir: str, include_path: str) -> list[str]:
        normalized = posixpath.normpath(posixpath.join(base_dir, include_path))
        candidates = [normalized]
        if not normalized.endswith(".tex"):
            candidates.append(f"{normalized}.tex")
        return candidates

    @staticmethod
    def _strip_latex_comments(text: str) -> str:
        return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())

    def _parse_tex(self, tex: str, *, paper_id: str) -> StructuredPaper:
        title = self._match_or_default(TITLE_RE, tex, paper_id.replace("_", " "))
        raw_authors = self._match_or_default(AUTHOR_RE, tex, "Unknown Author")
        authors = [
            item.strip()
            for item in re.split(r"\\\\|\\and|,", raw_authors.replace("\n", " "))
            if item.strip()
        ]
        abstract = self._match_or_default(ABSTRACT_RE, tex, "")
        sections = self._extract_sections(tex)
        return StructuredPaper(
            paper_id=paper_id,
            title=self._clean_tex(title),
            authors=[self._clean_tex(author) for author in authors] or ["Unknown Author"],
            abstract=self._clean_tex(abstract),
            categories=[],
            publish_date=datetime.now(UTC).date(),
            sections=sections,
            references=[],
            metadata={"source_format": "latex"},
        )

    def _extract_sections(self, tex: str) -> list[PaperSection]:
        matches = list(SECTION_RE.finditer(tex))
        if not matches:
            body = self._clean_tex(tex)
            return [PaperSection(section_id="section-1", title="Body", content=body)]
        sections: list[PaperSection] = []
        for index, match in enumerate(matches, start=1):
            start = match.end()
            end = matches[index].start() if index < len(matches) else len(tex)
            content = self._clean_tex(tex[start:end])
            sections.append(
                PaperSection(
                    section_id=f"section-{index}",
                    title=self._clean_tex(match.group("title")),
                    content=content,
                )
            )
        return sections

    @staticmethod
    def _clean_tex(value: str) -> str:
        cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^}]*)\})?", r"\1", value)
        cleaned = re.sub(r"\$+", "", cleaned)
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def _match_or_default(pattern: re.Pattern[str], text: str, fallback: str) -> str:
        match = pattern.search(text)
        return match.group("value") if match else fallback


class PDFParser:
    """Parse PDF text into a minimal StructuredPaper."""

    def parse(self, pdf_path: Path) -> StructuredPaper:
        with fitz.open(pdf_path) as document:
            pages = [page.get_text("text") for page in document]
        text = "\n".join(page for page in pages if page.strip())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"No extractable text found in {pdf_path}")
        title = lines[0]
        sections = self._extract_sections(lines)
        abstract = self._extract_abstract(lines)
        return StructuredPaper(
            paper_id=pdf_path.stem,
            title=title,
            authors=["Unknown Author"],
            abstract=abstract,
            categories=[],
            publish_date=datetime.now(UTC).date(),
            sections=sections,
            references=[],
            metadata={"source_format": "pdf"},
        )

    def _extract_sections(self, lines: list[str]) -> list[PaperSection]:
        headings = {"abstract", "introduction", "method", "methods", "results", "conclusion"}
        sections: list[PaperSection] = []
        current_title = "Body"
        buffer: list[str] = []
        index = 1
        for line in lines[1:]:
            normalized = line.lower().strip(":")
            if normalized in headings and buffer:
                sections.append(
                    PaperSection(
                        section_id=f"section-{index}",
                        title=current_title,
                        content=" ".join(buffer),
                    )
                )
                index += 1
                current_title = line.strip(":")
                buffer = []
                continue
            if normalized in headings:
                current_title = line.strip(":")
                continue
            buffer.append(line)
        if buffer:
            sections.append(
                PaperSection(
                    section_id=f"section-{index}",
                    title=current_title,
                    content=" ".join(buffer),
                )
            )
        if sections:
            return sections
        return [
            PaperSection(
                section_id="section-1",
                title="Body",
                content=" ".join(lines[1:]),
            )
        ]

    @staticmethod
    def _extract_abstract(lines: list[str]) -> str:
        lowered = [line.lower() for line in lines]
        if "abstract" in lowered:
            start = lowered.index("abstract") + 1
            end = next(
                (
                    index
                    for index in range(start, len(lines))
                    if lowered[index]
                    in {"introduction", "method", "methods", "results", "conclusion"}
                ),
                min(start + 5, len(lines)),
            )
            return " ".join(lines[start:end]).strip()
        return " ".join(lines[1:5]).strip()
