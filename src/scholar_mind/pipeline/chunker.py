from __future__ import annotations

import re

from scholar_mind.models.domain import ChunkType, PaperChunk, StructuredPaper

# Minimum number of consecutive lines that look table-like to be
# considered an actual table block.
_TABLE_MIN_LINES = 2

# Regex for markdown-style separator rows: | --- | --- | or +------+------+ etc.
_TABLE_SEP_RE = re.compile(r"^[\s|+:-]+$")


class StructureAwareChunker:
    """Structure-aware chunker with configurable overlap."""

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 80):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, paper: StructuredPaper) -> list[PaperChunk]:
        chunks = [self._metadata_chunk(paper)]
        for section in paper.sections:
            # 1. Extract structured chunks (FORMULA, TABLE, ALGORITHM) first
            structured = self._structured_chunks(
                paper.paper_id,
                section.section_id,
                section.title,
                section.content,
            )

            # 2. Strip structured blocks from prose so SECTION chunks don't duplicate them
            prose = self._strip_structured_blocks(section.content)

            # 3. Tokenize prose into words and create overlapping SECTION chunks
            words = prose.split()
            words = [w for w in words if w.strip()]
            if not words:
                chunks.extend(structured)
                continue

            part = 1
            start = 0
            while start < len(words):
                end = min(start + self.max_tokens, len(words))
                chunk_words = words[start:end]
                content_text = " ".join(chunk_words)
                chunks.append(
                    self._section_chunk(
                        paper.paper_id,
                        section.section_id,
                        section.title,
                        content_text, len(chunk_words), part,
                    )
                )
                part += 1
                if end >= len(words):
                    break
                # Advance by (max_tokens - overlap_tokens) for overlap
                start += self.max_tokens - self.overlap_tokens

            chunks.extend(structured)
        return chunks

    def _metadata_chunk(self, paper: StructuredPaper) -> PaperChunk:
        content = (
            f"[Paper: {paper.title}] "
            f"[Authors: {', '.join(paper.authors)}] "
            f"[Abstract] {paper.abstract}"
        )
        return PaperChunk(
            chunk_id=f"{paper.paper_id}::metadata",
            paper_id=paper.paper_id,
            chunk_type=ChunkType.METADATA,
            section="metadata",
            content=content,
            token_count=len(content.split()),
            metadata={"categories": paper.categories},
        )

    def _section_chunk(
        self,
        paper_id: str,
        section_id: str,
        section_title: str,
        content: str,
        token_count: int,
        part: int,
    ) -> PaperChunk:
        wrapped = f"[Section: {section_title}] {content.strip()}"
        section_key = self._section_key(section_id)
        return PaperChunk(
            chunk_id=f"{paper_id}::{section_key}::{part}",
            paper_id=paper_id,
            chunk_type=ChunkType.SECTION,
            section=section_title,
            content=wrapped,
            token_count=token_count,
        )

    def _structured_chunks(
        self,
        paper_id: str,
        section_id: str,
        section_title: str,
        content: str,
    ) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        section_key = self._section_key(section_id)
        for index, formula in enumerate(self._formula_blocks(content), start=1):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{paper_id}::{section_key}::formula::{index}",
                    paper_id=paper_id,
                    chunk_type=ChunkType.FORMULA,
                    section=section_title,
                    content=formula,
                    token_count=len(formula.split()),
                )
            )
        for index, table in enumerate(self._table_blocks(content), start=1):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{paper_id}::{section_key}::table::{index}",
                    paper_id=paper_id,
                    chunk_type=ChunkType.TABLE,
                    section=section_title,
                    content=table,
                    token_count=len(table.split()),
                )
            )
        for index, algorithm in enumerate(self._algorithm_blocks(content, section_title), start=1):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{paper_id}::{section_key}::algorithm::{index}",
                    paper_id=paper_id,
                    chunk_type=ChunkType.ALGORITHM,
                    section=section_title,
                    content=algorithm,
                    token_count=len(algorithm.split()),
                )
            )
        return chunks

    @staticmethod
    def _section_key(section_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", section_id.strip().lower())

    # ------------------------------------------------------------------
    # Helpers to strip structured blocks from prose
    # ------------------------------------------------------------------

    def _strip_structured_blocks(self, content: str) -> str:
        """Remove formula, table, and algorithm blocks from section content."""
        text = content
        text = re.sub(r"\$[^$]+\$", "", text)
        text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", text, flags=re.DOTALL)
        # Strip real table blocks (found by _table_blocks)
        for table_block in self._table_blocks(text):
            text = text.replace(table_block, "")
        return text

    # ------------------------------------------------------------------
    # Block extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _formula_blocks(content: str) -> list[str]:
        return [
            match.strip()
            for match in re.findall(
                r"\$[^$]+\$|\\begin\{equation\}.*?\\end\{equation\}",
                content,
                flags=re.DOTALL,
            )
            if match.strip()
        ]

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if a single line looks like a table row.

        A table row must contain at least two cells, while still
        rejecting incidental ``|`` characters (e.g. ``|x|``,
        ``f(x|y)``, ``\\bigl|``) in normal prose.
        """
        stripped = line.strip()
        if not stripped:
            return False

        # Pipe-delimited rows with at least two non-empty cells.
        if "|" in stripped:
            if "\\" in stripped or "{" in stripped or "}" in stripped:
                return False
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            nonempty_cells = [cell for cell in cells if cell]
            if len(nonempty_cells) >= 2 and (
                stripped.startswith("|")
                or stripped.endswith("|")
                or " | " in stripped
            ):
                return True

        # Markdown separator:  |---|---| or +------+------+ etc.
        if _TABLE_SEP_RE.match(stripped) and ("-" in stripped or "=" in stripped):
            return True

        # Tab-separated: at least 2 tabs
        if stripped.count("\t") >= 2:
            return True

        return False

    @staticmethod
    def _table_blocks(content: str) -> list[str]:
        """Extract contiguous blocks of table-like lines.

        A block is only returned when it contains at least
        ``_TABLE_MIN_LINES`` consecutive table rows, which filters out
        stray ``|`` characters in LaTeX prose.
        """
        tables: list[str] = []
        current: list[str] = []
        for line in content.splitlines():
            if StructureAwareChunker._is_table_row(line):
                current.append(line.rstrip())
                continue
            if len(current) >= _TABLE_MIN_LINES:
                tables.append("\n".join(current))
            current = []
        if len(current) >= _TABLE_MIN_LINES:
            tables.append("\n".join(current))
        return tables

    @staticmethod
    def _algorithm_blocks(content: str, section_title: str) -> list[str]:
        if "algorithm" in section_title.lower():
            return [content.strip()]
        blocks: list[str] = []
        for match in re.findall(
            r"(?:Algorithm[:\s].*?)(?=\n\n|\Z)|(?:Step\s+\d+.*?)(?=\n\n|\Z)",
            content,
            flags=re.DOTALL,
        ):
            if match.strip():
                blocks.append(match.strip())
        return blocks
