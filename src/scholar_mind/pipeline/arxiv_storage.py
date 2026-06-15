from __future__ import annotations

from pathlib import Path

DEFAULT_CATEGORY = "uncategorized"


def normalize_category_name(category: str | None) -> str:
    value = (category or "").strip()
    return value or DEFAULT_CATEGORY


def preferred_category_name(
    metadata: dict[str, object] | None,
    *,
    requested_categories: list[str] | None = None,
    category_override: str | None = None,
) -> str:
    if category_override:
        return normalize_category_name(category_override)
    if requested_categories and len(requested_categories) == 1:
        return normalize_category_name(requested_categories[0])
    if metadata is not None:
        raw_categories = metadata.get("categories")
        if isinstance(raw_categories, list):
            for item in raw_categories:
                if isinstance(item, str) and item.strip():
                    return normalize_category_name(item)
    return DEFAULT_CATEGORY


def metadata_path(raw_root: Path, paper_id: str, category: str) -> Path:
    return raw_root / "metadata" / normalize_category_name(category) / f"{paper_id}.json"


def source_dir(raw_root: Path, category: str) -> Path:
    return raw_root / "source" / normalize_category_name(category)


def pdf_dir(raw_root: Path, category: str) -> Path:
    return raw_root / "pdf" / normalize_category_name(category)


def find_metadata_path(raw_root: Path, paper_id: str, category: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if category:
        candidates.append(metadata_path(raw_root, paper_id, category))
    candidates.append(raw_root / "metadata" / f"{paper_id}.json")
    candidates.extend(sorted((raw_root / "metadata").glob(f"*/{paper_id}.json")))
    for path in candidates:
        if path.exists():
            return path
    return None


def find_local_artifact(raw_root: Path, paper_id: str, category: str | None = None) -> tuple[Path, str] | None:
    source_candidates: list[Path] = []
    pdf_candidates: list[Path] = []

    if category:
        normalized = normalize_category_name(category)
        source_candidates.extend(
            [
                raw_root / "source" / normalized / f"{paper_id}.tar.gz",
                raw_root / "source" / normalized / f"{paper_id}.tex",
                raw_root / "source" / normalized / f"{paper_id}.json",
            ]
        )
        pdf_candidates.append(raw_root / "pdf" / normalized / f"{paper_id}.pdf")

    source_candidates.extend(
        [
            raw_root / "source" / f"{paper_id}.tar.gz",
            raw_root / "source" / f"{paper_id}.tex",
            raw_root / "source" / f"{paper_id}.json",
        ]
    )
    source_candidates.extend(sorted((raw_root / "source").glob(f"*/{paper_id}.tar.gz")))
    source_candidates.extend(sorted((raw_root / "source").glob(f"*/{paper_id}.tex")))
    source_candidates.extend(sorted((raw_root / "source").glob(f"*/{paper_id}.json")))
    pdf_candidates.append(raw_root / "pdf" / f"{paper_id}.pdf")
    pdf_candidates.extend(sorted((raw_root / "pdf").glob(f"*/{paper_id}.pdf")))

    for path in source_candidates:
        if path.exists():
            return path, "latex"
    for path in pdf_candidates:
        if path.exists():
            return path, "pdf"
    return None


def list_category_paper_ids(raw_root: Path, category: str) -> list[str]:
    normalized = normalize_category_name(category)
    paper_ids: list[str] = []
    for path in sorted((raw_root / "metadata" / normalized).glob("*.json")):
        paper_id = path.stem
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
    for path in sorted((raw_root / "source" / normalized).glob("*.tar.gz")):
        paper_id = path.name[:-7]
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
    for path in sorted((raw_root / "source" / normalized).glob("*.tex")):
        paper_id = path.stem
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
    for path in sorted((raw_root / "source" / normalized).glob("*.json")):
        paper_id = path.stem
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
    for path in sorted((raw_root / "pdf" / normalized).glob("*.pdf")):
        paper_id = path.stem
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
    return paper_ids
