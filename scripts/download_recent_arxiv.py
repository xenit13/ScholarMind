#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import anyio

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholar_mind.config.settings import get_settings
from scholar_mind.pipeline.arxiv_storage import metadata_path, pdf_dir, preferred_category_name, source_dir
from scholar_mind.pipeline.downloader import (
    ArxivApiMetadataDownloader,
    ArxivMetadataDownloader,
    ArxivPdfDownloader,
    ArxivSourceDownloader,
)
from scholar_mind.pipeline.recent_ingest import normalize_requested_categories
from arxiv_run_logging import capture_run_output, resolve_run_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download arXiv metadata and source assets into the local raw data directory."
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=None,
        help="Number of recent papers to download.",
    )
    parser.add_argument(
        "--paper-id",
        "-p",
        action="append",
        default=[],
        help="Specific arXiv paper ID to download (repeatable).",
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        default=[],
        help="arXiv category such as cs.AI, cs.CL, cs.LG, or a broad prefix like cs.",
    )
    parser.add_argument(
        "--from-date",
        help="Start date in YYYY-MM-DD format. Only used with --count.",
    )
    parser.add_argument(
        "--to-date",
        help="End date in YYYY-MM-DD format. Only used with --count.",
    )
    return parser


def validate_date(date_str: str, param_name: str) -> None:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: {param_name} must be in YYYY-MM-DD format, got: {date_str}", file=sys.stderr)
        sys.exit(1)


def log_paper(path: Path, paper_id: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{paper_id}\n")


async def select_records(args) -> list[dict[str, object]]:
    metadata_downloader = ArxivMetadataDownloader()
    api_downloader = ArxivApiMetadataDownloader()

    if args.paper_id:
        records: list[dict[str, object]] = []
        for paper_id in dict.fromkeys(args.paper_id):
            try:
                record = await metadata_downloader.download_record(paper_id)
            except Exception:
                record = {"paper_id": paper_id}
            records.append(record)
        return records

    categories = normalize_requested_categories(args.category)
    return await api_downloader.download_recent(
        count=args.count,
        categories=categories or None,
        from_date=args.from_date,
        to_date=args.to_date,
    )


async def run(args) -> int:
    settings = get_settings()
    raw_root = settings.resolve_path(settings.raw_data_dir) / "arxiv"
    source_downloader = ArxivSourceDownloader()
    pdf_downloader = ArxivPdfDownloader()
    requested_categories = normalize_requested_categories(args.category)
    run_dir = resolve_run_dir(requested_categories)
    success_log = run_dir / "success_download.log"
    failed_log = run_dir / "failed_download.log"

    records = await select_records(args)
    total = len(records)
    print(f"Logs: {run_dir}")
    print(f"Selected {total} papers for download")
    print()

    failed = 0
    for index, record in enumerate(records, start=1):
        paper_id = str(record.get("paper_id", "")).strip()
        if not paper_id:
            continue
        title = str(record.get("title", "")).strip()
        print(f"[{index}/{total}] Downloading {paper_id}" + (f": {title}" if title else ""))
        try:
            category_name = preferred_category_name(record, requested_categories=requested_categories)
            if len(record) > 1:
                metadata_file = metadata_path(raw_root, paper_id, category_name)
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                metadata_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                artifact_path = await source_downloader.download_single(paper_id, source_dir(raw_root, category_name))
                source_format = "latex"
            except Exception:
                artifact_path = await pdf_downloader.download_single(paper_id, pdf_dir(raw_root, category_name))
                source_format = "pdf"
            log_paper(success_log, paper_id)
            print(f"  Done: [{category_name}] {source_format} -> {artifact_path}")
        except Exception as exc:
            failed += 1
            log_paper(failed_log, paper_id)
            print(f"  Failed: {exc}", file=sys.stderr)

    print()
    print(f"Finished: {total - failed}/{total} downloaded successfully")
    if failed:
        print(f"Failed paper IDs were appended to {failed_log}")
    return 1 if failed else 0


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(normalize_requested_categories(args.category))
    with capture_run_output(run_dir, argv=sys.argv) as _:
        if not args.paper_id and args.count is None:
            print("Error: specify --count or --paper-id", file=sys.stderr)
            sys.exit(1)

        if args.paper_id and args.count is not None:
            print("Error: --count and --paper-id are mutually exclusive", file=sys.stderr)
            sys.exit(1)

        if (args.from_date or args.to_date) and not args.count:
            print("Error: --from-date and --to-date require --count", file=sys.stderr)
            sys.exit(1)

        if args.from_date:
            validate_date(args.from_date, "--from-date")
        if args.to_date:
            validate_date(args.to_date, "--to-date")

        raise SystemExit(anyio.run(run, args))


if __name__ == "__main__":
    main()
