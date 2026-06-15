#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from arxiv_run_logging import capture_run_output, resolve_run_dir

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_PAPER_ID_FILE = Path("success_download.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the local ScholarMind API to ingest downloaded arXiv assets."
    )
    parser.add_argument(
        "--paper-id",
        "-p",
        action="append",
        default=[],
        help="Specific arXiv paper ID to ingest from local files (repeatable).",
    )
    parser.add_argument(
        "--paper-id-file",
        type=Path,
        default=None,
        help=f"Read paper IDs from a newline-delimited file. Defaults to {DEFAULT_PAPER_ID_FILE} if present.",
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        default=[],
        help="Ingest all downloaded papers under one or more local category directories.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"ScholarMind API base URL (default: {DEFAULT_API_URL})",
    )
    return parser


def load_paper_ids(args) -> list[str]:
    paper_ids: list[str] = []
    for paper_id in args.paper_id:
        normalized = paper_id.strip()
        if normalized and normalized not in paper_ids:
            paper_ids.append(normalized)

    paper_id_file = args.paper_id_file
    if paper_id_file is None and not paper_ids and DEFAULT_PAPER_ID_FILE.exists():
        paper_id_file = DEFAULT_PAPER_ID_FILE

    if paper_id_file is not None:
        if not paper_id_file.exists():
            print(f"Error: paper ID file not found: {paper_id_file}", file=sys.stderr)
            sys.exit(1)
        for line in paper_id_file.read_text(encoding="utf-8").splitlines():
            normalized = line.strip()
            if normalized and normalized not in paper_ids:
                paper_ids.append(normalized)

    if not paper_ids and not args.category:
        print("Error: specify --paper-id, --paper-id-file, or --category", file=sys.stderr)
        sys.exit(1)

    return paper_ids


def parse_sse_lines(line_iter):
    event = "message"
    data_lines: list[str] = []
    for raw_line in line_iter:
        line = raw_line.rstrip("\n").rstrip("\r")
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
    if data_lines:
        yield event, "\n".join(data_lines)


def log_paper(path: Path, paper_id: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{paper_id}\n")


def process_sse_stream(resp, *, success_log: Path, failed_log: Path) -> int:
    failed = 0

    for event_type, raw_data in parse_sse_lines(resp.iter_lines()):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue

        message = data.get("message", "")
        if event_type == "metadata_downloaded":
            print(f"[1/2] {message}")
            print()
        elif event_type == "paper_start":
            print(f"  {message}")
        elif event_type == "paper_ingested":
            print(f"  {message}")
            paper_id = data.get("paper_id", "")
            if paper_id:
                log_paper(success_log, paper_id)
        elif event_type == "paper_failed":
            print(f"  {message}", file=sys.stderr)
            paper_id = data.get("paper_id", "")
            if paper_id:
                log_paper(failed_log, paper_id)
            failed += 1
        elif event_type == "complete":
            print()
            result = data.get("result", {})
            ingested = result.get("ingested_count", 0)
            selected = result.get("selected_count", 0)
            print(f"[2/2] Done: {ingested}/{selected} ingested successfully")

    return 1 if failed else 0


def run(*, base_url: str, paper_ids: list[str], categories: list[str]) -> int:
    url = f"{base_url.rstrip('/')}/api/v1/ingest/local/stream"
    body = {"paper_ids": paper_ids, "categories": categories}
    run_dir = resolve_run_dir(categories)
    success_log = run_dir / "success_ingest.log"
    failed_log = run_dir / "failed_ingest.log"
    print(f"Logs: {run_dir}")
    print(f"==> POST {url}")
    print(f"    paper_ids={paper_ids or '[]'}  categories={categories or '[]'}")
    print()

    try:
        with httpx.stream("POST", url, json=body, timeout=None) as resp:
            if resp.status_code >= 400:
                error_text = resp.read().decode("utf-8", errors="replace")
                if resp.status_code == 404:
                    print(
                        "Error: the running ScholarMind service does not expose "
                        "/api/v1/ingest/local/stream yet. Restart the API service so it loads the latest code.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Error: {resp.status_code} — {error_text}", file=sys.stderr)
                raise SystemExit(1)
            return process_sse_stream(resp, success_log=success_log, failed_log=failed_log)
    except httpx.ConnectError as exc:
        print(
            f"Error: cannot connect to {base_url}. "
            "Make sure ScholarMind is running (bash scripts/deploy.sh).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.category)
    with capture_run_output(run_dir, argv=sys.argv) as _:
        paper_ids = load_paper_ids(args)
        raise SystemExit(run(base_url=args.api_url, paper_ids=paper_ids, categories=args.category))


if __name__ == "__main__":
    main()
