from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol


DEFAULT_LOOKBACK_WINDOWS_DAYS = (7, 30, 90, 180, 365, 730, 1825)


class MetadataDownloader(Protocol):
    async def download_incremental(self, since: datetime) -> list[dict[str, Any]]: ...


class PaperIngestor(Protocol):
    async def ingest_paper(self, paper_id: str) -> dict[str, Any]: ...


def normalize_requested_categories(categories: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_value in categories:
        for item in raw_value.split(","):
            category = item.strip()
            if category and category not in normalized:
                normalized.append(category)
    return normalized


def category_matches(record_categories: list[str], requested_categories: list[str]) -> bool:
    if not requested_categories:
        return True
    for requested in requested_categories:
        for actual in record_categories:
            if actual == requested or actual.startswith(f"{requested}."):
                return True
    return False


def record_recency_key(record: dict[str, Any]) -> tuple[datetime, datetime, str]:
    created = _parse_record_datetime(record.get("created"))
    updated = _parse_record_datetime(record.get("updated"))
    baseline = datetime.min.replace(tzinfo=UTC)
    primary = created or updated or baseline
    secondary = updated or created or baseline
    return primary, secondary, str(record.get("paper_id", ""))


def _parse_record_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


class RecentArxivBatchIngestor:
    def __init__(
        self,
        metadata_downloader: MetadataDownloader,
        paper_ingestor: PaperIngestor,
        *,
        api_metadata_downloader=None,
        lookback_windows_days: tuple[int, ...] = DEFAULT_LOOKBACK_WINDOWS_DAYS,
        now_fn=None,
    ):
        self.metadata_downloader = metadata_downloader
        self.paper_ingestor = paper_ingestor
        self.api_metadata_downloader = api_metadata_downloader
        self.lookback_windows_days = lookback_windows_days
        self.now_fn = now_fn or (lambda: datetime.now(UTC))

    async def select_recent_records(
        self,
        *,
        count: int,
        categories: list[str],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        if count <= 0:
            raise ValueError("count must be greater than 0")
        requested_categories = normalize_requested_categories(categories)
        selected_by_id: dict[str, dict[str, Any]] = {}
        windows_used: list[int] = []

        for window_days in self.lookback_windows_days:
            since = self.now_fn() - timedelta(days=window_days)
            windows_used.append(window_days)
            records = await self.metadata_downloader.download_incremental(since)
            for record in records:
                paper_id = str(record.get("paper_id", "")).strip()
                if not paper_id:
                    continue
                if not category_matches(record.get("categories", []), requested_categories):
                    continue
                current = selected_by_id.get(paper_id)
                if current is None or record_recency_key(record) > record_recency_key(current):
                    selected_by_id[paper_id] = record
            if len(selected_by_id) >= count:
                break

        selected = sorted(selected_by_id.values(), key=record_recency_key, reverse=True)
        return selected[:count], windows_used

    async def ingest_recent(
        self,
        *,
        count: int,
        categories: list[str],
    ) -> dict[str, Any]:
        collected: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        async for event in self.ingest_recent_stream(count=count, categories=categories):
            if event["type"] == "complete":
                return event["result"]
            if event["type"] == "paper_ingested":
                collected.append(event["paper"])
            if event["type"] == "metadata_downloaded":
                meta.update(event)
                del meta["type"]
        return {
            "requested_count": count,
            "selected_count": meta.get("selected_count", len(collected)),
            "ingested_count": len(collected),
            "categories": meta.get("categories", []),
            "lookback_windows_days": meta.get("lookback_windows_days", []),
            "selected_papers": meta.get("selected_papers", []),
            "ingested_papers": collected,
        }

    async def ingest_recent_stream(
        self,
        *,
        count: int,
        categories: list[str],
        from_date: str | None = None,
        to_date: str | None = None,
    ):
        requested_categories = normalize_requested_categories(categories)

        # Fast path: use arXiv search API (sorted by date, no time-window limit)
        if self.api_metadata_downloader is not None:
            yield {
                "type": "status",
                "step": "selecting",
                "message": "Downloading arXiv metadata via API...",
                "categories": requested_categories,
                "count": count,
            }
            selected_records = await self.api_metadata_downloader.download_recent(
                count=count,
                categories=requested_categories or None,
                from_date=from_date,
                to_date=to_date,
            )
            windows_used = []
        else:
            yield {
                "type": "status",
                "step": "selecting",
                "message": f"Downloading arXiv metadata (lookback up to {self.lookback_windows_days[0]} days)...",
                "categories": requested_categories,
                "count": count,
            }
            selected_records, windows_used = await self.select_recent_records(
                count=count,
                categories=requested_categories,
            )
        yield {
            "type": "metadata_downloaded",
            "step": "selected",
            "message": f"Selected {len(selected_records)} papers"
                + (f" from {windows_used[-1]}-day window(s)" if windows_used else " (sorted by date, no time limit)"),
            "selected_count": len(selected_records),
            "categories": requested_categories,
            "lookback_windows_days": windows_used,
            "selected_papers": [
                {
                    "paper_id": r["paper_id"],
                    "title": r.get("title", ""),
                    "categories": r.get("categories", []),
                    "created": r.get("created"),
                    "updated": r.get("updated"),
                }
                for r in selected_records
            ],
        }
        ingested: list[dict[str, Any]] = []
        for index, record in enumerate(selected_records, start=1):
            paper_id = record["paper_id"]
            title = record.get("title", "")
            yield {
                "type": "paper_start",
                "step": "ingesting",
                "index": index,
                "total": len(selected_records),
                "paper_id": paper_id,
                "title": title,
                "message": f"[{index}/{len(selected_records)}] Ingesting {paper_id}: {title}",
            }
            try:
                result = await self.paper_ingestor.ingest_paper(paper_id)
                ingested.append(result)
                yield {
                    "type": "paper_ingested",
                    "step": "ingesting",
                    "index": index,
                    "total": len(selected_records),
                    "paper_id": paper_id,
                    "title": title,
                    "paper": result,
                    "message": f"[{index}/{len(selected_records)}] Done: {paper_id} ({result.get('source_format', '?')}, {result.get('chunk_count', 0)} chunks)",
                }
            except Exception as exc:
                yield {
                    "type": "paper_failed",
                    "step": "ingesting",
                    "index": index,
                    "total": len(selected_records),
                    "paper_id": paper_id,
                    "title": title,
                    "error": str(exc),
                    "message": f"[{index}/{len(selected_records)}] Failed: {paper_id} — {exc}",
                }
        yield {
            "type": "complete",
            "step": "done",
            "result": {
                "requested_count": count,
                "selected_count": len(selected_records),
                "ingested_count": len(ingested),
                "categories": requested_categories,
                "lookback_windows_days": windows_used,
                "selected_papers": [
                    {
                        "paper_id": r["paper_id"],
                        "title": r.get("title", ""),
                        "categories": r.get("categories", []),
                        "created": r.get("created"),
                        "updated": r.get("updated"),
                    }
                    for r in selected_records
                ],
                "ingested_papers": ingested,
            },
            "message": f"All done: {len(ingested)}/{len(selected_records)} ingested successfully",
        }

    async def ingest_papers_stream(self, *, paper_ids: list[str]):
        """Ingest specific paper IDs with SSE streaming, continuing on individual failures."""
        total = len(paper_ids)
        ingested: list[dict[str, Any]] = []

        yield {
            "type": "metadata_downloaded",
            "step": "selected",
            "message": f"Selected {total} papers to ingest",
            "selected_count": total,
            "categories": [],
            "lookback_windows_days": [],
            "selected_papers": [
                {"paper_id": pid, "title": "", "categories": [], "created": None, "updated": None}
                for pid in paper_ids
            ],
        }

        for index, paper_id in enumerate(paper_ids, start=1):
            yield {
                "type": "paper_start",
                "step": "ingesting",
                "index": index,
                "total": total,
                "paper_id": paper_id,
                "title": "",
                "message": f"[{index}/{total}] Ingesting {paper_id}",
            }
            try:
                result = await self.paper_ingestor.ingest_paper(paper_id)
                ingested.append(result)
                yield {
                    "type": "paper_ingested",
                    "step": "ingesting",
                    "index": index,
                    "total": total,
                    "paper_id": paper_id,
                    "title": result.get("title", ""),
                    "paper": result,
                    "message": f"[{index}/{total}] Done: {paper_id} ({result.get('source_format', '?')}, {result.get('chunk_count', 0)} chunks)",
                }
            except Exception as exc:
                yield {
                    "type": "paper_failed",
                    "step": "ingesting",
                    "index": index,
                    "total": total,
                    "paper_id": paper_id,
                    "title": "",
                    "error": str(exc),
                    "message": f"[{index}/{total}] Failed: {paper_id} — {exc}",
                }

        yield {
            "type": "complete",
            "step": "done",
            "result": {
                "requested_count": total,
                "selected_count": total,
                "ingested_count": len(ingested),
                "categories": [],
                "lookback_windows_days": [],
                "selected_papers": [
                    {"paper_id": pid, "title": "", "categories": [], "created": None, "updated": None}
                    for pid in paper_ids
                ],
                "ingested_papers": ingested,
            },
            "message": f"All done: {len(ingested)}/{total} ingested successfully",
        }
