from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


class ArxivMetadataDownloader:
    """Download and parse arXiv OAI-PMH metadata pages."""

    OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def download_initial(self, max_papers: int = 100_000) -> list[dict[str, Any]]:
        return await self._download(
            {"verb": "ListRecords", "metadataPrefix": "arXiv"},
            limit=max_papers,
        )

    async def download_incremental(self, since: datetime) -> list[dict[str, Any]]:
        return await self._download(
            {
                "verb": "ListRecords",
                "metadataPrefix": "arXiv",
                "from": since.date().isoformat(),
            },
            limit=None,
        )

    async def download_record(self, paper_id: str) -> dict[str, Any]:
        xml_text = await self._fetch_page(
            {
                "verb": "GetRecord",
                "identifier": f"oai:arXiv.org:{paper_id}",
                "metadataPrefix": "arXiv",
            }
        )
        records, _ = self._parse_response(xml_text)
        for record in records:
            if record:
                return record
        raise ValueError(f"No metadata record found for {paper_id}")

    async def _download(self, params: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        next_params = dict(params)
        while next_params:
            xml_text = await self._fetch_page(next_params)
            page_records, token = self._parse_response(xml_text)
            records.extend(page_records)
            if limit is not None and len(records) >= limit:
                return records[:limit]
            next_params = {"verb": "ListRecords", "resumptionToken": token} if token else {}
        return records

    async def _fetch_page(self, params: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {"timeout": 120, "follow_redirects": True}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.get(self.OAI_ENDPOINT, params=params)
            response.raise_for_status()
        return response.text

    def _parse_response(self, xml_text: str) -> tuple[list[dict[str, Any]], str | None]:
        root = ElementTree.fromstring(xml_text)
        records = [self._parse_record(node) for node in root.findall(".//oai:record", OAI_NS)]
        token_node = root.find(".//oai:resumptionToken", OAI_NS)
        token = token_node.text.strip() if token_node is not None and token_node.text else None
        return records, token

    def _parse_record(self, record: ElementTree.Element) -> dict[str, Any]:
        metadata = record.find("oai:metadata/arxiv:arXiv", OAI_NS)
        if metadata is None:
            return {}
        authors = []
        for author in metadata.findall("arxiv:authors/arxiv:author", OAI_NS):
            keyname = author.findtext("arxiv:keyname", default="", namespaces=OAI_NS)
            forenames = author.findtext("arxiv:forenames", default="", namespaces=OAI_NS)
            authors.append(" ".join(part for part in [forenames, keyname] if part).strip())
        categories = metadata.findtext("arxiv:categories", default="", namespaces=OAI_NS).split()
        return {
            "paper_id": metadata.findtext("arxiv:id", default="", namespaces=OAI_NS),
            "title": " ".join(
                metadata.findtext("arxiv:title", default="", namespaces=OAI_NS).split()
            ),
            "abstract": " ".join(
                metadata.findtext("arxiv:abstract", default="", namespaces=OAI_NS).split()
            ),
            "authors": authors,
            "categories": categories,
            "created": metadata.findtext("arxiv:created", default="", namespaces=OAI_NS),
            "updated": metadata.findtext("arxiv:updated", default="", namespaces=OAI_NS),
        }


ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


ALL_ARXIV_ARCHIVES = [
    "astro-ph", "cond-mat", "cs", "econ", "eess",
    "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math", "math-ph", "nlin", "nucl-ex", "nucl-th",
    "physics", "q-bio", "q-fin", "quant-ph", "stat",
]


class ArxivApiMetadataDownloader:
    """Download recent arXiv papers via the arXiv search API (faster than OAI-PMH)."""

    API_ENDPOINT = "http://export.arxiv.org/api/query"

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def download_recent(
        self,
        count: int,
        categories: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        page_size = 2000
        all_records: list[dict[str, Any]] = []
        cat_query = (
            " OR ".join(f"cat:{cat}" for cat in categories)
            if categories
            else " OR ".join(f"cat:{a}" for a in ALL_ARXIV_ARCHIVES)
        )
        query_parts: list[str] = [f"({cat_query})"]
        if from_date or to_date:
            from_str = from_date.replace("-", "") + "000000" if from_date else "00000000000000"
            to_str = to_date.replace("-", "") + "235959" if to_date else "99991231235959"
            query_parts.append(f"submittedDate:[{from_str} TO {to_str}]")
        search_query = " AND ".join(query_parts)

        kwargs: dict[str, Any] = {"timeout": 60, "follow_redirects": True}
        if self.proxy:
            kwargs["proxy"] = self.proxy

        async with httpx.AsyncClient(**kwargs) as client:
            start = 0
            while len(all_records) < count:
                params: dict[str, Any] = {
                    "search_query": search_query,
                    "start": start,
                    "max_results": min(page_size, count - len(all_records)),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                response = await client.get(self.API_ENDPOINT, params=params)
                response.raise_for_status()
                page_records = self._parse_atom_response(response.text)
                if not page_records:
                    break
                all_records.extend(page_records)
                start += len(page_records)
                if len(page_records) < params["max_results"]:
                    break
        return all_records[:count]

    def _parse_atom_response(self, xml_text: str) -> list[dict[str, Any]]:
        root = ElementTree.fromstring(xml_text)
        records: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            raw_id = entry.findtext("atom:id", "", ATOM_NS)
            paper_id = raw_id.rsplit("/", 1)[-1] if "/" in raw_id else raw_id
            if "v" in paper_id:
                paper_id = paper_id.rsplit("v", 1)[0]

            authors = [
                a.findtext("atom:name", "", ATOM_NS)
                for a in entry.findall("atom:author", ATOM_NS)
            ]
            categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", ATOM_NS)
            ]
            published = entry.findtext("atom:published", "", ATOM_NS)[:10]
            updated = entry.findtext("atom:updated", "", ATOM_NS)[:10]

            records.append({
                "paper_id": paper_id,
                "title": " ".join(entry.findtext("atom:title", "", ATOM_NS).split()),
                "abstract": " ".join(entry.findtext("atom:summary", "", ATOM_NS).split()),
                "authors": authors,
                "categories": categories,
                "created": published,
                "updated": updated,
            })
        return records


class ArxivSourceDownloader:
    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def download_single(self, paper_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{paper_id}.tar.gz"
        if path.exists():
            return path
        kwargs: dict[str, Any] = {"timeout": 60, "follow_redirects": True}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.get(f"https://arxiv.org/e-print/{paper_id}")
            response.raise_for_status()
        path.write_bytes(response.content)
        return path

    async def download_many(self, paper_ids: list[str], output_dir: Path) -> list[Path]:
        paths = []
        for paper_id in paper_ids:
            paths.append(await self.download_single(paper_id, output_dir))
        return paths


class ArxivPdfDownloader:
    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def download_single(self, paper_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{paper_id}.pdf"
        if path.exists():
            return path
        kwargs: dict[str, Any] = {"timeout": 60, "follow_redirects": True}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.get(f"https://arxiv.org/pdf/{paper_id}.pdf")
            response.raise_for_status()
        path.write_bytes(response.content)
        return path

    async def download_many(self, paper_ids: list[str], output_dir: Path) -> list[Path]:
        paths = []
        for paper_id in paper_ids:
            paths.append(await self.download_single(paper_id, output_dir))
        return paths
