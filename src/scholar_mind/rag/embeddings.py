from __future__ import annotations

import httpx
from typing import Protocol

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional at runtime
    OpenAIEmbeddings = None


class EmbeddingService(Protocol):
    dimension: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingService:
    def __init__(self, model: str, base_url: str, api_key: str, dimension: int):
        if OpenAIEmbeddings is None:  # pragma: no cover - import guard
            raise RuntimeError("langchain-openai is required for OpenAI-compatible embeddings")
        self.dimension = dimension
        self.client = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)

    def embed_query(self, text: str) -> list[float]:
        return list(self.client.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(item) for item in self.client.embed_documents(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return list(await self.client.aembed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(item) for item in await self.client.aembed_documents(texts)]


class RemoteEmbeddingService:
    def __init__(
        self,
        model: str,
        base_url: str,
        dimension: int,
        api_key: str | None = None,
        request_dimensions: int | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.api_key = api_key
        self.request_dimensions = request_dimensions
        self.timeout_seconds = timeout_seconds

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self._endpoint(),
            json=self._payload(texts),
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self._parse_embeddings(response.json())

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self._endpoint(),
                json=self._payload(texts),
                headers=self._headers(),
            )
        response.raise_for_status()
        return self._parse_embeddings(response.json())

    def _endpoint(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, texts: list[str]) -> dict[str, object]:
        payload: dict[str, object] = {"input": texts}
        if self.model:
            payload["model"] = self.model
        if self.request_dimensions is not None:
            payload["dimensions"] = self.request_dimensions
        return payload

    @staticmethod
    def _parse_embeddings(payload: dict) -> list[list[float]]:
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and all(isinstance(item, list) for item in embeddings):
            return [[float(value) for value in item] for item in embeddings]

        rows = payload.get("data")
        if isinstance(rows, list):
            parsed: list[list[float]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                embedding = row.get("embedding")
                if isinstance(embedding, list):
                    parsed.append([float(value) for value in embedding])
            if parsed:
                return parsed

        raise ValueError("Unsupported embedding response payload")
