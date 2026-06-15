from __future__ import annotations

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency at runtime
    tiktoken = None


def _get_encoding(name: str):
    if tiktoken is None:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("tiktoken unavailable")
    return tiktoken.get_encoding(name)


def _normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return ""
    return model_name.rsplit("/", 1)[-1].strip().lower()


def _candidate_encodings(model_name: str | None) -> list[str]:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return ["cl100k_base"]
    if normalized.startswith(("gpt-5", "gpt-4o", "o1", "o3", "o4")):
        return ["o200k_base", "cl100k_base"]
    if normalized.startswith(("gpt-4", "gpt-3.5", "text-embedding-3")):
        return ["cl100k_base"]
    return ["cl100k_base"]


def estimate_text_tokens(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0

    encoding_name = _candidate_encodings(model_name)[0]
    encoding = _get_encoding(encoding_name)
    return len(encoding.encode(text))
