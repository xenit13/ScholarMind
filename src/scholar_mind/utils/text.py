from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
BM25_K1 = 1.5
BM25_B = 0.75
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "with",
}


@dataclass(slots=True)
class SparseCorpusStats:
    document_count: int
    average_length: float
    document_frequencies: dict[str, int]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stable_hash_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def stable_sparse_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647


def simple_hash_embedding(text: str, dim: int = 128) -> list[float]:
    vector = [0.0] * dim
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        index = stable_hash_index(token, dim)
        vector[index] += float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def overlap_score(query: str, content: str) -> float:
    q = set(tokenize(query))
    c = set(tokenize(content))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def top_keywords(text: str, limit: int = 5) -> list[str]:
    counts = Counter(token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2)
    return [token for token, _ in counts.most_common(limit)]


def build_sparse_corpus_stats(texts: list[str]) -> SparseCorpusStats:
    document_frequencies: dict[str, int] = Counter()
    total_length = 0
    for text in texts:
        tokens = tokenize(text)
        total_length += len(tokens)
        for token in set(tokens):
            document_frequencies[token] = document_frequencies.get(token, 0) + 1
    document_count = max(len(texts), 1)
    average_length = total_length / document_count if total_length else 1.0
    return SparseCorpusStats(
        document_count=document_count,
        average_length=average_length,
        document_frequencies=document_frequencies,
    )


def encode_sparse_text(
    text: str,
    stats: SparseCorpusStats,
    *,
    is_query: bool = False,
) -> tuple[list[int], list[float]]:
    counts = Counter(tokenize(text))
    if not counts:
        return [], []

    doc_length = sum(counts.values())
    weights: dict[int, float] = {}
    for token, tf in counts.items():
        df = stats.document_frequencies.get(token, 0)
        idf = math.log(1 + ((stats.document_count - df + 0.5) / (df + 0.5)))
        if is_query:
            score = idf * (1 + math.log(tf))
        else:
            denominator = tf + BM25_K1 * (
                1 - BM25_B + BM25_B * (doc_length / max(stats.average_length, 1e-6))
            )
            score = idf * ((tf * (BM25_K1 + 1)) / max(denominator, 1e-6))
        index = stable_sparse_index(token)
        weights[index] = weights.get(index, 0.0) + score

    ordered = sorted(weights.items())
    return [index for index, _ in ordered], [value for _, value in ordered]


def truncate(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
