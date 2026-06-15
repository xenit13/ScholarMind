from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1 / math.log2(index + 1)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg / ideal
