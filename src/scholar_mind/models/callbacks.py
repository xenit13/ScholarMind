from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


class UsageTracker:
    def __init__(self):
        self.usage = ModelUsage()

    def record(
        self, *, prompt_tokens: int = 0, 
        completion_tokens: int = 0, 
        total_tokens: int = 0, 
        latency_ms: int = 0
    ) -> None:
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += total_tokens
        self.usage.latency_ms += latency_ms

    def snapshot(self) -> ModelUsage:
        return ModelUsage(
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
            latency_ms=self.usage.latency_ms,
        )


def usage_dict(usage: ModelUsage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": usage.latency_ms,
    }
