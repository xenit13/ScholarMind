from __future__ import annotations

import json
from math import ceil

from langchain_core.messages import BaseMessage, SystemMessage

from scholar_mind.agents.common import empty_usage, invoke_structured_output
from scholar_mind.eval.context import get_eval_context, record_memory_event
from scholar_mind.models.domain import CompressionOutput
from scholar_mind.models.eval_models import MemoryCallEvent, MemoryOperation
from scholar_mind.utils.messages import serialize_messages


class MessageCompressor:
    def __init__(
        self,
        *,
        context_window_tokens: int = 32768,
        compact_threshold_ratio: float = 0.75,
        llm=None,
    ):
        self.context_window_tokens = context_window_tokens
        self.compact_threshold_ratio = compact_threshold_ratio
        self.llm = llm

    def compress(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        compressed, _ = self.compress_with_usage(messages)
        return compressed

    def compress_with_usage(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], dict[str, float]]:
        if not messages:
            return messages, empty_usage()

        threshold_tokens = int(self.context_window_tokens * self.compact_threshold_ratio)
        total_tokens = self._estimate_messages_tokens(messages)
        if total_tokens < threshold_tokens:
            return messages, empty_usage()

        reserve_tokens = max(
            int(self.context_window_tokens * (1 - self.compact_threshold_ratio)),
            1024,
        )
        recent_budget = max(threshold_tokens - reserve_tokens, threshold_tokens // 2)

        older, recent = self._split_messages_for_compaction(messages, recent_budget)
        if not older:
            return messages, empty_usage()

        snippets = self._build_snippets(older)
        if not snippets:
            return recent, empty_usage()

        summary, usage = self._summarize(snippets)
        compressed_messages = [SystemMessage(content=summary)] + recent
        compressed_tokens = self._estimate_messages_tokens(compressed_messages)

        ctx = get_eval_context()
        if ctx is not None:
            record_memory_event(
                MemoryCallEvent(
                    request_id=ctx.request_id,
                    operation=MemoryOperation.CONVERSATION_COMPRESS,
                    latency_ms=int(usage.get("latency_ms", 0)),
                    compression_before_tokens=total_tokens,
                    compression_after_tokens=compressed_tokens,
                )
            )

        return compressed_messages, usage

    def _split_messages_for_compaction(
        self, messages: list[BaseMessage], recent_budget: int
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        kept: list[BaseMessage] = []
        kept_tokens = 0

        for message in reversed(messages):
            message_tokens = self._estimate_messages_tokens([message])
            if kept and kept_tokens + message_tokens > recent_budget:
                break
            kept.append(message)
            kept_tokens += message_tokens

        recent = list(reversed(kept))
        older = messages[: len(messages) - len(recent)]
        return older, recent

    def _build_snippets(self, messages: list[BaseMessage]) -> list[str]:
        snippets = []
        for message in messages:
            content = str(message.content).strip()
            if not content:
                continue
            message_type = getattr(message, "type", "message")
            snippets.append(f"{message_type}: {content[:160]}")
        return snippets

    def _estimate_messages_tokens(self, messages: list[BaseMessage]) -> int:
        if not messages:
            return 0

        if self.llm is not None and hasattr(self.llm, "get_num_tokens_from_messages"):
            try:
                return int(self.llm.get_num_tokens_from_messages(messages))
            except Exception:
                pass

        serialized = serialize_messages(messages)
        payload = json.dumps(serialized, ensure_ascii=False, default=str)
        return max(1, ceil((len(payload) / 4) * 1.2))

    def _summarize(self, snippets: list[str]) -> tuple[str, dict[str, float]]:
        prompt = (
            "Summarize the earlier conversation into one concise system-memory note. "
            "Preserve user preferences, prior questions, and unresolved follow-ups.\n\n"
            + "\n".join(f"- {snippet}" for snippet in snippets[:10])
        )
        structured, usage = invoke_structured_output(self.llm, prompt, CompressionOutput)
        if structured and structured.summary.strip():
            return structured.summary.strip(), usage
        return "Earlier context summary: " + " | ".join(snippets[:8]), usage
