"""Runs a single RAG evaluation case through retrieval and answer generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from scholar_mind.eval.runner import AnswerRunner
from scholar_mind.models.domain import RetrievalStrategyName
from scholar_mind.models.rag_eval_models import RagEvalCase


@dataclass(slots=True)
class RagEvalObservation:
    case: RagEvalCase
    strategy: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    retrieval_latency_ms: int
    generated_at: datetime


class RagEvalRunner:
    def __init__(self, rag_engine, llm=None):
        self.rag_engine = rag_engine
        self.answer_runner = AnswerRunner(rag_engine, llm=llm)

    async def run_case(
        self,
        case: RagEvalCase,
        *,
        strategy: RetrievalStrategyName,
        top_k: int,
    ) -> RagEvalObservation:
        retrieved, latency = await self.rag_engine.retrieve(
            case.user_input,
            strategy=strategy,
            top_k=top_k,
        )
        if self.answer_runner.llm is None:
            raise RuntimeError("LLM is required for RAG evaluation answer generation")
        contexts = [item.content for item in retrieved]
        answer = await asyncio.to_thread(
            self.answer_runner._answer_with_llm,
            case.user_input,
            contexts,
            top_k,
        )
        if answer is None:
            raise RuntimeError("LLM answer generation failed for RAG evaluation")
        return RagEvalObservation(
            case=case,
            strategy=strategy.value,
            user_input=case.user_input,
            response=answer,
            retrieved_contexts=contexts,
            retrieved_chunk_ids=[item.chunk_id for item in retrieved],
            retrieval_latency_ms=latency,
            generated_at=datetime.now(UTC),
        )
