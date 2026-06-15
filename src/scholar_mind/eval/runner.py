from __future__ import annotations

from scholar_mind.agents.common import try_structured_output
from scholar_mind.models.domain import AnswerGenerationOutput, RetrievalStrategyName
from scholar_mind.rag.top_k import RAG_EVAL_CONTEXT_TOP_K


class AnswerRunner:
    def __init__(self, rag_engine, llm=None):
        self.rag_engine = rag_engine
        self.llm = llm

    async def answer(
        self, query: str, strategy: RetrievalStrategyName, top_k: int = RAG_EVAL_CONTEXT_TOP_K
    ) -> tuple[str, list]:
        retrieved, _ = await self.rag_engine.retrieve(query, strategy=strategy, top_k=top_k)
        contexts = [item.content for item in retrieved]
        answer = self._answer_with_llm(query, contexts, top_k=top_k)
        if answer is None:
            raise RuntimeError("LLM answer generation failed for RAG evaluation")
        return answer, retrieved

    def _answer_with_llm(self, query: str, contexts: list[str], top_k: int) -> str | None:
        if self.llm is None:
            return None
        prompt = (
            "Answer the question using only the provided evidence. "
            "If the evidence is insufficient, say so clearly.\n\n"
            f"Question: {query}\nEvidence: {contexts[:top_k]}"
        )
        structured = try_structured_output(self.llm, prompt, AnswerGenerationOutput)
        if not structured or not structured.answer.strip():
            return None
        return structured.answer.strip()
