from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from langchain_core.messages import AIMessage, ToolMessage

import scholar_mind.app as app_module
from scholar_mind.app import get_container
from scholar_mind.config.settings import get_settings
from scholar_mind.models.domain import PlannerOutput, QueryType, ReviewerOutput


class _TestEmbeddingService:
    dimension = 16

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        vector = [0.0] * 16
        for token in text.lower().split():
            vector[hash(token) % 16] += 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class _FakeStructuredRunnable:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        if self.schema is PlannerOutput:
            query = _extract_after_label(prompt, "Query:")
            lowered = query.lower()
            if "cross" in lowered or "transfer" in lowered or "跨" in query:
                payload = PlannerOutput(
                    query_type=QueryType.CROSS_DOMAIN,
                    source_papers=["2405.00005"],
                    target_domains=["机器人控制"] if "机器人" in query else ["code generation"],
                )
            elif "trend" in lowered or "趋势" in query:
                payload = PlannerOutput(query_type=QueryType.TREND)
            elif "study" in lowered or "学习计划" in query:
                payload = PlannerOutput(query_type=QueryType.STUDY_PLAN)
            elif "paper" in lowered or "精读" in query:
                payload = PlannerOutput(query_type=QueryType.PAPER_READING)
            elif "novelty" in lowered or "新颖性" in query:
                payload = PlannerOutput(query_type=QueryType.IDEA_NOVELTY)
            else:
                payload = PlannerOutput(query_type=QueryType.QA)
            return {
                "parsed": payload,
                "raw": AIMessage(content=payload.model_dump_json()),
                "parsing_error": None,
            }

        if self.schema is ReviewerOutput:
            prompt_text = _prompt_to_text(prompt)
            draft = _extract_between(prompt_text, "Draft:", "\nCitations:") or "Grounded answer."
            payload = ReviewerOutput(final_answer=draft.strip(), review_score=0.92, notes="fake-review")
            return {
                "parsed": payload,
                "raw": AIMessage(content=payload.model_dump_json()),
                "parsing_error": None,
            }

        return {"parsed": None, "raw": AIMessage(content=""), "parsing_error": None}


class _FakeChatModel:
    def __init__(self, tool_names: set[str] | None = None):
        self.tool_names = tool_names or set()

    def bind_tools(self, tools):
        names = {
            getattr(tool, "name", None) or getattr(tool, "__name__", "")
            for tool in tools
        }
        return _FakeChatModel(names)

    def with_structured_output(self, schema, include_raw: bool = False):
        assert include_raw is True
        return _FakeStructuredRunnable(schema)

    def invoke(self, prompt):
        if isinstance(prompt, list):
            return self._invoke_messages(prompt)
        return AIMessage(content="已生成一份简明结果。")

    def _invoke_messages(self, messages):
        system_prompt = str(getattr(messages[0], "content", "")) if messages else ""
        user_context = str(getattr(messages[1], "content", "")) if len(messages) > 1 else ""
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]

        if "Use the available retrieval tools" in system_prompt:
            if tool_messages:
                return AIMessage(
                    content="Hybrid retrieval improves recall by combining lexical and semantic evidence."
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_retrieve",
                        "args": {"query": _extract_after_label(user_context, "Retrieval query:")},
                        "id": "call_research",
                        "type": "tool_call",
                    }
                ],
            )

        if "analytics tools" in system_prompt:
            if tool_messages:
                return AIMessage(content="Trend summary grounded in the gathered statistics.")
            topic = _extract_after_label(user_context, "Topic:")
            keywords = _extract_list_after_label(user_context, "Suggested keywords:")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "paper_count_stats",
                        "args": {"topic": topic, "granularity": "quarterly"},
                        "id": "call_count",
                        "type": "tool_call",
                    },
                    {
                        "name": "keyword_trend_stats",
                        "args": {"keywords": keywords or [topic]},
                        "id": "call_keywords",
                        "type": "tool_call",
                    },
                    {
                        "name": "paper_search",
                        "args": {"query": topic, "page_size": 5},
                        "id": "call_search",
                        "type": "tool_call",
                    },
                ],
            )

        if "rag_top10_similar_papers" in system_prompt:
            if tool_messages:
                return AIMessage(
                    content='{"source_method_summary":"retrieval-grounded planning","candidates":[{"paper_id":"2401.00001","methodology_similarity":0.82,"transfer_rationale":"retrieval planning transfers well"}]}'
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_top10_similar_papers",
                        "args": {
                            "source_summary": _extract_after_label(
                                user_context, "Source methodology summary:"
                            ),
                            "target_domains": _extract_list_after_label(
                                user_context, "Target domains:"
                            ),
                            "exclude_paper_ids": [],
                            "exclude_primary_categories": [],
                            "strategy": "hybrid",
                        },
                        "id": "call_cross",
                        "type": "tool_call",
                    }
                ],
            )

        if "You are drafting an idea novelty report." in system_prompt:
            if tool_messages:
                return AIMessage(
                    content="The idea partially overlaps with retrieval-grounded planning work but remains novel in its target application."
                )
            paper_ids = _extract_list_after_label(user_context, "Candidate paper ids:")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "citation_lookup",
                        "args": {"paper_ids": paper_ids or ["2401.00001"]},
                        "id": "call_citation_idea",
                        "type": "tool_call",
                    }
                ],
            )

        if "Write a concise cross-domain transfer report in Chinese." in system_prompt:
            if tool_messages:
                return AIMessage(content="可以把检索增强规划迁移到机器人控制，但需要验证状态空间差异。")
            paper_ids = _extract_list_after_label(user_context, "Reference paper ids:")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "citation_lookup",
                        "args": {"paper_ids": paper_ids or ["2401.00001"]},
                        "id": "call_citation_cross",
                        "type": "tool_call",
                    }
                ],
            )

        if "Each hypothesis must combine the source method" in system_prompt:
            return AIMessage(
                content='{"hypotheses":[{"hypothesis":"将检索增强规划迁移到代码生成任务中以提升步骤一致性。","candidate_paper_ids":["2401.00001"],"novelty_is_novel":true,"novelty_confidence":0.78,"novelty_rationale":"当前样例语料里没有直接相同的迁移方案。","target_domain":"code generation","core_intervention":"retrieval-grounded planning loop","datasets_or_tasks":["HumanEval"],"baselines":["direct generation"],"metrics":["pass@1"],"ablations":["without retrieval"]}]}'
            )

        return AIMessage(content="已生成一份简明结果。")


def _extract_after_label(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _extract_list_after_label(text: str, label: str) -> list[str]:
    raw = _extract_after_label(text, label)
    if raw.startswith("[") and raw.endswith("]"):
        items = raw[1:-1].split(",")
        return [item.strip().strip("'\"") for item in items if item.strip()]
    return []


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(getattr(message, "content", "")) for message in prompt)
    return str(prompt)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_RERANKER_PROVIDER", raising=False)
    monkeypatch.delenv("SCHOLARMIND_RERANKER_MODEL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_RERANKER_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_RERANKER_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_RERANKER_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("SCHOLARMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("SCHOLARMIND_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv(
        "SCHOLARMIND_CHECKPOINT_DATABASE_URL", f"sqlite:///{tmp_path / 'checkpoints.db'}"
    )
    monkeypatch.setenv("SCHOLARMIND_QDRANT_LOCATION", ":memory:")
    monkeypatch.setenv(
        "SCHOLARMIND_PAPERS_SEED_PATH", str(root / "data/processed/sample_papers.json")
    )
    monkeypatch.setenv("SCHOLARMIND_LOG_DIR", str(tmp_path / "message_logs"))
    monkeypatch.setenv("SCHOLARMIND_MEMORY_ROOT_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("SCHOLARMIND_EVAL_ROOT_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("SCHOLARMIND_RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("SCHOLARMIND_PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(
        app_module,
        "build_chat_models",
        lambda _settings: {"reasoning": _FakeChatModel(), "light": _FakeChatModel()},
    )
    monkeypatch.setattr(app_module, "build_embedding_service", lambda _settings: _TestEmbeddingService())
    get_settings.cache_clear()
    get_container.cache_clear()
    yield
    if get_container.cache_info().currsize:
        anyio.run(get_container().aclose)
    get_settings.cache_clear()
    get_container.cache_clear()
