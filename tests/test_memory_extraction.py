from __future__ import annotations

from langchain_core.messages import HumanMessage

from scholar_mind.memory.extraction import (
    _build_candidate_extraction_prompt,
    extract_memory_candidates_from_round,
)
from scholar_mind.models.domain import MemoryCandidate, MemoryCandidateExtractionOutput
from scholar_mind.utils.messages import serialize_messages


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, llm, payload):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return self.payload


class _StructuredOutputLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, self.payload)


def test_extract_memory_candidates_recovers_structured_json_payload():
    llm = _StructuredOutputLLM(
        {
            "parsed": None,
            "raw": _RawResult(
                "```json\n"
                "{"
                '"candidates": ['
                "{"
                '"memory_type": "preference",'
                '"content": "用户偏好简洁回答，关键结论需要带引用。",'
                '"structured": {"subject": "user", "predicate": "prefers"},'
                '"keywords": ["简洁", "引用"],'
                '"importance": 0.8,'
                '"confidence": 0.9,'
                '"source": "conversation",'
                '"evidence": [{"message_id": "s1-1-0", "role": "human"}]'
                "}"
                "]"
                "}\n"
                "```",
                {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
            ),
            "parsing_error": ValueError("invalid json"),
        }
    )
    round_messages = [
        {"message": serialize_messages([HumanMessage(content="以后回答请简洁，但结论带引用")])[0]}
    ]

    candidates, usage, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert usage["total_tokens"] == 15
    assert len(candidates) == 1
    assert candidates[0].memory_type == "preference"
    assert candidates[0].content == "用户偏好简洁回答，关键结论需要带引用。"
    assert candidates[0].keywords == ["简洁", "引用"]


def test_extract_memory_candidates_accepts_core_memory_types():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(
                candidates=[
                    MemoryCandidate(
                        memory_type="preference",
                        content="用户偏好中文回答。",
                        source="conversation",
                    ),
                    MemoryCandidate(
                        memory_type="research_interest",
                        content="用户关注长期记忆评测。",
                        source="conversation",
                    ),
                    MemoryCandidate(
                        memory_type="knowledge_level",
                        content="用户熟悉 Transformer 基础概念。",
                        source="conversation",
                    ),
                ]
            ),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [{"message": serialize_messages([HumanMessage(content="请记住这些偏好")])[0]}]

    candidates, usage, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert usage["total_tokens"] == 5
    assert [candidate.memory_type for candidate in candidates] == [
        "preference",
        "research_interest",
        "knowledge_level",
    ]


def test_candidate_extraction_prompt_documents_management_operations():
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="请忘记我的回答风格偏好")]
            )[0]
        }
    ]

    prompt = _build_candidate_extraction_prompt(round_messages)

    assert "structured.operation" in prompt
    assert "DELETE" in prompt
    assert "ARCHIVE" in prompt
    assert "RESTORE" in prompt


def test_candidate_extraction_prompt_documents_discrete_memory_schema():
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="请记住我不喜欢 Java，之后别推荐 Java 项目")]
            )[0]
        }
    ]

    prompt = _build_candidate_extraction_prompt(round_messages)

    assert "memory_fact_v1" in prompt
    assert "structured.subject" in prompt
    assert "structured.entity" in prompt
    assert "structured.attribute" in prompt
    assert "structured.value" in prompt
    assert "structured.polarity" in prompt
    assert "structured.certainty" in prompt
    assert "structured.temporal" in prompt
    assert "structured.conflict_key" in prompt


def test_candidate_extraction_prompt_adds_locomo_event_guidance_for_official_transcript():
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "[2023-05-07] Caroline (D1:1): I went to an LGBTQ support "
                            "group today.\n"
                            "[2023-05-07] Melanie (D1:2): I painted a sunrise.\n"
                            "Image caption: a sunrise painting"
                        )
                    )
                ]
            )[0]
        }
    ]

    prompt = _build_candidate_extraction_prompt(round_messages)

    assert "LoCoMo benchmark" in prompt
    assert "event-level" in prompt
    assert "dialog id" in prompt
    assert "dates, times, places" in prompt
    assert "image captions" in prompt
    assert "Return every benchmark-relevant event candidate" in prompt
    assert "Return at most 5 candidates" not in prompt
    assert "D1:1" in prompt


def test_candidate_extraction_prompt_keeps_default_durable_guidance_for_normal_round():
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="以后回答请简洁，但结论带引用")]
            )[0]
        }
    ]

    prompt = _build_candidate_extraction_prompt(round_messages)

    assert "Extract durable user memory candidates" in prompt
    assert "LoCoMo benchmark" not in prompt


def test_extract_memory_candidates_marks_locomo_events_with_dialog_ids():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(
                candidates=[
                    MemoryCandidate(
                        memory_type="interaction_summary",
                        content=(
                            "D1:1 Caroline went to an LGBTQ support group on "
                            "2023-05-07."
                        ),
                        source="conversation",
                    )
                ]
            ),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "[2023-05-07] Caroline (D1:1): I went to an LGBTQ support "
                            "group today."
                        )
                    )
                ]
            )[0]
        }
    ]

    candidates, _, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert candidates[0].structured["schema_version"] == "locomo_event_v1"
    assert candidates[0].structured["dialog_ids"] == ["D1:1"]
    assert candidates[0].structured["source_mode"] == "locomo_benchmark"


def test_extract_memory_candidates_keeps_all_explicit_locomo_events_without_llm():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(candidates=[]),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content="\n".join(
                            f"[2023-05-07] Caroline (D1:{idx}): event {idx}"
                            for idx in range(1, 8)
                        )
                    )
                ]
            )[0]
        }
    ]
    explicit_memories = [
        f"D1:{idx} - On 2023-05-07, Caroline said: event {idx}"
        for idx in range(1, 8)
    ]

    candidates, usage, success = extract_memory_candidates_from_round(
        llm,
        round_messages,
        explicit_memories=explicit_memories,
    )

    assert success is True
    assert usage["total_tokens"] == 0
    assert len(candidates) == 7
    assert llm.prompts == []
    assert candidates[-1].structured["dialog_ids"] == ["D1:7"]
    assert all(
        candidate.structured["schema_version"] == "locomo_event_v1"
        for candidate in candidates
    )


def test_extract_memory_candidates_recovers_invalid_locomo_memory_type():
    llm = _StructuredOutputLLM(
        {
            "parsed": None,
            "raw": _RawResult(
                "```json\n"
                "{"
                '"candidates": ['
                "{"
                '"memory_type": "activity",'
                '"content": "D1:1 Caroline went to a support group on 2023-05-07.",'
                '"source": "conversation"'
                "}"
                "]"
                "}\n"
                "```",
                {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
            ),
            "parsing_error": ValueError("invalid enum"),
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "[2023-05-07] Caroline (D1:1): I went to a support "
                            "group today."
                        )
                    )
                ]
            )[0]
        }
    ]

    candidates, _, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert len(candidates) == 1
    assert candidates[0].memory_type == "interaction_summary"
    assert candidates[0].structured["schema_version"] == "locomo_event_v1"
    assert candidates[0].structured["dialog_ids"] == ["D1:1"]


def test_extract_memory_candidates_normalizes_discrete_fact_payload():
    llm = _StructuredOutputLLM(
        {
            "parsed": None,
            "raw": _RawResult(
                "```json\n"
                "{"
                '"candidates": ['
                "{"
                '"memory_type": "preference",'
                '"content": "用户不喜欢 Java。",'
                '"structured": {'
                '"subject": {"type": "user", "id": "u1", "label": "用户"},'
                '"entity": {"type": "language", "id": "java", "label": "Java"},'
                '"attribute": "preference",'
                '"value": {"canonical": "dislike", "text": "不喜欢"},'
                '"polarity": "negative",'
                '"certainty": "explicit",'
                '"temporal": {"tense": "current"}'
                "},"
                '"keywords": ["Java"],'
                '"importance": 0.8,'
                '"confidence": 0.9,'
                '"source": "conversation",'
                '"evidence": [{"message_id": "s1-1-0", "role": "human"}]'
                "}"
                "]"
                "}\n"
                "```",
                {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
            ),
            "parsing_error": ValueError("invalid json"),
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="请记住我不喜欢 Java，之后别推荐 Java 项目")]
            )[0]
        }
    ]

    candidates, _, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert candidates[0].structured["schema_version"] == "memory_fact_v1"
    assert candidates[0].structured["fact_kind"] == "discrete_fact"
    assert (
        candidates[0].structured["conflict_key"]
        == "subject:user|entity:language:java|attribute:preference"
    )


def test_paper_qa_preferences_keep_distinct_candidates_without_identity_keys():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(
                candidates=[
                    MemoryCandidate(
                        memory_type="preference",
                        content=(
                            "User prefers paper Q&A answers to start with the "
                            "conclusion, then evidence."
                        ),
                        source="conversation",
                    ),
                    MemoryCandidate(
                        memory_type="preference",
                        content=(
                            "When evidence in paper QA is insufficient, first explicitly state "
                            "that the evidence is insufficient, then provide actionable additional "
                            "retrieval suggestions."
                        ),
                        source="conversation",
                    ),
                ]
            ),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "如果论文问答里的检索证据不足，请先说明不足，"
                            "再给可执行的补充检索建议。"
                        )
                    )
                ]
            )[0]
        }
    ]

    candidates, _, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert [candidate.content for candidate in candidates] == [
        "User prefers paper Q&A answers to start with the conclusion, then evidence.",
        (
            "When evidence in paper QA is insufficient, first explicitly state "
            "that the evidence is insufficient, then provide actionable additional "
            "retrieval suggestions."
        ),
    ]


def test_paper_qa_identifier_preference_keeps_candidate_content_without_identity_key():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(
                candidates=[
                    MemoryCandidate(
                        memory_type="preference",
                        content=(
                            "When answering paper Q&A, preserve the arXiv ID and paper "
                            "title; do not use only an abbreviation."
                        ),
                        source="conversation",
                    )
                ]
            ),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "以后回答论文问答时，请保留 arXiv ID 和论文标题，"
                            "不要只写简称。"
                        )
                    )
                ]
            )[0]
        }
    ]

    candidates, _, success = extract_memory_candidates_from_round(llm, round_messages)

    assert success is True
    assert [candidate.content for candidate in candidates] == [
        (
            "When answering paper Q&A, preserve the arXiv ID and paper "
            "title; do not use only an abbreviation."
        ),
    ]


def test_memory_application_only_round_does_not_rewrite_preferences():
    llm = _StructuredOutputLLM(
        {
            "parsed": MemoryCandidateExtractionOutput(
                candidates=[
                    MemoryCandidate(
                        memory_type="preference",
                        content=(
                            "When answering paper questions, present the conclusion first, "
                            "then explain the method mainline in a clear order."
                        ),
                        source="conversation",
                    )
                ]
            ),
            "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            "parsing_error": None,
        }
    )
    round_messages = [
        {
            "message": serialize_messages(
                [
                    HumanMessage(
                        content=(
                            "基于刚才这些偏好，帮我回答一篇论文的方法主线时"
                            "应该怎么组织？"
                        )
                    )
                ]
            )[0]
        }
    ]

    candidates, usage, success = extract_memory_candidates_from_round(llm, round_messages)

    assert candidates == []
    assert usage["total_tokens"] == 0
    assert success is False
