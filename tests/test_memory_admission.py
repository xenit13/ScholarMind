from __future__ import annotations

from scholar_mind.memory.admission import MemoryAdmissionAction, MemoryAdmissionPolicy
from scholar_mind.models.domain import MemoryCandidate


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int] | None = None):
        self.content = content
        self.usage_metadata = usage_metadata or {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }


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


def _candidate(
    content: str,
    *,
    structured: dict | None = None,
    keywords: list[str] | None = None,
    evidence: list[dict] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type="preference",
        content=content,
        structured=structured or {},
        keywords=keywords or [],
        importance=0.8,
        confidence=0.9,
        source="conversation",
        evidence=evidence or [],
    )


def test_admission_uses_model_decision_when_model_returns_write():
    llm = _StructuredOutputLLM(
        {
            "parsed": {
                "action": "WRITE",
                "reason": "The candidate is a durable communication preference.",
                "matched_rules": [],
            },
            "raw": _RawResult('{"action":"WRITE","reason":"ok","matched_rules":[]}'),
            "parsing_error": None,
        }
    )

    decision, usage = MemoryAdmissionPolicy().evaluate(
        _candidate("用户偏好中文、结构化回答。"),
        llm=llm,
    )

    assert decision.action == MemoryAdmissionAction.WRITE
    assert decision.reason == "The candidate is a durable communication preference."
    assert decision.matched_rules == []
    assert usage["total_tokens"] == 5
    assert llm.prompts


def test_admission_uses_model_decision_when_model_returns_drop():
    llm = _StructuredOutputLLM(
        {
            "parsed": {
                "action": "DROP",
                "reason": "The candidate should not be stored.",
                "matched_rules": ["model_policy"],
            },
            "raw": _RawResult(
                '{"action":"DROP","reason":"model rejected","matched_rules":["model_policy"]}'
            ),
            "parsing_error": None,
        }
    )

    decision, usage = MemoryAdmissionPolicy().evaluate(
        _candidate("用户偏好中文、结构化回答。"),
        llm=llm,
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert decision.reason == "The candidate should not be stored."
    assert decision.matched_rules == ["model_policy"]
    assert usage["total_tokens"] == 5


def test_admission_falls_back_to_rules_when_model_fails():
    llm = _StructuredOutputLLM(
        {
            "parsed": None,
            "raw": _RawResult("not-json"),
            "parsing_error": ValueError("invalid json"),
        }
    )

    decision, usage = MemoryAdmissionPolicy().evaluate(
        _candidate("用户的 OpenAI API key 是 sk-abcdefghijklmnopqrstuvwxyz1234567890"),
        llm=llm,
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "secrets" in decision.matched_rules
    assert usage["total_tokens"] == 0


def test_admission_preserves_model_usage_when_invalid_model_output_falls_back_to_rules():
    llm = _StructuredOutputLLM(
        {
            "parsed": {
                "action": "UNKNOWN",
                "reason": "invalid",
                "matched_rules": [],
            },
            "raw": _RawResult('{"action":"UNKNOWN","reason":"invalid","matched_rules":[]}'),
            "parsing_error": None,
        }
    )

    decision, usage = MemoryAdmissionPolicy().evaluate(
        _candidate("用户的 OpenAI API key 是 sk-abcdefghijklmnopqrstuvwxyz1234567890"),
        llm=llm,
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "secrets" in decision.matched_rules
    assert usage["total_tokens"] == 5


def test_admission_writes_normal_durable_preference():
    decision, usage = MemoryAdmissionPolicy().evaluate(_candidate("用户偏好中文、结构化回答。"))

    assert decision.action == MemoryAdmissionAction.WRITE
    assert decision.matched_rules == []
    assert usage["total_tokens"] == 0


def test_admission_drops_secret_in_candidate_content():
    decision, _usage = MemoryAdmissionPolicy().evaluate(
        _candidate("用户的 OpenAI API key 是 sk-abcdefghijklmnopqrstuvwxyz1234567890")
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "secrets" in decision.matched_rules


def test_admission_drops_contact_info_in_structured_payload():
    decision, _usage = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户偏好通过邮件接收摘要。",
            structured={"contact": {"email": "alice@example.com"}},
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "precise_contact_or_address" in decision.matched_rules


def test_admission_drops_health_info_in_evidence_payload():
    decision, _usage = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户最近被诊断为偏头痛。",
            evidence=[{"role": "human", "content": "医生诊断我有偏头痛。"}],
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "medical_or_health" in decision.matched_rules


def test_admission_drops_tool_trace_payload():
    decision, _usage = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户项目使用内部检索服务。",
            structured={"tool_call_id": "call_123", "internal_url": "http://10.0.0.5/admin"},
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "model_or_tool_trace" in decision.matched_rules
