from __future__ import annotations

from scholar_mind.memory.admission import MemoryAdmissionAction, MemoryAdmissionPolicy
from scholar_mind.models.domain import MemoryCandidate


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


def test_admission_writes_normal_durable_preference():
    decision = MemoryAdmissionPolicy().evaluate(_candidate("用户偏好中文、结构化回答。"))

    assert decision.action == MemoryAdmissionAction.WRITE
    assert decision.matched_rules == []


def test_admission_drops_secret_in_candidate_content():
    decision = MemoryAdmissionPolicy().evaluate(
        _candidate("用户的 OpenAI API key 是 sk-abcdefghijklmnopqrstuvwxyz1234567890")
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "secrets" in decision.matched_rules


def test_admission_drops_contact_info_in_structured_payload():
    decision = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户偏好通过邮件接收摘要。",
            structured={"contact": {"email": "alice@example.com"}},
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "precise_contact_or_address" in decision.matched_rules


def test_admission_drops_health_info_in_evidence_payload():
    decision = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户最近被诊断为偏头痛。",
            evidence=[{"role": "human", "content": "医生诊断我有偏头痛。"}],
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "medical_or_health" in decision.matched_rules


def test_admission_drops_tool_trace_payload():
    decision = MemoryAdmissionPolicy().evaluate(
        _candidate(
            "用户项目使用内部检索服务。",
            structured={"tool_call_id": "call_123", "internal_url": "http://10.0.0.5/admin"},
        )
    )

    assert decision.action == MemoryAdmissionAction.DROP
    assert "model_or_tool_trace" in decision.matched_rules
