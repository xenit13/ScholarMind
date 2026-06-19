"""Deterministic WRITE/DROP admission checks for extracted memory candidates."""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from scholar_mind.models.domain import MemoryCandidate
from scholar_mind.models.structured_output import empty_usage, invoke_structured_output


class MemoryAdmissionAction(StrEnum):
    WRITE = "WRITE"
    DROP = "DROP"


class MemoryAdmissionDecision(BaseModel):
    action: MemoryAdmissionAction
    reason: str
    matched_rules: list[str] = Field(default_factory=list)


class MemoryAdmissionModelOutput(BaseModel):
    action: MemoryAdmissionAction
    reason: str
    matched_rules: list[str] = Field(default_factory=list)


class MemoryAdmissionPolicy:
    def evaluate(
        self, candidate: MemoryCandidate, *, llm=None
    ) -> tuple[MemoryAdmissionDecision, dict[str, float]]:
        if llm is not None:
            decision, usage = self._evaluate_with_model(candidate, llm)
            if decision is not None:
                return decision, usage
            return self._evaluate_with_rules(candidate), usage
        return self._evaluate_with_rules(candidate), empty_usage()

    def _evaluate_with_model(
        self, candidate: MemoryCandidate, llm
    ) -> tuple[MemoryAdmissionDecision | None, dict[str, float]]:
        output, usage = invoke_structured_output(
            llm,
            _build_model_admission_prompt(candidate),
            MemoryAdmissionModelOutput,
        )
        if output is None:
            return None, usage
        try:
            parsed = MemoryAdmissionModelOutput.model_validate(output)
        except Exception:
            return None, usage
        return (
            MemoryAdmissionDecision(
                action=parsed.action,
                reason=parsed.reason,
                matched_rules=parsed.matched_rules,
            ),
            usage,
        )

    def _evaluate_with_rules(self, candidate: MemoryCandidate) -> MemoryAdmissionDecision:
        payload = _candidate_payload_text(candidate)
        matched_rules = _match_prohibited_rules(payload)
        if matched_rules:
            return MemoryAdmissionDecision(
                action=MemoryAdmissionAction.DROP,
                reason="candidate contains prohibited memory content",
                matched_rules=matched_rules,
            )
        return MemoryAdmissionDecision(action=MemoryAdmissionAction.WRITE, reason="allowed")


def _build_model_admission_prompt(candidate: MemoryCandidate) -> str:
    return (
        "# Role\n"
        "You are the memory admission gate for ScholarMind.\n\n"
        "# Goal\n"
        "Decide whether this extracted memory candidate may be persisted.\n\n"
        "# Actions\n"
        "- WRITE: safe and useful durable memory for future interactions.\n"
        "- DROP: contains prohibited content or should not be stored.\n\n"
        "# Drop if the candidate contains\n"
        "- secrets, credentials, tokens, passwords, private keys, or tool credentials\n"
        "- payment data or bank/card details\n"
        "- government IDs or exact identity documents\n"
        "- precise contact details or physical addresses\n"
        "- medical, health, legal, financial, minor, or third-party private information\n"
        "- model/tool traces, system/developer prompts, request IDs, internal URLs, "
        "or raw retrieval text\n\n"
        "# Output\n"
        "Return valid JSON only with fields: action, reason, matched_rules.\n"
        "Use action WRITE or DROP. matched_rules should be an empty array for WRITE.\n\n"
        f"Candidate: {_candidate_payload_text(candidate)}"
    )


def _candidate_payload_text(candidate: MemoryCandidate) -> str:
    payload = {
        "content": candidate.content,
        "structured": candidate.structured,
        "keywords": candidate.keywords,
        "evidence": candidate.evidence,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _match_prohibited_rules(text: str) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for rule_name, patterns in _PROHIBITED_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            matched.append(rule_name)
    return matched


def _regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_PROHIBITED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "secrets": [
        _regex(r"\bsk-[a-z0-9_-]{20,}\b"),
        _regex(r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token)\b"),
        _regex(r"\b(password|passwd|pwd)\b"),
        _regex(r"-----begin (rsa |ec |openssh )?private key-----"),
        _regex(r"\beyj[a-z0-9_-]{20,}\.[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}\b"),
    ],
    "payment_data": [
        _regex(r"\b(?:\d[ -]*?){13,19}\b"),
        _regex(r"\b(cvv|cvc)\b|银行卡|信用卡|银行账号|卡号"),
    ],
    "government_ids": [
        _regex(r"\b\d{17}[\dx]\b"),
        _regex(r"\b(ssn|social security|passport)\b|护照|身份证|社保号|驾照"),
    ],
    "precise_contact_or_address": [
        _regex(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
        _regex(r"(?:\+?86[- ]?)?1[3-9]\d{9}"),
        _regex(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b"),
        _regex(r"家庭住址|详细地址|门牌号|收货地址|居住地址"),
    ],
    "medical_or_health": [
        _regex(r"\b(diagnosis|diagnosed|medication|therapy|depression|anxiety)\b"),
        _regex(r"诊断|医生|疾病|病史|用药|药物|处方|心理健康|抑郁|焦虑|偏头痛"),
    ],
    "legal_or_financial": [
        _regex(r"\b(lawsuit|bankruptcy|debt|salary|income|tax|portfolio)\b"),
        _regex(r"诉讼|债务|破产|收入|工资|资产|税务|纳税|投资组合"),
    ],
    "minors": [
        _regex(r"\b(minor|child|children|school name)\b"),
        _regex(r"未成年|儿童|孩子|小孩|学校名称"),
    ],
    "third_party_private_info": [
        _regex(r"\b(my|his|her|their) (spouse|partner|child|manager|coworker)\b"),
        _regex(r"\b(我朋友|我同事|我老板|我家人|我孩子|我妻子|我丈夫).*(隐私|电话|邮箱|地址|病|收入)\b"),
    ],
    "model_or_tool_trace": [
        _regex(r"\b(tool_call_id|trace_id|request_id|system prompt|developer message)\b"),
        _regex(r"\b(internal_url|localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"),
        _regex(r"内部url|工具调用|系统提示词|开发者消息|检索原文"),
    ],
}
