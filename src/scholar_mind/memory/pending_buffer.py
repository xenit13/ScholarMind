from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

PENDING_MEMORY_TOKEN_LIMIT = 30_000
PENDING_MEMORY_TTL = timedelta(hours=1)


class PendingContextPayload(BaseModel):
    context: str = ""
    hit_count: int = 0
    notices: list[str] = Field(default_factory=list)
    token_estimate: int = 0


class PendingConversationRound(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    round_index: int | None = None
    text: str
    user_question: str
    token_estimate: int
    created_at: datetime
    expires_at: datetime


class PendingMemoryNotice(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    text: str
    created_at: datetime
    expires_at: datetime


class PendingConversationBuffer:
    def __init__(
        self,
        *,
        max_tokens: int = PENDING_MEMORY_TOKEN_LIMIT,
        ttl: timedelta = PENDING_MEMORY_TTL,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.max_tokens = max(0, int(max_tokens))
        self.ttl = ttl
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._rounds: dict[tuple[str, str], deque[PendingConversationRound]] = {}
        self._notices: dict[tuple[str, str], deque[PendingMemoryNotice]] = {}

    def add_round(
        self,
        *,
        user_id: str,
        session_id: str | None,
        request_id: str,
        messages: Iterable[BaseMessage],
        round_index: int | None = None,
    ) -> None:
        session = _normalize_id(session_id)
        key = (user_id, session)
        self._cleanup_key(key)
        text, user_question = _round_text(messages, request_id=request_id, round_index=round_index)
        if not text:
            return
        now = self._now()
        rounds = self._rounds.setdefault(key, deque())
        rounds.append(
            PendingConversationRound(
                request_id=request_id,
                user_id=user_id,
                session_id=session,
                round_index=round_index,
                text=text,
                user_question=user_question,
                token_estimate=estimate_pending_tokens(text),
                created_at=now,
                expires_at=now + self.ttl,
            )
        )
        self._enforce_limit(key)

    def get_context_payload(
        self,
        *,
        user_id: str,
        session_id: str | None,
        consume_notices: bool = True,
    ) -> PendingContextPayload:
        key = (user_id, _normalize_id(session_id))
        self._cleanup_key(key)
        rounds = list(self._rounds.get(key, deque()))
        context = ""
        if rounds:
            context = "Pending conversation not yet extracted into memory:\n" + "\n\n".join(
                round_item.text for round_item in rounds
            )
        notices = [notice.text for notice in self._notices.get(key, deque())]
        if consume_notices:
            self._notices.pop(key, None)
        return PendingContextPayload(
            context=context,
            hit_count=len(rounds),
            notices=notices,
            token_estimate=sum(round_item.token_estimate for round_item in rounds),
        )

    def remove_round(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        request_id: str,
    ) -> None:
        keys = self._matching_keys(user_id=user_id, session_id=session_id)
        for key in keys:
            self._cleanup_key(key)
            rounds = self._rounds.get(key)
            if not rounds:
                continue
            kept = deque(item for item in rounds if item.request_id != request_id)
            if kept:
                self._rounds[key] = kept
            else:
                self._rounds.pop(key, None)

    def mark_rejected(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        request_id: str,
        round_index: int | None = None,
        user_question: str | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        key, round_item = self._find_round(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )
        session = key[1] if key is not None else _normalize_id(session_id)
        resolved_round = round_index
        resolved_question = (user_question or "").strip()
        if round_item is not None:
            resolved_round = (
                resolved_round if resolved_round is not None else round_item.round_index
            )
            resolved_question = resolved_question or round_item.user_question
        now = self._now()
        text = _notice_text(resolved_round, resolved_question, reasons)
        notice_key = (user_id, session)
        self._cleanup_key(notice_key)
        if any(
            item.request_id == request_id and item.text == text
            for item in self._notices.get(notice_key, deque())
        ):
            return
        notice = PendingMemoryNotice(
            request_id=request_id,
            user_id=user_id,
            session_id=session,
            text=text,
            created_at=now,
            expires_at=now + self.ttl,
        )
        self._notices.setdefault(notice_key, deque()).append(notice)

    def _cleanup_key(self, key: tuple[str, str]) -> None:
        now = self._now()
        rounds = self._rounds.get(key)
        if rounds is not None:
            kept_rounds = deque(item for item in rounds if item.expires_at > now)
            if kept_rounds:
                self._rounds[key] = kept_rounds
            else:
                self._rounds.pop(key, None)
        notices = self._notices.get(key)
        if notices is not None:
            kept_notices = deque(item for item in notices if item.expires_at > now)
            if kept_notices:
                self._notices[key] = kept_notices
            else:
                self._notices.pop(key, None)

    def _enforce_limit(self, key: tuple[str, str]) -> None:
        rounds = self._rounds.get(key)
        if not rounds:
            return
        while len(rounds) > 1 and _total_tokens(rounds) > self.max_tokens:
            rounds.popleft()
        if rounds and _total_tokens(rounds) > self.max_tokens:
            item = rounds[-1]
            keep_chars = int(self.max_tokens / 1.2 * 4) if self.max_tokens > 0 else 0
            text = item.text[-keep_chars:] if keep_chars else ""
            rounds[-1] = item.model_copy(
                update={"text": text, "token_estimate": estimate_pending_tokens(text)}
            )

    def _matching_keys(self, *, user_id: str, session_id: str | None) -> list[tuple[str, str]]:
        if session_id is not None:
            return [(user_id, _normalize_id(session_id))]
        return [key for key in set(self._rounds) | set(self._notices) if key[0] == user_id]

    def _find_round(
        self,
        *,
        user_id: str,
        session_id: str | None,
        request_id: str,
    ) -> tuple[tuple[str, str] | None, PendingConversationRound | None]:
        for key in self._matching_keys(user_id=user_id, session_id=session_id):
            self._cleanup_key(key)
            for item in self._rounds.get(key, deque()):
                if item.request_id == request_id:
                    return key, item
        return None, None

    def _now(self) -> datetime:
        now = self._now_fn()
        return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def estimate_pending_tokens(text: str) -> int:
    return int(len(text) / 4 * 1.2)


def _round_text(
    messages: Iterable[BaseMessage],
    *,
    request_id: str,
    round_index: int | None,
) -> tuple[str, str]:
    if round_index:
        lines = [f"[request_id={request_id} round={round_index}]"]
    else:
        lines = [f"[request_id={request_id}]"]
    user_question = ""
    for message in messages:
        role = ""
        if isinstance(message, HumanMessage):
            role = "Human"
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            role = "Assistant"
        if not role:
            continue
        content = _message_text(getattr(message, "content", ""))
        if not content:
            continue
        if role == "Human":
            user_question = content
        lines.append(f"{role}: {content}")
    if len(lines) == 1:
        return "", ""
    return "\n".join(lines), user_question


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def _notice_text(
    round_index: int | None,
    user_question: str,
    reasons: list[str] | None,
) -> str:
    prefix = _notice_subject(round_index, user_question)
    category = _notice_category(reasons)
    return f"{prefix}包含{category}，这类信息不适合保存到长期记忆，我没有将其写入记忆。"


def _notice_subject(round_index: int | None, user_question: str) -> str:
    question = _redact_notice_question(user_question)
    if question:
        return f"问题「{question}」"
    return f"第 {round_index} 轮问题" if round_index is not None else "问题"


def _redact_notice_question(user_question: str) -> str:
    question = " ".join(user_question.strip().split())
    if not question:
        return ""
    question = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[已隐藏]",
        question,
    )
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        question = pattern.sub("[已隐藏]", question)
    return question[:200]


def _notice_category(reasons: list[str] | None) -> str:
    normalized = " ".join(reasons or ()).lower()
    if any(
        marker in normalized
        for marker in (
            "secret",
            "credential",
            "token",
            "password",
            "private key",
            "api",
        )
    ):
        return "凭证、密钥或令牌类敏感信息"
    if "payment" in normalized or "credit" in normalized or "cvv" in normalized:
        return "支付或银行卡类敏感信息"
    if "government" in normalized or "passport" in normalized or "ssn" in normalized:
        return "证件号码类敏感信息"
    if "contact" in normalized or "address" in normalized:
        return "精确联系方式或地址类敏感信息"
    if "medical" in normalized or "health" in normalized:
        return "医疗或健康类敏感信息"
    if "legal" in normalized or "financial" in normalized:
        return "法律或财务类敏感信息"
    if "minor" in normalized or "child" in normalized:
        return "未成年人相关敏感信息"
    if "third_party" in normalized or "third party" in normalized:
        return "第三方私人信息"
    if "tool" in normalized or "trace" in normalized:
        return "模型或工具调用轨迹信息"
    return "敏感或不适合长期保存的信息"


def _normalize_id(value: str | None) -> str:
    return value or ""


def _total_tokens(rounds: Iterable[PendingConversationRound]) -> int:
    return sum(item.token_estimate for item in rounds)


_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|pwd)\b"
    r"(\s*(?:是|为|=|:|：)\s*)"
    r"([^，。；;,\s]+)",
    re.IGNORECASE,
)

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\bsk-[a-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(
        r"\beyj[a-z0-9_-]{20,}\.[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?:\+?86[- ]?)?1[3-9]\d{9}"),
    re.compile(r"\b\d{17}[\dx]\b", re.IGNORECASE),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)
