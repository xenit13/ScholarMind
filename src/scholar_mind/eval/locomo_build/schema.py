from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arxiv_id: str
    title: str
    category: str


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_id: str
    user_id: str
    background: str


class TemporalUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old: dict[str, Any]
    new: dict[str, Any]
    old_date: str
    new_date: str


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_id: str
    persona_id: str
    case_id: str
    case_topic: str
    papers: list[PaperRef]
    memory_type: Literal[
        "paper_read",
        "workflow",
        "preference",
        "feedback",
        "knowledge_level",
        "project_constraint",
    ]
    content: dict[str, Any]
    temporal: TemporalUpdate | None = None
    distractor_case_id: str


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker: Literal["user", "assistant"]
    dia_id: str
    text: str
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "seed_id": None,
            "memory_type": None,
            "is_distractor": True,
        }
    )


class Conversation(BaseModel):
    model_config = ConfigDict(extra="allow")
    speaker_a: str
    speaker_b: str
    session_1_date_time: str
    session_1: list[Turn] = Field(default_factory=list)


class QA(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str
    category: int
    evidence: list[str]
    metadata: dict[str, Any]


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    persona: Persona
    conversation: Conversation
    qa: list[QA]
