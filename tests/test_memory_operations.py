from __future__ import annotations

from datetime import UTC, datetime

from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.operations import MemoryOperationApplier
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import MemoryCandidate, StructuredMemoryRecord


class _Index:
    def __init__(self):
        self.upserts: list[object] = []

    def upsert_memory(self, record, _embedding):
        self.upserts.append(record)


class _FailingIndex:
    def upsert_memory(self, _record, _embedding):
        raise RuntimeError("index unavailable")


class _Embedder:
    def embed_query(self, _text: str):
        return [0.1, 0.2]


class _VectorEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_query(self, text: str):
        return self.vectors.get(text, [0.0, 1.0])


class _JudgeRunnable:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return {
            "parsed": {
                "relation": self.llm.relation,
                "confidence": self.llm.confidence,
                "reason": self.llm.reason,
            },
            "raw": None,
            "parsing_error": None,
        }


class _JudgeLLM:
    def __init__(
        self,
        relation: str,
        *,
        confidence: float = 0.9,
        reason: str = "test judge",
    ):
        self.relation = relation
        self.confidence = confidence
        self.reason = reason
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _JudgeRunnable(self)


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        bootstrap_sample_data=False,
    )


def _repository(tmp_path) -> MemoryRepository:
    settings = _settings(tmp_path)
    init_database(settings)
    return MemoryRepository(build_session_factory(settings))


def _candidate(
    content: str = "用户偏好简洁回答。",
    *,
    memory_type: str = "preference",
    confidence: float = 0.9,
    structured: dict | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=memory_type,
        content=content,
        structured=structured or {},
        keywords=["简洁"],
        importance=0.8,
        confidence=confidence,
        source="conversation",
        evidence=[{"message_id": "s1-1-0", "role": "human"}],
    )


def _discrete_structured(
    value: str,
    polarity: str,
    *,
    conflict_key: str = "subject:user:u1|entity:language:java|attribute:preference",
    certainty: str = "explicit",
    tense: str = "current",
    subject_id: str = "u1",
    entity_id: str = "java",
) -> dict:
    return {
        "schema_version": "memory_fact_v1",
        "fact_kind": "discrete_fact",
        "subject": {"type": "user", "id": subject_id, "label": "用户"},
        "entity": {"type": "language", "id": entity_id, "label": "Java"},
        "attribute": "preference",
        "value": {"canonical": value, "text": value},
        "polarity": polarity,
        "certainty": certainty,
        "temporal": {"tense": tense},
        "conflict_key": conflict_key,
        "source_mode": "conversation",
    }


def _record(
    memory_id: str,
    content: str = "用户偏好简洁回答。",
    *,
    memory_type: str = "preference",
    structured: dict | None = None,
) -> StructuredMemoryRecord:
    now = datetime(2026, 5, 19, 8, 0, tzinfo=UTC)
    return StructuredMemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        scope="user",
        memory_type=memory_type,
        content=content,
        structured=structured or {},
        source="conversation",
        importance=0.8,
        confidence=0.9,
        status="active",
        created_at=now,
        updated_at=now,
        decay_rate=0.01,
        decay_floor=0.5,
    )


def test_operation_applier_adds_new_candidate_and_records_event(tmp_path):
    repository = _repository(tmp_path)
    index = _Index()
    applier = MemoryOperationApplier(repository, index, _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", result.memory_id)
    events = repository.list_operation_events("u1")
    assert result.operation == "ADD"
    assert stored is not None
    assert stored.content == "用户偏好简洁回答。"
    assert index.upserts[0].memory_id == result.memory_id
    assert events[0].operation == "ADD"
    assert events[0].new_record["memory_id"] == result.memory_id


def test_operation_applier_keeps_audit_when_derived_index_upsert_fails(tmp_path):
    repository = _repository(tmp_path)
    applier = MemoryOperationApplier(repository, _FailingIndex(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", result.memory_id)
    events = repository.list_operation_events("u1")
    assert result.operation == "ADD"
    assert stored is not None
    assert events[0].operation == "ADD"
    assert events[0].new_record["memory_id"] == result.memory_id


def test_operation_applier_updates_semantic_match_and_increments_version(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing"))
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("用户偏好简洁回答，关键结论需要带引用。"),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    events = repository.list_operation_events("u1")
    assert result.operation == "UPDATE"
    assert result.memory_id == "mem_existing"
    assert stored is not None
    assert stored.content == "用户偏好简洁回答，关键结论需要带引用。"
    assert stored.version == 2
    assert events[0].operation == "UPDATE"
    assert events[0].old_record["content"] == "用户偏好简洁回答。"


def test_operation_applier_deletes_semantic_match_without_physical_delete(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing"))
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户要求忘记回答风格偏好。",
            structured={"operation": "DELETE"},
        ),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    events = repository.list_operation_events("u1")
    assert result.operation == "DELETE"
    assert stored is not None
    assert stored.status == "deleted"
    assert repository.list_active("u1") == []
    assert events[0].operation == "DELETE"
    assert events[0].old_record["memory_id"] == "mem_existing"


def test_operation_applier_records_none_for_duplicate_or_low_confidence_candidate(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing"))
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    duplicate = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(),
        request_id="req1",
        session_id="s1",
    )
    low_confidence = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("用户可能临时想要列表。", confidence=0.3),
        request_id="req2",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    events = repository.list_operation_events("u1")
    assert duplicate.operation == "NONE"
    assert low_confidence.operation == "NONE"
    assert stored is not None
    assert stored.version == 1
    assert [event.operation for event in events] == ["NONE", "NONE"]


def test_operation_applier_archives_and_restores_semantic_match(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing"))
    index = _Index()
    applier = MemoryOperationApplier(repository, index, _Embedder())

    archived = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("请暂时归档回答风格偏好。", structured={"operation": "ARCHIVE"}),
        request_id="req1",
        session_id="s1",
    )
    restored = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("请恢复回答风格偏好。", structured={"operation": "RESTORE"}),
        request_id="req2",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    events = repository.list_operation_events("u1")
    assert archived.operation == "ARCHIVE"
    assert restored.operation == "RESTORE"
    assert stored is not None
    assert stored.status == "active"
    assert [event.operation for event in events] == ["ARCHIVE", "RESTORE"]
    assert [record.status for record in index.upserts] == ["archived", "active"]


def test_operation_applier_updates_lexical_match_when_embedding_is_low(tmp_path):
    existing_content = "用户偏好简洁回答，结论要带引用。"
    candidate_content = "用户希望回答简洁，并在结论中附引用。"
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing", existing_content))
    applier = MemoryOperationApplier(
        repository,
        _Index(),
        _VectorEmbedder(
            {
                candidate_content: [1.0, 0.0],
                existing_content: [0.0, 1.0],
            }
        ),
    )

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(candidate_content),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "UPDATE"
    assert stored is not None
    assert stored.content == candidate_content
    assert stored.version == 2


def test_operation_applier_uses_judge_for_embedding_gray_zone_duplicate(tmp_path):
    existing_content = "用户偏好回答时先给结论。"
    candidate_content = "用户希望答案先说结论。"
    judge = _JudgeLLM("duplicate")
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing", existing_content))
    applier = MemoryOperationApplier(
        repository,
        _Index(),
        _VectorEmbedder(
            {
                candidate_content: [1.0, 0.0],
                existing_content: [0.75, 0.66],
            }
        ),
        llm=judge,
    )

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(candidate_content),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "NONE"
    assert result.memory_id == "mem_existing"
    assert stored is not None
    assert stored.version == 1
    assert len(judge.prompts) == 1


def test_operation_applier_uses_judge_for_embedding_gray_zone_distinct(tmp_path):
    existing_content = "用户偏好回答时先给结论。"
    candidate_content = "用户正在研究 RAG 评测。"
    judge = _JudgeLLM("distinct")
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_existing", existing_content))
    applier = MemoryOperationApplier(
        repository,
        _Index(),
        _VectorEmbedder(
            {
                candidate_content: [1.0, 0.0],
                existing_content: [0.75, 0.66],
            }
        ),
        llm=judge,
    )

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(candidate_content),
        request_id="req1",
        session_id="s1",
    )

    assert result.operation == "ADD"
    assert result.memory_id != "mem_existing"
    assert len(repository.list_active("u1")) == 2
    assert len(judge.prompts) == 1


def test_operation_applier_skips_discrete_duplicate_with_same_value(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户不喜欢 Java。",
            structured=_discrete_structured("dislike", "negative"),
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户明确不喜欢 Java。",
            structured=_discrete_structured("dislike", "negative"),
        ),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "NONE"
    assert result.memory_id == "mem_existing"
    assert stored is not None
    assert stored.status == "active"
    assert len(repository.list_active("u1")) == 1


def test_operation_applier_upgrades_matching_memory_with_discrete_structure(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户喜欢 Java。",
            structured={"explicit": True},
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户喜欢 Java。",
            structured=_discrete_structured("like", "positive"),
        ),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "UPDATE"
    assert result.memory_id == "mem_existing"
    assert stored is not None
    assert stored.structured["schema_version"] == "memory_fact_v1"
    assert (
        stored.structured["conflict_key"]
        == "subject:user|entity:language:java|attribute:preference"
    )


def test_operation_applier_supersedes_conflicting_discrete_memory(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户喜欢 Java。",
            structured=_discrete_structured("like", "positive"),
        )
    )
    index = _Index()
    applier = MemoryOperationApplier(repository, index, _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户不喜欢 Java。",
            structured=_discrete_structured("dislike", "negative"),
            confidence=0.9,
        ),
        request_id="req1",
        session_id="s1",
    )

    old_record = repository.get("u1", "mem_existing")
    new_record = repository.get("u1", result.memory_id)
    active = repository.list_active("u1")
    events = repository.list_operation_events("u1")
    assert result.operation == "UPDATE"
    assert result.memory_id != "mem_existing"
    assert old_record is not None
    assert old_record.status == "superseded"
    assert old_record.superseded_by == result.memory_id
    assert new_record is not None
    assert new_record.status == "active"
    assert new_record.supersedes == ["mem_existing"]
    assert [record.memory_id for record in active] == [result.memory_id]
    assert [record.status for record in index.upserts] == ["active", "superseded"]
    assert events[0].operation == "UPDATE"
    assert events[0].old_record["memory_id"] == "mem_existing"
    assert events[0].new_record["memory_id"] == result.memory_id


def test_operation_applier_supersedes_discrete_conflict_key_variants(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户喜欢 Java。",
            structured=_discrete_structured(
                "like",
                "positive",
                subject_id="u1",
                entity_id="java",
                conflict_key="subject:user:u1|entity:language:java|attribute:preference",
            ),
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户不喜欢 Java。",
            structured=_discrete_structured(
                "dislike",
                "negative",
                subject_id="user:unknown",
                entity_id="language:java",
                conflict_key=(
                    "subject:user:unknown|entity:language:java|attribute:preference"
                ),
            ),
            confidence=0.9,
        ),
        request_id="req1",
        session_id="s1",
    )

    old_record = repository.get("u1", "mem_existing")
    assert result.operation == "UPDATE"
    assert result.memory_id != "mem_existing"
    assert old_record is not None
    assert old_record.status == "superseded"


def test_operation_applier_rejects_low_confidence_discrete_conflict(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户喜欢 Java。",
            structured=_discrete_structured("like", "positive"),
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户可能不喜欢 Java。",
            structured=_discrete_structured("dislike", "negative"),
            confidence=0.74,
        ),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "NONE"
    assert result.memory_id == "mem_existing"
    assert stored is not None
    assert stored.status == "active"
    assert len(repository.list_active("u1")) == 1


def test_operation_applier_keeps_past_and_current_discrete_facts_active(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户现在喜欢 Java。",
            structured=_discrete_structured("like", "positive", tense="current"),
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate(
            "用户以前不喜欢 Java。",
            structured=_discrete_structured("dislike", "negative", tense="past"),
            confidence=0.9,
        ),
        request_id="req1",
        session_id="s1",
    )

    active = repository.list_active("u1")
    assert result.operation == "ADD"
    assert result.memory_id != "mem_existing"
    assert [record.memory_id for record in active] == ["mem_existing", result.memory_id]


def test_operation_applier_matches_specific_type_against_interaction_summary(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户偏好中文回答。",
            memory_type="interaction_summary",
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("用户偏好使用中文回答。", memory_type="preference"),
        request_id="req1",
        session_id="s1",
    )

    stored = repository.get("u1", "mem_existing")
    assert result.operation == "UPDATE"
    assert stored is not None
    assert stored.memory_type == "preference"
    assert stored.version == 2


def test_operation_applier_does_not_merge_unrelated_concrete_types(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(
        _record(
            "mem_existing",
            "用户偏好中文回答。",
            memory_type="workflow",
        )
    )
    applier = MemoryOperationApplier(repository, _Index(), _Embedder())

    result = applier.apply_candidate(
        user_id="u1",
        candidate=_candidate("用户偏好中文回答。", memory_type="preference"),
        request_id="req1",
        session_id="s1",
    )

    assert result.operation == "ADD"
    assert result.memory_id != "mem_existing"
    assert len(repository.list_active("u1")) == 2
