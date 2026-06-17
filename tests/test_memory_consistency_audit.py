from __future__ import annotations

from datetime import UTC, datetime

from scholar_mind.config import settings as settings_module
from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.consistency_audit import MemoryConsistencyAuditor
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import StructuredMemoryRecord


class _Index:
    def __init__(self):
        self.upserts: list[tuple[StructuredMemoryRecord, list[float]]] = []

    def upsert_memory(self, record, embedding):
        self.upserts.append((record, embedding))


class _Embedder:
    def embed_query(self, content: str):
        return [float(len(content)), 1.0]


class _Runnable:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return {
            "parsed": self.llm.outputs.pop(0),
            "raw": None,
            "parsing_error": None,
        }


class _AuditLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self)


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


def _discrete_structured(value: str, polarity: str) -> dict:
    return {
        "schema_version": "memory_fact_v1",
        "fact_kind": "discrete_fact",
        "subject": {"type": "user", "id": "u1", "label": "用户"},
        "entity": {"type": "language", "id": "java", "label": "Java"},
        "attribute": "preference",
        "value": {"canonical": value, "text": value},
        "polarity": polarity,
        "certainty": "explicit",
        "temporal": {"tense": "current"},
        "conflict_key": "subject:user:u1|entity:language:java|attribute:preference",
        "source_mode": "conversation",
    }


def _record(
    *,
    memory_id: str = "mem_java",
    content: str = "用户喜欢 Java。",
    structured: dict | None = None,
    evidence: list[dict] | None = None,
) -> StructuredMemoryRecord:
    now = datetime(2026, 6, 17, 8, 0, tzinfo=UTC)
    return StructuredMemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        scope="user",
        memory_type="preference",
        content=content,
        structured=structured or _discrete_structured("like", "positive"),
        keywords=["Java"],
        source="conversation",
        evidence=(
            evidence
            if evidence is not None
            else [{"role": "human", "content": "我其实不喜欢 Java。"}]
        ),
        importance=0.8,
        confidence=0.9,
        status="active",
        created_at=now,
        updated_at=now,
        decay_rate=0.01,
        decay_floor=0.5,
        version=1,
    )


def test_consistency_auditor_repairs_inconsistent_memory_from_evidence(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record())
    index = _Index()
    llm = _AuditLLM(
        {
            "verdict": "inconsistent",
            "source_of_truth": "evidence",
            "corrected_content": "用户不喜欢 Java。",
            "corrected_structured": _discrete_structured("dislike", "negative"),
            "corrected_keywords": ["Java", "偏好"],
            "confidence": 0.94,
            "reason": "原始证据明确表示用户不喜欢 Java。",
        }
    )
    auditor = MemoryConsistencyAuditor(
        repository=repository,
        index=index,
        embedder=_Embedder(),
        llm=llm,
        min_confidence=0.85,
    )

    result = auditor.run(user_id="u1")

    stored = repository.get("u1", "mem_java")
    events = repository.list_operation_events("u1")
    assert result["checked_count"] == 1
    assert result["repaired_count"] == 1
    assert stored is not None
    assert stored.content == "用户不喜欢 Java。"
    assert stored.structured["value"]["canonical"] == "dislike"
    assert stored.structured["polarity"] == "negative"
    assert stored.keywords == ["Java", "偏好"]
    assert stored.version == 2
    assert index.upserts[0][0].content == "用户不喜欢 Java。"
    assert events[0].operation == "UPDATE"
    assert events[0].model == "daily_consistency_audit"
    assert events[0].old_record["content"] == "用户喜欢 Java。"
    assert events[0].new_record["content"] == "用户不喜欢 Java。"
    assert events[0].candidate["audit_run_id"] == result["run_id"]
    assert events[0].candidate["source_of_truth"] == "evidence_json"
    assert events[0].candidate["evidence_hash"].startswith("sha256:")


def test_consistency_auditor_skips_low_confidence_correction(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record())
    llm = _AuditLLM(
        {
            "verdict": "inconsistent",
            "source_of_truth": "evidence",
            "corrected_content": "用户不喜欢 Java。",
            "corrected_structured": _discrete_structured("dislike", "negative"),
            "confidence": 0.6,
            "reason": "证据不足以自动修复。",
        }
    )
    auditor = MemoryConsistencyAuditor(
        repository=repository,
        index=_Index(),
        embedder=_Embedder(),
        llm=llm,
        min_confidence=0.85,
    )

    result = auditor.run(user_id="u1")

    stored = repository.get("u1", "mem_java")
    assert result["checked_count"] == 1
    assert result["repaired_count"] == 0
    assert result["skipped_count"] == 1
    assert stored is not None
    assert stored.content == "用户喜欢 Java。"
    assert repository.list_operation_events("u1") == []


def test_consistency_auditor_skips_memory_without_evidence_or_discrete_fact(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record(memory_id="mem_no_evidence", evidence=[]))
    repository.upsert(
        _record(
            memory_id="mem_no_discrete",
            structured={"schema_version": "legacy_summary"},
            evidence=[{"role": "human", "content": "请用中文回答。"}],
        )
    )
    llm = _AuditLLM()
    auditor = MemoryConsistencyAuditor(
        repository=repository,
        index=_Index(),
        embedder=_Embedder(),
        llm=llm,
        min_confidence=0.85,
    )

    result = auditor.run(user_id="u1")

    assert result["checked_count"] == 0
    assert result["skipped_count"] == 2
    assert llm.prompts == []


def test_memory_yaml_maps_consistency_audit_settings(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "memory.yaml").write_text(
        "memory:\n"
        "  consistency_audit:\n"
        "    enabled: false\n"
        "    auto_fix_enabled: false\n"
        "    min_confidence: 0.91\n"
        "    batch_size: 123\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "CONFIG_DIR", config_dir)

    payload = settings_module._memory_yaml_payload()

    assert payload["memory_consistency_audit_enabled"] is False
    assert payload["memory_consistency_audit_auto_fix_enabled"] is False
    assert payload["memory_consistency_audit_min_confidence"] == 0.91
    assert payload["memory_consistency_audit_batch_size"] == 123
