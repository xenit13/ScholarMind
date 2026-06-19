from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from scholar_mind.config.settings import Settings
from scholar_mind.memory.admission import MemoryAdmissionAction, MemoryAdmissionPolicy
from scholar_mind.memory.compressor import MessageCompressor
from scholar_mind.memory.context import get_memory_context, record_memory_event
from scholar_mind.memory.decay import MemoryScoreInput, rank_memory_candidates
from scholar_mind.memory.discrete import format_discrete_memory
from scholar_mind.memory.extraction import extract_memory_candidates_from_round
from scholar_mind.memory.operations import MemoryOperationApplier
from scholar_mind.memory.pending_buffer import PendingContextPayload, PendingConversationBuffer
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryExtractionOutput,
    MemoryOperationName,
    MemoryRecord,
    MemoryStatus,
    MessageLogEntry,
    StructuredMemoryRecord,
)
from scholar_mind.models.structured_output import (
    extract_json_candidate,
    invoke_structured_output,
    merge_usage,
    raw_output_text,
)
from scholar_mind.models.eval_models import MemoryCallEvent, MemoryOperation
from scholar_mind.rag.embeddings import EmbeddingService
from scholar_mind.rag.index import QdrantIndex
from scholar_mind.utils.messages import deserialize_messages, serialize_messages
from scholar_mind.utils.token_estimator import estimate_text_tokens

MEMORY_DUPLICATE_SCORE_THRESHOLD = 0.95
ROUND_MEMORY_DEDUP_SCORE_THRESHOLD = 0.9
ROUND_META_PREFIX = "__ROUND_META__"


def _local_now() -> datetime:
    return datetime.now().astimezone()


class MemoryManager:
    def __init__(
        self,
        settings: Settings,
        index: QdrantIndex,
        embedder: EmbeddingService,
        llm=None,
        metrics_repository=None,
        memory_eval_v2_repository=None,
        memory_repository: MemoryRepository | None = None,
    ):
        self.settings = settings
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.metrics_repository = metrics_repository
        self.memory_eval_v2_repository = memory_eval_v2_repository
        self.memory_repository = memory_repository
        self.operation_applier = (
            MemoryOperationApplier(memory_repository, index, embedder, llm=llm)
            if memory_repository is not None
            else None
        )
        self.admission_policy = MemoryAdmissionPolicy()
        self.pending_buffer = PendingConversationBuffer()
        self.compressor = MessageCompressor(
            context_window_tokens=settings.message_context_window_tokens,
            compact_threshold_ratio=settings.message_compact_threshold_ratio,
            llm=llm,
        )
        self.log_dir = settings.resolve_path(settings.log_dir)
        self.memory_root = settings.resolve_path(settings.memory_root_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.memory_root.mkdir(parents=True, exist_ok=True)

    async def get_context(self, user_id: str, current_query: str) -> tuple[str, int]:
        payload = self.get_context_payload_sync(
            user_id=user_id,
            session_id=None,
            current_query=current_query,
        )
        return payload.context, payload.hit_count

    async def get_context_payload(
        self,
        *,
        user_id: str,
        current_query: str,
        session_id: str | None = None,
    ) -> PendingContextPayload:
        return self.get_context_payload_sync(
            user_id=user_id,
            session_id=session_id,
            current_query=current_query,
        )

    def get_context_payload_sync(
        self,
        *,
        user_id: str,
        session_id: str | None,
        current_query: str,
    ) -> PendingContextPayload:
        persisted_context, persisted_hits = self.get_context_sync(
            user_id=user_id,
            current_query=current_query,
        )
        pending = self.pending_buffer.get_context_payload(
            user_id=user_id,
            session_id=session_id,
        )
        sections = []
        if persisted_context:
            sections.append(f"Persisted memory:\n{persisted_context}")
        if pending.context:
            sections.append(pending.context)
        return PendingContextPayload(
            context="\n\n".join(sections),
            hit_count=persisted_hits + pending.hit_count,
            notices=pending.notices,
            token_estimate=pending.token_estimate,
        )

    def get_context_sync(self, user_id: str, current_query: str) -> tuple[str, int]:
        embedding_started = perf_counter()
        vector = self.embedder.embed_query(current_query)
        embedding_latency_ms = int((perf_counter() - embedding_started) * 1000)
        max_hits = max(1, int(getattr(self.settings, "memory_top_k", 5)))
        min_score = float(getattr(self.settings, "memory_min_similarity_score", 0.6))
        search_limit = max_hits
        if self.memory_repository is not None:
            multiplier = max(1, int(getattr(self.settings, "memory_candidate_multiplier", 4)))
            search_limit = max_hits * multiplier
        search_started = perf_counter()
        raw_hits = self.index.search_memory(
            user_id=user_id,
            vector=vector,
            limit=search_limit,
        )
        vector_search_latency_ms = int((perf_counter() - search_started) * 1000)
        if self.memory_repository is not None:
            return self._get_structured_context_from_hits(
                user_id=user_id,
                current_query=current_query,
                raw_hits=raw_hits,
                max_hits=max_hits,
                min_score=min_score,
                embedding_latency_ms=embedding_latency_ms,
                vector_search_latency_ms=vector_search_latency_ms,
            )
        retrieved_memory_ids: list[str] = []
        retrieved_scores: list[float] = []
        hits = []
        for item in raw_hits:
            if not item.payload:
                continue
            record_id = _payload_memory_id(item.payload)
            score = float(getattr(item, "score", 0.0) or 0.0)
            if record_id:
                retrieved_memory_ids.append(record_id)
                retrieved_scores.append(score)
            if score < min_score:
                continue
            if not self._is_active_memory_payload(user_id, item.payload):
                continue
            hits.append(item)
            if len(hits) >= max_hits:
                break
        injected_memory_ids = [
            _payload_memory_id(item.payload)
            for item in hits
            if _payload_memory_id(item.payload)
        ]
        if not hits:
            self._record_memory_retrieval_event_v2(
                user_id=user_id,
                query=current_query,
                embedding_latency_ms=embedding_latency_ms,
                vector_search_latency_ms=vector_search_latency_ms,
                retrieved_memory_ids=retrieved_memory_ids,
                retrieved_scores=retrieved_scores,
                injected_memory_ids=[],
                injected_text="",
            )
            self._record_memory_eval_event(
                operation=MemoryOperation.CONTEXT_RETRIEVE,
                query=current_query,
                latency_ms=vector_search_latency_ms,
                hit_count=0,
            )
            return "", 0
        lines = [f"- {self._memory_content_from_payload(user_id, item.payload)}" for item in hits]
        injected_text = "\n".join(lines)
        self._record_memory_retrieval_event_v2(
            user_id=user_id,
            query=current_query,
            embedding_latency_ms=embedding_latency_ms,
            vector_search_latency_ms=vector_search_latency_ms,
            retrieved_memory_ids=retrieved_memory_ids,
            retrieved_scores=retrieved_scores,
            injected_memory_ids=injected_memory_ids,
            injected_text=injected_text,
        )
        self._record_memory_eval_event(
            operation=MemoryOperation.CONTEXT_RETRIEVE,
            query=current_query,
            latency_ms=vector_search_latency_ms,
            hit_count=len(lines),
            source_memory_ids=injected_memory_ids,
        )
        self._record_memory_eval_event(
            operation=MemoryOperation.MEMORY_INJECTION,
            query=current_query,
            latency_ms=vector_search_latency_ms,
            hit_count=len(lines),
            injected_text=injected_text,
            injected_chars=len(injected_text),
            source_memory_ids=injected_memory_ids,
        )
        return injected_text, len(lines)

    def _get_structured_context_from_hits(
        self,
        *,
        user_id: str,
        current_query: str,
        raw_hits: list,
        max_hits: int,
        min_score: float,
        embedding_latency_ms: int,
        vector_search_latency_ms: int,
    ) -> tuple[str, int]:
        retrieved_memory_ids: list[str] = []
        retrieved_scores: list[float] = []
        candidates: list[MemoryScoreInput] = []
        for item in raw_hits:
            if not item.payload:
                continue
            record_id = _payload_memory_id(item.payload)
            score = float(getattr(item, "score", 0.0) or 0.0)
            if record_id:
                retrieved_memory_ids.append(record_id)
                retrieved_scores.append(score)
            if score < min_score or not record_id:
                continue
            record = self.memory_repository.get(user_id, record_id)
            if record is None:
                continue
            status = record.status.value if hasattr(record.status, "value") else str(record.status)
            if status != MemoryStatus.ACTIVE.value:
                continue
            candidates.append(MemoryScoreInput(record=record, semantic_score=score))
        decay_enabled = bool(getattr(self.settings, "memory_decay_enabled", True))
        ranked = rank_memory_candidates(
            candidates,
            top_k=max_hits,
            min_final_score=float(
                getattr(self.settings, "memory_min_final_score", 0.05)
                if decay_enabled
                else 0.0
            ),
            enabled=decay_enabled,
            access_boost_factor=float(getattr(self.settings, "memory_access_boost_factor", 0.2)),
            access_boost_cap=float(getattr(self.settings, "memory_access_boost_cap", 1.5)),
        )
        injected_memory_ids = [item.record.memory_id for item in ranked]
        if not ranked:
            self._record_memory_retrieval_event_v2(
                user_id=user_id,
                query=current_query,
                embedding_latency_ms=embedding_latency_ms,
                vector_search_latency_ms=vector_search_latency_ms,
                retrieved_memory_ids=retrieved_memory_ids,
                retrieved_scores=retrieved_scores,
                injected_memory_ids=[],
                injected_text="",
            )
            self._record_memory_eval_event(
                operation=MemoryOperation.CONTEXT_RETRIEVE,
                query=current_query,
                latency_ms=vector_search_latency_ms,
                hit_count=0,
            )
            return "", 0
        self.memory_repository.record_access(user_id, injected_memory_ids)
        lines = [
            f"- {format_discrete_memory(item.record) or item.record.content}"
            for item in ranked
        ]
        injected_text = "\n".join(lines)
        self._record_memory_retrieval_event_v2(
            user_id=user_id,
            query=current_query,
            embedding_latency_ms=embedding_latency_ms,
            vector_search_latency_ms=vector_search_latency_ms,
            retrieved_memory_ids=retrieved_memory_ids,
            retrieved_scores=retrieved_scores,
            injected_memory_ids=injected_memory_ids,
            injected_text=injected_text,
        )
        self._record_memory_eval_event(
            operation=MemoryOperation.CONTEXT_RETRIEVE,
            query=current_query,
            latency_ms=vector_search_latency_ms,
            hit_count=len(lines),
            source_memory_ids=injected_memory_ids,
        )
        self._record_memory_eval_event(
            operation=MemoryOperation.MEMORY_INJECTION,
            query=current_query,
            latency_ms=vector_search_latency_ms,
            hit_count=len(lines),
            injected_text=injected_text,
            injected_chars=len(injected_text),
            source_memory_ids=injected_memory_ids,
        )
        return injected_text, len(lines)

    async def save(
        self, user_id: str, content: str, source: str = "conversation"
    ) -> MemoryRecord | None:
        content = content.strip()
        if not content:
            return None
        query_vector = await self.embedder.aembed_query(content)
        existing = await asyncio.to_thread(
            self.index.search_memory,
            user_id,
            query_vector,
            1,
        )
        if existing and existing[0].score >= MEMORY_DUPLICATE_SCORE_THRESHOLD:
            return None
        record = self._build_memory_record(user_id=user_id, content=content, source=source)
        await asyncio.to_thread(self._persist_memory_record, record)
        await asyncio.to_thread(self.index.upsert_memory, record, query_vector)
        self._record_memory_eval_event(
            operation=MemoryOperation.MEMORY_WRITE,
            query=content,
            injected_chars=len(content),
            source_memory_ids=[record.record_id],
        )
        return record

    def log_round(
        self,
        user_id: str,
        session_id: str,
        round_index: int,
        messages: list[BaseMessage],
        explicit_memories: list[str] | None = None,
    ) -> None:
        path = self._log_file_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized_messages = serialize_messages(messages)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f'__ROUND_START__ {{"thread_id":"{session_id}","round_index":{round_index}}}\n'
            )
            if explicit_memories:
                handle.write(
                    f"{ROUND_META_PREFIX} "
                    + json.dumps(
                        {"explicit_memories": explicit_memories},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for idx, message in enumerate(serialized_messages):
                entry = MessageLogEntry(
                    message_id=f"{session_id}-{round_index}-{idx}",
                    thread_id=session_id,
                    user_id=user_id,
                    message=message,
                    timestamp=_local_now(),
                    round_index=round_index,
                )
                handle.write(entry.model_dump_json() + "\n")
            handle.write(
                f'__ROUND_END__ {{"thread_id":"{session_id}","round_index":{round_index}}}\n\n'
            )

    def extract_pending_memories(self, user_id: str | None = None) -> int:
        state = self._load_extraction_state()
        users = (
            [user_id]
            if user_id
            else [path.name for path in self.log_dir.iterdir() if path.is_dir()]
        )
        extracted = 0
        for target_user in users:
            log_file = self._log_file_path(target_user)
            if not log_file.exists():
                continue
            checkpoint = state.get(target_user, {})
            last_offset = 0
            if checkpoint.get("file") == log_file.name:
                last_offset = int(checkpoint.get("offset", 0))
            started = perf_counter()
            new_offset, extracted_count, usage = self._extract_from_file(
                target_user, log_file, last_offset
            )
            state[target_user] = {"file": log_file.name, "offset": new_offset}
            if new_offset > last_offset:
                extracted += 1
                if self.metrics_repository is not None:
                    self.metrics_repository.record_memory_run(
                        user_id=target_user,
                        success=True,
                        extracted_count=extracted_count,
                        latency_ms=int((perf_counter() - started) * 1000),
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        total_tokens=int(usage.get("total_tokens", 0)),
                    )
        self._save_extraction_state(state)
        return extracted

    def _extract_from_file(
        self, user_id: str, path: Path, start_offset: int
    ) -> tuple[int, int, dict[str, float]]:
        current_round: list[dict] = []
        explicit_memories: list[str] = []
        in_round = False
        offset = start_offset
        committed_offset = start_offset
        current_round_offset = start_offset
        extracted_count = 0
        usage = merge_usage()
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(start_offset)
            for line in handle:
                line_start_offset = offset
                offset += len(line.encode("utf-8"))
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("__ROUND_START__"):
                    current_round = []
                    explicit_memories = []
                    in_round = True
                    current_round_offset = line_start_offset
                    continue
                if stripped.startswith(ROUND_META_PREFIX):
                    if in_round:
                        payload = json.loads(stripped.removeprefix(ROUND_META_PREFIX).strip())
                        explicit_memories = [
                            item.strip()
                            for item in payload.get("explicit_memories", [])
                            if isinstance(item, str) and item.strip()
                        ]
                    continue
                if stripped.startswith("__ROUND_END__"):
                    if current_round or explicit_memories:
                        if self.operation_applier is not None:
                            records, round_usage, success = (
                                self._extract_and_apply_structured_memories(
                                    user_id=user_id,
                                    round_messages=current_round,
                                    explicit_memories=explicit_memories,
                                    session_id=_round_session_id(current_round),
                                )
                            )
                            if not success:
                                return committed_offset, extracted_count, usage
                            usage = merge_usage(usage, round_usage)
                            extracted_count += len(records)
                        else:
                            memories, round_usage, success = self._extract_memories_from_round(
                                current_round,
                                explicit_memories=explicit_memories,
                            )
                            if not success:
                                return committed_offset, extracted_count, usage
                            usage = merge_usage(usage, round_usage)
                            for memory in memories:
                                self._save_sync(user_id, memory)
                                extracted_count += 1
                    committed_offset = offset
                    current_round = []
                    in_round = False
                    continue
                if in_round:
                    current_round.append(json.loads(stripped))
        if in_round:
            return current_round_offset, extracted_count, usage
        return committed_offset, extracted_count, usage

    def _save_sync(self, user_id: str, content: str) -> MemoryRecord | None:
        content = content.strip()
        if not content:
            return None
        vector = self.embedder.embed_query(content)
        existing = self.index.search_memory(user_id=user_id, vector=vector, limit=1)
        if existing and existing[0].score >= MEMORY_DUPLICATE_SCORE_THRESHOLD:
            return None
        record = self._build_memory_record(
            user_id=user_id,
            content=content,
            source="conversation",
        )
        self._persist_memory_record(record)
        self.index.upsert_memory(record, vector)
        return record

    def extract_request_memories(
        self,
        *,
        user_id: str,
        request_id: str,
        round_messages: list[dict],
        explicit_memories: list[str] | None = None,
    ) -> dict[str, object]:
        written_records: list[MemoryRecord | StructuredMemoryRecord] = []
        session_id = _round_session_id(round_messages)
        if self.operation_applier is not None:
            written_records, usage, success = self._extract_and_apply_structured_memories(
                user_id=user_id,
                round_messages=round_messages,
                explicit_memories=explicit_memories,
                request_id=request_id,
                session_id=session_id,
            )
        else:
            memories, usage, success = self._extract_memories_from_round(
                round_messages,
                explicit_memories=explicit_memories,
            )
            if success:
                for memory in memories:
                    record = self._save_sync(user_id, memory)
                    if record is not None:
                        written_records.append(record)

        if success:
            self.pending_buffer.remove_round(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
            )

        if self.memory_eval_v2_repository is not None:
            self.memory_eval_v2_repository.update_memory_extraction_result(
                request_id=request_id,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                written_memory_ids=[record.record_id for record in written_records],
                written_memory_texts=[record.content for record in written_records],
            )
        return {
            "request_id": request_id,
            "success": success,
            "written_count": len(written_records),
            "written_memory_ids": [record.record_id for record in written_records],
            "written_memory_texts": [record.content for record in written_records],
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
        }

    def _extract_and_apply_structured_memories(
        self,
        *,
        user_id: str,
        round_messages: list[dict],
        explicit_memories: list[str] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[list[StructuredMemoryRecord], dict[str, float], bool]:
        if self.operation_applier is None:
            return [], merge_usage(), False
        candidates, usage, success = extract_memory_candidates_from_round(
            self.llm,
            round_messages,
            explicit_memories=explicit_memories,
        )
        if not success:
            return [], usage, False
        written_records: list[StructuredMemoryRecord] = []
        for candidate in candidates:
            admission, admission_usage = self.admission_policy.evaluate(candidate, llm=self.llm)
            usage = merge_usage(usage, admission_usage)
            if admission.action == MemoryAdmissionAction.DROP:
                if request_id is not None:
                    self.pending_buffer.mark_rejected(
                        user_id=user_id,
                        session_id=session_id,
                        request_id=request_id,
                        round_index=_round_index(round_messages),
                        user_question=_latest_human_question(round_messages),
                        reasons=admission.matched_rules,
                    )
                self.operation_applier.reject_candidate(
                    user_id=user_id,
                    candidate=candidate,
                    request_id=request_id,
                    session_id=session_id,
                    reason=f"admission_drop:{','.join(admission.matched_rules)}",
                )
                continue
            result = self.operation_applier.apply_candidate(
                user_id=user_id,
                candidate=candidate,
                request_id=request_id,
                session_id=session_id,
            )
            if (
                result.operation in {MemoryOperationName.ADD, MemoryOperationName.UPDATE}
                and result.record is not None
                and result.record.status == MemoryStatus.ACTIVE
            ):
                written_records.append(result.record)
        return written_records, usage, True

    def _extract_memories_from_round(
        self, round_messages: list[dict], explicit_memories: list[str] | None = None
    ) -> tuple[list[str], dict[str, float], bool]:
        explicit_memories = [
            item.strip()
            for item in (explicit_memories or [])
            if isinstance(item, str) and item.strip()
        ]
        memories, usage, success = self._extract_memories_with_llm(round_messages)
        if explicit_memories:
            merged = self._dedupe_round_memories(explicit_memories + (memories if success else []))
            return merged, usage, True
        if not success:
            return [], usage, False
        return memories, usage, True

    def _build_memory_record(
        self,
        *,
        user_id: str,
        content: str,
        source: str,
    ) -> MemoryRecord | StructuredMemoryRecord:
        record_id = f"mem_{uuid4().hex}"
        created_at = datetime.now(UTC)
        if self.memory_repository is None:
            return MemoryRecord(
                record_id=record_id,
                user_id=user_id,
                created_at=created_at,
                source=source,
                content=content,
            )
        return StructuredMemoryRecord(
            memory_id=record_id,
            user_id=user_id,
            scope="user",
            memory_type="interaction_summary",
            content=content,
            source=source,
            importance=0.6,
            confidence=0.7,
            status="active",
            created_at=created_at,
            updated_at=created_at,
            decay_rate=0.03,
            decay_floor=0.3,
        )

    def _persist_memory_record(self, record: MemoryRecord | StructuredMemoryRecord) -> None:
        if self.memory_repository is not None and isinstance(record, StructuredMemoryRecord):
            self.memory_repository.upsert(record)
            return
        self._append_record_to_file(record)

    def _is_active_memory_payload(self, user_id: str, payload: dict) -> bool:
        status = payload.get("status", "active")
        if status != "active":
            return False
        if self.memory_repository is None:
            return True
        memory_id = _payload_memory_id(payload)
        if not memory_id:
            return True
        record = self.memory_repository.get(user_id, memory_id)
        status = (
            record.status.value
            if record is not None and hasattr(record.status, "value")
            else ""
        )
        return record is not None and status == "active"

    def _memory_content_from_payload(self, user_id: str, payload: dict) -> str:
        memory_id = _payload_memory_id(payload)
        if self.memory_repository is not None and memory_id:
            record = self.memory_repository.get(user_id, memory_id)
            if record is not None:
                return record.content
        return str(payload.get("content", ""))

    def _append_record_to_file(self, record: MemoryRecord) -> None:
        path = self._memory_file_path(record.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Memory\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"## {record.record_id}\n"
                f"- created_at: {record.created_at.isoformat()}\n"
                f"- source: {record.source}\n"
                f"- content: {record.content}\n\n"
            )

    def _memory_file_path(self, user_id: str) -> Path:
        return self.memory_root / user_id / "MEMORY.md"

    def _log_file_path(self, user_id: str) -> Path:
        stamp = _local_now().strftime("%Y-%m-%d")
        return self.log_dir / user_id / f"session_messages-{stamp}-1.jsonl"

    def _state_path(self) -> Path:
        return self.log_dir / "extraction_state.json"

    def _load_extraction_state(self) -> dict[str, dict[str, object]]:
        path = self._state_path()
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        migrated: dict[str, dict[str, object]] = {}
        for user_id, checkpoint in payload.items():
            if isinstance(checkpoint, int):
                migrated[user_id] = {
                    "file": self._log_file_path(user_id).name,
                    "offset": checkpoint,
                }
            else:
                migrated[user_id] = checkpoint
        return migrated

    def _save_extraction_state(self, state: dict[str, dict[str, object]]) -> None:
        self._state_path().write_text(json.dumps(state), encoding="utf-8")

    def _record_memory_eval_event(
        self,
        *,
        operation: MemoryOperation,
        query: str | None = None,
        latency_ms: int | None = None,
        hit_count: int | None = None,
        injected_text: str | None = None,
        injected_chars: int | None = None,
        source_memory_ids: list[str] | None = None,
        compression_before_tokens: int | None = None,
        compression_after_tokens: int | None = None,
    ) -> None:
        """Record a memory event to the current evaluation context."""
        eval_ctx = get_memory_context()
        if eval_ctx is None:
            return
        record_memory_event(
            MemoryCallEvent(
                request_id=eval_ctx.request_id,
                operation=operation,
                query=query,
                latency_ms=latency_ms,
                hit_count=hit_count,
                injected_text=injected_text[:500] if injected_text else None,
                injected_chars=injected_chars,
                source_memory_ids=source_memory_ids or [],
                compression_before_tokens=compression_before_tokens,
                compression_after_tokens=compression_after_tokens,
            )
        )

    def _record_memory_retrieval_event_v2(
        self,
        *,
        user_id: str,
        query: str,
        embedding_latency_ms: int,
        vector_search_latency_ms: int,
        retrieved_memory_ids: list[str],
        retrieved_scores: list[float],
        injected_memory_ids: list[str],
        injected_text: str,
    ) -> None:
        if self.memory_eval_v2_repository is None:
            return
        eval_ctx = get_memory_context()
        if eval_ctx is None:
            return
        self.memory_eval_v2_repository.save_memory_retrieval_event(
            {
                "request_id": eval_ctx.request_id,
                "user_id": user_id,
                "query": query,
                "embedding_latency_ms": embedding_latency_ms,
                "vector_search_latency_ms": vector_search_latency_ms,
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_scores": [round(float(score), 4) for score in retrieved_scores],
                "retrieved_count": len(retrieved_memory_ids),
                "injected_memory_ids": injected_memory_ids,
                "injected_count": len(injected_memory_ids),
                "injected_text": injected_text,
                "injected_tokens": estimate_text_tokens(
                    injected_text,
                    model_name=getattr(self.settings, "llm_reasoning_model", None),
                ),
            }
        )

    def _extract_memories_with_llm(
        self, round_messages: list[dict]
    ) -> tuple[list[str], dict[str, float], bool]:
        if self.llm is None:
            return [], merge_usage(), False
        messages = deserialize_messages([item["message"] for item in round_messages])
        prompt_messages = []
        for message in messages:
            payload: dict[str, object] = {
                "type": getattr(message, "type", "system"),
                "content": str(message.content),
            }
            if isinstance(message, AIMessage) and message.tool_calls:
                payload["tool_calls"] = message.tool_calls
            if isinstance(message, ToolMessage):
                payload["tool_call_id"] = message.tool_call_id
                payload["name"] = message.name
                payload["status"] = message.status
            prompt_messages.append(payload)
        prompt = (
            "# Role\n"
            "You are the memory extraction agent for ScholarMind.\n\n"
            "# Goal\n"
            "Extract up to 3 durable user memories from this conversation round.\n"
            "A durable memory is a stable fact that is likely to help future interactions.\n\n"
            "# Priorities\n"
            "1. Precision over recall\n"
            "2. Long-term usefulness over turn-specific detail\n"
            "3. User-stated facts over assistant interpretation\n"
            "4. Stable preferences, research interests, and recurring constraints over "
            "temporary requests\n\n"
            "# Extraction Rules\n"
            "- Keep only memories that are likely to remain useful beyond this round.\n"
            "- Prefer explicitly stated user preferences, research interests, goals, "
            "workflows, and recurring constraints.\n"
            "- Ignore assistant-only content unless it directly reflects a durable user "
            "preference or constraint.\n"
            "- Ignore one-off requests, temporary plans, transient context, and details "
            "tied only to the current turn.\n"
            "- If there are no clear durable memories, return an empty list.\n\n"
            "# Prohibitions\n"
            "- Do not invent memories that are not supported by the messages.\n"
            "- Do not restate or summarize the whole conversation.\n"
            "- Do not extract speculative inferences about the user.\n"
            "- Do not extract sensitive personal data unless the user explicitly asked the "
            "system to remember it.\n"
            "- Do not include assistant plans, tool traces, or temporary execution details "
            "as durable memory.\n\n"
            "# Good Memory Candidates\n"
            "- enduring topic interests\n"
            "- stable output preferences\n"
            "- recurring project constraints\n"
            "- persistent goals or domains of work\n\n"
            "# Bad Memory Candidates\n"
            "- a single one-time question\n"
            "- temporary task state\n"
            "- information already obsolete within the same round\n"
            "- speculative inferences about the user\n\n"
            "# Output\n"
            "Return valid JSON only. Do not add any prose outside the JSON object.\n"
            "Use exactly this top-level field:\n"
            "- `memories`: array of strings\n\n"
            "# Example\n"
            "```json\n"
            "{\n"
            "  \"memories\": [\n"
            "    \"User focuses on RAG and multi-agent systems research.\",\n"
            "    \"User prefers concise, structured answers.\",\n"
            "    \"User wants prompt rewrites to follow a production-grade structured format.\"\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            f"Messages: {json.dumps(prompt_messages, ensure_ascii=False)}"
        )
        structured, usage = invoke_structured_output(
            self.llm,
            prompt,
            MemoryExtractionOutput,
            recover=_recover_memory_extraction_output,
        )
        if not structured:
            return [], usage, False
        return [item.strip() for item in structured.memories if item.strip()][:3], usage, True

    def _dedupe_round_memories(self, memories: list[str]) -> list[str]:
        deduped: list[str] = []
        fingerprints: set[str] = set()
        embeddings: list[list[float] | None] = []

        for memory in memories:
            candidate = memory.strip()
            if not candidate:
                continue
            fingerprint = _normalize_memory_text(candidate)
            if fingerprint and fingerprint in fingerprints:
                continue
            embedding = None
            try:
                embedding = self.embedder.embed_query(candidate)
            except Exception:
                embedding = None
            if embedding is not None and any(
                existing is not None
                and _cosine_similarity(embedding, existing) >= ROUND_MEMORY_DEDUP_SCORE_THRESHOLD
                for existing in embeddings
            ):
                continue
            deduped.append(candidate)
            embeddings.append(embedding)
            if fingerprint:
                fingerprints.add(fingerprint)
            if len(deduped) >= 3:
                break
        return deduped


def _recover_memory_extraction_output(raw) -> MemoryExtractionOutput | None:
    payload = extract_json_candidate(raw_output_text(raw).strip())
    if not isinstance(payload, dict):
        return None
    memories = payload.get("memories") or payload.get("user_memories") or payload.get("items") or []
    if isinstance(memories, str):
        memories = [memories]
    normalized = []
    for item in memories:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            content = item.get("content") or item.get("memory") or item.get("text")
            if content:
                normalized.append(str(content))
    return MemoryExtractionOutput(memories=normalized)


def _payload_memory_id(payload: dict) -> str:
    return str(payload.get("memory_id") or payload.get("record_id") or "")


def _round_session_id(round_messages: list[dict]) -> str | None:
    for item in round_messages:
        if isinstance(item, dict) and item.get("thread_id"):
            return str(item["thread_id"])
    return None


def _round_index(round_messages: list[dict]) -> int | None:
    for item in round_messages:
        if not isinstance(item, dict):
            continue
        raw_round = item.get("round_index")
        if raw_round is None:
            continue
        try:
            return int(raw_round)
        except (TypeError, ValueError):
            return None
    return None


def _latest_human_question(round_messages: list[dict]) -> str:
    messages = deserialize_messages(
        [item["message"] for item in round_messages if isinstance(item, dict) and "message" in item]
    )
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return str(message.content).strip()
    return ""


def _normalize_memory_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[，。！？!?,.:：;；、\"'“”‘’()（）\[\]{}<>《》]", "", cleaned)
    return cleaned


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
