from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from scholar_mind.rag.top_k import IDEA_EVIDENCE_TOP_K


def utcnow() -> datetime:
    return datetime.now(UTC)


class QueryType(StrEnum):
    QA = "qa"
    IDEA_NOVELTY = "idea_novelty"
    TREND = "trend"
    CROSS_DOMAIN = "cross_domain"
    STUDY_PLAN = "study_plan"
    PAPER_READING = "paper_reading"


class RetrievalStrategyName(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED_HYBRID = "reranked_hybrid"


class ChunkType(StrEnum):
    METADATA = "metadata"
    SECTION = "section"
    TABLE = "table"
    FORMULA = "formula"
    ALGORITHM = "algorithm"
    FIGURE_DESC = "figure_desc"


class TimeRange(BaseModel):
    start: date | None = None
    end: date | None = None


class AgentTrace(BaseModel):
    agent: str
    duration_ms: int


class PaperReference(BaseModel):
    ref_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    arxiv_id: str | None = None


class PaperSection(BaseModel):
    section_id: str
    title: str
    content: str
    level: int = 1
    formulas: list[str] = Field(default_factory=list)
    has_algorithm: bool = False


class StructuredPaper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    publish_date: date
    citation_count: int | None = None
    sections: list[PaperSection] = Field(default_factory=list)
    references: list[PaperReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperChunk(BaseModel):
    chunk_id: str
    paper_id: str
    chunk_type: ChunkType = ChunkType.SECTION
    section: str
    subsection: str | None = None
    content: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str
    content: str
    score: float
    strategy: RetrievalStrategyName
    categories: list[str] = Field(default_factory=list)
    publish_date: date | None = None


class Citation(BaseModel):
    paper_id: str
    title: str
    section: str
    quote: str
    relevance_score: float


class RelatedPaper(BaseModel):
    paper_id: str
    title: str
    score: float


class ResearchAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    related_papers: list[RelatedPaper] = Field(default_factory=list)
    rag_info: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[AgentTrace] = Field(default_factory=list)
    session_id: str


class OverlapEvidence(BaseModel):
    section: str
    snippet: str


class OverlapPaper(BaseModel):
    paper_id: str
    title: str
    overlap_aspects: list[str] = Field(default_factory=list)
    evidence: list[OverlapEvidence] = Field(default_factory=list)


class DifferenceItem(BaseModel):
    aspect: str
    description: str


class UnexploredAspect(BaseModel):
    aspect: str
    reason: str


class NoveltySummary(BaseModel):
    summary: str = ""
    overall_judgement: Literal[
        "high_overlap", "partial_overlap", "no_direct_evidence"
    ] = "partial_overlap"


class IdeaNoveltyReport(BaseModel):
    idea_summary: str = ""
    overlapping_papers: list[OverlapPaper] = Field(default_factory=list)
    differences: list[DifferenceItem] = Field(default_factory=list)
    unexplored_aspects: list[UnexploredAspect] = Field(default_factory=list)
    novelty_report: NoveltySummary = Field(default_factory=NoveltySummary)
    references: list[dict[str, Any]] = Field(default_factory=list)


class TrendPoint(BaseModel):
    period: str
    count: int


class EmergingKeyword(BaseModel):
    keyword: str
    growth_rate: float


class HotSubtopic(BaseModel):
    topic: str
    paper_count: int
    key_papers: list[str] = Field(default_factory=list)


class TrendReport(BaseModel):
    paper_count_by_period: list[TrendPoint] = Field(default_factory=list)
    emerging_keywords: list[EmergingKeyword] = Field(default_factory=list)
    hot_subtopics: list[HotSubtopic] = Field(default_factory=list)
    representative_papers: list[dict[str, Any]] = Field(default_factory=list)


class TransferAnalysisItem(BaseModel):
    paper_id: str
    title: str
    categories: list[str] = Field(default_factory=list)
    summary: str = ""
    methodology_similarity: float
    transfer_rationale: str
    sources: list[SourceRecord] = Field(default_factory=list)


class HypothesisEvidence(BaseModel):
    paper_id: str
    title: str
    claim: str
    role: Literal["source", "candidate", "methodology_lookup"]
    sources: list[SourceRecord] = Field(default_factory=list)


class HypothesisNoveltyCheck(BaseModel):
    is_novel: bool = True
    confidence: float = 0.0
    rationale: str = ""
    sources: list[SourceRecord] = Field(default_factory=list)


class ExperimentDesign(BaseModel):
    target_domain: str
    core_intervention: str
    datasets_or_tasks: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    ablations: list[str] = Field(default_factory=list)


class CrossDomainHypothesis(BaseModel):
    hypothesis: str
    candidate_paper_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    novelty_check: HypothesisNoveltyCheck = Field(default_factory=HypothesisNoveltyCheck)
    experiment_design: ExperimentDesign
    sources: list[SourceRecord] = Field(default_factory=list)


class StudyPlanPhase(BaseModel):
    title: str
    weeks: str
    objectives: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    recommended_papers: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)


class StudyPlanCheckpoint(BaseModel):
    week: int
    checkpoint: str


class StudyPlanReport(BaseModel):
    goal_summary: str = ""
    plan_basis: Literal["memory_grounded", "input_grounded", "exploratory"] = "exploratory"
    plan_horizon_weeks: int = 8
    phases: list[StudyPlanPhase] = Field(default_factory=list)
    weekly_checkpoints: list[StudyPlanCheckpoint] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PaperReadingPaperState(BaseModel):
    paper_id: str
    title: str
    current_section: str
    current_paragraph_index: int = 0


class PaperReadingPassage(BaseModel):
    section: str
    paragraph_index: int
    text: str


class PaperReadingExplanation(BaseModel):
    plain_language: str = ""
    technical_detail: str = ""
    formula_notes: list[str] = Field(default_factory=list)
    figure_notes: list[str] = Field(default_factory=list)
    algorithm_notes: list[str] = Field(default_factory=list)


class PaperReadingKnowledgeLink(BaseModel):
    related_memory: str
    connection: str


class PaperReadingNotes(BaseModel):
    contribution: list[str] = Field(default_factory=list)
    methodology: list[str] = Field(default_factory=list)
    key_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PaperReadingNextStep(BaseModel):
    section: str
    paragraph_index: int
    suggestion: str


class CrossDomainReport(BaseModel):
    source_methodology: dict[str, Any]
    transfer_analysis: list[TransferAnalysisItem] = Field(default_factory=list)
    hypotheses: list[CrossDomainHypothesis] = Field(default_factory=list)
    agent_trace: list[AgentTrace] = Field(default_factory=list)
    session_id: str


class PaperReadingReport(BaseModel):
    paper: PaperReadingPaperState
    current_passage: PaperReadingPassage
    explanation: PaperReadingExplanation = Field(default_factory=PaperReadingExplanation)
    knowledge_links: list[PaperReadingKnowledgeLink] = Field(default_factory=list)
    notes: PaperReadingNotes = Field(default_factory=PaperReadingNotes)
    next_step: PaperReadingNextStep


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    closed_at: datetime | None = None
    message_count: int = 0
    topics_discussed: list[str] = Field(default_factory=list)
    papers_mentioned: list[str] = Field(default_factory=list)
    memory_context_loaded: bool = False


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    RESEARCH_INTEREST = "research_interest"
    KNOWLEDGE_LEVEL = "knowledge_level"
    GOAL = "goal"
    WORKFLOW = "workflow"
    PROJECT_CONSTRAINT = "project_constraint"
    PAPER_READ = "paper_read"
    INTERACTION_SUMMARY = "interaction_summary"
    FEEDBACK = "feedback"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryOperationName(StrEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"
    ARCHIVE = "ARCHIVE"
    RESTORE = "RESTORE"


class MemoryRecord(BaseModel):
    record_id: str
    user_id: str
    created_at: datetime
    source: str
    content: str


class StructuredMemoryRecord(BaseModel):
    memory_id: str
    user_id: str
    scope: Literal["user", "session", "org"] = "user"
    session_id: str | None = None
    request_id: str | None = None
    memory_type: MemoryType = MemoryType.INTERACTION_SUMMARY
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    source: Literal["explicit", "conversation", "system_extracted", "user_edited"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    sensitivity: Literal["normal", "sensitive"] = "normal"
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expires_at: datetime | None = None
    decay_rate: float = 0.03
    decay_floor: float = 0.3
    access_count: int = 0
    access_count_30d: int = 0
    last_decay_score: float | None = None
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    version: int = 1

    @property
    def record_id(self) -> str:
        return self.memory_id

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            record_id=self.memory_id,
            user_id=self.user_id,
            created_at=self.created_at,
            source=self.source,
            content=self.content,
        )


class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source: Literal["explicit", "conversation", "system_extracted"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class MemoryCandidateExtractionOutput(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)


class MemoryOperationEvent(BaseModel):
    event_id: str
    user_id: str
    operation: MemoryOperationName
    memory_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)
    old_record: dict[str, Any] | None = None
    new_record: dict[str, Any] | None = None
    reason: str = ""
    model: str = "rule"
    created_at: datetime


class MemoryOperationResult(BaseModel):
    operation: MemoryOperationName
    memory_id: str | None = None
    record: StructuredMemoryRecord | None = None
    event_id: str | None = None
    reason: str = ""


class MessageLogEntry(BaseModel):
    message_id: str
    thread_id: str
    user_id: str
    message: dict[str, Any]
    timestamp: datetime
    round_index: int


class RAGEvalSample(BaseModel):
    sample_id: str
    query: str
    query_type: str
    relevant_chunk_ids: list[str]
    relevance_levels: dict[str, int] = Field(default_factory=dict)
    category: str | None = None
    difficulty: str = "medium"


class BenchmarkStrategyResult(BaseModel):
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    hit_rate: float
    latency_p95_ms: float


class PlannerOutput(BaseModel):
    query_type: QueryType
    sub_queries: list[str] = Field(default_factory=list)
    plan: str = ""
    source_papers: list[str] = Field(default_factory=list)
    target_domains: list[str] = Field(default_factory=list)
    paper_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    paper_title: str | None = None
    section: str | None = None
    paragraph_index: int | None = None
    depth: str | None = None
    instruction: str | None = None
    topic: str | None = None
    goal: str | None = None
    current_progress: str | None = None
    read_papers: list[str] = Field(default_factory=list)
    known_topics: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    weekly_hours: int | None = None
    constraints: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    granularity: str | None = None
    max_papers: int | None = None
    max_hypotheses: int | None = None
    rag_strategy: str | None = None


class PaperReadingPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = 0
    section: str = ""
    purpose: str = ""


class PaperReadingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = "start_reading"
    target_section: str | None = None
    target_paragraph_index: int | None = None
    reading_mode: str = "paragraph_explain"
    depth: str | None = None
    reason: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    reading_goal: str = ""
    plan: list[PaperReadingPlanStep] = Field(default_factory=list)


class ReviewerOutput(BaseModel):
    final_answer: str = ""
    review_score: float = 0.0
    notes: str = ""


class MemoryExtractionOutput(BaseModel):
    memories: list[str] = Field(default_factory=list)


class CompressionOutput(BaseModel):
    summary: str = ""


class WriterInsightOutput(BaseModel):
    comparative_analysis: str = ""
    future_directions: list[str] = Field(default_factory=list)


class AnswerGenerationOutput(BaseModel):
    answer: str = ""
    grounded: bool = True


class TrendOutput(BaseModel):
    summary: str = ""
    emerging_keywords: list[str] = Field(default_factory=list)


class WriterOutput(BaseModel):
    draft: str = ""
    report_payload: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    kind: str
    label: str = ""
    paper_id: str | None = None
    title: str | None = None
    section: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrossDomainSourcePaper(BaseModel):
    requested_paper: str
    resolved: bool = False
    paper_id: str | None = None
    title: str
    categories: list[str] = Field(default_factory=list)
    summary: str = ""
    methodology_summary: str = ""
    sources: list[SourceRecord] = Field(default_factory=list)


class CrossDomainOutput(BaseModel):
    planner_intent: dict[str, Any] = Field(default_factory=dict)
    source_methodology: dict[str, Any] = Field(default_factory=dict)
    candidate_papers: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)


class ReportSummary(BaseModel):
    report_id: str
    type: str
    created_at: datetime
    config: dict[str, Any]
    results: dict[str, Any]


ASK_QUERY_MIN_LENGTH = 1


class AskRequest(BaseModel):
    query: str = Field(min_length=ASK_QUERY_MIN_LENGTH)
    user_id: str
    session_id: str | None = None
    mode: QueryType = QueryType.QA
    paper_ids: list[str] = Field(default_factory=list)
    rag_strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID
    conditional_memory_injection: bool = False
    memory_extraction_enabled: bool | None = None
    request_memory_extraction_enabled: bool | None = None
    wait_for_pending_extractions: bool = False


class ChatRequest(BaseModel):
    query: str = Field(min_length=ASK_QUERY_MIN_LENGTH)
    user_id: str
    session_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    rag_strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID
    conditional_memory_injection: bool = False
    memory_extraction_enabled: bool | None = None
    request_memory_extraction_enabled: bool | None = None
    wait_for_pending_extractions: bool = False


class TranscriptMemoryMessage(BaseModel):
    message: dict[str, Any]
    message_id: str
    thread_id: str | None = None
    round_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptMemoryExtractionRequest(BaseModel):
    user_id: str
    request_id: str
    session_id: str
    round_messages: list[TranscriptMemoryMessage] = Field(min_length=1)
    wait_for_pending_extractions: bool = True


class IdeaNoveltyRequest(BaseModel):
    idea: str = Field(min_length=5)
    user_id: str
    session_id: str | None = None
    max_papers: int = IDEA_EVIDENCE_TOP_K
    time_range: TimeRange | None = None
    categories: list[str] = Field(default_factory=list)
    rag_strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID
    conditional_memory_injection: bool = False


class TrendRequest(BaseModel):
    topic: str = Field(min_length=3)
    user_id: str
    session_id: str | None = None
    time_range: TimeRange | None = None
    granularity: Literal["yearly", "quarterly", "monthly"] = "quarterly"
    conditional_memory_injection: bool = False


class CrossDomainRequest(BaseModel):
    request: str = Field(min_length=5)
    user_id: str
    session_id: str | None = None
    max_hypotheses: int = 3
    rag_strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID
    conditional_memory_injection: bool = False


class StudyPlanRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    request: str | None = None
    goal: str | None = None
    current_progress: str | None = None
    read_papers: list[str] = Field(default_factory=list)
    known_topics: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    weekly_hours: int | None = None
    constraints: list[str] = Field(default_factory=list)
    conditional_memory_injection: bool = False


class PaperReadingRequest(BaseModel):
    paper_id: str | None = None
    user_id: str
    session_id: str | None = None
    instruction: str = "开始精读"
    section: str | None = None
    paragraph_index: int | None = None
    depth: Literal["brief", "standard", "deep"] = "standard"
    conditional_memory_injection: bool = False


class SessionCreateRequest(BaseModel):
    user_id: str


T = TypeVar("T")


class ResponseMeta(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    latency_ms: int | None = None


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: str | None = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ErrorPayload | None = None
    meta: ResponseMeta


class StreamEvent(BaseModel):
    event: str
    data: dict[str, Any]
