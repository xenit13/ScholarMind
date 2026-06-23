from __future__ import annotations

from dataclasses import dataclass

from scholar_mind.eval.locomo_build.schema import Persona

CASES_PER_PERSONA = 6
SESSIONS_PER_PERSONA = 6
PAPERS_PER_CASE = 5

MEMORY_TYPES = (
    "paper_read",
    "workflow",
    "preference",
    "feedback",
    "knowledge_level",
    "project_constraint",
)

SEEDS_PER_PERSONA = CASES_PER_PERSONA * len(MEMORY_TYPES)

PAPER_CATEGORIES = ("cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.HC", "stat.ML")

RESEARCH_TASKS = (
    "memory evaluation",
    "retrieval-grounded study planning",
    "cross-domain hypothesis design",
    "paper reading personalization",
    "research workflow automation",
    "long-context assistant reliability",
)

TEMPORAL_FRACTION = 0.30


@dataclass(slots=True, frozen=True)
class PaperRecord:
    arxiv_id: str
    title: str
    category: str


def _build_persona(persona_id: str, background: str) -> Persona:
    return Persona(
        persona_id=persona_id,
        user_id=f"locomo_v2_{persona_id}",
        background=background,
    )


PERSONAS: tuple[Persona, ...] = (
    _build_persona("p01", "ML 工程师,强 Python 工程能力、弱因果推断数学"),
    _build_persona("p02", "博士生,理论功底扎实、工程经验有限"),
    _build_persona("p03", "产品经理,跨域协作、关心交付节奏而非实现细节"),
    _build_persona("p04", "跨学科 researcher,关注方法迁移和方法论对比"),
    _build_persona("p05", "工业界 R&D,关注落地成本、可复现性、生产稳定性"),
)


def get_distractor_case_id(case_id: str) -> str:
    """Return the next case_id in the cyclic sequence case_001..case_006 → case_001.

    The distractor always differs from the input case (cycle length = CASES_PER_PERSONA).
    Raises ValueError if case_id is not in the form `case_NNN` where NNN ∈ [001, CASES_PER_PERSONA].
    """
    try:
        prefix, n_str = case_id.split("_")
    except ValueError as exc:
        raise ValueError(f"unexpected case_id format: {case_id}") from exc
    if prefix != "case" or not n_str.isdigit():
        raise ValueError(f"unexpected case_id format: {case_id}")
    n = int(n_str)
    if not 1 <= n <= CASES_PER_PERSONA:
        raise ValueError(
            f"case_id out of range [1,{CASES_PER_PERSONA}]: {case_id}"
        )
    next_n = (n % CASES_PER_PERSONA) + 1
    return f"case_{next_n:03d}"


def build_persona_case_topic(category: str, research_task: str) -> str:
    return f"{category} {research_task}"
