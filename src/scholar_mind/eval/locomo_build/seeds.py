from __future__ import annotations

import random
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


def sample_papers_for_persona(
    pool: list[PaperRecord],
    persona_id: str,
    *,
    papers_needed: int,
    rng: random.Random,
    used_arxiv_ids: set[str] | None = None,
) -> list[PaperRecord]:
    """Sample distinct papers for one persona, avoiding re-use across personas.

    Balances across PAPER_CATEGORIES so each persona sees all 6 categories.
    Raises ValueError if any category has insufficient unused papers.

    Note: ``persona_id`` is accepted for caller-side identification/logging and
    to keep the function signature self-documenting; it is not consulted by the
    sampling logic itself (cross-persona uniqueness is enforced via
    ``used_arxiv_ids``).
    """
    _ = persona_id  # accepted for API clarity; not used by sampling logic
    if used_arxiv_ids is None:
        used_arxiv_ids = set()
    available_by_cat: dict[str, list[PaperRecord]] = {
        cat: [] for cat in PAPER_CATEGORIES
    }
    for paper in pool:
        if paper.arxiv_id in used_arxiv_ids:
            continue
        if paper.category in available_by_cat:
            available_by_cat[paper.category].append(paper)

    per_cat = papers_needed // len(PAPER_CATEGORIES)
    remainder = papers_needed - per_cat * len(PAPER_CATEGORIES)
    chosen: list[PaperRecord] = []
    for idx, cat in enumerate(PAPER_CATEGORIES):
        target = per_cat + (1 if idx < remainder else 0)
        candidates = available_by_cat[cat]
        if len(candidates) < target:
            raise ValueError(
                f"paper pool exhausted for category {cat}: needed {target}, "
                f"available {len(candidates)}"
            )
        chosen.extend(rng.sample(candidates, target))
    return chosen
