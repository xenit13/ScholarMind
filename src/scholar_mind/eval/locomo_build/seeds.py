from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from scholar_mind.eval.locomo_build.schema import PaperRef, Persona, Seed, TemporalUpdate

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

    Returns papers grouped by PAPER_CATEGORIES order (cs.AI block, then cs.CL, ...).
    Callers that slice contiguously and read [0].category depend on this ordering.

    Balances across PAPER_CATEGORIES so each persona sees all 6 categories.
    Raises ValueError if any category has insufficient unused papers.
    ``persona_id`` is included in error messages for caller diagnostics.
    """
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
                f"paper pool exhausted for persona {persona_id} category {cat}: "
                f"needed {target}, available {len(candidates)}"
            )
        chosen.extend(rng.sample(candidates, target))
    return chosen


_DEFAULT_BASE_DATE = date(2026, 5, 3)
_SESSION_GAP_DAYS = 3


def _case_dates(case_index: int) -> tuple[str, str, str]:
    base = _DEFAULT_BASE_DATE + timedelta(days=case_index * _SESSION_GAP_DAYS)
    return (
        base.isoformat(),
        (base - timedelta(days=30)).isoformat(),
        (base + timedelta(days=30)).isoformat(),
    )


def _seed_content(
    memory_type: str,
    case_topic: str,
    papers: list[PaperRef],
    persona_background: str,
    rng: random.Random,
) -> dict:
    if memory_type == "paper_read":
        primary = papers[0]
        return {
            "role": rng.choice(["anchor paper", "baseline candidate", "ablation target"]),
            "paper_title": primary.title,
        }
    if memory_type == "workflow":
        return {
            "outputs": rng.sample(
                [
                    "method assumptions",
                    "failure modes",
                    "dataset fit",
                    "implementation risks",
                    "evaluation protocol",
                ],
                2,
            ),
            "paper_titles": [p.title for p in papers[:2]],
        }
    if memory_type == "preference":
        return {
            "default_depth": rng.choice(
                ["survey-first overview", "implementation-first notes", "results-first summary"]
            )
        }
    if memory_type == "feedback":
        return {
            "style_tag": rng.choice(
                [
                    "intuition-first formula explanation",
                    "rigor-first derivation",
                    "example-first walkthrough",
                ]
            )
        }
    if memory_type == "knowledge_level":
        return {"background": persona_background}
    if memory_type == "project_constraint":
        return {
            "requirement": rng.choice(
                [
                    "start with limitations before novelty",
                    "include reproducibility checklist",
                    "map methods into education technology",
                    "flag production stability concerns",
                ]
            )
        }
    raise ValueError(f"unknown memory_type {memory_type}")


def _maybe_temporal(
    memory_type: str,
    content: dict,
    rng: random.Random,
    case_index: int,
) -> TemporalUpdate | None:
    if memory_type not in {"preference", "feedback"}:
        return None
    if rng.random() > TEMPORAL_FRACTION:
        return None
    _, old_date, new_date = _case_dates(case_index)
    alt_content: dict[str, str] = {}
    for key, value in content.items():
        alt_pool = {
            "default_depth": [
                "survey-first overview",
                "implementation-first notes",
                "results-first summary",
            ],
            "style_tag": [
                "intuition-first formula explanation",
                "rigor-first derivation",
                "example-first walkthrough",
            ],
        }
        pool = alt_pool.get(key)
        if pool:
            alt = rng.choice([opt for opt in pool if opt != value])
            alt_content[key] = alt
    if not alt_content:
        return None
    return TemporalUpdate(
        old=alt_content, new=content, old_date=old_date, new_date=new_date
    )


def build_seeds_for_persona(
    persona: Persona,
    pool: list[PaperRecord],
    *,
    rng: random.Random,
    used_arxiv_ids: set[str] | None = None,
) -> list[Seed]:
    """Build 36 seeds for one persona: 6 cases x 6 memory_types.

    Papers are sampled distinctly (per-persona) and grouped 5 per case in
    PAPER_CATEGORIES order. See sample_papers_for_persona docstring.

    If ``used_arxiv_ids`` is provided, it is mutated in place to include the
    sampled papers.
    """
    chosen_papers = sample_papers_for_persona(
        pool,
        persona.persona_id,
        papers_needed=CASES_PER_PERSONA * PAPERS_PER_CASE,
        rng=rng,
        used_arxiv_ids=used_arxiv_ids,
    )
    seeds: list[Seed] = []
    for case_idx in range(CASES_PER_PERSONA):
        case_id = f"case_{case_idx + 1:03d}"
        case_papers = chosen_papers[
            case_idx * PAPERS_PER_CASE : (case_idx + 1) * PAPERS_PER_CASE
        ]
        category = case_papers[0].category
        research_task = RESEARCH_TASKS[case_idx % len(RESEARCH_TASKS)]
        case_topic = build_persona_case_topic(category, research_task)
        distractor_case_id = get_distractor_case_id(case_id)
        for memory_type in MEMORY_TYPES:
            content = _seed_content(
                memory_type, case_topic, case_papers, persona.background, rng
            )
            temporal = _maybe_temporal(memory_type, content, rng, case_idx)
            paper_refs = [PaperRef(**asdict(p)) for p in case_papers]
            seeds.append(
                Seed(
                    seed_id=f"{persona.persona_id}_{case_id}_{memory_type}",
                    persona_id=persona.persona_id,
                    case_id=case_id,
                    case_topic=case_topic,
                    papers=paper_refs,
                    memory_type=memory_type,
                    content=content,
                    temporal=temporal,
                    distractor_case_id=distractor_case_id,
                )
            )
        if used_arxiv_ids is not None:
            for p in case_papers:
                used_arxiv_ids.add(p.arxiv_id)
    return seeds


def build_all_seeds(
    pool: list[PaperRecord],
    *,
    rng: random.Random,
) -> dict[str, list[Seed]]:
    """Build seeds for all 5 personas, sharing the used-arxiv-id set across personas."""
    used: set[str] = set()
    out: dict[str, list[Seed]] = {}
    for persona in PERSONAS:
        out[persona.persona_id] = build_seeds_for_persona(
            persona, pool, rng=rng, used_arxiv_ids=used
        )
    return out
