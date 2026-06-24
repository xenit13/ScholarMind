from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import typer

from scholar_mind.config.settings import get_settings
from scholar_mind.eval.locomo_build.dialogue import build_persona_conversation
from scholar_mind.eval.locomo_build.questions import build_persona_qas
from scholar_mind.eval.locomo_build.schema import Sample
from scholar_mind.eval.locomo_build.seeds import (
    PERSONAS,
    build_all_seeds,
    load_paper_pool,
    write_seeds_json,
)
from scholar_mind.eval.locomo_build.validate import validate_samples
from scholar_mind.models.factory import build_chat_models

app = typer.Typer(help="LOCOMO v2 dataset builder")
logger = logging.getLogger(__name__)

# Module-level typer.Option singletons so ruff's B008 does not flag calls in
# argument defaults. Mirrors the pattern in scholar_mind.main.PAPER_IDS_OPTION.
SEEDS_OUT_OPTION = typer.Option(Path("data/eval/locomo_build/seeds.json"))
SEEDS_IN_OPTION = typer.Option(Path("data/eval/locomo_build/seeds.json"))
DIALOGUES_OUT_OPTION = typer.Option(Path("data/eval/locomo_build/dialogues.json"))
DIALOGUES_IN_OPTION = typer.Option(Path("data/eval/locomo_build/dialogues.json"))
SAMPLES_OUT_OPTION = typer.Option(Path("data/eval/locomo_build/scholarmind_locomo_v2.json"))
SAMPLES_IN_OPTION = typer.Option(Path("data/eval/locomo_build/scholarmind_locomo_v2.json"))
VALIDATION_OUT_OPTION = typer.Option(Path("data/eval/locomo_build/validation_report.json"))
RUN_OUT_DIR_OPTION = typer.Option(Path("data/eval/locomo_build"))


def _load_chat_model():
    settings = get_settings()
    models = build_chat_models(settings)
    model = models.get("reasoning") or models.get("light")
    if model is None:
        raise RuntimeError(
            "no chat model configured; set SCHOLARMIND_LLM_API_KEY and SCHOLARMIND_LLM_BASE_URL"
        )
    return model


@app.command()
def seeds(
    database_url: str = typer.Option(..., help="SQLite database URL"),
    out: Path = SEEDS_OUT_OPTION,
    seed: int = typer.Option(42),
):
    """Stage 1: build deterministic memory seeds."""
    rng = random.Random(seed)
    pool = load_paper_pool(database_url)
    by_persona = build_all_seeds(pool, rng=rng)
    write_seeds_json(by_persona, out)
    typer.echo(f"wrote seeds for {len(by_persona)} personas to {out}")


@app.command()
def dialogues(
    seeds_file: Path = SEEDS_IN_OPTION,
    out: Path = DIALOGUES_OUT_OPTION,
):
    """Stage 2: expand seeds into natural dialogues via LLM."""
    chat_model = _load_chat_model()
    raw = json.loads(seeds_file.read_text(encoding="utf-8"))
    conversations: dict[str, dict] = {}
    for persona in PERSONAS:
        persona_seeds = raw[persona.persona_id]
        by_case: dict[str, list[dict]] = {}
        for seed in persona_seeds:
            by_case.setdefault(seed["case_id"], []).append(
                {
                    "seed_id": seed["seed_id"],
                    "memory_type": seed["memory_type"],
                    "content": seed["content"],
                }
            )
        conversations[persona.persona_id] = build_persona_conversation(
            chat_model=chat_model,
            persona=persona,
            seeds_by_case=by_case,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"wrote dialogues for {len(conversations)} personas to {out}")


@app.command()
def qas(
    seeds_file: Path = SEEDS_IN_OPTION,
    dialogues_file: Path = DIALOGUES_IN_OPTION,
    out: Path = SAMPLES_OUT_OPTION,
):
    """Stage 3: generate QAs and assemble final test set."""
    chat_model = _load_chat_model()
    raw_seeds = json.loads(seeds_file.read_text(encoding="utf-8"))
    raw_dialogues = json.loads(dialogues_file.read_text(encoding="utf-8"))

    samples: list[Sample] = []
    for persona in PERSONAS:
        persona_seeds = raw_seeds[persona.persona_id]
        conversation = raw_dialogues[persona.persona_id]
        seed_to_dia: dict[str, list[str]] = {}
        dialogue_texts: list[str] = []
        for key, turns in conversation.items():
            if not key.startswith("session_") or key.endswith("_date_time"):
                continue
            for turn in turns:
                dialogue_texts.append(turn["text"])
                sid = turn.get("metadata", {}).get("seed_id")
                if sid:
                    seed_to_dia.setdefault(sid, []).append(turn["dia_id"])

        seeds_per_case: list[dict] = []
        for case_id in sorted({s["case_id"] for s in persona_seeds}):
            case_seeds = [s for s in persona_seeds if s["case_id"] == case_id]
            seeds_per_case.append(
                {
                    "case_id": case_id,
                    "case_topic": case_seeds[0]["case_topic"],
                    "seeds": case_seeds,
                }
            )
        qas = build_persona_qas(
            chat_model=chat_model,
            persona_id=persona.persona_id,
            seeds_per_case=seeds_per_case,
            seed_to_dia_lookup=seed_to_dia,
            dialogue_texts=dialogue_texts,
        )
        samples.append(
            Sample(
                sample_id=f"scholarmind_locomo_v2_{persona.persona_id}",
                persona=persona,
                conversation=conversation,
                qa=qas,
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([s.model_dump() for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"wrote {len(samples)} samples to {out}")


@app.command()
def validate(
    samples_file: Path = SAMPLES_IN_OPTION,
    out: Path = VALIDATION_OUT_OPTION,
):
    """Stage 4: run gold/random/structural sanity checks."""
    samples = json.loads(samples_file.read_text(encoding="utf-8"))
    report = validate_samples(samples)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"validation report: {report}")
    if not report["structural_check_passed"] or report["gold_overall_accuracy"] < 0.99:
        raise typer.Exit(code=1)


@app.command()
def run(
    database_url: str = typer.Option(...),
    out_dir: Path = RUN_OUT_DIR_OPTION,
    seed: int = typer.Option(42),
):
    """Run all 4 stages end-to-end."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = out_dir / "seeds.json"
    dialogues_path = out_dir / "dialogues.json"
    samples_path = out_dir / "scholarmind_locomo_v2.json"
    validation_path = out_dir / "validation_report.json"

    seeds(database_url=database_url, out=seeds_path, seed=seed)
    dialogues(seeds_file=seeds_path, out=dialogues_path)
    qas(seeds_file=seeds_path, dialogues_file=dialogues_path, out=samples_path)
    validate(samples_file=samples_path, out=validation_path)
