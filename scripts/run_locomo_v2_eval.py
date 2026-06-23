#!/usr/bin/env python3
"""Run end-to-end LOCOMO v2 evaluation against the running ScholarMind stack.

Usage:
    PYTHONPATH=src uv run python scripts/run_locomo_v2_eval.py \\
        --samples data/eval/locomo_build/scholarmind_locomo_v2.json \\
        --out data/eval/locomo_build/predictions.json \\
        [--limit 60]

Outputs:
    --out: predictions JSON (samples with scholarmind/scholarmind_context fields on each QA)
    --stats: scoring report JSON (per-category accuracy + recall)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import anyio  # noqa: E402

from scholar_mind.app import get_container  # noqa: E402
from scholar_mind.config.settings import get_settings  # noqa: E402
from scholar_mind.eval.locomo import score_locomo_samples  # noqa: E402
from scholar_mind.eval.locomo_v2_runner import run_locomo_v2_eval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LOCOMO v2 evaluation through ScholarMind's full stack.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("data/eval/locomo_build/scholarmind_locomo_v2.json"),
        help="Path to the test set produced by build_locomo_dataset.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/locomo_build/predictions.json"),
        help="Where to write predictions JSON (samples with prediction fields)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/eval/locomo_build/predictions_stats.json"),
        help="Where to write scoring stats JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total QAs across all personas (for smoke testing)",
    )
    parser.add_argument(
        "--prediction-key",
        default="scholarmind",
        help="Field name to write predictions under",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("data/eval/locomo_build/predictions_progress.json"),
        help="Incremental progress file (overwritten each QA)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("locomo_v2_eval")

    settings = get_settings()
    logger.info(
        "settings: env=%s, llm=%s, qdrant=%s",
        settings.environment,
        settings.llm_reasoning_model,
        settings.qdrant_location or settings.qdrant_url,
    )

    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    logger.info(
        "loaded %d samples, %d total QAs",
        len(samples),
        sum(len(s.get("qa", [])) for s in samples),
    )
    if args.limit:
        logger.info("limiting to %d QAs (smoke test)", args.limit)

    container = get_container()
    research_service = container.research_service

    predictions = await run_locomo_v2_eval(
        research_service=research_service,
        samples=samples,
        prediction_key=args.prediction_key,
        limit=args.limit,
        progress_file=args.progress,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote predictions to %s", args.out)

    _, report = score_locomo_samples(
        predictions,
        prediction_key=args.prediction_key,
        model_name=args.prediction_key,
        skip_missing_predictions=True,
    )
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote stats to %s", args.stats)

    summary = report[args.prediction_key]
    print("\n=== LOCOMO v2 Eval Summary ===")
    print(f"Total QAs scored: {summary['question_count']}")
    print(f"Non-empty predictions: {summary['prediction_nonempty']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    print("Per-category accuracy:")
    for cat in sorted(summary["accuracy_by_category"], key=int):
        acc = summary["accuracy_by_category"][cat]
        print(f"  cat {cat}: {acc:.3f}")
    if "recall_by_category" in summary:
        print("Per-category evidence recall:")
        for cat in sorted(summary["recall_by_category"], key=int):
            rec = summary["recall_by_category"][cat]
            print(f"  cat {cat}: {rec:.3f}")
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(anyio.run(main, args))
