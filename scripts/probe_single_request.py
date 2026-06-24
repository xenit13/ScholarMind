#!/usr/bin/env python3
"""Single-request probe: import ONE transcript round and inspect memory extraction.

Isolates the wait+flag mechanism from the full runner so we can see exactly
what's working and what isn't. Runs in ~30s instead of ~30min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import anyio  # noqa: E402

from scholar_mind.app import get_container  # noqa: E402
from scholar_mind.config.settings import get_settings  # noqa: E402
from scholar_mind.eval.locomo_v2_runner import (  # noqa: E402
    _iter_transcript_rounds,
    _round_messages_payload,
)


async def count_memories_for_user(memory_manager, user_id: str) -> int:
    """Query memory DB for how many records exist for this user."""
    memory_repository = getattr(memory_manager, "memory_repository", None)
    if memory_repository is None:
        return -1
    try:
        # List all memories regardless of status
        rows = memory_repository.list_memories(user_id=user_id, status=None)
        return len(rows)
    except Exception as exc:
        print(f"  query failed: {exc}")
        return -1


async def main() -> int:
    settings = get_settings()
    print(f"settings: model={settings.llm_reasoning_model}, env={settings.environment}")

    container = get_container()
    research_service = container.research_service
    memory_manager = research_service.memory_manager

    # Pick the first transcript round from p01
    samples_path = Path("data/eval/locomo_build/scholarmind_locomo_v2.json")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    sample = samples[0]
    user_id = sample["persona"]["user_id"]
    session_index, session_date, first_round_turns = _iter_transcript_rounds(
        sample["conversation"]
    )[0]
    first_turn = first_round_turns[0]
    first_qa = sample["qa"][0]
    round_messages = _round_messages_payload(
        turns=first_round_turns,
        session_id=f"{user_id}-probe-transcript",
        session_index=session_index,
        session_date=session_date,
        round_index=1,
    )

    print(f"user_id: {user_id}")
    print(f"first turn seed_id: {first_turn['metadata']['seed_id']}")
    print(f"first QA: {first_qa['question'][:80]}")
    print(f"gold answer: {first_qa['answer']}")

    print("\n=== Async dispatch probe: extract_transcript_memories ===")
    research_service.extract_transcript_memories(
        user_id=user_id,
        request_id="probe_transcript_round",
        session_id=f"{user_id}-probe-transcript",
        round_messages=round_messages,
    )
    pending_count_before = len(research_service._pending_extractions)
    print(f"pending tasks: {pending_count_before}")
    summary = research_service.wait_for_pending_extractions(timeout=120)
    print(f"wait result: {summary}")

    # === Direct probe: call extract_request_memories synchronously ===
    print("\n=== Direct probe: extract_memory_candidates_from_round ===")
    from scholar_mind.memory.extraction import extract_memory_candidates_from_round

    # Step 1: extract candidates
    candidates, ext_usage, ext_success = extract_memory_candidates_from_round(
        memory_manager.llm,
        round_messages,
        explicit_memories=[],
    )
    print(
        f"extraction success: {ext_success}, "
        f"candidates: {len(candidates)}, usage: {ext_usage}"
    )
    for i, c in enumerate(candidates):
        print(f"  candidate[{i}]: type={c.memory_type}, content={c.content[:80]!r}")

    # === Step 1.5: raw LLM call to see what model actually returns ===
    print("\n=== Raw LLM call ===")
    from scholar_mind.agents.common import invoke_structured_output_with_raw
    from scholar_mind.memory.extraction import _build_candidate_extraction_prompt
    from scholar_mind.models.domain import MemoryCandidateExtractionOutput

    prompt = _build_candidate_extraction_prompt(round_messages)
    parsed, raw, error = invoke_structured_output_with_raw(
        memory_manager.llm, prompt, MemoryCandidateExtractionOutput,
    )
    print(f"parsed: {parsed!r}")
    if hasattr(raw, "content"):
        print("raw content (first 2000 chars):")
        print(str(raw.content)[:2000])

    # Step 2: admission for each
    print("\n--- admission decisions ---")
    for i, c in enumerate(candidates):
        decision, _adm_usage = memory_manager.admission_policy.evaluate(
            c,
            llm=memory_manager.llm,
        )
        print(
            f"  candidate[{i}]: action={decision.action}, "
            f"reason={decision.reason}, matched_rules={decision.matched_rules}"
        )

    # Check memory count
    print("\n=== memory count AFTER direct call ===")
    after_direct = await count_memories_for_user(memory_manager, user_id)
    print(f"memories for {user_id}: {after_direct}")

    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
