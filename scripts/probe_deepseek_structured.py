#!/usr/bin/env python3
"""Debug probe: capture what DeepSeek returns for a memory extraction call.

Goal: see the raw model output when we call invoke_structured_output with
MemoryCandidateExtractionOutput. Tells us whether the model:
(a) returns valid JSON but missing required fields,
(b) returns JSON wrapped in thinking/reasoning tags,
(c) returns an empty response,
(d) something else.
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

from scholar_mind.agents.common import invoke_structured_output_with_raw  # noqa: E402
from scholar_mind.config.settings import get_settings  # noqa: E402
from scholar_mind.memory.admission import MemoryAdmissionPolicy, MemoryAdmissionModelOutput  # noqa: E402
from scholar_mind.models.factory import build_chat_models  # noqa: E402
from scholar_mind.models.domain import MemoryCandidate, MemoryCandidateExtractionOutput, MemoryType  # noqa: E402


PROMPT = """Extract memory candidates from this user message.

User message: 我把论文《SWE-chat》标为 anchor paper，用于 cs.AI memory evaluation。

Output a JSON object with a "candidates" array. Each candidate must have:
- memory_type: one of [preference, research_interest, knowledge_level, goal, workflow, project_constraint, paper_read, interaction_summary, feedback]
- content: the memory text
- source: one of [explicit, conversation, system_extracted]
"""


async def main() -> int:
    settings = get_settings()
    print(f"Model: {settings.llm_reasoning_model}")
    print(f"Base URL: {settings.llm_base_url}")
    print()

    models = build_chat_models(settings)
    llm = models.get("reasoning") or models.get("light")
    if llm is None:
        print("No LLM configured")
        return 1

    print(f"LLM type: {type(llm).__name__}")
    print(f"Model name attr: {getattr(llm, 'model_name', None)!r}")
    print()

    print("=" * 60)
    print("Test 1: Extraction")
    print("=" * 60)
    parsed, raw, error = invoke_structured_output_with_raw(
        llm, PROMPT, MemoryCandidateExtractionOutput
    )
    print(f"Parsed: {parsed!r}")
    print(f"Error: {error!r}")
    print()

    print("=" * 60)
    print("Test 2: Admission")
    print("=" * 60)
    candidate = MemoryCandidate(
        memory_type=MemoryType.PAPER_READ,
        content="论文《SWE-chat》被标为 anchor paper，用于 cs.AI memory evaluation。",
        source="explicit",
    )
    policy = MemoryAdmissionPolicy()
    decision, usage = policy.evaluate(candidate, llm=llm)
    print(f"Decision: {decision}")
    print(f"Usage: {usage}")
    print()

    print("=" * 60)
    print("Test 3: Admission raw LLM call")
    print("=" * 60)
    from scholar_mind.memory.admission import _build_model_admission_prompt
    admission_prompt = _build_model_admission_prompt(candidate)
    print(f"Prompt (first 500 chars):\n{admission_prompt[:500]}")
    print()
    parsed_adm, raw_adm, error_adm = invoke_structured_output_with_raw(
        llm, admission_prompt, MemoryAdmissionModelOutput
    )
    print(f"Parsed: {parsed_adm!r}")
    print(f"Error: {error_adm!r}")
    if hasattr(raw_adm, "content"):
        print(f"Raw content (first 1000 chars):")
        print(str(raw_adm.content)[:1000])
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
