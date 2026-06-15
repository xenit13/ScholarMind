# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholar_mind.config.settings import get_settings
from scripts.refine_rag_annotations import (
    ANNOTATIONS_PATH,
    CHINESE_REFERENCE_OVERRIDES_PATH,
    load_json_mapping,
    load_jsonl,
    requires_chinese_answer,
)


GUIDANCE_MARKERS = (
    "理想回答应",
    "围绕目标论文",
    "先交代",
    "再概括",
    "避免编造",
    "答案覆盖目标论文身份",
    "正文阅读重点放在",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Chinese-request RAG references into Chinese factual standards."
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--output", type=Path, default=CHINESE_REFERENCE_OVERRIDES_PATH)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.annotations)
    selected_case_ids = set(args.case_id)
    existing = load_json_mapping(args.output)
    targets = [
        row
        for row in rows
        if requires_chinese_answer(row)
        and (not selected_case_ids or str(row.get("case_id")) in selected_case_ids)
        and (args.force or str(row.get("case_id")) not in existing)
    ]
    if args.limit is not None:
        targets = targets[: args.limit]

    translations = asyncio.run(
        translate_rows(targets, timeout_seconds=args.request_timeout_seconds)
    )
    existing.update(translations)
    args.output.write_text(
        json.dumps(dict(sorted(existing.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "annotations": str(args.annotations),
                "output": str(args.output),
                "translated": len(translations),
                "total_overrides": len(existing),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


async def translate_rows(rows: list[dict], *, timeout_seconds: float) -> dict[str, str]:
    if not rows:
        return {}

    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_api_key:
        raise RuntimeError("LLM base URL and API key are required for translation")
    model = settings.llm_light_model
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    translations: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        reference = str(row.get("reference") or "")
        translated = await asyncio.wait_for(
            translate_reference(client, model, case_id, reference),
            timeout=timeout_seconds,
        )
        translations[case_id] = translated
        print(json.dumps({"case_id": case_id, "chars": len(translated)}, ensure_ascii=False))
    return translations


async def translate_reference(
    client: AsyncOpenAI,
    model: str,
    case_id: str,
    reference: str,
) -> str:
    feedback = ""
    last_error: Exception | None = None
    for _ in range(3):
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严谨的学术评测 reference 翻译器。把输入改写为中文事实型标准答案。"
                        "保留 arXiv ID、论文英文标题、类别、公式、模型名、数据集名、"
                        "指标名和必要缩写；"
                        "其余英文事实句和英文解释短语必须翻译成中文。专业名词可采用“中文译名（英文缩写）”。"
                        "普通技术术语也要中文化，例如 Large Language Models 写为"
                        "“大语言模型（LLMs）”，"
                        "Knowledge Graph Completion 写为“知识图谱补全（KGC）”，"
                        "LLM-generated text 写为“LLM 生成文本”，"
                        "memory retrieval policy 写为“记忆检索策略”。"
                        "除论文标题和必要缩写外，正文不得留下连续 5 个以上英文单词。"
                        "正文应以中文为主，不要把英文短语当作中文句子的主体。"
                        "不得新增事实、不得删除关键方法/实验/结论/局限信息。"
                        "禁止写成评分规则或指导语；禁止出现“理想回答应”“围绕目标论文”“先交代”"
                        "“再概括”“避免编造”等措辞。只输出翻译后的 reference 正文。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"case_id: {case_id}\nreference:\n{reference}{feedback}",
                },
            ],
        )
        content = response.choices[0].message.content or ""
        translated = strip_code_fence(content.strip())
        try:
            validate_translation(case_id, translated)
            return translated
        except ValueError as exc:
            last_error = exc
            feedback = (
                "\n\n上一次输出不合格，原因："
                f"{exc}。请重新输出，翻译掉正文中的长英文片段。"
            )
    assert last_error is not None
    raise last_error


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:\w+)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def validate_translation(case_id: str, reference: str) -> None:
    if not reference:
        raise ValueError(f"{case_id}: empty translated reference")
    for marker in GUIDANCE_MARKERS:
        if marker in reference:
            raise ValueError(f"{case_id}: guidance marker remains: {marker}")
    if not re.search(r"[\u4e00-\u9fff]", reference):
        raise ValueError(f"{case_id}: translated reference contains no Chinese text")
    body = re.sub(r"《[^》]+》", "", reference)
    body = re.sub(r"arXiv:\d{4}\.\d{5}", "", body)
    long_english = re.search(
        r"\b[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){5,}\b",
        body,
    )
    if long_english:
        raise ValueError(
            f"{case_id}: long English fragment remains: {long_english.group(0)[:80]}"
        )


if __name__ == "__main__":
    main()
