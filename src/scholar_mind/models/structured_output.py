from __future__ import annotations

import asyncio
import inspect
import json
import logging
from time import perf_counter
from typing import Any, Callable

from scholar_mind.models.callbacks import UsageTracker, usage_dict

logger = logging.getLogger(__name__)


async def _ainvoke_model(model: Any, prompt: Any):
    ainvoke = getattr(model, "ainvoke", None)
    if callable(ainvoke):
        result = ainvoke(prompt)
        if inspect.isawaitable(result):
            return await result
        return result
    await asyncio.sleep(0)
    return model.invoke(prompt)


def invoke_structured_output_once(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
) -> tuple[Any, Any, BaseException | None]:
    try:
        response = _structured_output_runnable(llm, schema).invoke(prompt)
        if not isinstance(response, dict):
            return response, response, None
        parsed = response.get("parsed")
        raw = response.get("raw")
        parsing_error = response.get("parsing_error")
        if parsed is None and raw is not None:
            recovered = recover(raw) if recover is not None else None
            if recovered is None:
                recovered = recover_structured_output(schema, raw)
            if recovered is not None:
                return recovered, raw, None
        return parsed, raw, parsing_error
    except BaseException as exc:
        return None, None, exc


async def ainvoke_structured_output_once(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
) -> tuple[Any, Any, BaseException | None]:
    try:
        runnable = _structured_output_runnable(llm, schema)
        response = await _ainvoke_model(runnable, prompt)
        if not isinstance(response, dict):
            return response, response, None
        parsed = response.get("parsed")
        raw = response.get("raw")
        parsing_error = response.get("parsing_error")
        if parsed is None and raw is not None:
            recovered = recover(raw) if recover is not None else None
            if recovered is None:
                recovered = recover_structured_output(schema, raw)
            if recovered is not None:
                return recovered, raw, None
        return parsed, raw, parsing_error
    except BaseException as exc:
        return None, None, exc


def invoke_structured_output(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
):
    output, usage, _ = invoke_structured_output_with_raw(
        llm,
        prompt,
        schema,
        recover=recover,
    )
    return output, usage


async def ainvoke_structured_output(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
):
    output, usage, _ = await ainvoke_structured_output_with_raw(
        llm,
        prompt,
        schema,
        recover=recover,
    )
    return output, usage


def invoke_structured_output_with_raw(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
):
    if llm is None:
        return None, empty_usage(), None
    try:
        started = perf_counter()
        output, raw, parsing_error = invoke_structured_output_once(
            llm,
            prompt,
            schema,
            recover=recover,
        )
        if output is None:
            expected_schema = expected_schema_text(schema)
            logger.error(
                "Structured output parse failed: reason=%s, expected_schema=%s, actual_schema=%s",
                str(parsing_error or "Unknown parsing error"),
                expected_schema,
                actual_schema_text(raw),
            )
            if not should_retry_structured_output(parsing_error, raw):
                return None, empty_usage(), raw
            first_usage = usage_from_result(
                prompt,
                raw_output_text(raw),
                perf_counter() - started,
                raw,
            )
            retry_prompt = (
                "You must output valid JSON matching this schema.\n"
                "Previous output failed to parse.\n"
                f"Error: {parsing_error or 'Unknown parsing error'}\n"
                f"Previous output: {raw_output_text(raw)}\n"
                f"correct output schema: {expected_schema}"
            )
            retry_started = perf_counter()
            retry_output, retry_raw, retry_error = invoke_structured_output_once(
                llm,
                retry_prompt,
                schema,
                recover=recover,
            )
            if retry_output is None:
                if retry_error is not None:
                    logger.error(
                        "Structured output retry failed: reason=%s, expected_schema=%s, actual_schema=%s",
                        str(retry_error),
                        expected_schema,
                        actual_schema_text(retry_raw),
                    )
                return None, empty_usage(), retry_raw or raw
            retry_usage = usage_from_result(
                retry_prompt,
                retry_output.model_dump_json()
                if hasattr(retry_output, "model_dump_json")
                else str(retry_output),
                perf_counter() - retry_started,
                retry_raw,
            )
            return retry_output, merge_usage(first_usage, retry_usage), retry_raw
        usage = usage_from_result(
            prompt,
            output.model_dump_json() if hasattr(output, "model_dump_json") else str(output),
            perf_counter() - started,
            raw,
        )
        return output, usage, raw
    except Exception:
        return None, empty_usage(), None


async def ainvoke_structured_output_with_raw(
    llm,
    prompt: str,
    schema,
    recover: Callable[[Any], Any | None] | None = None,
):
    if llm is None:
        return None, empty_usage(), None
    try:
        started = perf_counter()
        output, raw, parsing_error = await ainvoke_structured_output_once(
            llm,
            prompt,
            schema,
            recover=recover,
        )
        if output is None:
            expected_schema = expected_schema_text(schema)
            logger.error(
                "Structured output parse failed: reason=%s, expected_schema=%s, actual_schema=%s",
                str(parsing_error or "Unknown parsing error"),
                expected_schema,
                actual_schema_text(raw),
            )
            if not should_retry_structured_output(parsing_error, raw):
                return None, empty_usage(), raw
            first_usage = usage_from_result(
                prompt,
                raw_output_text(raw),
                perf_counter() - started,
                raw,
            )
            retry_prompt = (
                "You must output valid JSON matching this schema.\n"
                "Previous output failed to parse.\n"
                f"Error: {parsing_error or 'Unknown parsing error'}\n"
                f"Previous output: {raw_output_text(raw)}\n"
                f"correct output schema: {expected_schema}"
            )
            retry_started = perf_counter()
            retry_output, retry_raw, retry_error = await ainvoke_structured_output_once(
                llm,
                retry_prompt,
                schema,
                recover=recover,
            )
            if retry_output is None:
                if retry_error is not None:
                    logger.error(
                        "Structured output retry failed: reason=%s, expected_schema=%s, actual_schema=%s",
                        str(retry_error),
                        expected_schema,
                        actual_schema_text(retry_raw),
                    )
                return None, empty_usage(), retry_raw or raw
            retry_usage = usage_from_result(
                retry_prompt,
                retry_output.model_dump_json()
                if hasattr(retry_output, "model_dump_json")
                else str(retry_output),
                perf_counter() - retry_started,
                retry_raw,
            )
            return retry_output, merge_usage(first_usage, retry_usage), retry_raw
        usage = usage_from_result(
            prompt,
            output.model_dump_json() if hasattr(output, "model_dump_json") else str(output),
            perf_counter() - started,
            raw,
        )
        return output, usage, raw
    except Exception:
        return None, empty_usage(), None


def _structured_output_runnable(llm, schema):
    structured_output = llm.with_structured_output
    kwargs: dict[str, Any] = {"include_raw": True}
    if _accepts_keyword(structured_output, "method"):
        kwargs["method"] = "function_calling"
    return structured_output(schema, **kwargs)


def _accepts_keyword(func, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def recover_structured_output(schema, raw) -> Any:
    payload = extract_json_candidate(raw_output_text(raw).strip())
    if payload is None:
        return None
    if hasattr(schema, "model_validate"):
        try:
            return schema.model_validate(payload)
        except Exception:
            return None
    return payload


def expected_schema_text(schema) -> str:
    if isinstance(schema, dict):
        return json.dumps(schema, ensure_ascii=False, sort_keys=True)
    if hasattr(schema, "model_json_schema"):
        return json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    return str(schema)


def raw_output_text(raw) -> str:
    if raw is None:
        return ""
    content = getattr(raw, "content", raw)
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def extract_json_candidate(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    attempts = [candidate]
    for opening, closing in [("{", "}"), ("[", "]")]:
        start = candidate.find(opening)
        end = candidate.rfind(closing)
        if start != -1 and end != -1 and end > start:
            attempts.append(candidate[start : end + 1])
    for item in attempts:
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            continue
    return None


def schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: schema_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return ["unknown"]
        return [schema_shape(value[0])]
    return type(value).__name__


def actual_schema_text(raw) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, (dict, list)):
        return json.dumps(schema_shape(content), ensure_ascii=False, sort_keys=True)
    text = str(content).strip()
    if not text:
        return "empty"
    try:
        return json.dumps(schema_shape(json.loads(text)), ensure_ascii=False, sort_keys=True)
    except json.JSONDecodeError:
        return type(content).__name__


def should_retry_structured_output(parsing_error: BaseException | None, raw) -> bool:
    if parsing_error is None:
        return False
    previous_output = raw_output_text(raw).strip()
    if not previous_output:
        return False
    lowered = str(parsing_error).lower()
    return any(
        token in lowered for token in ["parse", "json", "schema", "validation", "field", "format"]
    ) or isinstance(parsing_error, (ValueError, TypeError, json.JSONDecodeError))


def empty_usage() -> dict[str, float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0,
    }


def merge_usage(*payloads: dict[str, float] | None) -> dict[str, float]:
    merged = empty_usage()
    for payload in payloads:
        if not payload:
            continue
        for key in merged:
            merged[key] += float(payload.get(key, 0))
    merged["prompt_tokens"] = int(merged["prompt_tokens"])
    merged["completion_tokens"] = int(merged["completion_tokens"])
    merged["total_tokens"] = int(merged["total_tokens"])
    merged["latency_ms"] = int(merged["latency_ms"])
    return merged


def usage_from_result(
    prompt: Any,
    completion: Any,
    timecost_s: float,
    result,
) -> dict[str, float]:
    usage = UsageTracker()
    latency_ms = int(timecost_s * 1000)
    usage_meta = getattr(result, "usage_metadata", None)
    if usage_meta:
        prompt_tokens = int(usage_meta.get("input_tokens", 0))
        completion_tokens = int(usage_meta.get("output_tokens", 0))
        total_tokens = int(usage_meta.get("total_tokens", 0))
    else:
        token_usage = getattr(result, "response_metadata", {}).get("token_usage", {})
        prompt_tokens = int(token_usage.get("prompt_tokens", 0))
        completion_tokens = int(token_usage.get("completion_tokens", 0))
        total_tokens = int(token_usage.get("total_tokens", 0))
    if total_tokens == 0:
        prompt_text = str(prompt)
        completion_text = str(completion)
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(completion_text.split())
        total_tokens = prompt_tokens + completion_tokens
    usage.record(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )
    return usage_dict(usage.usage)
