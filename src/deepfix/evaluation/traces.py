from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepfix.evaluation.models import RunUsage

_MAX_TRACE_LINE_CHARS = 1_000_000


def _valid_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def summarize_llm_trace(path: Path, task_id: str) -> RunUsage:
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    model_seconds = 0.0
    usage_seen = False

    if not path.is_file():
        return RunUsage(
            input_tokens=0,
            output_tokens=0,
            model_calls=0,
            tool_calls=0,
            wall_seconds=0,
            usage_estimated=True,
        )

    with path.open("r", encoding="utf-8", errors="replace") as trace:
        for line in trace:
            if len(line) > _MAX_TRACE_LINE_CHARS:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            if record.get("event") != "response" or record.get("task_id") != task_id:
                continue

            raw_input = record.get("input_tokens", 0)
            raw_output = record.get("output_tokens", 0)
            raw_duration = record.get("duration_seconds", 0)
            if not _valid_nonnegative_int(raw_input):
                continue
            if not _valid_nonnegative_int(raw_output):
                continue
            if not _valid_nonnegative_number(raw_duration):
                continue

            usage_seen = usage_seen or (
                "input_tokens" in record or "output_tokens" in record
            )
            input_tokens += raw_input
            output_tokens += raw_output
            model_seconds += float(raw_duration)
            model_calls += 1

    return RunUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_calls=model_calls,
        tool_calls=0,
        wall_seconds=model_seconds,
        usage_estimated=not usage_seen or input_tokens + output_tokens == 0,
    )
