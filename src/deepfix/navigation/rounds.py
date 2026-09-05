"""Deterministic detection of completed action tool rounds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)


@dataclass(frozen=True)
class ToolRoundDelta:
    count: int
    latest_round_id: str | None


def completed_tool_rounds_after(
    messages: Sequence[AnyMessage],
    cursor_round_id: str | None,
) -> ToolRoundDelta:
    """Count complete action rounds represented by ``messages`` after a cursor."""

    candidates: list[tuple[str, list[str], list[object], int, int]] = []
    all_round_ids: list[str] = []
    all_call_ids: list[str] = []
    result_ids: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        round_id = message.id
        if not isinstance(round_id, str) or not round_id.strip():
            continue
        all_round_ids.append(round_id)

        call_ids: list[str] = []
        tool_names: list[object] = []
        for call in message.tool_calls:
            call_id = call.get("id") if isinstance(call, dict) else None
            if not isinstance(call_id, str) or not call_id.strip():
                break
            call_ids.append(call_id)
            all_call_ids.append(call_id)
            tool_names.append(call.get("name"))
        else:
            boundary = _next_round_boundary(messages, index + 1)
            candidates.append((round_id, call_ids, tool_names, index + 1, boundary))

    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.id, str) and message.id.strip():
            result_ids.append(message.id)

    duplicate_round_ids = _duplicates(all_round_ids)
    duplicate_call_ids = _duplicates(all_call_ids)
    duplicate_result_ids = _duplicates(result_ids)
    complete_round_ids: list[str] = []
    checkpoint_index = max(
        (index for index, message in enumerate(messages) if _is_checkpoint(message)),
        default=-1,
    )
    for round_id, call_ids, tool_names, start, boundary in candidates:
        if start - 1 <= checkpoint_index:
            continue
        results = [
            message
            for message in messages[start:boundary]
            if isinstance(message, ToolMessage)
        ]
        paired_ids = [message.tool_call_id for message in results]
        if (
            round_id in duplicate_round_ids
            or any(call_id in duplicate_call_ids for call_id in call_ids)
            or any(message.id in duplicate_result_ids for message in results)
            or len(results) != len(call_ids)
            or len(set(paired_ids)) != len(paired_ids)
            or set(paired_ids) != set(call_ids)
            or all(name == "write_todos" for name in tool_names)
        ):
            continue
        complete_round_ids.append(round_id)

    history_replaced = checkpoint_index >= 0 and cursor_round_id not in complete_round_ids
    latest_round_id = (
        complete_round_ids[-1]
        if complete_round_ids
        else (None if history_replaced else cursor_round_id)
    )
    if history_replaced:
        cursor_round_id = None
    if history_replaced:
        return ToolRoundDelta(len(complete_round_ids), latest_round_id)
    if cursor_round_id is None:
        return ToolRoundDelta(len(complete_round_ids), latest_round_id)
    if cursor_round_id not in complete_round_ids:
        return ToolRoundDelta(0, latest_round_id)

    cursor_index = complete_round_ids.index(cursor_round_id)
    return ToolRoundDelta(len(complete_round_ids) - cursor_index - 1, latest_round_id)


def _duplicates(values: Sequence[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def _next_round_boundary(messages: Sequence[AnyMessage], start: int) -> int:
    for index in range(start, len(messages)):
        if _is_round_boundary(messages[index]):
            return index
    return len(messages)


def _is_round_boundary(message: AnyMessage) -> bool:
    """End an AI tool-result block at every intervening non-tool message."""

    return not isinstance(message, ToolMessage)


def _is_checkpoint(message: AnyMessage) -> bool:
    return isinstance(message, SystemMessage) and "_deepfix_snapshot_version" in message.additional_kwargs
