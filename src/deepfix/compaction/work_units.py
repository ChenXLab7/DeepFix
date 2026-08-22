from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from deepfix.compaction.identity import stable_work_unit_id
from deepfix.compaction.models import WorkUnit


@dataclass(frozen=True)
class WorkUnitPartition:
    units: list[WorkUnit]
    unassigned_message_ids: list[str]
    diagnostics: list[str]
    safe_cut_indices: list[int]


def partition_work_units(
    messages: Sequence[AnyMessage],
    conflicted_message_ids: set[str] | frozenset[str],
) -> WorkUnitPartition:
    task_id = _task_scope(messages)
    diagnostics: list[str] = []
    all_call_ids = [
        str(call["id"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    duplicate_calls = {
        call_id for call_id, count in Counter(all_call_ids).items() if count > 1
    }
    if duplicate_calls:
        diagnostics.extend(
            f"duplicate tool_call_id: {call_id}" for call_id in sorted(duplicate_calls)
        )

    call_owners = {
        str(call["id"]): index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if str(call["id"]) not in duplicate_calls
    }
    result_indices: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            result_indices.setdefault(str(message.tool_call_id), []).append(index)

    units: list[WorkUnit] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            if call_id not in call_owners or call_id in duplicate_calls:
                diagnostics.append(f"orphan ToolMessage: {message.id or index}")
                units.append(
                    _make_unit(
                        task_id,
                        messages,
                        index,
                        index,
                        [call_id],
                        "ambiguous",
                        conflicted_message_ids,
                    )
                )
                index += 1
                continue

        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = [str(call["id"]) for call in message.tool_calls]
            boundary = _next_hard_boundary(messages, index + 1)
            paired_indices = [
                result_index
                for call_id in call_ids
                for result_index in result_indices.get(call_id, [])
                if result_index > index
            ]
            crosses_boundary = any(result_index >= boundary for result_index in paired_indices)
            if crosses_boundary:
                end = max(paired_indices)
                state = "ambiguous"
                diagnostics.append(f"Tool result crossed user/tool boundary: {message.id}")
            else:
                end = boundary - 1
                while end > index and _starts_hard_boundary(messages[end]):
                    end -= 1
                matched = {
                    str(item.tool_call_id)
                    for item in messages[index + 1 : end + 1]
                    if isinstance(item, ToolMessage)
                }
                state = "complete" if set(call_ids) <= matched else "incomplete"
            if set(call_ids) & duplicate_calls:
                state = "ambiguous"
            units.append(
                _make_unit(
                    task_id,
                    messages,
                    index,
                    max(index, end),
                    call_ids,
                    state,
                    conflicted_message_ids,
                )
            )
            index = max(index, end) + 1
            continue

        start = index
        index += 1
        if isinstance(message, HumanMessage):
            while index < len(messages):
                current = messages[index]
                if isinstance(current, HumanMessage | ToolMessage):
                    break
                if isinstance(current, AIMessage) and current.tool_calls:
                    break
                index += 1
        unit = _make_unit(
            task_id,
            messages,
            start,
            index - 1,
            [],
            "complete",
            conflicted_message_ids,
        )
        units.append(unit)

    safe_cut_indices = [unit.end_index + 1 for unit in units]
    return WorkUnitPartition(units, [], diagnostics, safe_cut_indices)


def _make_unit(
    task_id: str,
    messages: Sequence[AnyMessage],
    start: int,
    end: int,
    tool_call_ids: list[str],
    state: str,
    conflicted_message_ids: set[str] | frozenset[str],
) -> WorkUnit:
    selected = list(messages[start : end + 1])
    message_ids = [str(message.id or f"missing-{index}") for index, message in enumerate(selected, start)]
    if any(message_id in conflicted_message_ids for message_id in message_ids):
        state = "ambiguous"
    categories = _categories(selected)
    purpose = _purpose(selected)
    return WorkUnit(
        unit_id=stable_work_unit_id(task_id, message_ids),
        purpose=purpose,
        message_ids=message_ids,
        tool_call_ids=tool_call_ids,
        state=state,
        categories=categories,
        start_index=start,
        end_index=end,
        must_keep=state != "complete",
    )


def _next_hard_boundary(messages: Sequence[AnyMessage], start: int) -> int:
    for index in range(start, len(messages)):
        if _starts_hard_boundary(messages[index]):
            return index
    return len(messages)


def _starts_hard_boundary(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) or (
        isinstance(message, AIMessage) and bool(message.tool_calls)
    )


def _categories(messages: list[AnyMessage]) -> set[str]:
    categories: set[str] = set()
    tool_names = [
        str(call["name"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    for name in tool_names:
        if name in {"read_file", "ls", "glob"}:
            categories.add("read")
        elif name in {"grep", "search_technical_sources", "fetch_external_evidence"}:
            categories.add("search")
        elif name in {"write_file", "edit_file", "delete"}:
            categories.add("modify")
        elif name == "execute":
            results = [
                message
                for message in messages
                if isinstance(message, ToolMessage)
            ]
            exit_codes = [
                message.artifact.get("exit_code")
                for message in results
                if isinstance(message.artifact, dict)
                and isinstance(message.artifact.get("exit_code"), int)
            ]
            categories.add("verify_pass" if exit_codes and exit_codes[-1] == 0 else "verify_fail")
        else:
            categories.add("other")
    return categories or {"other"}


def _purpose(messages: list[AnyMessage]) -> str:
    first = messages[0]
    if isinstance(first, AIMessage) and isinstance(first.content, str) and first.content.strip():
        return first.content.strip()[:500]
    if isinstance(first, HumanMessage) and isinstance(first.content, str):
        return first.content.strip()[:500]
    names = [
        str(call["name"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    return ", ".join(names)[:500] or "conversation"


def _task_scope(messages: Sequence[AnyMessage]) -> str:
    for message in messages:
        task_id = str(message.additional_kwargs.get("_deepfix_task_id", "")).strip()
        if task_id:
            return task_id
    return "graph"
