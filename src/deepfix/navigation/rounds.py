"""Deterministic detection of completed action tool rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


@dataclass(frozen=True)
class ToolRoundDelta:
    count: int
    latest_round_id: str | None


def completed_tool_rounds_after(
    messages: Sequence[AnyMessage],
    cursor_round_id: str | None,
) -> ToolRoundDelta:
    """Count complete action rounds represented by ``messages`` after a cursor."""

    complete_round_ids: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        round_id = message.id
        if not isinstance(round_id, str) or not round_id.strip():
            continue

        call_ids: list[str] = []
        tool_names: list[object] = []
        for call in message.tool_calls:
            call_id = call.get("id") if isinstance(call, dict) else None
            if not isinstance(call_id, str) or not call_id.strip():
                break
            call_ids.append(call_id)
            tool_names.append(call.get("name"))
        else:
            paired_ids = {
                candidate.tool_call_id
                for candidate in messages[index + 1 :]
                if isinstance(candidate, ToolMessage) and candidate.tool_call_id
            }
            if all(call_id in paired_ids for call_id in call_ids) and not all(
                name == "write_todos" for name in tool_names
            ):
                complete_round_ids.append(round_id)

    latest_round_id = complete_round_ids[-1] if complete_round_ids else cursor_round_id
    if cursor_round_id is None:
        return ToolRoundDelta(len(complete_round_ids), latest_round_id)
    if cursor_round_id not in complete_round_ids:
        return ToolRoundDelta(0, latest_round_id)

    cursor_index = complete_round_ids.index(cursor_round_id)
    return ToolRoundDelta(len(complete_round_ids) - cursor_index - 1, latest_round_id)
