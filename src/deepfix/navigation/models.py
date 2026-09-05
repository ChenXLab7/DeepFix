"""State and deterministic helpers for native LangChain Todo navigation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Annotated, NotRequired

from langchain.agents.middleware.todo import PlanningState, Todo
from langchain.agents.middleware.types import PrivateStateAttr
from pydantic import BaseModel


class TodoNavigationState(PlanningState):
    """Native planning state with private navigation bookkeeping."""

    _deepfix_todo_rounds_since_update: NotRequired[Annotated[int, PrivateStateAttr]]
    _deepfix_last_completed_tool_round_id: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_last_todo_progress_fingerprint: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_last_navigation_hint_fingerprint: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_pending_navigation_reminder: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]


class TodoValidation(BaseModel):
    valid: bool
    error_code: str | None = None
    message: str | None = None


_VALID_STATUSES = frozenset({"pending", "in_progress", "completed"})


def validate_todos(todos: Sequence[Todo]) -> TodoValidation:
    """Validate native Todo progress invariants without changing Todo items."""

    statuses: list[object] = []
    for todo in todos:
        status = todo.get("status") if isinstance(todo, dict) else None
        if status not in _VALID_STATUSES:
            return TodoValidation(
                valid=False,
                error_code="todo_invalid_status",
                message=f"Invalid Todo status: {status!r}",
            )
        statuses.append(status)

    in_progress_count = statuses.count("in_progress")
    if in_progress_count > 1:
        return TodoValidation(
            valid=False,
            error_code="todo_multiple_in_progress",
            message="Todo list must contain at most one in-progress item.",
        )
    if any(status != "completed" for status in statuses) and in_progress_count != 1:
        return TodoValidation(
            valid=False,
            error_code="todo_requires_one_in_progress",
            message="A non-completed Todo list must contain exactly one in-progress item.",
        )
    return TodoValidation(valid=True)


def todo_progress_fingerprint(todos: Sequence[Todo]) -> str:
    """Return a deterministic fingerprint of Todo cardinality and ordered statuses."""

    progress = [todo.get("status") if isinstance(todo, dict) else None for todo in todos]
    serialized = json.dumps(progress, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
