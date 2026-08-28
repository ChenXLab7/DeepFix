"""Request-local Todo navigation reminders and native Todo validation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from html import escape
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepfix.navigation.feedback import NavigationFeedback, NavigationFeedbackSource
from deepfix.navigation.models import (
    TodoNavigationState,
    todo_progress_fingerprint,
    validate_todos,
)
from deepfix.navigation.rounds import completed_tool_rounds_after

_MAX_TODOS = 12
_MAX_FEEDBACK_LINES = 8
_MAX_RENDERED_LINE_CHARS = 240


class TodoNavigationMiddleware(AgentMiddleware):
    """Schedule bounded reminders without writing native Todo or domain state."""

    state_schema = TodoNavigationState

    def __init__(
        self,
        feedback_source: NavigationFeedbackSource,
        *,
        reminder_rounds: int = 3,
    ) -> None:
        if reminder_rounds <= 0:
            raise ValueError("reminder_rounds must be positive")
        self.feedback_source = feedback_source
        self.reminder_rounds = reminder_rounds

    def before_model(
        self,
        state: TodoNavigationState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Advance private counters and schedule at most one request-local reminder."""

        return self._navigation_update(state, runtime)

    async def abefore_model(
        self,
        state: TodoNavigationState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Asynchronous state hook with the same deterministic projection."""

        return self._navigation_update(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Append a pending reminder to this model request's system message only."""

        return handler(self._model_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Asynchronous model wrapper equivalent to the synchronous wrapper."""

        return await handler(self._model_request(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Validate native ``write_todos`` payloads before framework execution."""

        error = _todo_validation_error(request)
        if error is not None:
            return error
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        """Asynchronously validate native Todo writes before delegation."""

        error = _todo_validation_error(request)
        if error is not None:
            return error
        return await handler(request)

    def _navigation_update(
        self,
        state: TodoNavigationState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        task_id = _runtime_task_id(runtime)
        if not task_id:
            return None

        todos = list(state.get("todos") or ())
        messages = list(state.get("messages") or ())
        current_todo_fingerprint = todo_progress_fingerprint(todos)
        stored_todo_fingerprint = state.get(
            "_deepfix_last_todo_progress_fingerprint"
        )
        todo_changed = (
            stored_todo_fingerprint is not None
            and stored_todo_fingerprint != current_todo_fingerprint
        )

        stored_cursor = state.get("_deepfix_last_completed_tool_round_id")
        delta = completed_tool_rounds_after(messages, stored_cursor)
        stored_rounds = _non_negative_counter(
            state.get("_deepfix_todo_rounds_since_update")
        )
        rounds = 0 if todo_changed else stored_rounds + delta.count

        feedback = self.feedback_source.build(task_id)
        stored_hint_fingerprint = state.get(
            "_deepfix_last_navigation_hint_fingerprint"
        )
        new_milestone = bool(feedback.milestone_ids) and (
            feedback.fingerprint != stored_hint_fingerprint
        )
        cadence_due = rounds >= self.reminder_rounds
        missing_plan_due = not todos and delta.count > 0
        reminder_due = new_milestone or cadence_due or missing_plan_due

        if reminder_due:
            rounds = 0
            pending_reminder = render_todo_navigation_reminder(todos, feedback)
        else:
            pending_reminder = None

        hint_fingerprint = (
            feedback.fingerprint if new_milestone else stored_hint_fingerprint
        )
        return {
            "_deepfix_todo_rounds_since_update": rounds,
            "_deepfix_last_completed_tool_round_id": delta.latest_round_id,
            "_deepfix_last_todo_progress_fingerprint": current_todo_fingerprint,
            "_deepfix_last_navigation_hint_fingerprint": hint_fingerprint,
            "_deepfix_pending_navigation_reminder": pending_reminder,
        }

    @staticmethod
    def _model_request(request: ModelRequest) -> ModelRequest:
        state = request.state
        pending = (
            state.get("_deepfix_pending_navigation_reminder")
            if isinstance(state, Mapping)
            else None
        )
        if not isinstance(pending, str) or not pending:
            return request

        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{pending}" if original else pending
        return request.override(system_message=SystemMessage(content=content))


def render_todo_navigation_reminder(
    todos: Sequence[Mapping[str, Any]],
    feedback: NavigationFeedback,
) -> str:
    """Render deterministic, escaped, bounded Todo navigation guidance."""

    if todos:
        todo_lines = [
            _escaped_bounded_line(
                f"- [{todo.get('status', '')}] {todo.get('content', '')}"
            )
            for todo in todos[:_MAX_TODOS]
        ]
    else:
        todo_lines = [
            (
                "- No native Todo exists; create a short native Todo before more "
                "multi-step work."
            )
        ]

    feedback_lines = [
        _escaped_bounded_line(f"- {line}")
        for line in feedback.lines[:_MAX_FEEDBACK_LINES]
    ]
    if not feedback_lines:
        feedback_lines = ["- No new trusted progress milestone."]

    return "\n".join(
        [
            "<todo_navigation_reminder>",
            "Current Todo:",
            *todo_lines,
            "",
            "Trusted progress signals:",
            *feedback_lines,
            "",
            "Review before continuing:",
            "1. Is the current in_progress item already complete?",
            (
                "2. Is there enough evidence to stop investigating? "
                "（是否已有足够证据停止继续调查）"
            ),
            (
                "3. Should you complete the current item and move to modification, "
                "verification, or wrap-up?"
            ),
            "4. Does the next action still serve the original user goal?",
            "</todo_navigation_reminder>",
        ]
    )


def _escaped_bounded_line(value: str) -> str:
    return escape(value[:_MAX_RENDERED_LINE_CHARS], quote=True)


def _runtime_task_id(runtime: Runtime) -> str:
    execution_info = getattr(runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    return thread_id.strip() if isinstance(thread_id, str) else ""


def _non_negative_counter(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _todo_validation_error(request: ToolCallRequest) -> ToolMessage | None:
    tool_call = request.tool_call
    if tool_call.get("name") != "write_todos":
        return None

    args = tool_call.get("args")
    todos = args.get("todos") if isinstance(args, Mapping) else None
    if not isinstance(todos, list) or any(
        not isinstance(todo, Mapping) for todo in todos
    ):
        return _todo_error_message(
            request,
            "todo_invalid_payload",
            "todos must be a list of native Todo mappings.",
        )

    validation = validate_todos([dict(todo) for todo in todos])
    if not validation.valid:
        return _todo_error_message(
            request,
            validation.error_code or "todo_invalid_payload",
            validation.message or "Todo validation failed.",
        )
    return None


def _todo_error_message(
    request: ToolCallRequest,
    code: str,
    message: str,
) -> ToolMessage:
    tool_call = request.tool_call
    return ToolMessage(
        content=f"{code}: {message}",
        name=str(tool_call.get("name", "write_todos")),
        tool_call_id=str(tool_call.get("id", "")),
        status="error",
    )
