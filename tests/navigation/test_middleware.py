from __future__ import annotations

import asyncio
from html import unescape
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.navigation.feedback import NavigationFeedback
from deepfix.navigation.middleware import TodoNavigationMiddleware
from deepfix.navigation.models import TodoNavigationState


class FeedbackSource:
    def __init__(self, feedback: NavigationFeedback | None = None) -> None:
        self.feedback = feedback or NavigationFeedback(
            milestone_ids=(),
            lines=(),
            fingerprint="empty",
        )
        self.calls: list[str] = []
        self.mutation_calls: list[str] = []

    def build(self, task_id: str) -> NavigationFeedback:
        self.calls.append(task_id)
        return self.feedback

    def record_progress(self, task_id: str) -> None:
        self.mutation_calls.append(task_id)


def runtime(task_id: str | None = "task-a") -> Runtime:
    return Runtime(
        execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="model-1",
            thread_id=task_id,
        )
    )


def action_round(round_id: str, *call_ids: str) -> list[Any]:
    calls = [
        {"id": call_id, "name": "read_file", "args": {"path": f"{call_id}.py"}}
        for call_id in call_ids
    ]
    return [
        AIMessage(id=round_id, content="", tool_calls=calls),
        *[
            ToolMessage(content="ok", tool_call_id=call_id, id=f"result-{call_id}")
            for call_id in call_ids
        ],
    ]


def state(
    *,
    todos: list[dict[str, str]] | None = None,
    messages: list[Any] | None = None,
    **private: Any,
) -> TodoNavigationState:
    return {
        "todos": todos or [],
        "messages": messages or [],
        **private,
    }


def tool_request(name: str, args: Any, call_id: str = "call-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id},
        tool=None,
        state={},
        runtime=runtime(),
    )


def model_request(
    pending: str | None,
    *,
    system_message: SystemMessage | None = None,
) -> ModelRequest:
    messages = [HumanMessage(id="user-1", content="fix it")]
    request_state = {
        "messages": messages,
        "todos": [{"content": "diagnose", "status": "in_progress"}],
        "_deepfix_pending_navigation_reminder": pending,
    }
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=messages,
        system_message=system_message or SystemMessage(content="base system"),
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        state=request_state,
        runtime=runtime(),
    )


@pytest.mark.parametrize("threshold", [0, -1])
def test_constructor_rejects_non_positive_threshold(threshold: int):
    with pytest.raises(ValueError, match="^reminder_rounds must be positive$"):
        TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=threshold)


def test_two_in_progress_items_return_error_without_calling_handler():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = tool_request(
        "write_todos",
        {
            "todos": [
                {"content": "diagnose", "status": "in_progress"},
                {"content": "fix", "status": "in_progress"},
            ]
        },
    )
    calls: list[ToolCallRequest] = []

    result = middleware.wrap_tool_call(request, lambda received: calls.append(received))

    assert calls == []
    assert result.status == "error"
    assert result.tool_call_id == "call-1"
    assert result.name == "write_todos"
    assert "todo_multiple_in_progress" in result.text


def test_non_completed_list_without_in_progress_returns_error():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = tool_request(
        "write_todos",
        {"todos": [{"content": "diagnose", "status": "pending"}]},
    )

    result = middleware.wrap_tool_call(
        request,
        lambda received: pytest.fail("invalid Todo must not reach handler"),
    )

    assert result.status == "error"
    assert "todo_requires_one_in_progress" in result.text


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"todos": "not-a-list"},
        {"todos": ["not-a-mapping"]},
        None,
    ],
)
def test_malformed_payload_returns_stable_error(args: Any):
    middleware = TodoNavigationMiddleware(FeedbackSource())

    result = middleware.wrap_tool_call(
        tool_request("write_todos", args),
        lambda received: pytest.fail("malformed Todo must not reach handler"),
    )

    assert result.status == "error"
    assert "todo_invalid_payload" in result.text


def test_valid_write_todos_delegates_exact_request():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = tool_request(
        "write_todos",
        {
            "todos": [
                {"content": "diagnose", "status": "in_progress"},
                {"content": "fix", "status": "pending"},
            ]
        },
    )
    received: list[ToolCallRequest] = []
    expected = ToolMessage(content="saved", name="write_todos", tool_call_id="call-1")

    result = middleware.wrap_tool_call(
        request,
        lambda candidate: received.append(candidate) or expected,
    )

    assert received == [request]
    assert received[0] is request
    assert result is expected


def test_non_todo_tool_delegates_exact_request():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = tool_request("read_file", {"path": "a.py"})
    received: list[ToolCallRequest] = []
    expected = ToolMessage(content="file", name="read_file", tool_call_id="call-1")

    result = middleware.wrap_tool_call(
        request,
        lambda candidate: received.append(candidate) or expected,
    )

    assert received == [request]
    assert received[0] is request
    assert result is expected


def test_async_validation_mirrors_sync_behavior():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    invalid = tool_request("write_todos", {"todos": [{"content": "later", "status": "pending"}]})
    valid = tool_request(
        "write_todos",
        {"todos": [{"content": "now", "status": "in_progress"}]},
        "call-2",
    )
    calls: list[ToolCallRequest] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request)
        return ToolMessage(content="saved", name="write_todos", tool_call_id="call-2")

    async def run() -> tuple[ToolMessage, ToolMessage]:
        error = await middleware.awrap_tool_call(invalid, handler)
        delegated = await middleware.awrap_tool_call(valid, handler)
        return error, delegated

    error, delegated = asyncio.run(run())

    assert error.status == "error"
    assert "todo_requires_one_in_progress" in error.text
    assert calls == [valid]
    assert delegated.status == "success"


def test_parallel_tool_round_increments_counter_once():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=3)
    messages = action_round("round-1", "call-a", "call-b")

    update = middleware.before_model(
        state(
            todos=[{"content": "diagnose", "status": "in_progress"}],
            messages=messages,
        ),
        runtime(),
    )

    assert update is not None
    assert update["_deepfix_todo_rounds_since_update"] == 1
    assert update["_deepfix_last_completed_tool_round_id"] == "round-1"


def test_same_messages_are_not_recounted():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=3)
    original = state(
        todos=[{"content": "diagnose", "status": "in_progress"}],
        messages=action_round("round-1", "call-a"),
    )
    first = middleware.before_model(original, runtime())
    assert first is not None

    second = middleware.before_model({**original, **first}, runtime())

    assert second is not None
    assert second["_deepfix_todo_rounds_since_update"] == 1
    assert second["_deepfix_last_completed_tool_round_id"] == "round-1"


def test_prose_only_todo_rewrite_does_not_reset_counter():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=5)
    initial = middleware.before_model(
        state(todos=[{"content": "diagnose", "status": "in_progress"}]),
        runtime(),
    )
    assert initial is not None
    prior = {**initial, "_deepfix_todo_rounds_since_update": 2}

    update = middleware.before_model(
        state(
            todos=[{"content": "carefully diagnose parser", "status": "in_progress"}],
            **prior,
        ),
        runtime(),
    )

    assert update is not None
    assert update["_deepfix_todo_rounds_since_update"] == 2


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            [{"content": "diagnose", "status": "in_progress"}],
            [{"content": "diagnose", "status": "completed"}],
        ),
        (
            [
                {"content": "diagnose", "status": "in_progress"},
                {"content": "fix", "status": "pending"},
            ],
            [
                {"content": "diagnose", "status": "completed"},
                {"content": "fix", "status": "in_progress"},
            ],
        ),
        (
            [{"content": "diagnose", "status": "in_progress"}],
            [
                {"content": "diagnose", "status": "in_progress"},
                {"content": "fix", "status": "pending"},
            ],
        ),
    ],
)
def test_meaningful_todo_change_resets_counter(before: list[dict[str, str]], after: list[dict[str, str]]):
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=5)
    initial = middleware.before_model(state(todos=before), runtime())
    assert initial is not None

    update = middleware.before_model(
        state(todos=after, **{**initial, "_deepfix_todo_rounds_since_update": 2}),
        runtime(),
    )

    assert update is not None
    assert update["_deepfix_todo_rounds_since_update"] == 0


def test_three_completed_rounds_schedule_cadence_reminder_and_reset_counter():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=3)
    messages = [
        *action_round("round-1", "call-1"),
        *action_round("round-2", "call-2"),
        *action_round("round-3", "call-3"),
    ]

    update = middleware.before_model(
        state(
            todos=[{"content": "diagnose", "status": "in_progress"}],
            messages=messages,
        ),
        runtime(),
    )

    assert update is not None
    assert update["_deepfix_todo_rounds_since_update"] == 0
    assert "是否已有足够证据停止继续调查" in (
        update["_deepfix_pending_navigation_reminder"]
    )


def test_new_milestone_schedules_before_threshold_and_identical_milestone_does_not_repeat():
    source = FeedbackSource(
        NavigationFeedback(
            milestone_ids=("closed-evidence-gap:gap-1",),
            lines=("Closed evidence gap: gap-1.",),
            fingerprint="milestone-a",
        )
    )
    middleware = TodoNavigationMiddleware(source, reminder_rounds=3)
    current = state(todos=[{"content": "diagnose", "status": "in_progress"}])

    first = middleware.before_model(current, runtime())
    assert first is not None
    repeated = middleware.before_model({**current, **first}, runtime())

    assert first["_deepfix_pending_navigation_reminder"] is not None
    assert first["_deepfix_last_navigation_hint_fingerprint"] == "milestone-a"
    assert repeated is not None
    assert repeated["_deepfix_pending_navigation_reminder"] is None
    assert repeated["_deepfix_last_navigation_hint_fingerprint"] == "milestone-a"


def test_changed_milestone_fingerprint_schedules_a_new_reminder():
    source = FeedbackSource(
        NavigationFeedback(
            milestone_ids=("supported-hypothesis:h1",),
            lines=("Supported hypothesis: h1.",),
            fingerprint="milestone-a",
        )
    )
    middleware = TodoNavigationMiddleware(source)
    current = state(todos=[{"content": "diagnose", "status": "in_progress"}])
    first = middleware.before_model(current, runtime())
    assert first is not None
    source.feedback = NavigationFeedback(
        milestone_ids=("supported-hypothesis:h1", "closed-evidence-gap:gap-1"),
        lines=("Supported hypothesis: h1.", "Closed evidence gap: gap-1."),
        fingerprint="milestone-b",
    )

    changed = middleware.before_model({**current, **first}, runtime())

    assert changed is not None
    assert changed["_deepfix_pending_navigation_reminder"] is not None
    assert changed["_deepfix_last_navigation_hint_fingerprint"] == "milestone-b"


def test_missing_todo_after_one_action_round_schedules_create_plan_advice():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=10)

    update = middleware.before_model(
        state(messages=action_round("round-1", "call-1")),
        runtime(),
    )

    assert update is not None
    reminder = update["_deepfix_pending_navigation_reminder"]
    assert reminder is not None
    assert "create a short native Todo" in reminder
    assert update["_deepfix_todo_rounds_since_update"] == 0


def test_no_trigger_explicitly_clears_prior_pending_reminder():
    middleware = TodoNavigationMiddleware(FeedbackSource(), reminder_rounds=3)

    update = middleware.before_model(
        state(
            todos=[{"content": "diagnose", "status": "in_progress"}],
            _deepfix_pending_navigation_reminder="stale reminder",
        ),
        runtime(),
    )

    assert update is not None
    assert update["_deepfix_pending_navigation_reminder"] is None


def test_cadence_does_not_mutate_investigation_progress():
    source = FeedbackSource()
    middleware = TodoNavigationMiddleware(source, reminder_rounds=1)
    current = state(
        todos=[{"content": "diagnose", "status": "in_progress"}],
        messages=action_round("round-1", "call-1"),
        no_progress_count=7,
    )

    update = middleware.before_model(current, runtime())

    assert update is not None
    assert source.mutation_calls == []
    assert current["no_progress_count"] == 7
    assert "no_progress_count" not in update
    assert set(update) == {
        "_deepfix_todo_rounds_since_update",
        "_deepfix_last_completed_tool_round_id",
        "_deepfix_last_todo_progress_fingerprint",
        "_deepfix_last_navigation_hint_fingerprint",
        "_deepfix_pending_navigation_reminder",
    }
    assert "messages" not in update
    assert "todos" not in update


def test_reminder_request_preserves_request_fields_and_only_changes_system_message():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = model_request("<todo_navigation_reminder>review</todo_navigation_reminder>")
    captured: list[ModelRequest] = []

    middleware.wrap_model_call(
        request,
        lambda candidate: captured.append(candidate)
        or ModelResponse(result=[AIMessage(content="ok")]),
    )

    updated = captured[0]
    assert updated is not request
    assert updated.model is request.model
    assert updated.messages is request.messages
    assert updated.tools is request.tools
    assert updated.state is request.state
    assert updated.runtime is request.runtime
    assert updated.system_message is not request.system_message
    assert updated.system_message.text == (
        "base system\n\n<todo_navigation_reminder>review</todo_navigation_reminder>"
    )
    assert request.state["messages"] == request.messages
    assert all(not isinstance(message, SystemMessage) for message in request.messages)


def test_no_pending_reminder_passes_exact_request_unchanged():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    request = model_request(None)
    captured: list[ModelRequest] = []

    middleware.wrap_model_call(
        request,
        lambda candidate: captured.append(candidate)
        or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert captured == [request]
    assert captured[0] is request


def test_xml_sensitive_todo_and_feedback_are_escaped_after_truncation():
    source = FeedbackSource(
        NavigationFeedback(
            milestone_ids=("milestone-1",),
            lines=('Trusted <signal> & "quote" and \'apostrophe\'.',),
            fingerprint="milestone-1",
        )
    )
    middleware = TodoNavigationMiddleware(source)

    update = middleware.before_model(
        state(todos=[{"content": 'inspect <x> & "y" and \'z\'', "status": "in_progress"}]),
        runtime(),
    )

    assert update is not None
    reminder = update["_deepfix_pending_navigation_reminder"]
    assert reminder is not None
    assert "&lt;x&gt;" in reminder
    assert "&amp;" in reminder
    assert "&quot;y&quot;" in reminder
    assert "&#x27;z&#x27;" in reminder
    assert "&lt;signal&gt;" in reminder
    assert "<x>" not in reminder
    assert "<signal>" not in reminder


def test_todo_feedback_and_character_bounds_are_enforced():
    todos = [
        {
            "content": f"todo-{index}-" + ("x" * 300) + f"-end-{index}",
            "status": "in_progress" if index == 0 else "pending",
        }
        for index in range(13)
    ]
    source = FeedbackSource(
        NavigationFeedback(
            milestone_ids=("milestone-1",),
            lines=tuple(f"feedback-{index}-" + ("y" * 300) + f"-end-{index}" for index in range(9)),
            fingerprint="milestone-1",
        )
    )
    middleware = TodoNavigationMiddleware(source)

    update = middleware.before_model(state(todos=todos), runtime())

    assert update is not None
    reminder = update["_deepfix_pending_navigation_reminder"]
    assert reminder is not None
    todo_lines = [line for line in reminder.splitlines() if line.startswith("- [")]
    feedback_lines = [line for line in reminder.splitlines() if line.startswith("- feedback-")]
    assert len(todo_lines) == 12
    assert len(feedback_lines) == 8
    assert all(len(unescape(line)) <= 240 for line in todo_lines)
    assert all(len(unescape(line)) <= 240 for line in feedback_lines)
    assert "end-12" not in reminder
    assert "feedback-8" not in reminder


def test_sync_and_async_model_wrappers_produce_equivalent_request_content():
    middleware = TodoNavigationMiddleware(FeedbackSource())
    reminder = "<todo_navigation_reminder>review</todo_navigation_reminder>"
    sync_request = model_request(reminder)
    async_request = model_request(reminder)
    captured_sync: list[ModelRequest] = []
    captured_async: list[ModelRequest] = []

    middleware.wrap_model_call(
        sync_request,
        lambda candidate: captured_sync.append(candidate)
        or ModelResponse(result=[AIMessage(content="sync")]),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        captured_async.append(candidate)
        return ModelResponse(result=[AIMessage(content="async")])

    asyncio.run(middleware.awrap_model_call(async_request, handler))

    assert captured_sync[0].system_message.text == captured_async[0].system_message.text
    assert captured_sync[0].messages == captured_async[0].messages
    assert captured_sync[0].tools == captured_async[0].tools
    assert captured_sync[0].state == captured_async[0].state


def test_blank_task_id_returns_no_update_without_querying_feedback():
    source = FeedbackSource()
    middleware = TodoNavigationMiddleware(source)

    assert middleware.before_model(state(), runtime("   ")) is None
    assert asyncio.run(middleware.abefore_model(state(), runtime(None))) is None
    assert source.calls == []
