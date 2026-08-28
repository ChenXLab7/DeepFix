from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deepfix.compaction.middleware import MessageIdentityMiddleware
from deepfix.navigation.feedback import NavigationFeedback
from deepfix.navigation.middleware import TodoNavigationMiddleware
from deepfix.navigation.prompts import DEEPFIX_TODO_SYSTEM_PROMPT

TODOS = [
    {"content": "reproduce failure", "status": "completed"},
    {"content": "identify root cause", "status": "in_progress"},
    {"content": "apply minimal fix", "status": "pending"},
    {"content": "run required verification", "status": "pending"},
]


def _call(name: str, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


class _NoFeedback:
    def build(self, task_id: str) -> NavigationFeedback:
        assert task_id == "workflow-task"
        return NavigationFeedback(milestone_ids=(), lines=(), fingerprint="empty")


class _ScriptedChatModel(FakeMessagesListChatModel):
    captured_system_prompts: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system_prompts = [
            message.text for message in messages if isinstance(message, SystemMessage)
        ]
        self.captured_system_prompts.append("\n\n".join(system_prompts))
        return super()._generate(messages, stop, run_manager, **kwargs)


def _read_file(file_path: str) -> str:
    """Read a deterministic offline fixture file."""

    return f"contents of {file_path}"


def _grep(pattern: str) -> str:
    """Search deterministic offline fixture content."""

    return f"match for {pattern}"


def _agent(model: _ScriptedChatModel, checkpointer: InMemorySaver):
    return create_agent(
        model=model,
        tools=[
            StructuredTool.from_function(_read_file, name="read_file"),
            StructuredTool.from_function(_grep, name="grep"),
        ],
        system_prompt="Repair the offline fixture.",
        middleware=[
            MessageIdentityMiddleware(),
            TodoListMiddleware(system_prompt=DEEPFIX_TODO_SYSTEM_PROMPT),
            TodoNavigationMiddleware(_NoFeedback(), reminder_rounds=3),
        ],
        checkpointer=checkpointer,
    )


def test_todo_navigation_survives_checkpoint_resume_and_parallel_tool_rounds():
    model = _ScriptedChatModel(
        responses=[
            AIMessage(
                id="plan-round",
                content="",
                tool_calls=[_call("write_todos", "todos-1", {"todos": TODOS})],
            ),
            AIMessage(
                id="parallel-read-round",
                content="",
                tool_calls=[
                    _call("read_file", "read-source", {"file_path": "src/value.py"}),
                    _call("read_file", "read-test", {"file_path": "tests/test_value.py"}),
                ],
            ),
            AIMessage(
                id="second-investigation-round",
                content="",
                tool_calls=[_call("grep", "grep-branch", {"pattern": "normalize"})],
            ),
            AIMessage(id="checkpoint-stop", content="Pause at the checkpoint."),
            AIMessage(
                id="third-investigation-round",
                content="",
                tool_calls=[
                    _call("read_file", "read-config", {"file_path": "pyproject.toml"})
                ],
            ),
            AIMessage(id="resume-stop", content="Ready to update the Todo."),
        ]
    )
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "workflow-task"}}

    first = _agent(model, checkpointer).invoke(
        {"messages": [HumanMessage(id="user-1", content="Fix the failure.")]},
        config,
    )
    checkpoint_before_resume = _agent(model, checkpointer).get_state(config).values
    assert first["todos"] == TODOS
    assert checkpoint_before_resume["todos"] == TODOS
    assert checkpoint_before_resume["_deepfix_todo_rounds_since_update"] == 2
    assert checkpoint_before_resume["_deepfix_last_completed_tool_round_id"] == (
        "second-investigation-round"
    )
    assert checkpoint_before_resume[
        "_deepfix_last_todo_progress_fingerprint"
    ] == "5f78ef6c5f272dfec1a91db55cc9a57d00166a64d93dc5e76a9af9a94efa2af1"
    assert checkpoint_before_resume["_deepfix_last_navigation_hint_fingerprint"] is None
    assert checkpoint_before_resume["_deepfix_pending_navigation_reminder"] is None

    captured_before_resume = len(model.captured_system_prompts)
    resumed_agent = _agent(model, checkpointer)
    result = resumed_agent.invoke(
        {"messages": [HumanMessage(id="user-2", content="Continue.")]},
        config,
    )

    captured_after_resume = model.captured_system_prompts[captured_before_resume:]
    captured_system_prompt = model.captured_system_prompts[-1]
    checkpoint_state = resumed_agent.get_state(config).values

    assert result["todos"] == TODOS
    assert checkpoint_state["todos"] == TODOS
    assert checkpoint_state["_deepfix_last_completed_tool_round_id"] == (
        "third-investigation-round"
    )
    assert checkpoint_state["_deepfix_todo_rounds_since_update"] == 0
    assert checkpoint_state[
        "_deepfix_last_todo_progress_fingerprint"
    ] == checkpoint_before_resume["_deepfix_last_todo_progress_fingerprint"]
    assert checkpoint_state["_deepfix_last_navigation_hint_fingerprint"] is None
    assert "<todo_navigation_reminder>" in checkpoint_state[
        "_deepfix_pending_navigation_reminder"
    ]
    assert len(captured_after_resume) == 2
    assert "<todo_navigation_reminder>" not in captured_after_resume[0]
    assert "<todo_navigation_reminder>" in captured_system_prompt
    assert captured_system_prompt.count("<todo_navigation_reminder>") == 1
    assert len(model.captured_system_prompts) == 6
    assert all(
        "<todo_navigation_reminder>" not in prompt
        for prompt in model.captured_system_prompts[:-1]
    )
    assert sum(
        prompt.count("<todo_navigation_reminder>")
        for prompt in model.captured_system_prompts
    ) == 1
