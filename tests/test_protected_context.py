from __future__ import annotations

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.compaction.errors import ProtectedContextLoadError
from deepfix.domain_repositories import DomainRepositories
from deepfix.protected_context import ProtectedContextBuilder, ProtectedContextMiddleware
from deepfix.task_domain.models import TaskDefinition


def _repositories(tmp_path, task_id="task-a", problem="修复解析错误"):
    repositories = DomainRepositories.create(tmp_path / "deepfix.sqlite3")
    repositories.tasks.create_definition(
        TaskDefinition(
            task_id=task_id,
            original_message_id=f"user-{task_id}",
            original_problem=problem,
            approval_mode="manual",
            source_project_root="C:/project",
            workspace_root=f"C:/workspaces/{task_id}",
            workspace_baseline_id="baseline-1",
            project_python="C:/Python/python.exe",
            confinement_level="guarded_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    return repositories


def test_protected_context_is_task_local_and_uses_immutable_definition(tmp_path):
    repositories = _repositories(tmp_path, "task-a", "修复 A 的解析错误")
    repositories.tasks.create_definition(
        TaskDefinition(
            task_id="task-b",
            original_message_id="user-task-b",
            original_problem="B TASK SECRET",
            approval_mode="manual",
            source_project_root="C:/project-b",
            workspace_root="C:/workspaces/task-b",
            project_python="C:/Python/python.exe",
            confinement_level="guarded_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    context = ProtectedContextBuilder(repositories).build(
        "task-a", [HumanMessage(id="later", content="继续调查")], None
    )
    assert context.task_anchor.task_goal == "修复 A 的解析错误"
    assert context.task_anchor.latest_user_message_id == "later"
    assert "B TASK SECRET" not in str(context)


def test_authoritative_read_failure_is_typed(tmp_path):
    repositories = _repositories(tmp_path)
    builder = ProtectedContextBuilder(repositories)

    class ThrowingTasks:
        def get_definition(self, task_id):
            raise RuntimeError("database unavailable")

    object.__setattr__(repositories, "tasks", ThrowingTasks())
    with pytest.raises(ProtectedContextLoadError) as captured:
        builder.build("task-a", [], None)
    assert captured.value.recovery.stage == "protected_context"
    assert captured.value.recovery.original_messages_preserved is True


def test_compatibility_middleware_keeps_context_request_local(tmp_path):
    middleware = ProtectedContextMiddleware(ProtectedContextBuilder(_repositories(tmp_path)))
    graph_messages = [HumanMessage(id="graph-user", content="继续")]
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=list(graph_messages),
        system_message=SystemMessage(content="base"),
        tools=[],
        state={"messages": list(graph_messages)},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-1",
                thread_id="task-a",
            )
        ),
    )
    captured = []

    def handler(updated):
        captured.append(updated)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)
    assert captured[0].system_message.text.count("<deepfix_protected_context>") == 1
    assert captured[0].messages == graph_messages
    assert request.state["messages"] == graph_messages
