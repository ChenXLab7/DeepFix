from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.context import (
    ContextMemoryMiddleware,
    build_save_progress_tool,
    render_working_memory,
)
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore, WorkingMemoryVersion


def snapshot(summary: str = "已复现失败") -> ProgressSnapshot:
    return ProgressSnapshot(
        phase="investigating",
        summary=summary,
        facts=["失败可复现"],
        evidence=[],
        active_hypotheses=[],
        rejected_hypotheses=[],
        checked_files=[],
        experiments=[],
        next_steps=["检查调用链"],
        unresolved_questions=[],
    )


def progress_call(call_id: str, **overrides):
    args = {
        "phase": "investigating",
        "summary": "已复现失败",
        "facts": ["失败可复现"],
        "evidence": [],
        "active_hypotheses": [],
        "rejected_hypotheses": [],
        "checked_files": [],
        "experiments": [],
        "next_steps": ["检查调用链"],
        "unresolved_questions": [],
    }
    args.update(overrides)
    return {
        "name": "save_progress",
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def invoke_tool(tool, call, thread_id: str):
    node = ToolNode([tool])
    return node.invoke(
        {"messages": [AIMessage(content="", tool_calls=[call])]},
        {"configurable": {"thread_id": thread_id}},
        runtime=Runtime(),
    )["messages"][0]


def model_request(thread_id: str) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="继续调查")],
        system_message=SystemMessage(content="基础系统提示"),
        tools=[],
        state={"messages": []},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-node-1",
                thread_id=thread_id,
            )
        ),
    )


def test_save_progress_uses_thread_id_and_does_not_expose_task_id(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    tool = build_save_progress_tool(store)

    result = invoke_tool(tool, progress_call("save-1"), "task-a")

    assert "task_id" not in tool.args
    assert store.latest("task-a").version == 1
    assert store.latest("task-a").snapshot.summary == "已复现失败"
    assert result.status == "success"
    assert result.artifact == {"version": 1}


def test_save_progress_rejects_invalid_snapshot_without_writing_version(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    tool = build_save_progress_tool(store)
    call = progress_call(
        "save-invalid",
        active_hypotheses=[f"假设 {index}" for index in range(11)],
    )

    result = invoke_tool(tool, call, "task-invalid")

    assert result.status == "error"
    assert "active_hypotheses" in result.text
    assert store.latest("task-invalid") is None


def test_context_middleware_injects_only_latest_task_memory_and_records_peak(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    store.save("task-a", snapshot("旧调查摘要"))
    store.save("task-a", snapshot("最新调查摘要"))
    middleware = ContextMemoryMiddleware(store)
    captured = []

    def handler(received):
        captured.append(received)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(model_request("task-a"), handler)

    injected = captured[0].system_message.text
    assert "<deepfix_working_memory" in injected
    assert "最新调查摘要" in injected
    assert "检查调用链" in injected
    assert "旧调查摘要" not in injected
    assert store.metrics("task-a").context_peak_tokens > 0

    captured.clear()
    middleware.wrap_model_call(model_request("task-b"), handler)
    assert captured[0].system_message.text == "基础系统提示"
    assert store.metrics("task-b").context_peak_tokens == 0


def test_render_working_memory_has_a_hard_injection_bound():
    bounded_snapshot = snapshot("摘要" * 1000).model_copy(
        update={"facts": [f"事实-{index}-" + "x" * 1000 for index in range(30)]}
    )
    rendered = render_working_memory(
        WorkingMemoryVersion(
            task_id="task-a",
            version=7,
            snapshot=bounded_snapshot,
            created_at="2026-08-21T00:00:00+00:00",
        )
    )

    assert "…[truncated]" in rendered
    assert "事实-0-" in rendered
    assert "事实-8-" not in rendered
    assert len(rendered) <= 12_000
