import asyncio
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Command

from deepfix.investigation.errors import (
    InvestigationStagnationError,
    InvestigationStateError,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.models import (
    AgentPhase,
    ContinueInvestigationInput,
    InvestigationCapability,
    InvestigationHypothesis,
    InvestigationRecoveryMetadata,
    NewInvestigationEvent,
)
from deepfix.investigation.receipts import ToolExecutionReceiptStore
from investigation.helpers import coordinator_fixture, force_phase


def named_tools(*names: str) -> list[BaseTool]:
    def run() -> str:
        return "ok"

    return [
        StructuredTool.from_function(
            run,
            name=name,
            description=f"{name} test",
        )
        for name in names
    ]


def model_request(*, tools: list[BaseTool]) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="continue")],
        system_message=SystemMessage(content="base"),
        tools=tools,
        state={"messages": []},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-node",
                thread_id="task-a",
            )
        ),
    )


def capture_model_request(
    middleware: InvestigationMiddleware,
    request: ModelRequest,
) -> ModelRequest:
    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request,
        lambda updated: captured.append(updated)
        or ModelResponse(result=[AIMessage(content="ok")]),
    )
    return captured[0]


def tool_request(
    name: str,
    call_id: str,
    args: dict[str, object],
) -> ToolCallRequest:
    tool = named_tools(name)[0]
    runtime = ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": "task-a"}},
        stream_writer=lambda value: None,
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={
            "name": name,
            "id": call_id,
            "args": args,
            "type": "tool_call",
        },
        tool=tool,
        state={"messages": []},
        runtime=runtime,
    )


def pytest_tool_request() -> ToolCallRequest:
    return tool_request(
        "execute",
        "pytest-1",
        {"command": "python -m pytest -q"},
    )


def read_request() -> ToolCallRequest:
    return tool_request(
        "read_file",
        "read-1",
        {"file_path": "/src/sign.py"},
    )


def edit_request(call_id: str) -> ToolCallRequest:
    return tool_request(
        "edit_file",
        call_id,
        {
            "file_path": "/src/sign.py",
            "old_string": "return -value",
            "new_string": "return value",
        },
    )


def pytest_result(exit_code: int) -> ToolMessage:
    return ToolMessage(
        id="msg-pytest-1",
        content=f"pytest exit {exit_code}",
        tool_call_id="pytest-1",
        artifact={"exit_code": exit_code},
    )


def edit_result(call_id: str) -> ToolMessage:
    return ToolMessage(
        id=f"msg-{call_id}",
        content="edited",
        tool_call_id=call_id,
        artifact={
            "operation": "edit",
            "status": "succeeded",
            "path": "/src/sign.py",
        },
    )


def read_result() -> ToolMessage:
    return ToolMessage(
        id="msg-read-1",
        content="return -value",
        tool_call_id="read-1",
    )


def all_test_tools() -> list[BaseTool]:
    return named_tools(
        "read_file",
        "grep",
        "execute",
        "edit_file",
        "record_hypothesis",
        "continue_investigation",
        "search_diagnostic_artifacts",
        "read_diagnostic_artifact",
        "save_progress",
        "compact_conversation",
    )


def middleware_fixture(
    tmp_path: Path,
    *,
    phase: str = "investigating",
    stagnation_level: int = 0,
    fail_next_event_commit: bool = False,
) -> InvestigationMiddleware:
    coordinator = coordinator_fixture(tmp_path)
    state = coordinator.store.ensure_started("task-a")
    state = force_phase(coordinator.store, state, AgentPhase(phase))
    candidate = InvestigationHypothesis(
        hypothesis_id="hyp-1",
        statement="sign flips in helper",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate",
    )
    state = coordinator.store.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id=stable_investigation_id(
                    "event", "task-a", "middleware-fixture"
                ),
                task_id="task-a",
                event_type="hypothesis_recorded",
                phase_before=state.agent_phase,
                phase_after=state.agent_phase,
            )
        ],
        state.model_copy(
            update={
                "hypotheses": [candidate],
                "stagnation_level": stagnation_level,
                "reevaluation_required": stagnation_level > 0,
            }
        ),
    )
    if fail_next_event_commit:
        original_record = coordinator.record_tool_result
        failed = False

        def fail_once(task_id, tool_call, result):
            nonlocal failed
            if not failed:
                failed = True
                current = coordinator.state(task_id)
                raise InvestigationStateError(
                    InvestigationRecoveryMetadata(
                        task_id=task_id,
                        error_code="investigation_event_commit_failed",
                        agent_phase=current.agent_phase,
                        state_version=current.version,
                        last_event_sequence=coordinator.store.last_sequence(task_id),
                        tool_call_id=str(tool_call["id"]),
                        checkpoint_available=True,
                        recovery_action="replay_event_commit_without_rerunning_tool",
                    )
                )
            return original_record(task_id, tool_call, result)

        coordinator.record_tool_result = fail_once
    capabilities = {
        "read_file": InvestigationCapability.READ,
        "grep": InvestigationCapability.SEARCH,
        "execute": InvestigationCapability.EXECUTE,
        "edit_file": InvestigationCapability.MODIFY,
        "record_hypothesis": InvestigationCapability.META,
        "continue_investigation": InvestigationCapability.META,
        "search_diagnostic_artifacts": InvestigationCapability.READ,
        "read_diagnostic_artifact": InvestigationCapability.READ,
        "save_progress": InvestigationCapability.MEMORY,
        "compact_conversation": InvestigationCapability.COMPACTION,
    }
    backend = FilesystemBackend(
        root_dir=tmp_path / "artifacts",
        virtual_mode=True,
    )
    return InvestigationMiddleware(
        coordinator,
        ToolExecutionReceiptStore(backend),
        capabilities,
    )


def test_diagnosing_request_hides_modify_tools_and_renders_bounded_state(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    request = model_request(
        tools=named_tools("read_file", "edit_file", "record_hypothesis")
    )

    captured = capture_model_request(middleware, request)

    assert [tool.name for tool in captured.tools] == [
        "read_file",
        "record_hypothesis",
    ]
    assert "<deepfix_investigation_state>" in captured.system_message.text
    assert "hyp-1" not in captured.system_message.text


def test_level_one_exposes_only_three_meta_tools(tmp_path):
    middleware = middleware_fixture(tmp_path, stagnation_level=1)

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert {tool.name for tool in captured.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
    }


def test_normal_investigation_exposes_diagnostic_artifact_tools(tmp_path):
    middleware = middleware_fixture(tmp_path)

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert {
        "search_diagnostic_artifacts",
        "read_diagnostic_artifact",
    } <= {tool.name for tool in captured.tools}


def test_level_one_exposes_only_the_unconsumed_permitted_artifact_tool(tmp_path):
    middleware = middleware_fixture(tmp_path, stagnation_level=1)
    middleware.coordinator.grant_investigation_permit(
        "task-a",
        ContinueInvestigationInput(
            hypothesis_ids=["hyp-1"],
            unresolved_question="which archived failure proves the branch?",
            expected_evidence="the original AssertionError and traceback",
            tool_name="search_diagnostic_artifacts",
            target="AssertionError",
            reason="the live ToolMessage contains only an offload preview",
        ),
    )

    before = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )
    assert {tool.name for tool in before.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
        "search_diagnostic_artifacts",
    }

    request = tool_request(
        "search_diagnostic_artifacts",
        "artifact-search-1",
        {"query": "  AssertionError  "},
    )
    middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(
            content="match",
            tool_call_id="artifact-search-1",
            name="search_diagnostic_artifacts",
        ),
    )

    after = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )
    assert {tool.name for tool in after.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
    }


def test_interrupt_command_does_not_mark_pytest_as_executed(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="editing")

    result = middleware.wrap_tool_call(
        pytest_tool_request(),
        lambda request: Command(goto="approval"),
    )

    assert isinstance(result, Command)
    assert middleware.coordinator.state("task-a").agent_phase is AgentPhase.EDITING


def test_real_pytest_tool_message_records_testing_then_failure(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="editing")

    result = middleware.wrap_tool_call(
        pytest_tool_request(),
        lambda request: pytest_result(exit_code=1),
    )

    events = middleware.coordinator.store.list_events("task-a")
    assert [item.event_type for item in events[-4:]] == [
        "verification_execution_observed",
        "phase_changed",
        "post_edit_test_observed",
        "phase_changed",
    ]
    assert result.artifact["exit_code"] == 1
    assert middleware.coordinator.state("task-a").agent_phase is AgentPhase.DIAGNOSING


def test_level_two_blocks_before_handler_execution(tmp_path):
    middleware = middleware_fixture(tmp_path, stagnation_level=2)
    called = False

    def handler(request):
        nonlocal called
        called = True
        return read_result()

    with pytest.raises(InvestigationStagnationError):
        middleware.wrap_tool_call(read_request(), handler)

    assert called is False


def test_committed_tool_receipt_prevents_side_effect_reexecution(tmp_path):
    middleware = middleware_fixture(
        tmp_path,
        phase="planning",
        fail_next_event_commit=True,
    )
    executions = 0

    def handler(request):
        nonlocal executions
        executions += 1
        return edit_result(call_id="edit-1")

    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(edit_request(call_id="edit-1"), handler)
    middleware.wrap_tool_call(edit_request(call_id="edit-1"), handler)

    assert executions == 1
    receipt = middleware.receipts.load("task-a", "edit-1")
    assert receipt is not None
    assert receipt.tool_message.tool_call_id == "edit-1"


def test_async_tool_hook_calls_handler_at_most_once_on_replay(tmp_path):
    middleware = middleware_fixture(
        tmp_path,
        phase="planning",
        fail_next_event_commit=True,
    )
    executions = 0

    async def handler(request):
        nonlocal executions
        executions += 1
        return edit_result(call_id="edit-async")

    async def run():
        request = edit_request("edit-async")
        with pytest.raises(InvestigationStateError):
            await middleware.awrap_tool_call(request, handler)
        await middleware.awrap_tool_call(request, handler)

    asyncio.run(run())

    assert executions == 1


def test_receipt_store_is_verified_and_task_isolated(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="planning")
    result = edit_result("edit-receipt")
    request = edit_request("edit-receipt")

    middleware.wrap_tool_call(request, lambda received: result)

    receipt = middleware.receipts.load("task-a", "edit-receipt")
    assert receipt is not None
    assert receipt.tool_message == result
    assert middleware.receipts.load("task-b", "edit-receipt") is None
