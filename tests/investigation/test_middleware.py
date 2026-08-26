import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
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
    RecordHypothesisInput,
    ToolObservation,
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


def model_request(
    *,
    tools: list[BaseTool],
    messages: list | None = None,
) -> ModelRequest:
    request_messages = messages or [HumanMessage(content="continue")]
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=request_messages,
        system_message=SystemMessage(content="base"),
        tools=tools,
        state={"messages": request_messages},
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


def offloaded_result_read_request(call_id: str = "artifact-read-1") -> ToolCallRequest:
    return tool_request(
        "read_file",
        call_id,
        {
            "file_path": (
                "/.deepfix-artifacts/large_tool_results/pytest-current"
            ),
            "offset": 60,
            "limit": 80,
        },
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


def save_progress_request(call_id: str) -> ToolCallRequest:
    return tool_request(
        "save_progress",
        call_id,
        {
            "phase": "testing",
            "summary": "tests passed",
            "facts": [],
            "evidence": [],
            "hypotheses": [],
            "checked_files": [],
            "experiments": [],
            "next_steps": [],
            "unresolved_questions": [],
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


def artifactless_edit_result(call_id: str) -> ToolMessage:
    return ToolMessage(
        id=f"msg-{call_id}",
        content="Successfully replaced 1 instance(s) of the string in '/src/sign.py'",
        name="edit_file",
        tool_call_id=call_id,
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
    return InvestigationMiddleware(
        coordinator,
        ToolExecutionReceiptStore(tmp_path / "artifacts" / "investigation_receipts"),
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
    assert '<hypothesis id="hyp-1" state="candidate">' in (
        captured.system_message.text
    )


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


def test_repeated_save_progress_failures_hide_memory_tool_for_generation(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="testing")

    for attempt in range(2):
        call_id = f"save-failed-{attempt}"
        middleware.wrap_tool_call(
            save_progress_request(call_id),
            lambda request, call_id=call_id: ToolMessage(
                content="保存工作记忆失败：source ref_id 不属于当前任务",
                name="save_progress",
                tool_call_id=call_id,
                status="error",
            ),
        )

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert "save_progress" not in {tool.name for tool in captured.tools}
    assert "本进展代次内不再调用 save_progress" in captured.system_message.text


def test_successful_save_progress_is_not_repeated_in_same_generation(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="testing")
    middleware.wrap_tool_call(
        save_progress_request("save-success"),
        lambda request: ToolMessage(
            content="工作记忆已保存为版本 1",
            name="save_progress",
            tool_call_id="save-success",
            status="success",
        ),
    )

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert "save_progress" not in {tool.name for tool in captured.tools}
    assert "Working Memory 已保存" in captured.system_message.text
    assert "直接返回 RepairOutcome" in captured.system_message.text


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


def test_diagnostic_read_requires_hypothesis_decision_and_renders_instruction(
    tmp_path,
):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    middleware.coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="artifact_read",
            tool_call_id="artifact-read",
            result_fingerprint="artifact-result",
            payload={
                "artifact_id": "artifact_" + "a" * 32,
                "start_line": 10,
                "end_line": 20,
            },
        ),
    )

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert {tool.name for tool in captured.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
    }
    assert "<diagnostic_decision_checkpoint>" in captured.system_message.text
    assert "record_hypothesis" in captured.system_message.text
    assert "不要继续读取相邻" in captured.system_message.text


def test_decision_checkpoint_returns_one_correction_then_pauses(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    middleware.coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="artifact_read",
            tool_call_id="artifact-read",
            result_fingerprint="artifact-result",
            payload={"artifact_id": "artifact_" + "a" * 32},
        ),
    )
    called = False

    def handler(request):
        nonlocal called
        called = True
        return read_result()

    correction = middleware.wrap_tool_call(read_request(), handler)

    assert called is False
    assert isinstance(correction, ToolMessage)
    assert correction.status == "error"
    assert correction.artifact["result_type"] == "diagnostic_decision_correction"
    assert "record_hypothesis" in correction.content
    assert middleware.coordinator.state("task-a").decision_correction_used is True

    with pytest.raises(InvestigationStagnationError) as caught:
        middleware.wrap_tool_call(
            tool_request("read_file", "read-2", {"file_path": "/src/other.py"}),
            handler,
        )

    assert caught.value.recovery.error_code == "diagnostic_decision_ignored"


def test_direct_read_of_offloaded_result_routes_to_diagnostic_tools(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    called = False

    def handler(request):
        nonlocal called
        called = True
        raise AssertionError("ordinary read_file must not read diagnostic artifacts")

    result = middleware.wrap_tool_call(
        offloaded_result_read_request(),
        handler,
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.artifact == {
        "result_type": "diagnostic_artifact_redirect",
        "error_code": "use_diagnostic_artifact_tools",
        "operation": "read_file",
    }
    assert "search_diagnostic_artifacts" in result.content
    state = middleware.coordinator.state("task-a")
    assert all("large_tool_results" not in item.path for item in state.checked_files)
    assert middleware.coordinator.store.list_events("task-a")[-1].event_type == (
        "tool_completed"
    )


def test_async_direct_read_of_offloaded_result_skips_handler(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    called = False

    async def handler(request):
        nonlocal called
        called = True
        raise AssertionError("ordinary read_file must not read diagnostic artifacts")

    result = asyncio.run(
        middleware.awrap_tool_call(
            offloaded_result_read_request("artifact-read-async"),
            handler,
        )
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


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


def test_post_edit_pytest_failure_requires_hypothesis_reevaluation(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="planning")
    middleware.wrap_tool_call(
        edit_request(call_id="edit-artifactless"),
        lambda request: artifactless_edit_result("edit-artifactless"),
    )
    assert middleware.coordinator.state("task-a").agent_phase is AgentPhase.EDITING

    middleware.wrap_tool_call(
        pytest_tool_request(),
        lambda request: pytest_result(exit_code=1),
    )

    state = middleware.coordinator.state("task-a")
    assert state.agent_phase is AgentPhase.DIAGNOSING
    assert state.repair_reevaluation_required is True
    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )
    assert {tool.name for tool in captured.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
    }
    assert "修改后验证失败" in captured.system_message.text


def test_two_diagnostic_pytest_runs_require_hypothesis_decision(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="investigating")
    for call_id, command in (
        ("pytest-diagnostic-1", "python -m pytest -q"),
        ("pytest-diagnostic-2", "python -m pytest -q -x"),
    ):
        middleware.wrap_tool_call(
            tool_request("execute", call_id, {"command": command}),
            lambda request, call_id=call_id: ToolMessage(
                id=f"msg-{call_id}",
                content="1 failed",
                name="execute",
                tool_call_id=call_id,
                artifact={"exit_code": 1},
            ),
        )

    state = middleware.coordinator.state("task-a")
    assert state.diagnostic_test_count_since_decision == 2
    assert state.diagnostic_decision_required is True
    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )
    assert {tool.name for tool in captured.tools} == {
        "record_hypothesis",
        "continue_investigation",
        "save_progress",
    }


def test_duplicate_execute_is_corrected_once_then_pauses(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="investigating")
    executions = 0

    def handler(request):
        nonlocal executions
        executions += 1
        call_id = str(request.tool_call["id"])
        return ToolMessage(
            id=f"msg-{call_id}",
            content="probe complete",
            name="execute",
            tool_call_id=call_id,
            artifact={"exit_code": 0},
        )

    command = {"command": 'python -c "print(1)"'}
    middleware.wrap_tool_call(tool_request("execute", "probe-1", command), handler)
    correction = middleware.wrap_tool_call(
        tool_request("execute", "probe-2", command), handler
    )

    assert executions == 1
    assert isinstance(correction, ToolMessage)
    assert correction.artifact["result_type"] == "duplicate_execute_correction"

    with pytest.raises(InvestigationStagnationError) as caught:
        middleware.wrap_tool_call(
            tool_request("execute", "probe-3", command), handler
        )

    assert executions == 1
    assert caught.value.recovery.error_code == "duplicate_execute_ignored"


def test_windows_unix_pipeline_is_blocked_before_execution(tmp_path, monkeypatch):
    middleware = middleware_fixture(tmp_path, phase="investigating")
    monkeypatch.setattr("deepfix.investigation.middleware._PLATFORM_NAME", "nt")
    called = False

    def handler(request):
        nonlocal called
        called = True
        raise AssertionError("non-portable Windows command must not execute")

    result = middleware.wrap_tool_call(
        tool_request(
            "execute",
            "pytest-head",
            {"command": "python -m pytest -q 2>&1 | head -50"},
        ),
        handler,
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.artifact["result_type"] == "windows_shell_correction"
    assert "--tb=short" in result.content


def test_redirected_pytest_is_corrected_then_bound_permit_executes(tmp_path, monkeypatch):
    middleware = middleware_fixture(tmp_path, stagnation_level=1)
    monkeypatch.setattr("deepfix.investigation.middleware._PLATFORM_NAME", "nt")
    command = "python -m pytest python_testcases/test_knapsack.py -v"
    middleware.coordinator.grant_investigation_permit(
        "task-a",
        ContinueInvestigationInput(
            hypothesis_ids=["hyp-1"],
            unresolved_question="does the test reproduce?",
            expected_evidence="pytest exit code",
            tool_name="execute",
            target=command,
            reason="run the requested test",
        ),
    )
    executions = 0
    permit_consumed_before_execution = False

    def handler(request):
        nonlocal executions, permit_consumed_before_execution
        executions += 1
        permit = middleware.coordinator.state("task-a").permit
        permit_consumed_before_execution = permit is not None and permit.consumed
        return ToolMessage(
            id="pytest-result",
            content="1 passed",
            name="execute",
            tool_call_id=str(request.tool_call["id"]),
            artifact={"exit_code": 0},
        )

    correction = middleware.wrap_tool_call(
        tool_request(
            "execute",
            "pytest-redirected",
            {"command": f"{command} 2>&1"},
        ),
        handler,
    )

    assert executions == 0
    assert isinstance(correction, ToolMessage)
    assert correction.artifact["result_type"] == "pytest_shell_correction"
    assert correction.artifact["suggested_command"] == command
    assert middleware.coordinator.state("task-a").permit.consumed is False

    result = middleware.wrap_tool_call(
        tool_request("execute", "pytest-clean", {"command": command}),
        handler,
    )

    assert result.artifact["exit_code"] == 0
    assert executions == 1
    assert permit_consumed_before_execution is True


def test_model_prompt_exposes_current_work_unit_ids_for_save_progress(tmp_path):
    middleware = middleware_fixture(tmp_path)
    messages = [
        HumanMessage(id="user-1", content="inspect sign"),
        AIMessage(
            id="read-request",
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "src/sign.py"},
                    "id": "read-sign",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            id="read-result",
            content="return -value",
            name="read_file",
            tool_call_id="read-sign",
        ),
    ]

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools(), messages=messages),
    )

    assert "<available_work_unit_refs>" in captured.system_message.text
    assert 'path="src/sign.py"' in captured.system_message.text
    assert 'ref_id="wu_' in captured.system_message.text


def test_model_prompt_exposes_current_hypothesis_ids_and_states(tmp_path):
    middleware = middleware_fixture(tmp_path)
    hypothesis = middleware.coordinator.record_hypothesis(
        "task-a",
        RecordHypothesisInput(
            statement="capacity=0 may expose a boundary bug",
            evidence_ids=[],
            checked_locations=[],
            target_state="candidate",
            reason="needs a targeted test",
        ),
        source_id="candidate-call",
    )

    captured = capture_model_request(
        middleware,
        model_request(tools=all_test_tools()),
    )

    assert "<current_hypotheses>" in captured.system_message.text
    assert f'id="{hypothesis.hypothesis_id}"' in captured.system_message.text
    assert 'state="candidate"' in captured.system_message.text


def test_alternating_duplicate_candidates_are_each_corrected_once_then_pause(tmp_path):
    middleware = middleware_fixture(tmp_path)
    statement = "capacity=0 may expose a boundary bug"
    existing = middleware.coordinator.record_hypothesis(
        "task-a",
        RecordHypothesisInput(
            statement=statement,
            evidence_ids=[],
            checked_locations=[],
            target_state="candidate",
            reason="needs a targeted test",
        ),
        source_id="candidate-original",
    )
    other_statement = "empty items may expose a boundary bug"
    other = middleware.coordinator.record_hypothesis(
        "task-a",
        RecordHypothesisInput(
            statement=other_statement,
            evidence_ids=[],
            checked_locations=[],
            target_state="candidate",
            reason="needs a targeted test",
        ),
        source_id="candidate-other",
    )
    arguments = {
        "statement": statement,
        "evidence_ids": [],
        "checked_locations": [],
        "target_state": "candidate",
        "reason": "needs a targeted test",
    }
    other_arguments = {**arguments, "statement": other_statement}
    handler_called = False

    def handler(request):
        nonlocal handler_called
        handler_called = True
        raise AssertionError("duplicate candidate must not reach the tool")

    correction = middleware.wrap_tool_call(
        tool_request(
            "record_hypothesis",
            "candidate-duplicate-1",
            arguments,
        ),
        handler,
    )

    assert handler_called is False
    assert correction.status == "error"
    assert correction.artifact["result_type"] == "duplicate_hypothesis_correction"
    assert correction.artifact["hypothesis_id"] == existing.hypothesis_id
    assert "continue_investigation" in correction.text

    other_correction = middleware.wrap_tool_call(
        tool_request(
            "record_hypothesis",
            "candidate-other-duplicate-1",
            other_arguments,
        ),
        handler,
    )

    assert other_correction.artifact["hypothesis_id"] == other.hypothesis_id

    with pytest.raises(InvestigationStagnationError) as captured:
        middleware.wrap_tool_call(
            tool_request(
                "record_hypothesis",
                "candidate-duplicate-2",
                arguments,
            ),
            handler,
        )

    assert captured.value.recovery.error_code == "duplicate_hypothesis_ignored"
    assert handler_called is False


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


def test_receipt_persistence_failure_keeps_bounded_recovery_diagnostic(
    tmp_path,
    monkeypatch,
):
    middleware = middleware_fixture(tmp_path)

    def fail_replace(source, destination):
        raise OSError("simulated receipt disk failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(InvestigationStateError) as captured:
        middleware.wrap_tool_call(read_request(), lambda request: read_result())

    recovery = captured.value.recovery
    assert recovery.error_code == "tool_receipt_persistence_failed"
    assert recovery.error_type == "OSError"
    assert recovery.error_detail == "simulated receipt disk failure"
    assert recovery.error_fingerprint.startswith("error_")
    assert len(recovery.error_detail) <= 500


def test_two_parallel_read_files_commit_both_receipts_and_observations(tmp_path):
    middleware = middleware_fixture(tmp_path)

    def run(call_id, path, content):
        request = tool_request(
            "read_file",
            call_id,
            {"file_path": path},
        )
        return middleware.wrap_tool_call(
            request,
            lambda received: ToolMessage(
                content=content,
                name="read_file",
                tool_call_id=call_id,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run, "read-source", "/src/code.py", "source"),
            pool.submit(run, "read-test", "/tests/test_code.py", "test"),
        ]
        results = [future.result() for future in futures]

    assert [result.status for result in results] == ["success", "success"]
    assert middleware.receipts.load("task-a", "read-source") is not None
    assert middleware.receipts.load("task-a", "read-test") is not None
    assert {item.path for item in middleware.coordinator.state("task-a").checked_files} >= {
        "src/code.py",
        "tests/test_code.py",
    }
