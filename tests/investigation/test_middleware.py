import asyncio
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

from deepfix.domain_repositories.execution import ExecutionRepository
from deepfix.investigation.errors import (
    InvestigationStateError,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.operations import OperationStatus
from investigation.helpers import coordinator_fixture


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
        lambda updated: captured.append(updated) or ModelResponse(result=[AIMessage(content="ok")]),
    )
    return captured[0]


def tool_request(
    name: str,
    call_id: str,
    args: dict[str, object],
    *,
    allowed_capabilities: list[str] | None = None,
) -> ToolCallRequest:
    tool = named_tools(name)[0]
    runtime = ToolRuntime(
        state={"messages": []},
        context=None,
        config={
            "configurable": {
                "thread_id": "task-a",
                **(
                    {"allowed_capabilities": allowed_capabilities}
                    if allowed_capabilities is not None
                    else {}
                ),
            }
        },
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
            "file_path": ("/.deepfix-artifacts/large_tool_results/pytest-current"),
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
        "compact_conversation",
    )


def middleware_fixture(
    tmp_path: Path, *, phase="investigating", stagnation_level=0,
    fail_next_event_commit=False, with_journal=False,
) -> InvestigationMiddleware:
    coordinator = coordinator_fixture(tmp_path)
    collector = coordinator.evidence_collector
    if fail_next_event_commit:
        original = collector.collect_pair
        failed = False
        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("simulated evidence commit failure")
            return original(*args, **kwargs)
        collector.collect_pair = fail_once
    return InvestigationMiddleware(
        coordinator.tasks,
        ExecutionRepository(coordinator.evidence_repository.database),
        ToolResultArtifactStorage(tmp_path / "artifacts" / "investigation_receipts"),
        collector,
    )


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


def test_unexpected_inner_interrupt_preserves_started_operation_for_recovery(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="editing")

    with pytest.raises(InvestigationStateError) as captured:
        middleware.wrap_tool_call(
            pytest_tool_request(),
            lambda request: Command(goto="approval"),
        )

    assert captured.value.recovery.error_code == "operation_result_not_durable"
    operation = middleware.execution.load_operation(
        stable_investigation_id("operation", "task-a", "pytest-1")
    )
    assert operation is not None
    assert operation.status is OperationStatus.STARTED


def test_missing_timeout_plugin_explains_isolated_environment_recovery(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="investigating")
    request = tool_request(
        "execute",
        "pytest-timeout-missing",
        {"command": ("python -m pytest python_testcases/test_find_in_sorted.py -q --timeout=5")},
    )
    raw_result = ToolMessage(
        id="msg-pytest-timeout-missing",
        content="ERROR: unrecognized arguments: --timeout=5",
        tool_call_id="pytest-timeout-missing",
        artifact={"exit_code": 4},
    )

    result = middleware.wrap_tool_call(request, lambda _request: raw_result)

    feedback = str(result.content)
    assert "PYTHONNOUSERSITE=1" in feedback
    assert "pytest-timeout" in feedback
    assert "独立虚拟环境" in feedback
    assert "--python" in feedback


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


def test_redirected_pytest_is_corrected_then_clean_command_executes(tmp_path, monkeypatch):
    middleware = middleware_fixture(tmp_path, stagnation_level=1)
    monkeypatch.setattr("deepfix.investigation.middleware._PLATFORM_NAME", "nt")
    command = "python -m pytest python_testcases/test_knapsack.py -v"
    executions = 0

    def handler(request):
        nonlocal executions
        executions += 1
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

    result = middleware.wrap_tool_call(
        tool_request("execute", "pytest-clean", {"command": command}),
        handler,
    )

    assert result.artifact["exit_code"] == 0
    assert executions == 1


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
    receipt = middleware.execution.load_receipt("task-a", "edit-1")
    assert receipt is not None
    assert receipt.tool_message.tool_call_id == "edit-1"


def test_atomic_observation_failure_after_edit_does_not_reexecute_handler(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "src" / "sign.py"
    target.parent.mkdir()
    target.write_text("return -value\n", encoding="utf-8")
    middleware = middleware_fixture(
        tmp_path,
        phase="planning",
        with_journal=True,
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        target.write_text("return value\n", encoding="utf-8")
        return edit_result("journal-edit-1")

    def fail_observation(*args, **kwargs):
        raise OSError("simulated execution commit failure")

    monkeypatch.setattr(
        middleware.execution,
        "observe_with_receipt",
        fail_observation,
    )

    with pytest.raises(InvestigationStateError) as first:
        middleware.wrap_tool_call(edit_request("journal-edit-1"), handler)
    with pytest.raises(InvestigationStateError) as second:
        middleware.wrap_tool_call(edit_request("journal-edit-1"), handler)

    assert first.value.recovery.error_code == "tool_receipt_persistence_failed"
    assert second.value.recovery.error_code == "operation_reconciliation_required"
    assert calls == 1
    entries = middleware.execution.list_incomplete("task-a")
    assert len(entries) == 1
    assert entries[0].status is OperationStatus.STARTED


def test_successful_edit_commits_journal_after_receipt_and_evidence(tmp_path):
    target = tmp_path / "src" / "sign.py"
    target.parent.mkdir()
    target.write_text("return -value\n", encoding="utf-8")
    middleware = middleware_fixture(
        tmp_path,
        phase="planning",
        with_journal=True,
    )

    def handler(request):
        target.write_text("return value\n", encoding="utf-8")
        return edit_result("journal-edit-success")

    result = middleware.wrap_tool_call(
        edit_request("journal-edit-success"),
        handler,
    )

    assert result.status == "success"
    assert middleware.execution.list_incomplete("task-a") == []
    entry = middleware.execution.load_operation(
        stable_investigation_id(
            "operation",
            "task-a",
            "journal-edit-success",
        )
    )
    assert entry is not None
    assert entry.status is OperationStatus.COMMITTED
    assert entry.pre_state.file_hash != entry.post_state.file_hash


def test_evidence_commit_failure_leaves_observed_operation_and_resume_does_not_edit_twice(
    tmp_path,
):
    target = tmp_path / "src" / "sign.py"
    target.parent.mkdir()
    target.write_text("return -value\n", encoding="utf-8")
    middleware = middleware_fixture(
        tmp_path,
        phase="planning",
        with_journal=True,
        fail_next_event_commit=True,
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        target.write_text("return value\n", encoding="utf-8")
        return edit_result("journal-edit-observed")

    request = edit_request("journal-edit-observed")
    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(request, handler)

    operation_id = stable_investigation_id("operation", "task-a", "journal-edit-observed")
    assert middleware.execution.load_operation(operation_id).status is OperationStatus.OBSERVED
    middleware.wrap_tool_call(request, handler)

    assert calls == 1
    assert middleware.execution.load_operation(operation_id).status is OperationStatus.COMMITTED


def test_committed_edit_uses_authoritative_receipt_without_legacy_file(tmp_path):
    target = tmp_path / "src" / "sign.py"
    target.parent.mkdir()
    target.write_text("return -value\n", encoding="utf-8")
    middleware = middleware_fixture(tmp_path, phase="planning", with_journal=True)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        target.write_text("return value\n", encoding="utf-8")
        return edit_result("journal-edit-missing-receipt")

    request = edit_request("journal-edit-missing-receipt")
    middleware.wrap_tool_call(request, handler)
    receipt_files = list(middleware.artifacts.root_dir.rglob("*.json"))
    assert receipt_files == []
    assert middleware.execution.load_receipt("task-a", "journal-edit-missing-receipt") is not None

    result = middleware.wrap_tool_call(request, handler)

    assert result.status == "success"
    assert calls == 1


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

    receipt = middleware.execution.load_receipt("task-a", "edit-receipt")
    assert receipt is not None
    assert receipt.tool_message == result
    assert middleware.execution.load_receipt("task-b", "edit-receipt") is None


def test_receipt_persistence_failure_keeps_bounded_recovery_diagnostic(
    tmp_path,
    monkeypatch,
):
    middleware = middleware_fixture(tmp_path)

    def fail_record(receipt):
        raise OSError("simulated receipt disk failure")

    monkeypatch.setattr(middleware.execution, "record_receipt", fail_record)

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
    assert middleware.execution.load_receipt("task-a", "read-source") is not None
    assert middleware.execution.load_receipt("task-a", "read-test") is not None


@pytest.mark.parametrize("tool_name", ["grep", "read_file"])
def test_read_after_external_file_modification_is_not_stale_duplicate(tmp_path, tool_name):
    middleware = middleware_fixture(tmp_path)
    target = tmp_path / "changed.py"
    target.write_text("before", encoding="utf-8")
    arguments = {"file_path": "/changed.py"} if tool_name == "read_file" else {"pattern": "after"}

    def handler(request):
        return ToolMessage(
            content=target.read_text(encoding="utf-8"),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )

    middleware.wrap_tool_call(tool_request(tool_name, "before", arguments), handler)
    target.write_text("after", encoding="utf-8")
    result = middleware.wrap_tool_call(tool_request(tool_name, "after", arguments), handler)
    assert result.status == "success"
    assert result.content == "after"
