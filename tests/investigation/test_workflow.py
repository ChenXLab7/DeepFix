from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from deepfix.domain_repositories.execution import ExecutionRepository
from deepfix.investigation.errors import (
    InvestigationStateError,
)
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.models import (
    InvestigationCapability,
    InvestigationRecoveryMetadata,
    RecordHypothesisInput,
)
from deepfix.investigation.receipts import ToolResultArtifactStorage
from investigation.helpers import (
    coordinator_fixture,
    seed_checked_location,
    seed_evidence,
    supported_input,
)


@dataclass(frozen=True)
class ScriptStep:
    name: str
    call_id: str
    args: dict[str, object]
    result: ToolMessage | None = None
    hypothesis: RecordHypothesisInput | None = None


class OfflineRepairHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.investigation = coordinator_fixture(tmp_path)
        self.middleware = InvestigationMiddleware(
            self.investigation,
            ExecutionRepository(self.investigation.evidence_repository.database),
            ToolResultArtifactStorage(tmp_path / "artifacts" / "investigation_receipts"),
            {
                "read_file": InvestigationCapability.READ,
                "grep": InvestigationCapability.SEARCH,
                "execute": InvestigationCapability.EXECUTE,
                "edit_file": InvestigationCapability.MODIFY,
                "record_hypothesis": InvestigationCapability.META,
                "continue_investigation": InvestigationCapability.META,
            },
        )
        self.tool_names: list[str] = []
        self.execution_counts: Counter[str] = Counter()

    @property
    def completed_read_calls(self) -> int:
        return sum(
            count for call_id, count in self.execution_counts.items() if call_id.startswith("read-")
        )

    def run(self, steps: list[ScriptStep]):
        for step in steps:
            self.tool_names.append(step.name)
            if step.hypothesis is not None:
                self.investigation.record_hypothesis(
                    "task-a",
                    step.hypothesis,
                    source_id=step.call_id,
                )
                continue
            request = _tool_request(step.name, step.call_id, step.args)

            def handler(received, current=step):
                del received
                self.execution_counts[current.call_id] += 1
                assert current.result is not None
                return current.result

            self.middleware.wrap_tool_call(request, handler)
        return self.investigation.state("task-a")


def _tool_request(
    name: str,
    call_id: str,
    args: dict[str, object],
) -> ToolCallRequest:
    def run() -> str:
        return "unused"

    tool = StructuredTool.from_function(
        run,
        name=name,
        description=f"{name} workflow tool",
    )
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


def _result(
    call_id: str,
    content: str,
    artifact: dict[str, object] | None = None,
) -> ToolMessage:
    return ToolMessage(
        id=f"msg-{call_id}",
        content=content,
        tool_call_id=call_id,
        artifact=artifact,
    )


def _normal_flow(evidence_id: str) -> list[ScriptStep]:
    return [
        ScriptStep(
            "execute",
            "test-fail",
            {"command": "python -m pytest -q"},
            _result("test-fail", "1 failed", {"exit_code": 1}),
        ),
        ScriptStep(
            "read_file",
            "read-test",
            {"file_path": "/tests/test_sign.py"},
            _result("read-test", "assert apply_sign(-2, -1) == 2"),
        ),
        ScriptStep(
            "read_file",
            "read-src",
            {"file_path": "/src/sign.py"},
            _result("read-src", "return -value"),
        ),
        ScriptStep(
            "record_hypothesis",
            "hyp-1",
            {},
            hypothesis=supported_input([evidence_id]),
        ),
        ScriptStep(
            "edit_file",
            "edit-1",
            {"file_path": "/src/sign.py"},
            _result(
                "edit-1",
                "edited",
                {
                    "operation": "edit",
                    "status": "succeeded",
                    "path": "/src/sign.py",
                },
            ),
        ),
        ScriptStep(
            "execute",
            "test-pass",
            {"command": "python -m pytest -q"},
            _result("test-pass", "1 passed", {"exit_code": 0}),
        ),
    ]


def _exploratory_reads(count: int) -> list[ScriptStep]:
    return [
        ScriptStep(
            "read_file",
            f"read-{index}",
            {"file_path": f"/unrelated/file_{index}.py"},
            _result(f"read-{index}", f"value_{index} = {index}"),
        )
        for index in range(count)
    ]


def _repeated_pytest_steps(hypothesis_id: str) -> list[ScriptStep]:
    def failed(call_id: str) -> ScriptStep:
        return ScriptStep(
            "execute",
            call_id,
            {"command": "python -m pytest -q"},
            _result(call_id, "same gcd failure", {"exit_code": 1}),
        )

    del hypothesis_id
    return [
        failed("repeat-1"),
        failed("repeat-2"),
        failed("repeat-3"),
        failed("repeat-4"),
        failed("repeat-5"),
        failed("repeat-6"),
    ]


def test_normal_bugfix_flow_records_domain_progress_without_phase_navigation(
    tmp_path: Path,
) -> None:
    harness = OfflineRepairHarness(tmp_path)
    evidence_id = "evidence-test-fail"
    seed_evidence(harness.investigation.evidence_repository, "task-a", evidence_id)
    seed_checked_location(
        harness.investigation.store,
        "task-a",
        "src/sign.py",
        1,
        20,
    )

    state = harness.run(_normal_flow(evidence_id))

    assert state.supported_hypothesis_ids
    assert len(state.test_evidence_ids) == 2
    assert state.stagnation_level == 0
    assert not any(
        event.event_type == "phase_changed"
        for event in harness.investigation.store.list_events("task-a")
    )
    assert harness.tool_names == [
        "execute",
        "read_file",
        "read_file",
        "record_hypothesis",
        "edit_file",
        "execute",
    ]


def test_bfs_repository_scan_is_advisory_not_a_hard_tool_gate(tmp_path: Path) -> None:
    harness = OfflineRepairHarness(tmp_path)

    state = harness.run(_exploratory_reads(20))

    assert harness.completed_read_calls == 20
    assert state.stagnation_level == 1


def test_gcd_exact_repeat_reevaluates_strategy_without_pausing(tmp_path: Path) -> None:
    harness = OfflineRepairHarness(tmp_path)
    hypothesis = harness.investigation.record_hypothesis(
        "task-a",
        RecordHypothesisInput(
            statement="negative normalization repeats",
            evidence_ids=[],
            checked_locations=[],
            target_state="candidate",
            reason="candidate for bounded reproduction",
        ),
        source_id="hyp-gcd-call",
    )

    state = harness.run(_repeated_pytest_steps(hypothesis.hypothesis_id))

    events = harness.investigation.store.list_events("task-a")
    assert sum(event.event_type == "investigation_permit_granted" for event in events) == 0
    assert harness.execution_counts.total() == 1
    assert state.reevaluation_required is True


def test_store_failure_after_tool_execution_does_not_rerun_tool(
    tmp_path: Path,
) -> None:
    coordinator = coordinator_fixture(tmp_path)
    coordinator.store.ensure_started("task-a")
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
                    state_version=current.version,
                    last_event_sequence=coordinator.store.last_sequence(task_id),
                    tool_call_id=str(tool_call["id"]),
                    checkpoint_available=True,
                    recovery_action="replay_event_commit_without_rerunning_tool",
                )
            )
        return original_record(task_id, tool_call, result)

    coordinator.record_tool_result = fail_once
    middleware = InvestigationMiddleware(
        coordinator,
        ExecutionRepository(coordinator.evidence_repository.database),
        ToolResultArtifactStorage(tmp_path / "artifacts" / "investigation_receipts"),
        {"edit_file": InvestigationCapability.MODIFY},
    )
    request = _tool_request("edit_file", "edit-1", {"file_path": "/src/sign.py"})
    result = _result(
        "edit-1",
        "edited",
        {"operation": "edit", "status": "succeeded", "path": "/src/sign.py"},
    )
    executions = 0

    def handler(received):
        nonlocal executions
        del received
        executions += 1
        return result

    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(request, handler)
    middleware.wrap_tool_call(request, handler)

    assert executions == 1
    event_ids = [item.event_id for item in coordinator.store.list_events("task-a")]
    assert len(event_ids) == len(set(event_ids))
