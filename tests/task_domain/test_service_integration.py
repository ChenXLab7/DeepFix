from __future__ import annotations

from collections import deque

import pytest
from langgraph.types import Interrupt

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.models import SystemTestEvidence
from deepfix.config import ApprovalMode, load_config
from deepfix.models import RepairOutcome, TaskStatus
from deepfix.models import TestResult as RepairTestResult
from deepfix.research.store import ResearchEvidenceStore
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.repository import TaskRepository
from deepfix.verification import VerificationPolicyStore
from deepfix.workspace import WorkspaceFactory


class FakeAgent:
    def __init__(self, *results) -> None:
        self.results = deque(results)

    def invoke(self, _value, _config):
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class RecordingTaskRepository(TaskRepository):
    def __init__(self, database) -> None:
        self.events: list[str] = []
        super().__init__(database)

    def create_definition(self, definition):
        self.events.append("create_definition")
        return super().create_definition(definition)

    def save_verification_policy(self, policy) -> None:
        self.events.append("save_verification_policy")
        super().save_verification_policy(policy)

    def transition_lifecycle(self, task_id, next_status, **kwargs):
        self.events.append(f"transition_lifecycle:{next_status.value}")
        return super().transition_lifecycle(task_id, next_status, **kwargs)

    def save_legacy_projection(self, task) -> None:
        self.events.append("save_legacy_projection")
        super().save_legacy_projection(task)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    project = tmp_path / "project"
    project.mkdir()
    return load_config(project, ApprovalMode.MANUAL)


def service_for(config, agent, *, repository=None, workspace=False):
    repository = repository or TaskRepository(config.database_path)
    return BugfixService(
        agent,
        repository,
        ApprovalPolicy(config.approval_mode),
        config,
        ResearchEvidenceStore(config.database_path),
        workspace_factory=(
            WorkspaceFactory(config.database_path.parent / "workspaces") if workspace else None
        ),
        verification_policy_store=VerificationPolicyStore(tasks=repository),
    )


def no_response() -> dict[str, list]:
    return {"messages": []}


def approval_interrupt():
    return {
        "__interrupt__": (
            Interrupt(
                value={
                    "action_requests": [
                        {
                            "name": "execute",
                            "args": {"command": "python -m pytest -q"},
                            "description": "运行测试",
                        }
                    ]
                },
                id="approval-1",
            ),
        )
    }


def test_service_creates_canonical_state_in_dependency_order(config) -> None:
    repository = RecordingTaskRepository(config.database_path)
    service = service_for(
        config,
        FakeAgent(no_response()),
        repository=repository,
        workspace=True,
    )

    task = service.start("原始修复问题")

    assert repository.events[:4] == [
        "create_definition",
        "save_verification_policy",
        "transition_lifecycle:running",
        "save_legacy_projection",
    ]
    definition = repository.get_definition(task.task_id)
    assert definition.original_problem == "原始修复问题"
    assert definition.original_message_id == task.conversation[0]["id"]


def test_later_user_message_does_not_rewrite_definition(config) -> None:
    service = service_for(config, FakeAgent(no_response(), no_response()))
    task = service.start("原始修复问题")
    original = service.repository.get_definition(task.task_id)

    service.continue_task(task.task_id, "补充约束")

    assert service.repository.get_definition(task.task_id) == original


def test_service_boundaries_update_business_lifecycle(config) -> None:
    running_service = service_for(config, FakeAgent(no_response()))
    running = running_service.start("保持运行")
    assert (
        running_service.repository.get_lifecycle(running.task_id).status
        is TaskLifecycleStatus.RUNNING
    )

    approval_service = service_for(config, FakeAgent(approval_interrupt()))
    waiting = approval_service.start("等待审批")
    assert (
        approval_service.repository.get_lifecycle(waiting.task_id).status
        is TaskLifecycleStatus.WAITING_APPROVAL
    )
    approval_service.pause_task(waiting.task_id, "用户暂停")
    assert (
        approval_service.repository.get_lifecycle(waiting.task_id).status
        is TaskLifecycleStatus.PAUSED
    )

    failed_service = service_for(config, FakeAgent(RuntimeError("model failed")))
    failed = failed_service.start("触发失败")
    assert (
        failed_service.repository.get_lifecycle(failed.task_id).status is TaskLifecycleStatus.FAILED
    )


def test_legacy_phase_change_does_not_increment_business_lifecycle(config) -> None:
    service = service_for(config, FakeAgent(no_response()))
    task = service.start("修复失败测试")
    before = service.repository.get_lifecycle(task.task_id)
    task.status = TaskStatus.EDITING

    service.repository.save_legacy_projection(task)

    after = service.repository.get_lifecycle(task.task_id)
    assert after.status is TaskLifecycleStatus.RUNNING
    assert after.version == before.version
    assert service.repository.get(task.task_id).status is TaskStatus.EDITING


def test_completed_outcome_records_adjudication_with_evidence_ids(config) -> None:
    service = service_for(config, FakeAgent(no_response()), workspace=True)
    task = service.start("验证当前实现，运行 python -m pytest -q")
    task.test_results = [RepairTestResult("python -m pytest -q", 0, "1 passed")]
    service.compaction_store.save_evidence(
        task.task_id,
        SystemTestEvidence(
            evidence_id="test-pass",
            command="python -m pytest -q",
            exit_code=0,
            summary="1 passed",
            tool_call_id="test-call",
            source_message_id="test-result",
            origin="user_specified",
            scope="targeted",
            timing="baseline",
            workspace_baseline_id="baseline-1",
            code_state_hash="code-state-1",
        ),
    )

    completed = service._apply_outcome(
        task,
        RepairOutcome(status="completed", summary="当前实现测试通过"),
    )

    decision = service.repository.latest_adjudication(task.task_id)
    assert completed.resolution == "not_reproduced"
    assert decision is not None
    assert decision.outcome == "not_reproduced"
    assert decision.evidence_ids == ["test-pass"]
    assert decision.operation_ids == []
