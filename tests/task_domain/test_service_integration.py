from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
from httpx import Request, Response, StreamConsumed
from langgraph.types import Interrupt
from openai import BadRequestError

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.errors import ContextRecoveryRequired
from deepfix.compaction.models import ContextRecoveryMetadata, SystemTestEvidence
from deepfix.config import ApprovalMode, load_config
from deepfix.domain_repositories.migration import DomainMigrationReport
from deepfix.investigation.errors import InvestigationStateError
from deepfix.investigation.models import InvestigationRecoveryMetadata
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.operations import (
    NewOperationEntry,
    OperationKind,
    OperationReconciler,
    OperationStateSnapshot,
)
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.outcome import RepairOutcomeCandidate
from deepfix.task_domain.repository import TaskRepository
from deepfix.workspace import WorkspaceFactory


class FakeAgent:
    def __init__(self, *results) -> None:
        self.results = deque(results)
        self.invocations = 0
        self.interrupts = ()

    def invoke(self, _value, _config):
        self.invocations += 1
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        self.interrupts = result.get("__interrupt__", ())
        return result

    def get_state(self, _config):
        return SimpleNamespace(tasks=[SimpleNamespace(interrupts=self.interrupts)])


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


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    project = tmp_path / "project"
    project.mkdir()
    return load_config(project, ApprovalMode.MANUAL)


def service_for(
    config,
    agent,
    *,
    repository=None,
    workspace=False,
    domain_migrator=None,
    operation_reconciler=None,
):
    repository = repository or TaskRepository(config.database_path)
    return BugfixService(
        agent,
        repository,
        ApprovalPolicy(config.approval_mode),
        config,
        workspace_factory=(
            WorkspaceFactory(config.database_path.parent / "workspaces") if workspace else None
        ),
        domain_migrator=domain_migrator,
        operation_reconciler=operation_reconciler,
    )


def no_response() -> dict[str, list]:
    return {"messages": []}


def _not_ready_report(task_id: str) -> DomainMigrationReport:
    return DomainMigrationReport(
        domain="evidence",
        task_id=task_id,
        source_count=1,
        target_count=0,
        source_hash="a" * 64,
        target_hash="b" * 64,
        identity_mismatches=["evidence:missing"],
        ready_to_switch=False,
    )


class NotReadyMigrator:
    def migrate_deterministic_evidence(self, task_id):
        return _not_ready_report(task_id)

    def __getattr__(self, _name):
        return lambda task_id: _not_ready_report(task_id)


def approval_interrupt(interrupt_id: str = "approval-1"):
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
                id=interrupt_id,
            ),
        )
    }


def provider_bad_request() -> BadRequestError:
    request = Request("POST", "https://provider.example/v1/chat/completions")
    return BadRequestError(
        "provider rejected one request",
        response=Response(400, request=request),
        body={"code": "InvalidParameter", "param": ""},
    )


def test_service_creates_canonical_state_in_dependency_order(config) -> None:
    repository = RecordingTaskRepository(config.database_path)
    service = service_for(
        config,
        FakeAgent(no_response()),
        repository=repository,
        workspace=True,
    )

    task = service.start("原始修复问题")

    assert repository.events[:3] == [
        "create_definition",
        "save_verification_policy",
        "transition_lifecycle:running",
    ]
    definition = repository.get_definition(task.task_id)
    assert definition.original_problem == "原始修复问题"
    assert definition.original_message_id
    with repository.database.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM legacy_task_projection WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            is None
        )


def test_unvalidated_domain_migration_pauses_before_agent_invocation(config) -> None:
    agent = FakeAgent(no_response())
    service = service_for(config, agent, domain_migrator=NotReadyMigrator())

    task = service.start("不得在迁移不完整时运行")

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason is not None
    assert "domain_authority_migration_not_ready:evidence" in task.pause_reason
    assert agent.invocations == 0


def test_missing_required_oracle_path_pauses_before_agent_invocation(config) -> None:
    agent = FakeAgent(no_response())
    service = service_for(config, agent, workspace=True)

    task = service.start(
        "修复问题，并运行 python -m pytest tests/test_missing.py -q 验证"
    )

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason == (
        "Required Oracle 路径不可用：tests/test_missing.py"
    )
    assert agent.invocations == 0
    policy = service.repository.load_verification_policy(task.task_id)
    assert policy is not None
    assert policy.required_oracles[0].relevant_paths == ["tests/test_missing.py"]


def test_missing_required_oracle_is_rechecked_before_continue_invocation(config) -> None:
    agent = FakeAgent(no_response())
    service = service_for(config, agent, workspace=True)
    task = service.start(
        "修复问题，并运行 python -m pytest tests/test_missing.py -q 验证"
    )

    resumed = service.continue_task(task.task_id, "测试文件仍然没有提供")

    assert resumed.lifecycle is TaskLifecycleStatus.PAUSED
    assert resumed.pause_reason == "Required Oracle 路径不可用：tests/test_missing.py"
    assert agent.invocations == 0


def test_pytest_output_option_is_not_treated_as_required_oracle_path(config) -> None:
    test_path = config.project_root / "tests" / "test_value.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_value(): assert True\n", encoding="utf-8")
    agent = FakeAgent(no_response())
    service = service_for(config, agent, workspace=True)

    task = service.start(
        "运行 python -m pytest --log-file reports/test.log "
        "tests/test_value.py -q 验证"
    )

    assert task.lifecycle is TaskLifecycleStatus.RUNNING
    assert agent.invocations == 1


def test_missing_oracle_after_boolean_pytest_flag_pauses_before_model(config) -> None:
    agent = FakeAgent(no_response())
    service = service_for(config, agent, workspace=True)

    task = service.start(
        "运行 python -m pytest --setup-show tests/test_missing.py -q 验证"
    )

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason == "Required Oracle 路径不可用：tests/test_missing.py"
    assert agent.invocations == 0


def test_missing_required_oracle_directory_pauses_before_model(config) -> None:
    agent = FakeAgent(no_response())
    service = service_for(config, agent, workspace=True)

    task = service.start("运行 python -m pytest integration/ -q 验证")

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason == "Required Oracle 路径不可用：integration/"
    assert agent.invocations == 0


def test_pytest_usage_error_does_not_consume_test_failure_budget(config) -> None:
    service = service_for(config, FakeAgent(no_response()))
    task = service.start("调查测试环境")
    service.repositories.evidence.record_deterministic(
        task.task_id,
        SystemTestEvidence(
            evidence_id="pytest-usage-error",
            command="python -m pytest -q --bad-option",
            exit_code=4,
            summary="pytest: error: unrecognized arguments: --bad-option",
            tool_call_id="usage-call",
            source_message_id="usage-result",
            origin="repository_existing",
            scope="full_suite",
            timing="post_change",
            workspace_baseline_id="baseline-1",
            code_state_hash="code-state-1",
        ),
        provenance_root_ids=["usage-call"],
    )

    assert service._consecutive_post_change_failures(task.task_id) == 0


def test_domain_migration_exception_pauses_without_leaking_secret(config) -> None:
    class FailingMigrator(NotReadyMigrator):
        def migrate_deterministic_evidence(self, _task_id):
            raise OSError("provider failed with secret")

    agent = FakeAgent(no_response())
    service = service_for(config, agent, domain_migrator=FailingMigrator())

    task = service.start("迁移异常必须暂停")

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason == "领域权威迁移需要恢复：domain_authority_migration_failed:evidence"
    assert "secret" not in task.pause_reason
    assert agent.invocations == 0


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


def test_provider_error_pauses_and_resume_can_reach_approval(config) -> None:
    agent = FakeAgent(provider_bad_request(), approval_interrupt())
    service = service_for(config, agent)

    paused = service.start("供应商临时拒绝后允许恢复")

    assert paused.lifecycle is TaskLifecycleStatus.PAUSED
    assert paused.pause_reason is not None
    assert "BadRequestError" in paused.pause_reason

    waiting = service.continue_task(paused.task_id, "重试并继续验证")

    assert waiting.lifecycle is TaskLifecycleStatus.WAITING_APPROVAL
    assert [action["name"] for action in waiting.pending_actions] == ["execute"]
    assert agent.invocations == 2


def test_failed_task_rejects_resume_before_agent_invocation(config) -> None:
    agent = FakeAgent(RuntimeError("internal bug"))
    service = service_for(config, agent)
    failed = service.start("内部错误保持终态")

    with pytest.raises(ValueError, match="终态任务不能继续"):
        service.continue_task(failed.task_id, "不要再次调用模型")

    assert agent.invocations == 1


def test_local_http_client_usage_error_remains_failed(config) -> None:
    service = service_for(config, FakeAgent(StreamConsumed()))

    failed = service.start("本地 HTTP 客户端使用错误不是可恢复的供应商错误")

    assert failed.lifecycle is TaskLifecycleStatus.FAILED


def test_rejected_approval_is_recorded_and_resumes_same_task(config) -> None:
    agent = FakeAgent(approval_interrupt(), no_response())
    service = service_for(config, agent)
    waiting = service.start("运行需要审批的测试")

    resumed = service.decide(waiting.task_id, ["reject"])

    approvals = service.repositories.execution.list_approvals(waiting.task_id)
    assert resumed.lifecycle is TaskLifecycleStatus.RUNNING
    assert [item.decision for item in approvals] == ["reject"]
    assert agent.invocations == 2


def test_shell_budget_counts_repository_approvals_before_resume(config) -> None:
    limited = replace(config, max_shell_calls=1)
    agent = FakeAgent(approval_interrupt(), approval_interrupt())
    service = service_for(limited, agent)
    waiting = service.start("不要超过 shell 预算")

    paused = service.decide(waiting.task_id, ["approve"])

    assert paused.lifecycle is TaskLifecycleStatus.PAUSED
    assert paused.pause_reason == "已达到 Shell 最大执行次数"
    assert agent.invocations == 2


def test_repeated_identical_action_uses_distinct_interrupt_approval_identity(config) -> None:
    roomy = replace(config, max_shell_calls=3)
    agent = FakeAgent(
        approval_interrupt("approval-round-1"),
        approval_interrupt("approval-round-2"),
        no_response(),
    )
    service = service_for(roomy, agent)
    first = service.start("允许重复运行相同测试")

    second = service.decide(first.task_id, ["approve"])
    completed_resume = service.decide(second.task_id, ["approve"])

    approvals = service.repositories.execution.list_approvals(first.task_id)
    assert completed_resume.lifecycle is TaskLifecycleStatus.RUNNING
    assert len(approvals) == 2
    assert len({item.approval_id for item in approvals}) == 2
    assert [item.decision for item in approvals] == ["approve", "approve"]


def test_unresolved_started_operation_blocks_next_agent_invocation(config) -> None:
    agent = FakeAgent(no_response(), no_response())
    service = service_for(config, agent)
    task = service.start("恢复未知副作用")
    operation = service.repositories.execution.prepare(
        NewOperationEntry(
            operation_id="operation-unknown",
            task_id=task.task_id,
            experiment_id=f"legacy-{task.task_id}",
            tool_call_id="execute-unknown",
            operation_kind=OperationKind.COMMAND,
            call_hash="a" * 64,
            workspace_baseline_id="legacy",
            pre_state=OperationStateSnapshot(command_hash="b" * 64),
        )
    )
    service.repositories.execution.mark_started(operation.operation_id)
    service.operation_reconciler = OperationReconciler(
        service.repositories.execution,
        ToolResultArtifactStorage(config.artifacts_path / "investigation_receipts"),
    )

    paused = service.continue_task(task.task_id, "继续")

    assert paused.lifecycle is TaskLifecycleStatus.PAUSED
    assert "副作用操作需要人工恢复" in paused.pause_reason
    assert agent.invocations == 1


def test_foreign_investigation_recovery_fails_closed(config) -> None:
    recovery = InvestigationRecoveryMetadata(
        task_id="other-task",
        error_code="investigation_event_commit_failed",
        state_version=1,
        last_event_sequence=0,
        checkpoint_available=True,
        recovery_action="pause",
    )
    service = service_for(config, FakeAgent(InvestigationStateError(recovery)))

    task = service.start("拒绝其他任务的恢复元数据")

    assert task.lifecycle is TaskLifecycleStatus.FAILED
    assert task.pause_reason == "Agent 返回了其他任务的调查恢复信息"


def test_context_recovery_pauses_at_service_boundary(config) -> None:
    recovery = ContextRecoveryMetadata(
        task_id="placeholder",
        stage="overflow_retry",
        error_code="context_overflow_retry_exhausted",
        original_messages_preserved=True,
    )
    agent = FakeAgent()
    service = service_for(config, agent)

    # Bind the generated task id by raising from the first invocation.
    def invoke(_value, graph_config):
        task_id = graph_config["configurable"]["thread_id"]
        raise ContextRecoveryRequired(recovery.model_copy(update={"task_id": task_id}))

    agent.invoke = invoke

    task = service.start("上下文恢复应暂停")

    assert task.lifecycle is TaskLifecycleStatus.PAUSED
    assert task.pause_reason == "上下文协调需要恢复：context_overflow_retry_exhausted"


def test_runtime_does_not_expose_mutable_legacy_phase(config) -> None:
    service = service_for(config, FakeAgent(no_response()))
    task = service.start("修复失败测试")
    before = service.repository.get_lifecycle(task.task_id)
    runtime = service.get_runtime(task.task_id)

    after = service.repository.get_lifecycle(task.task_id)
    assert "status" not in type(runtime).model_fields
    assert after.status is TaskLifecycleStatus.RUNNING
    assert after.version == before.version


def test_completed_outcome_records_adjudication_with_evidence_ids(config) -> None:
    from deepfix.workspace import compute_code_state_hash

    service = service_for(config, FakeAgent(no_response()), workspace=True)
    task = service.start("验证当前实现，运行 python -m pytest -q")
    definition = service.repository.get_definition(task.task_id)
    service.repositories.evidence.record_deterministic(
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
            workspace_baseline_id=definition.workspace_baseline_id,
            code_state_hash=compute_code_state_hash(definition.workspace_root),
        ),
        provenance_root_ids=["test-call"],
    )
    service.agent = FakeAgent(
        {
            "structured_response": RepairOutcomeCandidate(
                status="completed",
                summary="当前实现测试通过",
            ),
            "messages": [],
        }
    )
    completed = service.continue_task(task.task_id, "请根据已有证据裁决")

    decision = service.repository.latest_adjudication(task.task_id)
    assert completed.lifecycle is TaskLifecycleStatus.COMPLETED
    assert decision is not None
    assert decision.outcome == "not_reproduced"
    assert decision.evidence_ids == ["test-pass"]
    assert decision.operation_ids == []


def test_related_repository_suite_failure_blocks_fixed_claim(config) -> None:
    command = "python -m pytest tests/test_value.py -q"
    service = service_for(config, FakeAgent(no_response()), workspace=True)
    task = service.start(f"修复失败并运行 {command}")
    for evidence in (
        SystemTestEvidence(
            evidence_id="targeted-pass",
            command=command,
            exit_code=0,
            summary="1 passed",
            tool_call_id="targeted-call",
            source_message_id="targeted-result",
            origin="user_specified",
            scope="targeted",
            timing="post_change",
            workspace_baseline_id="baseline-1",
            code_state_hash="code-state-2",
        ),
        SystemTestEvidence(
            evidence_id="suite-fail",
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed, 20 passed",
            tool_call_id="suite-call",
            source_message_id="suite-result",
            origin="repository_existing",
            scope="full_suite",
            timing="post_change",
            workspace_baseline_id="baseline-1",
            code_state_hash="code-state-2",
        ),
    ):
        service.repositories.evidence.record_deterministic(
            task.task_id,
            evidence,
            provenance_root_ids=[evidence.tool_call_id],
        )
    service.agent = FakeAgent(
        {
            "structured_response": RepairOutcomeCandidate(
                status="completed",
                summary="目标测试通过",
            ),
            "messages": [],
        }
    )

    result = service.continue_task(task.task_id, "裁决修复结果")

    assert result.lifecycle is TaskLifecycleStatus.PAUSED
    assert service.repository.latest_adjudication(task.task_id) is None
