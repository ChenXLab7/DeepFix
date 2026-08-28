import json
from collections import deque
from dataclasses import asdict, replace
from uuid import uuid4

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, Interrupt

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    ContextRecoveryRequired,
    ProtectedContextLoadError,
)
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import (
    CompactionFailureRecord,
    ContextRecoveryMetadata,
    FileChangeEvidence,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode, load_config
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import (
    InvestigationStagnationError,
    InvestigationStateError,
)
from deepfix.investigation.models import (
    AgentPhase,
    InvestigationRecoveryMetadata,
)
from deepfix.investigation.receipts import (
    ToolExecutionReceiptStore,
    receipt_from_result,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore
from deepfix.models import Evidence, RepairOutcome, TaskState, TaskStatus
from deepfix.models import TestResult as RepairTestResult
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalStore,
    OperationKind,
    OperationReconciler,
    OperationStateSnapshot,
)
from deepfix.persistence import TaskRepository
from deepfix.reporting import render_report
from deepfix.research.models import ExternalEvidence, SearchCandidate
from deepfix.research.store import ResearchEvidenceStore
from deepfix.service import BugfixService
from deepfix.verification import VerificationPolicyBuilder, VerificationPolicyStore
from deepfix.workspace import WorkspaceFactory


class FakeAgent:
    def __init__(self, *results):
        self.results = deque(results)
        self.invoke_calls = []

    def invoke(self, value, config):
        self.invoke_calls.append((value, config))
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def test_service_passes_single_run_graph_step_limit(app_config):
    fake_agent = FakeAgent(outcome())
    service, _ = make_service(app_config, fake_agent)

    service.start("prevent internal tool loops")

    _, graph_config = fake_agent.invoke_calls[0]
    assert graph_config["recursion_limit"] == app_config.max_graph_steps


def test_graph_step_limit_pauses_instead_of_failing(app_config):
    service, repository = make_service(
        app_config,
        FakeAgent(GraphRecursionError("recursion limit reached")),
    )

    task = service.start("prevent internal tool loops")

    assert task.status is TaskStatus.PAUSED
    assert "单次 Agent 执行步骤上限" in task.pause_reason
    assert repository.get(task.task_id).status is TaskStatus.PAUSED


def make_interrupt(name: str, args: dict[str, object]):
    return {
        "__interrupt__": (
            Interrupt(
                value={
                    "action_requests": [
                        {"name": name, "args": args, "description": "待审批操作"}
                    ],
                    "review_configs": [
                        {
                            "action_name": name,
                            "allowed_decisions": ["approve", "reject"],
                        }
                    ],
                },
                id="interrupt-1",
            ),
        )
    }


def outcome(status="needs_input", **overrides):
    values = {
        "status": status,
        "question": "请补充失败堆栈" if status == "needs_input" else None,
        "summary": "等待补充" if status == "needs_input" else "处理完成",
    }
    values.update(overrides)
    return {"structured_response": RepairOutcome(**values), "messages": []}


def passing_outcome(command="pytest -q"):
    return {
        "structured_response": RepairOutcome(
            status="completed",
            diagnosis="边界条件错误",
            hypotheses=["计算分支选择错误"],
            repair_plan=["修正条件判断"],
            summary="缺陷已修复",
            review_summary="修改范围最小",
        ),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": command},
                        "id": "test-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                id="test-message-1",
                content="1 passed",
                name="execute",
                tool_call_id="test-call-1",
                artifact={"exit_code": 0},
            ),
        ],
    }


def test_recorded_test_result_keeps_tool_and_message_provenance(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "修复错误", ApprovalMode.MANUAL)

    service._record_tool_results(task, passing_outcome())
    service._record_tool_results(task, passing_outcome())

    assert len(task.test_results) == 1
    assert task.test_results[0].tool_call_id == "test-call-1"
    assert task.test_results[0].source_message_id == "test-message-1"


def test_pre_edit_pytest_failure_does_not_consume_repair_failure_budget(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "复现错误", ApprovalMode.MANUAL)
    failed = passing_outcome()
    failed["messages"][-1].artifact = {"exit_code": 1}
    failed["messages"][-1].content = "1 failed"

    service._record_tool_results(task, failed)

    assert task.test_results[0].exit_code == 1
    assert task.consecutive_test_failures == 0


def test_pause_syncs_successful_change_as_pending_verification(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "修复查找错误", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)
    service.compaction_store.save_evidence(
        task.task_id,
        FileChangeEvidence(
            evidence_id="file-change-1",
            path="python_programs/find_first_in_sorted.py",
            operation="edit",
            status="succeeded",
            tool_call_id="edit-call-1",
            source_message_id="edit-message-1",
        ),
    )

    paused = service._pause(task, "调查协调需要恢复：investigation_state_commit_failed")

    assert paused.successful_changed_files == [
        "python_programs/find_first_in_sorted.py"
    ]
    assert paused.latest_change_verification == "pending"
    assert paused.pause_reason == (
        "调查协调需要恢复：investigation_state_commit_failed"
    )


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    return load_config(tmp_path, ApprovalMode.MANUAL)


@pytest.fixture
def memory_store(app_config):
    return WorkingMemoryStore(app_config.database_path)


def make_service(
    config,
    fake_agent,
    memory_store=None,
    research_store=None,
    investigation=None,
    operation_reconciler=None,
    workspace_factory=None,
    verification_policy_store=None,
    execution_backend=None,
):
    repository = TaskRepository(config.database_path)
    memory_store = memory_store or WorkingMemoryStore(config.database_path)
    research_store = research_store or ResearchEvidenceStore(config.database_path)
    return (
        BugfixService(
            fake_agent,
            repository,
            ApprovalPolicy(config.approval_mode),
            config,
            memory_store,
            research_store,
            investigation=investigation,
            operation_reconciler=operation_reconciler,
            workspace_factory=workspace_factory,
            verification_policy_store=verification_policy_store,
            execution_backend=execution_backend,
        ),
        repository,
    )


def test_service_creates_workspace_and_policy_before_agent_invocation(app_config):
    fake_agent = FakeAgent(outcome())
    workspace_factory = WorkspaceFactory(
        app_config.database_path.parent / "workspaces"
    )
    policy_store = VerificationPolicyStore(app_config.database_path)
    service, _ = make_service(
        app_config,
        fake_agent,
        workspace_factory=workspace_factory,
        verification_policy_store=policy_store,
    )

    task = service.start("运行 python -m pytest -q 并修复失败")

    assert task.source_project_root == str(app_config.project_root)
    assert task.workspace_root != str(app_config.project_root)
    assert task.project_root == task.workspace_root
    assert task.workspace_baseline_id
    assert task.verification_policy_id
    assert task.verification_policy_version == 1
    _, graph_config = fake_agent.invoke_calls[0]
    assert graph_config["configurable"]["workspace_root"] == task.workspace_root
    assert policy_store.load(task.task_id).policy_id == task.verification_policy_id


def test_repository_suite_failure_blocks_fixed_at_service_boundary(app_config):
    test = app_config.project_root / "tests" / "test_value.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("def test_value(): assert True\n", encoding="utf-8")
    task = TaskState.create(
        app_config.project_root,
        "运行 python -m pytest tests/test_value.py -q 并修复",
        app_config.approval_mode,
    )
    workspace = WorkspaceFactory(
        app_config.database_path.parent / "policy-workspaces"
    ).create(task.task_id, app_config.project_root)
    task.project_root = str(workspace.root)
    task.workspace_root = str(workspace.root)
    task.workspace_baseline_id = workspace.baseline.baseline_id
    task.successful_changed_files = ["value.py"]
    task.changed_files = ["value.py"]
    task.test_results = [
        RepairTestResult("baseline", 1, "failed"),
        RepairTestResult("targeted", 0, "passed"),
    ]
    task.transition_to(TaskStatus.INVESTIGATING)
    policy = VerificationPolicyBuilder().build(task, workspace)
    policy_store = VerificationPolicyStore(app_config.database_path)
    policy_store.save(policy)
    task.verification_policy_id = policy.policy_id
    task.verification_policy_version = policy.version
    compaction = CompactionStore(app_config.database_path)
    for evidence in (
        FileChangeEvidence(
            evidence_id="file-change",
            path="value.py",
            operation="edit",
            status="succeeded",
            tool_call_id="edit-value",
            source_message_id="edit-result",
        ),
        SystemTestEvidence(
            evidence_id="targeted-pass",
            command="python -m pytest tests/test_value.py -q",
            exit_code=0,
            summary="1 passed",
            tool_call_id="targeted",
            source_message_id="targeted-result",
            origin="user_specified",
            scope="targeted",
            timing="post_change",
            workspace_baseline_id=workspace.baseline.baseline_id,
            code_state_hash="code-a",
        ),
        SystemTestEvidence(
            evidence_id="suite-failure",
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="suite",
            source_message_id="suite-result",
            origin="repository_existing",
            scope="full_suite",
            timing="post_change",
            workspace_baseline_id=workspace.baseline.baseline_id,
            code_state_hash="code-a",
        ),
    ):
        compaction.save_evidence(task.task_id, evidence)
    service, repository = make_service(
        app_config,
        FakeAgent(passing_outcome()),
        verification_policy_store=policy_store,
    )
    service.compaction_store = compaction
    repository.save(task)

    result = service._invoke(task, {"messages": []})

    assert result.status is TaskStatus.PAUSED
    assert "required verification oracle" in result.pause_reason


def test_unresolved_operation_pauses_before_agent_invocation(app_config):
    fake_agent = FakeAgent(outcome())
    journal = OperationJournalStore(app_config.database_path)
    receipts = ToolExecutionReceiptStore(
        app_config.artifacts_path / "investigation_receipts"
    )
    reconciler = OperationReconciler(journal, receipts)
    service, repository = make_service(
        app_config,
        fake_agent,
        operation_reconciler=reconciler,
    )
    task = TaskState.create(
        app_config.project_root,
        "recover interrupted edit",
        app_config.approval_mode,
        workspace_baseline_id="baseline-1",
    )
    task.transition_to(TaskStatus.INVESTIGATING)
    repository.save(task)
    target = app_config.project_root / "value.py"
    target.write_text("third-party\n", encoding="utf-8")
    entry = NewOperationEntry(
        operation_id="operation-conflict",
        task_id=task.task_id,
        experiment_id="legacy-task",
        tool_call_id="edit-conflict",
        operation_kind=OperationKind.FILE_EDIT,
        call_hash="a" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(
            target_path="value.py", target_exists=True, file_hash="b" * 64
        ),
        expected_post_state=OperationStateSnapshot(
            target_path="value.py", target_exists=True, file_hash="c" * 64
        ),
    )
    journal.prepare(entry)
    journal.mark_started(entry.operation_id)

    result = service._invoke(task, {"messages": []})

    assert result.status is TaskStatus.PAUSED
    assert "operation-conflict" in result.pause_reason
    assert fake_agent.invoke_calls == []


def service_coordinator(config):
    compaction = CompactionStore(config.database_path)
    research = ResearchEvidenceStore(config.database_path)
    return InvestigationCoordinator(
        store=InvestigationStore(config.database_path),
        tasks=TaskRepository(config.database_path),
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, research),
    )


def save_external_evidence(store, task_id, *, excerpt="EXTERNAL ONLY"):
    candidate = store.save_candidates(
        task_id,
        "pydantic model_copy",
        [
            SearchCandidate(
                candidate_id="draft",
                task_id="draft",
                source_type="official_docs",
                evidence_level="E1",
                title="Pydantic models",
                url="https://docs.pydantic.dev/models/",
                query="draft",
                repository="pydantic/pydantic",
                created_at="draft",
            )
        ],
    )[0]
    evidence = ExternalEvidence(
        evidence_id=uuid4().hex,
        task_id=task_id,
        candidate_id=candidate.candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title="Pydantic models",
        url=candidate.url,
        query=candidate.query,
        relevant_excerpt=excerpt,
        retrieved_at="2026-08-22T00:00:00+00:00",
        dependency_name="pydantic",
        documented_version="2.8",
        project_version="2.8.4",
        local_verification="unverified",
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
        artifact_path=f"/.deepfix-artifacts/research/{task_id}/evidence.md",
    )
    store.save_evidence(evidence)
    return evidence


def test_start_passes_problem_and_stable_thread_id(app_config):
    fake_agent = FakeAgent(outcome())
    service, _ = make_service(app_config, fake_agent)

    task = service.start("除法结果错误")

    value, config = fake_agent.invoke_calls[0]
    graph_message = value["messages"][0]
    assert isinstance(graph_message, HumanMessage)
    assert graph_message.content == "除法结果错误"
    assert graph_message.id == task.conversation[0]["id"]
    assert task.conversation[0] == {
        "id": graph_message.id,
        "role": "user",
        "content": "除法结果错误",
    }
    assert config["configurable"]["thread_id"] == task.task_id
    assert task.agent_invocations == 1
    assert task.project_python == str(app_config.project_python)


def test_service_syncs_only_current_task_research_summary_on_every_save(app_config):
    research_store = ResearchEvidenceStore(app_config.database_path)
    fake_agent = FakeAgent(outcome(), outcome(question="继续"))
    service, repository = make_service(
        app_config,
        fake_agent,
        research_store=research_store,
    )
    task = service.start("模型复制失败")
    current = save_external_evidence(research_store, task.task_id)
    save_external_evidence(research_store, "other-task", excerpt="OTHER TASK SECRET")
    for index in range(12):
        research_store.save_query(
            task.task_id,
            f"query-{index}",
            ["github"],
            [f"provider-{index}-" + "x" * 400],
        )

    continued = service.continue_task(task.task_id, "继续")
    persisted = repository.get(task.task_id)

    assert continued.external_evidence_ids == [current.evidence_id]
    assert persisted.external_evidence_ids == [current.evidence_id]
    assert continued.research_query_count == 12
    assert len(continued.research_provider_errors) == 10
    assert all(len(error) <= 300 for error in continued.research_provider_errors)
    assert all("OTHER TASK SECRET" not in error for error in continued.research_provider_errors)


def test_external_excerpt_never_enters_local_task_evidence(app_config):
    research_store = ResearchEvidenceStore(app_config.database_path)
    fake_agent = FakeAgent(outcome(), outcome(question="继续"))
    service, _ = make_service(
        app_config,
        fake_agent,
        research_store=research_store,
    )
    task = service.start("模型复制失败")
    save_external_evidence(
        research_store,
        task.task_id,
        excerpt="EXTERNAL EXCERPT MUST STAY OUT",
    )

    continued = service.continue_task(task.task_id, "继续")

    assert all(
        "EXTERNAL EXCERPT MUST STAY OUT" not in item.observation
        for item in continued.evidence
    )


def test_provider_failure_is_synchronized_without_failing_task(app_config):
    research_store = ResearchEvidenceStore(app_config.database_path)
    fake_agent = FakeAgent(outcome(), outcome(question="继续"))
    service, _ = make_service(
        app_config,
        fake_agent,
        research_store=research_store,
    )
    task = service.start("依赖行为未知")
    research_store.save_query(
        task.task_id,
        "pydantic validation",
        ["github"],
        ["github: rate limited"],
    )

    continued = service.continue_task(task.task_id, "继续")

    assert continued.status is TaskStatus.CLARIFYING
    assert continued.research_provider_errors == ["github: rate limited"]


def test_interrupt_is_persisted_as_waiting_approval(app_config):
    fake_agent = FakeAgent(make_interrupt("execute", {"command": "pytest -q"}))
    service, repository = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.WAITING_APPROVAL
    assert task.shell_calls == 1
    assert service.pending_actions(task.task_id)[0]["name"] == "execute"
    assert repository.get(task.task_id).pending_actions == task.pending_actions


def test_user_can_pause_and_resume_same_pending_approval(app_config):
    fake_agent = FakeAgent(make_interrupt("execute", {"command": "pytest -q"}))
    service, repository = make_service(app_config, fake_agent)
    task = service.start("测试失败")

    paused = service.pause_task(task.task_id, "用户从终端暂停")
    resumed = service.continue_task(task.task_id)

    assert paused.task_id == task.task_id
    assert paused.status is TaskStatus.PAUSED
    assert paused.pending_actions == task.pending_actions
    assert resumed.status is TaskStatus.WAITING_APPROVAL
    assert resumed.pending_actions == task.pending_actions
    assert repository.get(task.task_id).status is TaskStatus.WAITING_APPROVAL


def test_l3_operation_is_rejected_even_when_user_requests_approval(app_config):
    fake_agent = FakeAgent(
        make_interrupt("execute", {"command": "git reset --hard"}),
        outcome(),
    )
    service, _ = make_service(app_config, fake_agent)
    task = service.start("清理项目")

    resumed = service.decide(task.task_id, ["approve"])

    resume_value, _ = fake_agent.invoke_calls[-1]
    assert isinstance(resume_value, Command)
    assert resume_value.resume["decisions"][0]["type"] == "reject"
    assert resumed.approvals[-1].decision == "reject"


def test_guarded_l1_operation_is_automatically_approved(app_config):
    guarded = replace(app_config, approval_mode=ApprovalMode.GUARDED)
    fake_agent = FakeAgent(
        make_interrupt("execute", {"command": "pytest -q"}),
        passing_outcome(),
    )
    service, _ = make_service(guarded, fake_agent)

    task = service.start("测试失败")

    resume_value, _ = fake_agent.invoke_calls[1]
    assert resume_value.resume["decisions"] == [{"type": "approve"}]
    assert task.status is TaskStatus.COMPLETED


def test_agent_exception_is_saved_as_failed(app_config):
    fake_agent = FakeAgent(RuntimeError("model unavailable"))
    service, repository = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.FAILED
    assert "model unavailable" in task.final_summary
    assert repository.get(task.task_id).status is TaskStatus.FAILED


def test_agent_failure_redacts_both_model_role_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret-value")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret-value")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(
        tmp_path,
        ApprovalMode.MANUAL,
        env_file=tmp_path / "missing.env",
    )
    service, repository = make_service(
        config,
        FakeAgent(RuntimeError("auth failed main-secret-value compact-secret-value")),
    )

    task = service.start("修复错误")
    persisted = repository.get(task.task_id)

    assert task.status is TaskStatus.FAILED
    assert "main-secret-value" not in persisted.final_summary
    assert "compact-secret-value" not in persisted.final_summary
    assert (
        persisted.final_summary
        == "Agent 执行失败: auth failed [REDACTED] [REDACTED]"
    )


def test_shell_budget_pauses_before_command_is_resumed(app_config):
    limited = replace(app_config, max_shell_calls=0)
    fake_agent = FakeAgent(make_interrupt("execute", {"command": "pytest -q"}))
    service, _ = make_service(limited, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert len(fake_agent.invoke_calls) == 1
    assert task.pending_actions


def test_agent_invocation_budget_pauses_before_calling_agent(app_config):
    limited = replace(app_config, max_agent_invocations=0)
    fake_agent = FakeAgent()
    service, _ = make_service(limited, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert fake_agent.invoke_calls == []


def test_changed_file_budget_pauses_before_approved_edit_is_resumed(app_config):
    limited = replace(app_config, max_changed_files=0)
    fake_agent = FakeAgent(
        make_interrupt(
            "edit_file",
            {"file_path": "/src/calc.py", "old_string": "x", "new_string": "y"},
        )
    )
    service, _ = make_service(limited, fake_agent)
    task = service.start("计算错误")

    paused = service.decide(task.task_id, ["approve"])

    assert paused.status is TaskStatus.PAUSED
    assert len(fake_agent.invoke_calls) == 1
    assert paused.pending_actions


def test_consecutive_test_failure_budget_pauses_task(app_config):
    limited = replace(app_config, max_consecutive_test_failures=1)
    failed_result = passing_outcome()
    failed_result["structured_response"] = outcome()["structured_response"]
    failed_result["messages"][-1].artifact = {"exit_code": 1}
    failed_result["messages"][-1].content = "1 failed"
    fake_agent = FakeAgent(
        make_interrupt(
            "edit_file",
            {"file_path": "/src/calc.py", "old_string": "x", "new_string": "y"},
        ),
        failed_result,
    )
    service, _ = make_service(limited, fake_agent)

    awaiting_approval = service.start("测试失败")
    task = service.decide(awaiting_approval.task_id, ["approve"])

    assert task.status is TaskStatus.PAUSED
    assert task.consecutive_test_failures == 1
    assert "连续测试失败" in task.final_summary


def test_rejected_action_records_approval_and_resumes_agent(app_config):
    fake_agent = FakeAgent(
        make_interrupt("write_file", {"file_path": "/src/calc.py", "content": "x"}),
        outcome(),
    )
    service, repository = make_service(app_config, fake_agent)
    task = service.start("计算错误")

    result = service.decide(task.task_id, ["reject"])

    assert result.approvals[-1].operation == "write_file"
    assert result.approvals[-1].decision == "reject"
    assert repository.get(task.task_id).approvals[-1].decision == "reject"


def test_continue_clarifying_task_appends_message_without_new_task(app_config):
    fake_agent = FakeAgent(outcome(), outcome(question="请提供 Python 版本"))
    service, _ = make_service(app_config, fake_agent)
    task = service.start("测试失败")

    continued = service.continue_task(task.task_id, "Python 3.12")

    value, config = fake_agent.invoke_calls[-1]
    assert continued.task_id == task.task_id
    graph_message = value["messages"][0]
    assert isinstance(graph_message, HumanMessage)
    assert graph_message.content == "Python 3.12"
    assert graph_message.id == continued.conversation[-1]["id"]
    assert config["configurable"]["thread_id"] == task.task_id
    assert continued.conversation[-1]["content"] == "Python 3.12"


def test_completed_outcome_requires_and_records_passing_test(app_config):
    fake_agent = FakeAgent(passing_outcome())
    service, _ = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.COMPLETED
    assert task.test_results[0].command == "pytest -q"
    assert task.test_results[0].exit_code == 0
    assert task.resolution == "not_reproduced"
    assert task.final_summary == (
        "未复现用户描述的问题：当前环境中所运行的 pytest 测试通过，且未修改代码。"
    )


def test_offline_report_path_keeps_todo_navigation_in_graph_state_only(app_config):
    todos = [
        {"content": "reproduce failure", "status": "completed"},
        {"content": "identify root cause", "status": "in_progress"},
        {"content": "apply minimal fix", "status": "pending"},
        {"content": "run required verification", "status": "pending"},
    ]
    private_navigation_state = {
        "_deepfix_todo_rounds_since_update": 2,
        "_deepfix_last_completed_tool_round_id": "verification-round",
        "_deepfix_last_todo_progress_fingerprint": "todo-fingerprint",
        "_deepfix_last_navigation_hint_fingerprint": "oracle-milestone",
        "_deepfix_pending_navigation_reminder": "request-local reminder",
    }
    graph_result = passing_outcome()
    graph_result.update({"todos": todos, **private_navigation_state})
    graph_result["messages"] = [
        message.model_copy(
            update={
                "additional_kwargs": {
                    **message.additional_kwargs,
                    "todos": todos,
                    **private_navigation_state,
                }
            },
            deep=True,
        )
        for message in graph_result["messages"]
    ]
    navigation_bearing_messages = json.dumps(
        [message.model_dump(mode="json") for message in graph_result["messages"]],
        ensure_ascii=False,
    )
    service, _ = make_service(app_config, FakeAgent(graph_result))

    task = service.start("测试失败")
    deterministic_evidence = EvidenceCollector(
        service.compaction_store,
        service.research_evidence_store,
    ).collect(task.task_id, graph_result["messages"], task)
    task_payload = task.to_dict()
    boundary_texts = [
        json.dumps(task_payload, ensure_ascii=False, default=str),
        deterministic_evidence.model_dump_json(),
        json.dumps(asdict(task.context_metrics), ensure_ascii=False),
        render_report(task),
    ]

    assert graph_result["todos"] == todos
    assert "todos" not in task_payload
    assert "todos" in navigation_bearing_messages
    for forbidden in private_navigation_state:
        assert forbidden in navigation_bearing_messages
        assert all(forbidden not in text for text in boundary_texts)
    for todo in todos:
        assert todo["content"] in navigation_bearing_messages
        assert all(todo["content"] not in text for text in boundary_texts)


def test_not_reproduced_is_rejected_when_current_task_has_failed_test(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "测试失败", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)
    task.test_results.extend(
        [
            RepairTestResult("pytest -q", 1, "1 failed"),
            RepairTestResult("pytest -q", 0, "1 passed"),
        ]
    )

    result = service._apply_outcome(
        task,
        RepairOutcome(
            status="completed",
            resolution="not_reproduced",
            summary="没有复现",
        ),
    )

    assert result.status is TaskStatus.PAUSED
    assert result.resolution is None
    assert "失败测试证据" in result.final_summary


def test_not_reproduced_is_rejected_after_successful_file_change(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "测试失败", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)
    task.test_results.append(RepairTestResult("pytest -q", 0, "1 passed"))
    service.compaction_store.save_evidence(
        task.task_id,
        FileChangeEvidence(
            evidence_id="file-change-1",
            path="python_programs/example.py",
            operation="edit",
            status="succeeded",
            tool_call_id="edit-call-1",
            source_message_id="edit-message-1",
        ),
    )

    result = service._apply_outcome(
        task,
        RepairOutcome(
            status="completed",
            resolution="not_reproduced",
            summary="没有复现",
        ),
    )

    assert result.status is TaskStatus.PAUSED
    assert result.resolution is None
    assert "已经成功修改文件" in result.final_summary


def test_fixed_is_rejected_without_reproduced_failure(app_config):
    service, _ = make_service(app_config, FakeAgent())
    task = TaskState.create(app_config.project_root, "测试失败", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)
    task.test_results.append(RepairTestResult("pytest -q", 0, "1 passed"))
    service.compaction_store.save_evidence(
        task.task_id,
        FileChangeEvidence(
            evidence_id="file-change-1",
            path="python_programs/example.py",
            operation="edit",
            status="succeeded",
            tool_call_id="edit-call-1",
            source_message_id="edit-message-1",
        ),
    )

    result = service._apply_outcome(
        task,
        RepairOutcome(
            status="completed",
            resolution="fixed",
            summary="已经修复",
        ),
    )

    assert result.status is TaskStatus.PAUSED
    assert result.resolution is None
    assert "缺少修复前的失败测试证据" in result.final_summary


def test_completed_outcome_without_passing_test_is_paused(app_config):
    fake_agent = FakeAgent(outcome(status="completed", summary="声称已完成"))
    service, _ = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert "通过的测试证据" in task.final_summary


def test_repeated_graph_message_history_does_not_duplicate_test_result(app_config):
    first = passing_outcome()
    first["structured_response"] = outcome()["structured_response"]
    second = passing_outcome()
    second["structured_response"] = outcome(question="还需要版本信息")[
        "structured_response"
    ]
    fake_agent = FakeAgent(first, second)
    service, _ = make_service(app_config, fake_agent)
    task = service.start("测试失败")

    continued = service.continue_task(task.task_id, "Python 3.12")

    assert len(continued.test_results) == 1


def test_service_syncs_latest_memory_without_promoting_hypothesis_to_diagnosis(
    app_config,
    memory_store,
):
    fake_agent = FakeAgent(outcome(), outcome(question="请继续提供信息"))
    service, _ = make_service(app_config, fake_agent, memory_store)
    task = service.start("测试失败")
    memory_store.save(
        task.task_id,
        ProgressSnapshot(
            phase="investigating",
            summary="最新调查",
            facts=["失败可稳定复现"],
            evidence=[
                {
                    "source": "tests/test_calc.py:18",
                    "observation": "期望 2，实际 -2",
                }
            ],
            active_hypotheses=["符号处理重复取反"],
            rejected_hypotheses=[],
            checked_files=["src/calc.py"],
            experiments=["目标单测退出码为 1"],
            next_steps=["验证 normalize_sign"],
            unresolved_questions=[],
        ),
    )

    continued = service.continue_task(task.task_id, "继续")

    assert continued.working_memory_version == 1
    assert Evidence("tests/test_calc.py:18", "期望 2，实际 -2") in continued.evidence
    assert "符号处理重复取反" in continued.hypotheses
    assert continued.diagnosis is None


def test_tool_message_wording_cannot_increment_compaction_metric(
    app_config,
    memory_store,
):
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "compact_conversation",
                    "args": {},
                    "id": "compact-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="对话已经安全压缩完成。",
            tool_call_id="compact-1",
        ),
    ]
    first = outcome()
    first["messages"] = messages
    second = outcome(question="请继续")
    second["messages"] = messages
    service, _ = make_service(
        app_config,
        FakeAgent(first, second),
        memory_store,
    )
    task = service.start("超长测试失败")

    continued = service.continue_task(task.task_id, "继续")

    assert continued.context_metrics.active_compaction_count == 0
    assert memory_store.metrics(task.task_id).active_compaction_count == 0


def test_nothing_to_compact_does_not_increment_success_metric(
    app_config,
    memory_store,
):
    result = outcome()
    result["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "compact_conversation",
                    "args": {},
                    "id": "compact-noop",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Nothing to compact yet — conversation is within the token budget.",
            tool_call_id="compact-noop",
        ),
    ]
    service, _ = make_service(
        app_config,
        FakeAgent(result),
        memory_store,
    )

    task = service.start("短任务")

    assert task.context_metrics.active_compaction_count == 0
    assert memory_store.metrics(task.task_id).active_compaction_count == 0


def test_uncoordinated_context_overflow_is_an_ordinary_agent_failure(
    app_config, memory_store
):
    service, repository = make_service(
        app_config,
        FakeAgent(ContextOverflowError("context exceeded")),
        memory_store,
    )

    task = service.start("超长任务")

    assert task.status is TaskStatus.FAILED
    assert task.context_metrics.context_overflow_count == 0
    assert repository.get(task.task_id).status is TaskStatus.FAILED


class RecoveryAgent:
    def __init__(self, error_type):
        self.error_type = error_type

    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        recovery = ContextRecoveryMetadata(
            task_id=task_id,
            stage="protected_context",
            error_code="protected_context_read_failed",
            original_messages_preserved=True,
        )
        raise self.error_type(recovery)


def investigation_recovery(task_id):
    return InvestigationRecoveryMetadata(
        task_id=task_id,
        error_code="investigation_stagnated",
        agent_phase=AgentPhase.DIAGNOSING,
        state_version=3,
        last_event_sequence=7,
        checkpoint_available=True,
        recovery_action="request_user_direction",
    )


class InvestigationErrorAgent:
    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        raise InvestigationStagnationError(investigation_recovery(task_id))


class ForeignInvestigationErrorAgent:
    def invoke(self, value, config):
        raise InvestigationStagnationError(investigation_recovery("other-task"))


class DiagnosticArtifactErrorAgent:
    def __init__(self, recovery_task_id: str | None = None):
        self.recovery_task_id = recovery_task_id

    def invoke(self, value, config):
        task_id = self.recovery_task_id or config["configurable"]["thread_id"]
        recovery = investigation_recovery(task_id).model_copy(
            update={
                "error_code": "diagnostic_artifact_backend_read_failed",
                "recovery_action": "pause_and_retry_diagnostic_artifact_read",
            }
        )
        raise InvestigationStateError(recovery)


class DetailedInvestigationErrorAgent:
    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        recovery = investigation_recovery(task_id).model_copy(
            update={
                "error_code": "tool_receipt_persistence_failed",
                "error_type": "OSError",
                "error_detail": "disk rejected secret",
                "error_fingerprint": "error_deadbeef",
                "recovery_action": "do_not_retry_tool_without_manual_recovery",
            }
        )
        raise InvestigationStateError(recovery)


def test_investigation_error_pauses_and_persists_metadata(app_config):
    coordinator = service_coordinator(app_config)
    service, repository = make_service(
        app_config,
        InvestigationErrorAgent(),
        investigation=coordinator,
    )

    task = service.start("repository scan loops")

    assert task.status is TaskStatus.PAUSED
    assert task.investigation_recovery.error_code == "investigation_stagnated"
    assert (
        repository.get(task.task_id).investigation_recovery
        == task.investigation_recovery
    )


def test_investigation_recovery_redacts_detail_and_writes_debug_event(app_config):
    service, repository = make_service(
        app_config,
        DetailedInvestigationErrorAgent(),
    )

    task = service.start("receipt persistence fails")
    persisted = repository.get(task.task_id)
    log_path = app_config.artifacts_path / "debug" / "llm_calls.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert persisted.investigation_recovery.error_detail == (
        "disk rejected [REDACTED]"
    )
    assert all("secret" not in line for line in log_path.read_text(encoding="utf-8").splitlines())
    assert records[-1] == {
        "event": "investigation_recovery",
        "task_id": task.task_id,
        "error_code": "tool_receipt_persistence_failed",
        "error_type": "OSError",
        "error_detail": "disk rejected [REDACTED]",
        "error_fingerprint": "error_deadbeef",
        "recovery_action": "do_not_retry_tool_without_manual_recovery",
    }


def test_foreign_investigation_recovery_task_id_fails_closed(app_config):
    service, _ = make_service(app_config, ForeignInvestigationErrorAgent())

    task = service.start("bug")

    assert task.status is TaskStatus.FAILED
    assert "其他任务" in task.final_summary


def test_diagnostic_artifact_system_error_pauses_only_at_service_boundary(
    app_config,
):
    service, repository = make_service(
        app_config,
        DiagnosticArtifactErrorAgent(),
    )

    task = service.start("recover archived traceback")

    assert task.status is TaskStatus.PAUSED
    assert task.investigation_recovery.error_code == (
        "diagnostic_artifact_backend_read_failed"
    )
    assert repository.get(task.task_id).status is TaskStatus.PAUSED


def test_foreign_diagnostic_artifact_recovery_fails_closed(app_config):
    service, repository = make_service(
        app_config,
        DiagnosticArtifactErrorAgent("other-task"),
    )

    task = service.start("recover archived traceback")

    assert task.status is TaskStatus.FAILED
    assert task.investigation_recovery is None
    assert "其他任务" in task.final_summary
    assert repository.get(task.task_id).status is TaskStatus.FAILED


def test_needs_input_synchronizes_agent_phase_to_clarifying(app_config):
    coordinator = service_coordinator(app_config)
    service, _ = make_service(
        app_config,
        FakeAgent(outcome(question="which Python version?")),
        investigation=coordinator,
    )

    task = service.start("version-sensitive bug")
    state = coordinator.state(task.task_id)

    assert task.status is TaskStatus.CLARIFYING
    assert state.agent_phase is AgentPhase.CLARIFYING
    assert state.paused_agent_phase is AgentPhase.INVESTIGATING


def test_user_information_records_lifecycle_and_clears_recovery(app_config):
    coordinator = service_coordinator(app_config)
    service, _ = make_service(
        app_config,
        FakeAgent(
            InvestigationStagnationError(investigation_recovery("placeholder")),
        ),
        investigation=coordinator,
    )
    task = TaskState.create(
        app_config.project_root,
        "bug",
        app_config.approval_mode,
        app_config.project_python,
    )
    task.status = TaskStatus.PAUSED
    task.paused_from = TaskStatus.INVESTIGATING
    task.investigation_recovery = investigation_recovery(task.task_id)
    service.repository.save(task)
    coordinator.ensure_started(task.task_id)
    service.agent = FakeAgent(outcome(question="continue"))

    resumed = service.continue_task(task.task_id, "Python 3.12")
    event_types = [
        event.event_type
        for event in coordinator.store.list_events(task.task_id)
    ]

    assert resumed.investigation_recovery is None
    assert "user_information_received" in event_types


@pytest.mark.parametrize(
    "error_type", [ProtectedContextLoadError, ContextRecoveryRequired]
)
def test_context_recovery_error_pauses_and_persists_metadata(
    app_config,
    error_type,
):
    service, repository = make_service(app_config, RecoveryAgent(error_type))

    task = service.start("long bug")

    assert task.status is TaskStatus.PAUSED
    assert task.context_recovery.error_code == "protected_context_read_failed"
    assert repository.get(task.task_id).context_recovery == task.context_recovery


def test_preparation_error_leak_remains_an_ordinary_failure(app_config):
    failure = CompactionFailureRecord(
        attempt_id="attempt-1",
        task_id="task-a",
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="artifact_write",
        error_code="artifact_write_failed",
        input_hash="f" * 64,
        original_messages_preserved=True,
        recorded_at="2026-08-22T00:00:00+00:00",
    )
    service, _ = make_service(
        app_config,
        FakeAgent(ArtifactPersistenceError(failure)),
    )

    assert service.start("bug").status is TaskStatus.FAILED


def test_recovery_metadata_clears_only_after_successful_resumed_invoke(app_config):
    service, _ = make_service(
        app_config,
        RecoveryAgent(ContextRecoveryRequired),
    )
    paused = service.start("long bug")
    assert paused.context_recovery is not None
    service.agent = FakeAgent(outcome(question="continue"))

    resumed = service.continue_task(paused.task_id, "more detail")

    assert resumed.context_recovery is None


def test_service_records_only_current_task_artifacts(app_config, memory_store):
    service, _ = make_service(app_config, FakeAgent(), memory_store)
    task = TaskState.create(app_config.project_root, "测试失败", ApprovalMode.MANUAL)
    current_history = (
        app_config.artifacts_path / "conversation_history" / f"{task.task_id}.md"
    )
    foreign_history = (
        app_config.artifacts_path / "conversation_history" / "other-task.md"
    )
    current_research = (
        app_config.artifacts_path / "research" / task.task_id / "evidence.md"
    )
    foreign_research = (
        app_config.artifacts_path / "research" / "other-task" / "evidence.md"
    )
    for path in (
        current_history,
        foreign_history,
        current_research,
        foreign_research,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    receipt_store = ToolExecutionReceiptStore(
        app_config.artifacts_path / "investigation_receipts"
    )
    current_receipt = receipt_from_result(
        task.task_id,
        {
            "name": "read_file",
            "id": "current-read",
            "args": {"file_path": "src/current.py"},
        },
        ToolMessage(
            id="current-result",
            content="current",
            name="read_file",
            tool_call_id="current-read",
        ),
    )
    foreign_receipt = receipt_from_result(
        "other-task",
        {
            "name": "read_file",
            "id": "foreign-read",
            "args": {"file_path": "src/foreign.py"},
        },
        ToolMessage(
            id="foreign-result",
            content="foreign",
            name="read_file",
            tool_call_id="foreign-read",
        ),
    )
    receipt_store.save(current_receipt)
    receipt_store.save(foreign_receipt)
    current_receipt_path = next(
        path
        for path in (app_config.artifacts_path / "investigation_receipts").rglob(
            "*.json"
        )
        if task.task_id in path.read_text(encoding="utf-8")
    ).relative_to(app_config.artifacts_path).as_posix()

    service._save(task)

    assert task.offloaded_artifacts == [
        f"conversation_history/{task.task_id}.md",
        current_receipt_path,
        f"research/{task.task_id}/evidence.md",
    ]
