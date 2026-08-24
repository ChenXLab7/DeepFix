from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend, LocalShellBackend
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime
from pydantic import SecretStr

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator, CompactionRequest
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.identity import stable_work_unit_id
from deepfix.compaction.middleware import DeepFixCompactionMiddleware
from deepfix.compaction.models import (
    CompactionDelta,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    FactCandidate,
    HypothesisProgressInput,
    ProvenanceRef,
    ResearchStatusEvidence,
    SnapshotCoverage,
    TaskAnchor,
    UserConstraint,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import partition_work_units
from deepfix.config import AppConfig, ApprovalMode, ModelRoleConfig
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import WorkingMemoryStore
from deepfix.models import ApprovalRecord, TaskState, TaskStatus
from deepfix.models import TestResult as RepairTestResult
from deepfix.persistence import TaskRepository
from deepfix.protected_context import ProtectedContext, render_protected_context
from deepfix.research.store import ResearchEvidenceStore
from deepfix.service import BugfixService


def _empty_delta() -> CompactionDelta:
    return CompactionDelta(
        user_constraint_candidates=[],
        confirmed_fact_candidates=[],
        hypothesis_transitions=[],
        experiments=[],
        conflict_candidates=[],
        unresolved_questions=[],
        next_steps=[],
    )


class _FakeDeltaModel:
    def generate(self, model, units):
        return _empty_delta()

    async def agenerate(self, model, units):
        return _empty_delta()


def _budget(zone: str, ratio: float) -> ContextBudgetReport:
    return ContextBudgetReport(
        request_tokens=int(ratio * 1000),
        max_input_tokens=1000,
        output_reserve_tokens=0,
        usage_ratio=ratio,
        zone=zone,
        target_ratio=0.75 if zone == "normal_compaction" else 0.65,
    )


def _call(name, call_id, args):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _scoped(task_id: str, messages):
    return [
        message.model_copy(
            update={
                "additional_kwargs": {
                    **message.additional_kwargs,
                    "_deepfix_task_id": task_id,
                }
            },
            deep=True,
        )
        for message in messages
    ]


@dataclass
class _WorkflowResult:
    task: TaskState
    original_constraint: str
    final_snapshot: object
    rejected_reason: str
    final_protected_context: str
    history_contains_all_compressed_message_ids: bool
    no_work_unit_was_split: bool
    target_project_has_no_deepfix_artifacts: bool


class _LongBugWorkflow:
    def __init__(self, tmp_path: Path) -> None:
        fixture = Path(__file__).parent / "fixtures" / "bug_project"
        self.project = tmp_path / "target-project"
        shutil.copytree(fixture, self.project)
        self.state = tmp_path / "state"

    def run(self) -> _WorkflowResult:
        database = self.state / "deepfix.sqlite3"
        tasks = TaskRepository(database)
        memory = WorkingMemoryStore(database)
        snapshots = CompactionStore(database)
        research = ResearchEvidenceStore(database)
        artifact_adapter = DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=self.state / "artifacts", virtual_mode=True)
        )
        coordinator = CompactionCoordinator(
            adapter=artifact_adapter,
            delta_generator=_FakeDeltaModel(),
            snapshot_builder=CompactionSnapshotBuilder(),
            snapshot_store=snapshots,
            memory_store=memory,
        )
        project_backend = LocalShellBackend(
            root_dir=self.project,
            virtual_mode=True,
            env={"PATH": str(Path(sys.executable).parent)},
            inherit_env=False,
        )
        command = f'"{sys.executable}" -m pytest -q'
        failed = project_backend.execute(command)
        assert failed.exit_code == 1

        constraint = "只修复符号归一化，不得修改公开 API"
        task = TaskState.create(self.project, constraint, ApprovalMode.MANUAL)
        task.transition_to(TaskStatus.INVESTIGATING)
        tasks.save(task)
        messages = [
            HumanMessage(id="m-user", content=constraint),
            AIMessage(
                id="m-read",
                content="并行读取实现和测试",
                tool_calls=[
                    _call("read_file", "read-src", {"file_path": "src/calculator.py"}),
                    _call("read_file", "read-test", {"file_path": "tests/test_calculator.py"}),
                ],
            ),
            ToolMessage(id="m-src", content="return abs on both branches", tool_call_id="read-src"),
            ToolMessage(id="m-test", content="negative sign must return -4", tool_call_id="read-test"),
            AIMessage(id="m-read-explain", content="失败分支丢失了负号"),
            AIMessage(
                id="m-fail-call",
                content="运行基线测试",
                tool_calls=[_call("execute", "pytest-fail", {"command": command})],
            ),
            ToolMessage(
                id="m-fail-result",
                content=failed.output,
                tool_call_id="pytest-fail",
                artifact={"exit_code": failed.exit_code},
            ),
            AIMessage(id="m-fail-explain", content="负号用例稳定失败"),
        ]
        read_unit_id = stable_work_unit_id(
            task.task_id, ["m-read", "m-src", "m-test", "m-read-explain"]
        )
        source = ProvenanceRef(kind="work_unit", ref_id=read_unit_id)
        first_memory = memory.save_progress(
            task.task_id,
            phase="investigating",
            summary="已定位符号分支",
            facts=[FactCandidate(text="负号用例稳定失败", sources=[source])],
            evidence=[],
            hypotheses=[
                HypothesisProgressInput(
                    text="缓存污染导致失败",
                    target_state="active",
                    sources=[source],
                )
            ],
            checked_files=["src/calculator.py", "tests/test_calculator.py"],
            experiments=["基线 pytest exit_code=1"],
            next_steps=["排除缓存后修复负号分支"],
            unresolved_questions=[],
            coverage=SnapshotCoverage(covered_work_unit_ids=[read_unit_id]),
            valid_source_ids={read_unit_id},
        )
        hypothesis_id = first_memory.snapshot.active_hypotheses[0].hypothesis_id
        rejected_reason = "全新进程仍失败，缓存假设被排除"
        memory.save_progress(
            task.task_id,
            phase="planning",
            summary="缓存已排除，准备最小修改",
            facts=[],
            evidence=[],
            hypotheses=[
                HypothesisProgressInput(
                    hypothesis_id=hypothesis_id,
                    text="缓存污染导致失败",
                    target_state="rejected",
                    reason=rejected_reason,
                    sources=[source],
                )
            ],
            checked_files=["src/calculator.py", "tests/test_calculator.py"],
            experiments=["新进程 pytest exit_code=1"],
            next_steps=["修改 else 分支"],
            unresolved_questions=[],
            coverage=SnapshotCoverage(covered_work_unit_ids=[read_unit_id]),
            valid_source_ids={read_unit_id},
        )
        snapshots.save_evidence(
            task.task_id,
            ResearchStatusEvidence(
                evidence_id="research-sign-semantics",
                verification="verified",
                artifact_path="/.deepfix-artifacts/research/sign.md",
            ),
        )
        collector = EvidenceCollector(snapshots, research)
        events = []
        active_snapshot = None

        def compact(current_messages):
            nonlocal active_snapshot
            evidence = collector.collect(task.task_id, current_messages, task)
            anchor = TaskAnchor(
                task_id=task.task_id,
                task_goal=constraint,
                user_constraints=[
                    UserConstraint(
                        constraint_id="constraint-public-api",
                        text=constraint,
                        source_user_message_id="m-user",
                    )
                ],
                latest_user_message_id="m-user",
                project_root=str(self.project),
                project_python=sys.executable,
                task_status=task.status.value,
            )
            protected = ProtectedContext(
                anchor,
                memory.latest(task.task_id),
                evidence,
                active_snapshot,
            )
            result = coordinator.invoke_automatic(
                CompactionRequest(
                    task_id=task.task_id,
                    entrypoint="automatic",
                    messages=tuple(current_messages),
                    active_event=events[-1] if events else None,
                    protected_context=protected,
                    budget=_budget("normal_compaction", 0.85),
                    model=object(),
                ),
                lambda compacted: ModelResponse(result=[AIMessage(content="继续")]),
            )
            event = DeepFixCompactionEvent.model_validate(
                result.command.update["_deepfix_compaction_event"]
            )
            active_snapshot = snapshots.activate_from_event(task.task_id, event)
            events.append(event)

        compact(messages)
        task.approvals.append(ApprovalRecord("edit_file", "approve", "L1"))
        task.changed_files.append("src/calculator.py")
        edited = project_backend.edit(
            "/src/calculator.py",
            "normalized = abs(normalized)",
            "normalized = -abs(normalized)",
        )
        assert edited.error is None
        messages.extend(
            [
                AIMessage(
                    id="m-edit-call",
                    content="执行已审批的最小修改",
                    tool_calls=[_call("edit_file", "edit-sign", {"file_path": "src/calculator.py"})],
                ),
                ToolMessage(
                    id="m-edit-result",
                    content="edit succeeded",
                    tool_call_id="edit-sign",
                    artifact={"operation": "edit", "status": "succeeded", "path": "src/calculator.py"},
                ),
                AIMessage(id="m-edit-explain", content="仅恢复负号，公开 API 未变化"),
            ]
        )
        compact(messages)
        passed = project_backend.execute(command)
        assert passed.exit_code == 0
        messages.extend(
            [
                AIMessage(
                    id="m-pass-call",
                    content="运行回归测试",
                    tool_calls=[_call("execute", "pytest-pass", {"command": command})],
                ),
                ToolMessage(
                    id="m-pass-result",
                    content=passed.output,
                    tool_call_id="pytest-pass",
                    artifact={"exit_code": passed.exit_code},
                ),
                AIMessage(id="m-pass-explain", content="真实 pytest 已通过"),
            ]
        )
        task.test_results.extend(
            [
                RepairTestResult(command, 1, failed.output, "pytest-fail", "m-fail-result"),
                RepairTestResult(command, 0, passed.output, "pytest-pass", "m-pass-result"),
            ]
        )
        compact(messages)
        task.transition_to(TaskStatus.REVIEWING)
        task.transition_to(TaskStatus.COMPLETED)
        tasks.save(task)
        final_evidence = collector.collect(task.task_id, messages, task)
        final_context = ProtectedContext(
            TaskAnchor(
                task_id=task.task_id,
                task_goal=constraint,
                user_constraints=active_snapshot.user_constraints,
                latest_user_message_id="m-user",
                project_root=str(self.project),
                project_python=sys.executable,
                task_status=task.status.value,
            ),
            memory.latest(task.task_id),
            final_evidence,
            active_snapshot,
        )
        history = artifact_adapter.read_verified(events[-1].conversation_artifact.path)
        scoped_partition = partition_work_units(_scoped(task.task_id, messages), set())
        units = {unit.unit_id: unit for unit in scoped_partition.units}
        compressed_ids = {
            unit_id
            for event in events
            for unit_id in event.conversation_artifact.work_unit_ids
        }
        no_split = all(
            all(message_id in history for message_id in units[unit_id].message_ids)
            for unit_id in compressed_ids
            if unit_id in units
        )
        return _WorkflowResult(
            task=tasks.get(task.task_id),
            original_constraint=constraint,
            final_snapshot=active_snapshot,
            rejected_reason=rejected_reason,
            final_protected_context=render_protected_context(final_context),
            history_contains_all_compressed_message_ids=all(
                message_id in history
                for unit_id in compressed_ids
                for message_id in units[unit_id].message_ids
            ),
            no_work_unit_was_split=no_split,
            target_project_has_no_deepfix_artifacts=not any(
                ".deepfix" in path.name for path in self.project.rglob("*")
            ),
        )


@pytest.fixture
def workflow(tmp_path):
    return _LongBugWorkflow(tmp_path)


def test_long_bugfix_preserves_evidence_across_three_compactions(workflow):
    result = workflow.run()
    assert result.task.test_results[-1].exit_code == 0
    assert result.original_constraint == result.final_snapshot.user_constraints[0].text
    assert result.rejected_reason in result.final_protected_context
    assert result.history_contains_all_compressed_message_ids
    assert result.no_work_unit_was_split
    assert result.target_project_has_no_deepfix_artifacts
    assert result.task.status is TaskStatus.COMPLETED


class _FailingAdapter:
    def __init__(self):
        self.calls = 0

    def persist_history(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("injected artifact failure")


def _failure_coordinator(tmp_path, adapter):
    database = tmp_path / "deepfix.sqlite3"
    return CompactionCoordinator(
        adapter=adapter,
        delta_generator=_FakeDeltaModel(),
        snapshot_builder=CompactionSnapshotBuilder(),
        snapshot_store=CompactionStore(database),
        memory_store=WorkingMemoryStore(database),
    )


def _failure_request(task_id, ratio, zone):
    messages = (HumanMessage(id="m1", content="保留这些原始消息"), AIMessage(id="m2", content="调查"))
    return CompactionRequest(
        task_id=task_id,
        entrypoint="automatic",
        messages=messages,
        active_event=None,
        protected_context=ProtectedContext(
            TaskAnchor(
                task_id=task_id,
                task_goal="fix",
                user_constraints=[],
                latest_user_message_id="m1",
                project_root="C:/project",
                project_python=sys.executable,
                task_status="investigating",
            ),
            None,
            DeterministicEvidenceBlock(),
            None,
        ),
        budget=_budget(zone, ratio),
        model=object(),
    )


def test_artifact_failure_at_085_continues_once_without_removing_messages(tmp_path):
    adapter = _FailingAdapter()
    coordinator = _failure_coordinator(tmp_path, adapter)
    request = _failure_request("task-normal", 0.85, "normal_compaction")
    original = tuple(request.messages)
    calls = []

    coordinator.invoke_automatic(
        request,
        lambda messages: calls.append(messages) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert calls == [original]
    assert tuple(request.messages) == original
    assert adapter.calls == 1


class _FailureAgent:
    def __init__(self, coordinator, ratio, zone):
        self.coordinator = coordinator
        self.ratio = ratio
        self.zone = zone

    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        request = _failure_request(task_id, self.ratio, self.zone)
        self.coordinator.invoke_automatic(
            request,
            lambda messages: ModelResponse(result=[AIMessage(content="unsafe")]),
        )
        return {}


def _service(tmp_path, agent):
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "deepfix.sqlite3"
    config = AppConfig(
        project_root=project,
        database_path=database,
        artifacts_path=tmp_path / "artifacts",
        main_model=ModelRoleConfig(
            model_name="deepseek-v4-pro",
            api_key=SecretStr("offline-main"),
            base_url="https://api.deepseek.com",
        ),
        compaction_model=ModelRoleConfig(
            model_name="deepseek-v4-flash",
            api_key=SecretStr("offline-compaction"),
            base_url="https://api.deepseek.com",
        ),
        approval_mode=ApprovalMode.MANUAL,
        project_python=Path(sys.executable),
    )
    repository = TaskRepository(database)
    memory = WorkingMemoryStore(database)
    research = ResearchEvidenceStore(database)
    compaction = CompactionStore(database)
    investigation = InvestigationCoordinator(
        store=InvestigationStore(database),
        tasks=repository,
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, research),
    )
    return BugfixService(
        agent,
        repository,
        ApprovalPolicy(ApprovalMode.MANUAL),
        config,
        memory,
        research,
        compaction,
        investigation,
    )


def test_artifact_failure_at_091_pauses_service_with_recovery_metadata(tmp_path):
    adapter = _FailingAdapter()
    coordinator = _failure_coordinator(tmp_path, adapter)
    service = _service(tmp_path, _FailureAgent(coordinator, 0.91, "emergency"))

    task = service.start("修复错误")

    assert task.status is TaskStatus.PAUSED
    assert task.context_recovery is not None
    assert task.context_recovery.original_messages_preserved is True
    assert adapter.calls == 1


class _DynamicProtectedBuilder:
    def build(self, task_id, messages, event):
        return _failure_request(task_id, 0.70, "normal").protected_context


class _NormalBudget:
    def measure(self, request, blocks):
        return _budget("normal", 0.70)

    def should_emit_memory_hint(self, version, unit_id):
        return False


class _OverflowAgent:
    def __init__(self, middleware):
        self.middleware = middleware
        self.handler_calls = 0

    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        messages = [
            HumanMessage(id="m1", content="old"),
            AIMessage(id="m2", content="old answer"),
            HumanMessage(id="m3", content="latest"),
            AIMessage(id="m4", content="working"),
        ]
        runtime = Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="cp",
                checkpoint_ns="",
                task_id="model",
                thread_id=task_id,
            )
        )
        request = ModelRequest(
            model=FakeListChatModel(responses=["unused"], profile={"max_input_tokens": 100}),
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[],
            state={"messages": messages},
            runtime=runtime,
        )

        def overflow(_request):
            self.handler_calls += 1
            raise ContextOverflowError("injected overflow")

        self.middleware.wrap_model_call(request, overflow)
        return {}


def test_two_overflows_call_handler_twice_then_pause_service(tmp_path):
    coordinator = _failure_coordinator(
        tmp_path,
        DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        ),
    )
    middleware = DeepFixCompactionMiddleware(
        _DynamicProtectedBuilder(), _NormalBudget(), coordinator
    )
    agent = _OverflowAgent(middleware)
    service = _service(tmp_path, agent)

    task = service.start("修复超长任务")

    assert agent.handler_calls == 2
    assert task.status is TaskStatus.PAUSED
    assert task.context_recovery.error_code == "context_overflow_after_single_retry"
