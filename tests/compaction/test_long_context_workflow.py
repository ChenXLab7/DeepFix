from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend, LocalShellBackend
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import ExecutionInfo, Runtime
from pydantic import SecretStr

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator, CompactionRequest
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.middleware import (
    DeepFixCompactionMiddleware,
    DeepFixCompactionState,
)
from deepfix.compaction.models import (
    CompactionDelta,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    ResearchStatusEvidence,
    TaskAnchor,
    UserConstraint,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import partition_work_units
from deepfix.config import AppConfig, ApprovalMode, ModelRoleConfig
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.models import InvestigationHypothesis
from deepfix.investigation.store import InvestigationStore
from deepfix.models import ApprovalRecord, TaskState, TaskStatus
from deepfix.models import TestResult as RepairTestResult
from deepfix.navigation.models import TodoNavigationState
from deepfix.persistence import TaskRepository
from deepfix.protected_context import (
    ProtectedContext,
    ProtectedContextBuilder,
    render_protected_context,
)
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


_NAVIGATION_STATE = {
    "todos": [
        {"content": "reproduce failure", "status": "completed"},
        {"content": "identify root cause", "status": "in_progress"},
        {"content": "apply minimal fix", "status": "pending"},
        {"content": "run required verification", "status": "pending"},
    ],
    "_deepfix_todo_rounds_since_update": 2,
    "_deepfix_last_completed_tool_round_id": "m-read",
    "_deepfix_last_todo_progress_fingerprint": "todo-progress-fingerprint",
    "_deepfix_last_navigation_hint_fingerprint": "milestone-fingerprint",
    "_deepfix_pending_navigation_reminder": "request-local reminder",
}


class _CompactionNavigationState(TodoNavigationState, DeepFixCompactionState):
    pass


class _ForcedCompactionBudget:
    def measure(self, request, blocks):
        return _budget("normal_compaction", 0.85)


class _RecordingCompactionMiddleware(DeepFixCompactionMiddleware):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.navigation_inputs: list[dict[str, object]] = []

    def wrap_model_call(self, request, handler):
        self.navigation_inputs.append({key: request.state.get(key) for key in _NAVIGATION_STATE})
        return super().wrap_model_call(request, handler)


class _ConstraintProtectedBuilder:
    def __init__(self, delegate: ProtectedContextBuilder, constraint: str) -> None:
        self.delegate = delegate
        self.constraint = constraint

    def build(self, task_id, messages, event):
        context = self.delegate.build(task_id, messages, event)
        anchor = context.task_anchor.model_copy(
            update={
                "user_constraints": [
                    UserConstraint(
                        constraint_id="constraint-public-api",
                        text=self.constraint,
                        source_user_message_id="m-user",
                    )
                ]
            }
        )
        return replace(context, task_anchor=anchor)


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
    conversation_artifact_text: str
    restored_navigation_state: dict[str, object]
    compaction_navigation_inputs: list[dict[str, object]]
    final_snapshot_message: SystemMessage


class _LongBugWorkflow:
    def __init__(self, tmp_path: Path) -> None:
        fixture = Path(__file__).parent / "fixtures" / "bug_project"
        self.project = tmp_path / "target-project"
        shutil.copytree(fixture, self.project)
        self.state = tmp_path / "state"

    def run(self) -> _WorkflowResult:
        database = self.state / "deepfix.sqlite3"
        constraint = "只修复符号归一化，不得修改公开 API"
        repositories = DomainRepositories.create(database)
        tasks = repositories.tasks
        snapshots = CompactionStore(database, repositories=repositories)
        artifact_adapter = DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=self.state / "artifacts", virtual_mode=True)
        )
        coordinator = CompactionCoordinator(
            adapter=artifact_adapter,
            delta_generator=_FakeDeltaModel(),
            snapshot_builder=CompactionSnapshotBuilder(),
            snapshot_store=snapshots,
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
            ToolMessage(
                id="m-test", content="negative sign must return -4", tool_call_id="read-test"
            ),
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
        hypothesis_id = "hyp-cache-pollution"
        rejected_reason = "全新进程仍失败，缓存假设被排除"
        snapshots.save_evidence(
            task.task_id,
            ResearchStatusEvidence(
                evidence_id="research-sign-semantics",
                verification="verified",
                artifact_path="/.deepfix-artifacts/research/sign.md",
            ),
        )
        repositories.investigation.backfill_current_hypothesis(
            task.task_id,
            InvestigationHypothesis(
                hypothesis_id=hypothesis_id,
                statement="缓存污染导致失败",
                state="rejected",
                evidence_ids=[],
                checked_locations=[],
                reason=rejected_reason,
            ),
        )
        events = []
        active_snapshot = None
        protected_builder = ProtectedContextBuilder(repositories)
        compaction_middleware = _RecordingCompactionMiddleware(
            _ConstraintProtectedBuilder(
                protected_builder,
                constraint,
            ),
            _ForcedCompactionBudget(),
            coordinator,
        )
        navigation_checkpointer = InMemorySaver()
        navigation_config = {"configurable": {"thread_id": task.task_id}}
        navigation_graph_builder = StateGraph(_CompactionNavigationState)
        compaction_model_inputs: list[list[object]] = []

        def force_compaction(state):
            runtime = Runtime(
                execution_info=ExecutionInfo(
                    checkpoint_id="workflow-checkpoint",
                    checkpoint_ns="",
                    task_id="compaction-model",
                    thread_id=task.task_id,
                )
            )
            request = ModelRequest(
                model=FakeListChatModel(
                    responses=["continue"],
                    profile={"max_input_tokens": 100},
                ),
                messages=list(state["messages"]),
                system_message=SystemMessage(content="repair"),
                tools=[],
                state=state,
                runtime=runtime,
            )
            response = compaction_middleware.wrap_model_call(
                request,
                lambda compacted: (
                    compaction_model_inputs.append(list(compacted.messages))
                    or ModelResponse(result=[AIMessage(content="continue")])
                ),
            )
            command_update = getattr(response, "command", None)
            failures = snapshots.list_failures(task.task_id)
            assert command_update is not None, [
                (failure.stage, failure.error_code) for failure in failures
            ]
            return command_update.update

        navigation_graph_builder.add_node("compact", force_compaction)
        navigation_graph_builder.add_edge(START, "compact")
        navigation_graph_builder.add_edge("compact", END)
        navigation_graph = navigation_graph_builder.compile(checkpointer=navigation_checkpointer)
        restored_navigation_state: dict[str, object] = {}

        def compact(current_messages):
            nonlocal active_snapshot, restored_navigation_state
            graph_input: dict[str, object] = {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *current_messages,
                ],
            }
            if not events:
                graph_input.update(_NAVIGATION_STATE)
            navigation_graph.invoke(graph_input, navigation_config)
            restored_navigation_state = dict(navigation_graph.get_state(navigation_config).values)
            event = DeepFixCompactionEvent.model_validate(
                restored_navigation_state["_deepfix_compaction_event"]
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
                    tool_calls=[
                        _call("edit_file", "edit-sign", {"file_path": "src/calculator.py"})
                    ],
                ),
                ToolMessage(
                    id="m-edit-result",
                    content="edit succeeded",
                    tool_call_id="edit-sign",
                    artifact={
                        "operation": "edit",
                        "status": "succeeded",
                        "path": "src/calculator.py",
                    },
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
        final_context = protected_builder.build(task.task_id, messages, events[-1])
        history = artifact_adapter.read_verified(events[-1].conversation_artifact.path)
        scoped_partition = partition_work_units(_scoped(task.task_id, messages), set())
        units = {unit.unit_id: unit for unit in scoped_partition.units}
        compressed_ids = {
            unit_id for event in events for unit_id in event.conversation_artifact.work_unit_ids
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
                if unit_id in units
                for message_id in units[unit_id].message_ids
            ),
            no_work_unit_was_split=no_split,
            target_project_has_no_deepfix_artifacts=not any(
                ".deepfix" in path.name for path in self.project.rglob("*")
            ),
            conversation_artifact_text=history,
            restored_navigation_state=restored_navigation_state,
            compaction_navigation_inputs=compaction_middleware.navigation_inputs,
            final_snapshot_message=next(
                message
                for message in compaction_model_inputs[-1]
                if isinstance(message, SystemMessage)
                and "_deepfix_snapshot_version" in message.additional_kwargs
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
    assert result.compaction_navigation_inputs == [_NAVIGATION_STATE] * 3
    for field, value in _NAVIGATION_STATE.items():
        assert result.restored_navigation_state[field] == value
    for forbidden in _NAVIGATION_STATE:
        assert forbidden not in result.final_snapshot.model_dump_json()
        assert forbidden not in result.conversation_artifact_text


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
    )


def _failure_request(task_id, ratio, zone):
    messages = (
        HumanMessage(id="m1", content="保留这些原始消息"),
        AIMessage(id="m2", content="调查"),
    )
    return CompactionRequest(
        task_id=task_id,
        entrypoint="automatic",
        messages=messages,
        active_event=None,
        protected_context=ProtectedContext(
            task_anchor=TaskAnchor(
                task_id=task_id,
                task_goal="fix",
                user_constraints=[],
                latest_user_message_id="m1",
                project_root="C:/project",
                project_python=sys.executable,
                task_status="investigating",
            ),
            deterministic_evidence=DeterministicEvidenceBlock(),
            confirmed_facts=(),
            hypotheses=(),
            unresolved_questions=(),
            execution_integrity=ExecutionIntegrity(receipt_count=0, approval_count=0),
            external_evidence=(),
            active_snapshot=None,
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
