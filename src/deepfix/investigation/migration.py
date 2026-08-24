from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime

from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.investigation.classification import is_pytest_verification
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    AgentPhase,
    InvestigationEventType,
    InvestigationHypothesis,
    InvestigationState,
    NewInvestigationEvent,
    ProgressKind,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import WorkingMemoryStore
from deepfix.models import TaskStatus
from deepfix.persistence import TaskRepository

_FILE_OPERATIONS = {
    "write_file": "write",
    "edit_file": "edit",
    "delete": "delete",
}


@dataclass(frozen=True)
class _LegacyObservation:
    kind: str
    source_message_id: str
    tool_call_id: str
    evidence_id: str | None = None
    path: str | None = None
    exit_code: int | None = None


class InvestigationMigrator:
    VERSION = 1

    def __init__(
        self,
        *,
        tasks: TaskRepository,
        store: InvestigationStore,
        compaction_store: CompactionStore,
        memory: WorkingMemoryStore,
    ) -> None:
        self.tasks = tasks
        self.store = store
        self.compaction_store = compaction_store
        self.memory = memory

    def migrate(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
    ) -> InvestigationState:
        normalized_task_id = task_id.strip()
        if not normalized_task_id:
            raise ValueError("task_id 不能为空")
        existing = self.store.load(normalized_task_id)
        if existing is not None and existing.migration_version >= self.VERSION:
            return existing

        task = self.tasks.get(normalized_task_id)
        identified = ensure_message_ids(normalized_task_id, messages).messages
        observations = _durable_observations(
            normalized_task_id,
            identified,
            task.project_python,
            self.compaction_store,
        )
        reconstructed_phase = _reconstruct_phase(observations)
        hypotheses = _working_memory_candidates(
            normalized_task_id,
            self.memory,
            existing.hypotheses if existing is not None else [],
        )
        state = (existing or InvestigationState.new(normalized_task_id)).model_copy(
            update={
                "agent_phase": (
                    AgentPhase.CLARIFYING
                    if task.status is TaskStatus.CLARIFYING
                    else reconstructed_phase
                ),
                "paused_agent_phase": (
                    reconstructed_phase
                    if task.status is TaskStatus.CLARIFYING
                    else None
                ),
                "hypotheses": hypotheses,
                "supported_hypothesis_ids": [],
                "test_evidence_ids": _test_evidence_ids(observations),
                "migration_version": self.VERSION,
            }
        )
        events = _migration_events(normalized_task_id, observations, state.agent_phase)
        if existing is None:
            events.insert(0, NewInvestigationEvent.task_started(normalized_task_id))
        return self.store.commit(existing.version if existing else 0, events, state)


class InvestigationMigrationMiddleware(AgentMiddleware):
    def __init__(self, migrator: InvestigationMigrator) -> None:
        self.migrator = migrator

    def before_agent(
        self,
        state: dict[str, Any],
        runtime: Runtime,
    ) -> dict[str, object] | None:
        execution = runtime.execution_info
        task_id = (
            str(execution.thread_id).strip()
            if execution is not None and execution.thread_id
            else ""
        )
        if not task_id:
            return None
        messages = state.get("messages", ())
        self.migrator.migrate(
            task_id,
            messages if isinstance(messages, Sequence) else (),
        )
        return None


def _durable_observations(
    task_id: str,
    messages: Sequence[AnyMessage],
    project_python: str,
    compaction_store: CompactionStore,
) -> list[_LegacyObservation]:
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results: dict[str, list[ToolMessage]] = defaultdict(list)
    result_order: dict[str, int] = {}
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[str(call.get("id", ""))].append(call)
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            results[call_id].append(message)
            result_order.setdefault(call_id, index)

    evidence_by_call = {
        item.tool_call_id: item
        for item in compaction_store.list_evidence(task_id)
        if isinstance(item, (SystemTestEvidence, FileChangeEvidence))
        and item.tool_call_id
    }
    observations: list[tuple[int, _LegacyObservation]] = []
    for call_id, call_items in calls.items():
        result_items = results.get(call_id, [])
        if not call_id or len(call_items) != 1 or len(result_items) != 1:
            continue
        call = call_items[0]
        result = result_items[0]
        args = call.get("args", {})
        args = args if isinstance(args, Mapping) else {}
        name = str(call.get("name", ""))
        evidence = evidence_by_call.get(call_id)
        observation = _observation_from_pair(
            name,
            call_id,
            args,
            result,
            project_python,
            evidence.evidence_id if evidence is not None else None,
        )
        if observation is not None:
            observations.append((result_order[call_id], observation))
    return [item for _, item in sorted(observations, key=lambda pair: pair[0])]


def _observation_from_pair(
    name: str,
    call_id: str,
    args: Mapping[str, object],
    result: ToolMessage,
    project_python: str,
    evidence_id: str | None,
) -> _LegacyObservation | None:
    artifact = result.artifact
    if not isinstance(artifact, Mapping):
        return None
    source_message_id = str(result.id or "").strip()
    if not source_message_id:
        return None

    if name == "execute":
        command = str(args.get("command", "")).strip()
        exit_code = artifact.get("exit_code")
        if (
            not is_pytest_verification(command, project_python)
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
        ):
            return None
        return _LegacyObservation(
            kind="pytest",
            source_message_id=source_message_id,
            tool_call_id=call_id,
            evidence_id=evidence_id,
            exit_code=exit_code,
        )

    expected_operation = _FILE_OPERATIONS.get(name)
    if (
        expected_operation is None
        or artifact.get("operation") != expected_operation
        or artifact.get("status") != "succeeded"
        or result.status == "error"
    ):
        return None
    path = artifact.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    return _LegacyObservation(
        kind="edit",
        source_message_id=source_message_id,
        tool_call_id=call_id,
        evidence_id=evidence_id,
        path=path.strip(),
    )


def _reconstruct_phase(observations: Sequence[_LegacyObservation]) -> AgentPhase:
    phase = AgentPhase.INVESTIGATING
    edit_seen = False
    for observation in observations:
        if observation.kind == "edit":
            edit_seen = True
            phase = AgentPhase.EDITING
        elif observation.kind == "pytest":
            if edit_seen:
                phase = (
                    AgentPhase.REVIEWING
                    if observation.exit_code == 0
                    else AgentPhase.DIAGNOSING
                )
            elif observation.exit_code != 0:
                phase = AgentPhase.DIAGNOSING
    return phase


def _working_memory_candidates(
    task_id: str,
    memory: WorkingMemoryStore,
    existing: Sequence[InvestigationHypothesis],
) -> list[InvestigationHypothesis]:
    candidates = {item.hypothesis_id: item for item in existing}
    latest = memory.latest(task_id)
    if latest is None:
        return list(candidates.values())
    for item in latest.snapshot.active_hypotheses:
        hypothesis_id = stable_investigation_id(
            "hyp", task_id, "legacy-working-memory", item.hypothesis_id, item.text
        )
        candidates.setdefault(
            hypothesis_id,
            InvestigationHypothesis(
                hypothesis_id=hypothesis_id,
                statement=item.text,
                state="candidate",
                evidence_ids=[],
                checked_locations=[],
                reason="legacy Working Memory candidate; unverified",
            ),
        )
    return list(candidates.values())


def _test_evidence_ids(
    observations: Sequence[_LegacyObservation],
) -> list[str]:
    return list(
        dict.fromkeys(
            item.evidence_id
            for item in observations
            if item.kind == "pytest" and item.evidence_id is not None
        )
    )


def _migration_events(
    task_id: str,
    observations: Sequence[_LegacyObservation],
    final_phase: AgentPhase,
) -> list[NewInvestigationEvent]:
    phase = AgentPhase.INVESTIGATING
    edit_seen = False
    events: list[NewInvestigationEvent] = []
    for observation in observations:
        before = phase
        if observation.kind == "edit":
            edit_seen = True
            phase = AgentPhase.EDITING
            event_type = InvestigationEventType.FILE_CHANGED
            progress_kind = ProgressKind.FILE_CHANGE
        else:
            if edit_seen:
                phase = (
                    AgentPhase.REVIEWING
                    if observation.exit_code == 0
                    else AgentPhase.DIAGNOSING
                )
                event_type = InvestigationEventType.POST_EDIT_TEST_OBSERVED
                progress_kind = ProgressKind.POST_EDIT_TEST
            else:
                phase = (
                    AgentPhase.DIAGNOSING
                    if observation.exit_code != 0
                    else phase
                )
                event_type = InvestigationEventType.TEST_OBSERVED
                progress_kind = ProgressKind.TEST_EVIDENCE
        events.append(
            NewInvestigationEvent(
                event_id=stable_investigation_id(
                    "event",
                    task_id,
                    "legacy-migration-v1",
                    observation.source_message_id,
                    observation.tool_call_id,
                    event_type.value,
                ),
                task_id=task_id,
                event_type=event_type,
                source_message_id=observation.source_message_id,
                tool_call_id=observation.tool_call_id,
                phase_before=before,
                phase_after=phase,
                progress_kind=progress_kind,
                payload={
                    "evidence_id": observation.evidence_id,
                    "path": observation.path,
                    "exit_code": observation.exit_code,
                    "migration_version": 1,
                },
            )
        )
    if not events:
        return []
    if final_phase is AgentPhase.CLARIFYING:
        events[-1] = events[-1].model_copy(update={"phase_after": final_phase})
    return events
