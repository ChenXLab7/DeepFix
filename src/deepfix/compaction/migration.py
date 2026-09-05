from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from pydantic import ValidationError

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.identity import ensure_message_ids, stable_generated_message_id
from deepfix.compaction.middleware import DeepFixCompactionState
from deepfix.compaction.models import (
    ApprovalEvidence,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    FileChangeEvidence,
    SystemTestEvidence,
    UserConstraint,
)
from deepfix.compaction.snapshot import snapshot_content_hash
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.history import history_record_from_snapshot
from deepfix.persistence import TaskRepository
from deepfix.protected_context import ProtectedContextBuilder
from deepfix.task_domain.legacy_payload import LegacyTaskPayload

_MIGRATION_VERSION = 1


@dataclass(frozen=True)
class LegacyContextStores:
    tasks: TaskRepository
    repositories: DomainRepositories


class LegacyContextMigrationState(DeepFixCompactionState):
    _summarization_event: NotRequired[Annotated[dict[str, object] | None, PrivateStateAttr]]


class LegacyContextMigrationMiddleware(AgentMiddleware):
    state_schema = LegacyContextMigrationState

    def __init__(
        self,
        stores: LegacyContextStores,
        adapter: DeepAgentsArtifactAdapter,
    ) -> None:
        self.stores = stores
        self.adapter = adapter

    def before_agent(
        self,
        state: LegacyContextMigrationState,
        runtime: Runtime,
    ) -> dict[str, object] | None:
        if state.get("_summarization_event") is None:
            return None
        execution = runtime.execution_info
        task_id = (
            str(execution.thread_id).strip()
            if execution is not None and execution.thread_id
            else ""
        )
        migrated = migrate_legacy_context_state(
            task_id,
            dict(state),
            self.stores,
            self.adapter,
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *migrated["messages"],
            ],
            "_summarization_event": None,
            "_deepfix_compaction_event": migrated["_deepfix_compaction_event"],
        }


def migrate_legacy_context_state(
    task_id: str,
    graph_state: dict[str, object],
    stores: LegacyContextStores,
    adapter: DeepAgentsArtifactAdapter,
) -> dict[str, object]:
    """Lazily convert one serialized Deep Agents summary into DeepFix state."""
    task_id = task_id.strip()
    if not task_id:
        raise ValueError("task_id 不能为空")
    state = dict(graph_state)
    messages = ensure_message_ids(task_id, list(state.get("messages", ()))).messages
    state["messages"] = messages
    legacy_event = state.get("_summarization_event")
    if legacy_event is None:
        state.pop("_summarization_event", None)
        return state

    recorded = stores.repositories.history.migrated_event(task_id)
    if recorded is not None:
        state.pop("_summarization_event", None)
        state["_deepfix_compaction_event"] = recorded.model_dump(mode="json")
        return state

    legacy = _load_legacy_payload(stores.repositories, task_id)
    task = legacy.definition
    if task.task_id != task_id:
        raise ValueError("legacy migration task_id 不匹配")
    summary = _legacy_summary_message(legacy_event)
    artifact_messages: list[AnyMessage] = [*messages]
    if summary is not None:
        artifact_messages.append(summary)
    artifact_messages = ensure_message_ids(task_id, artifact_messages).messages
    attempt_id = _stable_id("legacy-migration", task_id, *_message_ids(artifact_messages))
    retained_ids = set(_message_ids(messages))
    artifact = adapter.persist_history(
        task_id,
        attempt_id,
        artifact_messages,
        retained_ids,
    )

    existing_snapshots = stores.repositories.history.list_for_task(task_id)
    version = (existing_snapshots[-1].version if existing_snapshots else 0) + 1
    evidence = _legacy_evidence(task_id, legacy)
    for item in [
        *evidence.tests,
        *evidence.files,
        *evidence.approvals,
        *evidence.research,
    ]:
        stores.repositories.evidence.record_deterministic(
            task_id,
            item,
            provenance_root_ids=[item.evidence_id],
        )
    current = ProtectedContextBuilder(stores.repositories).build(
        task_id,
        messages,
        None,
    )
    source_message_id = next(
        (str(message.id) for message in reversed(messages) if isinstance(message, HumanMessage)),
        stable_generated_message_id(task_id, "legacy-task-goal", "user"),
    )
    constraint = UserConstraint(
        constraint_id=_stable_id(
            "constraint", task_id, source_message_id, task.original_problem, "active"
        ),
        text=task.original_problem,
        source_user_message_id=source_message_id,
    )
    snapshot = CompactionSnapshot(
        task_id=task_id,
        version=version,
        previous_version=None,
        lifecycle="prepared",
        created_at=_now(),
        source_work_unit_ids=[],
        task_goal=task.original_problem,
        user_constraints=[constraint],
        confirmed_facts=list(current.confirmed_facts),
        deterministic_evidence=evidence,
        active_hypotheses=[item for item in current.hypotheses if item.state == "active"],
        rejected_hypotheses=[item for item in current.hypotheses if item.state == "rejected"],
        confirmed_hypotheses=[item for item in current.hypotheses if item.state == "confirmed"],
        changed_files=evidence.files,
        experiments=[],
        test_results=evidence.tests,
        conflicts=[],
        unresolved_questions=list(current.unresolved_questions),
        next_steps=[],
        artifact_references=[artifact],
        content_hash="pending",
    )
    snapshot = snapshot.model_copy(update={"content_hash": snapshot_content_hash(snapshot)})
    prepared_record = stores.repositories.history.save_prepared(
        history_record_from_snapshot(snapshot, artifact.content_hash)
    )
    prepared = stores.repositories.history.project_snapshot(
        task_id,
        prepared_record.version,
        evidence=stores.repositories.evidence,
        investigation=stores.repositories.investigation,
    )
    snapshot_message_id = stable_generated_message_id(
        task_id, f"snapshot-{prepared.version}", "compaction-snapshot"
    )
    event = DeepFixCompactionEvent(
        event_id=_stable_id("event", task_id, artifact.content_hash),
        task_id=task_id,
        active_snapshot_version=prepared.version,
        snapshot_message_id=snapshot_message_id,
        retained_message_ids=sorted(retained_ids),
        conversation_artifact=artifact,
        input_hash=artifact.content_hash,
    )
    stores.repositories.history.activate(task_id, prepared.version, event)
    stores.repositories.history.record_migration(task_id, _MIGRATION_VERSION, event)
    stores.repositories.history.record_compaction_outcome(
        task_id,
        event_id=event.event_id,
        snapshot_version=prepared.version,
        artifact_path=artifact.path,
        emergency=False,
    )
    state.pop("_summarization_event", None)
    state["_deepfix_compaction_event"] = event.model_dump(mode="json")
    return state


def _legacy_summary_message(value: object) -> HumanMessage | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("summary_message")
    if isinstance(raw, HumanMessage):
        return raw.model_copy(deep=True)
    if isinstance(raw, dict):
        try:
            return HumanMessage.model_validate(raw)
        except ValidationError:
            return None
    return None


def _legacy_evidence(task_id: str, task: LegacyTaskPayload) -> DeterministicEvidenceBlock:
    payload = task.legacy_fields
    tests = [
        SystemTestEvidence(
            evidence_id=_stable_id("evidence", task_id, "legacy-test", str(index)),
            command=str(result.get("command", "")),
            exit_code=int(result.get("exit_code", -1)),
            summary=str(result.get("summary", "")),
            tool_call_id=str(result.get("tool_call_id") or f"legacy:{task_id}:test:{index}"),
            source_message_id=str(
                result.get("source_message_id")
                or f"legacy:{task_id}:test-result:{index}"
            ),
        )
        for index, result in enumerate(payload.get("test_results", []))
        if isinstance(result, dict)
    ]
    files = [
        FileChangeEvidence(
            evidence_id=_stable_id("evidence", task_id, "legacy-file", path),
            path=path,
            operation="approved_target",
            status="approved_target",
        )
        for path in dict.fromkeys(str(item) for item in payload.get("changed_files", []))
    ]
    approvals = [
        ApprovalEvidence(
            evidence_id=_stable_id("evidence", task_id, "legacy-approval", str(index)),
            operation=str(item.get("operation", "")),
            decision=str(item.get("decision", "")),
            risk=str(item.get("risk", "")),
        )
        for index, item in enumerate(payload.get("approvals", []))
        if isinstance(item, dict)
    ]
    return DeterministicEvidenceBlock(
        tests=tests,
        files=files,
        approvals=approvals,
        research=[],
    )


def _load_legacy_payload(
    repositories: DomainRepositories,
    task_id: str,
) -> LegacyTaskPayload:
    with repositories.database.connection() as connection:
        row = connection.execute(
            "SELECT payload FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    if row is None:
        raise KeyError(task_id)
    return LegacyTaskPayload.parse_json(str(row[0]))


def _message_ids(messages: list[AnyMessage]) -> list[str]:
    return [str(message.id) for message in messages]


def _stable_id(prefix: str, *parts: str) -> str:
    value = "|".join((f"{prefix}:v1", *parts))
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()[:32]}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
