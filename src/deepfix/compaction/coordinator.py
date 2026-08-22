from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.types import Command

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import (
    ContextBudgetReport,
    RetentionPlan,
    select_retained_units,
)
from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    CompactionPreparationError,
    ContextRecoveryRequired,
    SnapshotBuildError,
    SnapshotPersistenceError,
)
from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_generated_message_id,
)
from deepfix.compaction.models import (
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    ContextRecoveryMetadata,
    DeepFixCompactionEvent,
)
from deepfix.compaction.snapshot import (
    BuildSnapshotInput,
    CompactionDeltaGenerator,
    CompactionSnapshotBuilder,
)
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import WorkUnitPartition, partition_work_units
from deepfix.memory import WorkingMemoryStore
from deepfix.protected_context import ProtectedContext


@dataclass(frozen=True)
class CompactionRequest:
    task_id: str
    entrypoint: Literal["automatic", "manual_tool", "overflow_recovery"]
    messages: tuple[AnyMessage, ...]
    active_event: DeepFixCompactionEvent | None
    protected_context: ProtectedContext
    budget: ContextBudgetReport
    model: BaseChatModel
    tool_call_id: str | None = None


@dataclass(frozen=True)
class PreparedCompaction:
    attempt_id: str
    input_hash: str
    artifact_reference: ArtifactReference
    snapshot: CompactionSnapshot
    snapshot_message: SystemMessage
    retention: RetentionPlan
    event: DeepFixCompactionEvent


class CompactionCoordinator:
    def __init__(
        self,
        *,
        adapter: DeepAgentsArtifactAdapter | Any,
        delta_generator: CompactionDeltaGenerator | Any,
        snapshot_builder: CompactionSnapshotBuilder | Any,
        snapshot_store: CompactionStore | Any,
        memory_store: WorkingMemoryStore,
        partitioner: Callable[
            [Sequence[AnyMessage], set[str] | frozenset[str]],
            WorkUnitPartition,
        ] = partition_work_units,
    ) -> None:
        self.adapter = adapter
        self.delta_generator = delta_generator
        self.snapshot_builder = snapshot_builder
        self.snapshot_store = snapshot_store
        self.memory_store = memory_store
        self.partitioner = partitioner
        self._failed_attempts: set[str] = set()

    def prepare(self, request: CompactionRequest) -> PreparedCompaction:
        identities = ensure_message_ids(request.task_id, request.messages)
        messages = tuple(
            _with_task_scope(message, request.task_id)
            for message in identities.messages
        )
        partition = self.partitioner(messages, identities.conflicted_message_ids)
        input_hash = _input_hash(request, messages)
        attempt_id = _attempt_id(request.task_id, input_hash)
        retention = select_retained_units(
            request.budget,
            partition.units,
            request.protected_context.task_anchor.latest_user_message_id,
        )
        compressed_ids = {unit.unit_id for unit in retention.compressed_units}

        try:
            artifact = self.adapter.persist_history(
                request.task_id,
                attempt_id,
                messages,
                retention.retained_message_ids,
                work_unit_ids=compressed_ids,
            )
        except CompactionPreparationError as exc:
            raise _rebind_error(exc, request, attempt_id, input_hash) from exc
        except Exception as exc:
            raise _new_preparation_error(
                ArtifactPersistenceError,
                request,
                attempt_id,
                input_hash,
                "artifact_write",
                "artifact_write_failed",
            ) from exc

        try:
            delta = self.delta_generator.generate(
                request.model,
                retention.compressed_units,
            )
        except CompactionPreparationError as exc:
            raise _rebind_error(
                exc,
                request,
                attempt_id,
                input_hash,
                artifact=artifact,
            ) from exc
        except Exception as exc:
            raise _new_preparation_error(
                SnapshotBuildError,
                request,
                attempt_id,
                input_hash,
                "delta_generation",
                "delta_generation_failed",
                artifact=artifact,
            ) from exc

        build_input = BuildSnapshotInput(
            task_id=request.task_id,
            previous_snapshot=request.protected_context.active_snapshot,
            compressed_units=retention.compressed_units,
            latest_memory=request.protected_context.working_memory,
            task_anchor=request.protected_context.task_anchor,
            deterministic_evidence=request.protected_context.deterministic_evidence,
            delta=delta,
            artifact_reference=artifact,
            input_hash=input_hash,
        )
        try:
            candidate = self.snapshot_builder.build(build_input)
        except CompactionPreparationError as exc:
            raise _rebind_error(
                exc,
                request,
                attempt_id,
                input_hash,
                artifact=artifact,
            ) from exc
        except Exception as exc:
            raise _new_preparation_error(
                SnapshotBuildError,
                request,
                attempt_id,
                input_hash,
                "snapshot_validate",
                "snapshot_build_failed",
                artifact=artifact,
            ) from exc

        try:
            saved = self.snapshot_store.save_prepared_snapshot(candidate, input_hash)
        except CompactionPreparationError as exc:
            raise _rebind_error(
                exc,
                request,
                attempt_id,
                input_hash,
                artifact=artifact,
            ) from exc
        except Exception as exc:
            raise _new_preparation_error(
                SnapshotPersistenceError,
                request,
                attempt_id,
                input_hash,
                "snapshot_write",
                "snapshot_write_failed",
                artifact=artifact,
            ) from exc

        try:
            verified = self.snapshot_store.get_snapshot(
                request.task_id,
                saved.version,
            )
            if (
                verified.lifecycle != "prepared"
                or verified.content_hash != saved.content_hash
                or verified.model_dump(mode="json") != saved.model_dump(mode="json")
            ):
                raise ValueError("Snapshot 写后校验不一致")
        except CompactionPreparationError as exc:
            raise _rebind_error(
                exc,
                request,
                attempt_id,
                input_hash,
                artifact=artifact,
                prepared_version=saved.version,
            ) from exc
        except Exception as exc:
            raise _new_preparation_error(
                SnapshotPersistenceError,
                request,
                attempt_id,
                input_hash,
                "snapshot_verify",
                "snapshot_persistence_failed",
                artifact=artifact,
                prepared_version=saved.version,
            ) from exc

        snapshot_message = _snapshot_message(request.task_id, verified, artifact)
        event = DeepFixCompactionEvent(
            event_id=_stable_id("event", request.task_id, input_hash),
            task_id=request.task_id,
            active_snapshot_version=verified.version,
            snapshot_message_id=str(snapshot_message.id),
            retained_message_ids=sorted(retention.retained_message_ids),
            conversation_artifact=artifact,
            input_hash=input_hash,
        )
        return PreparedCompaction(
            attempt_id=attempt_id,
            input_hash=input_hash,
            artifact_reference=artifact,
            snapshot=verified,
            snapshot_message=snapshot_message,
            retention=retention,
            event=event,
        )

    def invoke_automatic(
        self,
        request: CompactionRequest,
        handler: Callable[[tuple[AnyMessage, ...]], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        self.memory_store.record_budget(
            request.task_id,
            request.budget.usage_ratio,
            request.budget.zone,
        )
        input_hash = _input_hash(request, request.messages)
        attempt_id = _attempt_id(request.task_id, input_hash)
        if attempt_id in self._failed_attempts:
            response = handler(request.messages)
            self.memory_store.record_passthrough(request.task_id)
            return response
        try:
            prepared = self.prepare(request)
        except CompactionPreparationError as exc:
            failure = exc.failure
            self.snapshot_store.record_failure(failure)
            self.memory_store.record_compaction_failure(
                request.task_id,
                failure.error_code,
            )
            self._failed_attempts.add(failure.attempt_id)
            if request.budget.zone == "emergency" or request.entrypoint == "overflow_recovery":
                raise _recovery_required(request, failure) from exc
            response = handler(request.messages)
            self.memory_store.record_passthrough(request.task_id)
            if failure.prepared_snapshot_version is not None:
                self.snapshot_store.abandon_snapshot(
                    request.task_id,
                    failure.prepared_snapshot_version,
                    "passthrough produced newer conversation state",
                )
            return response

        effective = (
            prepared.snapshot_message,
            *(
                message
                for message in request.messages
                if str(message.id) in prepared.retention.retained_message_ids
            ),
        )
        response = handler(effective)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "_deepfix_compaction_event": prepared.event.model_dump(
                        mode="json"
                    )
                }
            ),
        )


def _with_task_scope(message: AnyMessage, task_id: str) -> AnyMessage:
    additional = dict(message.additional_kwargs)
    additional["_deepfix_task_id"] = task_id
    return message.model_copy(update={"additional_kwargs": additional}, deep=True)


def _input_hash(
    request: CompactionRequest,
    messages: Sequence[AnyMessage],
) -> str:
    normalized_messages = ensure_message_ids(request.task_id, messages).messages
    evidence = request.protected_context.deterministic_evidence.model_dump(mode="json")
    payload = {
        "task_id": request.task_id,
        "active_snapshot_version": (
            request.active_event.active_snapshot_version
            if request.active_event
            else None
        ),
        "message_ids": [str(message.id) for message in normalized_messages],
        "working_memory_version": (
            request.protected_context.working_memory.version
            if request.protected_context.working_memory
            else None
        ),
        "deterministic_evidence": evidence,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _attempt_id(task_id: str, input_hash: str) -> str:
    return _stable_id("attempt", task_id, input_hash)


def _snapshot_message(
    task_id: str,
    snapshot: CompactionSnapshot,
    artifact: ArtifactReference,
) -> SystemMessage:
    content = snapshot.model_dump_json()
    if len(content) > 16_000:
        content = (
            content[:15_000]
            + "\n[Snapshot details truncated; full history: "
            + artifact.path
            + "]"
        )
    return SystemMessage(
        id=stable_generated_message_id(
            task_id,
            f"snapshot-{snapshot.version}-{snapshot.content_hash}",
            "compaction_snapshot",
        ),
        content=content,
        additional_kwargs={"_deepfix_snapshot_version": snapshot.version},
    )


def _rebind_error(
    error: CompactionPreparationError,
    request: CompactionRequest,
    attempt_id: str,
    input_hash: str,
    *,
    artifact: ArtifactReference | None = None,
    prepared_version: int | None = None,
) -> CompactionPreparationError:
    failure = error.failure.model_copy(
        update={
            "attempt_id": attempt_id,
            "task_id": request.task_id,
            "entrypoint": request.entrypoint,
            "budget_zone": request.budget.zone,
            "input_hash": input_hash,
            "original_messages_preserved": True,
            "artifact_reference": (
                artifact.path if artifact else error.failure.artifact_reference
            ),
            "prepared_snapshot_version": (
                prepared_version
                if prepared_version is not None
                else error.failure.prepared_snapshot_version
            ),
            "recorded_at": _utc_now(),
        }
    )
    return type(error)(failure)


def _new_preparation_error(
    error_type: type[CompactionPreparationError],
    request: CompactionRequest,
    attempt_id: str,
    input_hash: str,
    stage: str,
    error_code: str,
    *,
    artifact: ArtifactReference | None = None,
    prepared_version: int | None = None,
) -> CompactionPreparationError:
    return error_type(
        CompactionFailureRecord(
            attempt_id=attempt_id,
            task_id=request.task_id,
            entrypoint=request.entrypoint,
            budget_zone=request.budget.zone,
            stage=stage,
            error_code=error_code,
            input_hash=input_hash,
            original_messages_preserved=True,
            artifact_reference=artifact.path if artifact else None,
            prepared_snapshot_version=prepared_version,
            recorded_at=_utc_now(),
        )
    )


def _recovery_required(
    request: CompactionRequest,
    failure: CompactionFailureRecord,
) -> ContextRecoveryRequired:
    allowed_stages = {
        "artifact_write",
        "artifact_verify",
        "delta_generation",
        "snapshot_validate",
        "snapshot_write",
        "snapshot_verify",
        "compacted_model_call",
        "overflow_retry",
    }
    stage = failure.stage if failure.stage in allowed_stages else "snapshot_validate"
    active_version = (
        request.active_event.active_snapshot_version
        if request.active_event
        else None
    )
    return ContextRecoveryRequired(
        ContextRecoveryMetadata(
            task_id=request.task_id,
            stage=stage,
            error_code=failure.error_code,
            usage_ratio=request.budget.usage_ratio,
            working_memory_version=(
                request.protected_context.working_memory.version
                if request.protected_context.working_memory
                else None
            ),
            active_snapshot_version=active_version,
            prepared_snapshot_version=failure.prepared_snapshot_version,
            prepared_snapshot_lifecycle=(
                "prepared" if failure.prepared_snapshot_version else None
            ),
            conversation_artifact=failure.artifact_reference,
            original_messages_preserved=True,
        )
    )


def _stable_id(prefix: str, *parts: str) -> str:
    value = "|".join((f"{prefix}:v1", *parts))
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
