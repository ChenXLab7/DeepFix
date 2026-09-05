from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Literal

from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import (
    ContextBudgetMonitor,
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
    WorkUnit,
)
from deepfix.compaction.snapshot import (
    BuildSnapshotInput,
    CompactionDeltaGenerator,
    CompactionSnapshotBuilder,
)
from deepfix.compaction.work_units import WorkUnitPartition, partition_work_units
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.history import (
    HistoryRepository,
    history_record_from_snapshot,
)
from deepfix.domain_repositories.investigation import InvestigationRepository
from deepfix.protected_context import (
    ProtectedContext,
    ProtectedContextBuilder,
    ProtectedContextProjector,
)


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


class _StaticArtifactAdapter:
    def __init__(self, artifact: ArtifactReference) -> None:
        self.artifact = artifact

    def persist_history(self, *args: Any, **kwargs: Any) -> ArtifactReference:
        return self.artifact


class _StaticDeltaGenerator:
    def __init__(self, delta: Any) -> None:
        self.delta = delta

    def generate(self, model: Any, units: Any) -> Any:
        return self.delta


class CompactionCoordinator:
    def __init__(
        self,
        *,
        adapter: DeepAgentsArtifactAdapter | Any,
        delta_generator: CompactionDeltaGenerator | Any,
        snapshot_builder: CompactionSnapshotBuilder | Any,
        history_repository: HistoryRepository,
        evidence_repository: EvidenceRepository,
        investigation_repository: InvestigationRepository,
        budget_monitor: ContextBudgetMonitor | None = None,
        protected_builder: ProtectedContextBuilder | None = None,
        model: BaseChatModel | None = None,
        partitioner: Callable[
            [Sequence[AnyMessage], set[str] | frozenset[str]],
            WorkUnitPartition,
        ] = partition_work_units,
    ) -> None:
        self.adapter = adapter
        self.delta_generator = delta_generator
        self.snapshot_builder = snapshot_builder
        self.history_repository = history_repository
        self.evidence_repository = evidence_repository
        self.investigation_repository = investigation_repository
        self.budget_monitor = budget_monitor
        self.protected_builder = protected_builder
        self.model = model
        self.partitioner = partitioner
        self._failed_attempts: set[str] = set()

    def _snapshot(self, task_id: str, version: int) -> CompactionSnapshot:
        return self.history_repository.project_snapshot(
            task_id,
            version,
            evidence=self.evidence_repository,
            investigation=self.investigation_repository,
        )

    def active_snapshot_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent | None,
    ) -> CompactionSnapshot | None:
        if event is None:
            return None
        record = self.history_repository.active_from_event(task_id, event)
        return None if record is None else self._snapshot(task_id, record.version)

    def activate_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent,
    ) -> CompactionSnapshot:
        self.history_repository.activate(task_id, event.active_snapshot_version, event)
        return self._snapshot(task_id, event.active_snapshot_version)

    def _abandon_if_prepared(self, task_id: str, version: int, reason: str) -> None:
        """Best-effort cleanup must never mask the original recovery path."""
        try:
            snapshot = self._snapshot(task_id, version)
            if snapshot.lifecycle == "prepared":
                self.history_repository.abandon(task_id, version, reason)
        except Exception:  # noqa: BLE001 - cleanup must not mask recovery
            return

    def compact_manually(self, runtime: Any) -> Command:
        request, immediate = self._manual_request(runtime)
        if immediate is not None:
            return immediate
        assert request is not None
        try:
            prepared = self.prepare(request)
        except CompactionPreparationError as exc:
            return self._manual_failure(request, exc)
        return self._manual_success(request, prepared)

    async def acompact_manually(self, runtime: Any) -> Command:
        request, immediate = self._manual_request(runtime)
        if immediate is not None:
            return immediate
        assert request is not None
        try:
            prepared = await self.aprepare(request)
        except CompactionPreparationError as exc:
            return self._manual_failure(request, exc)
        return self._manual_success(request, prepared)

    def _manual_request(
        self,
        runtime: Any,
    ) -> tuple[CompactionRequest | None, Command | None]:
        if self.budget_monitor is None or self.protected_builder is None or self.model is None:
            raise RuntimeError("manual compaction dependencies are not configured")
        task_id = _runtime_task_id(runtime)
        messages = tuple(runtime.state.get("messages", ()))
        event_value = runtime.state.get("_deepfix_compaction_event")
        event = (
            DeepFixCompactionEvent.model_validate(event_value) if event_value is not None else None
        )
        protected = self.protected_builder.build(task_id, messages, event)
        projected = ProtectedContextProjector().project(protected)
        budget = self.budget_monitor.measure(
            SimpleNamespace(
                model=self.model,
                system_message=None,
                messages=list(messages),
                tools=list(getattr(runtime, "tools", ()) or ()),
            ),
            [
                projected.task_anchor_xml,
                projected.current_state_xml,
                projected.deterministic_evidence_xml,
                projected.execution_integrity_xml,
            ],
        )
        tool_call_id = str(getattr(runtime, "tool_call_id", "") or "")
        if budget.zone in {"normal", "observe"}:
            message = _manual_tool_message(
                task_id,
                tool_call_id or "manual-noop",
                "manual_noop",
                tool_call_id,
                "当前上下文尚未达到压缩条件。",
                "success",
            )
            return None, Command(update={"messages": [message]})

        request = CompactionRequest(
            task_id=task_id,
            entrypoint="manual_tool",
            messages=messages,
            active_event=event,
            protected_context=protected,
            budget=budget,
            model=self.model,
            tool_call_id=tool_call_id or None,
        )
        return request, None

    def _manual_failure(
        self,
        request: CompactionRequest,
        error: CompactionPreparationError,
    ) -> Command:
        failure = error.failure
        self.history_repository.record_failure(failure)
        self.history_repository.record_compaction_failure(
            request.task_id,
            event_id=f"{failure.attempt_id}:failure",
            error_code=failure.error_code,
        )
        self.history_repository.record_manual_compaction_error(
            request.task_id,
            event_id=f"{failure.attempt_id}:manual-error",
        )
        if request.budget.zone == "emergency":
            if failure.prepared_snapshot_version is not None:
                self.history_repository.abandon(
                    request.task_id,
                    failure.prepared_snapshot_version,
                    "emergency manual compaction failed",
                )
            raise _recovery_required(request, failure) from error
        if failure.prepared_snapshot_version is not None:
            self.history_repository.abandon(
                request.task_id,
                failure.prepared_snapshot_version,
                "manual compaction returned error",
            )
        message = _manual_tool_message(
            request.task_id,
            failure.attempt_id,
            "manual_error",
            request.tool_call_id or "",
            f"上下文压缩失败（{failure.error_code}），可稍后重试。",
            "error",
        )
        return Command(update={"messages": [message]})

    @staticmethod
    def _manual_success(
        request: CompactionRequest,
        prepared: PreparedCompaction,
    ) -> Command:
        message = _manual_tool_message(
            request.task_id,
            prepared.attempt_id,
            "manual_success",
            request.tool_call_id or "",
            f"上下文已安全压缩为 Snapshot v{prepared.snapshot.version}。",
            "success",
        )
        return Command(
            update={
                "_deepfix_compaction_event": prepared.event.model_dump(mode="json"),
                "_deepfix_compaction_session_id": prepared.attempt_id,
                "messages": [message],
            }
        )

    def prepare(self, request: CompactionRequest) -> PreparedCompaction:
        identities = ensure_message_ids(request.task_id, request.messages)
        messages = tuple(
            _with_task_scope(message, request.task_id) for message in identities.messages
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
                self.model if self.model is not None else request.model,
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
            task_anchor=request.protected_context.task_anchor,
            deterministic_evidence=request.protected_context.deterministic_evidence,
            current_facts=request.protected_context.confirmed_facts,
            current_hypotheses=request.protected_context.hypotheses,
            current_unresolved_questions=request.protected_context.unresolved_questions,
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
            saved_record = self.history_repository.save_prepared(
                history_record_from_snapshot(candidate, input_hash)
            )
            saved = self._snapshot(saved_record.task_id, saved_record.version)
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
            verified = self._snapshot(
                request.task_id,
                saved.version,
            )
            _validate_history_coverage(
                verified,
                retention.compressed_units,
                request.protected_context.task_anchor.latest_user_message_id,
            )
            if (
                verified.lifecycle not in {"prepared", "active"}
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
        try:
            active = self.activate_from_event(
                request.task_id,
                event,
            )
            if active.lifecycle != "active" or active.content_hash != verified.content_hash:
                raise ValueError("Snapshot activation verification failed")
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
                "snapshot_activate",
                "snapshot_activation_failed",
                artifact=artifact,
                prepared_version=saved.version,
            ) from exc
        snapshot_message = _snapshot_message(request.task_id, active, artifact)
        return PreparedCompaction(
            attempt_id=attempt_id,
            input_hash=input_hash,
            artifact_reference=artifact,
            snapshot=active,
            snapshot_message=snapshot_message,
            retention=retention,
            event=event,
        )

    async def aprepare(self, request: CompactionRequest) -> PreparedCompaction:
        identities = ensure_message_ids(request.task_id, request.messages)
        messages = tuple(
            _with_task_scope(message, request.task_id) for message in identities.messages
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
            artifact = await self.adapter.apersist_history(
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
            delta = await self.delta_generator.agenerate(
                self.model if self.model is not None else request.model,
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
        coordinator = CompactionCoordinator(
            adapter=_StaticArtifactAdapter(artifact),
            delta_generator=_StaticDeltaGenerator(delta),
            snapshot_builder=self.snapshot_builder,
            history_repository=self.history_repository,
            evidence_repository=self.evidence_repository,
            investigation_repository=self.investigation_repository,
            partitioner=self.partitioner,
        )
        return coordinator.prepare(request)

    def invoke_automatic(
        self,
        request: CompactionRequest,
        handler: Callable[[tuple[AnyMessage, ...]], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        input_hash = _input_hash(request, request.messages)
        attempt_id = _attempt_id(request.task_id, input_hash)
        self.history_repository.record_budget_observation(
            request.task_id,
            event_id=f"{attempt_id}:automatic-budget",
            estimated_tokens=request.budget.request_tokens,
            usage_ratio=request.budget.usage_ratio,
            zone=request.budget.zone,
        )
        if attempt_id in self._failed_attempts:
            response = handler(request.messages)
            self.history_repository.record_passthrough(
                request.task_id,
                event_id=f"{attempt_id}:repeat-passthrough",
            )
            return response
        try:
            prepared = self.prepare(request)
        except CompactionPreparationError as exc:
            failure = exc.failure
            self.history_repository.record_failure(failure)
            self.history_repository.record_compaction_failure(
                request.task_id,
                event_id=f"{failure.attempt_id}:failure",
                error_code=failure.error_code,
            )
            self._failed_attempts.add(failure.attempt_id)
            if request.budget.zone == "emergency" or request.entrypoint == "overflow_recovery":
                if failure.prepared_snapshot_version is not None:
                    self._abandon_if_prepared(
                        request.task_id,
                        failure.prepared_snapshot_version,
                        "emergency preparation failed",
                    )
                raise _recovery_required(request, failure) from exc
            response = handler(request.messages)
            self.history_repository.record_passthrough(
                request.task_id,
                event_id=f"{failure.attempt_id}:failure-passthrough",
            )
            if failure.prepared_snapshot_version is not None:
                self._abandon_if_prepared(
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
                update={"_deepfix_compaction_event": prepared.event.model_dump(mode="json")}
            ),
        )

    async def ainvoke_automatic(
        self,
        request: CompactionRequest,
        handler: Callable[[tuple[AnyMessage, ...]], Any],
    ) -> ModelResponse | ExtendedModelResponse:
        input_hash = _input_hash(request, request.messages)
        attempt_id = _attempt_id(request.task_id, input_hash)
        self.history_repository.record_budget_observation(
            request.task_id,
            event_id=f"{attempt_id}:automatic-budget",
            estimated_tokens=request.budget.request_tokens,
            usage_ratio=request.budget.usage_ratio,
            zone=request.budget.zone,
        )
        if attempt_id in self._failed_attempts:
            response = await handler(request.messages)
            self.history_repository.record_passthrough(
                request.task_id,
                event_id=f"{attempt_id}:repeat-passthrough",
            )
            return response
        try:
            prepared = await self.aprepare(request)
        except CompactionPreparationError as exc:
            failure = exc.failure
            self.history_repository.record_failure(failure)
            self.history_repository.record_compaction_failure(
                request.task_id,
                event_id=f"{failure.attempt_id}:failure",
                error_code=failure.error_code,
            )
            self._failed_attempts.add(failure.attempt_id)
            if request.budget.zone == "emergency" or request.entrypoint == "overflow_recovery":
                if failure.prepared_snapshot_version is not None:
                    self._abandon_if_prepared(
                        request.task_id,
                        failure.prepared_snapshot_version,
                        "emergency preparation failed",
                    )
                raise _recovery_required(request, failure) from exc
            response = await handler(request.messages)
            self.history_repository.record_passthrough(
                request.task_id,
                event_id=f"{failure.attempt_id}:failure-passthrough",
            )
            if failure.prepared_snapshot_version is not None:
                self._abandon_if_prepared(
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
        response = await handler(effective)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={"_deepfix_compaction_event": prepared.event.model_dump(mode="json")}
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
            request.active_event.active_snapshot_version if request.active_event else None
        ),
        "message_ids": [str(message.id) for message in normalized_messages],
        "current_hypothesis_ids": [
            item.hypothesis_id for item in request.protected_context.hypotheses
        ],
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
            content[:15_000] + "\n[Snapshot details truncated; full history: " + artifact.path + "]"
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


def _validate_history_coverage(
    snapshot: CompactionSnapshot,
    compressed_units: Sequence[WorkUnit],
    latest_user_message_id: str,
) -> None:
    expected_message_ids = list(
        dict.fromkeys(message_id for unit in compressed_units for message_id in unit.message_ids)
    )
    expected_work_unit_ids = list(dict.fromkeys(unit.unit_id for unit in compressed_units))
    if snapshot.coverage.last_user_message_id != latest_user_message_id:
        raise ValueError("History coverage latest User Message does not match")
    if snapshot.coverage.covered_message_ids != expected_message_ids:
        raise ValueError("History coverage Message IDs do not match")
    if snapshot.coverage.covered_work_unit_ids != expected_work_unit_ids:
        raise ValueError("History coverage WorkUnit IDs do not match")


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
            "artifact_reference": (artifact.path if artifact else error.failure.artifact_reference),
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
        "snapshot_activate",
        "compacted_model_call",
        "overflow_retry",
    }
    stage = failure.stage if failure.stage in allowed_stages else "snapshot_validate"
    active_version = request.active_event.active_snapshot_version if request.active_event else None
    return ContextRecoveryRequired(
        ContextRecoveryMetadata(
            task_id=request.task_id,
            stage=stage,
            error_code=failure.error_code,
            usage_ratio=request.budget.usage_ratio,
            active_snapshot_version=active_version,
            prepared_snapshot_version=failure.prepared_snapshot_version,
            prepared_snapshot_lifecycle=("prepared" if failure.prepared_snapshot_version else None),
            conversation_artifact=failure.artifact_reference,
            original_messages_preserved=True,
        )
    )


def _stable_id(prefix: str, *parts: str) -> str:
    value = "|".join((f"{prefix}:v1", *parts))
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]}"


def _runtime_task_id(runtime: Any) -> str:
    execution_info = getattr(runtime, "execution_info", None)
    if execution_info is not None and execution_info.thread_id:
        return str(execution_info.thread_id).strip()
    task_id = str(
        getattr(runtime, "config", {}).get("configurable", {}).get("thread_id", "")
    ).strip()
    if not task_id:
        raise ValueError("manual compaction runtime 缺少 thread_id")
    return task_id


def _manual_tool_message(
    task_id: str,
    scope_id: str,
    result_type: str,
    tool_call_id: str,
    content: str,
    status: Literal["success", "error"],
) -> ToolMessage:
    return ToolMessage(
        id=stable_generated_message_id(task_id, scope_id, result_type),
        content=content[:500],
        name="compact_conversation",
        tool_call_id=tool_call_id,
        status=status,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
