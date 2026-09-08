from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import ExtendedModelResponse, PrivateStateAttr
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from deepfix.compaction.budget import ContextBudgetMonitor, ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator, CompactionRequest
from deepfix.compaction.errors import ArtifactPersistenceError, ContextRecoveryRequired
from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.models import (
    ContextRecoveryMetadata,
    DeepFixCompactionEvent,
)
from deepfix.compaction.work_units import partition_work_units
from deepfix.protected_context import (
    ProtectedContext,
    ProtectedContextBuilder,
    ProtectedContextProjector,
    render_protected_context,
)


class DeepFixCompactionState(AgentState):
    _deepfix_compaction_event: NotRequired[
        Annotated[dict[str, object] | None, PrivateStateAttr]
    ]
    _deepfix_compaction_session_id: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]


class MessageIdentityMiddleware(AgentMiddleware):
    state_schema = DeepFixCompactionState

    def before_agent(
        self,
        state: DeepFixCompactionState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._normalize(state, runtime)

    def before_model(
        self,
        state: DeepFixCompactionState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._normalize(state, runtime)

    @staticmethod
    def _normalize(
        state: DeepFixCompactionState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        task_id = _runtime_task_id(runtime)
        if not task_id:
            return None
        messages = list(state.get("messages", ()))
        identities = ensure_message_ids(task_id, messages)
        if not identities.assigned_message_ids:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *identities.messages,
            ]
        }


class DeepFixCompactionMiddleware(AgentMiddleware):
    state_schema = DeepFixCompactionState

    def __init__(
        self,
        protected_builder: ProtectedContextBuilder | Any,
        budget_monitor: ContextBudgetMonitor | Any,
        coordinator: CompactionCoordinator | Any,
        projector: ProtectedContextProjector | None = None,
    ) -> None:
        self.protected_builder = protected_builder
        self.budget_monitor = budget_monitor
        self.coordinator = coordinator
        self.projector = projector or ProtectedContextProjector()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        prepared = self._prepare_request(request)
        try:
            return self._invoke_zone(prepared, handler)
        except ContextOverflowError as first_overflow:
            return self._recover_overflow(prepared, handler, first_overflow)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        prepared = self._prepare_request(request)
        try:
            return await self._ainvoke_zone(prepared, handler)
        except ContextOverflowError as first_overflow:
            return await self._arecover_overflow(
                prepared,
                handler,
                first_overflow,
            )

    def _prepare_request(self, request: ModelRequest) -> _PreparedModelRequest:
        task_id = _model_request_task_id(request)
        event = _event_from_state(request.state)
        context = self.protected_builder.build(task_id, request.messages, event)
        context, effective_messages = self._reconcile_event(
            task_id,
            event,
            context,
            request.messages,
        )
        protected_xml = render_protected_context(
            context,
            projector=self.projector,
        )
        original_system = request.system_message.text if request.system_message else ""
        if "<deepfix_task_anchor>" in original_system:
            system_content = original_system
        else:
            system_content = (
                f"{original_system}\n\n{protected_xml}"
                if original_system
                else protected_xml
            )
        updated = request.override(
            messages=list(effective_messages),
            system_message=SystemMessage(content=system_content),
        )
        report = self.budget_monitor.measure(updated, [])
        view = self._lightweight_view(task_id, updated.messages, report)
        if view is not None:
            updated = updated.override(messages=view)
            report = self.budget_monitor.measure(updated, [])
        self.coordinator.history_repository.record_budget_observation(
            task_id,
            event_id=_telemetry_event_id(
                task_id,
                "model-budget",
                report.request_tokens,
                report.zone,
                *(str(message.id) for message in effective_messages),
            ),
            estimated_tokens=report.request_tokens,
            usage_ratio=report.usage_ratio,
            zone=report.zone,
        )
        return _PreparedModelRequest(
            task_id=task_id,
            request=updated,
            context=context,
            event=event,
            report=report,
        )

    def _lightweight_view(
        self, task_id: str, messages: Sequence[AnyMessage], report: ContextBudgetReport,
    ) -> list[AnyMessage] | None:
        """Request-local only: never return a checkpoint Command or mutate a Receipt."""
        long_history = len(messages) > 200
        if report.zone == "normal" and not long_history:
            return None
        if ensure_message_ids(task_id, messages).conflicted_message_ids:
            return None
        partition = partition_work_units(_scoped_messages(task_id, messages), set())
        recent = {unit.unit_id for unit in partition.units[-8:]}
        replacements: dict[str, AnyMessage] = {}
        omitted: set[str] = set()
        history_path = ""
        for unit in partition.units:
            originals = list(messages[unit.start_index:unit.end_index + 1])
            if (unit.state != "complete" or unit.unit_id in recent
                    or any(isinstance(m, (HumanMessage, SystemMessage)) for m in originals)):
                continue
            drop = long_history and unit.end_index < len(messages) - 80
            large = [m for m in originals if isinstance(m, ToolMessage)
                     and isinstance(m.content, str) and len(m.content) > 8000]
            if not drop and not large:
                continue
            attempt = "snip-" + hashlib.sha256(
                "".join(m.model_dump_json() for m in originals).encode("utf-8")
            ).hexdigest()[:32]
            try:
                reference = self.coordinator.adapter.persist_history(
                    task_id, attempt, originals, set(), work_unit_ids={unit.unit_id},
                )
            except ArtifactPersistenceError:
                # An unsuccessful archive cannot justify hiding original content.
                return None
            history_path = reference.path
            if drop:
                omitted.update(unit.message_ids)
            else:
                for message in large:
                    replacements[str(message.id)] = message.model_copy(update={"content": (
                        f"[Archived tool result: {reference.path}; event={attempt}; message={message.id}]\n"
                        + message.content[:2000] + "\n[...snipped; original is recoverable...]\n"
                        + message.content[-1000:]
                    )}, deep=True)
        if not omitted and not replacements:
            return None
        view: list[AnyMessage] = []
        note_added = False
        for message in messages:
            if str(message.id) in omitted:
                if not note_added:
                    view.append(SystemMessage(id=f"snip-view-{task_id}", content=(
                        f"Older complete tool rounds archived at {history_path}. "
                        "Use read_file with offset/limit, or search_diagnostic_artifacts / "
                        "read_diagnostic_artifact to retrieve originals."
                    )))
                    note_added = True
                continue
            view.append(replacements.get(str(message.id), message))
        return view

    def _invoke_zone(
        self,
        prepared: _PreparedModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        request = self._with_observation_hint(prepared)
        if prepared.report.zone in {"normal", "observe"}:
            return handler(request)
        compaction_request = _compaction_request(prepared, request)
        return self.coordinator.invoke_automatic(
            compaction_request,
            lambda messages: handler(request.override(messages=list(messages))),
        )

    async def _ainvoke_zone(
        self,
        prepared: _PreparedModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        request = self._with_observation_hint(prepared)
        if prepared.report.zone in {"normal", "observe"}:
            return await handler(request)
        compaction_request = _compaction_request(prepared, request)
        return await self.coordinator.ainvoke_automatic(
            compaction_request,
            lambda messages: handler(request.override(messages=list(messages))),
        )

    def _recover_overflow(
        self,
        prepared: _PreparedModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        first_overflow: ContextOverflowError,
    ) -> ModelResponse | ExtendedModelResponse:
        overflow_event_id = _prepared_telemetry_event_id(prepared, "overflow")
        self.coordinator.history_repository.record_overflow(
            prepared.task_id,
            event_id=overflow_event_id,
        )
        self.coordinator.history_repository.record_overflow_retry(
            prepared.task_id,
            event_id=f"{overflow_event_id}:retry",
        )
        recovery_request = _overflow_request(prepared)
        try:
            return self.coordinator.invoke_automatic(
                recovery_request,
                lambda messages: handler(
                    prepared.request.override(messages=list(messages))
                ),
            )
        except ContextOverflowError:
            raise self._overflow_exhausted(
                recovery_request,
                first_overflow,
            ) from first_overflow

    async def _arecover_overflow(
        self,
        prepared: _PreparedModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        first_overflow: ContextOverflowError,
    ) -> ModelResponse | ExtendedModelResponse:
        overflow_event_id = _prepared_telemetry_event_id(prepared, "overflow")
        self.coordinator.history_repository.record_overflow(
            prepared.task_id,
            event_id=overflow_event_id,
        )
        self.coordinator.history_repository.record_overflow_retry(
            prepared.task_id,
            event_id=f"{overflow_event_id}:retry",
        )
        recovery_request = _overflow_request(prepared)
        try:
            return await self.coordinator.ainvoke_automatic(
                recovery_request,
                lambda messages: handler(
                    prepared.request.override(messages=list(messages))
                ),
            )
        except ContextOverflowError:
            error = await self._aoverflow_exhausted(
                recovery_request,
                first_overflow,
            )
            raise error from first_overflow

    def _overflow_exhausted(
        self,
        request: CompactionRequest,
        _first_overflow: ContextOverflowError,
    ) -> ContextRecoveryRequired:
        prepared = self.coordinator.prepare(request)
        return ContextRecoveryRequired(
            ContextRecoveryMetadata(
                task_id=request.task_id,
                stage="overflow_retry",
                error_code="context_overflow_after_single_retry",
                usage_ratio=request.budget.usage_ratio,
                active_snapshot_version=(
                    request.active_event.active_snapshot_version
                    if request.active_event
                    else None
                ),
                prepared_snapshot_version=prepared.snapshot.version,
                prepared_snapshot_lifecycle=prepared.snapshot.lifecycle,
                conversation_artifact=prepared.artifact_reference.path,
                original_messages_preserved=True,
            )
        )

    async def _aoverflow_exhausted(
        self,
        request: CompactionRequest,
        _first_overflow: ContextOverflowError,
    ) -> ContextRecoveryRequired:
        prepared = await self.coordinator.aprepare(request)
        return ContextRecoveryRequired(
            ContextRecoveryMetadata(
                task_id=request.task_id,
                stage="overflow_retry",
                error_code="context_overflow_after_single_retry",
                usage_ratio=request.budget.usage_ratio,
                active_snapshot_version=(
                    request.active_event.active_snapshot_version
                    if request.active_event
                    else None
                ),
                prepared_snapshot_version=prepared.snapshot.version,
                prepared_snapshot_lifecycle=prepared.snapshot.lifecycle,
                conversation_artifact=prepared.artifact_reference.path,
                original_messages_preserved=True,
            )
        )

    def _with_observation_hint(
        self,
        prepared: _PreparedModelRequest,
    ) -> ModelRequest:
        return prepared.request

    def _reconcile_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent | None,
        context: ProtectedContext,
        messages: Sequence[AnyMessage],
    ) -> tuple[ProtectedContext, tuple[AnyMessage, ...]]:
        if event is None:
            return context, tuple(messages)
        snapshot = self.coordinator.active_snapshot_from_event(
            task_id,
            event,
        )
        if snapshot is None:
            return replace(context, active_snapshot=None), tuple(messages)
        if snapshot.lifecycle == "prepared":
            snapshot = self.coordinator.activate_from_event(
                task_id,
                event,
            )
        metrics = self.coordinator.history_repository.context_telemetry(task_id)
        if metrics.active_snapshot_version != event.active_snapshot_version:
            self.coordinator.history_repository.record_compaction_outcome(
                task_id,
                event_id=event.event_id,
                snapshot_version=event.active_snapshot_version,
                artifact_path=event.conversation_artifact.path,
                emergency=metrics.latest_budget_zone == "emergency",
            )
        scoped = _scoped_messages(task_id, messages)
        partition = partition_work_units(scoped, set())
        compressed_unit_ids = set(event.conversation_artifact.work_unit_ids)
        compressed_message_ids = {
            message_id
            for unit in partition.units
            if unit.unit_id in compressed_unit_ids
            for message_id in unit.message_ids
        }
        retained = set(event.retained_message_ids)
        effective = tuple(
            message
            for message in messages
            if str(message.id) not in compressed_message_ids
            or str(message.id) in retained
        )
        snapshot_message = _event_snapshot_message(event, snapshot)
        return replace(context, active_snapshot=snapshot), (
            snapshot_message,
            *effective,
        )


@dataclass(frozen=True)
class _PreparedModelRequest:
    task_id: str
    request: ModelRequest
    context: ProtectedContext
    event: DeepFixCompactionEvent | None
    report: ContextBudgetReport


def _compaction_request(
    prepared: _PreparedModelRequest,
    request: ModelRequest,
) -> CompactionRequest:
    return CompactionRequest(
        task_id=prepared.task_id,
        entrypoint="automatic",
        messages=tuple(request.messages),
        active_event=prepared.event,
        protected_context=prepared.context,
        budget=prepared.report,
        model=request.model,
    )


def _overflow_request(prepared: _PreparedModelRequest) -> CompactionRequest:
    report = replace(
        prepared.report,
        zone="emergency",
        target_ratio=0.50,
    )
    return CompactionRequest(
        task_id=prepared.task_id,
        entrypoint="overflow_recovery",
        messages=tuple(prepared.request.messages),
        active_event=prepared.event,
        protected_context=prepared.context,
        budget=report,
        model=prepared.request.model,
    )


def _event_from_state(state: dict[str, Any]) -> DeepFixCompactionEvent | None:
    value = state.get("_deepfix_compaction_event")
    return DeepFixCompactionEvent.model_validate(value) if value else None


def _prepared_telemetry_event_id(
    prepared: _PreparedModelRequest,
    kind: str,
) -> str:
    return _telemetry_event_id(
        prepared.task_id,
        kind,
        prepared.report.request_tokens,
        *(str(message.id) for message in prepared.request.messages),
    )


def _telemetry_event_id(task_id: str, kind: str, *parts: object) -> str:
    material = json.dumps(
        [task_id, kind, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"context_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _model_request_task_id(request: ModelRequest) -> str:
    task_id = _runtime_task_id(request.runtime)
    if not task_id:
        raise ValueError("DeepFixCompactionMiddleware 缺少 thread_id")
    return task_id


def _runtime_task_id(runtime: Runtime | None) -> str:
    execution_info = runtime.execution_info if runtime is not None else None
    return (
        str(execution_info.thread_id).strip()
        if execution_info is not None and execution_info.thread_id
        else ""
    )


def _scoped_messages(
    task_id: str,
    messages: Sequence[AnyMessage],
) -> tuple[AnyMessage, ...]:
    identities = ensure_message_ids(task_id, messages)
    scoped = []
    for message in identities.messages:
        additional = dict(message.additional_kwargs)
        additional["_deepfix_task_id"] = task_id
        scoped.append(
            message.model_copy(
                update={"additional_kwargs": additional},
                deep=True,
            )
        )
    return tuple(scoped)


def _event_snapshot_message(
    event: DeepFixCompactionEvent,
    snapshot: Any,
) -> SystemMessage:
    content = snapshot.model_dump_json()
    if len(content) > 16_000:
        content = (
            content[:15_000]
            + "\n[Snapshot details truncated; full history: "
            + event.conversation_artifact.path
            + "]"
        )
    return SystemMessage(
        id=event.snapshot_message_id,
        content=content,
        additional_kwargs={"_deepfix_snapshot_version": snapshot.version},
    )
