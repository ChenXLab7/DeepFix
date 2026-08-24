from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from langchain_core.messages import ToolMessage

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.investigation.classification import (
    is_pytest_verification,
    result_fingerprint,
    tool_signature,
)
from deepfix.investigation.errors import (
    InvestigationCoordinationError,
    InvestigationStagnationError,
    InvestigationStateError,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    AgentPhase,
    CheckedFile,
    CheckedLocation,
    ContinueInvestigationInput,
    InvestigationCapability,
    InvestigationEventType,
    InvestigationHypothesis,
    InvestigationPermit,
    InvestigationRecoveryMetadata,
    InvestigationState,
    NewInvestigationEvent,
    RecordHypothesisInput,
    ToolObservation,
)
from deepfix.investigation.phase import PhaseResolver
from deepfix.investigation.progress import ProgressEvaluator
from deepfix.investigation.stagnation import StagnationDetector
from deepfix.investigation.store import InvestigationStore
from deepfix.persistence import TaskRepository


@dataclass(frozen=True)
class ToolAuthorization:
    allowed: bool
    permit_id: str | None = None


_PHASE_CAPABILITIES: dict[AgentPhase, frozenset[InvestigationCapability]] = {
    AgentPhase.CLARIFYING: frozenset(
        {
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
    AgentPhase.INVESTIGATING: frozenset(
        {
            InvestigationCapability.READ,
            InvestigationCapability.SEARCH,
            InvestigationCapability.EXECUTE,
            InvestigationCapability.RESEARCH,
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
    AgentPhase.DIAGNOSING: frozenset(
        {
            InvestigationCapability.READ,
            InvestigationCapability.SEARCH,
            InvestigationCapability.EXECUTE,
            InvestigationCapability.RESEARCH,
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
    AgentPhase.PLANNING: frozenset(InvestigationCapability),
    AgentPhase.EDITING: frozenset(
        {
            InvestigationCapability.READ,
            InvestigationCapability.SEARCH,
            InvestigationCapability.EXECUTE,
            InvestigationCapability.MODIFY,
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
    AgentPhase.TESTING: frozenset(
        {
            InvestigationCapability.READ,
            InvestigationCapability.SEARCH,
            InvestigationCapability.EXECUTE,
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
    AgentPhase.REVIEWING: frozenset(
        {
            InvestigationCapability.READ,
            InvestigationCapability.SEARCH,
            InvestigationCapability.EXECUTE,
            InvestigationCapability.META,
            InvestigationCapability.MEMORY,
            InvestigationCapability.COMPACTION,
        }
    ),
}
_REEVALUATION_TOOL_NAMES = frozenset(
    {"record_hypothesis", "continue_investigation", "save_progress"}
)


class InvestigationCoordinator:
    def __init__(
        self,
        *,
        store: InvestigationStore,
        tasks: TaskRepository,
        compaction_store: CompactionStore,
        evidence_collector: EvidenceCollector,
        phase_resolver: PhaseResolver | None = None,
        progress_evaluator: ProgressEvaluator | None = None,
        stagnation_detector: StagnationDetector | None = None,
    ) -> None:
        self.store = store
        self.tasks = tasks
        self.compaction_store = compaction_store
        self.evidence_collector = evidence_collector
        self.phase_resolver = phase_resolver or PhaseResolver()
        self.progress_evaluator = progress_evaluator or ProgressEvaluator()
        self.stagnation_detector = stagnation_detector or StagnationDetector()

    def ensure_started(self, task_id: str) -> InvestigationState:
        return self.state(task_id)

    def project_python(self, task_id: str) -> str:
        return self.tasks.get(task_id).project_python

    def allowed_tool_names(
        self,
        state: InvestigationState,
        capabilities: Mapping[str, InvestigationCapability],
    ) -> set[str]:
        if state.stagnation_level > 0 or state.reevaluation_required:
            return set(_REEVALUATION_TOOL_NAMES) & set(capabilities)
        allowed_capabilities = _PHASE_CAPABILITIES[state.agent_phase]
        return {
            name
            for name, capability in capabilities.items()
            if capability in allowed_capabilities
        }

    def state(self, task_id: str) -> InvestigationState:
        try:
            self.tasks.get(task_id)
            return self.store.load(task_id) or self.store.ensure_started(task_id)
        except InvestigationCoordinationError:
            raise
        except Exception as exc:
            raise InvestigationStateError(
                self.recovery(
                    task_id,
                    "investigation_state_read_failed",
                    checkpoint_available=True,
                    recovery_action="retry_state_read_without_model_call",
                )
            ) from exc

    def record_tool_result(
        self,
        task_id: str,
        call: Mapping[str, object],
        result: ToolMessage,
    ) -> InvestigationState:
        state = self.state(task_id)
        task = self.tasks.get(task_id)
        name = str(call.get("name", "")).strip()
        call_id = str(call.get("id", "")).strip()
        args_value = call.get("args", {})
        args = args_value if isinstance(args_value, Mapping) else {}
        signature = tool_signature(name, args)
        fingerprint = result_fingerprint(result)
        evidence = self.evidence_collector.collect_pair(task_id, call, result, task)

        if name == "execute" and is_pytest_verification(
            str(args.get("command", "")),
            task.project_python,
        ):
            if not isinstance(evidence, SystemTestEvidence):
                return self.record_observation(
                    task_id,
                    self._tool_observation(
                        InvestigationEventType.TOOL_COMPLETED,
                        call_id,
                        result,
                        signature,
                        fingerprint,
                    ),
                )
            if state.agent_phase is AgentPhase.EDITING:
                state = self.record_observation(
                    task_id,
                    self._tool_observation(
                        InvestigationEventType.VERIFICATION_EXECUTION_OBSERVED,
                        call_id,
                        result,
                        signature,
                        fingerprint,
                    ),
                )
            event_type = (
                InvestigationEventType.POST_EDIT_TEST_OBSERVED
                if state.agent_phase is AgentPhase.TESTING
                else InvestigationEventType.TEST_OBSERVED
            )
            return self.record_observation(
                task_id,
                self._tool_observation(
                    event_type,
                    call_id,
                    result,
                    signature,
                    fingerprint,
                    evidence_id=evidence.evidence_id,
                    exit_code=evidence.exit_code,
                ),
            )

        if name == "read_file":
            path = _normalized_path(str(args.get("file_path", "")))
            offset = _non_negative_int(args.get("offset"), default=0)
            line_count = max(1, len(_message_text(result).splitlines()))
            return self.record_observation(
                task_id,
                self._tool_observation(
                    InvestigationEventType.FILE_CHECKED,
                    call_id,
                    result,
                    signature,
                    fingerprint,
                    path=path,
                    payload={
                        "start_line": offset + 1,
                        "end_line": offset + line_count,
                        "content_fingerprint": fingerprint,
                    },
                ),
            )

        if isinstance(evidence, FileChangeEvidence):
            event_type = (
                InvestigationEventType.FILE_CHANGED
                if evidence.status == "succeeded"
                else InvestigationEventType.FILE_CHANGE_FAILED
            )
            return self.record_observation(
                task_id,
                self._tool_observation(
                    event_type,
                    call_id,
                    result,
                    signature,
                    fingerprint,
                    evidence_id=evidence.evidence_id,
                    path=evidence.path,
                ),
            )

        return self.record_observation(
            task_id,
            self._tool_observation(
                InvestigationEventType.TOOL_COMPLETED,
                call_id,
                result,
                signature,
                fingerprint,
            ),
        )

    def record_observation(
        self,
        task_id: str,
        observation: ToolObservation,
    ) -> InvestigationState:
        state = self.state(task_id)
        evaluated, progress = self.progress_evaluator.apply(state, observation)
        if observation.progress_kind is not None:
            progress = observation.progress_kind
        effective_observation = observation.model_copy(
            update={"progress_kind": progress}
        )
        event_id = self._observation_event_id(task_id, observation)
        if self.store.has_event(task_id, event_id):
            return state
        resolution = self.phase_resolver.transition(
            task_id,
            state.agent_phase,
            observation,
        )
        origin = NewInvestigationEvent(
            event_id=event_id,
            task_id=task_id,
            event_type=observation.event_type,
            source_message_id=observation.source_message_id,
            tool_call_id=observation.tool_call_id,
            phase_before=state.agent_phase,
            phase_after=resolution.phase,
            progress_kind=progress,
            payload=self._event_payload(observation),
        )
        updated = self._apply_domain_state(evaluated, effective_observation)
        if progress is not None:
            updated = self.stagnation_detector.after_event(
                updated,
                effective_observation,
            )
            updated = updated.model_copy(
                update={
                    "last_progress_event_id": event_id,
                    "last_progress_at": datetime.now(UTC),
                }
            )
        else:
            updated = self.stagnation_detector.after_tool(
                updated,
                effective_observation,
            )
        updated = updated.model_copy(update={"agent_phase": resolution.phase})
        events = [origin]
        if resolution.phase_event is not None:
            events.append(resolution.phase_event)
        try:
            return self.store.commit(state.version, events, updated)
        except Exception as exc:
            raise InvestigationStateError(
                self.recovery(
                    task_id,
                    "investigation_state_commit_failed",
                    state=state,
                    tool_call_id=observation.tool_call_id,
                    checkpoint_available=True,
                    recovery_action="replay_event_commit_without_model_call",
                )
            ) from exc

    def record_hypothesis(
        self,
        task_id: str,
        command: RecordHypothesisInput,
        *,
        source_id: str,
    ) -> InvestigationHypothesis:
        state = self.state(task_id)
        evidence_ids = {
            item.evidence_id
            for item in self.compaction_store.list_evidence(task_id)
        }
        if not set(command.evidence_ids) <= evidence_ids:
            raise ValueError("假设证据不属于当前任务")
        if not all(
            self._location_was_checked(state, item)
            for item in command.checked_locations
        ):
            raise ValueError("假设位置尚未被当前任务检查")
        first_source = command.evidence_ids[0] if command.evidence_ids else source_id
        hypothesis_id = command.hypothesis_id or stable_investigation_id(
            "hyp",
            task_id,
            first_source,
            command.statement,
        )
        existing = self._hypothesis(state, hypothesis_id)
        if command.hypothesis_id and existing is None:
            raise ValueError("hypothesis_id 不存在于当前任务")
        if existing is not None and existing.statement != command.statement:
            raise ValueError("hypothesis_id 不能更换假设陈述")
        record = InvestigationHypothesis(
            hypothesis_id=hypothesis_id,
            statement=command.statement,
            state={
                "candidate": "candidate",
                "rejected": "rejected",
                "supported": "supported",
            }[command.target_state],
            evidence_ids=command.evidence_ids,
            checked_locations=command.checked_locations,
            proposed_change=command.proposed_change,
            expected_effect=command.expected_effect,
            reason=command.reason,
        )
        event_type = {
            "candidate": InvestigationEventType.HYPOTHESIS_RECORDED,
            "rejected": InvestigationEventType.HYPOTHESIS_REJECTED,
            "supported": InvestigationEventType.HYPOTHESIS_SUPPORTED,
        }[command.target_state]
        observation = ToolObservation(
            event_type=event_type,
            tool_call_id=source_id,
            hypothesis_id=hypothesis_id,
            result_fingerprint=stable_investigation_id(
                "result",
                task_id,
                hypothesis_id,
                command.target_state,
            ),
            payload={"hypothesis": record.model_dump(mode="json")},
        )
        updated = self.record_observation(task_id, observation)
        return next(
            item for item in updated.hypotheses if item.hypothesis_id == hypothesis_id
        )

    def record_user_information(
        self,
        task_id: str,
        source_message_id: str,
    ) -> InvestigationState:
        return self.record_observation(
            task_id,
            ToolObservation(
                event_type=InvestigationEventType.USER_INFORMATION_RECEIVED,
                source_message_id=source_message_id,
                result_fingerprint=stable_investigation_id(
                    "result", task_id, "user-information", source_message_id
                ),
            ),
        )

    def grant_investigation_permit(
        self,
        task_id: str,
        command: ContinueInvestigationInput,
    ) -> InvestigationPermit:
        state = self.state(task_id)
        known_hypotheses = {item.hypothesis_id for item in state.hypotheses}
        if not command.hypothesis_ids or not set(command.hypothesis_ids) <= known_hypotheses:
            raise ValueError("continue_investigation 必须引用当前任务假设")
        if state.stagnation_level != 1 or not state.reevaluation_required:
            raise ValueError("当前任务不需要额外调查许可")
        target_hash = _permit_target_hash(command.tool_name, command.target)
        permit_id = stable_investigation_id(
            "permit",
            task_id,
            str(state.progress_generation),
            *sorted(command.hypothesis_ids),
            command.tool_name.lower(),
            target_hash,
        )
        if state.permit is not None:
            if state.permit.permit_id == permit_id:
                return state.permit
            raise ValueError("当前进展代次已经发放调查许可")
        permit = InvestigationPermit(
            permit_id=permit_id,
            tool_name=command.tool_name.strip().lower(),
            target_hash=target_hash,
            granted_in_generation=state.progress_generation,
        )
        event = NewInvestigationEvent(
            event_id=stable_investigation_id(
                "event", task_id, "permit-granted", permit_id
            ),
            task_id=task_id,
            event_type=InvestigationEventType.INVESTIGATION_PERMIT_GRANTED,
            phase_before=state.agent_phase,
            phase_after=state.agent_phase,
            payload={
                "permit_id": permit_id,
                "hypothesis_ids": command.hypothesis_ids,
                "tool_name": permit.tool_name,
                "target_hash": target_hash,
            },
        )
        updated = state.model_copy(update={"permit": permit})
        try:
            committed = self.store.commit(state.version, [event], updated)
        except Exception as exc:
            raise InvestigationStateError(
                self.recovery(
                    task_id,
                    "investigation_permit_commit_failed",
                    state=state,
                    checkpoint_available=True,
                    recovery_action="retry_permit_commit_without_model_call",
                )
            ) from exc
        if committed.permit is None:
            raise InvestigationStateError(
                self.recovery(
                    task_id,
                    "investigation_permit_missing_after_commit",
                    state=committed,
                    checkpoint_available=True,
                    recovery_action="reload_investigation_state",
                )
            )
        return committed.permit

    def authorize_tool(
        self,
        task_id: str,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> ToolAuthorization:
        state = self.state(task_id)
        normalized_tool = tool_name.strip().lower()
        if state.stagnation_level == 0 and not state.reevaluation_required:
            return ToolAuthorization(allowed=True)
        if normalized_tool in {
            "record_hypothesis",
            "continue_investigation",
            "save_progress",
        }:
            return ToolAuthorization(allowed=True)
        permit = state.permit
        target = _target_from_arguments(normalized_tool, arguments)
        target_hash = _permit_target_hash(normalized_tool, target)
        if (
            state.stagnation_level == 1
            and permit is not None
            and not permit.consumed
            and permit.tool_name == normalized_tool
            and permit.target_hash == target_hash
        ):
            consumed = permit.model_copy(update={"consumed": True})
            event = NewInvestigationEvent(
                event_id=stable_investigation_id(
                    "event", task_id, "permit-consumed", permit.permit_id
                ),
                task_id=task_id,
                event_type=InvestigationEventType.INVESTIGATION_INTENT_RECORDED,
                phase_before=state.agent_phase,
                phase_after=state.agent_phase,
                payload={"permit_id": permit.permit_id},
            )
            updated = state.model_copy(
                update={
                    "permit": consumed,
                    "post_permit_review_pending": True,
                }
            )
            try:
                self.store.commit(state.version, [event], updated)
            except Exception as exc:
                raise InvestigationStateError(
                    self.recovery(
                        task_id,
                        "investigation_permit_consume_failed",
                        state=state,
                        permit_id=permit.permit_id,
                        checkpoint_available=True,
                        recovery_action="retry_authorization_without_tool_execution",
                    )
                ) from exc
            return ToolAuthorization(allowed=True, permit_id=permit.permit_id)
        raise InvestigationStagnationError(
            self.recovery(
                task_id,
                "investigation_stagnated",
                state=state,
                permit_id=permit.permit_id if permit else None,
                checkpoint_available=True,
                recovery_action="pause_and_request_hypothesis_reevaluation",
            )
        )

    def recovery(
        self,
        task_id: str,
        error_code: str,
        *,
        state: InvestigationState | None = None,
        tool_call_id: str | None = None,
        permit_id: str | None = None,
        checkpoint_available: bool,
        recovery_action: str,
    ) -> InvestigationRecoveryMetadata:
        current = state
        if current is None:
            try:
                current = self.store.load(task_id)
            except Exception:  # noqa: BLE001 - recovery must preserve the original failure
                current = None
        try:
            sequence = self.store.last_sequence(task_id)
        except Exception:  # noqa: BLE001 - recovery must preserve the original failure
            sequence = 0
        return InvestigationRecoveryMetadata(
            task_id=task_id,
            error_code=error_code,
            agent_phase=(
                current.agent_phase if current else AgentPhase.INVESTIGATING
            ),
            state_version=current.version if current else 0,
            last_event_sequence=sequence,
            tool_call_id=tool_call_id,
            permit_id=permit_id,
            checkpoint_available=checkpoint_available,
            recovery_action=recovery_action,
        )

    @staticmethod
    def _hypothesis(
        state: InvestigationState,
        hypothesis_id: str,
    ) -> InvestigationHypothesis | None:
        return next(
            (
                item
                for item in state.hypotheses
                if item.hypothesis_id == hypothesis_id
            ),
            None,
        )

    @staticmethod
    def _location_was_checked(
        state: InvestigationState,
        location: CheckedLocation,
    ) -> bool:
        path = _normalized_path(location.path)
        return any(
            _normalized_path(item.path) == path
            and any(
                checked.start_line <= location.start_line
                and checked.end_line >= location.end_line
                for checked in item.ranges
            )
            for item in state.checked_files
        )

    @staticmethod
    def _tool_observation(
        event_type: InvestigationEventType,
        call_id: str,
        result: ToolMessage,
        signature: str,
        fingerprint: str,
        **updates: Any,
    ) -> ToolObservation:
        return ToolObservation(
            event_type=event_type,
            tool_call_id=call_id or None,
            source_message_id=str(result.id or "") or None,
            signature=signature,
            result_fingerprint=fingerprint,
            **updates,
        )

    @staticmethod
    def _observation_event_id(
        task_id: str,
        observation: ToolObservation,
    ) -> str:
        identity = (
            observation.tool_call_id
            or observation.source_message_id
            or observation.hypothesis_id
            or observation.signature
            or "command"
        )
        result = observation.result_fingerprint or "no-result"
        return stable_investigation_id(
            "event",
            task_id,
            observation.event_type.value,
            identity,
            result,
        )

    @staticmethod
    def _event_payload(observation: ToolObservation) -> dict[str, Any]:
        payload = dict(observation.payload)
        for key, value in {
            "signature": observation.signature or None,
            "result_fingerprint": observation.result_fingerprint or None,
            "scope": observation.scope.value,
            "evidence_id": observation.evidence_id,
            "hypothesis_id": observation.hypothesis_id,
            "path": observation.path,
            "exit_code": observation.exit_code,
        }.items():
            if value is not None:
                payload[key] = value
        return payload

    @staticmethod
    def _apply_domain_state(
        state: InvestigationState,
        observation: ToolObservation,
    ) -> InvestigationState:
        updates: dict[str, Any] = {}
        if observation.event_type is InvestigationEventType.FILE_CHECKED:
            checked = _merge_checked_file(state, observation)
            updates["checked_files"] = checked
        if observation.event_type in {
            InvestigationEventType.HYPOTHESIS_RECORDED,
            InvestigationEventType.HYPOTHESIS_REJECTED,
            InvestigationEventType.HYPOTHESIS_SUPPORTED,
        }:
            raw = observation.payload.get("hypothesis")
            record = InvestigationHypothesis.model_validate(raw)
            hypotheses = [
                item
                for item in state.hypotheses
                if item.hypothesis_id != record.hypothesis_id
            ]
            hypotheses.append(record)
            updates["hypotheses"] = hypotheses[-64:]
            updates["supported_hypothesis_ids"] = [
                item.hypothesis_id
                for item in hypotheses
                if item.state == "supported"
            ][-64:]
        if (
            observation.event_type
            in {
                InvestigationEventType.TEST_OBSERVED,
                InvestigationEventType.POST_EDIT_TEST_OBSERVED,
            }
            and observation.evidence_id
        ):
            updates["test_evidence_ids"] = list(
                dict.fromkeys(
                    [*state.test_evidence_ids, observation.evidence_id]
                )
            )[-64:]
        return state.model_copy(update=updates) if updates else state


def _merge_checked_file(
    state: InvestigationState,
    observation: ToolObservation,
) -> list[CheckedFile]:
    path = _normalized_path(observation.path or "")
    start = int(observation.payload["start_line"])
    end = int(observation.payload["end_line"])
    location = CheckedLocation(path=path, start_line=start, end_line=end)
    event_id = InvestigationCoordinator._observation_event_id(
        state.task_id,
        observation,
    )
    existing = next(
        (item for item in state.checked_files if _normalized_path(item.path) == path),
        None,
    )
    if existing is None:
        item = CheckedFile(
            path=path,
            content_fingerprint=str(observation.payload["content_fingerprint"]),
            ranges=[location],
            scope=observation.scope,
            first_event_id=event_id,
            latest_event_id=event_id,
        )
        return [*state.checked_files, item][-64:]
    ranges = [*existing.ranges, location]
    replacement = existing.model_copy(
        update={
            "content_fingerprint": str(
                observation.payload["content_fingerprint"]
            ),
            "ranges": ranges,
            "latest_event_id": event_id,
        }
    )
    return [
        replacement if item is existing else item for item in state.checked_files
    ]


def _normalized_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("/")
    if not normalized:
        raise ValueError("文件路径不能为空")
    return str(PurePosixPath(normalized))


def _target_from_arguments(
    tool_name: str,
    arguments: Mapping[str, object],
) -> str:
    if tool_name in {"grep", "search"}:
        path = str(arguments.get("path") or arguments.get("file_path") or "")
        pattern = str(arguments.get("pattern") or arguments.get("query") or "").strip()
        return f"{_normalized_path(path)}|{pattern}"
    if tool_name == "execute":
        return " ".join(str(arguments.get("command", "")).split())
    if tool_name == "read_file":
        return _normalized_path(str(arguments.get("file_path", "")))
    target = arguments.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _permit_target_hash(tool_name: str, target: str) -> str:
    normalized_tool = tool_name.strip().lower()
    normalized_target = target.strip().replace("\\", "/").lstrip("/")
    if not normalized_tool or not normalized_target:
        raise ValueError("调查许可的工具和目标不能为空")
    material = f"{normalized_tool}|{normalized_target}"
    return f"target_{hashlib.sha256(material.encode()).hexdigest()[:32]}"


def _non_negative_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)
