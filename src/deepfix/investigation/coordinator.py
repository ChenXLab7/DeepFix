from __future__ import annotations

from collections.abc import Mapping
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
    InvestigationStateError,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    AgentPhase,
    CheckedFile,
    CheckedLocation,
    InvestigationEventType,
    InvestigationHypothesis,
    InvestigationRecoveryMetadata,
    InvestigationState,
    NewInvestigationEvent,
    RecordHypothesisInput,
    ToolObservation,
)
from deepfix.investigation.phase import PhaseResolver
from deepfix.investigation.progress import ProgressEvaluator
from deepfix.investigation.store import InvestigationStore
from deepfix.persistence import TaskRepository


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
    ) -> None:
        self.store = store
        self.tasks = tasks
        self.compaction_store = compaction_store
        self.evidence_collector = evidence_collector
        self.phase_resolver = phase_resolver or PhaseResolver()
        self.progress_evaluator = progress_evaluator or ProgressEvaluator()

    def ensure_started(self, task_id: str) -> InvestigationState:
        return self.state(task_id)

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
        updated = self._apply_domain_state(evaluated, observation)
        if progress is not None:
            updated = updated.model_copy(
                update={
                    "progress_generation": state.progress_generation + 1,
                    "last_progress_event_id": event_id,
                    "last_progress_at": datetime.now(UTC),
                }
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

    def recovery(
        self,
        task_id: str,
        error_code: str,
        *,
        state: InvestigationState | None = None,
        tool_call_id: str | None = None,
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


def _non_negative_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)
