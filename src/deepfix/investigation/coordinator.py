from __future__ import annotations

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
    CheckedFile,
    CheckedLocation,
    InvestigationCapability,
    InvestigationEventType,
    InvestigationHypothesis,
    InvestigationRecoveryMetadata,
    InvestigationState,
    NewInvestigationEvent,
    ProgressKind,
    RecordHypothesisInput,
    ToolObservation,
)
from deepfix.investigation.progress import ProgressEvaluator
from deepfix.investigation.stagnation import StagnationDetector
from deepfix.investigation.store import (
    InvestigationStateConflict,
    InvestigationStore,
)
from deepfix.persistence import TaskRepository


@dataclass(frozen=True)
class ToolAuthorization:
    allowed: bool
    correction_required: bool = False
    correction_kind: str | None = None
    correction_ref_id: str | None = None


_DIAGNOSTIC_RESULT_EVENTS = {
    (
        "search_diagnostic_artifacts",
        "diagnostic_artifact_search",
    ): InvestigationEventType.ARTIFACT_SEARCHED,
    (
        "read_diagnostic_artifact",
        "diagnostic_artifact_read",
    ): InvestigationEventType.ARTIFACT_READ,
}
_STATE_COMMIT_MAX_ATTEMPTS = 4
_DIAGNOSTIC_PAYLOAD_FIELDS = {
    "diagnostic_artifact_search": (
        "artifact_ids",
        "match_count",
        "searched_artifact_count",
        "omitted_artifact_count",
        "truncated",
        "content_hashes",
        "query_terms_hash",
    ),
    "diagnostic_artifact_read": (
        "artifact_id",
        "kind",
        "start_line",
        "end_line",
        "total_lines",
        "content_hash",
        "truncated",
    ),
}


class InvestigationCoordinator:
    def __init__(
        self,
        *,
        store: InvestigationStore,
        tasks: TaskRepository,
        compaction_store: CompactionStore,
        evidence_collector: EvidenceCollector,
        progress_evaluator: ProgressEvaluator | None = None,
        stagnation_detector: StagnationDetector | None = None,
    ) -> None:
        self.store = store
        self.tasks = tasks
        self.compaction_store = compaction_store
        self.evidence_collector = evidence_collector
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
        return self._filter_memory_tool(state, set(capabilities))

    @staticmethod
    def _filter_memory_tool(
        state: InvestigationState,
        allowed: set[str],
    ) -> set[str]:
        if state.progress_generation in {
            state.memory_save_blocked_generation,
            state.memory_saved_generation,
        }:
            allowed.discard("save_progress")
        return allowed

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
        task = self.tasks.get(task_id)
        name = str(call.get("name", "")).strip()
        call_id = str(call.get("id", "")).strip()
        args_value = call.get("args", {})
        args = args_value if isinstance(args_value, Mapping) else {}
        signature = tool_signature(name, args)
        fingerprint = result_fingerprint(result)
        artifact = result.artifact if isinstance(result.artifact, Mapping) else {}
        tool_payload = {
            "tool_name": name,
            "status": result.status,
            "result_type": str(artifact.get("result_type", "")),
        }
        diagnostic = _diagnostic_artifact_observation(
            name,
            call_id,
            result,
            signature,
            fingerprint,
        )
        if diagnostic is not None:
            return self.record_observation(task_id, diagnostic)
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
                        payload=tool_payload,
                    ),
                )
            has_successful_change = any(
                isinstance(item, FileChangeEvidence) and item.status == "succeeded"
                for item in self.compaction_store.list_evidence(task_id)
            )
            event_type = (
                InvestigationEventType.POST_EDIT_TEST_OBSERVED
                if evidence.timing in {"post_change", "post_recovery"}
                or has_successful_change
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
                    payload=tool_payload,
                ),
            )

        if name == "read_file" and result.status == "success":
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
                        **tool_payload,
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
                    payload=tool_payload,
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
                payload=tool_payload,
            ),
        )

    def record_observation(
        self,
        task_id: str,
        observation: ToolObservation,
    ) -> InvestigationState:
        event_id = self._observation_event_id(task_id, observation)
        recovery_state: InvestigationState | None = None
        for attempt in range(_STATE_COMMIT_MAX_ATTEMPTS):
            state = self.state(task_id)
            try:
                if self.store.has_event(task_id, event_id):
                    return self.state(task_id)
            except InvestigationStateError:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve recovery boundary
                failure = exc
                recovery_state = state
                break
            events, updated = self._prepare_observation_commit(
                task_id,
                state,
                observation,
                event_id,
            )
            try:
                return self.store.commit(state.version, events, updated)
            except InvestigationStateConflict as exc:
                if attempt + 1 < _STATE_COMMIT_MAX_ATTEMPTS:
                    continue
                failure = exc
                try:
                    recovery_state = self.state(task_id)
                except Exception:  # noqa: BLE001 - preserve the commit failure
                    recovery_state = state
                break
            except Exception as exc:  # noqa: BLE001 - preserve recovery boundary
                failure = exc
                recovery_state = state
                break
        raise InvestigationStateError(
            self.recovery(
                task_id,
                "investigation_state_commit_failed",
                state=recovery_state,
                tool_call_id=observation.tool_call_id,
                checkpoint_available=True,
                recovery_action="replay_event_commit_without_model_call",
            )
        ) from failure

    def _prepare_observation_commit(
        self,
        task_id: str,
        state: InvestigationState,
        observation: ToolObservation,
        event_id: str,
    ) -> tuple[list[NewInvestigationEvent], InvestigationState]:
        evaluated, progress = self.progress_evaluator.apply(state, observation)
        if observation.progress_kind is not None:
            progress = observation.progress_kind
        effective_observation = observation.model_copy(
            update={"progress_kind": progress}
        )
        origin = NewInvestigationEvent(
            event_id=event_id,
            task_id=task_id,
            event_type=observation.event_type,
            source_message_id=observation.source_message_id,
            tool_call_id=observation.tool_call_id,
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
        if observation.payload.get("tool_name") == "save_progress":
            if observation.payload.get("status") == "success":
                updated = updated.model_copy(
                    update={
                        "memory_save_failure_count": 0,
                        "memory_save_blocked_generation": None,
                        "memory_saved_generation": updated.progress_generation,
                    }
                )
            else:
                failures = state.memory_save_failure_count + 1
                updated = updated.model_copy(
                    update={
                        "memory_save_failure_count": failures,
                        "memory_save_blocked_generation": (
                            updated.progress_generation if failures >= 2 else None
                        ),
                    }
                )
        if observation.event_type is InvestigationEventType.TEST_OBSERVED:
            diagnostic_tests = (
                state.diagnostic_test_count_since_decision + 1
                if observation.exit_code != 0
                else 0
            )
            updated = updated.model_copy(
                update={
                    "diagnostic_test_count_since_decision": diagnostic_tests,
                    "diagnostic_decision_required": diagnostic_tests >= 2,
                    "decision_correction_used": False,
                }
            )
        if (
            observation.event_type is InvestigationEventType.POST_EDIT_TEST_OBSERVED
            and observation.exit_code != 0
        ):
            updated = updated.model_copy(
                update={
                    "diagnostic_decision_required": True,
                    "diagnostic_test_count_since_decision": 0,
                    "repair_reevaluation_required": True,
                    "decision_correction_used": False,
                    "reevaluation_required": False,
                    "stagnation_level": 0,
                }
            )
        elif observation.event_type is InvestigationEventType.ARTIFACT_READ:
            updated = updated.model_copy(
                update={
                    "diagnostic_decision_required": True,
                    "decision_correction_used": False,
                    "reevaluation_required": False,
                    "stagnation_level": 0,
                }
            )
        elif observation.event_type is InvestigationEventType.HYPOTHESIS_RECORDED:
            if state.diagnostic_decision_required:
                updated = updated.model_copy(
                    update={
                        "diagnostic_decision_required": True,
                        "diagnostic_test_count_since_decision": 0,
                        "decision_correction_used": False,
                    }
                )
        elif observation.event_type in {
            InvestigationEventType.HYPOTHESIS_REJECTED,
            InvestigationEventType.HYPOTHESIS_SUPPORTED,
        }:
            updated = updated.model_copy(
                update={
                    "diagnostic_decision_required": False,
                    "diagnostic_test_count_since_decision": 0,
                    "repair_reevaluation_required": False,
                    "decision_correction_used": False,
                }
            )
        if (
            observation.payload.get("tool_name") == "execute"
            and observation.payload.get("result_type")
            != "duplicate_execute_correction"
        ):
            updated = updated.model_copy(
                update={
                    "last_execute_signature": observation.signature,
                    "last_execute_generation": updated.progress_generation,
                    "duplicate_execute_correction_signature": None,
                }
            )
        return [origin], updated

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
        semantic_statement = _semantic_statement(command.statement)
        hypothesis_id = command.hypothesis_id or (
            stable_investigation_id(
                "hyp",
                task_id,
                command.evidence_ids[0],
                semantic_statement,
            )
            if command.evidence_ids
            else _candidate_hypothesis_id(task_id, semantic_statement)
        )
        existing = self._hypothesis(state, hypothesis_id)
        if command.hypothesis_id and existing is None:
            raise ValueError("hypothesis_id 不存在于当前任务")
        if (
            existing is not None
            and _semantic_statement(existing.statement) != semantic_statement
        ):
            raise ValueError("hypothesis_id 不能更换假设陈述")
        if (
            state.repair_reevaluation_required
            and existing is not None
            and existing.state == "supported"
            and command.target_state == "supported"
        ):
            raise ValueError(
                "修改后验证失败，不能再次支持同一假设；请先排除旧假设或建立新假设"
            )
        record = InvestigationHypothesis(
            hypothesis_id=hypothesis_id,
            statement=semantic_statement,
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
        state = self.state(task_id)
        updated = state.model_copy(
            update={
                "progress_generation": state.progress_generation + 1,
                "no_progress_count": 0,
                "exploratory_without_progress": 0,
                "reevaluation_required": False,
                "stagnation_level": 0,
                "diagnostic_decision_required": False,
                "diagnostic_test_count_since_decision": 0,
                "repair_reevaluation_required": False,
                "decision_correction_used": False,
            }
        )
        return self._record_lifecycle(
            state,
            updated,
            InvestigationEventType.USER_INFORMATION_RECEIVED,
            source_message_id,
            progress_kind=ProgressKind.USER_INFORMATION,
        )

    def record_needs_input(self, task_id: str, source_id: str) -> InvestigationState:
        state = self.state(task_id)
        return self._record_lifecycle(
            state,
            state,
            InvestigationEventType.NEEDS_INPUT,
            source_id,
        )

    def record_paused(self, task_id: str, reason: str) -> InvestigationState:
        state = self.state(task_id)
        return self._record_lifecycle(
            state,
            state,
            InvestigationEventType.TASK_PAUSED,
            reason,
        )

    def record_resumed(self, task_id: str, source_id: str) -> InvestigationState:
        state = self.state(task_id)
        return self._record_lifecycle(
            state,
            state,
            InvestigationEventType.TASK_RESUMED,
            source_id,
        )

    def _record_lifecycle(
        self,
        state: InvestigationState,
        updated: InvestigationState,
        event_type: InvestigationEventType,
        source_id: str,
        *,
        progress_kind: ProgressKind | None = None,
    ) -> InvestigationState:
        event = NewInvestigationEvent(
            event_id=stable_investigation_id(
                "event",
                state.task_id,
                "lifecycle",
                event_type.value,
                source_id,
                str(state.version),
            ),
            task_id=state.task_id,
            event_type=event_type,
            source_message_id=source_id,
            progress_kind=progress_kind,
        )
        try:
            return self.store.commit(state.version, [event], updated)
        except Exception as exc:
            raise InvestigationStateError(
                self.recovery(
                    state.task_id,
                    "investigation_lifecycle_commit_failed",
                    state=state,
                    checkpoint_available=True,
                    recovery_action="retry_lifecycle_commit_without_model_call",
                )
            ) from exc

    def authorize_tool(
        self,
        task_id: str,
        tool_name: str,
        arguments: Mapping[str, object],
        *,
        tool_call_id: str | None = None,
    ) -> ToolAuthorization:
        state = self.state(task_id)
        normalized_tool = tool_name.strip().lower()
        signature = tool_signature(normalized_tool, arguments)
        duplicate_hypothesis_id = _duplicate_candidate_id(
            task_id,
            normalized_tool,
            arguments,
            state,
        )
        if duplicate_hypothesis_id is not None:
            if duplicate_hypothesis_id in state.duplicate_hypothesis_correction_ids:
                raise InvestigationStagnationError(
                    self.recovery(
                        task_id,
                        "duplicate_hypothesis_ignored",
                        state=state,
                        tool_call_id=tool_call_id,
                        checkpoint_available=True,
                        recovery_action="pause_and_change_investigation_action",
                    )
                )
            source_id = tool_call_id or stable_investigation_id(
                "duplicate-hypothesis-correction",
                task_id,
                duplicate_hypothesis_id,
            )
            self._record_lifecycle(
                state,
                state.model_copy(
                    update={
                        "duplicate_hypothesis_correction_ids": list(
                            dict.fromkeys(
                                [
                                    *state.duplicate_hypothesis_correction_ids,
                                    duplicate_hypothesis_id,
                                ]
                            )
                        )[-16:]
                    }
                ),
                InvestigationEventType.REEVALUATION_REQUIRED,
                source_id,
            )
            return ToolAuthorization(
                allowed=False,
                correction_required=True,
                correction_kind="duplicate_hypothesis",
                correction_ref_id=duplicate_hypothesis_id,
            )
        if (
            normalized_tool == "execute"
            and state.last_execute_signature == signature
            and state.last_execute_generation == state.progress_generation
        ):
            if state.duplicate_execute_correction_signature == signature:
                raise InvestigationStagnationError(
                    self.recovery(
                        task_id,
                        "duplicate_execute_ignored",
                        state=state,
                        tool_call_id=tool_call_id,
                        checkpoint_available=True,
                        recovery_action="pause_and_change_investigation_action",
                    )
                )
            source_id = tool_call_id or stable_investigation_id(
                "duplicate-execute-correction",
                task_id,
                signature,
            )
            self._record_lifecycle(
                state,
                state.model_copy(
                    update={"duplicate_execute_correction_signature": signature}
                ),
                InvestigationEventType.REEVALUATION_REQUIRED,
                source_id,
            )
            return ToolAuthorization(
                allowed=False,
                correction_required=True,
                correction_kind="duplicate_execute",
            )
        # Navigation checkpoints are advisory. Approval, workspace confinement,
        # Receipt/Journal recovery, experiment scope, and duplicate-side-effect
        # checks remain authoritative; legacy phase/stagnation state cannot gate tools.
        return ToolAuthorization(allowed=True)

    def recovery(
        self,
        task_id: str,
        error_code: str,
        *,
        state: InvestigationState | None = None,
        tool_call_id: str | None = None,
        checkpoint_available: bool,
        recovery_action: str,
        error_type: str | None = None,
        error_detail: str | None = None,
        error_fingerprint: str | None = None,
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
            state_version=current.version if current else 0,
            last_event_sequence=sequence,
            tool_call_id=tool_call_id,
            checkpoint_available=checkpoint_available,
            recovery_action=recovery_action,
            error_type=error_type,
            error_detail=error_detail,
            error_fingerprint=error_fingerprint,
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
        hypothesis_events = {
            InvestigationEventType.HYPOTHESIS_RECORDED,
            InvestigationEventType.HYPOTHESIS_REJECTED,
            InvestigationEventType.HYPOTHESIS_SUPPORTED,
        }
        identity = (
            observation.hypothesis_id
            if observation.event_type in hypothesis_events
            else observation.tool_call_id
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
        if observation.event_type in {
            InvestigationEventType.ARTIFACT_SEARCHED,
            InvestigationEventType.ARTIFACT_READ,
        }:
            return payload
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


def _semantic_statement(value: str) -> str:
    return " ".join(value.split())


def _candidate_hypothesis_id(task_id: str, statement: str) -> str:
    return stable_investigation_id(
        "hyp",
        task_id,
        "candidate-statement",
        _semantic_statement(statement),
    )


def _duplicate_candidate_id(
    task_id: str,
    tool_name: str,
    arguments: Mapping[str, object],
    state: InvestigationState,
) -> str | None:
    if (
        tool_name != "record_hypothesis"
        or str(arguments.get("target_state", "")) != "candidate"
        or str(arguments.get("hypothesis_id", "")).strip()
    ):
        return None
    statement = _semantic_statement(str(arguments.get("statement", "")))
    if not statement:
        return None
    hypothesis_id = _candidate_hypothesis_id(task_id, statement)
    return (
        hypothesis_id
        if any(item.hypothesis_id == hypothesis_id for item in state.hypotheses)
        else None
    )


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


def _diagnostic_artifact_observation(
    tool_name: str,
    call_id: str,
    result: ToolMessage,
    signature: str,
    fingerprint: str,
) -> ToolObservation | None:
    artifact = result.artifact if isinstance(result.artifact, Mapping) else {}
    result_type = str(artifact.get("result_type", ""))
    event_type = _DIAGNOSTIC_RESULT_EVENTS.get((tool_name, result_type))
    if result.status != "success" or event_type is None:
        return None
    payload = {
        key: artifact[key]
        for key in _DIAGNOSTIC_PAYLOAD_FIELDS[result_type]
        if key in artifact
    }
    return ToolObservation(
        event_type=event_type,
        tool_call_id=call_id or None,
        source_message_id=str(result.id or "") or None,
        signature=signature,
        result_fingerprint=fingerprint,
        payload=payload,
    )


def _normalized_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("/")
    if not normalized:
        raise ValueError("文件路径不能为空")
    return str(PurePosixPath(normalized))


def _target_from_arguments(
    tool_name: str,
    arguments: Mapping[str, object],
) -> str:
    if tool_name == "search_diagnostic_artifacts":
        return " ".join(str(arguments.get("query", "")).split())
    if tool_name == "read_diagnostic_artifact":
        return str(arguments.get("artifact_id", "")).strip()
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


def _non_negative_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)
