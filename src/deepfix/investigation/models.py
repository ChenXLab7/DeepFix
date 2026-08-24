from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from deepfix.compaction.models import StrictModel


class AgentPhase(StrEnum):
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    EDITING = "editing"
    TESTING = "testing"
    REVIEWING = "reviewing"


class ProgressKind(StrEnum):
    TEST_EVIDENCE = "test_evidence"
    DECISION_EVIDENCE = "decision_evidence"
    HYPOTHESIS_TRANSITION = "hypothesis_transition"
    FILE_CHANGE = "file_change"
    POST_EDIT_TEST = "post_edit_test"
    USER_INFORMATION = "user_information"


class ScopeKind(StrEnum):
    DIRECT = "direct"
    DEPENDENCY = "dependency"
    EXPLORATORY = "exploratory"


class InvestigationCapability(StrEnum):
    READ = "read"
    SEARCH = "search"
    EXECUTE = "execute"
    MODIFY = "modify"
    RESEARCH = "research"
    MEMORY = "memory"
    COMPACTION = "compaction"
    META = "meta"


class InvestigationEventType(StrEnum):
    TASK_STARTED = "task_started"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    USER_INFORMATION_RECEIVED = "user_information_received"
    NEEDS_INPUT = "needs_input"
    TOOL_COMPLETED = "tool_completed"
    ARTIFACT_SEARCHED = "artifact_searched"
    ARTIFACT_READ = "artifact_read"
    TEST_OBSERVED = "test_observed"
    FILE_CHECKED = "file_checked"
    FILE_CHANGED = "file_changed"
    FILE_CHANGE_FAILED = "file_change_failed"
    DECISION_EVIDENCE_OBSERVED = "decision_evidence_observed"
    HYPOTHESIS_RECORDED = "hypothesis_recorded"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    HYPOTHESIS_SUPPORTED = "hypothesis_supported"
    VERIFICATION_EXECUTION_OBSERVED = "verification_execution_observed"
    POST_EDIT_TEST_OBSERVED = "post_edit_test_observed"
    INVESTIGATION_INTENT_RECORDED = "investigation_intent_recorded"
    PHASE_CHANGED = "phase_changed"
    REEVALUATION_REQUIRED = "reevaluation_required"
    INVESTIGATION_PERMIT_GRANTED = "investigation_permit_granted"
    INVESTIGATION_STAGNATED = "investigation_stagnated"


class CheckedLocation(StrictModel):
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_line_order(self) -> CheckedLocation:
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")
        return self


class ProposedChange(StrictModel):
    path: str = Field(min_length=1)
    description: str = Field(min_length=1)


class CheckedFile(StrictModel):
    path: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=1)
    ranges: list[CheckedLocation] = Field(default_factory=list)
    scope: ScopeKind
    first_event_id: str = Field(min_length=1)
    latest_event_id: str = Field(min_length=1)


class RelationEdge(StrictModel):
    relation_id: str = Field(min_length=1)
    relation: Literal[
        "import",
        "call",
        "traceback",
        "grep_reference",
        "symbol_reference",
        "test_collection",
    ]
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    source_message_id: str = Field(min_length=1)
    scope: Literal[ScopeKind.DEPENDENCY] = ScopeKind.DEPENDENCY


class InvestigationHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    state: Literal["candidate", "rejected", "supported"]
    evidence_ids: list[str]
    checked_locations: list[CheckedLocation]
    proposed_change: ProposedChange | None = None
    expected_effect: str | None = None
    reason: str = Field(min_length=1)


class InvestigationPermit(StrictModel):
    permit_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    target_hash: str = Field(min_length=1)
    granted_in_generation: int = Field(ge=0)
    consumed: bool = False


class RecordHypothesisInput(StrictModel):
    hypothesis_id: str | None = None
    statement: str = Field(min_length=1)
    evidence_ids: list[str]
    checked_locations: list[CheckedLocation]
    proposed_change: ProposedChange | None = None
    expected_effect: str | None = None
    target_state: Literal["candidate", "rejected", "supported"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_transition(self) -> RecordHypothesisInput:
        if self.target_state == "supported" and (
            not self.evidence_ids
            or not self.checked_locations
            or self.proposed_change is None
            or not (self.expected_effect or "").strip()
        ):
            raise ValueError("supported 假设缺少证据、位置、修改目标或预期效果")
        if self.target_state == "rejected" and self.hypothesis_id is None:
            raise ValueError("rejected 迁移必须提供 hypothesis_id")
        return self


class ContinueInvestigationInput(StrictModel):
    hypothesis_ids: list[str]
    unresolved_question: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ToolObservation(StrictModel):
    event_type: InvestigationEventType
    tool_call_id: str | None = None
    source_message_id: str | None = None
    signature: str = ""
    result_fingerprint: str = ""
    scope: ScopeKind = ScopeKind.DIRECT
    progress_kind: ProgressKind | None = None
    evidence_id: str | None = None
    hypothesis_id: str | None = None
    path: str | None = None
    exit_code: int | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class NewInvestigationEvent(StrictModel):
    event_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    event_type: InvestigationEventType
    source_message_id: str | None = None
    tool_call_id: str | None = None
    phase_before: AgentPhase
    phase_after: AgentPhase
    progress_kind: ProgressKind | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def task_started(cls, task_id: str) -> NewInvestigationEvent:
        from deepfix.investigation.identity import stable_investigation_id

        return cls(
            event_id=stable_investigation_id("event", task_id, "task_started"),
            task_id=task_id,
            event_type=InvestigationEventType.TASK_STARTED,
            phase_before=AgentPhase.INVESTIGATING,
            phase_after=AgentPhase.INVESTIGATING,
        )


class InvestigationEvent(NewInvestigationEvent):
    sequence: int = Field(ge=1)
    created_at: datetime


class InvestigationState(StrictModel):
    task_id: str = Field(min_length=1)
    version: int = 0
    migration_version: int = 0
    agent_phase: AgentPhase = AgentPhase.INVESTIGATING
    paused_agent_phase: AgentPhase | None = None
    checked_files: list[CheckedFile] = Field(default_factory=list, max_length=64)
    relation_edges: list[RelationEdge] = Field(default_factory=list, max_length=128)
    recent_tool_signatures: list[str] = Field(default_factory=list, max_length=32)
    test_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list, max_length=64)
    supported_hypothesis_ids: list[str] = Field(default_factory=list, max_length=64)
    seen_progress_fingerprints: list[str] = Field(default_factory=list, max_length=64)
    progress_generation: int = 0
    last_progress_event_id: str | None = None
    last_progress_at: datetime | None = None
    no_progress_count: int = 0
    exploratory_without_progress: int = 0
    reevaluation_required: bool = False
    stagnation_level: Literal[0, 1, 2] = 0
    permit: InvestigationPermit | None = None
    post_permit_review_pending: bool = False

    @classmethod
    def new(cls, task_id: str) -> InvestigationState:
        return cls(task_id=task_id.strip())


class InvestigationRecoveryMetadata(StrictModel):
    task_id: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    agent_phase: AgentPhase
    state_version: int
    last_event_sequence: int
    tool_call_id: str | None = None
    permit_id: str | None = None
    checkpoint_available: bool
    recovery_action: str = Field(min_length=1)
