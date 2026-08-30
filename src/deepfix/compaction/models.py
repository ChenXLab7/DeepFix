from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceRef(StrictModel):
    kind: Literal[
        "user_message",
        "work_unit",
        "working_memory",
        "system_evidence",
        "artifact",
        "snapshot_record",
    ]
    ref_id: str = Field(min_length=1)


class ProvenancedText(StrictModel):
    text: str = Field(min_length=1)
    sources: list[ProvenanceRef] = Field(default_factory=list)


class FactCandidate(ProvenancedText):
    pass


class ProvenancedClaim(ProvenancedText):
    claim_id: str = Field(min_length=1)
    state: Literal["confirmed", "conflict"] = "confirmed"


class UserConstraint(StrictModel):
    constraint_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    state: Literal["active", "revoked"] = "active"
    source_user_message_id: str = Field(min_length=1)
    supersedes_constraint_id: str | None = None


class UserConstraintCandidate(StrictModel):
    text: str = Field(min_length=1)
    source_user_message_id: str = Field(min_length=1)
    requested_state: Literal["active", "revoked"]
    supersedes_constraint_id: str | None = None


class HypothesisProgressInput(StrictModel):
    hypothesis_id: str | None = None
    text: str = Field(min_length=1)
    target_state: Literal["active", "rejected", "confirmed"]
    reason: str | None = None
    reopens_hypothesis_id: str | None = None
    sources: list[ProvenanceRef] = Field(default_factory=list)

    @field_validator("target_state", mode="before")
    @classmethod
    def normalize_investigation_state(cls, value: object) -> object:
        return {
            "candidate": "active",
            "supported": "active",
        }.get(value, value)


class HypothesisTransition(HypothesisProgressInput):
    pass


class HypothesisRecord(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    state: Literal["active", "rejected", "confirmed"]
    reason: str | None = None
    reopens_hypothesis_id: str | None = None
    sources: list[ProvenanceRef] = Field(default_factory=list)
    updated_in_version: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_reopen_identity(self) -> HypothesisRecord:
        if self.reopens_hypothesis_id == self.hypothesis_id:
            raise ValueError("reopens_hypothesis_id 必须引用不同的旧假设")
        return self


class ExperimentRecord(StrictModel):
    experiment_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    action: str = Field(min_length=1)
    result: str = Field(min_length=1)
    sources: list[ProvenanceRef] = Field(default_factory=list)


class SystemTestEvidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    exit_code: int
    summary: str
    tool_call_id: str = Field(min_length=1)
    source_message_id: str = Field(min_length=1)
    origin: Literal[
        "user_specified",
        "repository_existing",
        "agent_generated",
        "minimal_reproduction",
    ] = "repository_existing"
    scope: Literal["targeted", "module", "full_suite"] = "targeted"
    timing: Literal["baseline", "post_change", "post_recovery"] = "baseline"
    workspace_baseline_id: str = Field(default="legacy-untracked", min_length=1)
    code_state_hash: str = Field(default="legacy-untracked", min_length=1)
    test_target_paths: list[str] = Field(default_factory=list)
    test_content_hashes: dict[str, str] = Field(default_factory=dict)


class FileChangeEvidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    operation: Literal["write", "edit", "delete", "approved_target"]
    status: Literal["approved_target", "succeeded", "failed"]
    tool_call_id: str | None = None
    source_message_id: str | None = None


class ApprovalEvidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    risk: str = Field(min_length=1)


class ResearchStatusEvidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    verification: Literal["unverified", "verified", "contradicted"]
    artifact_path: str | None = None


class DeterministicEvidenceBlock(StrictModel):
    tests: list[SystemTestEvidence] = Field(default_factory=list)
    files: list[FileChangeEvidence] = Field(default_factory=list)
    approvals: list[ApprovalEvidence] = Field(default_factory=list)
    research: list[ResearchStatusEvidence] = Field(default_factory=list)


class ConflictRecord(StrictModel):
    conflict_id: str = Field(min_length=1)
    information_type: Literal[
        "user_constraint",
        "test",
        "file_operation",
        "approval",
        "research_status",
        "semantic",
    ]
    subject: str = Field(min_length=1)
    alternatives: list[ProvenancedText]
    source_of_truth: ProvenanceRef | None = None
    resolution: Literal["source_of_truth_applied", "unresolved_semantic"]


class ConflictCandidate(StrictModel):
    information_type: Literal[
        "user_constraint",
        "test",
        "file_operation",
        "approval",
        "research_status",
        "semantic",
    ]
    subject: str = Field(min_length=1)
    alternatives: list[ProvenancedText]


class ArtifactReference(StrictModel):
    path: str = Field(min_length=1)
    kind: Literal[
        "conversation_history",
        "large_tool_result",
        "operation_result",
        "research",
        "snapshot_detail",
    ]
    content_hash: str = Field(min_length=1)
    work_unit_ids: list[str] = Field(default_factory=list)


class SnapshotCoverage(StrictModel):
    last_user_message_id: str | None = None
    covered_message_ids: list[str] = Field(default_factory=list)
    covered_work_unit_ids: list[str] = Field(default_factory=list)


class WorkUnit(StrictModel):
    unit_id: str = Field(min_length=1)
    purpose: str
    message_ids: list[str] = Field(min_length=1)
    tool_call_ids: list[str] = Field(default_factory=list)
    state: Literal["complete", "incomplete", "ambiguous"]
    categories: set[
        Literal["read", "search", "modify", "verify_pass", "verify_fail", "other"]
    ] = Field(default_factory=set)
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    must_keep: bool = False


class TaskAnchor(StrictModel):
    task_id: str = Field(min_length=1)
    task_goal: str = Field(min_length=1)
    user_constraints: list[UserConstraint] = Field(default_factory=list)
    latest_user_message_id: str = Field(min_length=1)
    project_root: str = Field(min_length=1)
    project_python: str = Field(min_length=1)
    task_status: str = Field(min_length=1)


class CompactionDelta(StrictModel):
    user_constraint_candidates: list[UserConstraintCandidate]
    confirmed_fact_candidates: list[FactCandidate]
    hypothesis_transitions: list[HypothesisTransition]
    experiments: list[ExperimentRecord]
    conflict_candidates: list[ConflictCandidate]
    unresolved_questions: list[ProvenancedText]
    next_steps: list[ProvenancedText]


class CompactionSnapshot(StrictModel):
    task_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    previous_version: int | None = None
    lifecycle: Literal["prepared", "active", "abandoned"] = "prepared"
    created_at: str
    activated_at: str | None = None
    abandoned_at: str | None = None
    abandon_reason: str | None = None
    source_work_unit_ids: list[str]
    task_goal: str
    user_constraints: list[UserConstraint]
    confirmed_facts: list[ProvenancedClaim]
    deterministic_evidence: DeterministicEvidenceBlock
    active_hypotheses: list[HypothesisRecord]
    rejected_hypotheses: list[HypothesisRecord]
    confirmed_hypotheses: list[HypothesisRecord]
    changed_files: list[FileChangeEvidence]
    experiments: list[ExperimentRecord]
    test_results: list[SystemTestEvidence]
    conflicts: list[ConflictRecord]
    unresolved_questions: list[ProvenancedText]
    next_steps: list[ProvenancedText]
    artifact_references: list[ArtifactReference]
    content_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> CompactionSnapshot:
        if self.lifecycle == "prepared" and (
            self.activated_at is not None or self.abandoned_at is not None
        ):
            raise ValueError("prepared Snapshot 不能有生命周期结束时间")
        if self.lifecycle == "active" and self.activated_at is None:
            raise ValueError("active Snapshot 必须提供 activated_at")
        if self.lifecycle == "abandoned" and self.abandoned_at is None:
            raise ValueError("abandoned Snapshot 必须提供 abandoned_at")
        return self


class DeepFixCompactionEvent(StrictModel):
    event_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    active_snapshot_version: int = Field(ge=1)
    snapshot_message_id: str = Field(min_length=1)
    retained_message_ids: list[str]
    conversation_artifact: ArtifactReference
    input_hash: str = Field(min_length=1)


class CompactionFailureRecord(StrictModel):
    attempt_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    entrypoint: Literal["automatic", "manual_tool", "overflow_recovery"]
    budget_zone: Literal["normal", "observe", "normal_compaction", "emergency"]
    stage: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    original_messages_preserved: bool
    artifact_reference: str | None = None
    prepared_snapshot_version: int | None = None
    recorded_at: str


class ContextRecoveryMetadata(StrictModel):
    task_id: str = Field(min_length=1)
    stage: Literal[
        "protected_context",
        "artifact_write",
        "artifact_verify",
        "delta_generation",
        "snapshot_validate",
        "snapshot_write",
        "snapshot_verify",
        "compacted_model_call",
        "overflow_retry",
    ]
    error_code: str = Field(min_length=1)
    usage_ratio: float | None = None
    working_memory_version: int | None = None
    active_snapshot_version: int | None = None
    prepared_snapshot_version: int | None = None
    prepared_snapshot_lifecycle: Literal["prepared", "active", "abandoned"] | None = None
    conversation_artifact: str | None = None
    original_messages_preserved: bool
