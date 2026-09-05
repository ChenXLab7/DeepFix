from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from deepfix.compaction.models import ProvenancedClaim, ProvenanceRef, StrictModel
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import InvestigationCapability


class ExperimentIntent(StrEnum):
    REPRODUCE = "reproduce"
    INVESTIGATE = "investigate"
    EDIT = "edit"
    VERIFY = "verify"
    RESEARCH = "research"
    REVIEW = "review"


class EvidenceType(StrEnum):
    USER_MESSAGE = "user_message"
    TOOL_RECEIPT = "tool_receipt"
    OBSERVATION = "observation"
    TEST_RESULT = "test_result"
    FILE_CHANGE = "file_change"
    APPROVAL = "approval"
    RESEARCH = "research"
    CLAIM = "claim"


class TestPassedCheck(StrictModel):
    check_type: Literal["test_passed"] = "test_passed"
    oracle_id: str = Field(min_length=1)


class PathChangedCheck(StrictModel):
    check_type: Literal["path_changed"] = "path_changed"
    path: str = Field(min_length=1)


class PathUnchangedCheck(StrictModel):
    check_type: Literal["path_unchanged"] = "path_unchanged"
    path: str = Field(min_length=1)


class CommandCompletedCheck(StrictModel):
    check_type: Literal["command_completed"] = "command_completed"
    command_fingerprint: str = Field(min_length=1)
    expected_exit_code: int = 0
    allow_timeout: bool = False


class NoScopeViolationCheck(StrictModel):
    check_type: Literal["no_scope_violation"] = "no_scope_violation"


DeterministicCriterionCheck = Annotated[
    TestPassedCheck
    | PathChangedCheck
    | PathUnchangedCheck
    | CommandCompletedCheck
    | NoScopeViolationCheck,
    Field(discriminator="check_type"),
]


class DeterministicSuccessCriterion(StrictModel):
    criterion_id: str = Field(min_length=1)
    kind: Literal["deterministic"] = "deterministic"
    description: str = Field(min_length=1)
    check: DeterministicCriterionCheck
    required: bool = True


class SemanticSuccessCriterion(StrictModel):
    criterion_id: str = Field(min_length=1)
    kind: Literal["semantic"] = "semantic"
    description: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_evidence_types: set[EvidenceType] = Field(min_length=1)
    minimum_independent_root_count: int = Field(default=1, gt=0)
    required: bool = True


SuccessCriterion = Annotated[
    DeterministicSuccessCriterion | SemanticSuccessCriterion,
    Field(discriminator="kind"),
]


class SuggestedAction(StrictModel):
    capability: InvestigationCapability
    purpose: str = Field(min_length=1)
    tool_name: str | None = None
    target: str | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ExperimentSpec(StrictModel):
    experiment_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    intents: set[ExperimentIntent] = Field(min_length=1)
    goal: str = Field(min_length=1)
    target_hypothesis_id: str | None = None
    evidence_gap_ids: list[str]
    success_criteria: list[SuccessCriterion] = Field(min_length=1)
    allowed_capabilities: set[InvestigationCapability]
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    step_budget: int = Field(gt=0)
    model_call_budget: int = Field(gt=0)
    token_budget: int | None = Field(default=None, gt=0)
    time_budget_seconds: int = Field(gt=0)
    fallback: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_criterion_ids(self) -> ExperimentSpec:
        criterion_ids = [item.criterion_id for item in self.success_criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("success criterion IDs must be unique")
        return self


class EvidenceCandidate(StrictModel):
    evidence_type: EvidenceType
    summary: str = Field(min_length=1)
    source_refs: list[ProvenanceRef] = Field(default_factory=list)


class HypothesisUpdateCandidate(StrictModel):
    hypothesis_id: str | None = None
    statement: str = Field(min_length=1)
    target_state: Literal["candidate", "rejected", "supported"]
    reason: str = Field(min_length=1)
    source_refs: list[ProvenanceRef] = Field(default_factory=list)


class ExecutorNarrativeResult(StrictModel):
    claimed_completed_criterion_ids: list[str]
    evidence_candidates: list[EvidenceCandidate]
    hypothesis_updates: list[HypothesisUpdateCandidate]
    remaining_questions: list[str]
    executor_recommendation: str = Field(min_length=1)


class ExperimentResultStatus(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TIMED_OUT = "timed_out"
    POLICY_BLOCKED = "policy_blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExperimentResult(StrictModel):
    experiment_id: str = Field(min_length=1)
    status: ExperimentResultStatus
    executor_narrative: ExecutorNarrativeResult
    tool_receipt_ids: list[str]
    observation_ids: list[str]
    changed_file_evidence_ids: list[str]
    test_evidence_ids: list[str]


class StrategyAlternative(StrictModel):
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class StrategyReflection(StrictModel):
    alternatives: list[StrategyAlternative] = Field(min_length=2)
    prior_strategy_weakness: str = Field(min_length=1)
    selected_alternative_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_selected_alternative(self) -> StrategyReflection:
        if self.selected_alternative_index >= len(self.alternatives):
            raise ValueError("selected alternative index is out of range")
        return self


class StrategyDecisionCandidate(StrictModel):
    decision_type: Literal["run_experiment", "ask_user", "conclude"]
    current_assessment: str = Field(min_length=1)
    selected_hypothesis_id: str | None = None
    evidence_gap_ids: list[str]
    experiment_spec: ExperimentSpec | None = None
    reflection: StrategyReflection | None = None
    uncertainty: float = Field(ge=0, le=1)
    rationale_refs: list[ProvenanceRef]
    question_for_user: str | None = None
    candidate_outcome: str | None = None


class StrategyDecision(StrategyDecisionCandidate):
    decision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    blackboard_fingerprint: str = Field(min_length=1)


class CriterionAssessment(StrictModel):
    criterion_id: str = Field(min_length=1)
    criterion_kind: Literal["deterministic", "semantic"]
    completed: bool
    evidence_ids: list[str]
    explanation: str = Field(min_length=1)


class RejectedClaim(StrictModel):
    summary: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_refs: list[ProvenanceRef] = Field(default_factory=list)


class EvidenceGap(StrictModel):
    gap_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_evidence_types: set[EvidenceType] = Field(default_factory=set)


class ExperimentAssessmentOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    POLICY_BLOCKED = "policy_blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExperimentProgressKind(StrEnum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class ExperimentAssessment(StrictModel):
    experiment_id: str = Field(min_length=1)
    outcome: ExperimentAssessmentOutcome
    criterion_assessments: list[CriterionAssessment]
    deterministic_evidence_ids: list[str]
    accepted_claims: list[ProvenancedClaim]
    rejected_claims: list[RejectedClaim]
    closed_evidence_gap_ids: list[str]
    opened_evidence_gaps: list[EvidenceGap]
    supported_hypothesis_ids: list[str]
    rejected_hypothesis_ids: list[str]
    conflict_ids: list[str]
    progress_kind: ExperimentProgressKind
    recommended_strategy_change: str | None = None


def validate_executor_narrative(
    spec: ExperimentSpec,
    narrative: ExecutorNarrativeResult,
) -> None:
    declared = {item.criterion_id for item in spec.success_criteria}
    unknown = sorted(set(narrative.claimed_completed_criterion_ids) - declared)
    if unknown:
        raise ValueError(f"unknown criterion IDs: {', '.join(unknown)}")


def build_strategy_decision(
    task_id: str,
    blackboard_fingerprint: str,
    candidate: StrategyDecisionCandidate,
) -> StrategyDecision:
    if candidate.decision_type == "run_experiment" and candidate.experiment_spec is None:
        raise ValueError("run_experiment requires experiment_spec")
    if candidate.decision_type == "ask_user" and not candidate.question_for_user:
        raise ValueError("ask_user requires question_for_user")
    if candidate.decision_type == "conclude" and not candidate.candidate_outcome:
        raise ValueError("conclude requires candidate_outcome")
    decision_id = stable_investigation_id(
        "decision",
        task_id,
        blackboard_fingerprint,
        _canonical_json(candidate),
    )
    return StrategyDecision(
        decision_id=decision_id,
        task_id=task_id,
        blackboard_fingerprint=blackboard_fingerprint,
        **candidate.model_dump(mode="python"),
    )


def build_experiment_id(
    task_id: str,
    strategy_decision_id: str,
    spec: ExperimentSpec,
) -> str:
    definition = spec.model_dump(mode="python", exclude={"experiment_id", "task_id"})
    return stable_investigation_id(
        "experiment",
        task_id,
        strategy_decision_id,
        _canonical_json(definition),
    )


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    return json.dumps(_normalize(value), ensure_ascii=False, separators=(",", ":"))


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value
