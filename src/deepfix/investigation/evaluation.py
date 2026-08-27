from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field

from deepfix.compaction.models import FileChangeEvidence, StrictModel, SystemTestEvidence
from deepfix.investigation.experiments import (
    CommandCompletedCheck,
    CriterionAssessment,
    DeterministicSuccessCriterion,
    EvidenceType,
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentAssessmentOutcome,
    ExperimentProgressKind,
    ExperimentResult,
    ExperimentResultStatus,
    ExperimentSpec,
    NoScopeViolationCheck,
    PathChangedCheck,
    PathUnchangedCheck,
    SemanticSuccessCriterion,
    TestPassedCheck,
    validate_executor_narrative,
)
from deepfix.verification import OracleEvaluation


class SystemExperimentRuntimeRecord(StrictModel):
    status: ExperimentResultStatus
    tool_receipt_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    changed_file_evidence_ids: list[str] = Field(default_factory=list)
    test_evidence_ids: list[str] = Field(default_factory=list)


class SemanticEvidenceRecord(StrictModel):
    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    provenance_root_ids: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)


class CommandEvidenceRecord(StrictModel):
    evidence_id: str = Field(min_length=1)
    command_fingerprint: str = Field(min_length=1)
    exit_code: int | None = None
    timed_out: bool = False


class ExperimentEvidenceBundle(StrictModel):
    tests: list[SystemTestEvidence] = Field(default_factory=list)
    files: list[FileChangeEvidence] = Field(default_factory=list)
    semantic: list[SemanticEvidenceRecord] = Field(default_factory=list)
    commands: list[CommandEvidenceRecord] = Field(default_factory=list)
    oracle_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    path_unchanged_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    scope_audit_evidence_ids: list[str] = Field(default_factory=list)
    scope_violation_evidence_ids: list[str] = Field(default_factory=list)


class SemanticJudgment(StrictModel):
    completed: bool
    evidence_ids: list[str]
    explanation: str = Field(min_length=1)


class SemanticCriterionJudge(Protocol):
    def judge(
        self,
        criterion: SemanticSuccessCriterion,
        evidence: list[SemanticEvidenceRecord],
    ) -> SemanticJudgment: ...


class ExperimentResultBuilder:
    def build(
        self,
        spec: ExperimentSpec,
        narrative: ExecutorNarrativeResult,
        runtime_record: SystemExperimentRuntimeRecord,
    ) -> ExperimentResult:
        validate_executor_narrative(spec, narrative)
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status=runtime_record.status,
            executor_narrative=narrative,
            tool_receipt_ids=runtime_record.tool_receipt_ids,
            observation_ids=runtime_record.observation_ids,
            changed_file_evidence_ids=runtime_record.changed_file_evidence_ids,
            test_evidence_ids=runtime_record.test_evidence_ids,
        )


class EvidenceEvaluator:
    def __init__(
        self,
        evidence: ExperimentEvidenceBundle,
        semantic_judge: SemanticCriterionJudge,
    ) -> None:
        self.evidence = evidence
        self.semantic_judge = semantic_judge

    def evaluate(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
    ) -> ExperimentAssessment:
        if result.experiment_id != spec.experiment_id:
            raise ValueError("experiment result/spec mismatch")
        validate_executor_narrative(spec, result.executor_narrative)
        assessments = [
            self._evaluate_criterion(criterion, result)
            for criterion in spec.success_criteria
        ]
        required_ids = {
            item.criterion_id for item in spec.success_criteria if item.required
        }
        completed_ids = {
            item.criterion_id for item in assessments if item.completed
        }
        outcome = _assessment_outcome(result.status, required_ids, completed_ids)
        deterministic_ids = list(
            dict.fromkeys(
                evidence_id
                for item in assessments
                if item.criterion_kind == "deterministic"
                for evidence_id in item.evidence_ids
            )
        )
        progress = (
            ExperimentProgressKind.STRONG
            if completed_ids or deterministic_ids
            else (
                ExperimentProgressKind.WEAK
                if result.executor_narrative.evidence_candidates
                or result.executor_narrative.hypothesis_updates
                else ExperimentProgressKind.NONE
            )
        )
        return ExperimentAssessment(
            experiment_id=spec.experiment_id,
            outcome=outcome,
            criterion_assessments=assessments,
            deterministic_evidence_ids=deterministic_ids,
            accepted_claims=[],
            rejected_claims=[],
            closed_evidence_gap_ids=[],
            opened_evidence_gaps=[],
            supported_hypothesis_ids=[],
            rejected_hypothesis_ids=[],
            conflict_ids=list(self.evidence.scope_violation_evidence_ids),
            progress_kind=progress,
        )

    def _evaluate_criterion(
        self,
        criterion: DeterministicSuccessCriterion | SemanticSuccessCriterion,
        result: ExperimentResult,
    ) -> CriterionAssessment:
        if isinstance(criterion, DeterministicSuccessCriterion):
            completed, evidence_ids, explanation = self._evaluate_deterministic(
                criterion, result
            )
            return CriterionAssessment(
                criterion_id=criterion.criterion_id,
                criterion_kind="deterministic",
                completed=completed,
                evidence_ids=evidence_ids,
                explanation=explanation,
            )
        return self._evaluate_semantic(criterion, result)

    def _evaluate_deterministic(
        self,
        criterion: DeterministicSuccessCriterion,
        result: ExperimentResult,
    ) -> tuple[bool, list[str], str]:
        check = criterion.check
        if isinstance(check, TestPassedCheck):
            allowed_ids = set(result.test_evidence_ids)
            linked_ids = self.evidence.oracle_evidence_ids.get(check.oracle_id, [])
            selected = [
                item
                for item in self.evidence.tests
                if item.evidence_id in allowed_ids and item.evidence_id in linked_ids
            ]
            completed = bool(selected) and selected[-1].exit_code == 0
            return (
                completed,
                [item.evidence_id for item in selected],
                "required test passed" if completed else "required test did not pass",
            )
        if isinstance(check, PathChangedCheck):
            selected = [
                item
                for item in self.evidence.files
                if item.evidence_id in result.changed_file_evidence_ids
                and item.path == check.path
                and item.status == "succeeded"
            ]
            return bool(selected), [item.evidence_id for item in selected], (
                "path changed" if selected else "path change was not proven"
            )
        if isinstance(check, PathUnchangedCheck):
            evidence_ids = self.evidence.path_unchanged_evidence_ids.get(
                check.path, []
            )
            return bool(evidence_ids), evidence_ids, (
                "path remained unchanged"
                if evidence_ids
                else "unchanged path was not proven"
            )
        if isinstance(check, CommandCompletedCheck):
            selected = [
                item
                for item in self.evidence.commands
                if item.command_fingerprint == check.command_fingerprint
                and item.evidence_id in result.observation_ids
            ]
            completed = bool(selected) and (
                selected[-1].exit_code == check.expected_exit_code
                and (check.allow_timeout or not selected[-1].timed_out)
            )
            return completed, [item.evidence_id for item in selected], (
                "command completed" if completed else "command completion was not proven"
            )
        if isinstance(check, NoScopeViolationCheck):
            completed = bool(self.evidence.scope_audit_evidence_ids) and not (
                self.evidence.scope_violation_evidence_ids
            )
            return completed, list(self.evidence.scope_audit_evidence_ids), (
                "scope audit found no violation"
                if completed
                else "scope safety was not established"
            )
        raise TypeError(f"unsupported deterministic criterion: {type(check).__name__}")

    def _evaluate_semantic(
        self,
        criterion: SemanticSuccessCriterion,
        result: ExperimentResult,
    ) -> CriterionAssessment:
        referenced_ids = set(result.observation_ids)
        referenced_ids.update(
            source.ref_id
            for candidate in result.executor_narrative.evidence_candidates
            for source in candidate.source_refs
        )
        selected = [
            item
            for item in self.evidence.semantic
            if item.evidence_id in referenced_ids
        ]
        evidence_types = {item.evidence_type for item in selected}
        root_ids = {
            root_id for item in selected for root_id in item.provenance_root_ids
        }
        if not criterion.required_evidence_types.issubset(evidence_types):
            return CriterionAssessment(
                criterion_id=criterion.criterion_id,
                criterion_kind="semantic",
                completed=False,
                evidence_ids=[item.evidence_id for item in selected],
                explanation="required semantic evidence types are missing",
            )
        if len(root_ids) < criterion.minimum_independent_root_count:
            return CriterionAssessment(
                criterion_id=criterion.criterion_id,
                criterion_kind="semantic",
                completed=False,
                evidence_ids=[item.evidence_id for item in selected],
                explanation="independent provenance root threshold was not met",
            )
        judgment = self.semantic_judge.judge(criterion, selected)
        selected_ids = {item.evidence_id for item in selected}
        if not set(judgment.evidence_ids).issubset(selected_ids):
            raise ValueError("semantic judgment referenced unknown evidence")
        return CriterionAssessment(
            criterion_id=criterion.criterion_id,
            criterion_kind="semantic",
            completed=judgment.completed,
            evidence_ids=judgment.evidence_ids,
            explanation=judgment.explanation,
        )


class RepairLoopOutcome(StrEnum):
    FIXED = "fixed"
    NOT_REPRODUCED = "not_reproduced"
    CONTINUE = "continue"
    REVIEW = "review"
    BLOCKED = "blocked"


def adjudicate_outcome(
    oracle_evaluation: OracleEvaluation,
    *,
    changed_files: list[FileChangeEvidence],
    unresolved_operation_ids: list[str],
    scope_violation_evidence_ids: list[str],
    reproduction_state: Literal["unknown", "reproduced", "not_reproduced"],
) -> RepairLoopOutcome:
    if unresolved_operation_ids:
        return RepairLoopOutcome.BLOCKED
    if scope_violation_evidence_ids or oracle_evaluation.conflicting_evidence_ids:
        return RepairLoopOutcome.REVIEW
    real_change = any(item.status == "succeeded" for item in changed_files)
    if oracle_evaluation.fixed_allowed and real_change:
        return RepairLoopOutcome.FIXED
    if reproduction_state == "not_reproduced" and not real_change:
        return RepairLoopOutcome.NOT_REPRODUCED
    return RepairLoopOutcome.CONTINUE


def _assessment_outcome(
    status: ExperimentResultStatus,
    required_ids: set[str],
    completed_ids: set[str],
) -> ExperimentAssessmentOutcome:
    terminal = {
        ExperimentResultStatus.TIMED_OUT: ExperimentAssessmentOutcome.TIMED_OUT,
        ExperimentResultStatus.POLICY_BLOCKED: (
            ExperimentAssessmentOutcome.POLICY_BLOCKED
        ),
        ExperimentResultStatus.BUDGET_EXHAUSTED: (
            ExperimentAssessmentOutcome.BUDGET_EXHAUSTED
        ),
        ExperimentResultStatus.ABANDONED: ExperimentAssessmentOutcome.FAILED,
    }
    if status in terminal:
        return terminal[status]
    if required_ids and required_ids.issubset(completed_ids):
        return ExperimentAssessmentOutcome.SUCCEEDED
    if completed_ids:
        return ExperimentAssessmentOutcome.PARTIALLY_SUCCEEDED
    return ExperimentAssessmentOutcome.INCONCLUSIVE
