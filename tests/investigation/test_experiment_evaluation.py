from __future__ import annotations

from dataclasses import dataclass

from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.investigation.evaluation import (
    EvidenceEvaluator,
    ExperimentEvidenceBundle,
    ExperimentResultBuilder,
    RepairLoopOutcome,
    SemanticEvidenceRecord,
    SemanticJudgment,
    SystemExperimentRuntimeRecord,
    adjudicate_outcome,
)
from deepfix.investigation.experiments import (
    DeterministicSuccessCriterion,
    EvidenceType,
    ExecutorNarrativeResult,
    ExperimentIntent,
    ExperimentSpec,
    SemanticSuccessCriterion,
)
from deepfix.investigation.experiments import (
    TestPassedCheck as PassedTestCheck,
)
from deepfix.investigation.models import InvestigationCapability
from deepfix.verification import OracleEvaluation


@dataclass
class RecordingSemanticJudge:
    calls: int = 0

    def judge(self, criterion, evidence):
        self.calls += 1
        return SemanticJudgment(
            completed=True,
            evidence_ids=[item.evidence_id for item in evidence],
            explanation="Independent evidence supports the cause",
        )


def _narrative(*claimed: str) -> ExecutorNarrativeResult:
    return ExecutorNarrativeResult(
        claimed_completed_criterion_ids=list(claimed),
        evidence_candidates=[],
        hypothesis_updates=[],
        remaining_questions=[],
        executor_recommendation="finish",
    )


def _spec(criterion) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="experiment-1",
        task_id="task-1",
        intents={ExperimentIntent.VERIFY},
        goal="Evaluate one criterion",
        evidence_gap_ids=[],
        success_criteria=[criterion],
        allowed_capabilities={InvestigationCapability.EXECUTE},
        step_budget=2,
        model_call_budget=1,
        time_budget_seconds=30,
        fallback="Return the unmet criterion",
    )


def _test_evidence(exit_code: int) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id="test-evidence-1",
        command="python -m pytest tests/test_target.py -q",
        exit_code=exit_code,
        summary="test result",
        tool_call_id="call-1",
        source_message_id="message-1",
        origin="user_specified",
        scope="targeted",
        timing="post_change",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-2",
    )


def test_executor_claim_does_not_complete_failed_pytest() -> None:
    criterion = DeterministicSuccessCriterion(
        criterion_id="criterion-test",
        description="Required test passes",
        check=PassedTestCheck(oracle_id="oracle-1"),
    )
    runtime = SystemExperimentRuntimeRecord(
        status="completed",
        test_evidence_ids=["test-evidence-1"],
    )
    result = ExperimentResultBuilder().build(
        _spec(criterion),
        _narrative("criterion-test"),
        runtime,
    )
    evidence = ExperimentEvidenceBundle(
        tests=[_test_evidence(exit_code=1)],
        oracle_evidence_ids={"oracle-1": ["test-evidence-1"]},
    )
    judge = RecordingSemanticJudge()

    assessment = EvidenceEvaluator(evidence, judge).evaluate(
        _spec(criterion), result
    )

    assert assessment.criterion_assessments[0].completed is False
    assert judge.calls == 0


def test_semantic_sources_count_independent_roots() -> None:
    criterion = SemanticSuccessCriterion(
        criterion_id="criterion-cause",
        description="Root cause is supported",
        question="Is foo() the root cause?",
        required_evidence_types={EvidenceType.OBSERVATION, EvidenceType.CLAIM},
        minimum_independent_root_count=2,
    )
    evidence = ExperimentEvidenceBundle(
        semantic=[
            SemanticEvidenceRecord(
                evidence_id="observation-1",
                evidence_type="observation",
                provenance_root_ids=["root-1"],
                summary="Observed the wrong branch",
            ),
            SemanticEvidenceRecord(
                evidence_id="claim-1",
                evidence_type="claim",
                provenance_root_ids=["root-1"],
                summary="The branch is the root cause",
            ),
        ]
    )
    result = ExperimentResultBuilder().build(
        _spec(criterion),
        _narrative("criterion-cause"),
        SystemExperimentRuntimeRecord(
            status="completed",
            observation_ids=["observation-1", "claim-1"],
        ),
    )
    judge = RecordingSemanticJudge()

    assessment = EvidenceEvaluator(evidence, judge).evaluate(
        _spec(criterion), result
    )

    assert assessment.criterion_assessments[0].completed is False
    assert judge.calls == 0


def test_result_builder_uses_only_system_runtime_evidence_ids() -> None:
    criterion = DeterministicSuccessCriterion(
        criterion_id="criterion-test",
        description="Required test passes",
        check=PassedTestCheck(oracle_id="oracle-1"),
    )
    runtime = SystemExperimentRuntimeRecord(
        status="completed",
        tool_receipt_ids=["receipt-1"],
        changed_file_evidence_ids=["file-change-1"],
        test_evidence_ids=["test-evidence-1"],
    )

    result = ExperimentResultBuilder().build(
        _spec(criterion), _narrative("criterion-test"), runtime
    )

    assert result.tool_receipt_ids == ["receipt-1"]
    assert result.changed_file_evidence_ids == ["file-change-1"]
    assert result.test_evidence_ids == ["test-evidence-1"]


def test_required_oracle_conflict_blocks_fixed() -> None:
    oracle = OracleEvaluation(
        passed_required_oracle_ids=["oracle-1"],
        conflicting_evidence_ids=["suite-failure"],
        all_required_satisfied=True,
        fixed_allowed=False,
    )

    outcome = adjudicate_outcome(
        oracle,
        changed_files=[
            FileChangeEvidence(
                evidence_id="file-change-1",
                path="pkg/target.py",
                operation="edit",
                status="succeeded",
                tool_call_id="call-edit",
                source_message_id="message-edit",
            )
        ],
        unresolved_operation_ids=[],
        scope_violation_evidence_ids=[],
        reproduction_state="reproduced",
    )

    assert outcome is RepairLoopOutcome.REVIEW
