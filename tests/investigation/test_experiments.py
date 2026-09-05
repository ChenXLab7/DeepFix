from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from deepfix.compaction.models import ProvenanceRef
from deepfix.investigation.experiments import (
    DeterministicSuccessCriterion,
    ExecutorNarrativeResult,
    ExperimentIntent,
    ExperimentResult,
    ExperimentSpec,
    PathChangedCheck,
    SemanticSuccessCriterion,
    StrategyDecisionCandidate,
    SuccessCriterion,
    build_experiment_id,
    build_strategy_decision,
    validate_executor_narrative,
)
from deepfix.investigation.models import InvestigationCapability


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp-1",
        task_id="task-1",
        intents={ExperimentIntent.INVESTIGATE, ExperimentIntent.EDIT},
        goal="Confirm and repair the boundary defect",
        evidence_gap_ids=["gap-1"],
        success_criteria=[
            DeterministicSuccessCriterion(
                criterion_id="criterion-change",
                description="The allowed source file changed",
                check=PathChangedCheck(path="pkg/target.py"),
            )
        ],
        allowed_capabilities={
            InvestigationCapability.READ,
            InvestigationCapability.MODIFY,
        },
        step_budget=4,
        model_call_budget=2,
        time_budget_seconds=60,
        fallback="Return the remaining evidence gap",
    )


def test_executor_claim_must_reference_declared_criterion() -> None:
    narrative = ExecutorNarrativeResult(
        claimed_completed_criterion_ids=["unknown"],
        evidence_candidates=[],
        hypothesis_updates=[],
        remaining_questions=[],
        executor_recommendation="finish",
    )

    with pytest.raises(ValueError, match="unknown criterion"):
        validate_executor_narrative(_spec(), narrative)


def test_result_rejects_model_supplied_file_facts() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(
            {
                "experiment_id": "exp-1",
                "status": "completed",
                "executor_narrative": {
                    "claimed_completed_criterion_ids": [],
                    "evidence_candidates": [],
                    "hypothesis_updates": [],
                    "remaining_questions": [],
                    "executor_recommendation": "continue",
                },
                "tool_receipt_ids": [],
                "observation_ids": [],
                "changed_file_evidence_ids": [],
                "test_evidence_ids": [],
                "changed_files": ["foo.py"],
            }
        )


def test_success_criterion_uses_kind_discriminator() -> None:
    semantic = TypeAdapter(SuccessCriterion).validate_python(
        {
            "criterion_id": "criterion-root-cause",
            "kind": "semantic",
            "description": "The evidence supports the root cause",
            "question": "Does the evidence establish the off-by-one cause?",
            "required_evidence_types": ["observation", "test_result"],
            "minimum_independent_root_count": 2,
        }
    )

    assert isinstance(semantic, SemanticSuccessCriterion)
    assert semantic.minimum_independent_root_count == 2


def test_strategy_and_experiment_ids_are_stable_for_normalized_content() -> None:
    first = StrategyDecisionCandidate(
        decision_type="run_experiment",
        current_assessment="Need one focused experiment",
        selected_hypothesis_id="hypothesis-1",
        evidence_gap_ids=["gap-1"],
        experiment_spec=_spec(),
        uncertainty=0.25,
        rationale_refs=[ProvenanceRef(kind="system_evidence", ref_id="evidence-1")],
    )
    second = first.model_copy(deep=True)

    first_decision = build_strategy_decision("task-1", "blackboard-7", first)
    second_decision = build_strategy_decision("task-1", "blackboard-7", second)

    assert first_decision.decision_id == second_decision.decision_id
    assert build_experiment_id("task-1", first_decision.decision_id, _spec()) == (
        build_experiment_id("task-1", second_decision.decision_id, _spec())
    )


def test_run_experiment_strategy_requires_experiment_spec() -> None:
    candidate = StrategyDecisionCandidate(
        decision_type="run_experiment",
        current_assessment="Need evidence",
        evidence_gap_ids=["gap-1"],
        uncertainty=0.5,
        rationale_refs=[],
    )

    with pytest.raises(ValueError, match="experiment_spec"):
        build_strategy_decision("task-1", "blackboard-1", candidate)
