from __future__ import annotations

import pytest

from deepfix.compaction.models import ProvenancedClaim, ProvenanceRef
from deepfix.investigation.experiments import (
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentAssessmentOutcome,
    ExperimentProgressKind,
    ExperimentResult,
)
from deepfix.investigation.reducer import InvestigationStateReducer
from deepfix.investigation.store import InvestigationStateConflict, InvestigationStore


def _result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="experiment-1",
        status="completed",
        executor_narrative=ExecutorNarrativeResult(
            claimed_completed_criterion_ids=[],
            evidence_candidates=[],
            hypothesis_updates=[],
            remaining_questions=[],
            executor_recommendation="continue",
        ),
        tool_receipt_ids=["receipt-1"],
        observation_ids=["evidence-a", "evidence-b"],
        changed_file_evidence_ids=[],
        test_evidence_ids=[],
    )


def _assessment(*, claims: list[ProvenancedClaim] | None = None) -> ExperimentAssessment:
    return ExperimentAssessment(
        experiment_id="experiment-1",
        outcome=ExperimentAssessmentOutcome.PARTIALLY_SUCCEEDED,
        criterion_assessments=[],
        deterministic_evidence_ids=["evidence-a", "evidence-b"],
        accepted_claims=claims or [],
        rejected_claims=[],
        closed_evidence_gap_ids=["gap-1"],
        opened_evidence_gaps=[],
        supported_hypothesis_ids=[],
        rejected_hypothesis_ids=[],
        conflict_ids=[],
        progress_kind=ExperimentProgressKind.STRONG,
    )


def test_reducer_replay_is_idempotent(tmp_path) -> None:
    store = InvestigationStore(tmp_path / "state.db")
    reducer = InvestigationStateReducer("task-1", store)

    first = reducer.commit(0, _result(), _assessment())
    second = reducer.commit(first.version, _result(), _assessment())

    assert second.version == first.version
    assert store.count_experiment_events("task-1") == 1
    assert second.completed_experiment_ids == ["experiment-1"]


def test_reducer_propagates_parent_roots(tmp_path) -> None:
    store = InvestigationStore(tmp_path / "state.db")
    roots = {
        "evidence-a": ["root-a"],
        "evidence-b": ["root-b"],
    }
    reducer = InvestigationStateReducer(
        "task-1",
        store,
        provenance_root_resolver=lambda evidence_id: roots.get(evidence_id, []),
    )
    claim = ProvenancedClaim(
        claim_id="claim-1",
        text="The boundary check causes the failure",
        sources=[
            ProvenanceRef(kind="system_evidence", ref_id="evidence-a"),
            ProvenanceRef(kind="system_evidence", ref_id="evidence-b"),
        ],
    )

    state = reducer.commit(0, _result(), _assessment(claims=[claim]))

    assert state.experiment_claims[-1].provenance_root_ids == ["root-a", "root-b"]


def test_reducer_rejects_stale_non_replay_commit(tmp_path) -> None:
    store = InvestigationStore(tmp_path / "state.db")
    reducer = InvestigationStateReducer("task-1", store)
    reducer.commit(0, _result(), _assessment())
    different = _result().model_copy(update={"experiment_id": "experiment-2"})
    assessment = _assessment().model_copy(update={"experiment_id": "experiment-2"})

    with pytest.raises(InvestigationStateConflict):
        reducer.commit(0, different, assessment)
