from __future__ import annotations

import pytest

from deepfix.compaction.models import ProvenancedClaim, ProvenanceRef, SystemTestEvidence
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.investigation import (
    InvestigationRepository,
    InvestigationStateConflict,
)
from deepfix.investigation.experiments import (
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentAssessmentOutcome,
    ExperimentProgressKind,
    ExperimentResult,
)
from deepfix.investigation.models import (
    InvestigationHypothesis,
    NewInvestigationEvent,
)
from deepfix.investigation.reducer import InvestigationStateReducer


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
    store = InvestigationRepository(tmp_path / "state.db")
    reducer = InvestigationStateReducer("task-1", store)

    first = reducer.commit(0, _result(), _assessment())
    second = reducer.commit(first.version, _result(), _assessment())

    assert second.version == first.version
    assert store.count_experiment_events("task-1") == 1
    assert second.completed_experiment_ids == ["experiment-1"]


def test_reducer_propagates_parent_roots(tmp_path) -> None:
    store = InvestigationRepository(tmp_path / "state.db")
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
    store = InvestigationRepository(tmp_path / "state.db")
    reducer = InvestigationStateReducer("task-1", store)
    reducer.commit(0, _result(), _assessment())
    different = _result().model_copy(update={"experiment_id": "experiment-2"})
    assessment = _assessment().model_copy(update={"experiment_id": "experiment-2"})

    with pytest.raises(InvestigationStateConflict):
        reducer.commit(0, different, assessment)


def test_reducer_attaches_assessment_evidence_before_supporting_hypothesis(
    tmp_path,
) -> None:
    database = tmp_path / "state.db"
    evidence = EvidenceRepository(database)
    for index, evidence_id in enumerate(["evidence-a", "evidence-b"], start=1):
        evidence.record_deterministic(
            "task-1",
            SystemTestEvidence(
                evidence_id=evidence_id,
                command="python -m pytest -q",
                exit_code=1,
                summary="1 failed",
                tool_call_id=f"call-{index}",
                source_message_id=f"message-{index}",
            ),
            provenance_root_ids=[f"call-{index}"],
        )
    store = InvestigationRepository(database)
    state = store.ensure_started("task-1")
    candidate = InvestigationHypothesis(
        hypothesis_id="h-1",
        statement="the boundary condition is reversed",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate",
    )
    state = store.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id="event-h-1",
                task_id="task-1",
                event_type="hypothesis_recorded",
            )
        ],
        state.model_copy(update={"hypotheses": [candidate]}),
    )
    reducer = InvestigationStateReducer("task-1", store)
    assessment = _assessment().model_copy(
        update={"supported_hypothesis_ids": ["h-1"]}
    )

    committed = reducer.commit(state.version, _result(), assessment)

    supported = committed.hypotheses[0]
    assert supported.state == "supported"
    assert supported.evidence_ids == ["evidence-a", "evidence-b"]
    assert store.get_hypothesis("task-1", "h-1") == supported
