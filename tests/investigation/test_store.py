import pytest

from deepfix.compaction.models import SystemTestEvidence
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.investigation.experiments import (
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentAssessmentOutcome,
    ExperimentProgressKind,
    ExperimentResult,
    StrategyDecision,
)
from deepfix.investigation.models import (
    AgentPhase,
    InvestigationHypothesis,
    NewInvestigationEvent,
)
from deepfix.investigation.store import InvestigationStateConflict, InvestigationStore


def test_strategy_decision_save_is_idempotent(tmp_path) -> None:
    store = InvestigationStore(tmp_path / "state.db")
    decision = StrategyDecision(
        decision_id="decision-1",
        task_id="task-a",
        blackboard_fingerprint="blackboard-1",
        decision_type="ask_user",
        current_assessment="Local evidence cannot answer the question",
        evidence_gap_ids=["gap-1"],
        uncertainty=0.8,
        rationale_refs=[],
        question_for_user="Which behavior is expected?",
    )

    store.save_strategy_decision(decision)
    store.save_strategy_decision(decision)

    assert store.count_strategy_decisions("task-a") == 1


def new_event(
    task_id: str,
    event_id: str,
    event_type: str,
    phase: AgentPhase,
) -> NewInvestigationEvent:
    return NewInvestigationEvent(
        event_id=event_id,
        task_id=task_id,
        event_type=event_type,
        phase_before=phase,
        phase_after=phase,
    )


def test_store_commits_event_and_state_atomically(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    event = new_event("task-a", "event-1", "tool_completed", state.agent_phase)
    updated = state.model_copy(update={"no_progress_count": 1})

    committed = store.commit(state.version, [event], updated)

    assert committed.version == 2
    assert store.load("task-a").no_progress_count == 1
    assert [item.sequence for item in store.list_events("task-a")] == [1, 2]


def test_store_replay_is_idempotent_and_task_isolated(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    a = store.ensure_started("task-a")
    b = store.ensure_started("task-b")
    event = new_event("task-a", "event-a", "file_checked", a.agent_phase)

    committed = store.commit(a.version, [event], a)
    replayed = store.commit(a.version, [event], a)

    assert replayed == committed
    assert len(store.list_events("task-a")) == 2
    assert len(store.list_events("task-b")) == 1
    assert b.task_id == "task-b"


def test_store_rejects_stale_non_replay_commit(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    store.commit(
        state.version,
        [new_event("task-a", "event-1", "file_checked", state.agent_phase)],
        state,
    )

    with pytest.raises(InvestigationStateConflict):
        store.commit(
            state.version,
            [new_event("task-a", "event-2", "file_checked", state.agent_phase)],
            state,
        )


def test_store_rejects_same_event_id_with_different_payload(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    original = new_event("task-a", "event-1", "file_checked", state.agent_phase)
    store.commit(state.version, [original], state)
    conflicting = new_event("task-a", "event-1", "file_changed", state.agent_phase)

    with pytest.raises(InvestigationStateConflict, match="事件内容冲突"):
        store.commit(state.version, [conflicting], state)


def test_store_rejects_events_from_another_task(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    foreign = new_event("task-b", "event-1", "file_checked", state.agent_phase)

    with pytest.raises(InvestigationStateConflict, match="任务不一致"):
        store.commit(state.version, [foreign], state)


def test_last_sequence_is_zero_for_unknown_task(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")

    assert store.load("missing") is None
    assert store.last_sequence("missing") == 0


def test_store_projects_hypotheses_to_domain_repository_atomically(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    hypothesis = InvestigationHypothesis(
        hypothesis_id="h-1",
        statement="empty input follows the wrong branch",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate from inspected code",
    )
    event = new_event(
        "task-a",
        "event-h-1",
        "hypothesis_recorded",
        state.agent_phase,
    )

    committed = store.commit(
        state.version,
        [event],
        state.model_copy(update={"hypotheses": [hypothesis]}),
    )

    assert committed.hypotheses == [hypothesis]
    assert store.repository.list_hypotheses("task-a") == [hypothesis]


def test_store_load_overlays_repository_hypothesis_over_stale_legacy_payload(
    tmp_path,
) -> None:
    database = tmp_path / "deepfix.sqlite3"
    store = InvestigationStore(database)
    state = store.ensure_started("task-a")
    candidate = InvestigationHypothesis(
        hypothesis_id="h-1",
        statement="empty input follows the wrong branch",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate",
    )
    event = new_event(
        "task-a",
        "event-h-1",
        "hypothesis_recorded",
        state.agent_phase,
    )
    state = store.commit(
        state.version,
        [event],
        state.model_copy(update={"hypotheses": [candidate]}),
    )
    EvidenceRepository(database).record_deterministic(
        "task-a",
        SystemTestEvidence(
            evidence_id="e-1",
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="call-1",
            source_message_id="message-1",
        ),
        provenance_root_ids=["call-1"],
    )
    supported = candidate.model_copy(
        update={"state": "supported", "evidence_ids": ["e-1"]}
    )
    store.repository.record_hypothesis("task-a", supported)

    loaded = store.load("task-a")

    assert loaded is not None
    assert loaded.version == state.version
    assert loaded.hypotheses == [supported]
    assert loaded.supported_hypothesis_ids == ["h-1"]

    replayed = store.commit(1, [event], state)

    assert replayed.hypotheses == [supported]
    assert replayed.supported_hypothesis_ids == ["h-1"]


def test_experiment_replay_returns_repository_overlay(tmp_path) -> None:
    database = tmp_path / "deepfix.sqlite3"
    store = InvestigationStore(database)
    state = store.ensure_started("task-a")
    candidate = InvestigationHypothesis(
        hypothesis_id="h-1",
        statement="empty input follows the wrong branch",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate",
    )
    state = store.commit(
        state.version,
        [new_event("task-a", "event-h-1", "hypothesis_recorded", state.agent_phase)],
        state.model_copy(update={"hypotheses": [candidate]}),
    )
    result = ExperimentResult(
        experiment_id="experiment-1",
        status="completed",
        executor_narrative=ExecutorNarrativeResult(
            claimed_completed_criterion_ids=[],
            evidence_candidates=[],
            hypothesis_updates=[],
            remaining_questions=[],
            executor_recommendation="continue",
        ),
        tool_receipt_ids=[],
        observation_ids=[],
        changed_file_evidence_ids=[],
        test_evidence_ids=[],
    )
    assessment = ExperimentAssessment(
        experiment_id="experiment-1",
        outcome=ExperimentAssessmentOutcome.PARTIALLY_SUCCEEDED,
        criterion_assessments=[],
        deterministic_evidence_ids=[],
        accepted_claims=[],
        rejected_claims=[],
        closed_evidence_gap_ids=[],
        opened_evidence_gaps=[],
        supported_hypothesis_ids=[],
        rejected_hypothesis_ids=[],
        conflict_ids=[],
        progress_kind=ExperimentProgressKind.WEAK,
    )
    committed = store.commit_experiment(
        state.version,
        event_id="experiment-event-1",
        event_payload="payload-1",
        result=result,
        assessment=assessment,
        next_state=state,
    )
    EvidenceRepository(database).record_deterministic(
        "task-a",
        SystemTestEvidence(
            evidence_id="e-1",
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="call-1",
            source_message_id="message-1",
        ),
        provenance_root_ids=["call-1"],
    )
    supported = candidate.model_copy(
        update={"state": "supported", "evidence_ids": ["e-1"]}
    )
    store.repository.record_hypothesis("task-a", supported)

    replayed = store.commit_experiment(
        state.version,
        event_id="experiment-event-1",
        event_payload="payload-1",
        result=result,
        assessment=assessment,
        next_state=state,
    )

    assert committed.hypotheses == [candidate]
    assert replayed.hypotheses == [supported]
