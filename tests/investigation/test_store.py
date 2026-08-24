import pytest

from deepfix.investigation.models import AgentPhase, NewInvestigationEvent
from deepfix.investigation.store import InvestigationStateConflict, InvestigationStore


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
