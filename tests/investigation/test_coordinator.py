import sqlite3

import pytest
from langchain_core.messages import ToolMessage

from deepfix.investigation.errors import InvestigationStateError
from deepfix.investigation.models import (
    AgentPhase,
    RecordHypothesisInput,
    ToolObservation,
)
from investigation.helpers import (
    coordinator_fixture,
    seed_checked_location,
    seed_evidence,
    supported_input,
)


def test_supported_hypothesis_rejects_cross_task_evidence(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    seed_evidence(coordinator.compaction_store, "task-b", "evidence-b")
    seed_checked_location(coordinator.store, "task-a", "src/sign.py", 1, 20)

    with pytest.raises(ValueError, match="当前任务"):
        coordinator.record_hypothesis(
            "task-a",
            supported_input(evidence_ids=["evidence-b"]),
            source_id="tool-hyp-1",
        )


def test_supported_hypothesis_unlocks_planning_with_stable_id(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    seed_evidence(coordinator.compaction_store, "task-a", "evidence-a")
    seed_checked_location(coordinator.store, "task-a", "src/sign.py", 1, 20)

    first = coordinator.record_hypothesis(
        "task-a",
        supported_input(evidence_ids=["evidence-a"]),
        source_id="tool-hyp-1",
    )
    replay = coordinator.record_hypothesis(
        "task-a",
        supported_input(evidence_ids=["evidence-a"]),
        source_id="tool-hyp-1",
    )

    assert first.hypothesis_id == replay.hypothesis_id
    assert coordinator.state("task-a").agent_phase is AgentPhase.PLANNING


def test_candidate_hypothesis_does_not_unlock_planning(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    command = RecordHypothesisInput(
        statement="the parser may change the sign",
        evidence_ids=[],
        checked_locations=[],
        target_state="candidate",
        reason="needs verification",
    )

    coordinator.record_hypothesis("task-a", command, source_id="tool-hyp-1")

    assert coordinator.state("task-a").agent_phase is AgentPhase.INVESTIGATING


def test_state_read_failure_becomes_typed_recovery_error(tmp_path, monkeypatch):
    coordinator = coordinator_fixture(tmp_path)

    def fail_load(task_id):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(coordinator.store, "load", fail_load)

    with pytest.raises(InvestigationStateError) as caught:
        coordinator.state("task-a")

    assert caught.value.recovery.error_code == "investigation_state_read_failed"
    assert caught.value.recovery.task_id == "task-a"


def test_tool_result_uses_shared_deterministic_evidence_identity(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    result = ToolMessage(
        id="msg-test-1",
        content="1 failed",
        tool_call_id="test-1",
        artifact={"exit_code": 1},
    )

    coordinator.record_tool_result(
        "task-a",
        {
            "name": "execute",
            "id": "test-1",
            "args": {"command": "python -m pytest -q"},
        },
        result,
    )

    evidence = coordinator.evidence_collector.store.list_evidence("task-a")
    assert coordinator.state("task-a").test_evidence_ids == [evidence[0].evidence_id]


def test_read_file_automatically_records_checked_range_without_strong_progress(
    tmp_path,
):
    coordinator = coordinator_fixture(tmp_path)
    before = coordinator.state("task-a")

    coordinator.record_tool_result(
        "task-a",
        {
            "name": "read_file",
            "id": "read-1",
            "args": {"file_path": "/src/sign.py", "offset": 10, "limit": 20},
        },
        ToolMessage(
            id="msg-read-1",
            content="return -value",
            tool_call_id="read-1",
        ),
    )

    after = coordinator.state("task-a")
    assert after.checked_files[0].ranges[0].start_line == 11
    assert after.checked_files[0].ranges[0].end_line == 11
    assert after.progress_generation == before.progress_generation


def test_originating_progress_increments_generation_but_phase_event_does_not(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    before = coordinator.state("task-a")
    observation = ToolObservation(
        event_type="test_observed",
        source_message_id="msg-test-1",
        tool_call_id="test-1",
        signature="pytest-q",
        result_fingerprint="failed-result",
        exit_code=1,
    )

    after = coordinator.record_observation("task-a", observation)

    assert after.progress_generation == before.progress_generation + 1
    events = coordinator.store.list_events("task-a")
    assert events[-2].progress_kind == "test_evidence"
    assert events[-1].event_type == "phase_changed"
    assert events[-1].progress_kind is None
