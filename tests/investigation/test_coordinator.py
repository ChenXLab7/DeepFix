import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import ToolMessage

from deepfix.investigation.errors import InvestigationStateError
from deepfix.investigation.models import (
    AgentPhase,
    RecordHypothesisInput,
    ToolObservation,
)
from deepfix.investigation.store import InvestigationStateConflict
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
    coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="artifact_read",
            tool_call_id="artifact-read",
            result_fingerprint="artifact-result",
            payload={"artifact_id": "artifact_" + "a" * 32},
        ),
    )

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
    assert coordinator.state("task-a").diagnostic_decision_required is False


def test_failed_post_edit_test_cannot_resupport_same_hypothesis(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    seed_evidence(coordinator.compaction_store, "task-a", "evidence-a")
    seed_checked_location(coordinator.store, "task-a", "src/sign.py", 1, 20)
    supported = coordinator.record_hypothesis(
        "task-a",
        supported_input(evidence_ids=["evidence-a"]),
        source_id="tool-hyp-supported",
    )
    coordinator.record_tool_result(
        "task-a",
        {
            "name": "edit_file",
            "id": "edit-artifactless",
            "args": {"file_path": "src/sign.py"},
        },
        ToolMessage(
            id="msg-edit-artifactless",
            content="Successfully replaced 1 instance(s) of the string in '/src/sign.py'",
            tool_call_id="edit-artifactless",
        ),
    )
    coordinator.record_tool_result(
        "task-a",
        {
            "name": "execute",
            "id": "pytest-post-edit",
            "args": {"command": "python -m pytest -q"},
        },
        ToolMessage(
            id="msg-pytest-post-edit",
            content="1 failed",
            tool_call_id="pytest-post-edit",
            artifact={"exit_code": 1},
        ),
    )
    command = supported_input(evidence_ids=["evidence-a"]).model_copy(
        update={"hypothesis_id": supported.hypothesis_id}
    )

    with pytest.raises(ValueError, match="修改后验证失败"):
        coordinator.record_hypothesis(
            "task-a",
            command,
            source_id="tool-hyp-resupported",
        )


def test_candidate_hypothesis_does_not_unlock_planning(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="artifact_read",
            tool_call_id="artifact-read",
            result_fingerprint="artifact-result",
            payload={"artifact_id": "artifact_" + "a" * 32},
        ),
    )
    command = RecordHypothesisInput(
        statement="the parser may change the sign",
        evidence_ids=[],
        checked_locations=[],
        target_state="candidate",
        reason="needs verification",
    )

    coordinator.record_hypothesis("task-a", command, source_id="tool-hyp-1")

    assert coordinator.state("task-a").agent_phase is AgentPhase.INVESTIGATING
    assert coordinator.state("task-a").diagnostic_decision_required is True


def test_repeated_candidate_without_evidence_reuses_semantic_identity(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    command = RecordHypothesisInput(
        statement="  capacity=0   may expose a boundary bug  ",
        evidence_ids=[],
        checked_locations=[],
        target_state="candidate",
        reason="needs a targeted test",
    )

    first = coordinator.record_hypothesis(
        "task-a",
        command,
        source_id="candidate-call-1",
    )
    version_after_first = coordinator.state("task-a").version
    replay = coordinator.record_hypothesis(
        "task-a",
        command.model_copy(
            update={"statement": "capacity=0 may expose a boundary bug"}
        ),
        source_id="candidate-call-2",
    )
    state = coordinator.state("task-a")

    assert replay.hypothesis_id == first.hypothesis_id
    assert state.hypotheses == [first]
    assert state.version == version_after_first


def test_candidate_hypothesis_resets_diagnostic_test_counter(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    for index in range(2):
        coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="test_observed",
                tool_call_id=f"diagnostic-test-{index}",
                result_fingerprint=f"diagnostic-result-{index}",
                exit_code=1,
            ),
        )
    command = RecordHypothesisInput(
        statement="the midpoint update may skip the remaining interval",
        evidence_ids=[],
        checked_locations=[],
        target_state="candidate",
        reason="needs a targeted experiment",
    )

    coordinator.record_hypothesis("task-a", command, source_id="tool-hyp-new")

    state = coordinator.state("task-a")
    assert state.diagnostic_test_count_since_decision == 0
    assert state.diagnostic_decision_required is True


def test_user_information_clears_diagnostic_repair_checkpoint(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="file_changed",
            tool_call_id="edit-before-user",
            result_fingerprint="edit-before-user-result",
        ),
    )
    coordinator.record_observation(
        "task-a",
        ToolObservation(
            event_type="post_edit_test_observed",
            tool_call_id="failed-test-before-user",
            result_fingerprint="failed-test-before-user-result",
            exit_code=1,
        ),
    )
    for index in range(2):
        coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="test_observed",
                tool_call_id=f"extra-test-before-user-{index}",
                result_fingerprint=f"extra-test-before-user-result-{index}",
                exit_code=1,
            ),
        )

    coordinator.record_user_information("task-a", "user-message-2")

    state = coordinator.state("task-a")
    assert state.diagnostic_test_count_since_decision == 0
    assert state.diagnostic_decision_required is False
    assert state.repair_reevaluation_required is False


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


def test_parallel_file_observations_retry_state_conflict_without_losing_updates(
    tmp_path,
    monkeypatch,
):
    coordinator = coordinator_fixture(tmp_path)
    coordinator.state("task-a")
    original_state = coordinator.state
    first_state_reads: set[int] = set()
    state_lock = threading.Lock()
    first_state_barrier = threading.Barrier(2)

    def synchronize_first_state_read(task_id):
        state = original_state(task_id)
        thread_id = threading.get_ident()
        with state_lock:
            first_read = thread_id not in first_state_reads
            first_state_reads.add(thread_id)
        if first_read:
            first_state_barrier.wait(timeout=5)
        return state

    monkeypatch.setattr(coordinator, "state", synchronize_first_state_read)

    def record(path: str, call_id: str):
        return coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="file_checked",
                tool_call_id=call_id,
                source_message_id=f"msg-{call_id}",
                signature=f"read:{path}",
                result_fingerprint=f"result:{path}",
                path=path,
                payload={
                    "path": path,
                    "start_line": 1,
                    "end_line": 20,
                    "content_fingerprint": f"content:{path}",
                },
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(record, "python_programs/mergesort.py", "read-source"),
            executor.submit(
                record,
                "python_testcases/test_mergesort.py",
                "read-test",
            ),
        ]
        results = [future.result(timeout=10) for future in futures]

    monkeypatch.setattr(coordinator, "state", original_state)
    final_state = coordinator.state("task-a")
    assert {item.path for item in final_state.checked_files} == {
        "python_programs/mergesort.py",
        "python_testcases/test_mergesort.py",
    }
    assert len(
        [
            event
            for event in coordinator.store.list_events("task-a")
            if event.event_type == "file_checked"
        ]
    ) == 2
    assert max(result.version for result in results) == final_state.version


def test_concurrent_replay_returns_state_containing_committed_observation(
    tmp_path,
    monkeypatch,
):
    coordinator = coordinator_fixture(tmp_path)
    initial = coordinator.state("task-a")
    original_state = coordinator.state
    replay_loaded = threading.Event()
    first_committed = threading.Event()

    def pause_replay_after_state_load(task_id):
        state = original_state(task_id)
        if threading.current_thread().name.startswith("replay"):
            replay_loaded.set()
            assert first_committed.wait(timeout=5)
        return state

    monkeypatch.setattr(coordinator, "state", pause_replay_after_state_load)
    observation = ToolObservation(
        event_type="file_checked",
        tool_call_id="read-source",
        source_message_id="msg-read-source",
        signature="read:source",
        result_fingerprint="result:source",
        path="python_programs/mergesort.py",
        payload={
            "path": "python_programs/mergesort.py",
            "start_line": 1,
            "end_line": 20,
            "content_fingerprint": "content:source",
        },
    )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay") as executor:
        replay = executor.submit(
            coordinator.record_observation,
            "task-a",
            observation,
        )
        assert replay_loaded.wait(timeout=5)
        committed = coordinator.record_observation("task-a", observation)
        first_committed.set()
        replayed = replay.result(timeout=5)

    assert committed.version == initial.version + 1
    assert replayed.version == committed.version
    assert replayed.checked_files == committed.checked_files


def test_event_precheck_failure_becomes_typed_recovery_error(
    tmp_path,
    monkeypatch,
):
    coordinator = coordinator_fixture(tmp_path)

    def fail_has_event(task_id, event_id):
        raise OSError("disk")

    monkeypatch.setattr(coordinator.store, "has_event", fail_has_event)

    with pytest.raises(InvestigationStateError) as caught:
        coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="file_checked",
                tool_call_id="read-source",
                source_message_id="msg-read-source",
                signature="read:source",
                result_fingerprint="result:source",
                path="python_programs/mergesort.py",
                payload={
                    "path": "python_programs/mergesort.py",
                    "start_line": 1,
                    "end_line": 20,
                    "content_fingerprint": "content:source",
                },
            ),
        )

    assert caught.value.recovery.error_code == "investigation_state_commit_failed"


def test_conflict_retry_exhaustion_reports_fresh_recovery_state(
    tmp_path,
    monkeypatch,
):
    coordinator = coordinator_fixture(tmp_path)
    initial = coordinator.state("task-a")
    stale = initial.model_copy(update={"version": 1})
    fresh = initial.model_copy(update={"version": 9})
    reads = iter([stale, stale, stale, stale, fresh])

    monkeypatch.setattr(coordinator, "state", lambda task_id: next(reads))
    monkeypatch.setattr(
        coordinator.store,
        "commit",
        lambda expected_version, events, next_state: (_ for _ in ()).throw(
            InvestigationStateConflict("conflict")
        ),
    )

    with pytest.raises(InvestigationStateError) as caught:
        coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="file_checked",
                tool_call_id="read-source",
                source_message_id="msg-read-source",
                signature="read:source",
                result_fingerprint="result:source",
                path="python_programs/mergesort.py",
                payload={
                    "path": "python_programs/mergesort.py",
                    "start_line": 1,
                    "end_line": 20,
                    "content_fingerprint": "content:source",
                },
            ),
        )

    assert caught.value.recovery.state_version == fresh.version


def test_non_conflict_commit_failure_is_not_retried(tmp_path, monkeypatch):
    coordinator = coordinator_fixture(tmp_path)
    coordinator.state("task-a")
    calls = 0

    def fail_commit(expected_version, events, next_state):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(coordinator.store, "commit", fail_commit)

    with pytest.raises(InvestigationStateError) as caught:
        coordinator.record_observation(
            "task-a",
            ToolObservation(
                event_type="file_checked",
                tool_call_id="read-source",
                source_message_id="msg-read-source",
                signature="read:source",
                result_fingerprint="result:source",
                path="python_programs/mergesort.py",
                payload={
                    "path": "python_programs/mergesort.py",
                    "start_line": 1,
                    "end_line": 20,
                    "content_fingerprint": "content:source",
                },
            ),
        )

    assert calls == 1
    assert caught.value.recovery.error_code == "investigation_state_commit_failed"


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


def artifact_search_result(
    call_id: str,
    *,
    content_hash: str = "b" * 64,
) -> ToolMessage:
    return ToolMessage(
        id=f"msg-{call_id}",
        content=f"bounded excerpt {call_id}",
        tool_call_id=call_id,
        status="success",
        artifact={
            "result_type": "diagnostic_artifact_search",
            "artifact_ids": ["artifact_" + "a" * 32],
            "match_count": 1,
            "searched_artifact_count": 2,
            "omitted_artifact_count": 0,
            "truncated": False,
            "content_hashes": [content_hash],
            "query_terms_hash": "c" * 64,
            "backend_path": "/.deepfix-artifacts/private",
            "query": "secret query",
        },
    )


def test_successful_artifact_search_records_whitelisted_no_progress_event(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    before = coordinator.state("task-a")
    result = artifact_search_result("search-1")

    after = coordinator.record_tool_result(
        "task-a",
        {
            "name": "search_diagnostic_artifacts",
            "id": "search-1",
            "args": {"query": "secret query"},
        },
        result,
    )

    event = coordinator.store.list_events("task-a")[-1]
    assert event.event_type == "artifact_searched"
    assert event.source_message_id == "msg-search-1"
    assert event.tool_call_id == "search-1"
    assert event.progress_kind is None
    assert event.payload == {
        "artifact_ids": ["artifact_" + "a" * 32],
        "match_count": 1,
        "searched_artifact_count": 2,
        "omitted_artifact_count": 0,
        "truncated": False,
        "content_hashes": ["b" * 64],
        "query_terms_hash": "c" * 64,
    }
    assert "secret" not in str(event.payload)
    assert "backend_path" not in event.payload
    assert after.progress_generation == before.progress_generation
    assert after.no_progress_count == before.no_progress_count + 1


def test_successful_artifact_read_records_range_without_evidence(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    result = ToolMessage(
        id="msg-read-artifact-1",
        content="10: failure",
        tool_call_id="read-artifact-1",
        status="success",
        artifact={
            "result_type": "diagnostic_artifact_read",
            "artifact_id": "artifact_" + "a" * 32,
            "kind": "large_tool_result",
            "start_line": 10,
            "end_line": 12,
            "total_lines": 20,
            "content_hash": "b" * 64,
            "truncated": True,
            "content": "must not persist",
        },
    )

    state = coordinator.record_tool_result(
        "task-a",
        {
            "name": "read_diagnostic_artifact",
            "id": "read-artifact-1",
            "args": {"artifact_id": "artifact_" + "a" * 32},
        },
        result,
    )

    event = coordinator.store.list_events("task-a")[-1]
    assert event.event_type == "artifact_read"
    assert event.payload == {
        "artifact_id": "artifact_" + "a" * 32,
        "kind": "large_tool_result",
        "start_line": 10,
        "end_line": 12,
        "total_lines": 20,
        "content_hash": "b" * 64,
        "truncated": True,
    }
    assert state.test_evidence_ids == []


def test_artifact_event_replay_is_idempotent(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    call = {
        "name": "search_diagnostic_artifacts",
        "id": "search-replay",
        "args": {"query": "failure"},
    }
    result = artifact_search_result("search-replay")

    first = coordinator.record_tool_result("task-a", call, result)
    replay = coordinator.record_tool_result("task-a", call, result)

    events = [
        item
        for item in coordinator.store.list_events("task-a")
        if item.event_type == "artifact_searched"
    ]
    assert len(events) == 1
    assert replay.version == first.version
    assert replay.no_progress_count == first.no_progress_count


def test_six_distinct_artifact_searches_trigger_existing_stagnation_gate(tmp_path):
    coordinator = coordinator_fixture(tmp_path)

    for index in range(6):
        call_id = f"search-{index}"
        coordinator.record_tool_result(
            "task-a",
            {
                "name": "search_diagnostic_artifacts",
                "id": call_id,
                "args": {"query": f"failure-{index}"},
            },
            artifact_search_result(
                call_id,
                content_hash=f"{index:x}" * 64,
            ),
        )

    state = coordinator.state("task-a")
    assert state.no_progress_count == 6
    assert state.progress_generation == 0
    assert state.stagnation_level == 1
    assert state.reevaluation_required is True


def test_error_artifact_tool_message_is_not_recorded_as_successful_retrieval(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    result = ToolMessage(
        id="msg-search-error",
        content="no matches",
        tool_call_id="search-error",
        status="error",
        artifact={
            "result_type": "diagnostic_artifact_error",
            "error_code": "artifact_no_matches",
        },
    )

    coordinator.record_tool_result(
        "task-a",
        {
            "name": "search_diagnostic_artifacts",
            "id": "search-error",
            "args": {"query": "failure"},
        },
        result,
    )

    assert coordinator.store.list_events("task-a")[-1].event_type == "tool_completed"
