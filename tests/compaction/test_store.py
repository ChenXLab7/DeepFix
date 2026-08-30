from deepfix.compaction.models import (
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.models import TaskState
from deepfix.persistence import TaskRepository, open_sqlite_connection
from deepfix.research.store import ResearchEvidenceStore


def _snapshot(version: int, *, input_marker: str):
    return CompactionSnapshot(
        task_id="task-a",
        version=version,
        previous_version=version - 1 or None,
        lifecycle="prepared",
        created_at=f"2026-08-22T00:00:0{version}+00:00",
        source_work_unit_ids=[],
        task_goal="修复符号错误",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[
            ArtifactReference(
                path="/.deepfix-artifacts/conversation_history/task-a.md",
                kind="conversation_history",
                content_hash="c" * 64,
                work_unit_ids=[],
            )
        ],
        content_hash=input_marker * 64,
    )


def _event(version: int):
    return DeepFixCompactionEvent(
        event_id=f"event-{version}",
        task_id="task-a",
        active_snapshot_version=version,
        snapshot_message_id=f"snapshot-message-{version}",
        retained_message_ids=[],
        conversation_artifact=ArtifactReference(
            path="/.deepfix-artifacts/conversation_history/task-a.md",
            kind="conversation_history",
            content_hash="c" * 64,
            work_unit_ids=[],
        ),
        input_hash=f"input-{version}",
    )


def test_initialization_preserves_task_and_research_tables(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    repository = TaskRepository(database)
    task = TaskState.create(tmp_path, "修复错误", ApprovalMode.MANUAL)
    repository.save(task)
    ResearchEvidenceStore(database)

    store = CompactionStore(database)

    assert repository.get(task.task_id).user_problem == "修复错误"
    assert store.list_evidence("other-task") == []


def test_deterministic_evidence_is_idempotent_and_task_scoped(tmp_path):
    store = CompactionStore(tmp_path / "deepfix.sqlite3")
    evidence = SystemTestEvidence(
        evidence_id="test-c1",
        command="pytest -q",
        exit_code=1,
        summary="1 failed",
        tool_call_id="c1",
        source_message_id="m2",
    )

    store.save_evidence("task-a", evidence)
    store.save_evidence("task-a", evidence)

    assert store.list_evidence("task-a") == [evidence]
    assert store.list_evidence("task-b") == []


def test_event_not_latest_row_selects_and_activates_snapshot(tmp_path):
    store = CompactionStore(tmp_path / "deepfix.sqlite3")
    first = store.save_prepared_snapshot(_snapshot(1, input_marker="a"), "input-1")
    store.save_prepared_snapshot(_snapshot(2, input_marker="b"), "input-2")

    selected = store.active_snapshot_from_event("task-a", _event(1))
    active = store.activate_from_event("task-a", _event(1))

    assert selected.version == 1
    assert active.lifecycle == "active"
    assert active.content_hash == first.content_hash
    assert store.get_snapshot("task-a", 2).lifecycle == "prepared"


def test_abandoned_snapshot_never_resolves_as_active(tmp_path):
    store = CompactionStore(tmp_path / "deepfix.sqlite3")
    store.save_prepared_snapshot(_snapshot(1, input_marker="a"), "input-1")

    abandoned = store.abandon_snapshot("task-a", 1, "input changed")

    assert abandoned.lifecycle == "abandoned"
    assert abandoned.abandon_reason == "input changed"
    assert store.active_snapshot_from_event("task-a", _event(1)) is None


def test_new_prepared_snapshot_advances_past_abandoned_version(tmp_path):
    store = CompactionStore(tmp_path / "deepfix.sqlite3")
    store.save_prepared_snapshot(_snapshot(1, input_marker="a"), "input-1")
    store.abandon_snapshot("task-a", 1, "superseded")

    saved = store.save_prepared_snapshot(
        _snapshot(1, input_marker="b"),
        "input-2",
    )

    assert saved.version == 2
    assert saved.lifecycle == "prepared"


def test_failure_records_are_idempotent_by_attempt_and_stage(tmp_path):
    store = CompactionStore(tmp_path / "deepfix.sqlite3")
    failure = CompactionFailureRecord(
        attempt_id="attempt-1",
        task_id="task-a",
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="artifact_write",
        error_code="artifact_write_failed",
        input_hash="d" * 64,
        original_messages_preserved=True,
        recorded_at="2026-08-22T00:00:00+00:00",
    )

    store.record_failure(failure)
    store.record_failure(failure)

    assert store.list_failures("task-a") == [failure]


def test_compaction_store_writes_history_authority_not_legacy_snapshot_table(
    tmp_path,
):
    database = tmp_path / "deepfix.sqlite3"
    store = CompactionStore(database)

    saved = store.save_prepared_snapshot(_snapshot(1, input_marker="a"), "input-1")

    assert saved.version == 1
    with open_sqlite_connection(database) as connection:
        history_count = connection.execute(
            "SELECT COUNT(*) FROM history_snapshots"
        ).fetchone()[0]
        legacy_count = connection.execute(
            "SELECT COUNT(*) FROM compaction_snapshots"
        ).fetchone()[0]
    assert history_count == 1
    assert legacy_count == 0
