from __future__ import annotations

import json
import sqlite3

import pytest

from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.config import ApprovalMode
from deepfix.database import SQLiteDatabase
from deepfix.models import TaskState, TaskStatus
from deepfix.task_domain.models import (
    AdjudicationDecision,
    TaskDefinitionConflict,
    TaskLifecycleStatus,
)
from deepfix.task_domain.repository import TaskRepository
from deepfix.verification import VerificationPolicy


@pytest.fixture
def repository(tmp_path) -> TaskRepository:
    return TaskRepository(SQLiteDatabase(tmp_path / "deepfix.db"))


@pytest.fixture
def task(tmp_path) -> TaskState:
    value = TaskState.create(tmp_path, "修复排序错误", ApprovalMode.MANUAL)
    message_id = stable_conversation_message_id(
        value.task_id,
        0,
        "user",
        value.user_problem,
    )
    value.conversation.append(
        {"id": message_id, "role": "user", "content": value.user_problem}
    )
    value.transition_to(TaskStatus.INVESTIGATING)
    return value


def test_backfill_uses_original_message_and_is_idempotent(repository, task) -> None:
    repository.save_legacy_projection(task)
    first = repository.get_definition(task.task_id)

    repository.save_legacy_projection(task)
    second = repository.get_definition(task.task_id)

    assert first == second
    assert first.original_message_id == task.conversation[0]["id"]
    assert repository.get_lifecycle(task.task_id).status is TaskLifecycleStatus.RUNNING


def test_legacy_save_cannot_overwrite_immutable_problem(repository, task) -> None:
    repository.save_legacy_projection(task)
    task.user_problem = "模型改写后的目标"

    with pytest.raises(TaskDefinitionConflict, match="immutable"):
        repository.save_legacy_projection(task)


def test_unmigrated_fields_continue_to_round_trip(repository, task) -> None:
    task.hypotheses = ["H1"]
    task.changed_files = ["src/value.py"]
    task.pending_actions = [
        {"name": "execute", "args": {"command": "python -m pytest -q"}}
    ]

    repository.save_legacy_projection(task)
    restored = repository.get(task.task_id)

    assert restored.hypotheses == ["H1"]
    assert restored.changed_files == ["src/value.py"]
    assert restored.pending_actions == task.pending_actions


def test_legacy_payload_does_not_own_definition_or_lifecycle(repository, task) -> None:
    repository.save_legacy_projection(task)

    with repository.checkpoint_connection() as connection:
        row = connection.execute(
            "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()
    payload = json.loads(row[0])

    for key in (
        "task_id",
        "project_root",
        "user_problem",
        "approval_mode",
        "source_project_root",
        "workspace_root",
        "workspace_baseline_id",
        "project_python",
        "confinement_level",
        "status",
        "paused_from",
        "pause_reason",
        "verification_policy_id",
        "verification_policy_version",
        "resolution",
    ):
        assert key not in payload


def test_policy_and_adjudication_override_legacy_semantic_projection(
    repository,
    task,
) -> None:
    task.verification_policy_id = "stale-policy"
    task.verification_policy_version = 99
    task.resolution = "not_reproduced"
    repository.save_legacy_projection(task)
    policy = VerificationPolicy(
        policy_id="canonical-policy",
        task_id=task.task_id,
        version=1,
        required_oracles=[],
        supplemental_oracles=[],
        conflict_rules=[],
    )
    repository.save_verification_policy(policy)
    lifecycle = repository.get_lifecycle(task.task_id)
    repository.record_adjudication(
        AdjudicationDecision(
            decision_id="decision-1",
            task_id=task.task_id,
            outcome="fixed",
            evidence_ids=["evidence-1"],
            operation_ids=[],
            decided_at=lifecycle.updated_at,
        )
    )

    restored = repository.get(task.task_id)

    assert restored.verification_policy_id == "canonical-policy"
    assert restored.verification_policy_version == 1
    assert restored.resolution == "fixed"


def test_legacy_phase_changes_do_not_increment_business_lifecycle(
    repository,
    task,
) -> None:
    repository.save_legacy_projection(task)
    before = repository.get_lifecycle(task.task_id)
    task.transition_to(TaskStatus.EDITING)

    repository.save_legacy_projection(task)
    after = repository.get_lifecycle(task.task_id)

    assert after.status is TaskLifecycleStatus.RUNNING
    assert after.version == before.version
    assert repository.get(task.task_id).status is TaskStatus.EDITING


def test_paused_legacy_phase_round_trips_while_lifecycle_is_business_only(
    repository,
    task,
) -> None:
    repository.save_legacy_projection(task)
    task.transition_to(TaskStatus.PAUSED)

    repository.save_legacy_projection(task)
    restored = repository.get(task.task_id)
    lifecycle = repository.get_lifecycle(task.task_id)

    assert restored.status is TaskStatus.PAUSED
    assert restored.paused_from is TaskStatus.INVESTIGATING
    assert lifecycle.status is TaskLifecycleStatus.PAUSED
    assert lifecycle.paused_from is TaskLifecycleStatus.RUNNING


def test_historical_whole_payload_backfills_without_rewriting_source(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    historical = TaskState.create(tmp_path, "历史问题", ApprovalMode.GUARDED)
    historical.conversation = [{"role": "user", "content": "历史问题"}]
    historical.transition_to(TaskStatus.INVESTIGATING)
    historical_payload = json.dumps(historical.to_dict(), ensure_ascii=False)
    with database.connection() as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (historical.task_id, historical_payload, "2026-08-01T00:00:00+00:00"),
        )
        connection.commit()
    repository = TaskRepository(database)

    restored = repository.get(historical.task_id)
    first_definition = repository.get_definition(historical.task_id)
    restored.hypotheses = ["H1"]
    repository.save_legacy_projection(restored)

    with database.connection() as connection:
        source_payload = connection.execute(
            "SELECT payload FROM tasks WHERE task_id = ?",
            (historical.task_id,),
        ).fetchone()[0]
    assert source_payload == historical_payload
    assert repository.get_definition(historical.task_id) == first_definition
    assert repository.get(historical.task_id).hypotheses == ["H1"]


def test_historical_task_without_conversation_gets_stable_source_id(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    historical = TaskState.create(tmp_path, "没有历史消息", ApprovalMode.MANUAL)
    with database.connection() as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (
                historical.task_id,
                json.dumps(historical.to_dict(), ensure_ascii=False),
                "2026-08-01T00:00:00+00:00",
            ),
        )
        connection.commit()
    repository = TaskRepository(database)

    repository.get(historical.task_id)
    first = repository.get_definition(historical.task_id).original_message_id
    second = repository.get_definition(historical.task_id).original_message_id

    assert first == second
    assert first == stable_conversation_message_id(
        historical.task_id,
        0,
        "user",
        historical.user_problem,
    )


def test_historical_resolution_backfills_to_canonical_adjudication(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    historical = TaskState.create(tmp_path, "历史已完成任务", ApprovalMode.MANUAL)
    historical.resolution = "not_reproduced"
    historical.status = TaskStatus.COMPLETED
    payload = json.dumps(historical.to_dict(), ensure_ascii=False)
    with database.connection() as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (historical.task_id, payload, "2026-08-01T00:00:00+00:00"),
        )
        connection.commit()
    repository = TaskRepository(database)

    restored = repository.get(historical.task_id)

    assert restored.resolution == "not_reproduced"
    decision = repository.latest_adjudication(historical.task_id)
    assert decision is not None
    assert decision.outcome == "not_reproduced"


def test_first_backfill_is_atomic_when_projection_write_fails(
    repository,
    task,
) -> None:
    with repository.checkpoint_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_legacy_projection
            BEFORE INSERT ON legacy_task_projection
            BEGIN
                SELECT RAISE(ABORT, 'projection rejected');
            END
            """
        )
        connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="projection rejected"):
        repository.save_legacy_projection(task)

    with pytest.raises(KeyError):
        repository.get_definition(task.task_id)


def test_later_user_message_does_not_replace_fallback_original_message_id(
    repository,
    tmp_path,
) -> None:
    task = TaskState.create(tmp_path, "原始问题", ApprovalMode.MANUAL)
    repository.save_legacy_projection(task)
    original = repository.get_definition(task.task_id)
    task.conversation.append(
        {"id": "supplemental-message", "role": "user", "content": "Python 3.12"}
    )

    repository.save_legacy_projection(task)

    assert repository.get_definition(task.task_id) == original
