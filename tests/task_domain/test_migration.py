from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.database import SQLiteDatabase
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.repository import TaskRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "legacy_tasks.json"


@pytest.fixture
def historical_database(tmp_path) -> tuple[SQLiteDatabase, list[dict[str, object]]]:
    payloads = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            [
                (
                    payload["task_id"],
                    json.dumps(payload, ensure_ascii=False),
                    f"2026-08-01T00:00:0{index}+00:00",
                )
                for index, payload in enumerate(payloads)
            ],
        )
    return database, payloads


def test_historical_payloads_restore_into_bounded_authorities_without_rewrite(
    historical_database,
) -> None:
    database, payloads = historical_database
    before = _source_payloads(database)
    repository = TaskRepository(database)

    definitions = repository.list_recent_definitions(limit=10)

    assert {item.task_id for item in definitions} == {
        str(payload["task_id"]) for payload in payloads
    }
    assert _source_payloads(database) == before
    assert repository.get_lifecycle("legacy-created").status is TaskLifecycleStatus.CREATED
    assert repository.get_lifecycle("legacy-running").status is TaskLifecycleStatus.RUNNING
    assert (
        repository.get_lifecycle("legacy-approval").status is TaskLifecycleStatus.WAITING_APPROVAL
    )
    assert repository.get_lifecycle("legacy-paused").status is TaskLifecycleStatus.PAUSED
    assert repository.get_lifecycle("legacy-completed").status is TaskLifecycleStatus.COMPLETED
    decision = repository.latest_adjudication("legacy-completed")
    assert decision is not None
    assert decision.outcome == "fixed"


def test_historical_backfill_is_idempotent_and_preserves_original_message_identity(
    historical_database,
) -> None:
    database, _ = historical_database
    repository = TaskRepository(database)

    first = repository.get_definition("legacy-running")
    second = repository.get_definition("legacy-running")

    assert second == first
    assert first.original_message_id == stable_conversation_message_id(
        "legacy-running",
        0,
        "user",
        "repair parser branch",
    )
    assert repository.get_definition("legacy-completed").original_message_id == ("user-existing-id")


def test_historical_payload_is_not_exposed_as_runtime_projection(
    historical_database,
) -> None:
    database, _ = historical_database
    repository = TaskRepository(database)
    repository.get_definition("legacy-paused")

    assert not hasattr(repository, "get")
    assert not hasattr(repository, "save")
    assert not hasattr(repository, "save_legacy_projection")
    with database.connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM legacy_task_projection WHERE task_id = ?",
            ("legacy-paused",),
        ).fetchone()
    assert row is None


def test_historical_backfill_rolls_back_definition_when_adjudication_fails(
    historical_database,
) -> None:
    database, _ = historical_database
    repository = TaskRepository(database)
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_historical_adjudication
            BEFORE INSERT ON adjudication_decisions
            BEGIN
                SELECT RAISE(ABORT, 'simulated adjudication failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated adjudication failure"):
        repository.get_definition("legacy-completed")

    with database.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM task_definitions WHERE task_id = ?",
                ("legacy-completed",),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM task_lifecycle WHERE task_id = ?",
                ("legacy-completed",),
            ).fetchone()
            is None
        )


def _source_payloads(database: SQLiteDatabase) -> dict[str, str]:
    with database.connection() as connection:
        rows = connection.execute("SELECT task_id, payload FROM tasks ORDER BY task_id").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}
