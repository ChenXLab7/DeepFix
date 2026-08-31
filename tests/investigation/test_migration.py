from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.compaction.store import CompactionStore
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.investigation.migration import (
    InvestigationMigrationMiddleware,
    InvestigationMigrator,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository


def _migrator_fixture(
    tmp_path: Path,
    status: TaskStatus,
    *,
    memory_phase: str | None = None,
    active_hypothesis: str | None = None,
) -> InvestigationMigrator:
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    tasks.save(
        TaskState(
            task_id="task-a",
            project_root=str(tmp_path),
            project_python=sys.executable,
            user_problem="legacy bug",
            approval_mode="manual",
            status=status,
        )
    )
    if memory_phase is not None:
        sqlite = SQLiteDatabase(database)
        payload = {
            "phase": memory_phase,
            "summary": "legacy memory",
            "facts": [],
            "evidence": [],
            "active_hypotheses": (
                []
                if active_hypothesis is None
                else [
                    {
                        "hypothesis_id": "hyp-legacy",
                        "text": active_hypothesis,
                        "state": "active",
                        "reason": "legacy candidate",
                        "sources": [],
                        "updated_in_version": 1,
                    }
                ]
            ),
            "rejected_hypotheses": [],
            "confirmed_hypotheses": [],
            "checked_files": [],
            "experiments": [],
            "next_steps": [],
            "unresolved_questions": [],
            "coverage": {"covered_message_ids": [], "covered_work_unit_ids": []},
        }
        with sqlite.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE working_memory (
                    task_id TEXT NOT NULL, version INTEGER NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, version)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO working_memory(task_id, version, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                ("task-a", 1, json.dumps(payload), "2026-08-31T00:00:00+00:00"),
            )
        assert DomainMigrator(sqlite).migrate_working_memory("task-a").ready_to_switch
    return InvestigationMigrator(
        tasks=tasks,
        store=InvestigationStore(database),
        compaction_store=CompactionStore(database),
    )


def _tool_pair(
    name: str,
    call_id: str,
    args: dict[str, object],
    artifact: dict[str, object],
) -> list[AnyMessage]:
    return [
        AIMessage(
            id=f"ai-{call_id}",
            content="",
            tool_calls=[{"name": name, "id": call_id, "args": args, "type": "tool_call"}],
        ),
        ToolMessage(
            id=f"tool-{call_id}",
            content="result",
            tool_call_id=call_id,
            artifact=artifact,
        ),
    ]


def _pytest_pair(exit_code: int, call_id: str = "pytest-1") -> list[AnyMessage]:
    return _tool_pair(
        "execute",
        call_id,
        {"command": "python -m pytest -q"},
        {"exit_code": exit_code},
    )


def _edit_pair() -> list[AnyMessage]:
    return _tool_pair(
        "edit_file",
        "edit-1",
        {"file_path": "/src/sign.py"},
        {"operation": "edit", "status": "succeeded", "path": "/src/sign.py"},
    )


@pytest.mark.parametrize(
    ("messages", "task_status", "expected_observation_count"),
    [
        ([], TaskStatus.INVESTIGATING, 0),
        (_pytest_pair(1), TaskStatus.TESTING, 1),
        (
            [*_pytest_pair(1), *_edit_pair()],
            TaskStatus.EDITING,
            2,
        ),
        (
            [*_edit_pair(), *_pytest_pair(0)],
            TaskStatus.TESTING,
            2,
        ),
        ([], TaskStatus.CLARIFYING, 0),
    ],
)
def test_legacy_messages_migrate_durable_evidence_without_phase(
    tmp_path: Path,
    messages: list[AnyMessage],
    task_status: TaskStatus,
    expected_observation_count: int,
) -> None:
    migrator = _migrator_fixture(tmp_path, task_status)

    state = migrator.migrate("task-a", messages)

    assert len(migrator.store.list_events("task-a")) == 1 + expected_observation_count
    assert "agent_phase" not in type(state).model_fields


def test_legacy_hypothesis_moves_to_domain_store_without_unlocking_runtime_state(
    tmp_path: Path,
) -> None:
    migrator = _migrator_fixture(
        tmp_path,
        TaskStatus.INVESTIGATING,
        memory_phase="planning",
        active_hypothesis="maybe cache",
    )

    state = migrator.migrate("task-a", [])

    assert state.supported_hypothesis_ids == []
    assert state.hypotheses == []
    repository = DomainMigrator(SQLiteDatabase(tmp_path / "deepfix.sqlite3")).investigation
    assert [(item.statement, item.state) for item in repository.list_hypotheses("task-a")] == [
        ("maybe cache", "candidate")
    ]


def test_unpaired_execution_and_approval_do_not_count_as_execution(
    tmp_path: Path,
) -> None:
    migrator = _migrator_fixture(tmp_path, TaskStatus.EDITING)
    messages = [
        *_edit_pair(),
        AIMessage(
            id="ai-unpaired",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "id": "pytest-unpaired",
                    "args": {"command": "pytest -q"},
                    "type": "tool_call",
                }
            ],
        ),
    ]

    state = migrator.migrate("task-a", messages)

    assert state.test_evidence_ids == []


def test_migration_replay_is_idempotent(tmp_path: Path) -> None:
    migrator = _migrator_fixture(tmp_path, TaskStatus.TESTING)
    first = migrator.migrate("task-a", _pytest_pair(1))
    event_ids = [item.event_id for item in migrator.store.list_events("task-a")]

    second = migrator.migrate("task-a", _pytest_pair(1))

    assert second == first
    assert [item.event_id for item in migrator.store.list_events("task-a")] == event_ids


def test_middleware_migrates_using_runtime_task_id(tmp_path: Path) -> None:
    migrator = _migrator_fixture(tmp_path, TaskStatus.TESTING)
    middleware = InvestigationMigrationMiddleware(migrator)
    runtime = Runtime(
        execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="agent-1",
            thread_id="task-a",
        )
    )

    update = middleware.before_agent({"messages": _pytest_pair(1)}, runtime)

    assert update is None
    assert len(migrator.store.list_events("task-a")) == 2
