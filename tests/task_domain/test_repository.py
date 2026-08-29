from __future__ import annotations

import pytest

from deepfix.database import SQLiteDatabase
from deepfix.task_domain.models import (
    TaskDefinition,
    TaskDefinitionConflict,
    TaskLifecycleConflict,
    TaskLifecycleStatus,
)
from deepfix.task_domain.repository import TaskRepository


@pytest.fixture
def repository(tmp_path) -> TaskRepository:
    return TaskRepository(SQLiteDatabase(tmp_path / "deepfix.db"))


@pytest.fixture
def definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="task-1",
        original_message_id="message-1",
        original_problem="修复排序错误",
        approval_mode="manual",
        source_project_root="C:/repo",
        workspace_root="C:/workspaces/task-1",
        workspace_baseline_id="baseline-1",
        project_python="C:/Python/python.exe",
        confinement_level="guarded_local",
        created_at="2026-08-29T00:00:00+00:00",
    )


def test_create_definition_is_idempotent_but_rejects_mutation(
    repository,
    definition,
) -> None:
    assert repository.create_definition(definition) == definition
    assert repository.create_definition(definition) == definition

    changed = definition.model_copy(update={"original_problem": "changed"})
    with pytest.raises(TaskDefinitionConflict, match="immutable"):
        repository.create_definition(changed)


def test_original_message_cannot_define_two_tasks(repository, definition) -> None:
    repository.create_definition(definition)
    conflicting = definition.model_copy(update={"task_id": "task-2"})

    with pytest.raises(TaskDefinitionConflict, match="original message"):
        repository.create_definition(conflicting)


def test_definition_and_lifecycle_are_separate_rows(repository, definition) -> None:
    repository.create_definition(definition)

    assert repository.get_definition(definition.task_id) == definition
    lifecycle = repository.get_lifecycle(definition.task_id)
    assert lifecycle.status is TaskLifecycleStatus.CREATED
    assert lifecycle.version == 1
    assert lifecycle.paused_from is None


def test_lifecycle_uses_optimistic_version_and_transition_rules(
    repository,
    definition,
) -> None:
    repository.create_definition(definition)
    running = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
        expected_version=1,
    )

    assert running.version == 2
    assert running.status is TaskLifecycleStatus.RUNNING
    with pytest.raises(TaskLifecycleConflict, match="version"):
        repository.transition_lifecycle(
            definition.task_id,
            TaskLifecycleStatus.COMPLETED,
            expected_version=1,
        )


def test_pause_records_business_status_and_resume_source(repository, definition) -> None:
    repository.create_definition(definition)
    repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    paused = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.PAUSED,
        reason="需要人工输入",
    )
    resumed = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    assert paused.paused_from is TaskLifecycleStatus.RUNNING
    assert paused.reason == "需要人工输入"
    assert resumed.paused_from is None
    assert resumed.reason is None


def test_replaying_current_lifecycle_is_idempotent(repository, definition) -> None:
    repository.create_definition(definition)
    first = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    replay = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    assert replay == first


def test_normalized_task_tables_exclude_other_domains(repository) -> None:
    forbidden = {
        "messages",
        "todos",
        "hypotheses",
        "evidence",
        "test_results",
        "changed_files",
        "operations",
        "receipts",
        "approvals",
        "snapshots",
        "artifacts",
        "summary",
        "report",
    }

    with repository.checkpoint_connection() as connection:
        for table in ("task_definitions", "task_lifecycle"):
            columns = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert columns.isdisjoint(forbidden)


def test_unknown_task_raises_key_error(repository) -> None:
    with pytest.raises(KeyError, match="missing-task"):
        repository.get_definition("missing-task")
    with pytest.raises(KeyError, match="missing-task"):
        repository.get_lifecycle("missing-task")
