import sqlite3

import pytest

from deepfix.config import ApprovalMode
from deepfix.models import Evidence, TaskState, TaskStatus
from deepfix.persistence import TaskRepository


def test_repository_saves_task(tmp_path):
    repository = TaskRepository(tmp_path / "state" / "deepfix.sqlite3")
    task = TaskState.create(tmp_path, "排序结果不稳定", ApprovalMode.MANUAL)
    task.evidence.append(Evidence("tests/test_sort.py:9", "相同元素顺序变化"))

    repository.save(task)

    restored = repository.get(task.task_id)
    assert restored == task
    assert isinstance(restored.evidence[0], Evidence)


def test_repository_updates_existing_task(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")
    task = TaskState.create(tmp_path, "排序结果不稳定", ApprovalMode.MANUAL)
    repository.save(task)
    task.transition_to(TaskStatus.INVESTIGATING)

    repository.save(task)

    assert repository.get(task.task_id).status is TaskStatus.INVESTIGATING
    assert [item.task_id for item in repository.list_recent()] == [task.task_id]


def test_repository_lists_most_recent_first_and_applies_limit(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")
    first = TaskState.create(tmp_path, "first", ApprovalMode.MANUAL)
    second = TaskState.create(tmp_path, "second", ApprovalMode.GUARDED)
    repository.save(first)
    repository.save(second)

    result = repository.list_recent(limit=1)

    assert [item.task_id for item in result] == [second.task_id]


def test_repository_raises_for_unknown_task(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")

    with pytest.raises(KeyError, match="missing-task"):
        repository.get("missing-task")


def test_checkpoint_connection_closes_after_context(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")

    with repository.checkpoint_connection() as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_checkpoint_connection_enables_wal_and_busy_timeout(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")

    with repository.checkpoint_connection() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert journal_mode == "wal"
    assert busy_timeout == 5000
