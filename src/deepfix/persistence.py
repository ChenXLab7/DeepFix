from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from deepfix.models import TaskState


class TaskRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(self, task: TaskState) -> None:
        payload = json.dumps(task.to_dict(), ensure_ascii=False)
        updated_at = datetime.now(UTC).isoformat(timespec="microseconds")
        with self.checkpoint_connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks(task_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (task.task_id, payload, updated_at),
            )
            connection.commit()

    def get(self, task_id: str) -> TaskState:
        with self.checkpoint_connection() as connection:
            row = connection.execute(
                "SELECT payload FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        payload = cast(dict[str, object], json.loads(row[0]))
        return TaskState.from_dict(payload)

    def list_recent(self, limit: int = 20) -> list[TaskState]:
        with self.checkpoint_connection() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM tasks
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            TaskState.from_dict(cast(dict[str, object], json.loads(row[0])))
            for row in rows
        ]

    @contextmanager
    def checkpoint_connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.checkpoint_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
