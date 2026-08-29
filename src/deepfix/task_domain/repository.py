from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from deepfix.database import SQLiteDatabase
from deepfix.task_domain.models import (
    TaskDefinition,
    TaskDefinitionConflict,
    TaskLifecycle,
    TaskLifecycleConflict,
    TaskLifecycleStatus,
    validate_lifecycle_transition,
)


class TaskRepository:
    """Persist only bounded Task-domain state."""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.database_path = self.database.path
        self._initialize_schema()

    @contextmanager
    def checkpoint_connection(self) -> Iterator[sqlite3.Connection]:
        with self.database.connection() as connection:
            yield connection

    def create_definition(self, definition: TaskDefinition) -> TaskDefinition:
        definition_hash = _definition_hash(definition)
        try:
            with self.database.unit_of_work(immediate=True) as connection:
                existing = self._get_definition(connection, definition.task_id)
                if existing is not None:
                    if _definition_hash(existing) != definition_hash:
                        raise TaskDefinitionConflict(
                            "task definition is immutable"
                        )
                    return existing
                message_owner = connection.execute(
                    "SELECT task_id FROM task_definitions WHERE original_message_id = ?",
                    (definition.original_message_id,),
                ).fetchone()
                if message_owner is not None:
                    raise TaskDefinitionConflict(
                        "original message already defines another task"
                    )
                connection.execute(
                    """
                    INSERT INTO task_definitions(
                        task_id, original_message_id, original_problem,
                        approval_mode, source_project_root, workspace_root,
                        workspace_baseline_id, project_python, confinement_level,
                        created_at, definition_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        definition.task_id,
                        definition.original_message_id,
                        definition.original_problem,
                        definition.approval_mode,
                        definition.source_project_root,
                        definition.workspace_root,
                        definition.workspace_baseline_id,
                        definition.project_python,
                        definition.confinement_level,
                        definition.created_at,
                        definition_hash,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO task_lifecycle(
                        task_id, status, version, paused_from, reason, updated_at
                    ) VALUES (?, ?, 1, NULL, NULL, ?)
                    """,
                    (
                        definition.task_id,
                        TaskLifecycleStatus.CREATED.value,
                        definition.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise TaskDefinitionConflict(
                "task definition identity conflict"
            ) from exc
        return definition

    def get_definition(self, task_id: str) -> TaskDefinition:
        with self.database.connection() as connection:
            definition = self._get_definition(connection, task_id)
        if definition is None:
            raise KeyError(task_id)
        return definition

    def get_lifecycle(self, task_id: str) -> TaskLifecycle:
        with self.database.connection() as connection:
            lifecycle = self._get_lifecycle(connection, task_id)
        if lifecycle is None:
            raise KeyError(task_id)
        return lifecycle

    def transition_lifecycle(
        self,
        task_id: str,
        next_status: TaskLifecycleStatus,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> TaskLifecycle:
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._get_lifecycle(connection, task_id)
            if current is None:
                raise KeyError(task_id)
            if expected_version is not None and current.version != expected_version:
                raise TaskLifecycleConflict(
                    "task lifecycle version conflict"
                )
            if next_status is current.status:
                return current
            validate_lifecycle_transition(current.status, next_status)
            paused_from = (
                current.status if next_status is TaskLifecycleStatus.PAUSED else None
            )
            next_reason = reason if next_status is TaskLifecycleStatus.PAUSED else None
            updated_at = _now()
            cursor = connection.execute(
                """
                UPDATE task_lifecycle
                SET status = ?, version = version + 1,
                    paused_from = ?, reason = ?, updated_at = ?
                WHERE task_id = ? AND version = ?
                """,
                (
                    next_status.value,
                    paused_from.value if paused_from is not None else None,
                    next_reason,
                    updated_at,
                    task_id,
                    current.version,
                ),
            )
            if cursor.rowcount != 1:
                raise TaskLifecycleConflict(
                    "task lifecycle version conflict"
                )
            return TaskLifecycle(
                task_id=task_id,
                status=next_status,
                version=current.version + 1,
                paused_from=paused_from,
                reason=next_reason,
                updated_at=updated_at,
            )

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_definitions (
                    task_id TEXT PRIMARY KEY,
                    original_message_id TEXT NOT NULL UNIQUE,
                    original_problem TEXT NOT NULL,
                    approval_mode TEXT NOT NULL,
                    source_project_root TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    workspace_baseline_id TEXT,
                    project_python TEXT NOT NULL,
                    confinement_level TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    definition_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_lifecycle (
                    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    paused_from TEXT,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _get_definition(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> TaskDefinition | None:
        row = connection.execute(
            """
            SELECT task_id, original_message_id, original_problem, approval_mode,
                   source_project_root, workspace_root, workspace_baseline_id,
                   project_python, confinement_level, created_at
            FROM task_definitions WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return TaskDefinition(
            task_id=str(row[0]),
            original_message_id=str(row[1]),
            original_problem=str(row[2]),
            approval_mode=str(row[3]),
            source_project_root=str(row[4]),
            workspace_root=str(row[5]),
            workspace_baseline_id=str(row[6]) if row[6] is not None else None,
            project_python=str(row[7]),
            confinement_level=str(row[8]),
            created_at=str(row[9]),
        )

    @staticmethod
    def _get_lifecycle(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> TaskLifecycle | None:
        row = connection.execute(
            """
            SELECT task_id, status, version, paused_from, reason, updated_at
            FROM task_lifecycle WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return TaskLifecycle(
            task_id=str(row[0]),
            status=TaskLifecycleStatus(str(row[1])),
            version=int(row[2]),
            paused_from=(
                TaskLifecycleStatus(str(row[3])) if row[3] is not None else None
            ),
            reason=str(row[4]) if row[4] is not None else None,
            updated_at=str(row[5]),
        )


def _definition_hash(definition: TaskDefinition) -> str:
    payload = json.dumps(
        definition.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
