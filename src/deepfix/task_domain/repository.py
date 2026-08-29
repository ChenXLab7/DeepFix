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
        try:
            with self.database.unit_of_work(immediate=True) as connection:
                result = self._create_definition(connection, definition)
        except sqlite3.IntegrityError as exc:
            raise TaskDefinitionConflict(
                "task definition identity conflict"
            ) from exc
        return result

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
            return self._transition_lifecycle(
                connection,
                task_id,
                next_status,
                reason=reason,
                expected_version=expected_version,
            )

    def save_legacy_projection(self, task) -> None:
        from deepfix.task_domain.migration import (
            legacy_payload_from_task,
            lifecycle_status_for_legacy,
            task_definition_from_legacy,
        )

        payload = legacy_payload_from_task(task)
        serialized = _canonical_json(payload)
        payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        updated_at = _now()
        with self.database.unit_of_work(immediate=True) as connection:
            existing_definition = self._get_definition(connection, task.task_id)
            definition = task_definition_from_legacy(
                task,
                created_at=(
                    existing_definition.created_at
                    if existing_definition is not None
                    else None
                ),
            )
            if existing_definition is not None:
                definition = definition.model_copy(
                    update={
                        "original_message_id": existing_definition.original_message_id,
                    }
                )
            self._create_definition(connection, definition)
            self._synchronize_legacy_lifecycle(
                connection,
                task.task_id,
                lifecycle_status_for_legacy(task),
                reason=task.pause_reason,
            )
            connection.execute(
                """
                INSERT INTO legacy_task_projection(
                    task_id, legacy_phase_status, legacy_paused_from,
                    payload, payload_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    legacy_phase_status = excluded.legacy_phase_status,
                    legacy_paused_from = excluded.legacy_paused_from,
                    payload = excluded.payload,
                    payload_hash = excluded.payload_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    task.task_id,
                    task.status.value,
                    task.paused_from.value if task.paused_from is not None else None,
                    serialized,
                    payload_hash,
                    updated_at,
                ),
            )
            stored_definition = self._get_definition(connection, task.task_id)
            stored_projection = connection.execute(
                "SELECT payload_hash FROM legacy_task_projection WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if (
                stored_definition is None
                or _definition_hash(stored_definition) != _definition_hash(definition)
                or stored_projection is None
                or str(stored_projection[0]) != payload_hash
            ):
                raise sqlite3.IntegrityError("task migration verification failed")

    def save(self, task) -> None:
        """Compatibility alias retained until Plan 4 removes TaskState writes."""
        self.save_legacy_projection(task)

    def get(self, task_id: str):
        from deepfix.task_domain.migration import reconstruct_task_state

        try:
            definition = self.get_definition(task_id)
        except KeyError:
            self._backfill_historical_task(task_id)
            definition = self.get_definition(task_id)
        lifecycle = self.get_lifecycle(task_id)
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT legacy_phase_status, legacy_paused_from, payload
                FROM legacy_task_projection WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return reconstruct_task_state(
            definition,
            lifecycle,
            json.loads(str(row[2])),
            legacy_phase_status=str(row[0]) if row[0] is not None else None,
            legacy_paused_from=str(row[1]) if row[1] is not None else None,
        )

    def list_recent(self, limit: int = 20) -> list:
        self._backfill_all_historical_tasks()
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT task_id FROM legacy_task_projection
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self.get(str(row[0])) for row in rows]

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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legacy_task_projection (
                    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
                    legacy_phase_status TEXT,
                    legacy_paused_from TEXT,
                    payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _synchronize_legacy_lifecycle(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        desired: TaskLifecycleStatus,
        *,
        reason: str | None,
    ) -> TaskLifecycle:
        current = self._get_lifecycle(connection, task_id)
        if current is None:
            raise KeyError(task_id)
        if current.status is desired:
            return current
        if current.status is TaskLifecycleStatus.CREATED and desired in {
            TaskLifecycleStatus.WAITING_APPROVAL,
            TaskLifecycleStatus.COMPLETED,
        }:
            current = self._transition_lifecycle(
                connection,
                task_id,
                TaskLifecycleStatus.RUNNING,
                expected_version=current.version,
            )
        return self._transition_lifecycle(
            connection,
            task_id,
            desired,
            reason=reason,
            expected_version=current.version,
        )

    def _create_definition(
        self,
        connection: sqlite3.Connection,
        definition: TaskDefinition,
    ) -> TaskDefinition:
        definition_hash = _definition_hash(definition)
        existing = self._get_definition(connection, definition.task_id)
        if existing is not None:
            if _definition_hash(existing) != definition_hash:
                raise TaskDefinitionConflict("task definition is immutable")
            return existing
        message_owner = connection.execute(
            "SELECT task_id FROM task_definitions WHERE original_message_id = ?",
            (definition.original_message_id,),
        ).fetchone()
        if message_owner is not None:
            raise TaskDefinitionConflict("original message already defines another task")
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
        return definition

    def _transition_lifecycle(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        next_status: TaskLifecycleStatus,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> TaskLifecycle:
        current = self._get_lifecycle(connection, task_id)
        if current is None:
            raise KeyError(task_id)
        if expected_version is not None and current.version != expected_version:
            raise TaskLifecycleConflict("task lifecycle version conflict")
        if next_status is current.status:
            return current
        validate_lifecycle_transition(current.status, next_status)
        paused_from = current.status if next_status is TaskLifecycleStatus.PAUSED else None
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
            raise TaskLifecycleConflict("task lifecycle version conflict")
        return TaskLifecycle(
            task_id=task_id,
            status=next_status,
            version=current.version + 1,
            paused_from=paused_from,
            reason=next_reason,
            updated_at=updated_at,
        )

    def _backfill_historical_task(self, task_id: str) -> None:
        from deepfix.models import TaskState

        with self.database.connection() as connection:
            if not _table_exists(connection, "tasks"):
                raise KeyError(task_id)
            row = connection.execute(
                "SELECT payload FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        self.save_legacy_projection(TaskState.from_dict(json.loads(str(row[0]))))

    def _backfill_all_historical_tasks(self) -> None:
        with self.database.connection() as connection:
            if not _table_exists(connection, "tasks"):
                return
            rows = connection.execute(
                """
                SELECT task_id FROM tasks
                WHERE task_id NOT IN (SELECT task_id FROM task_definitions)
                ORDER BY updated_at ASC, rowid ASC
                """
            ).fetchall()
        for row in rows:
            self._backfill_historical_task(str(row[0]))

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
    payload = _canonical_json(
        definition.model_dump(mode="json"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None
