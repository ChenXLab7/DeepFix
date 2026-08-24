from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from deepfix.investigation.models import (
    InvestigationEvent,
    InvestigationState,
    NewInvestigationEvent,
)
from deepfix.persistence import open_sqlite_connection


class InvestigationStateConflict(RuntimeError):
    pass


class InvestigationStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def load(self, task_id: str) -> InvestigationState | None:
        with open_sqlite_connection(self.database_path) as connection:
            return self._load(connection, task_id)

    def ensure_started(self, task_id: str) -> InvestigationState:
        existing = self.load(task_id)
        if existing is not None:
            return existing
        state = InvestigationState.new(task_id)
        event = NewInvestigationEvent.task_started(task_id)
        try:
            return self.commit(0, [event], state)
        except InvestigationStateConflict:
            loaded = self.load(task_id)
            if loaded is None:
                raise
            return loaded

    def commit(
        self,
        expected_version: int,
        events: list[NewInvestigationEvent],
        next_state: InvestigationState,
    ) -> InvestigationState:
        if any(event.task_id != next_state.task_id for event in events):
            raise InvestigationStateConflict("事件与物化状态的任务不一致")

        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load(connection, next_state.task_id)
                current_version = 0 if current is None else current.version
                existing = self._existing_events(connection, next_state.task_id, events)
                if len(existing) == len(events):
                    self._validate_replay(events, existing)
                    connection.rollback()
                    if current is None:
                        raise InvestigationStateConflict("重放事件缺少物化状态")
                    return current
                if existing or current_version != expected_version:
                    raise InvestigationStateConflict("investigation state 版本冲突")

                committed = next_state.model_copy(update={"version": current_version + 1})
                self._insert_events(connection, events)
                self._upsert_state(connection, committed)
                connection.commit()
                return committed
            except Exception:
                connection.rollback()
                raise

    def list_events(self, task_id: str) -> list[InvestigationEvent]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload, sequence, created_at
                FROM investigation_events
                WHERE task_id = ?
                ORDER BY sequence
                """,
                (task_id,),
            ).fetchall()
        return [
            InvestigationEvent.model_validate(
                {
                    **cast(dict[str, Any], json.loads(row[0])),
                    "sequence": row[1],
                    "created_at": row[2],
                }
            )
            for row in rows
        ]

    def last_sequence(self, task_id: str) -> int:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM investigation_events
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        return int(row[0])

    def has_event(self, task_id: str, event_id: str) -> bool:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM investigation_events
                WHERE task_id = ? AND event_id = ?
                """,
                (task_id, event_id),
            ).fetchone()
        return row is not None

    def _initialize(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_events (
                    task_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, event_id),
                    UNIQUE (task_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_state (
                    task_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def _load(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> InvestigationState | None:
        row = connection.execute(
            "SELECT payload FROM investigation_state WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return InvestigationState.model_validate_json(row[0])

    @staticmethod
    def _existing_events(
        connection: sqlite3.Connection,
        task_id: str,
        events: list[NewInvestigationEvent],
    ) -> dict[str, str]:
        if not events:
            return {}
        event_ids = [event.event_id for event in events]
        placeholders = ",".join("?" for _ in event_ids)
        rows = connection.execute(
            f"""
            SELECT event_id, payload
            FROM investigation_events
            WHERE task_id = ? AND event_id IN ({placeholders})
            """,
            (task_id, *event_ids),
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    @staticmethod
    def _validate_replay(
        events: list[NewInvestigationEvent],
        existing: dict[str, str],
    ) -> None:
        for event in events:
            if existing[event.event_id] != _canonical_event(event):
                raise InvestigationStateConflict("相同 event_id 的事件内容冲突")

    @staticmethod
    def _insert_events(
        connection: sqlite3.Connection,
        events: list[NewInvestigationEvent],
    ) -> None:
        if not events:
            return
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0)
            FROM investigation_events
            WHERE task_id = ?
            """,
            (events[0].task_id,),
        ).fetchone()
        first_sequence = int(row[0]) + 1
        created_at = datetime.now(UTC).isoformat(timespec="microseconds")
        connection.executemany(
            """
            INSERT INTO investigation_events(
                task_id, event_id, sequence, event_type, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event.task_id,
                    event.event_id,
                    first_sequence + offset,
                    event.event_type.value,
                    _canonical_event(event),
                    created_at,
                )
                for offset, event in enumerate(events)
            ],
        )

    @staticmethod
    def _upsert_state(
        connection: sqlite3.Connection,
        state: InvestigationState,
    ) -> None:
        updated_at = datetime.now(UTC).isoformat(timespec="microseconds")
        connection.execute(
            """
            INSERT INTO investigation_state(task_id, version, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                version = excluded.version,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (state.task_id, state.version, state.model_dump_json(), updated_at),
        )


def _canonical_event(event: NewInvestigationEvent) -> str:
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
