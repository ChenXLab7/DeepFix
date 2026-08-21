from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field, StringConstraints

from deepfix.models import ContextMetrics, Evidence
from deepfix.persistence import open_sqlite_connection

MemoryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
]
SummaryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class ProgressSnapshot(BaseModel):
    phase: Literal[
        "clarifying",
        "investigating",
        "planning",
        "editing",
        "testing",
        "reviewing",
    ]
    summary: SummaryText
    facts: list[MemoryText] = Field(default_factory=list, max_length=30)
    evidence: list[Evidence] = Field(default_factory=list, max_length=50)
    active_hypotheses: list[MemoryText] = Field(default_factory=list, max_length=10)
    rejected_hypotheses: list[MemoryText] = Field(default_factory=list, max_length=20)
    checked_files: list[MemoryText] = Field(default_factory=list, max_length=50)
    experiments: list[MemoryText] = Field(default_factory=list, max_length=30)
    next_steps: list[MemoryText] = Field(default_factory=list, max_length=10)
    unresolved_questions: list[MemoryText] = Field(default_factory=list, max_length=10)


@dataclass(frozen=True)
class WorkingMemoryVersion:
    task_id: str
    version: int
    snapshot: ProgressSnapshot
    created_at: str


class WorkingMemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(
        self,
        task_id: str,
        snapshot: ProgressSnapshot,
    ) -> WorkingMemoryVersion:
        normalized_task_id = self._normalize_task_id(task_id)
        created_at = self._now()
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM working_memory WHERE task_id = ?",
                    (normalized_task_id,),
                ).fetchone()
                version = int(row[0])
                connection.execute(
                    """
                    INSERT INTO working_memory(task_id, version, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        normalized_task_id,
                        version,
                        snapshot.model_dump_json(),
                        created_at,
                    ),
                )
                metrics = self._read_metrics(connection, normalized_task_id)
                metrics.working_memory_version = version
                self._write_metrics(connection, normalized_task_id, metrics, created_at)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return WorkingMemoryVersion(
            task_id=normalized_task_id,
            version=version,
            snapshot=snapshot,
            created_at=created_at,
        )

    def latest(self, task_id: str) -> WorkingMemoryVersion | None:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT version, payload, created_at
                FROM working_memory
                WHERE task_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (normalized_task_id,),
            ).fetchone()
        if row is None:
            return None
        return self._version_from_row(normalized_task_id, row)

    def list_versions(self, task_id: str) -> list[WorkingMemoryVersion]:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT version, payload, created_at
                FROM working_memory
                WHERE task_id = ?
                ORDER BY version ASC
                """,
                (normalized_task_id,),
            ).fetchall()
        return [self._version_from_row(normalized_task_id, row) for row in rows]

    def metrics(self, task_id: str) -> ContextMetrics:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            return self._read_metrics(connection, normalized_task_id)

    def record_peak_tokens(self, task_id: str, estimate: int) -> ContextMetrics:
        if estimate < 0:
            raise ValueError("token estimate 不能为负数")

        def update(metrics: ContextMetrics) -> None:
            metrics.context_peak_tokens = max(metrics.context_peak_tokens, estimate)

        return self._update_metrics(task_id, update)

    def record_overflow(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.context_overflow_count += 1

        return self._update_metrics(task_id, update)

    def record_compaction(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.active_compaction_count += 1
            metrics.last_compaction_at = self._now()

        return self._update_metrics(task_id, update)

    def _update_metrics(self, task_id: str, update) -> ContextMetrics:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                metrics = self._read_metrics(connection, normalized_task_id)
                update(metrics)
                self._write_metrics(
                    connection,
                    normalized_task_id,
                    metrics,
                    self._now(),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return metrics

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memory (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS context_metrics (
                    task_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connection(self) -> sqlite3.Connection:
        return open_sqlite_connection(self.database_path)

    @staticmethod
    def _read_metrics(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> ContextMetrics:
        row = connection.execute(
            "SELECT payload FROM context_metrics WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return ContextMetrics()
        payload = cast(dict[str, object], json.loads(row[0]))
        return ContextMetrics(**payload)

    @staticmethod
    def _write_metrics(
        connection: sqlite3.Connection,
        task_id: str,
        metrics: ContextMetrics,
        updated_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO context_metrics(task_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                json.dumps(asdict(metrics), ensure_ascii=False),
                updated_at,
            ),
        )

    @staticmethod
    def _version_from_row(
        task_id: str,
        row: sqlite3.Row | tuple[object, ...],
    ) -> WorkingMemoryVersion:
        return WorkingMemoryVersion(
            task_id=task_id,
            version=int(row[0]),
            snapshot=ProgressSnapshot.model_validate_json(str(row[1])),
            created_at=str(row[2]),
        )

    @staticmethod
    def _normalize_task_id(task_id: str) -> str:
        normalized = task_id.strip()
        if not normalized:
            raise ValueError("task_id 不能为空")
        return normalized

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds")
