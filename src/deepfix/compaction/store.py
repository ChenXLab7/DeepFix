from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from deepfix.compaction.models import (
    ApprovalEvidence,
    CompactionFailureRecord,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    FileChangeEvidence,
    ResearchStatusEvidence,
    SystemTestEvidence,
)
from deepfix.compaction.snapshot import snapshot_content_hash
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    EvidenceRepository,
    deterministic_provenance_roots,
    restore_deterministic_evidence,
)
from deepfix.domain_repositories.migration import domain_is_switched
from deepfix.persistence import open_sqlite_connection

DeterministicEvidence: TypeAlias = (
    SystemTestEvidence | FileChangeEvidence | ApprovalEvidence | ResearchStatusEvidence
)

_EVIDENCE_TYPES: dict[str, type[BaseModel]] = {
    "test": SystemTestEvidence,
    "file": FileChangeEvidence,
    "approval": ApprovalEvidence,
    "research": ResearchStatusEvidence,
}


class CompactionStore:
    def __init__(self, database_path: SQLiteDatabase | str | Path) -> None:
        self.database = (
            database_path
            if isinstance(database_path, SQLiteDatabase)
            else SQLiteDatabase(database_path)
        )
        self.database_path = self.database.path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.evidence_repository = EvidenceRepository(self.database)

    def save_evidence(self, task_id: str, evidence: DeterministicEvidence) -> None:
        self.evidence_repository.record_deterministic(
            _required(task_id, "task_id"),
            evidence,
            provenance_root_ids=deterministic_provenance_roots(evidence),
        )

    def list_evidence(self, task_id: str) -> list[DeterministicEvidence]:
        task_id = _required(task_id, "task_id")
        current = {
            item.evidence_id: restore_deterministic_evidence(item)
            for item in self.evidence_repository.list_for_task(task_id)
            if item.kind in {
                EvidenceKind.TEST,
                EvidenceKind.FILE_CHANGE,
                EvidenceKind.APPROVAL,
                EvidenceKind.RESEARCH_STATUS,
            }
        }
        if domain_is_switched(self.database, "deterministic_evidence", task_id):
            return list(current.values())
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT kind, payload FROM deterministic_evidence
                WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        legacy = [
            _EVIDENCE_TYPES[str(kind)].model_validate_json(str(payload))
            for kind, payload in rows
        ]
        merged = {item.evidence_id: item for item in legacy}
        merged.update(current)
        return list(merged.values())

    def save_prepared_snapshot(
        self,
        snapshot: CompactionSnapshot,
        input_hash: str,
    ) -> CompactionSnapshot:
        if snapshot.lifecycle != "prepared":
            raise ValueError("只能保存 prepared Snapshot")
        input_hash = _required(input_hash, "input_hash")
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT payload FROM compaction_snapshots
                    WHERE task_id = ? AND input_hash = ?
                    """,
                    (snapshot.task_id, input_hash),
                ).fetchone()
                if existing is not None:
                    connection.rollback()
                    return CompactionSnapshot.model_validate_json(str(existing[0]))
                row = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM compaction_snapshots WHERE task_id = ?",
                    (snapshot.task_id,),
                ).fetchone()
                expected_version = int(row[0])
                if snapshot.version != expected_version:
                    snapshot = _rebase_snapshot(snapshot, expected_version)
                connection.execute(
                    """
                    INSERT INTO compaction_snapshots(
                        task_id, version, lifecycle, input_hash, payload, created_at,
                        activated_at, abandoned_at, abandon_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (
                        snapshot.task_id,
                        snapshot.version,
                        snapshot.lifecycle,
                        input_hash,
                        snapshot.model_dump_json(),
                        snapshot.created_at,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_snapshot(snapshot.task_id, snapshot.version)

    def get_snapshot(self, task_id: str, version: int) -> CompactionSnapshot:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT payload FROM compaction_snapshots
                WHERE task_id = ? AND version = ?
                """,
                (_required(task_id, "task_id"), version),
            ).fetchone()
        if row is None:
            raise KeyError((task_id, version))
        return CompactionSnapshot.model_validate_json(str(row[0]))

    def active_snapshot_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent | None,
    ) -> CompactionSnapshot | None:
        if event is None:
            return None
        if event.task_id != task_id:
            raise ValueError("compaction event task_id 不匹配")
        snapshot = self.get_snapshot(task_id, event.active_snapshot_version)
        return None if snapshot.lifecycle == "abandoned" else snapshot

    def activate_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent,
    ) -> CompactionSnapshot:
        snapshot = self.active_snapshot_from_event(task_id, event)
        if snapshot is None:
            raise ValueError("event 不能激活 abandoned Snapshot")
        if snapshot.lifecycle == "active":
            return snapshot
        activated = snapshot.model_copy(
            update={"lifecycle": "active", "activated_at": _utc_now()}
        )
        self._update_snapshot_lifecycle(activated)
        return activated

    def abandon_snapshot(
        self,
        task_id: str,
        version: int,
        reason: str,
    ) -> CompactionSnapshot:
        snapshot = self.get_snapshot(task_id, version)
        if snapshot.lifecycle == "active":
            raise ValueError("active Snapshot 不能 abandoned")
        if snapshot.lifecycle == "abandoned":
            return snapshot
        abandoned = snapshot.model_copy(
            update={
                "lifecycle": "abandoned",
                "abandoned_at": _utc_now(),
                "abandon_reason": _bounded_reason(reason),
            }
        )
        self._update_snapshot_lifecycle(abandoned)
        return abandoned

    def record_failure(self, failure: CompactionFailureRecord) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO compaction_failures(task_id, attempt_id, stage, payload, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id, attempt_id, stage) DO NOTHING
                """,
                (
                    failure.task_id,
                    failure.attempt_id,
                    failure.stage,
                    failure.model_dump_json(),
                    failure.recorded_at,
                ),
            )
            connection.commit()

    def list_failures(self, task_id: str) -> list[CompactionFailureRecord]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload FROM compaction_failures
                WHERE task_id = ? ORDER BY recorded_at, rowid
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [CompactionFailureRecord.model_validate_json(str(row[0])) for row in rows]

    def list_snapshots(self, task_id: str) -> list[CompactionSnapshot]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload FROM compaction_snapshots
                WHERE task_id = ? ORDER BY version
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [CompactionSnapshot.model_validate_json(str(row[0])) for row in rows]

    def migration_version(self, task_id: str) -> int | None:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT version FROM context_migrations WHERE task_id = ?",
                (_required(task_id, "task_id"),),
            ).fetchone()
        return None if row is None else int(row[0])

    def migrated_event(self, task_id: str) -> DeepFixCompactionEvent | None:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT event_payload FROM context_migrations WHERE task_id = ?",
                (_required(task_id, "task_id"),),
            ).fetchone()
        return (
            None
            if row is None
            else DeepFixCompactionEvent.model_validate_json(str(row[0]))
        )

    def record_migration(
        self,
        task_id: str,
        version: int,
        event: DeepFixCompactionEvent,
    ) -> None:
        task_id = _required(task_id, "task_id")
        if version < 1 or event.task_id != task_id:
            raise ValueError("legacy migration 元数据无效")
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO context_migrations(task_id, version, event_payload, migrated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO NOTHING
                """,
                (task_id, version, event.model_dump_json(), _utc_now()),
            )
            connection.commit()

    def _update_snapshot_lifecycle(self, snapshot: CompactionSnapshot) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE compaction_snapshots
                SET lifecycle = ?, payload = ?, activated_at = ?,
                    abandoned_at = ?, abandon_reason = ?
                WHERE task_id = ? AND version = ?
                """,
                (
                    snapshot.lifecycle,
                    snapshot.model_dump_json(),
                    snapshot.activated_at,
                    snapshot.abandoned_at,
                    snapshot.abandon_reason,
                    snapshot.task_id,
                    snapshot.version,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError((snapshot.task_id, snapshot.version))
            connection.commit()

    def _initialize(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS deterministic_evidence (
                    task_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, evidence_id)
                );
                CREATE TABLE IF NOT EXISTS compaction_snapshots (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    lifecycle TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    abandoned_at TEXT,
                    abandon_reason TEXT,
                    PRIMARY KEY(task_id, version),
                    UNIQUE(task_id, input_hash)
                );
                CREATE TABLE IF NOT EXISTS compaction_failures (
                    task_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, attempt_id, stage)
                );
                CREATE TABLE IF NOT EXISTS context_migrations (
                    task_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    event_payload TEXT NOT NULL,
                    migrated_at TEXT NOT NULL
                );
                """
            )
            connection.commit()


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    return normalized


def _bounded_reason(value: str) -> str:
    normalized = _required(value, "abandon reason")
    return normalized if len(normalized) <= 300 else normalized[:299] + "…"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _rebase_snapshot(
    snapshot: CompactionSnapshot,
    version: int,
) -> CompactionSnapshot:
    old_version = snapshot.version

    def rebase_hypotheses(items):
        return [
            item.model_copy(update={"updated_in_version": version})
            if item.updated_in_version == old_version
            else item
            for item in items
        ]

    rebased = snapshot.model_copy(
        update={
            "version": version,
            "active_hypotheses": rebase_hypotheses(snapshot.active_hypotheses),
            "rejected_hypotheses": rebase_hypotheses(
                snapshot.rejected_hypotheses
            ),
            "confirmed_hypotheses": rebase_hypotheses(
                snapshot.confirmed_hypotheses
            ),
        }
    )
    return rebased.model_copy(
        update={"content_hash": snapshot_content_hash(rebased)}
    )
