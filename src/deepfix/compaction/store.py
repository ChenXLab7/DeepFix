from __future__ import annotations

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
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    EvidenceRepository,
    deterministic_provenance_roots,
    restore_deterministic_evidence,
)
from deepfix.domain_repositories.history import (
    HistoryRepository,
    history_record_from_snapshot,
)
from deepfix.domain_repositories.investigation import InvestigationRepository
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
    def __init__(
        self,
        database_path: SQLiteDatabase | str | Path,
        *,
        repositories=None,
    ) -> None:
        self.database = (
            repositories.database
            if repositories is not None
            else (
                database_path
                if isinstance(database_path, SQLiteDatabase)
                else SQLiteDatabase(database_path)
            )
        )
        self.database_path = self.database.path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.evidence_repository = (
            repositories.evidence
            if repositories is not None
            else EvidenceRepository(self.database)
        )
        self.investigation_repository = (
            repositories.investigation
            if repositories is not None
            else InvestigationRepository(self.database)
        )
        self.history_repository = (
            repositories.history
            if repositories is not None
            else HistoryRepository(self.database)
        )

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
        record = history_record_from_snapshot(
            snapshot,
            _required(input_hash, "input_hash"),
        )
        saved = self.history_repository.save_prepared(record)
        return self.get_snapshot(saved.task_id, saved.version)

    def get_snapshot(self, task_id: str, version: int) -> CompactionSnapshot:
        return self.history_repository.project_snapshot(
            _required(task_id, "task_id"),
            version,
            evidence=self.evidence_repository,
            investigation=self.investigation_repository,
        )

    def active_snapshot_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent | None,
    ) -> CompactionSnapshot | None:
        if event is None:
            return None
        record = self.history_repository.active_from_event(task_id, event)
        return None if record is None else self.get_snapshot(task_id, record.version)

    def activate_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent,
    ) -> CompactionSnapshot:
        self.history_repository.activate(task_id, event.active_snapshot_version, event)
        return self.get_snapshot(task_id, event.active_snapshot_version)

    def abandon_snapshot(
        self,
        task_id: str,
        version: int,
        reason: str,
    ) -> CompactionSnapshot:
        self.history_repository.abandon(task_id, version, _bounded_reason(reason))
        return self.get_snapshot(task_id, version)

    def record_failure(self, failure: CompactionFailureRecord) -> None:
        self.history_repository.record_failure(failure)

    def list_failures(self, task_id: str) -> list[CompactionFailureRecord]:
        return self.history_repository.list_failures(_required(task_id, "task_id"))

    def list_snapshots(self, task_id: str) -> list[CompactionSnapshot]:
        return [
            self.get_snapshot(task_id, record.version)
            for record in self.history_repository.list_for_task(
                _required(task_id, "task_id")
            )
        ]

    def migration_version(self, task_id: str) -> int | None:
        return self.history_repository.migration_version(
            _required(task_id, "task_id")
        )

    def migrated_event(self, task_id: str) -> DeepFixCompactionEvent | None:
        return self.history_repository.migrated_event(_required(task_id, "task_id"))

    def record_migration(
        self,
        task_id: str,
        version: int,
        event: DeepFixCompactionEvent,
    ) -> None:
        self.history_repository.record_migration(
            _required(task_id, "task_id"), version, event
        )

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
