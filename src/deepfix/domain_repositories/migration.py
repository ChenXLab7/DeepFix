from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field

from deepfix.compaction.models import (
    ApprovalEvidence,
    FileChangeEvidence,
    ResearchStatusEvidence,
    StrictModel,
    SystemTestEvidence,
)
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    EvidenceRepository,
    deterministic_provenance_roots,
    restore_deterministic_evidence,
)


class DomainMigrationReport(StrictModel):
    domain: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_mismatches: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)
    missing_references: list[str] = Field(default_factory=list)
    ready_to_switch: bool


class DomainMigrator:
    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.evidence = EvidenceRepository(self.database)
        self._initialize_schema()

    def migrate_deterministic_evidence(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("deterministic_evidence", task_id)
        if existing is not None:
            return existing
        source = self._legacy_rows(task_id)
        for _evidence_id, kind, payload in source:
            evidence = _restore_legacy(kind, payload)
            self.evidence.record_deterministic(
                task_id,
                evidence,
                provenance_root_ids=deterministic_provenance_roots(evidence),
            )
        target = [
            item
            for item in self.evidence.list_for_task(task_id)
            if item.kind in _DETERMINISTIC_KINDS
        ]
        source_by_id = {
            evidence_id: _canonical_payload(json.loads(payload))
            for evidence_id, _kind, payload in source
        }
        target_by_id = {
            item.evidence_id: _canonical_payload(
                restore_deterministic_evidence(item).model_dump(mode="json")
            )
            for item in target
        }
        source_ids = set(source_by_id)
        target_ids = set(target_by_id)
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            evidence_id
            for evidence_id in source_ids & target_ids
            if source_by_id[evidence_id] != target_by_id[evidence_id]
        )
        source_projection = [
            (evidence_id, source_by_id[evidence_id])
            for evidence_id in sorted(source_by_id)
        ]
        target_projection = [
            (evidence_id, target_by_id[evidence_id])
            for evidence_id in sorted(target_by_id)
        ]
        report = DomainMigrationReport(
            domain="deterministic_evidence",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source_projection),
            target_hash=_canonical_hash(target_projection),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=[],
            ready_to_switch=(
                len(source) == len(target)
                and not identity_mismatches
                and not hash_mismatches
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    def _legacy_rows(self, task_id: str) -> list[tuple[str, str, str]]:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'deterministic_evidence'
                """
            ).fetchone()
            if table is None:
                return []
            rows = connection.execute(
                """
                SELECT evidence_id, kind, payload
                FROM deterministic_evidence
                WHERE task_id = ? ORDER BY evidence_id
                """,
                (task_id,),
            ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def _load_report(self, domain: str, task_id: str) -> DomainMigrationReport | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT report_json FROM domain_migrations
                WHERE domain = ? AND task_id = ?
                """,
                (domain, task_id),
            ).fetchone()
        return None if row is None else DomainMigrationReport.model_validate_json(str(row[0]))

    def _save_report(self, report: DomainMigrationReport) -> None:
        with self.database.unit_of_work(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO domain_migrations(domain, task_id, report_json, switched_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(domain, task_id) DO NOTHING
                """,
                (report.domain, report.task_id, report.model_dump_json()),
            )

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_migrations (
                    domain TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    switched_at TEXT NOT NULL,
                    PRIMARY KEY(domain, task_id)
                )
                """
            )


_LEGACY_TYPES = {
    "test": SystemTestEvidence,
    "file": FileChangeEvidence,
    "approval": ApprovalEvidence,
    "research": ResearchStatusEvidence,
}

_DETERMINISTIC_KINDS = {
    EvidenceKind.TEST,
    EvidenceKind.FILE_CHANGE,
    EvidenceKind.APPROVAL,
    EvidenceKind.RESEARCH_STATUS,
}


def domain_is_switched(
    database: SQLiteDatabase,
    domain: str,
    task_id: str,
) -> bool:
    with database.connection() as connection:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'domain_migrations'
            """
        ).fetchone()
        if table is None:
            return False
        row = connection.execute(
            """
            SELECT 1 FROM domain_migrations WHERE domain = ? AND task_id = ?
            """,
            (domain, task_id),
        ).fetchone()
    return row is not None


def _restore_legacy(kind: str, payload: str):
    model = _LEGACY_TYPES.get(kind)
    if model is None:
        raise ValueError(f"Unsupported legacy deterministic Evidence kind: {kind}")
    return model.model_validate_json(payload)


def _canonical_payload(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value).encode("utf-8")).hexdigest()


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized
