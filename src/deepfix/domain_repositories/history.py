from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    ConflictRecord,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    ExperimentRecord,
    FileChangeEvidence,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    ProvenanceRef,
    ResearchStatusEvidence,
    SnapshotCoverage,
    StrictModel,
    SystemTestEvidence,
    UserConstraint,
)
from deepfix.compaction.snapshot import snapshot_content_hash
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    EvidenceRepository,
    restore_deterministic_evidence,
)
from deepfix.domain_repositories.investigation import (
    InvestigationRepository,
    stable_question_id,
)


class HistoryIdentityConflict(RuntimeError):
    pass


class HistoricalSemanticItem(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    snapshot_version: int = Field(ge=1)
    kind: Literal[
        "task_goal",
        "constraint",
        "fact",
        "evidence",
        "hypothesis",
        "experiment",
        "conflict",
        "unresolved_question",
        "next_step",
    ]
    semantic_id: str = Field(min_length=1)
    payload_type: str = Field(min_length=1)
    payload: dict[str, JsonValue]
    provenance_root_ids: list[str] = Field(default_factory=list)


class HistorySnapshotRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    previous_version: int | None = None
    lifecycle: Literal["prepared", "active", "abandoned"] = "prepared"
    input_hash: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    activated_at: str | None = None
    abandoned_at: str | None = None
    abandon_reason: str | None = None
    coverage: SnapshotCoverage
    source_work_unit_ids: list[str] = Field(default_factory=list)
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    content_hash: str = Field(min_length=1)
    semantic_items: list[HistoricalSemanticItem] = Field(
        default_factory=list,
        exclude=True,
    )


ArtifactVerifier = Callable[[ArtifactReference], bool]


class HistoryRepository:
    """Own compacted history and provenance, never current domain truth."""

    def __init__(
        self,
        database: SQLiteDatabase | str | Path,
        *,
        artifact_verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.database_path = self.database.path
        self.artifact_verifier = artifact_verifier
        self._initialize_schema()

    def save_prepared(self, record: HistorySnapshotRecord) -> HistorySnapshotRecord:
        if record.lifecycle != "prepared":
            raise ValueError("History Snapshot must be prepared before save")
        self._verify_artifacts(record.artifact_references)
        with self.database.unit_of_work(immediate=True) as connection:
            existing_row = connection.execute(
                """
                SELECT version FROM history_snapshots
                WHERE task_id = ? AND input_hash = ?
                """,
                (record.task_id, record.input_hash),
            ).fetchone()
            if existing_row is not None:
                existing = self._get(connection, record.task_id, int(existing_row[0]))
                if (
                    existing.content_hash != record.content_hash
                    or existing.coverage != record.coverage
                    or existing.source_work_unit_ids != record.source_work_unit_ids
                    or existing.artifact_references != record.artifact_references
                    or existing.semantic_items != record.semantic_items
                ):
                    raise HistoryIdentityConflict("history input identity conflict")
                return existing
            row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 FROM history_snapshots
                WHERE task_id = ?
                """,
                (record.task_id,),
            ).fetchone()
            version = int(row[0])
            saved = _rebase_record(record, version)
            connection.execute(
                """
                INSERT INTO history_snapshots(
                    task_id, version, previous_version, lifecycle, input_hash,
                    created_at, activated_at, abandoned_at, abandon_reason,
                    source_work_unit_ids, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    saved.task_id,
                    saved.version,
                    saved.previous_version,
                    saved.lifecycle,
                    saved.input_hash,
                    saved.created_at,
                    json.dumps(saved.source_work_unit_ids),
                    saved.content_hash,
                ),
            )
            connection.execute(
                """
                INSERT INTO history_coverage(task_id, version, payload)
                VALUES (?, ?, ?)
                """,
                (saved.task_id, saved.version, saved.coverage.model_dump_json()),
            )
            for ordinal, reference in enumerate(saved.artifact_references):
                connection.execute(
                    """
                    INSERT INTO history_artifact_refs(
                        task_id, version, ordinal, path, content_hash, payload
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        saved.task_id,
                        saved.version,
                        ordinal,
                        reference.path,
                        reference.content_hash,
                        reference.model_dump_json(),
                    ),
                )
            for item in saved.semantic_items:
                connection.execute(
                    """
                    INSERT INTO history_semantic_items(
                        task_id, version, item_id, kind, semantic_id,
                        payload_type, payload, provenance_roots
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.task_id,
                        item.snapshot_version,
                        item.item_id,
                        item.kind,
                        item.semantic_id,
                        item.payload_type,
                        json.dumps(
                            item.payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        json.dumps(item.provenance_root_ids),
                    ),
                )
            return saved

    def backfill(self, record: HistorySnapshotRecord) -> HistorySnapshotRecord:
        prepared = record.model_copy(
            update={
                "lifecycle": "prepared",
                "activated_at": None,
                "abandoned_at": None,
                "abandon_reason": None,
            }
        )
        saved = self.save_prepared(prepared)
        if record.lifecycle == "prepared":
            return saved
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._get_required(connection, record.task_id, saved.version)
            if current.lifecycle == record.lifecycle:
                return current
            if current.lifecycle != "prepared":
                raise HistoryIdentityConflict("history lifecycle backfill conflict")
            connection.execute(
                """
                UPDATE history_snapshots
                SET lifecycle = ?, activated_at = ?, abandoned_at = ?,
                    abandon_reason = ?
                WHERE task_id = ? AND version = ? AND lifecycle = 'prepared'
                """,
                (
                    record.lifecycle,
                    record.activated_at,
                    record.abandoned_at,
                    record.abandon_reason,
                    record.task_id,
                    saved.version,
                ),
            )
            return self._get_required(connection, record.task_id, saved.version)

    def get(self, task_id: str, version: int) -> HistorySnapshotRecord:
        with self.database.connection() as connection:
            record = self._get(connection, task_id, version)
        if record is None:
            raise KeyError((task_id, version))
        return record

    def list_for_task(self, task_id: str) -> list[HistorySnapshotRecord]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT version FROM history_snapshots
                WHERE task_id = ? ORDER BY version
                """,
                (task_id,),
            ).fetchall()
            return [self._get(connection, task_id, int(row[0])) for row in rows]

    def activate(
        self,
        task_id: str,
        version: int,
        event: DeepFixCompactionEvent,
    ) -> HistorySnapshotRecord:
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._get_required(connection, task_id, version)
            _validate_event_match(task_id, current, event)
            if current.lifecycle == "abandoned":
                raise ValueError("abandoned History Snapshot cannot become active")
            if current.lifecycle == "active":
                return current
            connection.execute(
                """
                UPDATE history_snapshots
                SET lifecycle = 'active', activated_at = datetime('now')
                WHERE task_id = ? AND version = ? AND lifecycle = 'prepared'
                """,
                (task_id, version),
            )
            return self._get_required(connection, task_id, version)

    def active_from_event(
        self,
        task_id: str,
        event: DeepFixCompactionEvent | None,
    ) -> HistorySnapshotRecord | None:
        if event is None:
            return None
        record = self.get(task_id, event.active_snapshot_version)
        _validate_event_match(task_id, record, event)
        return None if record.lifecycle == "abandoned" else record

    def abandon(
        self,
        task_id: str,
        version: int,
        reason: str,
    ) -> HistorySnapshotRecord:
        normalized = reason.strip()
        if not normalized:
            raise ValueError("abandon reason must not be empty")
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._get_required(connection, task_id, version)
            if current.lifecycle == "active":
                raise ValueError("active History Snapshot cannot be abandoned")
            if current.lifecycle == "abandoned":
                return current
            connection.execute(
                """
                UPDATE history_snapshots
                SET lifecycle = 'abandoned', abandoned_at = datetime('now'),
                    abandon_reason = ?
                WHERE task_id = ? AND version = ? AND lifecycle = 'prepared'
                """,
                (normalized[:300], task_id, version),
            )
            return self._get_required(connection, task_id, version)

    def project_snapshot(
        self,
        task_id: str,
        version: int,
        *,
        evidence: EvidenceRepository,
        investigation: InvestigationRepository,
    ) -> CompactionSnapshot:
        record = self.get(task_id, version)
        by_kind: dict[str, list[HistoricalSemanticItem]] = {}
        for item in record.semantic_items:
            by_kind.setdefault(item.kind, []).append(item)

        task_goal_items = by_kind.get("task_goal", [])
        task_goal = (
            str(task_goal_items[-1].payload.get("text", ""))
            if task_goal_items
            else f"Historical task {task_id}"
        )
        constraints = [
            UserConstraint.model_validate(item.payload)
            for item in by_kind.get("constraint", [])
        ]
        facts = {
            item.semantic_id: ProvenancedClaim.model_validate(item.payload)
            for item in by_kind.get("fact", [])
        }
        hypotheses = {
            item.semantic_id: HypothesisRecord.model_validate(item.payload)
            for item in by_kind.get("hypothesis", [])
        }
        experiments = [
            ExperimentRecord.model_validate(item.payload)
            for item in by_kind.get("experiment", [])
        ]
        conflicts = [
            ConflictRecord.model_validate(item.payload)
            for item in by_kind.get("conflict", [])
        ]
        questions = {
            item.semantic_id: ProvenancedText.model_validate(item.payload)
            for item in by_kind.get("unresolved_question", [])
        }
        next_steps = [
            ProvenancedText.model_validate(item.payload)
            for item in by_kind.get("next_step", [])
        ]

        deterministic: list[
            SystemTestEvidence
            | FileChangeEvidence
            | ApprovalEvidence
            | ResearchStatusEvidence
        ] = []
        for envelope in evidence.list_for_task(task_id):
            if envelope.kind in {
                EvidenceKind.TEST,
                EvidenceKind.FILE_CHANGE,
                EvidenceKind.APPROVAL,
                EvidenceKind.RESEARCH_STATUS,
            }:
                deterministic.append(restore_deterministic_evidence(envelope))
            elif envelope.kind is EvidenceKind.SEMANTIC_CLAIM:
                claim = ProvenancedClaim.model_validate(envelope.payload)
                facts[claim.claim_id] = claim

        for current in investigation.list_hypotheses(task_id):
            hypotheses[current.hypothesis_id] = HypothesisRecord(
                hypothesis_id=current.hypothesis_id,
                text=current.statement,
                state={
                    "candidate": "active",
                    "supported": "confirmed",
                    "rejected": "rejected",
                }[current.state],
                reason=current.reason,
                reopens_hypothesis_id=current.reopens_hypothesis_id,
                sources=[
                    ProvenanceRef(kind="system_evidence", ref_id=evidence_id)
                    for evidence_id in current.evidence_ids
                ],
                updated_in_version=record.version,
            )
        for question in investigation.list_questions(task_id):
            if question.status == "resolved":
                questions.pop(question.question_id, None)
                continue
            questions[question.question_id] = ProvenancedText(
                text=question.text,
                sources=[
                    ProvenanceRef(kind="snapshot_record", ref_id=source_id)
                    for source_id in question.source_ids
                ],
            )

        block = DeterministicEvidenceBlock(
            tests=[item for item in deterministic if isinstance(item, SystemTestEvidence)],
            files=[item for item in deterministic if isinstance(item, FileChangeEvidence)],
            approvals=[item for item in deterministic if isinstance(item, ApprovalEvidence)],
            research=[item for item in deterministic if isinstance(item, ResearchStatusEvidence)],
        )
        snapshot = CompactionSnapshot(
            task_id=record.task_id,
            version=record.version,
            previous_version=record.previous_version,
            lifecycle=record.lifecycle,
            created_at=record.created_at,
            activated_at=record.activated_at,
            abandoned_at=record.abandoned_at,
            abandon_reason=record.abandon_reason,
            source_work_unit_ids=record.source_work_unit_ids,
            coverage=record.coverage,
            task_goal=task_goal,
            user_constraints=constraints,
            confirmed_facts=list(facts.values()),
            deterministic_evidence=block,
            active_hypotheses=[item for item in hypotheses.values() if item.state == "active"],
            rejected_hypotheses=[
                item for item in hypotheses.values() if item.state == "rejected"
            ],
            confirmed_hypotheses=[
                item for item in hypotheses.values() if item.state == "confirmed"
            ],
            changed_files=block.files,
            experiments=experiments,
            test_results=block.tests,
            conflicts=conflicts,
            unresolved_questions=list(questions.values()),
            next_steps=next_steps,
            artifact_references=record.artifact_references,
            content_hash=record.content_hash,
        )
        return snapshot.model_copy(
            update={"content_hash": snapshot_content_hash(snapshot)}
        )

    def record_failure(self, failure: CompactionFailureRecord) -> None:
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT payload FROM compaction_history_failures
                WHERE task_id = ? AND attempt_id = ? AND stage = ?
                """,
                (failure.task_id, failure.attempt_id, failure.stage),
            ).fetchone()
            if row is not None:
                existing = CompactionFailureRecord.model_validate_json(str(row[0]))
                if existing != failure:
                    raise HistoryIdentityConflict("compaction failure identity conflict")
                return
            connection.execute(
                """
                INSERT INTO compaction_history_failures(
                    task_id, attempt_id, stage, payload, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    failure.task_id,
                    failure.attempt_id,
                    failure.stage,
                    failure.model_dump_json(),
                    failure.recorded_at,
                ),
            )

    def list_failures(self, task_id: str) -> list[CompactionFailureRecord]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM compaction_history_failures
                WHERE task_id = ? ORDER BY recorded_at, rowid
                """,
                (task_id,),
            ).fetchall()
        return [CompactionFailureRecord.model_validate_json(str(row[0])) for row in rows]

    def record_migration(
        self,
        task_id: str,
        version: int,
        event: DeepFixCompactionEvent,
    ) -> None:
        if version < 1 or event.task_id != task_id:
            raise ValueError("legacy migration metadata is invalid")
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT version, event_payload FROM context_history_migrations
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is not None:
                if int(row[0]) != version or str(row[1]) != event.model_dump_json():
                    raise HistoryIdentityConflict("context migration identity conflict")
                return
            connection.execute(
                """
                INSERT INTO context_history_migrations(
                    task_id, version, event_payload, migrated_at
                ) VALUES (?, ?, ?, datetime('now'))
                """,
                (task_id, version, event.model_dump_json()),
            )

    def migration_version(self, task_id: str) -> int | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT version FROM context_history_migrations WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        return None if row is None else int(row[0])

    def migrated_event(self, task_id: str) -> DeepFixCompactionEvent | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT event_payload FROM context_history_migrations WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        return (
            None
            if row is None
            else DeepFixCompactionEvent.model_validate_json(str(row[0]))
        )

    def _verify_artifacts(self, references: list[ArtifactReference]) -> None:
        if self.artifact_verifier is None:
            return
        for reference in references:
            if not self.artifact_verifier(reference):
                raise ValueError(f"Artifact verification failed: {reference.path}")

    @staticmethod
    def _get(connection, task_id: str, version: int) -> HistorySnapshotRecord | None:
        row = connection.execute(
            """
            SELECT previous_version, lifecycle, input_hash, created_at,
                   activated_at, abandoned_at, abandon_reason,
                   source_work_unit_ids, content_hash
            FROM history_snapshots WHERE task_id = ? AND version = ?
            """,
            (task_id, version),
        ).fetchone()
        if row is None:
            return None
        coverage_row = connection.execute(
            "SELECT payload FROM history_coverage WHERE task_id = ? AND version = ?",
            (task_id, version),
        ).fetchone()
        artifact_rows = connection.execute(
            """
            SELECT payload FROM history_artifact_refs
            WHERE task_id = ? AND version = ? ORDER BY ordinal
            """,
            (task_id, version),
        ).fetchall()
        semantic_rows = connection.execute(
            """
            SELECT item_id, kind, semantic_id, payload_type, payload, provenance_roots
            FROM history_semantic_items
            WHERE task_id = ? AND version = ? ORDER BY rowid
            """,
            (task_id, version),
        ).fetchall()
        return HistorySnapshotRecord(
            task_id=task_id,
            version=version,
            previous_version=None if row[0] is None else int(row[0]),
            lifecycle=str(row[1]),
            input_hash=str(row[2]),
            created_at=str(row[3]),
            activated_at=None if row[4] is None else str(row[4]),
            abandoned_at=None if row[5] is None else str(row[5]),
            abandon_reason=None if row[6] is None else str(row[6]),
            coverage=SnapshotCoverage.model_validate_json(str(coverage_row[0])),
            source_work_unit_ids=list(json.loads(str(row[7]))),
            artifact_references=[
                ArtifactReference.model_validate_json(str(item[0]))
                for item in artifact_rows
            ],
            content_hash=str(row[8]),
            semantic_items=[
                HistoricalSemanticItem(
                    item_id=str(item[0]),
                    task_id=task_id,
                    snapshot_version=version,
                    kind=str(item[1]),
                    semantic_id=str(item[2]),
                    payload_type=str(item[3]),
                    payload=json.loads(str(item[4])),
                    provenance_root_ids=list(json.loads(str(item[5]))),
                )
                for item in semantic_rows
            ],
        )

    @classmethod
    def _get_required(
        cls,
        connection,
        task_id: str,
        version: int,
    ) -> HistorySnapshotRecord:
        record = cls._get(connection, task_id, version)
        if record is None:
            raise KeyError((task_id, version))
        return record

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS history_snapshots (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    previous_version INTEGER,
                    lifecycle TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    abandoned_at TEXT,
                    abandon_reason TEXT,
                    source_work_unit_ids TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY(task_id, version),
                    UNIQUE(task_id, input_hash)
                );
                CREATE TABLE IF NOT EXISTS history_semantic_items (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    item_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    semantic_id TEXT NOT NULL,
                    payload_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    provenance_roots TEXT NOT NULL,
                    PRIMARY KEY(task_id, version, item_id)
                );
                CREATE TABLE IF NOT EXISTS history_artifact_refs (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(task_id, version, ordinal)
                );
                CREATE TABLE IF NOT EXISTS history_coverage (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(task_id, version)
                );
                CREATE TABLE IF NOT EXISTS compaction_history_failures (
                    task_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, attempt_id, stage)
                );
                CREATE TABLE IF NOT EXISTS context_history_migrations (
                    task_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    event_payload TEXT NOT NULL,
                    migrated_at TEXT NOT NULL
                );
                """
            )


def _rebase_record(
    record: HistorySnapshotRecord,
    version: int,
) -> HistorySnapshotRecord:
    if record.version == version:
        return record
    return record.model_copy(
        update={
            "version": version,
            "previous_version": version - 1 or None,
            "semantic_items": [
                item.model_copy(update={"snapshot_version": version})
                for item in record.semantic_items
            ],
        }
    )


def _validate_event_match(
    task_id: str,
    record: HistorySnapshotRecord,
    event: DeepFixCompactionEvent,
) -> None:
    if event.task_id != task_id or event.active_snapshot_version != record.version:
        raise ValueError("compaction event version does not match History Snapshot")
    if event.input_hash != record.input_hash:
        raise ValueError("compaction event input_hash does not match History Snapshot")
    if event.conversation_artifact not in record.artifact_references:
        raise ValueError("compaction event Artifact does not match History Snapshot")


def history_record_from_snapshot(
    snapshot: CompactionSnapshot,
    input_hash: str,
) -> HistorySnapshotRecord:
    items: list[HistoricalSemanticItem] = []

    def add_item(
        kind: str,
        semantic_id: str,
        value,
        roots: list[str],
    ) -> None:
        item_id = _stable_item_id(snapshot.task_id, kind, semantic_id)
        payload = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else {"text": str(value)}
        )
        items.append(
            HistoricalSemanticItem(
                item_id=item_id,
                task_id=snapshot.task_id,
                snapshot_version=snapshot.version,
                kind=kind,
                semantic_id=semantic_id,
                payload_type=type(value).__name__,
                payload=payload,
                provenance_root_ids=list(dict.fromkeys(roots)),
            )
        )

    add_item(
        "task_goal",
        "task_goal",
        snapshot.task_goal,
        [f"snapshot:{snapshot.version}"],
    )
    for item in snapshot.user_constraints:
        add_item(
            "constraint",
            item.constraint_id,
            item,
            [item.source_user_message_id],
        )
    for item in snapshot.confirmed_facts:
        add_item("fact", item.claim_id, item, _source_roots(item.sources))
    for item in [
        *snapshot.deterministic_evidence.tests,
        *snapshot.deterministic_evidence.files,
        *snapshot.deterministic_evidence.approvals,
        *snapshot.deterministic_evidence.research,
    ]:
        add_item(
            "evidence",
            item.evidence_id,
            item,
            _deterministic_roots(item),
        )
    for item in [
        *snapshot.active_hypotheses,
        *snapshot.rejected_hypotheses,
        *snapshot.confirmed_hypotheses,
    ]:
        add_item(
            "hypothesis",
            item.hypothesis_id,
            item,
            _source_roots(item.sources),
        )
    for item in snapshot.experiments:
        add_item("experiment", item.experiment_id, item, _source_roots(item.sources))
    for item in snapshot.conflicts:
        roots = [
            source.ref_id
            for alternative in item.alternatives
            for source in alternative.sources
        ]
        add_item("conflict", item.conflict_id, item, roots)
    for item in snapshot.unresolved_questions:
        semantic_id = stable_question_id(snapshot.task_id, item.text)
        add_item(
            "unresolved_question",
            semantic_id,
            item,
            _source_roots(item.sources),
        )
    for item in snapshot.next_steps:
        semantic_id = _text_id("next", snapshot.task_id, item.text)
        add_item("next_step", semantic_id, item, _source_roots(item.sources))
    return HistorySnapshotRecord(
        task_id=snapshot.task_id,
        version=snapshot.version,
        previous_version=snapshot.previous_version,
        lifecycle=snapshot.lifecycle,
        input_hash=input_hash,
        created_at=snapshot.created_at,
        activated_at=snapshot.activated_at,
        abandoned_at=snapshot.abandoned_at,
        abandon_reason=snapshot.abandon_reason,
        coverage=snapshot.coverage,
        source_work_unit_ids=snapshot.source_work_unit_ids,
        artifact_references=snapshot.artifact_references,
        content_hash=snapshot.content_hash,
        semantic_items=items,
    )


def _source_roots(sources: list[ProvenanceRef]) -> list[str]:
    return list(
        dict.fromkeys(f"{source.kind}:{source.ref_id}" for source in sources)
    )


def _deterministic_roots(
    item: SystemTestEvidence
    | FileChangeEvidence
    | ApprovalEvidence
    | ResearchStatusEvidence,
) -> list[str]:
    roots = [item.evidence_id]
    if isinstance(item, SystemTestEvidence):
        roots.extend([item.tool_call_id, item.source_message_id])
    elif isinstance(item, FileChangeEvidence):
        roots.extend(
            value
            for value in (item.tool_call_id, item.source_message_id)
            if value is not None
        )
    elif isinstance(item, ResearchStatusEvidence) and item.artifact_path is not None:
        roots.append(item.artifact_path)
    return list(dict.fromkeys(roots))


def _stable_item_id(task_id: str, kind: str, semantic_id: str) -> str:
    return _text_id("history", task_id, kind, semantic_id)


def _text_id(prefix: str, *parts: str) -> str:
    material = "\0".join((prefix, *parts))
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"
