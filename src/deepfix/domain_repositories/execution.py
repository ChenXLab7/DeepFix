from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ConfigDict, Field

from deepfix.compaction.models import ArtifactReference, StrictModel
from deepfix.database import SQLiteDatabase
from deepfix.investigation.receipts import ToolExecutionReceipt
from deepfix.operations import (
    NewOperationEntry,
    OperationConflictError,
    OperationJournalEntry,
    OperationStateSnapshot,
    OperationStatus,
    OperationTransitionError,
    _prepared_identity,
)


class ExecutionIdentityConflict(RuntimeError):
    """A stable execution identity was replayed with different content."""


class ExecutionApproval(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    risk: str = Field(min_length=1)
    source_tool_call_id: str | None = None
    created_at: str = Field(min_length=1)


def create_execution_approval(
    *,
    task_id: str,
    operation: str,
    decision: str,
    risk: str,
    source_tool_call_id: str | None = None,
    legacy_ordinal: int | None = None,
    created_at: str | None = None,
) -> ExecutionApproval:
    normalized_operation = " ".join(operation.split())
    source_identity = source_tool_call_id or (
        f"legacy:{legacy_ordinal}" if legacy_ordinal is not None else ""
    )
    if not source_identity:
        raise ValueError("approval requires source_tool_call_id or legacy_ordinal")
    semantic = json.dumps(
        [task_id, source_identity, decision, risk, normalized_operation],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    approval_id = f"approval_{hashlib.sha256(semantic.encode('utf-8')).hexdigest()[:32]}"
    return ExecutionApproval(
        approval_id=approval_id,
        task_id=task_id,
        operation=normalized_operation,
        decision=decision,
        risk=risk,
        source_tool_call_id=source_tool_call_id,
        created_at=created_at or _now().isoformat(),
    )


class ExecutionIntegrity(StrictModel):
    status_counts: dict[str, int] = Field(default_factory=dict)
    receipt_count: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    incomplete_operation_ids: list[str] = Field(default_factory=list)
    unknown_operation_ids: list[str] = Field(default_factory=list)

    @property
    def has_unknown_operations(self) -> bool:
        return bool(self.unknown_operation_ids)


ArtifactVerifier = Callable[[ArtifactReference], bool]


class ExecutionRepository:
    """Own Operation, Receipt, and Approval execution facts."""

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

    def prepare(self, entry: NewOperationEntry) -> OperationJournalEntry:
        with self.database.unit_of_work(immediate=True) as connection:
            existing = self._load_operation(connection, entry.operation_id)
            if existing is not None:
                if _prepared_identity(existing) != entry:
                    raise OperationConflictError("operation prepare conflict")
                return existing
            now = _now()
            prepared = OperationJournalEntry(
                **entry.model_dump(mode="python"),
                status=OperationStatus.PREPARED,
                created_at=now,
                updated_at=now,
            )
            self._insert_operation(connection, prepared, [])
            return prepared

    def backfill_operation(
        self,
        entry: OperationJournalEntry,
        *,
        artifact_references: list[ArtifactReference],
    ) -> OperationJournalEntry:
        references = list(artifact_references)
        self._verify_artifacts(references)
        with self.database.unit_of_work(immediate=True) as connection:
            existing = self._load_operation(connection, entry.operation_id)
            if existing is not None:
                stored = self._load_artifact_references(connection, entry.operation_id)
                if existing != entry or stored != references:
                    raise ExecutionIdentityConflict("operation identity conflict")
                return existing
            self._insert_operation(connection, entry, references)
            return entry

    def load_operation(self, operation_id: str) -> OperationJournalEntry | None:
        with self.database.connection() as connection:
            return self._load_operation(connection, operation_id)

    def mark_started(self, operation_id: str) -> OperationJournalEntry:
        return self._transition(
            operation_id,
            expected=OperationStatus.PREPARED,
            target=OperationStatus.STARTED,
            replay_statuses={
                OperationStatus.STARTED,
                OperationStatus.OBSERVED,
                OperationStatus.COMMITTED,
            },
        )

    def record_receipt(self, receipt: ToolExecutionReceipt) -> ToolExecutionReceipt:
        with self.database.unit_of_work(immediate=True) as connection:
            return self._record_receipt(connection, receipt)

    def load_receipt(
        self,
        task_id: str,
        tool_call_id: str,
    ) -> ToolExecutionReceipt | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM receipts WHERE task_id = ? AND tool_call_id = ?",
                (task_id, tool_call_id),
            ).fetchone()
        return None if row is None else ToolExecutionReceipt.model_validate_json(row[0])

    def load_artifact_references(
        self,
        operation_id: str,
    ) -> list[ArtifactReference]:
        with self.database.connection() as connection:
            return self._load_artifact_references(connection, operation_id)

    def observe_with_receipt(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt: ToolExecutionReceipt,
        artifact_references: list[ArtifactReference],
    ) -> OperationJournalEntry:
        references = list(artifact_references)
        self._verify_artifacts(references)
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._load_required(connection, operation_id)
            self._validate_receipt_for_operation(current, receipt)
            paths = [reference.path for reference in references]
            if current.status in {OperationStatus.OBSERVED, OperationStatus.COMMITTED}:
                stored_references = self._load_artifact_references(connection, operation_id)
                if (
                    current.post_state != post_state
                    or current.receipt_id != receipt.result_fingerprint
                    or stored_references != references
                ):
                    raise OperationConflictError("operation observation conflict")
                self._record_receipt(connection, receipt)
                return current
            if current.status is not OperationStatus.STARTED:
                raise OperationTransitionError(
                    f"operation must be started before observed: {operation_id}"
                )
            self._record_receipt(connection, receipt)
            observed = current.model_copy(
                update={
                    "status": OperationStatus.OBSERVED,
                    "post_state": post_state,
                    "receipt_id": receipt.result_fingerprint,
                    "artifact_references": paths,
                    "updated_at": _now(),
                }
            )
            self._update_operation(connection, current.status, observed, references)
            return observed

    def commit(self, operation_id: str) -> OperationJournalEntry:
        return self._transition(
            operation_id,
            expected=OperationStatus.OBSERVED,
            target=OperationStatus.COMMITTED,
            replay_statuses={OperationStatus.COMMITTED},
        )

    def mark_unknown(self, operation_id: str, reason: str) -> OperationJournalEntry:
        normalized = reason.strip()
        if not normalized:
            raise ValueError("unknown operation reason must not be empty")
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._load_required(connection, operation_id)
            if current.status is OperationStatus.UNKNOWN:
                if current.unknown_reason != normalized:
                    raise OperationConflictError("operation unknown reason conflict")
                return current
            if current.status not in {OperationStatus.STARTED, OperationStatus.OBSERVED}:
                raise OperationTransitionError(
                    f"operation cannot become unknown from {current.status.value}"
                )
            unknown = current.model_copy(
                update={
                    "status": OperationStatus.UNKNOWN,
                    "unknown_reason": normalized,
                    "updated_at": _now(),
                }
            )
            references = self._load_artifact_references(connection, operation_id)
            self._update_operation(connection, current.status, unknown, references)
            return unknown

    def list_incomplete(self, task_id: str) -> list[OperationJournalEntry]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM operations
                WHERE task_id = ? AND status != ?
                ORDER BY created_at, operation_id
                """,
                (task_id, OperationStatus.COMMITTED.value),
            ).fetchall()
        return [OperationJournalEntry.model_validate_json(row[0]) for row in rows]

    def list_operations(self, task_id: str) -> list[OperationJournalEntry]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM operations
                WHERE task_id = ? ORDER BY created_at, operation_id
                """,
                (task_id,),
            ).fetchall()
        return [OperationJournalEntry.model_validate_json(row[0]) for row in rows]

    def record_approval(self, approval: ExecutionApproval) -> ExecutionApproval:
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload FROM approvals WHERE task_id = ? AND approval_id = ?",
                (approval.task_id, approval.approval_id),
            ).fetchone()
            if row is not None:
                existing = ExecutionApproval.model_validate_json(row[0])
                if existing != approval:
                    raise ExecutionIdentityConflict("approval identity conflict")
                return existing
            connection.execute(
                """
                INSERT INTO approvals(task_id, approval_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    approval.task_id,
                    approval.approval_id,
                    approval.model_dump_json(),
                    approval.created_at,
                ),
            )
        return approval

    def list_approvals(self, task_id: str) -> list[ExecutionApproval]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM approvals
                WHERE task_id = ? ORDER BY created_at, approval_id
                """,
                (task_id,),
            ).fetchall()
        return [ExecutionApproval.model_validate_json(row[0]) for row in rows]

    def integrity_view(self, task_id: str) -> ExecutionIntegrity:
        with self.database.connection() as connection:
            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) FROM operations
                WHERE task_id = ? GROUP BY status ORDER BY status
                """,
                (task_id,),
            ).fetchall()
            receipt_count = connection.execute(
                "SELECT COUNT(*) FROM receipts WHERE task_id = ?", (task_id,)
            ).fetchone()[0]
            approval_count = connection.execute(
                "SELECT COUNT(*) FROM approvals WHERE task_id = ?", (task_id,)
            ).fetchone()[0]
            unknown = connection.execute(
                """
                SELECT operation_id FROM operations
                WHERE task_id = ? AND status = ? ORDER BY operation_id
                """,
                (task_id, OperationStatus.UNKNOWN.value),
            ).fetchall()
            incomplete = connection.execute(
                """
                SELECT operation_id FROM operations
                WHERE task_id = ? AND status != ? ORDER BY operation_id
                """,
                (task_id, OperationStatus.COMMITTED.value),
            ).fetchall()
        return ExecutionIntegrity(
            status_counts={str(row[0]): int(row[1]) for row in status_rows},
            receipt_count=int(receipt_count),
            approval_count=int(approval_count),
            incomplete_operation_ids=[str(row[0]) for row in incomplete],
            unknown_operation_ids=[str(row[0]) for row in unknown],
        )

    def _transition(
        self,
        operation_id: str,
        *,
        expected: OperationStatus,
        target: OperationStatus,
        replay_statuses: set[OperationStatus],
    ) -> OperationJournalEntry:
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._load_required(connection, operation_id)
            if current.status in replay_statuses:
                return current
            if current.status is not expected:
                raise OperationTransitionError(
                    f"operation must be {expected.value} before {target.value}: "
                    f"{operation_id}"
                )
            transitioned = current.model_copy(
                update={"status": target, "updated_at": _now()}
            )
            references = self._load_artifact_references(connection, operation_id)
            self._update_operation(connection, current.status, transitioned, references)
            return transitioned

    def _record_receipt(
        self,
        connection,
        receipt: ToolExecutionReceipt,
    ) -> ToolExecutionReceipt:
        row = connection.execute(
            "SELECT payload FROM receipts WHERE task_id = ? AND tool_call_id = ?",
            (receipt.task_id, receipt.tool_call_id),
        ).fetchone()
        if row is not None:
            existing = ToolExecutionReceipt.model_validate_json(row[0])
            if existing != receipt:
                raise ExecutionIdentityConflict("receipt identity conflict")
            return existing
        connection.execute(
            """
            INSERT INTO receipts(
                task_id, tool_call_id, call_hash, result_fingerprint, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.task_id,
                receipt.tool_call_id,
                receipt.call_hash,
                receipt.result_fingerprint,
                receipt.model_dump_json(),
                _now().isoformat(),
            ),
        )
        return receipt

    @staticmethod
    def _validate_receipt_for_operation(
        operation: OperationJournalEntry,
        receipt: ToolExecutionReceipt,
    ) -> None:
        if (
            receipt.task_id != operation.task_id
            or receipt.tool_call_id != operation.tool_call_id
            or receipt.call_hash != operation.call_hash
        ):
            raise ExecutionIdentityConflict("receipt does not match operation")

    def _verify_artifacts(self, references: list[ArtifactReference]) -> None:
        if self.artifact_verifier is None:
            return
        for reference in references:
            if not self.artifact_verifier(reference):
                raise ValueError(f"Artifact verification failed: {reference.path}")

    @staticmethod
    def _load_operation(connection, operation_id: str) -> OperationJournalEntry | None:
        row = connection.execute(
            "SELECT payload FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return None if row is None else OperationJournalEntry.model_validate_json(row[0])

    @classmethod
    def _load_required(cls, connection, operation_id: str) -> OperationJournalEntry:
        entry = cls._load_operation(connection, operation_id)
        if entry is None:
            raise OperationTransitionError(f"missing operation: {operation_id}")
        return entry

    @staticmethod
    def _load_artifact_references(
        connection,
        operation_id: str,
    ) -> list[ArtifactReference]:
        row = connection.execute(
            "SELECT artifact_references FROM operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            raise OperationTransitionError(f"missing operation: {operation_id}")
        return [ArtifactReference.model_validate(item) for item in json.loads(row[0])]

    @staticmethod
    def _insert_operation(
        connection,
        entry: OperationJournalEntry,
        references: list[ArtifactReference],
    ) -> None:
        connection.execute(
            """
            INSERT INTO operations(
                operation_id, task_id, tool_call_id, call_hash, status, payload,
                artifact_references, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.operation_id,
                entry.task_id,
                entry.tool_call_id,
                entry.call_hash,
                entry.status.value,
                entry.model_dump_json(),
                _references_json(references),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
            ),
        )

    @staticmethod
    def _update_operation(
        connection,
        expected_status: OperationStatus,
        entry: OperationJournalEntry,
        references: list[ArtifactReference],
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE operations
            SET status = ?, payload = ?, artifact_references = ?, updated_at = ?
            WHERE operation_id = ? AND status = ?
            """,
            (
                entry.status.value,
                entry.model_dump_json(),
                _references_json(references),
                entry.updated_at.isoformat(),
                entry.operation_id,
                expected_status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise OperationConflictError("operation transition conflict")

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    call_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    artifact_references TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS operations_task_status
                ON operations(task_id, status, created_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipts (
                    task_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    call_hash TEXT NOT NULL,
                    result_fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, tool_call_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    task_id TEXT NOT NULL,
                    approval_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, approval_id)
                )
                """
            )


def _references_json(references: list[ArtifactReference]) -> str:
    return json.dumps(
        [reference.model_dump(mode="json") for reference in references],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _now() -> datetime:
    return datetime.now(UTC)
