from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from langchain_core.messages import ToolMessage, message_to_dict
from pydantic import Field

from deepfix.compaction.models import StrictModel
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
)
from deepfix.persistence import open_sqlite_connection
from deepfix.workspace import (
    WorkspacePathPolicy,
    WorkspaceScopeError,
    compute_code_state_hash,
)


class OperationTransitionError(RuntimeError):
    pass


class OperationConflictError(RuntimeError):
    pass


class OperationStatus(StrEnum):
    PREPARED = "prepared"
    STARTED = "started"
    OBSERVED = "observed"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


class OperationKind(StrEnum):
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_DELETE = "file_delete"
    COMMAND = "command"


class OperationStateSnapshot(StrictModel):
    target_path: str | None = None
    target_exists: bool | None = None
    file_hash: str | None = None
    code_state_hash: str | None = None
    command_hash: str | None = None
    exit_code: int | None = None


class NewOperationEntry(StrictModel):
    operation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    operation_kind: OperationKind
    call_hash: str = Field(min_length=1)
    workspace_baseline_id: str = Field(min_length=1)
    pre_state: OperationStateSnapshot
    expected_post_state: OperationStateSnapshot | None = None


class OperationJournalEntry(NewOperationEntry):
    status: OperationStatus
    post_state: OperationStateSnapshot | None = None
    receipt_id: str | None = None
    artifact_references: list[str] = Field(default_factory=list)
    unknown_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class OperationJournalStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def prepare(self, entry: NewOperationEntry) -> OperationJournalEntry:
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = self._load(connection, entry.operation_id)
                if existing is not None:
                    if _prepared_identity(existing) != entry:
                        raise OperationConflictError("operation prepare conflict")
                    connection.rollback()
                    return existing
                now = _now()
                prepared = OperationJournalEntry(
                    **entry.model_dump(mode="python"),
                    status=OperationStatus.PREPARED,
                    created_at=now,
                    updated_at=now,
                )
                self._insert(connection, prepared)
                connection.commit()
                return prepared
            except WorkspaceScopeError:
                connection.rollback()
                raise

    def load(self, operation_id: str) -> OperationJournalEntry | None:
        with open_sqlite_connection(self.database_path) as connection:
            return self._load(connection, operation_id)

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

    def observe(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt_id: str,
        artifact_references: list[str],
    ) -> OperationJournalEntry:
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load_required(connection, operation_id)
                if current.status in {
                    OperationStatus.OBSERVED,
                    OperationStatus.COMMITTED,
                }:
                    if (
                        current.post_state != post_state
                        or current.receipt_id != receipt_id
                        or current.artifact_references != artifact_references
                    ):
                        raise OperationConflictError("operation observation conflict")
                    connection.rollback()
                    return current
                if current.status is not OperationStatus.STARTED:
                    raise OperationTransitionError(
                        f"operation must be started before observed: {operation_id}"
                    )
                observed = current.model_copy(
                    update={
                        "status": OperationStatus.OBSERVED,
                        "post_state": post_state,
                        "receipt_id": receipt_id,
                        "artifact_references": list(artifact_references),
                        "updated_at": _now(),
                    }
                )
                self._update(connection, current.status, observed)
                connection.commit()
                return observed
            except Exception:
                connection.rollback()
                raise

    def commit(self, operation_id: str) -> OperationJournalEntry:
        return self._transition(
            operation_id,
            expected=OperationStatus.OBSERVED,
            target=OperationStatus.COMMITTED,
            replay_statuses={OperationStatus.COMMITTED},
        )

    def mark_unknown(
        self,
        operation_id: str,
        reason: str,
    ) -> OperationJournalEntry:
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("unknown operation reason must not be empty")
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load_required(connection, operation_id)
                if current.status is OperationStatus.UNKNOWN:
                    if current.unknown_reason != normalized_reason:
                        raise OperationConflictError("operation unknown reason conflict")
                    connection.rollback()
                    return current
                if current.status not in {
                    OperationStatus.STARTED,
                    OperationStatus.OBSERVED,
                }:
                    raise OperationTransitionError(
                        f"operation cannot become unknown from {current.status.value}"
                    )
                unknown = current.model_copy(
                    update={
                        "status": OperationStatus.UNKNOWN,
                        "unknown_reason": normalized_reason,
                        "updated_at": _now(),
                    }
                )
                self._update(connection, current.status, unknown)
                connection.commit()
                return unknown
            except Exception:
                connection.rollback()
                raise

    def list_incomplete(self, task_id: str) -> list[OperationJournalEntry]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM operation_journal
                WHERE task_id = ? AND status != ?
                ORDER BY created_at, operation_id
                """,
                (task_id, OperationStatus.COMMITTED.value),
            ).fetchall()
        return [OperationJournalEntry.model_validate_json(row[0]) for row in rows]

    def _transition(
        self,
        operation_id: str,
        *,
        expected: OperationStatus,
        target: OperationStatus,
        replay_statuses: set[OperationStatus],
    ) -> OperationJournalEntry:
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load_required(connection, operation_id)
                if current.status in replay_statuses:
                    connection.rollback()
                    return current
                if current.status is not expected:
                    raise OperationTransitionError(
                        f"operation must be {expected.value} before {target.value}: "
                        f"{operation_id}"
                    )
                transitioned = current.model_copy(
                    update={"status": target, "updated_at": _now()}
                )
                self._update(connection, current.status, transitioned)
                connection.commit()
                return transitioned
            except Exception:
                connection.rollback()
                raise

    def _initialize(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_journal (
                    operation_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    call_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS operation_journal_task_status
                ON operation_journal(task_id, status, created_at)
                """
            )
            connection.commit()


    @staticmethod
    def _load(
        connection: sqlite3.Connection,
        operation_id: str,
    ) -> OperationJournalEntry | None:
        row = connection.execute(
            "SELECT payload FROM operation_journal WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        return OperationJournalEntry.model_validate_json(row[0])

    @classmethod
    def _load_required(
        cls,
        connection: sqlite3.Connection,
        operation_id: str,
    ) -> OperationJournalEntry:
        entry = cls._load(connection, operation_id)
        if entry is None:
            raise OperationTransitionError(f"missing operation: {operation_id}")
        return entry

    @staticmethod
    def _insert(
        connection: sqlite3.Connection,
        entry: OperationJournalEntry,
    ) -> None:
        connection.execute(
            """
            INSERT INTO operation_journal(
                operation_id, task_id, call_hash, status, payload,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.operation_id,
                entry.task_id,
                entry.call_hash,
                entry.status.value,
                entry.model_dump_json(),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
            ),
        )

    @staticmethod
    def _update(
        connection: sqlite3.Connection,
        expected_status: OperationStatus,
        entry: OperationJournalEntry,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE operation_journal
            SET status = ?, payload = ?, updated_at = ?
            WHERE operation_id = ? AND status = ?
            """,
            (
                entry.status.value,
                entry.model_dump_json(),
                entry.updated_at.isoformat(),
                entry.operation_id,
                expected_status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise OperationConflictError("operation transition conflict")


class ReconciliationResult(StrictModel):
    reconstructed_operation_ids: list[str] = Field(default_factory=list)
    replayable_operation_ids: list[str] = Field(default_factory=list)
    conflict_operation_ids: list[str] = Field(default_factory=list)
    unknown_operation_ids: list[str] = Field(default_factory=list)
    prepared_operation_ids: list[str] = Field(default_factory=list)

    @property
    def blocks_agent_invocation(self) -> bool:
        return bool(self.conflict_operation_ids or self.unknown_operation_ids)


class OperationReconciler:
    def __init__(
        self,
        journal: OperationJournalStore,
        receipts: ToolExecutionReceiptStore,
        *,
        terminate_process_group=None,
    ) -> None:
        self.journal = journal
        self.receipts = receipts
        self.terminate_process_group = terminate_process_group or (lambda _entry: None)

    def reconcile_task(
        self,
        task_id: str,
        workspace: str | Path,
    ) -> ReconciliationResult:
        root = Path(workspace).expanduser().resolve(strict=True)
        paths = WorkspacePathPolicy(root)
        reconstructed: list[str] = []
        replayable: list[str] = []
        conflicts: list[str] = []
        unknown: list[str] = []
        prepared: list[str] = []
        for entry in self.journal.list_incomplete(task_id):
            if entry.status is OperationStatus.PREPARED:
                prepared.append(entry.operation_id)
                continue
            if entry.status is OperationStatus.UNKNOWN:
                unknown.append(entry.operation_id)
                continue
            if entry.status is OperationStatus.OBSERVED:
                if self.receipts.load(task_id, entry.tool_call_id) is not None:
                    replayable.append(entry.operation_id)
                else:
                    self.journal.mark_unknown(
                        entry.operation_id, "observed operation receipt is missing"
                    )
                    conflicts.append(entry.operation_id)
                continue
            if entry.operation_kind is OperationKind.COMMAND:
                if self._reconcile_command(entry, root):
                    reconstructed.append(entry.operation_id)
                    replayable.append(entry.operation_id)
                else:
                    unknown.append(entry.operation_id)
                continue
            expected = entry.expected_post_state
            if expected is None or not expected.target_path:
                self.journal.mark_unknown(
                    entry.operation_id, "file operation lacks expected post state"
                )
                unknown.append(entry.operation_id)
                continue
            try:
                target = paths.resolve_allowed(expected.target_path)
            except WorkspaceScopeError:
                self.journal.mark_unknown(
                    entry.operation_id, "file operation target escapes workspace"
                )
                conflicts.append(entry.operation_id)
                continue
            current = _file_snapshot(root, target)
            if _matches_expected(current, expected):
                receipt = _reconstructed_file_receipt(entry)
                self.receipts.save(receipt)
                self.journal.observe(
                    entry.operation_id,
                    post_state=current,
                    receipt_id=receipt.result_fingerprint,
                    artifact_references=[],
                )
                reconstructed.append(entry.operation_id)
                replayable.append(entry.operation_id)
            elif _matches_expected(current, entry.pre_state):
                self.journal.mark_unknown(
                    entry.operation_id,
                    "operation started but workspace still matches pre-state",
                )
                unknown.append(entry.operation_id)
            else:
                self.journal.mark_unknown(
                    entry.operation_id,
                    "workspace state matches neither operation pre-state nor post-state",
                )
                conflicts.append(entry.operation_id)
        return ReconciliationResult(
            reconstructed_operation_ids=reconstructed,
            replayable_operation_ids=replayable,
            conflict_operation_ids=conflicts,
            unknown_operation_ids=unknown,
            prepared_operation_ids=prepared,
        )

    def _reconcile_command(
        self,
        entry: OperationJournalEntry,
        workspace: Path,
    ) -> bool:
        loaded = self.receipts.load_result_artifact(entry.task_id, entry.tool_call_id)
        if loaded is None or loaded[1].exit_code is None:
            self.terminate_process_group(entry)
            self.journal.mark_unknown(
                entry.operation_id,
                "command exit status cannot be verified",
            )
            return False
        reference, artifact = loaded
        message = ToolMessage(
            id=f"recovered-{entry.operation_id}",
            content=artifact.output,
            name="execute",
            tool_call_id=entry.tool_call_id,
            artifact={
                "exit_code": artifact.exit_code,
                "result_type": "reconstructed_command_result",
            },
        )
        receipt = ToolExecutionReceipt(
            task_id=entry.task_id,
            tool_call_id=entry.tool_call_id,
            tool_name="execute",
            call_hash=entry.call_hash,
            tool_message_data=message_to_dict(message),
            result_fingerprint=result_fingerprint(message),
        )
        self.receipts.save(receipt)
        self.journal.observe(
            entry.operation_id,
            post_state=OperationStateSnapshot(
                code_state_hash=compute_code_state_hash(workspace),
                command_hash=entry.pre_state.command_hash,
                exit_code=artifact.exit_code,
            ),
            receipt_id=receipt.result_fingerprint,
            artifact_references=[reference],
        )
        return True


def _file_snapshot(root: Path, target: Path) -> OperationStateSnapshot:
    exists = target.is_file()
    return OperationStateSnapshot(
        target_path=target.relative_to(root).as_posix(),
        target_exists=exists,
        file_hash=_file_hash(target) if exists else None,
        code_state_hash=compute_code_state_hash(root),
    )


def _matches_expected(
    current: OperationStateSnapshot,
    expected: OperationStateSnapshot,
) -> bool:
    return all(
        getattr(expected, field) is None
        or getattr(current, field) == getattr(expected, field)
        for field in ("target_path", "target_exists", "file_hash")
    )


def _reconstructed_file_receipt(
    entry: OperationJournalEntry,
) -> ToolExecutionReceipt:
    names = {
        OperationKind.FILE_WRITE: ("write_file", "write"),
        OperationKind.FILE_EDIT: ("edit_file", "edit"),
        OperationKind.FILE_DELETE: ("delete", "delete"),
    }
    tool_name, operation = names[entry.operation_kind]
    path = f"/{entry.expected_post_state.target_path}"
    message = ToolMessage(
        id=f"recovered-{entry.operation_id}",
        content=f"Recovered successful {operation} for {path}",
        name=tool_name,
        tool_call_id=entry.tool_call_id,
        artifact={
            "operation": operation,
            "status": "succeeded",
            "path": path,
            "result_type": "reconstructed_file_result",
        },
    )
    return ToolExecutionReceipt(
        task_id=entry.task_id,
        tool_call_id=entry.tool_call_id,
        tool_name=tool_name,
        call_hash=entry.call_hash,
        tool_message_data=message_to_dict(message),
        result_fingerprint=result_fingerprint(message),
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_identity(entry: OperationJournalEntry) -> NewOperationEntry:
    return NewOperationEntry.model_validate(
        entry.model_dump(
            include={
                "operation_id",
                "task_id",
                "experiment_id",
                "tool_call_id",
                "operation_kind",
                "call_hash",
                "workspace_baseline_id",
                "pre_state",
                "expected_post_state",
            }
        )
    )


def _now() -> datetime:
    return datetime.now(UTC)
