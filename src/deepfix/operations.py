from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from langchain_core.messages import ToolMessage, message_to_dict
from pydantic import Field

from deepfix.compaction.models import ArtifactReference, StrictModel
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
)
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
        from deepfix.domain_repositories.execution import ExecutionRepository

        self.repository = ExecutionRepository(self.database_path)

    def prepare(self, entry: NewOperationEntry) -> OperationJournalEntry:
        return self.repository.prepare(entry)

    def load(self, operation_id: str) -> OperationJournalEntry | None:
        return self.repository.load_operation(operation_id)

    def mark_started(self, operation_id: str) -> OperationJournalEntry:
        return self.repository.mark_started(operation_id)

    def observe_with_receipt(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt: ToolExecutionReceipt,
        artifact_references: list[ArtifactReference],
    ) -> OperationJournalEntry:
        return self.repository.observe_with_receipt(
            operation_id,
            post_state=post_state,
            receipt=receipt,
            artifact_references=artifact_references,
        )

    def load_receipt(
        self,
        task_id: str,
        tool_call_id: str,
    ) -> ToolExecutionReceipt | None:
        return self.repository.load_receipt(task_id, tool_call_id)

    def artifact_references(self, operation_id: str) -> list[ArtifactReference]:
        return self.repository.load_artifact_references(operation_id)

    def observe(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt_id: str,
        artifact_references: list[str],
    ) -> OperationJournalEntry:
        current = self.repository.load_operation(operation_id)
        if current is None:
            raise OperationTransitionError(f"missing operation: {operation_id}")
        tool_name = {
            OperationKind.FILE_WRITE: "write_file",
            OperationKind.FILE_EDIT: "edit_file",
            OperationKind.FILE_DELETE: "delete",
            OperationKind.COMMAND: "execute",
        }[current.operation_kind]
        message = ToolMessage(
            content="legacy operation observation",
            name=tool_name,
            tool_call_id=current.tool_call_id,
        )
        receipt = ToolExecutionReceipt(
            task_id=current.task_id,
            tool_call_id=current.tool_call_id,
            tool_name=tool_name,
            call_hash=current.call_hash,
            tool_message_data=message_to_dict(message),
            result_fingerprint=receipt_id,
        )
        references = [
            ArtifactReference(
                path=path,
                kind="operation_result",
                content_hash=hashlib.sha256(path.encode("utf-8")).hexdigest(),
            )
            for path in artifact_references
        ]
        return self.observe_with_receipt(
            operation_id,
            post_state=post_state,
            receipt=receipt,
            artifact_references=references,
        )

    def commit(self, operation_id: str) -> OperationJournalEntry:
        return self.repository.commit(operation_id)

    def mark_unknown(
        self,
        operation_id: str,
        reason: str,
    ) -> OperationJournalEntry:
        return self.repository.mark_unknown(operation_id, reason)

    def list_incomplete(self, task_id: str) -> list[OperationJournalEntry]:
        return self.repository.list_incomplete(task_id)


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
        if receipts.repository is None:
            receipts.repository = journal.repository
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
                if self.journal.load_receipt(task_id, entry.tool_call_id) is not None:
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
                self.journal.observe_with_receipt(
                    entry.operation_id,
                    post_state=current,
                    receipt=receipt,
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
        self.journal.observe_with_receipt(
            entry.operation_id,
            post_state=OperationStateSnapshot(
                code_state_hash=compute_code_state_hash(workspace),
                command_hash=entry.pre_state.command_hash,
                exit_code=artifact.exit_code,
            ),
            receipt=receipt,
            artifact_references=[self.receipts.artifact_reference(reference)],
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
