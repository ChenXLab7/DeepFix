from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage, message_to_dict

from deepfix.compaction.models import ArtifactReference
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.execution import (
    ExecutionApproval,
    ExecutionIdentityConflict,
    ExecutionRepository,
    create_execution_approval,
)
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.receipts import ToolExecutionReceipt, receipt_task_segment
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalEntry,
    OperationKind,
    OperationStateSnapshot,
    OperationStatus,
    OperationTransitionError,
)


def _receipt(call_id: str, *, content: str = "ok") -> ToolExecutionReceipt:
    message = ToolMessage(
        content=content,
        name="execute",
        tool_call_id=call_id,
    )
    return ToolExecutionReceipt(
        task_id="task-1",
        tool_call_id=call_id,
        tool_name="execute",
        call_hash=f"hash-{call_id}",
        tool_message_data=message_to_dict(message),
        result_fingerprint=result_fingerprint(message),
    )


def _operation(operation_id: str, call_id: str) -> NewOperationEntry:
    return NewOperationEntry(
        operation_id=operation_id,
        task_id="task-1",
        experiment_id="experiment-1",
        tool_call_id=call_id,
        operation_kind=OperationKind.COMMAND,
        call_hash=f"hash-{call_id}",
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(command_hash=f"command-{call_id}"),
    )


def test_parallel_receipts_are_atomic_and_idempotent(tmp_path: Path) -> None:
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    receipts = [_receipt("call-1"), _receipt("call-2")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        saved = list(pool.map(repository.record_receipt, receipts))

    assert {item.tool_call_id for item in saved} == {"call-1", "call-2"}
    assert repository.record_receipt(receipts[0]) == receipts[0]


def test_observation_commits_receipt_and_operation_state_together(
    tmp_path: Path,
) -> None:
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    prepared = repository.prepare(_operation("op-1", "call-1"))
    repository.mark_started(prepared.operation_id)

    observed = repository.observe_with_receipt(
        prepared.operation_id,
        post_state=OperationStateSnapshot(exit_code=0),
        receipt=_receipt("call-1"),
        artifact_references=[],
    )

    assert observed.status is OperationStatus.OBSERVED
    assert repository.load_receipt("task-1", "call-1") == _receipt("call-1")
    assert repository.integrity_view("task-1").status_counts == {"observed": 1}


def test_failed_observation_rolls_back_receipt_and_operation_state(
    tmp_path: Path,
) -> None:
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    repository.prepare(_operation("op-1", "call-1"))

    with pytest.raises(OperationTransitionError):
        repository.observe_with_receipt(
            "op-1",
            post_state=OperationStateSnapshot(exit_code=0),
            receipt=_receipt("call-1"),
            artifact_references=[],
        )

    assert repository.load_operation("op-1").status is OperationStatus.PREPARED
    assert repository.load_receipt("task-1", "call-1") is None


def test_observation_rejects_unverified_artifact_without_mutation(
    tmp_path: Path,
) -> None:
    repository = ExecutionRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda reference: reference.path == "verified.json",
    )
    repository.prepare(_operation("op-1", "call-1"))
    repository.mark_started("op-1")
    reference = ArtifactReference(
        path="missing.json",
        kind="operation_result",
        content_hash="a" * 64,
    )

    with pytest.raises(ValueError, match="Artifact verification failed"):
        repository.observe_with_receipt(
            "op-1",
            post_state=OperationStateSnapshot(exit_code=0),
            receipt=_receipt("call-1"),
            artifact_references=[reference],
        )

    assert repository.load_operation("op-1").status is OperationStatus.STARTED
    assert repository.load_receipt("task-1", "call-1") is None


def test_approval_is_immutable_and_included_in_integrity_view(tmp_path: Path) -> None:
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    approval = ExecutionApproval(
        approval_id="approval-1",
        task_id="task-1",
        operation="execute pytest",
        decision="approve",
        risk="L1",
        source_tool_call_id="call-1",
        created_at="2026-08-29T00:00:00+00:00",
    )

    assert repository.record_approval(approval) == approval
    assert repository.record_approval(approval) == approval
    assert (
        repository.record_approval(
            approval.model_copy(update={"created_at": "2026-08-30T00:00:00+00:00"})
        )
        == approval
    )
    with pytest.raises(ExecutionIdentityConflict):
        repository.record_approval(approval.model_copy(update={"decision": "deny"}))

    view = repository.integrity_view("task-1")
    assert view.approval_count == 1
    assert view.receipt_count == 0


def test_approval_builder_assigns_stable_identity_from_trusted_source() -> None:
    first = create_execution_approval(
        task_id="task-1",
        operation=" execute   pytest ",
        decision="approve",
        risk="L1",
        source_tool_call_id="call-1",
        created_at="2026-08-29T00:00:00+00:00",
    )
    replay = create_execution_approval(
        task_id="task-1",
        operation="execute pytest",
        decision="approve",
        risk="L1",
        source_tool_call_id="call-1",
        created_at="2026-08-29T00:00:00+00:00",
    )

    assert first == replay
    assert first.approval_id.startswith("approval_")


def test_operation_lifecycle_is_stored_in_execution_repository(tmp_path: Path) -> None:
    database_path = tmp_path / "deepfix.db"
    store = ExecutionRepository(database_path)
    store.prepare(_operation("op-1", "call-1"))
    store.mark_started("op-1")
    store.observe_with_receipt(
        "op-1",
        post_state=OperationStateSnapshot(exit_code=0),
        receipt=_receipt("call-1"),
        artifact_references=[],
    )

    assert ExecutionRepository(database_path).load_operation("op-1").status is (
        OperationStatus.OBSERVED
    )


def test_receipt_is_stored_in_execution_repository_without_legacy_file(
    tmp_path: Path,
) -> None:
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    root = tmp_path / "artifacts" / "investigation_receipts"
    expected = _receipt("call-1")

    repository.record_receipt(expected)

    assert repository.load_receipt("task-1", "call-1") == expected
    assert not list(root.rglob("*.json"))


def test_execution_migration_is_idempotent_and_preserves_legacy_sources(
    tmp_path: Path,
) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    artifact_root = tmp_path / "artifacts"
    now = datetime(2026, 8, 29, tzinfo=UTC)
    legacy_operation = OperationJournalEntry(
        **_operation("op-legacy", "call-legacy").model_dump(mode="python"),
        status=OperationStatus.OBSERVED,
        post_state=OperationStateSnapshot(exit_code=0),
        receipt_id=_receipt("call-legacy").result_fingerprint,
        created_at=now,
        updated_at=now,
    )
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE operation_journal (
                operation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                call_hash TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO operation_journal VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                legacy_operation.operation_id,
                legacy_operation.task_id,
                legacy_operation.call_hash,
                legacy_operation.status.value,
                legacy_operation.model_dump_json(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            CREATE TABLE legacy_task_projection (
                task_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO legacy_task_projection VALUES (?, ?)",
            (
                "task-1",
                json.dumps(
                    {
                        "approvals": [
                            {
                                "operation": "execute pytest",
                                "decision": "approve",
                                "risk": "L1",
                            }
                        ]
                    }
                ),
            ),
        )
    legacy_receipt_dir = artifact_root / "investigation_receipts" / receipt_task_segment("task-1")
    legacy_receipt_dir.mkdir(parents=True)
    (legacy_receipt_dir / "call-legacy.json").write_text(
        _receipt("call-legacy").model_dump_json(indent=2), encoding="utf-8"
    )
    before = hashlib.sha256(legacy_operation.model_dump_json().encode("utf-8")).hexdigest()

    migrator = DomainMigrator(database, artifact_root=artifact_root)
    first = migrator.migrate_execution("task-1")
    second = migrator.migrate_execution("task-1")

    assert first == second
    assert first.ready_to_switch
    assert first.source_count == first.target_count == 3
    execution = ExecutionRepository(database)
    assert execution.load_operation("op-legacy") == legacy_operation
    assert execution.load_receipt("task-1", "call-legacy") == _receipt("call-legacy")
    approvals = execution.list_approvals("task-1")
    assert len(approvals) == 1
    assert approvals[0].operation == "execute pytest"
    assert approvals[0].decision == "approve"
    with database.connection() as connection:
        payload = connection.execute(
            "SELECT payload FROM operation_journal WHERE operation_id = 'op-legacy'"
        ).fetchone()[0]
    assert hashlib.sha256(str(payload).encode("utf-8")).hexdigest() == before
