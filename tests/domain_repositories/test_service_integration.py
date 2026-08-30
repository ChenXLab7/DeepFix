from __future__ import annotations

import json

from langchain_core.messages import ToolMessage, message_to_dict

from deepfix.compaction.models import (
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.models import ApprovalRecord, Evidence, TaskState
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalStore,
    OperationKind,
    OperationStateSnapshot,
)
from deepfix.research.store import ResearchEvidenceStore


def _mark_switched(database: SQLiteDatabase, task_id: str, *domains: str) -> None:
    with database.unit_of_work() as connection:
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
        connection.executemany(
            """
            INSERT INTO domain_migrations(domain, task_id, report_json, switched_at)
            VALUES (?, ?, '{}', '2026-08-30T00:00:00+00:00')
            """,
            [(domain, task_id) for domain in domains],
        )


def _legacy_row_count(connection, table: str) -> int:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        return 0
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_production_repositories_share_one_database_and_facade_instances(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")

    repositories = DomainRepositories.create(database)
    compaction = CompactionStore(database, repositories=repositories)
    research = ResearchEvidenceStore(database, repositories=repositories)
    investigation = InvestigationStore(database, repositories=repositories)
    operations = OperationJournalStore(database.path, repositories=repositories)

    paths = {
        repositories.tasks.database.path,
        repositories.evidence.database.path,
        repositories.investigation.database.path,
        repositories.execution.database.path,
        repositories.history.database.path,
    }
    assert paths == {database.path}
    assert compaction.evidence_repository is repositories.evidence
    assert compaction.investigation_repository is repositories.investigation
    assert compaction.history_repository is repositories.history
    assert research.evidence_repository is repositories.evidence
    assert investigation.repository is repositories.investigation
    assert operations.repository is repositories.execution


def test_legacy_projection_omits_only_fields_with_switched_domain_authority(
    tmp_path,
):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    repositories = DomainRepositories.create(database)
    task = TaskState.create(tmp_path, "fix parser", ApprovalMode.MANUAL)
    task.hypotheses = ["old hypothesis"]
    task.changed_files = ["parser.py"]
    task.successful_changed_files = ["parser.py"]
    task.approvals = [ApprovalRecord("edit_file", "approve", "L1")]
    task.external_evidence_ids = ["external-1"]
    task.research_query_count = 2
    task.evidence = [Evidence("parser.py:10", "legacy observation")]
    task.final_summary = "compatibility conclusion"
    _mark_switched(
        database,
        task.task_id,
        "evidence",
        "research",
        "investigation",
        "execution",
        "history",
    )

    repositories.tasks.save_legacy_projection(task)

    with database.connection() as connection:
        raw = connection.execute(
            "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()[0]
    payload = json.loads(str(raw))
    assert "hypotheses" not in payload
    assert "changed_files" not in payload
    assert "successful_changed_files" not in payload
    assert "approvals" not in payload
    assert "external_evidence_ids" not in payload
    assert "research_query_count" not in payload
    assert payload["evidence"] == [{"source": "parser.py:10", "observation": "legacy observation"}]
    assert payload["final_summary"] == "compatibility conclusion"
    assert "conversation" in payload


def test_evidence_marker_accepts_existing_deterministic_evidence_migration_name(
    tmp_path,
):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    repositories = DomainRepositories.create(database)
    task = TaskState.create(tmp_path, "fix parser", ApprovalMode.MANUAL)
    task.changed_files = ["parser.py"]
    _mark_switched(database, task.task_id, "deterministic_evidence")

    repositories.tasks.save_legacy_projection(task)

    with database.connection() as connection:
        raw = connection.execute(
            "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()[0]
    assert "changed_files" not in json.loads(str(raw))


def test_production_facades_do_not_write_retired_legacy_sources(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    repositories = DomainRepositories.create(database)
    compaction = CompactionStore(database, repositories=repositories)
    research = ResearchEvidenceStore(database, repositories=repositories)
    operations = OperationJournalStore(database.path, repositories=repositories)
    receipt_root = tmp_path / "artifacts" / "investigation_receipts"
    receipts = ToolExecutionReceiptStore(
        receipt_root,
        repository=repositories.execution,
    )
    task_id = "task-1"
    artifact = ArtifactReference(
        path="/.deepfix-artifacts/conversation_history/task-1.md",
        kind="conversation_history",
        content_hash="a" * 64,
        work_unit_ids=[],
    )
    snapshot = CompactionSnapshot(
        task_id=task_id,
        version=1,
        lifecycle="prepared",
        created_at="2026-08-30T00:00:00+00:00",
        source_work_unit_ids=[],
        task_goal="fix parser",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[artifact],
        content_hash="b" * 64,
    )

    compaction.save_evidence(
        task_id,
        SystemTestEvidence(
            evidence_id="test-1",
            command="python -m pytest -q",
            exit_code=0,
            summary="1 passed",
            tool_call_id="call-test",
            source_message_id="message-test",
        ),
    )
    research.save_query(task_id, "pytest timeout", ["github"], [])
    compaction.save_prepared_snapshot(snapshot, "input-1")
    compaction.record_failure(
        CompactionFailureRecord(
            attempt_id="attempt-1",
            task_id=task_id,
            entrypoint="automatic",
            budget_zone="normal_compaction",
            stage="artifact_write",
            error_code="artifact_write_failed",
            input_hash="c" * 64,
            original_messages_preserved=True,
            recorded_at="2026-08-30T00:00:01+00:00",
        )
    )
    compaction.record_migration(
        task_id,
        1,
        DeepFixCompactionEvent(
            event_id="event-1",
            task_id=task_id,
            active_snapshot_version=1,
            snapshot_message_id="snapshot-message-1",
            retained_message_ids=[],
            conversation_artifact=artifact,
            input_hash="input-1",
        ),
    )
    operation = operations.prepare(
        NewOperationEntry(
            operation_id="operation-1",
            task_id=task_id,
            experiment_id="experiment-1",
            tool_call_id="call-execute",
            operation_kind=OperationKind.COMMAND,
            call_hash="command-hash",
            workspace_baseline_id="baseline-1",
            pre_state=OperationStateSnapshot(command_hash="command-hash"),
        )
    )
    message = ToolMessage(
        content="ok",
        name="execute",
        tool_call_id=operation.tool_call_id,
    )
    receipts.save(
        ToolExecutionReceipt(
            task_id=task_id,
            tool_call_id=operation.tool_call_id,
            tool_name="execute",
            call_hash=operation.call_hash,
            tool_message_data=message_to_dict(message),
            result_fingerprint=result_fingerprint(message),
        )
    )

    with database.connection() as connection:
        counts = {
            table: _legacy_row_count(connection, table)
            for table in (
                "deterministic_evidence",
                "research_queries",
                "search_candidates",
                "external_evidence",
                "operation_journal",
                "compaction_snapshots",
                "compaction_failures",
                "context_migrations",
            )
        }
    assert counts == {table: 0 for table in counts}
    assert not list(receipt_root.rglob("*.json"))
