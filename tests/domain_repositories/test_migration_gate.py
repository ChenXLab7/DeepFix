from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import ToolMessage, message_to_dict

from deepfix.compaction.models import (
    ArtifactReference,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    SnapshotCoverage,
    SystemTestEvidence,
)
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.migration import (
    DomainMigrationInProgress,
    DomainMigrator,
)
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.models import InvestigationHypothesis, InvestigationState
from deepfix.investigation.receipts import ToolExecutionReceipt, receipt_task_segment
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalEntry,
    OperationKind,
    OperationStateSnapshot,
    OperationStatus,
)
from deepfix.research.models import ExternalEvidence, ResearchQuery, SearchCandidate

TASK_ID = "task-1"


@pytest.fixture
def migrated(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    artifacts = tmp_path / "artifacts"
    project = tmp_path / "project"
    project.mkdir()
    test = SystemTestEvidence(
        evidence_id="test-evidence-1",
        command="python -m pytest tests/test_parser.py -q",
        exit_code=0,
        summary="required oracle passed",
        tool_call_id="call-test",
        source_message_id="message-test",
    )
    task_payload = {
        "task_id": TASK_ID,
        "project_root": str(project),
        "user_problem": "fix parser failure",
        "approval_mode": "manual",
        "project_python": "python",
        "status": "investigating",
        "conversation": [
            {"id": "message-user-1", "role": "user", "content": "fix parser failure"}
        ],
        "approvals": [
            {"operation": "execute pytest", "decision": "approve", "risk": "L1"}
        ],
    }
    query = ResearchQuery(
        query_id="query-1",
        task_id=TASK_ID,
        sanitized_query="pytest parser",
        providers=["github"],
        provider_errors=[],
        created_at="2026-08-30T00:01:00+00:00",
    )
    candidate = SearchCandidate(
        candidate_id="candidate-1",
        task_id=TASK_ID,
        source_type="official_docs",
        evidence_level="E1",
        title="Parser documentation",
        url="https://example.test/parser",
        query=query.sanitized_query,
        repository=None,
        created_at="2026-08-30T00:01:01+00:00",
    )
    research_path = "/.deepfix-artifacts/research/task-1/evidence.md"
    research_file = artifacts / "research" / TASK_ID / "evidence.md"
    research_file.parent.mkdir(parents=True)
    research_file.write_text("verified research", encoding="utf-8")
    external = ExternalEvidence(
        evidence_id="external-evidence-1",
        task_id=TASK_ID,
        candidate_id=candidate.candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title=candidate.title,
        url=candidate.url,
        query=query.sanitized_query,
        relevant_excerpt="Parser behavior is documented.",
        retrieved_at="2026-08-30T00:01:02+00:00",
        dependency_name=None,
        documented_version=None,
        project_version=None,
        local_verification="unverified",
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
        artifact_path=research_path,
    )
    hypothesis = InvestigationHypothesis(
        hypothesis_id="hypothesis-1",
        statement="parser branch is the root cause",
        state="supported",
        evidence_ids=[test.evidence_id],
        checked_locations=[],
        reason="required oracle supports diagnosis",
    )
    operation = OperationJournalEntry(
        **NewOperationEntry(
            operation_id="operation-1",
            task_id=TASK_ID,
            experiment_id="experiment-1",
            tool_call_id="call-execute",
            operation_kind=OperationKind.COMMAND,
            call_hash="hash-call-execute",
            workspace_baseline_id="baseline-1",
            pre_state=OperationStateSnapshot(command_hash="command-hash"),
        ).model_dump(mode="python"),
        status=OperationStatus.OBSERVED,
        post_state=OperationStateSnapshot(exit_code=0),
        receipt_id="receipt-1",
        created_at=datetime(2026, 8, 30, 0, 4, tzinfo=UTC),
        updated_at=datetime(2026, 8, 30, 0, 4, tzinfo=UTC),
    )
    receipt_message = ToolMessage(
        id="tool-message-call-execute",
        content="pytest passed",
        name="execute",
        tool_call_id="call-execute",
    )
    receipt = ToolExecutionReceipt(
        task_id=TASK_ID,
        tool_call_id="call-execute",
        tool_name="execute",
        call_hash="hash-call-execute",
        tool_message_data=message_to_dict(receipt_message),
        result_fingerprint=result_fingerprint(receipt_message),
    )
    history_path = f"/.deepfix-artifacts/conversation_history/{TASK_ID}/history.md"
    history_file = artifacts / "conversation_history" / TASK_ID / "history.md"
    history_file.parent.mkdir(parents=True)
    history_file.write_text("legacy graph history", encoding="utf-8")
    history_artifact = ArtifactReference(
        path=history_path,
        kind="conversation_history",
        content_hash=hashlib.sha256(history_file.read_bytes()).hexdigest(),
        work_unit_ids=["unit-1"],
    )
    snapshot = CompactionSnapshot(
        task_id=TASK_ID,
        version=1,
        lifecycle="prepared",
        created_at="2026-08-30T00:05:00+00:00",
        source_work_unit_ids=["unit-1"],
        coverage=SnapshotCoverage(
            last_user_message_id="message-user-1",
            covered_message_ids=["message-user-1"],
            covered_work_unit_ids=["unit-1"],
        ),
        task_goal="fix parser failure",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(tests=[test]),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[history_artifact],
        content_hash="c" * 64,
    )
    event = DeepFixCompactionEvent(
        event_id="event-1",
        task_id=TASK_ID,
        active_snapshot_version=1,
        snapshot_message_id="snapshot-message-1",
        retained_message_ids=["message-user-1"],
        conversation_artifact=history_artifact,
        input_hash="b" * 64,
    )

    with database.unit_of_work() as connection:
        connection.executescript(
            """
            CREATE TABLE tasks(task_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT);
            CREATE TABLE deterministic_evidence(
                task_id TEXT, evidence_id TEXT, kind TEXT, payload TEXT, created_at TEXT,
                PRIMARY KEY(task_id, evidence_id)
            );
            CREATE TABLE research_queries(
                task_id TEXT, query_id TEXT, payload TEXT, created_at TEXT,
                PRIMARY KEY(task_id, query_id)
            );
            CREATE TABLE search_candidates(
                task_id TEXT, candidate_id TEXT, payload TEXT, created_at TEXT,
                PRIMARY KEY(task_id, candidate_id)
            );
            CREATE TABLE external_evidence(
                task_id TEXT, evidence_id TEXT, candidate_id TEXT, payload TEXT, updated_at TEXT,
                PRIMARY KEY(task_id, evidence_id)
            );
            CREATE TABLE investigation_state(
                task_id TEXT PRIMARY KEY, version INTEGER, payload TEXT, updated_at TEXT
            );
            CREATE TABLE operation_journal(
                operation_id TEXT PRIMARY KEY, task_id TEXT, call_hash TEXT, status TEXT,
                payload TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE compaction_snapshots(
                task_id TEXT, version INTEGER, lifecycle TEXT, input_hash TEXT, payload TEXT,
                created_at TEXT, activated_at TEXT, abandoned_at TEXT, abandon_reason TEXT,
                PRIMARY KEY(task_id, version), UNIQUE(task_id, input_hash)
            );
            CREATE TABLE context_migrations(
                task_id TEXT PRIMARY KEY, version INTEGER, event_payload TEXT, migrated_at TEXT
            );
            CREATE TABLE checkpoints(
                thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, payload BLOB,
                PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (TASK_ID, json.dumps(task_payload), "2026-08-30T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO deterministic_evidence VALUES (?, ?, ?, ?, ?)",
            (TASK_ID, test.evidence_id, "test", test.model_dump_json(), snapshot.created_at),
        )
        connection.execute(
            "INSERT INTO research_queries VALUES (?, ?, ?, ?)",
            (TASK_ID, query.query_id, query.model_dump_json(), query.created_at),
        )
        connection.execute(
            "INSERT INTO search_candidates VALUES (?, ?, ?, ?)",
            (TASK_ID, candidate.candidate_id, candidate.model_dump_json(), candidate.created_at),
        )
        connection.execute(
            "INSERT INTO external_evidence VALUES (?, ?, ?, ?, ?)",
            (
                TASK_ID,
                external.evidence_id,
                external.candidate_id,
                external.model_dump_json(),
                external.retrieved_at,
            ),
        )
        connection.execute(
            "INSERT INTO investigation_state VALUES (?, ?, ?, ?)",
            (
                TASK_ID,
                1,
                InvestigationState(
                    task_id=TASK_ID,
                    hypotheses=[hypothesis],
                    supported_hypothesis_ids=[hypothesis.hypothesis_id],
                ).model_dump_json(),
                snapshot.created_at,
            ),
        )
        connection.execute(
            "INSERT INTO operation_journal VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                operation.operation_id,
                TASK_ID,
                operation.call_hash,
                operation.status.value,
                operation.model_dump_json(),
                operation.created_at.isoformat(),
                operation.updated_at.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO compaction_snapshots VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (TASK_ID, 1, "prepared", event.input_hash, snapshot.model_dump_json(), snapshot.created_at),
        )
        connection.execute(
            "INSERT INTO context_migrations VALUES (?, ?, ?, ?)",
            (TASK_ID, 1, event.model_dump_json(), snapshot.created_at),
        )
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?)",
            (TASK_ID, "", "checkpoint-1", b"opaque-langgraph-checkpoint"),
        )

    receipt_dir = artifacts / "investigation_receipts" / receipt_task_segment(TASK_ID)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "call-execute.json").write_text(
        receipt.model_dump_json(),
        encoding="utf-8",
    )
    before = _legacy_hash(database, receipt_dir)
    migrator = DomainMigrator(database, artifact_root=artifacts)
    reports = [
        migrator.migrate_deterministic_evidence(TASK_ID),
        migrator.migrate_research(TASK_ID),
        migrator.migrate_investigation(TASK_ID),
        migrator.migrate_execution(TASK_ID),
        migrator.migrate_history(TASK_ID),
    ]
    return database, artifacts, DomainRepositories.create(database), reports, before, receipt_dir


def test_all_domain_migrations_are_lossless_and_leave_legacy_sources_unchanged(migrated):
    database, _artifacts, _repositories, reports, before, receipt_dir = migrated

    assert all(report.ready_to_switch for report in reports)
    assert all(report.source_count == report.target_count for report in reports)
    assert all(not report.identity_mismatches for report in reports)
    assert all(not report.hash_mismatches for report in reports)
    assert all(not report.missing_references for report in reports)
    assert _legacy_hash(database, receipt_dir) == before


def test_graph_checkpoint_and_current_authorities_restore_after_migration(migrated):
    database, _artifacts, repositories, _reports, _before, _receipt_dir = migrated

    assert repositories.evidence.get(TASK_ID, "test-evidence-1").evidence_id == (
        "test-evidence-1"
    )
    assert repositories.investigation.get_hypothesis(
        TASK_ID, "hypothesis-1"
    ).state == "supported"
    assert repositories.execution.load_receipt(TASK_ID, "call-execute") is not None
    assert repositories.history.migrated_event(TASK_ID) is not None
    assert repositories.tasks.get_definition(TASK_ID).original_problem == "fix parser failure"
    with database.connection() as connection:
        checkpoint = connection.execute(
            "SELECT payload FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
            (TASK_ID, "checkpoint-1"),
        ).fetchone()
    assert checkpoint == (b"opaque-langgraph-checkpoint",)


def test_active_migration_fence_rejects_a_second_owner(migrated):
    database, artifacts, _repositories, _reports, _before, _receipt_dir = migrated
    migrator = DomainMigrator(database, artifact_root=artifacts)
    with (
        migrator._migration_fence("new-domain", TASK_ID),
        pytest.raises(DomainMigrationInProgress),
        migrator._migration_fence("new-domain", TASK_ID),
    ):
        pass


def _legacy_hash(database: SQLiteDatabase, receipt_dir) -> str:
    tables = (
        "tasks",
        "deterministic_evidence",
        "research_queries",
        "search_candidates",
        "external_evidence",
        "investigation_state",
        "operation_journal",
        "compaction_snapshots",
        "context_migrations",
        "checkpoints",
    )
    with database.connection() as connection:
        payload = {
            table: [
                tuple(
                    item.hex() if isinstance(item, bytes) else item
                    for item in row
                )
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            ]
            for table in tables
        }
    payload["receipts"] = [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(receipt_dir.glob("*.json"))
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
