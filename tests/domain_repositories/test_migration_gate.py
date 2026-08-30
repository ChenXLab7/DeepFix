from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest
from langchain_core.messages import HumanMessage, ToolMessage, message_to_dict
from pydantic import SecretStr

from deepfix.approval import ApprovalPolicy
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import (
    ArtifactReference,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    FileChangeEvidence,
    HypothesisRecord,
    ProvenancedText,
    ProvenanceRef,
    SnapshotCoverage,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore
from deepfix.config import AppConfig, ApprovalMode, ModelRoleConfig
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.migration import (
    DomainMigrationInProgress,
    DomainMigrationReport,
    DomainMigrator,
)
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.models import InvestigationHypothesis, InvestigationState
from deepfix.investigation.receipts import (
    LegacyReceiptWritesDisabled,
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
    legacy_receipt_task_guard,
)
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import WorkingMemoryStore
from deepfix.models import ApprovalRecord, TaskState, TaskStatus
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalEntry,
    OperationJournalStore,
    OperationKind,
    OperationReconciler,
    OperationStateSnapshot,
    OperationStatus,
)
from deepfix.protected_context import ProtectedContextBuilder, render_protected_context
from deepfix.reporting import render_report
from deepfix.research.models import ExternalEvidence, ResearchQuery, SearchCandidate
from deepfix.research.store import ResearchEvidenceStore
from deepfix.service import BugfixService
from deepfix.task_domain.migration import MIGRATED_LEGACY_FIELDS
from deepfix.task_domain.repository import TaskRepository

TASK_ID = "task-1"
CURRENT_TEST_ID = "test-evidence-1"
CURRENT_HYPOTHESIS_ID = "hypothesis-1"
RESEARCH_EVIDENCE_ID = "external-evidence-1"


def _receipt(call_id: str, tool_name: str = "execute") -> ToolExecutionReceipt:
    message = ToolMessage(
        id=f"tool-message-{call_id}",
        content=f"result for {call_id}",
        name=tool_name,
        tool_call_id=call_id,
    )
    return ToolExecutionReceipt(
        task_id=TASK_ID,
        tool_call_id=call_id,
        tool_name=tool_name,
        call_hash=f"hash-{call_id}",
        tool_message_data=message_to_dict(message),
        result_fingerprint=result_fingerprint(message),
    )


def _snapshot(
    artifact: ArtifactReference,
    *,
    version: int = 1,
    stale: bool = True,
) -> CompactionSnapshot:
    historical_test = SystemTestEvidence(
        evidence_id=CURRENT_TEST_ID,
        command="python -m pytest tests/test_parser.py -q",
        exit_code=1 if stale else 0,
        summary="stale historical failure" if stale else "current pass",
        tool_call_id="call-test",
        source_message_id="message-test",
    )
    return CompactionSnapshot(
        task_id=TASK_ID,
        version=version,
        previous_version=version - 1 or None,
        lifecycle="prepared",
        created_at=f"2026-08-30T00:00:0{version}+00:00",
        source_work_unit_ids=["work-unit-1"],
        coverage=SnapshotCoverage(
            last_user_message_id="message-user-1",
            covered_message_ids=["message-user-1"],
            covered_work_unit_ids=["work-unit-1"],
        ),
        task_goal="fix parser failure",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(tests=[historical_test]),
        active_hypotheses=[
            HypothesisRecord(
                hypothesis_id=CURRENT_HYPOTHESIS_ID,
                text=(
                    "stale snapshot root cause"
                    if stale
                    else "current parser branch root cause"
                ),
                state="active",
                sources=[ProvenanceRef(kind="work_unit", ref_id="work-unit-1")],
                updated_in_version=version,
            )
        ],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[
            ProvenancedText(
                text="Why does parsing fail only on Windows?",
                sources=[ProvenanceRef(kind="work_unit", ref_id="work-unit-1")],
            )
        ],
        next_steps=[],
        artifact_references=[artifact],
        content_hash=("c" if stale else "d") * 64,
    )


def _table_rows(connection, table: str, columns: str, order_by: str) -> list[tuple]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        return []
    return connection.execute(
        f"SELECT {columns} FROM {table} ORDER BY {order_by}"
    ).fetchall()


@dataclass
class MigratedFixture:
    root: Path
    artifact_root: Path
    database: SQLiteDatabase
    repositories: DomainRepositories
    reports: dict[str, DomainMigrationReport]
    legacy_hashes_before: dict[str, str]
    conversation_artifact: ArtifactReference
    research_evidence_id: str
    research_artifact: ArtifactReference

    def report(self, domain: str) -> DomainMigrationReport:
        key = "deterministic_evidence" if domain == "evidence" else domain
        return self.reports[key]

    def legacy_source_hashes(self) -> dict[str, str]:
        payload: dict[str, object] = {}
        with self.database.connection() as connection:
            for table, columns, order_by in (
                (
                    "deterministic_evidence",
                    "task_id, evidence_id, kind, payload",
                    "task_id, evidence_id",
                ),
                (
                    "research_queries",
                    "task_id, query_id, payload",
                    "task_id, query_id",
                ),
                (
                    "search_candidates",
                    "task_id, candidate_id, payload",
                    "task_id, candidate_id",
                ),
                (
                    "external_evidence",
                    "task_id, evidence_id, payload",
                    "task_id, evidence_id",
                ),
                (
                    "investigation_state",
                    "task_id, version, payload",
                    "task_id",
                ),
                (
                    "working_memory",
                    "task_id, version, payload",
                    "task_id, version",
                ),
                (
                    "operation_journal",
                    "operation_id, task_id, status, payload",
                    "operation_id",
                ),
                (
                    "compaction_snapshots",
                    "task_id, version, input_hash, payload",
                    "task_id, version",
                ),
                (
                    "context_migrations",
                    "task_id, version, event_payload",
                    "task_id",
                ),
            ):
                payload[table] = _table_rows(connection, table, columns, order_by)
            projection = connection.execute(
                "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
                (TASK_ID,),
            ).fetchone()
            legacy_task = json.loads(str(projection[0]))
            migrated_fields = frozenset().union(*MIGRATED_LEGACY_FIELDS.values())
            payload["legacy_task_migrated_fields"] = {
                key: legacy_task.get(key)
                for key in sorted(migrated_fields)
            }
        receipt_root = self.artifact_root / "investigation_receipts"
        payload["receipt_files"] = [
            (path.relative_to(receipt_root).as_posix(), path.read_text(encoding="utf-8"))
            for path in sorted(receipt_root.rglob("*.json"))
        ]
        return {
            key: hashlib.sha256(
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode(
                    "utf-8"
                )
            ).hexdigest()
            for key, value in payload.items()
        }

    def current_counts(self) -> dict[str, int]:
        with self.database.connection() as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in (
                    "evidence_records",
                    "research_attempts",
                    "hypotheses",
                    "receipts",
                    "history_snapshots",
                )
            }

    def run_normal_agent_persistence_cycle(self) -> tuple[dict[str, int], dict[str, int]]:
        before = self.current_counts()
        compaction = CompactionStore(self.database, repositories=self.repositories)
        research = ResearchEvidenceStore(self.database, repositories=self.repositories)
        compaction.save_evidence(
            TASK_ID,
            SystemTestEvidence(
                evidence_id="test-evidence-2",
                command="python -m pytest -q",
                exit_code=0,
                summary="full suite passed",
                tool_call_id="call-test-2",
                source_message_id="message-test-2",
            ),
        )
        research.save_query(TASK_ID, "pytest windows parser", ["github"], [])
        self.repositories.investigation.record_hypothesis(
            TASK_ID,
            InvestigationHypothesis(
                hypothesis_id="hypothesis-2",
                statement="newline normalization is relevant",
                state="candidate",
                evidence_ids=[],
                checked_locations=[],
                reason="new current investigation candidate",
            ),
        )
        ToolExecutionReceiptStore(
            self.artifact_root / "investigation_receipts",
            repository=self.repositories.execution,
        ).save(_receipt("call-current"))
        compaction.save_prepared_snapshot(
            _snapshot(self.conversation_artifact, version=2, stale=False),
            "input-current-2",
        )
        task = self.repositories.tasks.get(TASK_ID)
        task.final_summary = "current persistence cycle completed"
        self.repositories.tasks.save_legacy_projection(task)
        return before, self.current_counts()


@pytest.fixture
def migrated_fixture(tmp_path: Path) -> MigratedFixture:
    project = tmp_path / "project"
    project.mkdir()
    artifact_root = tmp_path / "artifacts"
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    tasks = TaskRepository(database)
    task = TaskState.create(project, "fix parser failure", ApprovalMode.MANUAL)
    task.task_id = TASK_ID
    task.status = TaskStatus.INVESTIGATING
    task.conversation = [
        {"id": "message-user-1", "role": "user", "content": task.user_problem}
    ]
    task.changed_files = ["src/parser.py"]
    task.hypotheses = ["current parser branch root cause"]
    task.approvals = [ApprovalRecord("execute pytest", "approve", "L1")]
    tasks.save(task)

    current_test = SystemTestEvidence(
        evidence_id=CURRENT_TEST_ID,
        command="python -m pytest tests/test_parser.py -q",
        exit_code=0,
        summary="current required oracle passed",
        tool_call_id="call-test",
        source_message_id="message-test",
    )
    current_file = FileChangeEvidence(
        evidence_id="file-evidence-1",
        path="src/parser.py",
        operation="edit",
        status="succeeded",
        tool_call_id="call-edit",
        source_message_id="message-edit",
    )
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE deterministic_evidence (
                task_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, evidence_id)
            )
            """
        )
        connection.executemany(
            "INSERT INTO deterministic_evidence VALUES (?, ?, ?, ?, ?)",
            [
                (
                    TASK_ID,
                    current_test.evidence_id,
                    "test",
                    current_test.model_dump_json(),
                    "2026-08-30T00:00:00+00:00",
                ),
                (
                    TASK_ID,
                    current_file.evidence_id,
                    "file",
                    current_file.model_dump_json(),
                    "2026-08-30T00:00:01+00:00",
                ),
            ],
        )

    query = ResearchQuery(
        query_id="query-1",
        task_id=TASK_ID,
        sanitized_query="pytest windows parser",
        providers=["github"],
        provider_errors=[],
        created_at="2026-08-30T00:01:00+00:00",
    )
    candidate = SearchCandidate(
        candidate_id="candidate-1",
        task_id=TASK_ID,
        source_type="official_docs",
        evidence_level="E1",
        title="Official parser documentation",
        url="https://example.test/parser",
        query=query.sanitized_query,
        repository=None,
        created_at="2026-08-30T00:01:01+00:00",
    )
    research_path = "/.deepfix-artifacts/research/task-1/evidence.md"
    research_file = artifact_root / "research" / TASK_ID / "evidence.md"
    research_file.parent.mkdir(parents=True)
    research_file.write_text("verified legacy research", encoding="utf-8")
    research_artifact = ArtifactReference(
        path=research_path,
        kind="research",
        content_hash=hashlib.sha256(research_file.read_bytes()).hexdigest(),
        work_unit_ids=[],
    )
    external = ExternalEvidence(
        evidence_id=RESEARCH_EVIDENCE_ID,
        task_id=TASK_ID,
        candidate_id=candidate.candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title=candidate.title,
        url=candidate.url,
        query=query.sanitized_query,
        relevant_excerpt="Parser behavior is platform independent.",
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
    with database.unit_of_work() as connection:
        connection.executescript(
            """
            CREATE TABLE research_queries (
                task_id TEXT NOT NULL, query_id TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, query_id)
            );
            CREATE TABLE search_candidates (
                task_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, candidate_id)
            );
            CREATE TABLE external_evidence (
                task_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL, payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id, evidence_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO research_queries VALUES (?, ?, ?, ?)",
            (TASK_ID, query.query_id, query.model_dump_json(), query.created_at),
        )
        connection.execute(
            "INSERT INTO search_candidates VALUES (?, ?, ?, ?)",
            (
                TASK_ID,
                candidate.candidate_id,
                candidate.model_dump_json(),
                candidate.created_at,
            ),
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

    hypothesis = InvestigationHypothesis(
        hypothesis_id=CURRENT_HYPOTHESIS_ID,
        statement="current parser branch root cause",
        state="supported",
        evidence_ids=[CURRENT_TEST_ID],
        checked_locations=[],
        reason="current deterministic test supports this diagnosis",
    )
    investigation_state = InvestigationState(
        task_id=TASK_ID,
        hypotheses=[hypothesis],
        supported_hypothesis_ids=[CURRENT_HYPOTHESIS_ID],
    )
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE investigation_state (
                task_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO investigation_state VALUES (?, ?, ?, ?)",
            (
                TASK_ID,
                1,
                investigation_state.model_dump_json(),
                "2026-08-30T00:02:00+00:00",
            ),
        )

    conversation_content = "legacy conversation artifact"
    conversation_file = artifact_root / "conversation_history" / TASK_ID / "history.md"
    conversation_file.parent.mkdir(parents=True)
    conversation_file.write_text(conversation_content, encoding="utf-8")
    conversation_artifact = ArtifactReference(
        path=f"/.deepfix-artifacts/conversation_history/{TASK_ID}/history.md",
        kind="conversation_history",
        content_hash=hashlib.sha256(conversation_content.encode("utf-8")).hexdigest(),
        work_unit_ids=["work-unit-1"],
    )
    snapshot = _snapshot(conversation_artifact)
    event = DeepFixCompactionEvent(
        event_id="event-1",
        task_id=TASK_ID,
        active_snapshot_version=1,
        snapshot_message_id="snapshot-message-1",
        retained_message_ids=["message-user-1"],
        conversation_artifact=conversation_artifact,
        input_hash="b" * 64,
    )
    with database.unit_of_work() as connection:
        connection.executescript(
            """
            CREATE TABLE compaction_snapshots (
                task_id TEXT NOT NULL, version INTEGER NOT NULL,
                lifecycle TEXT NOT NULL, input_hash TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                activated_at TEXT, abandoned_at TEXT, abandon_reason TEXT,
                PRIMARY KEY(task_id, version), UNIQUE(task_id, input_hash)
            );
            CREATE TABLE context_migrations (
                task_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                event_payload TEXT NOT NULL, migrated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO compaction_snapshots VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (
                TASK_ID,
                1,
                "prepared",
                "b" * 64,
                snapshot.model_dump_json(),
                snapshot.created_at,
            ),
        )
        connection.execute(
            "INSERT INTO context_migrations VALUES (?, ?, ?, ?)",
            (TASK_ID, 1, event.model_dump_json(), "2026-08-30T00:03:00+00:00"),
        )

    receipts = [_receipt("call-execute"), _receipt("call-read", "read_file")]
    legacy_receipts = ToolExecutionReceiptStore(
        artifact_root / "investigation_receipts"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(legacy_receipts.save, receipts))
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
        receipt_id=receipts[0].result_fingerprint,
        created_at=datetime(2026, 8, 30, 0, 4, tzinfo=UTC),
        updated_at=datetime(2026, 8, 30, 0, 4, tzinfo=UTC),
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
                operation.operation_id,
                TASK_ID,
                operation.call_hash,
                operation.status.value,
                operation.model_dump_json(),
                operation.created_at.isoformat(),
                operation.updated_at.isoformat(),
            ),
        )

    migrator = DomainMigrator(database, artifact_root=artifact_root)
    reports = {
        "deterministic_evidence": migrator.migrate_deterministic_evidence(TASK_ID),
        "research": migrator.migrate_research(TASK_ID),
        "investigation": migrator.migrate_investigation(TASK_ID),
        "execution": migrator.migrate_execution(TASK_ID),
        "history": migrator.migrate_history(TASK_ID),
    }
    repositories = DomainRepositories.create(database)
    fixture = MigratedFixture(
        root=project,
        artifact_root=artifact_root,
        database=database,
        repositories=repositories,
        reports=reports,
        legacy_hashes_before={},
        conversation_artifact=conversation_artifact,
        research_evidence_id=RESEARCH_EVIDENCE_ID,
        research_artifact=research_artifact,
    )
    fixture.legacy_hashes_before = fixture.legacy_source_hashes()
    return fixture


@pytest.mark.parametrize(
    ("domain", "expected_count"),
    [
        ("evidence", 2),
        ("research", 3),
        ("investigation", 2),
        ("execution", 4),
        ("history", 2),
    ],
)
def test_migration_report_has_no_identity_hash_or_reference_loss(
    migrated_fixture: MigratedFixture,
    domain: str,
    expected_count: int,
):
    report = migrated_fixture.report(domain)
    assert report.source_count == report.target_count == expected_count
    assert report.source_hash == report.target_hash
    assert report.identity_mismatches == []
    assert report.hash_mismatches == []
    assert report.missing_references == []
    assert report.ready_to_switch


def test_rollback_sources_are_read_only_after_switch(
    migrated_fixture: MigratedFixture,
):
    before_counts, after_counts = migrated_fixture.run_normal_agent_persistence_cycle()

    assert migrated_fixture.legacy_source_hashes() == (
        migrated_fixture.legacy_hashes_before
    )
    assert all(after_counts[name] > before_counts[name] for name in before_counts)


def test_legacy_projection_write_is_rejected_while_source_migration_is_fenced(
    migrated_fixture: MigratedFixture,
):
    task = migrated_fixture.repositories.tasks.get(TASK_ID)
    task.approvals = [ApprovalRecord("execute", "approve", "L2")]
    migrator = DomainMigrator(
        migrated_fixture.database,
        artifact_root=migrated_fixture.artifact_root,
    )

    with (
        migrator._migration_fence("execution", TASK_ID),
        pytest.raises(DomainMigrationInProgress),
    ):
        migrated_fixture.repositories.tasks.save_legacy_projection(task)


def test_legacy_receipt_write_waits_for_execution_migration_fence(
    migrated_fixture: MigratedFixture,
):
    receipt_root = migrated_fixture.artifact_root / "investigation_receipts"
    store = ToolExecutionReceiptStore(receipt_root)
    started = Event()

    def save_late_receipt():
        started.set()
        store.save(_receipt("call-after-fence"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with legacy_receipt_task_guard(receipt_root, TASK_ID):
            future = pool.submit(save_late_receipt)
            assert started.wait(timeout=2)
            assert not future.done()
        with pytest.raises(LegacyReceiptWritesDisabled):
            future.result(timeout=5)

    assert store.load(TASK_ID, "call-after-fence") is None


@pytest.mark.parametrize("marker_exists", [False, True])
def test_stale_migration_fence_is_recoverable_after_crash(
    migrated_fixture: MigratedFixture,
    marker_exists: bool,
):
    domain = "crash-test-domain"
    migrator = DomainMigrator(
        migrated_fixture.database,
        artifact_root=migrated_fixture.artifact_root,
    )
    with migrated_fixture.database.unit_of_work() as connection:
        connection.execute(
            """
            INSERT INTO domain_migration_fences(
                domain, task_id, owner_id, started_at
            ) VALUES (?, ?, 'crashed-owner', '2000-01-01 00:00:00')
            """,
            (domain, TASK_ID),
        )
        if marker_exists:
            connection.execute(
                """
                INSERT INTO domain_migrations(
                    domain, task_id, report_json, switched_at
                ) VALUES (?, ?, '{}', '2000-01-01 00:00:01')
                """,
                (domain, TASK_ID),
            )

    with migrator._migration_fence(domain, TASK_ID):
        with migrated_fixture.database.connection() as connection:
            row = connection.execute(
                """
                SELECT owner_id FROM domain_migration_fences
                WHERE domain = ? AND task_id = ?
                """,
                (domain, TASK_ID),
            ).fetchone()
        assert row is not None
        assert str(row[0]) != "crashed-owner"

    with migrated_fixture.database.connection() as connection:
        remaining = connection.execute(
            """
            SELECT 1 FROM domain_migration_fences
            WHERE domain = ? AND task_id = ?
            """,
            (domain, TASK_ID),
        ).fetchone()
    assert remaining is None


def test_expired_owner_cannot_delete_replacement_migration_lease(
    migrated_fixture: MigratedFixture,
):
    domain = "lease-owner-test-domain"
    migrator = DomainMigrator(
        migrated_fixture.database,
        artifact_root=migrated_fixture.artifact_root,
    )
    old_lease = migrator._migration_fence(domain, TASK_ID)
    new_lease = migrator._migration_fence(domain, TASK_ID)
    old_lease.__enter__()
    try:
        with migrated_fixture.database.unit_of_work() as connection:
            old_owner = str(
                connection.execute(
                    """
                    SELECT owner_id FROM domain_migration_fences
                    WHERE domain = ? AND task_id = ?
                    """,
                    (domain, TASK_ID),
                ).fetchone()[0]
            )
            connection.execute(
                """
                UPDATE domain_migration_fences
                SET started_at = '2000-01-01 00:00:00'
                WHERE domain = ? AND task_id = ?
                """,
                (domain, TASK_ID),
            )
        new_lease.__enter__()
        with migrated_fixture.database.connection() as connection:
            new_owner = str(
                connection.execute(
                    """
                    SELECT owner_id FROM domain_migration_fences
                    WHERE domain = ? AND task_id = ?
                    """,
                    (domain, TASK_ID),
                ).fetchone()[0]
            )
        assert new_owner != old_owner

        old_lease.__exit__(None, None, None)
        with migrated_fixture.database.connection() as connection:
            remaining_owner = str(
                connection.execute(
                    """
                    SELECT owner_id FROM domain_migration_fences
                    WHERE domain = ? AND task_id = ?
                    """,
                    (domain, TASK_ID),
                ).fetchone()[0]
            )
        assert remaining_owner == new_owner
    finally:
        new_lease.__exit__(None, None, None)


def test_restore_context_reporting_and_execution_reconciliation_use_current_authority(
    migrated_fixture: MigratedFixture,
):
    repositories = migrated_fixture.repositories
    compaction = CompactionStore(
        migrated_fixture.database,
        repositories=repositories,
    )
    research = ResearchEvidenceStore(
        migrated_fixture.database,
        repositories=repositories,
    )
    investigation_store = InvestigationStore(
        migrated_fixture.database,
        repositories=repositories,
    )
    memory = WorkingMemoryStore(migrated_fixture.database.path)
    coordinator = InvestigationCoordinator(
        store=investigation_store,
        tasks=repositories.tasks,
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, research),
    )
    config = AppConfig(
        project_root=migrated_fixture.root,
        database_path=migrated_fixture.database.path,
        artifacts_path=migrated_fixture.artifact_root,
        main_model=ModelRoleConfig(
            model_name="offline-main",
            api_key=SecretStr("offline-main-key"),
            base_url="https://example.test/v1",
        ),
        compaction_model=ModelRoleConfig(
            model_name="offline-compaction",
            api_key=SecretStr("offline-compaction-key"),
            base_url="https://example.test/v1",
        ),
        approval_mode=ApprovalMode.MANUAL,
        project_python=Path(__import__("sys").executable).resolve(),
    )
    service = BugfixService(
        object(),
        repositories.tasks,
        ApprovalPolicy(ApprovalMode.MANUAL),
        config,
        memory,
        research,
        compaction,
        coordinator,
        repositories=repositories,
    )
    restored = service._load_current_task(TASK_ID)
    event = repositories.history.migrated_event(TASK_ID)
    assert event is not None
    context = ProtectedContextBuilder(
        repositories.tasks,
        memory,
        compaction,
        research,
        EvidenceCollector(compaction, research),
        investigation_store,
    ).build(
        TASK_ID,
        [HumanMessage(id="message-user-1", content="fix parser failure")],
        event,
    )
    rendered_context = render_protected_context(context)
    report = render_report(restored, research.list_evidence(TASK_ID))
    receipts = ToolExecutionReceiptStore(
        migrated_fixture.artifact_root / "investigation_receipts",
        repository=repositories.execution,
    )
    reconciler = OperationReconciler(
        OperationJournalStore(
            migrated_fixture.database.path,
            repositories=repositories,
        ),
        receipts,
    )

    first = reconciler.reconcile_task(TASK_ID, migrated_fixture.root)
    second = reconciler.reconcile_task(TASK_ID, migrated_fixture.root)
    migrated_receipt = _receipt("call-execute")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                repositories.execution.record_receipt,
                [migrated_receipt] * 4,
            )
        )
    projected = repositories.history.project_snapshot(
        TASK_ID,
        1,
        evidence=repositories.evidence,
        investigation=repositories.investigation,
    )

    assert restored.changed_files == ["src/parser.py"]
    assert [item.evidence_id for item in projected.deterministic_evidence.tests] == [
        CURRENT_TEST_ID
    ]
    assert projected.deterministic_evidence.tests[0].exit_code == 0
    assert projected.confirmed_hypotheses[0].text == "current parser branch root cause"
    assert "current parser branch root cause" in rendered_context
    assert "stale snapshot root cause" not in rendered_context
    assert "current required oracle passed" in report
    assert migrated_fixture.research_evidence_id in restored.external_evidence_ids
    research_envelope = repositories.evidence.get(
        TASK_ID,
        migrated_fixture.research_evidence_id,
    )
    assert research_envelope.artifact_references == [
        migrated_fixture.research_artifact
    ]
    assert [
        question.text
        for question in repositories.investigation.list_questions(TASK_ID)
    ] == ["Why does parsing fail only on Windows?"]
    history_record = repositories.history.get(TASK_ID, 1)
    assert history_record.artifact_references[0] == (
        migrated_fixture.conversation_artifact
    )
    assert repositories.execution.integrity_view(TASK_ID).receipt_count == 2
    assert len(repositories.execution.list_operations(TASK_ID)) == 1
    assert first.replayable_operation_ids == ["operation-1"]
    assert second.replayable_operation_ids == ["operation-1"]
