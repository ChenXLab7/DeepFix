from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deepfix.compaction.models import (
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    ProvenanceRef,
    SnapshotCoverage,
    SystemTestEvidence,
)
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.history import (
    HistoricalSemanticItem,
    HistoryRepository,
    HistorySnapshotRecord,
)
from deepfix.domain_repositories.investigation import InvestigationRepository, stable_question_id
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.investigation.models import InvestigationHypothesis, UnresolvedQuestion


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        path="/.deepfix-artifacts/conversation_history/task-1.md",
        kind="conversation_history",
        content_hash="a" * 64,
        work_unit_ids=["wu-1"],
    )


def _item(
    kind: str,
    semantic_id: str,
    payload_type: str,
    payload: dict[str, object],
) -> HistoricalSemanticItem:
    return HistoricalSemanticItem(
        item_id=f"history-{kind}-{semantic_id}",
        task_id="task-1",
        snapshot_version=1,
        kind=kind,
        semantic_id=semantic_id,
        payload_type=payload_type,
        payload=payload,
        provenance_root_ids=["wu-1"],
    )


def _record(*items: HistoricalSemanticItem) -> HistorySnapshotRecord:
    return HistorySnapshotRecord(
        task_id="task-1",
        version=1,
        previous_version=None,
        lifecycle="prepared",
        input_hash="b" * 64,
        created_at="2026-08-30T00:00:00+00:00",
        coverage=SnapshotCoverage(
            last_user_message_id="m-2",
            covered_message_ids=["m-1", "m-2"],
            covered_work_unit_ids=["wu-1"],
        ),
        source_work_unit_ids=["wu-1"],
        artifact_references=[_artifact()],
        content_hash="c" * 64,
        semantic_items=list(items),
    )


def _event(version: int = 1) -> DeepFixCompactionEvent:
    return DeepFixCompactionEvent(
        event_id=f"event-{version}",
        task_id="task-1",
        active_snapshot_version=version,
        snapshot_message_id=f"snapshot-message-{version}",
        retained_message_ids=["m-2"],
        conversation_artifact=_artifact(),
        input_hash="b" * 64,
    )


def test_history_record_contains_coverage_and_sources_but_not_current_domain_tables(
    tmp_path: Path,
) -> None:
    repository = HistoryRepository(tmp_path / "deepfix.db")

    record = repository.save_prepared(_record())

    assert record.coverage.covered_message_ids == ["m-1", "m-2"]
    assert record.artifact_references[0].content_hash == "a" * 64
    assert not hasattr(record, "deterministic_evidence")
    assert not hasattr(record, "active_hypotheses")


def test_prepared_replay_is_idempotent_and_only_matching_event_activates(
    tmp_path: Path,
) -> None:
    repository = HistoryRepository(tmp_path / "deepfix.db")
    first = repository.save_prepared(_record())
    replay = repository.save_prepared(_record())

    assert replay == first
    with pytest.raises(ValueError, match="version"):
        repository.activate("task-1", 1, _event(2))

    active = repository.activate("task-1", 1, _event())
    assert active.lifecycle == "active"
    with pytest.raises(ValueError, match="active"):
        repository.abandon("task-1", 1, "superseded")


def test_active_from_event_validates_identity_and_excludes_abandoned(
    tmp_path: Path,
) -> None:
    repository = HistoryRepository(tmp_path / "deepfix.db")
    repository.save_prepared(_record())

    assert repository.active_from_event("task-1", _event()).lifecycle == "prepared"
    with pytest.raises(ValueError, match="version"):
        repository.active_from_event(
            "task-1",
            _event().model_copy(update={"task_id": "other-task"}),
        )
    with pytest.raises(ValueError, match="input_hash"):
        repository.active_from_event(
            "task-1",
            _event().model_copy(update={"input_hash": "wrong"}),
        )
    with pytest.raises(ValueError, match="Artifact"):
        repository.active_from_event(
            "task-1",
            _event().model_copy(
                update={
                    "conversation_artifact": _artifact().model_copy(
                        update={"content_hash": "wrong"}
                    )
                }
            ),
        )

    repository.abandon("task-1", 1, "superseded")
    assert repository.active_from_event("task-1", _event()) is None


def test_unverified_artifact_prevents_prepared_record(tmp_path: Path) -> None:
    repository = HistoryRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda reference: False,
    )

    with pytest.raises(ValueError, match="Artifact verification failed"):
        repository.save_prepared(_record())

    assert repository.list_for_task("task-1") == []


def test_current_investigation_record_overrides_stale_history_item(
    tmp_path: Path,
) -> None:
    database = tmp_path / "deepfix.db"
    old = HypothesisRecord(
        hypothesis_id="h-1",
        text="old parser hypothesis",
        state="active",
        reason="historical",
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
        updated_in_version=1,
    )
    history = HistoryRepository(database)
    history.save_prepared(
        _record(
            _item(
                "hypothesis",
                "h-1",
                "HypothesisRecord",
                old.model_dump(mode="json"),
            )
        )
    )
    investigation = InvestigationRepository(database)
    investigation.record_hypothesis(
        "task-1",
        InvestigationHypothesis(
            hypothesis_id="h-1",
            statement="current parser hypothesis",
            state="candidate",
            evidence_ids=[],
            checked_locations=[],
            reason="current domain state",
        ),
    )

    projected = history.project_snapshot(
        "task-1",
        1,
        evidence=EvidenceRepository(database),
        investigation=investigation,
    )
    hypotheses = [
        *projected.active_hypotheses,
        *projected.rejected_hypotheses,
        *projected.confirmed_hypotheses,
    ]

    assert [(item.hypothesis_id, item.text) for item in hypotheses] == [
        ("h-1", "current parser hypothesis")
    ]


def test_current_evidence_claim_overrides_stale_history_claim(tmp_path: Path) -> None:
    database = tmp_path / "deepfix.db"
    stale = ProvenancedClaim(
        claim_id="claim-1",
        text="old parser fact",
        state="confirmed",
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
    )
    current = stale.model_copy(update={"text": "current parser fact"})
    history = HistoryRepository(database)
    history.save_prepared(
        _record(
            _item(
                "fact",
                "claim-1",
                "ProvenancedClaim",
                stale.model_dump(mode="json"),
            )
        )
    )
    evidence = EvidenceRepository(database)
    evidence.record_semantic_candidate(
        "task-1",
        current,
        provenance_root_ids=["wu-current"],
    )

    projected = history.project_snapshot(
        "task-1",
        1,
        evidence=evidence,
        investigation=InvestigationRepository(database),
    )

    assert [(item.claim_id, item.text) for item in projected.confirmed_facts] == [
        ("claim-1", "current parser fact")
    ]


def test_resolved_current_question_suppresses_stale_history_question(
    tmp_path: Path,
) -> None:
    database = tmp_path / "deepfix.db"
    question_id = stable_question_id("task-1", "why does parsing fail?")
    stale = ProvenancedText(
        text="why does parsing fail?",
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
    )
    history = HistoryRepository(database)
    history.save_prepared(
        _record(
            _item(
                "unresolved_question",
                question_id,
                "ProvenancedText",
                stale.model_dump(mode="json"),
            )
        )
    )
    evidence = EvidenceRepository(database)
    evidence.record_deterministic(
        "task-1",
        SystemTestEvidence(
            evidence_id="e-1",
            command="pytest -q",
            exit_code=0,
            summary="passed",
            tool_call_id="call-1",
            source_message_id="message-1",
        ),
        provenance_root_ids=["call-1"],
    )
    investigation = InvestigationRepository(database)
    investigation.open_question(
        UnresolvedQuestion(
            question_id=question_id,
            task_id="task-1",
            text=stale.text,
            status="open",
            source_ids=["wu-1"],
            created_at="2026-08-30T00:00:00+00:00",
        )
    )
    investigation.resolve_question("task-1", question_id, evidence_ids=["e-1"])

    projected = history.project_snapshot(
        "task-1",
        1,
        evidence=evidence,
        investigation=investigation,
    )

    assert projected.unresolved_questions == []


def test_history_migration_backfills_snapshot_failure_and_event_without_mutating_legacy(
    tmp_path: Path,
) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    historical_test = SystemTestEvidence(
        evidence_id="legacy-test-1",
        command="pytest legacy.py -q",
        exit_code=1,
        summary="legacy failure",
        tool_call_id="legacy-call-1",
        source_message_id="legacy-message-1",
    )
    snapshot = CompactionSnapshot(
        task_id="task-1",
        version=1,
        lifecycle="prepared",
        created_at="2026-08-30T00:00:00+00:00",
        source_work_unit_ids=["wu-1"],
        coverage=SnapshotCoverage(
            covered_message_ids=["m-1"],
            covered_work_unit_ids=["wu-1"],
        ),
        task_goal="fix parser",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(tests=[historical_test]),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[_artifact()],
        content_hash="c" * 64,
    )
    failure = CompactionFailureRecord(
        attempt_id="attempt-1",
        task_id="task-1",
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="snapshot_write",
        error_code="snapshot_write_failed",
        input_hash="d" * 64,
        original_messages_preserved=True,
        recorded_at="2026-08-30T00:01:00+00:00",
    )
    event = _event()
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
            CREATE TABLE compaction_failures (
                task_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
                stage TEXT NOT NULL, payload TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(task_id, attempt_id, stage)
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
                "task-1",
                1,
                "prepared",
                "b" * 64,
                snapshot.model_dump_json(),
                snapshot.created_at,
            ),
        )
        connection.execute(
            "INSERT INTO compaction_failures VALUES (?, ?, ?, ?, ?)",
            (
                "task-1",
                failure.attempt_id,
                failure.stage,
                failure.model_dump_json(),
                failure.recorded_at,
            ),
        )
        connection.execute(
            "INSERT INTO context_migrations VALUES (?, ?, ?, ?)",
            ("task-1", 1, event.model_dump_json(), "2026-08-30T00:02:00+00:00"),
        )
    before = hashlib.sha256(snapshot.model_dump_json().encode("utf-8")).hexdigest()

    migrator = DomainMigrator(database)
    first = migrator.migrate_history("task-1")
    second = migrator.migrate_history("task-1")

    assert first == second
    assert first.ready_to_switch
    assert first.source_count == first.target_count == 3
    history = HistoryRepository(database)
    record = history.get("task-1", 1)
    assert record.coverage.covered_message_ids == ["m-1"]
    historical_evidence = [
        item for item in record.semantic_items if item.kind == "evidence"
    ]
    assert [item.semantic_id for item in historical_evidence] == ["legacy-test-1"]
    assert historical_evidence[0].provenance_root_ids == [
        "legacy-test-1",
        "legacy-call-1",
        "legacy-message-1",
    ]
    projected = history.project_snapshot(
        "task-1",
        1,
        evidence=EvidenceRepository(database),
        investigation=InvestigationRepository(database),
    )
    assert projected.deterministic_evidence.tests == []
    assert history.list_failures("task-1") == [failure]
    assert history.migrated_event("task-1") == event
    with database.connection() as connection:
        payload = connection.execute(
            "SELECT payload FROM compaction_snapshots WHERE task_id = 'task-1'"
        ).fetchone()[0]
    assert hashlib.sha256(str(payload).encode("utf-8")).hexdigest() == before


def test_history_migration_rejects_missing_conversation_artifact(
    tmp_path: Path,
) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    snapshot = CompactionSnapshot(
        task_id="task-1",
        version=1,
        lifecycle="prepared",
        created_at="2026-08-30T00:00:00+00:00",
        source_work_unit_ids=["wu-1"],
        coverage=SnapshotCoverage(covered_work_unit_ids=["wu-1"]),
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
        artifact_references=[_artifact()],
        content_hash="c" * 64,
    )
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE compaction_snapshots (
                task_id TEXT NOT NULL, version INTEGER NOT NULL,
                lifecycle TEXT NOT NULL, input_hash TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                activated_at TEXT, abandoned_at TEXT, abandon_reason TEXT,
                PRIMARY KEY(task_id, version), UNIQUE(task_id, input_hash)
            )
            """
        )
        connection.execute(
            "INSERT INTO compaction_snapshots VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (
                "task-1",
                1,
                "prepared",
                "b" * 64,
                snapshot.model_dump_json(),
                snapshot.created_at,
            ),
        )

    report = DomainMigrator(database, artifact_root=artifact_root).migrate_history(
        "task-1"
    )

    assert not report.ready_to_switch
    assert report.missing_references == [_artifact().path]
    assert HistoryRepository(database).list_for_task("task-1") == []
