from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from deepfix.compaction.models import (
    CompactionSnapshot,
    DeterministicEvidenceBlock,
    HypothesisRecord,
    ProvenancedText,
    ProvenanceRef,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.investigation import (
    HypothesisIdentityConflict,
    HypothesisTransitionError,
    InvestigationEvidenceMissing,
    InvestigationRepository,
)
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.investigation.models import (
    InvestigationHypothesis,
    InvestigationState,
    UnresolvedQuestion,
)


def _hypothesis(
    hypothesis_id: str,
    *,
    statement: str = "parser branch mishandles empty input",
    state: str = "candidate",
    evidence_ids: list[str] | None = None,
    reopens_hypothesis_id: str | None = None,
) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        hypothesis_id=hypothesis_id,
        statement=statement,
        state=state,
        evidence_ids=evidence_ids or [],
        checked_locations=[],
        reason=f"{state} reason",
        reopens_hypothesis_id=reopens_hypothesis_id,
    )


def _question(question_id: str = "q-1") -> UnresolvedQuestion:
    return UnresolvedQuestion(
        question_id=question_id,
        task_id="task-1",
        text="Why does the parser fail only on Windows?",
        status="open",
        source_ids=["message-1"],
        resolution_evidence_ids=[],
        created_at="2026-08-29T00:00:00+00:00",
    )


def _record_test_evidence(database, evidence_id: str = "e-1") -> None:
    EvidenceRepository(database).record_deterministic(
        "task-1",
        SystemTestEvidence(
            evidence_id=evidence_id,
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="call-1",
            source_message_id="message-2",
        ),
        provenance_root_ids=["call-1"],
    )


def test_hypothesis_transition_requires_stable_id_and_current_evidence(tmp_path):
    database = tmp_path / "deepfix.db"
    _record_test_evidence(database)
    repository = InvestigationRepository(database)
    candidate = _hypothesis("h-1")

    repository.record_hypothesis("task-1", candidate)
    supported = candidate.model_copy(update={"state": "supported", "evidence_ids": ["e-1"]})

    assert repository.record_hypothesis("task-1", supported) == supported
    with pytest.raises(HypothesisIdentityConflict):
        repository.record_hypothesis(
            "task-1",
            _hypothesis("h-1", statement="different statement"),
        )


def test_hypothesis_transition_rejects_missing_or_invalid_evidence(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")
    candidate = _hypothesis("h-1")
    repository.record_hypothesis("task-1", candidate)

    with pytest.raises(InvestigationEvidenceMissing):
        repository.record_hypothesis(
            "task-1",
            candidate.model_copy(update={"state": "supported", "evidence_ids": ["missing"]}),
        )


def test_rejected_hypothesis_reopens_only_under_new_identity(tmp_path):
    database = tmp_path / "deepfix.db"
    _record_test_evidence(database)
    repository = InvestigationRepository(database)
    candidate = _hypothesis("h-1")
    repository.record_hypothesis("task-1", candidate)
    rejected = candidate.model_copy(update={"state": "rejected", "evidence_ids": ["e-1"]})
    repository.record_hypothesis("task-1", rejected)

    with pytest.raises(HypothesisTransitionError):
        repository.record_hypothesis(
            "task-1",
            rejected.model_copy(update={"state": "candidate"}),
        )
    with pytest.raises(HypothesisTransitionError, match="reopens_hypothesis_id"):
        repository.record_hypothesis(
            "task-1",
            _hypothesis("h-2", statement=rejected.statement),
        )

    reopened = _hypothesis(
        "h-2",
        statement=rejected.statement,
        evidence_ids=["e-1"],
        reopens_hypothesis_id="h-1",
    )
    assert repository.record_hypothesis("task-1", reopened) == reopened


def test_question_is_not_todo_or_task_lifecycle_state(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")

    question = repository.open_question(_question())

    assert question.status == "open"
    assert not hasattr(question, "todo_status")
    assert not hasattr(repository, "transition_task")
    assert not hasattr(repository, "can_execute")
    assert "owner" not in inspect.signature(repository.open_question).parameters


def test_question_resolution_requires_current_evidence_id(tmp_path):
    database = tmp_path / "deepfix.db"
    repository = InvestigationRepository(database)
    repository.open_question(_question())

    with pytest.raises(InvestigationEvidenceMissing):
        repository.resolve_question("task-1", "q-1", evidence_ids=["missing"])

    _record_test_evidence(database)
    resolved = repository.resolve_question("task-1", "q-1", evidence_ids=["e-1"])

    assert resolved.status == "resolved"
    assert resolved.resolution_evidence_ids == ["e-1"]
    assert resolved.resolved_at is not None


def test_migration_keeps_current_hypothesis_over_stale_snapshot_projection(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    current = _hypothesis("h-1", statement="current parser root cause")
    state = InvestigationState(task_id="task-1", hypotheses=[current])
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE investigation_state (
                task_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO investigation_state VALUES (?, ?, ?, ?)",
            ("task-1", 1, state.model_dump_json(), "2026-08-29T00:00:00+00:00"),
        )

    snapshot = CompactionSnapshot(
        task_id="task-1",
        version=1,
        lifecycle="prepared",
        created_at="2026-08-29T00:00:00+00:00",
        source_work_unit_ids=[],
        task_goal="fix parser",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(),
        active_hypotheses=[
            HypothesisRecord(
                hypothesis_id="h-1",
                text="stale snapshot explanation",
                state="active",
                sources=[ProvenanceRef(kind="snapshot_record", ref_id="snapshot-1")],
                updated_in_version=1,
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
                text="Why does this fail only on Windows?",
                sources=[ProvenanceRef(kind="snapshot_record", ref_id="snapshot-question-1")],
            )
        ],
        next_steps=[],
        artifact_references=[],
        content_hash="a" * 64,
    )
    CompactionStore(database).save_prepared_snapshot(snapshot, "input-1")
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE working_memory (
                task_id TEXT NOT NULL, version INTEGER NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, version)
            )
            """
        )
        connection.execute(
            "INSERT INTO working_memory(task_id, version, payload, created_at) VALUES (?, ?, ?, ?)",
            (
                "task-1",
                1,
                json.dumps(
                    {
                        "phase": "investigating",
                        "summary": "legacy memory",
                        "facts": [],
                        "evidence": [],
                        "active_hypotheses": [],
                        "rejected_hypotheses": [],
                        "confirmed_hypotheses": [],
                        "checked_files": [],
                        "experiments": [],
                        "next_steps": [],
                        "unresolved_questions": ["Why does this fail only on Windows?"],
                        "coverage": {"covered_message_ids": [], "covered_work_unit_ids": []},
                    }
                ),
                "2026-08-31T00:00:00+00:00",
            ),
        )
    legacy_hash_before = hashlib.sha256(state.model_dump_json().encode("utf-8")).hexdigest()

    report = DomainMigrator(database).migrate_investigation("task-1")
    repository = InvestigationRepository(database)

    assert report.source_count == report.target_count == 2
    assert report.ready_to_switch
    assert report.identity_mismatches == []
    assert report.hash_mismatches == []
    assert repository.get_hypothesis("task-1", "h-1").statement == current.statement
    questions = repository.list_questions("task-1")
    assert [item.text for item in questions] == ["Why does this fail only on Windows?"]
    assert questions[0].source_ids == [
        "snapshot_record:snapshot-question-1",
        "working_memory:1",
    ]
    assert repository.list_conflicts("task-1") == [
        {
            "hypothesis_id": "h-1",
            "current_statement": current.statement,
            "historical_statement": "stale snapshot explanation",
            "source_id": "snapshot_record:snapshot-1",
        }
    ]
    with database.connection() as connection:
        payload = connection.execute(
            "SELECT payload FROM investigation_state WHERE task_id = 'task-1'"
        ).fetchone()[0]
    assert hashlib.sha256(str(payload).encode("utf-8")).hexdigest() == legacy_hash_before


def test_migration_refuses_terminal_hypothesis_with_missing_evidence(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    dangling = _hypothesis(
        "h-dangling",
        state="supported",
        evidence_ids=["missing-evidence"],
    )
    state = InvestigationState(task_id="task-1", hypotheses=[dangling])
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE investigation_state (
                task_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO investigation_state VALUES (?, ?, ?, ?)",
            ("task-1", 1, state.model_dump_json(), "2026-08-29T00:00:00+00:00"),
        )

    report = DomainMigrator(database).migrate_investigation("task-1")

    assert not report.ready_to_switch
    assert report.missing_references == ["hypothesis:h-dangling:evidence:missing-evidence"]
    with database.connection() as connection:
        marker = connection.execute(
            """
            SELECT 1 FROM domain_migrations
            WHERE domain = 'investigation' AND task_id = 'task-1'
            """
        ).fetchone()
    assert marker is None
