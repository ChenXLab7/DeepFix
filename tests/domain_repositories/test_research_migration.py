from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from deepfix.compaction.models import ArtifactReference
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceAuthority,
    EvidenceRepository,
    EvidenceVerification,
)
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.models import Evidence
from deepfix.research.models import ExternalEvidence, ResearchQuery, SearchCandidate
from deepfix.research.store import ResearchEvidenceStore


def _candidate(*, candidate_id: str = "candidate-1") -> SearchCandidate:
    return SearchCandidate(
        candidate_id=candidate_id,
        task_id="task-1",
        source_type="official_docs",
        evidence_level="E1",
        title="Official pytest timeout documentation",
        url="https://example.test/docs/timeout",
        query="pytest timeout",
        repository=None,
        created_at="2026-08-29T00:00:00+00:00",
    )


def _external(
    *,
    evidence_id: str | None = None,
    candidate_id: str = "candidate-1",
    verification: str = "unverified",
) -> ExternalEvidence:
    return ExternalEvidence(
        evidence_id=evidence_id or uuid4().hex,
        task_id="task-1",
        candidate_id=candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title="Official pytest timeout documentation",
        url="https://example.test/docs/timeout",
        query="pytest timeout",
        relevant_excerpt="The timeout option terminates hanging tests.",
        retrieved_at="2026-08-29T00:00:01+00:00",
        dependency_name="pytest-timeout",
        documented_version="2.4.0",
        project_version="2.4.0",
        local_verification=verification,
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
        artifact_path="/.deepfix-artifacts/research/task-1/evidence.md",
    )


def _reference(content: str) -> ArtifactReference:
    return ArtifactReference(
        path="/.deepfix-artifacts/research/task-1/evidence.md",
        kind="research",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        work_unit_ids=[],
    )


def test_research_attempt_audit_redacts_provider_secrets_and_tracks_candidates(
    tmp_path: Path,
):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    attempt = repository.record_research_attempt(
        task_id="task-1",
        query_id="query-1",
        sanitized_query="pytest timeout",
        providers=["github"],
        provider_errors=["Authorization: Bearer secret-token"],
    )
    repository.record_research_candidates("query-1", [_candidate()])

    restored = repository.get_research_attempt("task-1", "query-1")

    assert attempt.query_id == "query-1"
    assert restored.provider_errors == ["Authorization: Bearer [REDACTED]"]
    assert restored.candidate_ids == ["candidate-1"]
    assert repository.get_research_candidate("task-1", "candidate-1") == _candidate()
    assert repository.research_summary("task-1") == (
        1,
        ["Authorization: Bearer [REDACTED]"],
    )


def test_external_evidence_update_preserves_immutable_revision_history(tmp_path: Path):
    content = "verified artifact body"
    reference = _reference(content)
    repository = EvidenceRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda item: item == reference,
    )
    original = _external()
    first = repository.accept_external(
        original,
        provenance_root_ids=["url:https://example.test/docs/timeout"],
        artifact_references=[reference],
    )
    verified = original.model_copy(
        update={
            "local_verification": "verified",
            "local_evidence": [Evidence("tests/test_timeout.py", "1 passed")],
            "linked_test_tool_call_ids": ["pytest-1"],
            "verification_explanation": "Local test passed",
        }
    )

    current = repository.update_external(verified)
    revisions = repository.list_revisions("task-1", original.evidence_id)

    assert first.authority is EvidenceAuthority.RESEARCH
    assert current.verification_state is EvidenceVerification.VERIFIED
    assert [item.verification_state for item in revisions] == [
        EvidenceVerification.UNVERIFIED,
        EvidenceVerification.VERIFIED,
    ]
    assert revisions[0].content_hash == first.content_hash
    assert revisions[1].content_hash == current.content_hash


def test_research_store_facade_writes_new_authority_not_legacy_tables(tmp_path: Path):
    content = "artifact body"
    reference = _reference(content)
    store = ResearchEvidenceStore(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda item: item == reference,
    )
    query = store.save_query("task-1", "pytest timeout", ["github"], [])
    candidate = store.save_candidates("task-1", query.sanitized_query, [_candidate()])[0]
    evidence = _external(candidate_id=candidate.candidate_id)

    store.save_evidence(evidence, artifact_reference=reference)

    assert store.get_evidence("task-1", evidence.evidence_id) == evidence
    assert store.list_evidence("task-1") == [evidence]
    with SQLiteDatabase(tmp_path / "deepfix.db").connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM research_queries").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM search_candidates").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM external_evidence").fetchone()[0] == 0


def _seed_legacy_research(database: SQLiteDatabase, artifact_root: Path) -> ExternalEvidence:
    query = ResearchQuery(
        query_id="query-1",
        task_id="task-1",
        sanitized_query="pytest timeout",
        providers=["github"],
        provider_errors=[],
        created_at="2026-08-29T00:00:00+00:00",
    )
    candidate = _candidate()
    evidence = _external(evidence_id=uuid4().hex)
    artifact = artifact_root / "research" / "task-1" / "evidence.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("legacy research artifact", encoding="utf-8")
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
            (query.task_id, query.query_id, query.model_dump_json(), query.created_at),
        )
        connection.execute(
            "INSERT INTO search_candidates VALUES (?, ?, ?, ?)",
            (
                candidate.task_id,
                candidate.candidate_id,
                candidate.model_dump_json(),
                candidate.created_at,
            ),
        )
        connection.execute(
            "INSERT INTO external_evidence VALUES (?, ?, ?, ?, ?)",
            (
                evidence.task_id,
                evidence.evidence_id,
                evidence.candidate_id,
                evidence.model_dump_json(),
                evidence.retrieved_at,
            ),
        )
    return evidence


def _legacy_research_hash(database: SQLiteDatabase) -> str:
    payload: list[tuple[str, str, str]] = []
    with database.connection() as connection:
        for table, id_column in (
            ("research_queries", "query_id"),
            ("search_candidates", "candidate_id"),
            ("external_evidence", "evidence_id"),
        ):
            rows = connection.execute(
                f"SELECT {id_column}, payload FROM {table} ORDER BY {id_column}"
            ).fetchall()
            payload.extend((table, str(row[0]), str(row[1])) for row in rows)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_research_backfill_preserves_ids_hashes_artifact_and_legacy_rows(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    artifact_root = tmp_path / "artifacts"
    evidence = _seed_legacy_research(database, artifact_root)
    before = _legacy_research_hash(database)

    report = DomainMigrator(
        database,
        artifact_root=artifact_root,
    ).migrate_research("task-1")
    restored = ResearchEvidenceStore(
        database,
        artifact_root=artifact_root,
    ).get_evidence("task-1", evidence.evidence_id)

    assert report.source_count == report.target_count == 3
    assert report.source_hash == report.target_hash
    assert report.identity_mismatches == []
    assert report.hash_mismatches == []
    assert report.missing_references == []
    assert report.ready_to_switch
    assert restored == evidence
    assert _legacy_research_hash(database) == before
