from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from deepfix.compaction.models import ArtifactReference
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    ArtifactVerifier,
    EvidenceKind,
    EvidenceRepository,
    restore_external_evidence,
)
from deepfix.domain_repositories.migration import domain_is_switched
from deepfix.models import Evidence
from deepfix.research.models import (
    ExternalEvidence,
    ResearchQuery,
    SearchCandidate,
    VerificationStatus,
)


class ResearchEvidenceStore:
    """Compatibility facade over EvidenceRepository research persistence."""

    def __init__(
        self,
        database_path: SQLiteDatabase | str | Path,
        *,
        artifact_verifier: ArtifactVerifier | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.database = (
            database_path
            if isinstance(database_path, SQLiteDatabase)
            else SQLiteDatabase(database_path)
        )
        self.database_path = self.database.path
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root is not None
            else None
        )
        verifier = artifact_verifier or (
            _artifact_verifier(self.artifact_root)
            if self.artifact_root is not None
            else None
        )
        self.evidence_repository = EvidenceRepository(
            self.database,
            artifact_verifier=verifier,
        )
        self._initialize_legacy_tables()

    def save_query(
        self,
        task_id: str,
        sanitized_query: str,
        providers: list[str],
        errors: list[str],
    ) -> ResearchQuery:
        query = ResearchQuery(
            query_id=uuid4().hex,
            task_id=task_id,
            sanitized_query=sanitized_query,
            providers=providers,
            provider_errors=errors,
            created_at=_utc_now(),
        )
        attempt = self.evidence_repository.record_research_attempt(
            task_id=query.task_id,
            query_id=query.query_id,
            sanitized_query=query.sanitized_query,
            providers=query.providers,
            provider_errors=query.provider_errors,
            created_at=query.created_at,
        )
        return ResearchQuery(
            query_id=attempt.query_id,
            task_id=attempt.task_id,
            sanitized_query=attempt.sanitized_query,
            providers=attempt.providers,
            provider_errors=attempt.provider_errors,
            created_at=attempt.created_at,
        )

    def save_candidates(
        self,
        task_id: str,
        query: str,
        candidates: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        attempt = self.evidence_repository.latest_research_attempt(task_id, query)
        if attempt is None:
            generated = self.save_query(task_id, query, [], [])
            attempt = self.evidence_repository.get_research_attempt(
                task_id, generated.query_id
            )
        created_at = _utc_now()
        saved = [
            candidate.model_copy(
                update={
                    "candidate_id": uuid4().hex,
                    "task_id": task_id,
                    "query": query,
                    "created_at": created_at,
                }
            )
            for candidate in candidates
        ]
        return self.evidence_repository.record_research_candidates(
            attempt.query_id,
            saved,
        )

    def get_candidate(self, task_id: str, candidate_id: str) -> SearchCandidate:
        try:
            return self.evidence_repository.get_research_candidate(task_id, candidate_id)
        except KeyError:
            if domain_is_switched(self.database, "research", task_id):
                raise
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM search_candidates
                WHERE task_id = ? AND candidate_id = ?
                """,
                (task_id, candidate_id),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return SearchCandidate.model_validate_json(str(row[0]))

    def save_evidence(
        self,
        evidence: ExternalEvidence,
        *,
        artifact_reference: ArtifactReference | None = None,
        artifact_content: str | None = None,
    ) -> None:
        _validate_uuid4(evidence.evidence_id, "evidence_id")
        self.get_candidate(evidence.task_id, evidence.candidate_id)
        roots = [f"url:{evidence.url}"]
        if artifact_reference is not None and artifact_content is not None:
            self.evidence_repository.accept_external_with_content(
                evidence,
                provenance_root_ids=roots,
                artifact_reference=artifact_reference,
                artifact_content=artifact_content,
            )
            return
        self.evidence_repository.accept_external(
            evidence,
            provenance_root_ids=roots,
            artifact_references=(
                [artifact_reference] if artifact_reference is not None else []
            ),
        )

    def get_evidence(self, task_id: str, evidence_id: str) -> ExternalEvidence:
        try:
            envelope = self.evidence_repository.get(task_id, evidence_id)
            return restore_external_evidence(envelope)
        except (KeyError, TypeError):
            if domain_is_switched(self.database, "research", task_id):
                raise KeyError(evidence_id) from None
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM external_evidence
                WHERE task_id = ? AND evidence_id = ?
                """,
                (task_id, evidence_id),
            ).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        return ExternalEvidence.model_validate_json(str(row[0]))

    def list_evidence(self, task_id: str) -> list[ExternalEvidence]:
        current = {
            item.evidence_id: restore_external_evidence(item)
            for item in self.evidence_repository.list_for_task(task_id)
            if item.kind is EvidenceKind.EXTERNAL_RESEARCH
        }
        if domain_is_switched(self.database, "research", task_id):
            return list(current.values())
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM external_evidence
                WHERE task_id = ? ORDER BY updated_at, rowid
                """,
                (task_id,),
            ).fetchall()
        legacy = [ExternalEvidence.model_validate_json(str(row[0])) for row in rows]
        merged = {item.evidence_id: item for item in legacy}
        merged.update(current)
        return list(merged.values())

    def update_verification(
        self,
        task_id: str,
        evidence_id: str,
        *,
        local_verification: VerificationStatus,
        local_evidence: list[Evidence],
        linked_test_tool_call_ids: list[str],
        verification_explanation: str | None,
    ) -> ExternalEvidence:
        current = self.get_evidence(task_id, evidence_id)
        payload = current.model_dump(mode="python")
        payload.update(
            {
                "local_verification": local_verification,
                "local_evidence": local_evidence,
                "linked_test_tool_call_ids": linked_test_tool_call_ids,
                "verification_explanation": verification_explanation,
            }
        )
        updated = ExternalEvidence.model_validate(payload)
        self.evidence_repository.update_external(updated)
        return updated

    def query_summary(self, task_id: str) -> tuple[int, list[str]]:
        current_count, current_errors = self.evidence_repository.research_summary(task_id)
        if domain_is_switched(self.database, "research", task_id):
            return current_count, current_errors
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM research_queries
                WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        legacy = [ResearchQuery.model_validate_json(str(row[0])) for row in rows]
        errors = list(
            dict.fromkeys(
                [
                    *current_errors,
                    *(error for query in legacy for error in query.provider_errors),
                ]
            )
        )
        return current_count + len(legacy), errors

    def _initialize_legacy_tables(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_queries (
                    task_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, query_id)
                );
                CREATE TABLE IF NOT EXISTS search_candidates (
                    task_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, candidate_id)
                );
                CREATE TABLE IF NOT EXISTS external_evidence (
                    task_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, evidence_id)
                );
                """
            )


def _artifact_verifier(root: Path) -> ArtifactVerifier:
    def verify(reference: ArtifactReference) -> bool:
        prefix = "/.deepfix-artifacts/"
        if not reference.path.startswith(prefix):
            return False
        relative = Path(reference.path.removeprefix(prefix))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        return hashlib.sha256(candidate.read_bytes()).hexdigest() == reference.content_hash

    return verify


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _validate_uuid4(value: str, field_name: Literal["evidence_id"]) -> None:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须是 UUID4") from exc
    if parsed.version != 4:
        raise ValueError(f"{field_name} 必须是 UUID4")
