from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from pydantic import ConfigDict, Field, JsonValue

from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    FileChangeEvidence,
    ProvenancedClaim,
    ResearchStatusEvidence,
    StrictModel,
    SystemTestEvidence,
)
from deepfix.database import SQLiteDatabase
from deepfix.research.models import ExternalEvidence, SearchCandidate


class EvidenceIdentityConflict(RuntimeError):
    """A stable Evidence identity was replayed with different content."""


class ArtifactIntegrityError(RuntimeError):
    """An Artifact reference could not be verified before persistence."""


class EvidenceKind(StrEnum):
    TEST = "test"
    FILE_CHANGE = "file_change"
    APPROVAL = "approval"
    RESEARCH_STATUS = "research_status"
    EXTERNAL_RESEARCH = "external_research"
    SEMANTIC_CLAIM = "semantic_claim"


class EvidenceAuthority(StrEnum):
    SYSTEM = "system"
    USER = "user"
    RESEARCH = "research"
    MODEL_SEMANTIC = "model_semantic"


class EvidenceVerification(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"


class EvidenceEnvelope(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    kind: EvidenceKind
    origin: str = Field(min_length=1)
    authority: EvidenceAuthority
    verification_state: EvidenceVerification
    payload_type: str = Field(min_length=1)
    payload: dict[str, JsonValue]
    provenance_root_ids: list[str] = Field(min_length=1)
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(min_length=1)


class VerificationEvidenceView(StrictModel):
    evidence_ids: list[str] = Field(default_factory=list)
    verified_evidence_ids: list[str] = Field(default_factory=list)
    contradicted_evidence_ids: list[str] = Field(default_factory=list)
    unverified_evidence_ids: list[str] = Field(default_factory=list)


class ResearchAttempt(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    sanitized_query: str = Field(min_length=1)
    providers: list[str] = Field(default_factory=list)
    provider_errors: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(min_length=1)


DeterministicEvidence: TypeAlias = (
    SystemTestEvidence | FileChangeEvidence | ApprovalEvidence | ResearchStatusEvidence
)
ArtifactVerifier: TypeAlias = Callable[[ArtifactReference], bool]

_DETERMINISTIC_PAYLOAD_TYPES: dict[str, type[DeterministicEvidence]] = {
    "SystemTestEvidence": SystemTestEvidence,
    "FileChangeEvidence": FileChangeEvidence,
    "ApprovalEvidence": ApprovalEvidence,
    "ResearchStatusEvidence": ResearchStatusEvidence,
}


class EvidenceRepository:
    """Own immutable current Evidence metadata, never acquisition behavior."""

    def __init__(
        self,
        database: SQLiteDatabase | str | Path,
        *,
        artifact_verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.database_path = self.database.path
        self.artifact_verifier = artifact_verifier
        self._initialize_schema()

    def record_deterministic(
        self,
        task_id: str,
        evidence: DeterministicEvidence,
        *,
        provenance_root_ids: list[str],
        artifact_references: list[ArtifactReference] | None = None,
    ) -> EvidenceEnvelope:
        kind, authority, origin, verification = _deterministic_policy(evidence)
        return self._record(
            evidence_id=evidence.evidence_id,
            task_id=task_id,
            kind=kind,
            origin=origin,
            authority=authority,
            verification_state=verification,
            payload_type=type(evidence).__name__,
            payload=evidence.model_dump(mode="json"),
            provenance_root_ids=provenance_root_ids,
            artifact_references=artifact_references or [],
        )

    def accept_external(
        self,
        evidence: ExternalEvidence,
        *,
        provenance_root_ids: list[str],
        artifact_references: list[ArtifactReference],
    ) -> EvidenceEnvelope:
        return self._record(
            evidence_id=evidence.evidence_id,
            task_id=evidence.task_id,
            kind=EvidenceKind.EXTERNAL_RESEARCH,
            origin=evidence.source_type,
            authority=EvidenceAuthority.RESEARCH,
            verification_state=_verification_from_text(evidence.local_verification),
            payload_type=type(evidence).__name__,
            payload=evidence.model_dump(mode="json"),
            provenance_root_ids=provenance_root_ids,
            artifact_references=artifact_references,
        )

    def accept_external_with_content(
        self,
        evidence: ExternalEvidence,
        *,
        provenance_root_ids: list[str],
        artifact_reference: ArtifactReference,
        artifact_content: str,
    ) -> EvidenceEnvelope:
        if artifact_reference.path != evidence.artifact_path:
            raise ArtifactIntegrityError("External Evidence Artifact path mismatch")
        actual_hash = hashlib.sha256(artifact_content.encode("utf-8")).hexdigest()
        if actual_hash != artifact_reference.content_hash:
            raise ArtifactIntegrityError(
                f"Artifact verification failed: {artifact_reference.path}"
            )
        return self._record(
            evidence_id=evidence.evidence_id,
            task_id=evidence.task_id,
            kind=EvidenceKind.EXTERNAL_RESEARCH,
            origin=evidence.source_type,
            authority=EvidenceAuthority.RESEARCH,
            verification_state=_verification_from_text(evidence.local_verification),
            payload_type=type(evidence).__name__,
            payload=evidence.model_dump(mode="json"),
            provenance_root_ids=provenance_root_ids,
            artifact_references=[artifact_reference],
            artifacts_preverified=True,
        )

    def update_external(self, evidence: ExternalEvidence) -> EvidenceEnvelope:
        current = self.get(evidence.task_id, evidence.evidence_id)
        if current.kind is not EvidenceKind.EXTERNAL_RESEARCH:
            raise EvidenceIdentityConflict("Only external Evidence can update verification")
        semantic = {
            "evidence_id": current.evidence_id,
            "task_id": current.task_id,
            "kind": current.kind.value,
            "origin": current.origin,
            "authority": current.authority.value,
            "verification_state": _verification_from_text(
                evidence.local_verification
            ).value,
            "payload_type": type(evidence).__name__,
            "payload": evidence.model_dump(mode="json"),
            "provenance_root_ids": current.provenance_root_ids,
            "artifact_references": [
                reference.model_dump(mode="json")
                for reference in current.artifact_references
            ],
        }
        content_hash = _canonical_hash(semantic)
        if content_hash == current.content_hash:
            return current
        updated = EvidenceEnvelope(
            **semantic,
            content_hash=content_hash,
            created_at=_now(),
        )
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT content_hash, revision FROM evidence_records
                WHERE task_id = ? AND evidence_id = ?
                """,
                (evidence.task_id, evidence.evidence_id),
            ).fetchone()
            if row is None:
                raise KeyError((evidence.task_id, evidence.evidence_id))
            if str(row[0]) != current.content_hash:
                raise EvidenceIdentityConflict("External Evidence update conflict")
            revision = int(row[1]) + 1
            connection.execute(
                """
                INSERT INTO evidence_revisions(
                    task_id, evidence_id, revision, envelope_json,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    updated.task_id,
                    updated.evidence_id,
                    revision,
                    updated.model_dump_json(),
                    updated.content_hash,
                    updated.created_at,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE evidence_records
                SET verification_state = ?, envelope_json = ?,
                    content_hash = ?, created_at = ?, revision = ?
                WHERE task_id = ? AND evidence_id = ? AND content_hash = ?
                """,
                (
                    updated.verification_state.value,
                    updated.model_dump_json(),
                    updated.content_hash,
                    updated.created_at,
                    revision,
                    updated.task_id,
                    updated.evidence_id,
                    current.content_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise EvidenceIdentityConflict("External Evidence update conflict")
        return updated

    def list_revisions(self, task_id: str, evidence_id: str) -> list[EvidenceEnvelope]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT envelope_json FROM evidence_revisions
                WHERE task_id = ? AND evidence_id = ? ORDER BY revision
                """,
                (_required(task_id, "task_id"), _required(evidence_id, "evidence_id")),
            ).fetchall()
        return [EvidenceEnvelope.model_validate_json(str(row[0])) for row in rows]

    def record_research_attempt(
        self,
        *,
        task_id: str,
        query_id: str,
        sanitized_query: str,
        providers: list[str],
        provider_errors: list[str],
        created_at: str | None = None,
    ) -> ResearchAttempt:
        attempt = ResearchAttempt(
            query_id=_required(query_id, "query_id"),
            task_id=_required(task_id, "task_id"),
            sanitized_query=_required(sanitized_query, "sanitized_query"),
            providers=_ordered_unique(providers),
            provider_errors=_ordered_unique(
                [_redact_provider_error(error) for error in provider_errors]
            ),
            candidate_ids=[],
            created_at=created_at or _now(),
        )
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload FROM research_attempts WHERE query_id = ?",
                (attempt.query_id,),
            ).fetchone()
            if row is not None:
                existing = ResearchAttempt.model_validate_json(str(row[0]))
                if existing != attempt:
                    raise EvidenceIdentityConflict("Research attempt identity conflict")
                return existing
            connection.execute(
                """
                INSERT INTO research_attempts(
                    query_id, task_id, sanitized_query, payload, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attempt.query_id,
                    attempt.task_id,
                    attempt.sanitized_query,
                    attempt.model_dump_json(),
                    attempt.created_at,
                ),
            )
        return attempt

    def record_research_candidates(
        self,
        query_id: str,
        candidates: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        query_id = _required(query_id, "query_id")
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload FROM research_attempts WHERE query_id = ?",
                (query_id,),
            ).fetchone()
            if row is None:
                raise KeyError(query_id)
            attempt = ResearchAttempt.model_validate_json(str(row[0]))
            for candidate in candidates:
                if candidate.task_id != attempt.task_id:
                    raise ValueError("Research candidate task_id mismatch")
                payload = candidate.model_dump_json()
                existing = connection.execute(
                    "SELECT payload FROM research_candidates WHERE candidate_id = ?",
                    (candidate.candidate_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != payload:
                        raise EvidenceIdentityConflict(
                            "Research candidate identity conflict"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO research_candidates(
                        candidate_id, query_id, task_id, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        query_id,
                        candidate.task_id,
                        payload,
                        candidate.created_at,
                    ),
                )
            candidate_ids = _ordered_unique(
                [*attempt.candidate_ids, *(item.candidate_id for item in candidates)]
            )
            updated = attempt.model_copy(update={"candidate_ids": candidate_ids})
            connection.execute(
                "UPDATE research_attempts SET payload = ? WHERE query_id = ?",
                (updated.model_dump_json(), query_id),
            )
        return list(candidates)

    def get_research_attempt(self, task_id: str, query_id: str) -> ResearchAttempt:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM research_attempts
                WHERE task_id = ? AND query_id = ?
                """,
                (_required(task_id, "task_id"), _required(query_id, "query_id")),
            ).fetchone()
        if row is None:
            raise KeyError(query_id)
        return ResearchAttempt.model_validate_json(str(row[0]))

    def list_research_attempts(self, task_id: str) -> list[ResearchAttempt]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM research_attempts
                WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [ResearchAttempt.model_validate_json(str(row[0])) for row in rows]

    def latest_research_attempt(
        self,
        task_id: str,
        sanitized_query: str,
    ) -> ResearchAttempt | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM research_attempts
                WHERE task_id = ? AND sanitized_query = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (_required(task_id, "task_id"), sanitized_query),
            ).fetchone()
        return None if row is None else ResearchAttempt.model_validate_json(str(row[0]))

    def get_research_candidate(self, task_id: str, candidate_id: str) -> SearchCandidate:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM research_candidates
                WHERE task_id = ? AND candidate_id = ?
                """,
                (_required(task_id, "task_id"), _required(candidate_id, "candidate_id")),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return SearchCandidate.model_validate_json(str(row[0]))

    def list_research_candidates(self, task_id: str) -> list[SearchCandidate]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM research_candidates
                WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [SearchCandidate.model_validate_json(str(row[0])) for row in rows]

    def research_summary(self, task_id: str) -> tuple[int, list[str]]:
        attempts = self.list_research_attempts(task_id)
        return (
            len(attempts),
            _ordered_unique(
                [error for attempt in attempts for error in attempt.provider_errors]
            ),
        )

    def record_semantic_candidate(
        self,
        task_id: str,
        claim: ProvenancedClaim,
        *,
        provenance_root_ids: list[str],
    ) -> EvidenceEnvelope:
        return self._record(
            evidence_id=claim.claim_id,
            task_id=task_id,
            kind=EvidenceKind.SEMANTIC_CLAIM,
            origin="model",
            authority=EvidenceAuthority.MODEL_SEMANTIC,
            verification_state=(
                EvidenceVerification.CONTRADICTED
                if claim.state == "conflict"
                else EvidenceVerification.UNVERIFIED
            ),
            payload_type=type(claim).__name__,
            payload=claim.model_dump(mode="json"),
            provenance_root_ids=provenance_root_ids,
            artifact_references=[],
        )

    def get(self, task_id: str, evidence_id: str) -> EvidenceEnvelope:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT envelope_json FROM evidence_records
                WHERE task_id = ? AND evidence_id = ?
                """,
                (_required(task_id, "task_id"), _required(evidence_id, "evidence_id")),
            ).fetchone()
        if row is None:
            raise KeyError((task_id, evidence_id))
        return EvidenceEnvelope.model_validate_json(str(row[0]))

    def list_for_task(self, task_id: str) -> list[EvidenceEnvelope]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT envelope_json FROM evidence_records
                WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [EvidenceEnvelope.model_validate_json(str(row[0])) for row in rows]

    def verification_view(self, task_id: str) -> VerificationEvidenceView:
        items = self.list_for_task(task_id)
        return VerificationEvidenceView(
            evidence_ids=[item.evidence_id for item in items],
            verified_evidence_ids=[
                item.evidence_id
                for item in items
                if item.verification_state is EvidenceVerification.VERIFIED
            ],
            contradicted_evidence_ids=[
                item.evidence_id
                for item in items
                if item.verification_state is EvidenceVerification.CONTRADICTED
            ],
            unverified_evidence_ids=[
                item.evidence_id
                for item in items
                if item.verification_state is EvidenceVerification.UNVERIFIED
            ],
        )

    def _record(
        self,
        *,
        evidence_id: str,
        task_id: str,
        kind: EvidenceKind,
        origin: str,
        authority: EvidenceAuthority,
        verification_state: EvidenceVerification,
        payload_type: str,
        payload: dict[str, JsonValue],
        provenance_root_ids: list[str],
        artifact_references: list[ArtifactReference],
        artifacts_preverified: bool = False,
    ) -> EvidenceEnvelope:
        task_id = _required(task_id, "task_id")
        evidence_id = _required(evidence_id, "evidence_id")
        roots = _normalize_roots(provenance_root_ids)
        references = list(artifact_references)
        if not artifacts_preverified:
            self._verify_artifacts(references)
        semantic = {
            "evidence_id": evidence_id,
            "task_id": task_id,
            "kind": kind.value,
            "origin": _required(origin, "origin"),
            "authority": authority.value,
            "verification_state": verification_state.value,
            "payload_type": _required(payload_type, "payload_type"),
            "payload": payload,
            "provenance_root_ids": roots,
            "artifact_references": [
                reference.model_dump(mode="json") for reference in references
            ],
        }
        content_hash = _canonical_hash(semantic)
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT envelope_json, content_hash FROM evidence_records
                WHERE task_id = ? AND evidence_id = ?
                """,
                (task_id, evidence_id),
            ).fetchone()
            if row is not None:
                if str(row[1]) != content_hash:
                    raise EvidenceIdentityConflict(
                        f"Evidence identity conflict: {task_id}/{evidence_id}"
                    )
                return EvidenceEnvelope.model_validate_json(str(row[0]))
            envelope = EvidenceEnvelope(
                **semantic,
                content_hash=content_hash,
                created_at=_now(),
            )
            serialized = envelope.model_dump_json()
            connection.execute(
                """
                INSERT INTO evidence_records(
                    task_id, evidence_id, kind, authority,
                    verification_state, envelope_json, content_hash, created_at,
                    revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    task_id,
                    evidence_id,
                    kind.value,
                    authority.value,
                    verification_state.value,
                    serialized,
                    content_hash,
                    envelope.created_at,
                ),
            )
            persisted = connection.execute(
                """
                SELECT envelope_json, content_hash FROM evidence_records
                WHERE task_id = ? AND evidence_id = ?
                """,
                (task_id, evidence_id),
            ).fetchone()
            if persisted is None or str(persisted[1]) != content_hash:
                raise EvidenceIdentityConflict("Evidence persistence verification failed")
            restored = EvidenceEnvelope.model_validate_json(str(persisted[0]))
            if restored != envelope:
                raise EvidenceIdentityConflict("Evidence persistence verification failed")
            connection.execute(
                """
                INSERT INTO evidence_revisions(
                    task_id, evidence_id, revision, envelope_json,
                    content_hash, created_at
                ) VALUES (?, ?, 1, ?, ?, ?)
                """,
                (
                    task_id,
                    evidence_id,
                    serialized,
                    content_hash,
                    envelope.created_at,
                ),
            )
            return restored

    def _verify_artifacts(self, references: list[ArtifactReference]) -> None:
        if not references:
            return
        if self.artifact_verifier is None:
            raise ArtifactIntegrityError("Artifact verifier is required")
        for reference in references:
            if not self.artifact_verifier(reference):
                raise ArtifactIntegrityError(
                    f"Artifact verification failed: {reference.path}"
                )

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_records (
                    task_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    verification_state TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(task_id, evidence_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS evidence_records_task_kind
                ON evidence_records(task_id, kind, created_at)
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(evidence_records)"
                ).fetchall()
            }
            if "revision" not in columns:
                connection.execute(
                    "ALTER TABLE evidence_records ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_revisions (
                    task_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    envelope_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, evidence_id, revision)
                );
                CREATE TABLE IF NOT EXISTS research_attempts (
                    query_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    sanitized_query TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_attempts_task_query
                ON research_attempts(task_id, sanitized_query, created_at);
                CREATE TABLE IF NOT EXISTS research_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_candidates_task
                ON research_candidates(task_id, created_at);
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_revisions(
                    task_id, evidence_id, revision, envelope_json,
                    content_hash, created_at
                )
                SELECT task_id, evidence_id, revision, envelope_json,
                       content_hash, created_at
                FROM evidence_records
                """
            )


def _deterministic_policy(
    evidence: DeterministicEvidence,
) -> tuple[EvidenceKind, EvidenceAuthority, str, EvidenceVerification]:
    if isinstance(evidence, SystemTestEvidence):
        return (
            EvidenceKind.TEST,
            EvidenceAuthority.SYSTEM,
            evidence.origin,
            EvidenceVerification.VERIFIED,
        )
    if isinstance(evidence, FileChangeEvidence):
        return (
            EvidenceKind.FILE_CHANGE,
            EvidenceAuthority.SYSTEM,
            "tool_execution",
            EvidenceVerification.VERIFIED,
        )
    if isinstance(evidence, ApprovalEvidence):
        return (
            EvidenceKind.APPROVAL,
            EvidenceAuthority.SYSTEM,
            "approval_record",
            EvidenceVerification.VERIFIED,
        )
    if isinstance(evidence, ResearchStatusEvidence):
        return (
            EvidenceKind.RESEARCH_STATUS,
            EvidenceAuthority.RESEARCH,
            "research_status",
            _verification_from_text(evidence.verification),
        )
    raise TypeError(f"Unsupported deterministic Evidence: {type(evidence).__name__}")


def deterministic_provenance_roots(evidence: DeterministicEvidence) -> list[str]:
    if isinstance(evidence, SystemTestEvidence):
        return [evidence.tool_call_id]
    if isinstance(evidence, FileChangeEvidence):
        source = evidence.tool_call_id or evidence.source_message_id
        return [source or f"file-projection:{evidence.evidence_id}"]
    if isinstance(evidence, ApprovalEvidence):
        return [f"approval:{evidence.evidence_id}"]
    if isinstance(evidence, ResearchStatusEvidence):
        return [f"research:{evidence.evidence_id}"]
    raise TypeError(f"Unsupported deterministic Evidence: {type(evidence).__name__}")


def restore_deterministic_evidence(envelope: EvidenceEnvelope) -> DeterministicEvidence:
    model = _DETERMINISTIC_PAYLOAD_TYPES.get(envelope.payload_type)
    if model is None:
        raise TypeError(f"Envelope is not deterministic Evidence: {envelope.payload_type}")
    return model.model_validate(envelope.payload)


def restore_external_evidence(envelope: EvidenceEnvelope) -> ExternalEvidence:
    if envelope.kind is not EvidenceKind.EXTERNAL_RESEARCH:
        raise TypeError(f"Envelope is not external Evidence: {envelope.payload_type}")
    return ExternalEvidence.model_validate(envelope.payload)


def _verification_from_text(value: str) -> EvidenceVerification:
    return {
        "verified": EvidenceVerification.VERIFIED,
        "contradicted": EvidenceVerification.CONTRADICTED,
        "unverified": EvidenceVerification.UNVERIFIED,
    }[value]


def _normalize_roots(values: list[str]) -> list[str]:
    roots = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not roots:
        raise ValueError("Evidence provenance_root_ids must not be empty")
    return roots


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _redact_provider_error(value: str) -> str:
    bounded = value.strip()[:500]
    bounded = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)\S+",
        r"\1[REDACTED]",
        bounded,
    )
    bounded = re.sub(
        r"(?i)((?:api[_ -]?key|token|secret)\s*[=:]\s*)\S+",
        r"\1[REDACTED]",
        bounded,
    )
    return bounded


def _canonical_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
