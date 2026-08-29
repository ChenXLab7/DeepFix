from __future__ import annotations

import hashlib
import json
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
from deepfix.research.models import ExternalEvidence


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
    ) -> EvidenceEnvelope:
        task_id = _required(task_id, "task_id")
        evidence_id = _required(evidence_id, "evidence_id")
        roots = _normalize_roots(provenance_root_ids)
        references = list(artifact_references)
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
                    verification_state, envelope_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
