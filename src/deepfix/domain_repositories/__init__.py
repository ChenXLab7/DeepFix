"""Bounded DeepFix domain repository exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    ArtifactIntegrityError,
    EvidenceAuthority,
    EvidenceEnvelope,
    EvidenceIdentityConflict,
    EvidenceKind,
    EvidenceRepository,
    EvidenceVerification,
    VerificationEvidenceView,
)
from deepfix.domain_repositories.execution import ExecutionRepository
from deepfix.domain_repositories.history import HistoryRepository
from deepfix.domain_repositories.investigation import (
    HypothesisIdentityConflict,
    HypothesisTransitionError,
    InvestigationEvidenceMissing,
    InvestigationRepository,
    QuestionIdentityConflict,
)
from deepfix.task_domain.repository import TaskRepository


@dataclass(frozen=True)
class DomainRepositories:
    """Single production composition root for bounded persistence authorities."""

    database: SQLiteDatabase
    tasks: TaskRepository
    evidence: EvidenceRepository
    investigation: InvestigationRepository
    execution: ExecutionRepository
    history: HistoryRepository

    @classmethod
    def create(
        cls,
        database: SQLiteDatabase | str | Path,
    ) -> DomainRepositories:
        resolved = database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        return cls(
            database=resolved,
            tasks=TaskRepository(resolved),
            evidence=EvidenceRepository(resolved),
            investigation=InvestigationRepository(resolved),
            execution=ExecutionRepository(resolved),
            history=HistoryRepository(resolved),
        )


__all__ = [
    "ArtifactIntegrityError",
    "DomainRepositories",
    "EvidenceAuthority",
    "EvidenceEnvelope",
    "EvidenceIdentityConflict",
    "EvidenceKind",
    "EvidenceRepository",
    "EvidenceVerification",
    "ExecutionRepository",
    "HistoryRepository",
    "HypothesisIdentityConflict",
    "HypothesisTransitionError",
    "InvestigationEvidenceMissing",
    "InvestigationRepository",
    "QuestionIdentityConflict",
    "VerificationEvidenceView",
]
