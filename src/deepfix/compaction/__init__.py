"""Evidence-preserving context compaction for DeepFix."""

from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    CompactionPreparationError,
    ContextCoordinationError,
    ContextRecoveryRequired,
    ProtectedContextLoadError,
    SnapshotBuildError,
    SnapshotPersistenceError,
)

__all__ = [
    "ArtifactPersistenceError",
    "CompactionPreparationError",
    "ContextCoordinationError",
    "ContextRecoveryRequired",
    "ProtectedContextLoadError",
    "SnapshotBuildError",
    "SnapshotPersistenceError",
]

