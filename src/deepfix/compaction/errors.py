from __future__ import annotations

from deepfix.compaction.models import CompactionFailureRecord, ContextRecoveryMetadata


class ContextCoordinationError(RuntimeError):
    def __init__(self, recovery: ContextRecoveryMetadata) -> None:
        self.recovery = recovery
        super().__init__(recovery.error_code)


class ProtectedContextLoadError(ContextCoordinationError):
    pass


class ContextRecoveryRequired(ContextCoordinationError):
    pass


class CompactionPreparationError(RuntimeError):
    def __init__(self, failure: CompactionFailureRecord) -> None:
        self.failure = failure
        super().__init__(failure.error_code)


class ArtifactPersistenceError(CompactionPreparationError):
    pass


class SnapshotBuildError(CompactionPreparationError):
    pass


class SnapshotPersistenceError(CompactionPreparationError):
    pass
