from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    CompactionPreparationError,
    ContextCoordinationError,
    ContextRecoveryRequired,
)
from deepfix.compaction.models import CompactionFailureRecord, ContextRecoveryMetadata


def _failure_record():
    return CompactionFailureRecord(
        attempt_id="attempt-1",
        task_id="task-a",
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="artifact_write",
        error_code="artifact_write_failed",
        input_hash="b" * 64,
        original_messages_preserved=True,
        artifact_reference=None,
        prepared_snapshot_version=None,
        recorded_at="2026-08-22T00:00:00+00:00",
    )


def _recovery_metadata():
    return ContextRecoveryMetadata(
        task_id="task-a",
        stage="overflow_retry",
        error_code="overflow_after_retry",
        usage_ratio=0.95,
        active_snapshot_version=2,
        prepared_snapshot_version=3,
        prepared_snapshot_lifecycle="prepared",
        conversation_artifact="/.deepfix-artifacts/conversation_history/task-a.md",
        original_messages_preserved=True,
    )


def test_preparation_error_does_not_cross_service_recovery_boundary():
    error = ArtifactPersistenceError(_failure_record())

    assert isinstance(error, CompactionPreparationError)
    assert not isinstance(error, ContextCoordinationError)
    assert error.failure.stage == "artifact_write"


def test_recovery_error_carries_safe_metadata():
    error = ContextRecoveryRequired(_recovery_metadata())

    assert isinstance(error, ContextCoordinationError)
    assert error.recovery.original_messages_preserved is True
    assert str(error) == "overflow_after_retry"
