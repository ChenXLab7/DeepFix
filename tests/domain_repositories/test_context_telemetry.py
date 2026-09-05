from __future__ import annotations

from pathlib import Path

import pytest

from deepfix.domain_repositories.history import (
    ContextTelemetryIdentityConflict,
    HistoryRepository,
)

TASK_ID = "task-context-telemetry"


def test_budget_observation_tracks_monotonic_peak_and_latest_zone(tmp_path: Path) -> None:
    history = HistoryRepository(tmp_path / "deepfix.db")

    initial = history.context_telemetry(TASK_ID)
    first = history.record_budget_observation(
        TASK_ID,
        event_id="budget-1",
        estimated_tokens=8_200,
        usage_ratio=0.82,
        zone="compact",
    )
    second = history.record_budget_observation(
        TASK_ID,
        event_id="budget-2",
        estimated_tokens=7_500,
        usage_ratio=0.75,
        zone="observe",
    )

    assert initial.task_id == TASK_ID
    assert initial.context_peak_tokens == 0
    assert first.context_peak_tokens == 8_200
    assert second.context_peak_tokens == 8_200
    assert second.latest_usage_ratio == 0.75
    assert second.latest_budget_zone == "observe"


def test_counter_event_replay_is_idempotent_and_conflicting_replay_is_rejected(
    tmp_path: Path,
) -> None:
    history = HistoryRepository(tmp_path / "deepfix.db")

    first = history.record_overflow(TASK_ID, event_id="overflow-1")
    replay = history.record_overflow(TASK_ID, event_id="overflow-1")

    assert first.context_overflow_count == 1
    assert replay.context_overflow_count == 1
    with pytest.raises(ContextTelemetryIdentityConflict):
        history.record_overflow_retry(TASK_ID, event_id="overflow-1")


def test_compaction_telemetry_records_outcomes_failures_and_bounded_counters(
    tmp_path: Path,
) -> None:
    history = HistoryRepository(tmp_path / "deepfix.db")

    history.record_overflow_retry(TASK_ID, event_id="retry-1")
    history.record_passthrough(TASK_ID, event_id="passthrough-1")
    history.record_manual_compaction_error(TASK_ID, event_id="manual-1")
    history.record_compaction_failure(
        TASK_ID,
        event_id="failure-1",
        error_code="snapshot_validation_failed",
    )
    history.record_compaction_outcome(
        TASK_ID,
        event_id="compaction-1",
        snapshot_version=4,
        artifact_path="/.deepfix-artifacts/conversation_history/task.md",
        emergency=True,
        recorded_at="2026-08-31T00:00:00+00:00",
    )

    metrics = history.context_telemetry(TASK_ID)
    assert metrics.overflow_retry_count == 1
    assert metrics.normal_zone_passthrough_count == 1
    assert metrics.manual_compaction_error_count == 1
    assert metrics.compaction_failure_count == 1
    assert metrics.last_compaction_error == "snapshot_validation_failed"
    assert metrics.active_compaction_count == 1
    assert metrics.normal_compaction_count == 0
    assert metrics.emergency_compaction_count == 1
    assert metrics.active_snapshot_version == 4
    assert metrics.last_compaction_artifact.endswith("task.md")
    assert metrics.last_compaction_at == "2026-08-31T00:00:00+00:00"
