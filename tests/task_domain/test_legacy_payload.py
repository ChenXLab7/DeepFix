from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepfix.task_domain.legacy_payload import LegacyTaskPayload
from deepfix.task_domain.models import TaskLifecycleStatus

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_tasks.json"


def _payloads() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("task_id", "expected_lifecycle"),
    [
        ("legacy-created", TaskLifecycleStatus.CREATED),
        ("legacy-running", TaskLifecycleStatus.RUNNING),
        ("legacy-approval", TaskLifecycleStatus.WAITING_APPROVAL),
        ("legacy-paused", TaskLifecycleStatus.PAUSED),
        ("legacy-completed", TaskLifecycleStatus.COMPLETED),
    ],
)
def test_historical_json_projects_to_immutable_task_records(
    task_id: str,
    expected_lifecycle: TaskLifecycleStatus,
) -> None:
    raw = next(item for item in _payloads() if item["task_id"] == task_id)

    legacy = LegacyTaskPayload.parse(raw, created_at="2026-08-01T00:00:00+00:00")

    assert legacy.definition.task_id == task_id
    assert legacy.definition.original_problem == raw["user_problem"]
    assert legacy.lifecycle.status is expected_lifecycle
    assert legacy.lifecycle.task_id == task_id
    assert legacy.definition.original_message_id


def test_historical_message_ids_are_stable_and_existing_ids_are_preserved() -> None:
    running_raw = next(item for item in _payloads() if item["task_id"] == "legacy-running")
    completed_raw = next(item for item in _payloads() if item["task_id"] == "legacy-completed")

    first = LegacyTaskPayload.parse(running_raw).graph_messages
    second = LegacyTaskPayload.parse(running_raw).graph_messages
    completed = LegacyTaskPayload.parse(completed_raw).graph_messages

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len({item["id"] for item in first}) == len(first)
    assert completed[0]["id"] == "user-existing-id"


def test_historical_payload_is_one_way_and_exposes_migration_inputs_only() -> None:
    paused_raw = next(item for item in _payloads() if item["task_id"] == "legacy-paused")

    legacy = LegacyTaskPayload.parse(paused_raw)

    assert legacy.context_recovery["error_code"] == "context_overflow"
    assert legacy.investigation_recovery["error_code"] == "investigation_state_commit_failed"
    assert legacy.artifact_references == (
        "/.deepfix-artifacts/conversation_history/legacy-paused.md",
    )
    assert not hasattr(legacy, "save")
    assert not hasattr(legacy, "to_dict")


def test_completed_payload_projects_adjudication_reference_without_fact_copy() -> None:
    raw = next(item for item in _payloads() if item["task_id"] == "legacy-completed")

    legacy = LegacyTaskPayload.parse(raw, created_at="2026-08-01T00:00:00+00:00")

    assert legacy.adjudication is not None
    assert legacy.adjudication.outcome == "fixed"
    assert legacy.adjudication.evidence_ids == ["external-1"]
    assert legacy.adjudication.task_id == "legacy-completed"
