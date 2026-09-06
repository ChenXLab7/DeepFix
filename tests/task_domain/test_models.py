from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepfix.task_domain.models import (
    AdjudicationDecision,
    TaskDefinition,
    TaskLifecycleConflict,
    TaskLifecycleStatus,
    validate_lifecycle_transition,
)


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="task-1",
        original_message_id="message-1",
        original_problem="修复排序错误",
        approval_mode="manual",
        source_project_root="C:/repo",
        workspace_root="C:/workspaces/task-1",
        workspace_baseline_id="baseline-1",
        project_python="C:/Python/python.exe",
        confinement_level="guarded_local",
        created_at="2026-08-29T00:00:00+00:00",
    )


def test_task_definition_is_frozen_and_tracks_original_message() -> None:
    definition = _definition()

    with pytest.raises(ValidationError, match="frozen"):
        definition.original_problem = "被模型改写"

    assert definition.original_problem == "修复排序错误"
    assert definition.original_message_id == "message-1"


def test_task_definition_rejects_unknown_fact_fields() -> None:
    payload = _definition().model_dump()
    payload["evidence"] = ["not-task-owned"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskDefinition.model_validate(payload)


def test_lifecycle_excludes_agent_phases() -> None:
    values = {item.value for item in TaskLifecycleStatus}

    assert values == {
        "created",
        "running",
        "waiting_approval",
        "waiting_input",
        "paused",
        "completed",
        "failed",
        "cancelled",
    }
    assert "investigating" not in values
    assert "editing" not in values
    assert "testing" not in values


@pytest.mark.parametrize(
    ("current", "next_status"),
    [
        (TaskLifecycleStatus.CREATED, TaskLifecycleStatus.RUNNING),
        (TaskLifecycleStatus.RUNNING, TaskLifecycleStatus.WAITING_APPROVAL),
        (TaskLifecycleStatus.WAITING_APPROVAL, TaskLifecycleStatus.RUNNING),
        (TaskLifecycleStatus.RUNNING, TaskLifecycleStatus.PAUSED),
        (TaskLifecycleStatus.PAUSED, TaskLifecycleStatus.RUNNING),
        (TaskLifecycleStatus.PAUSED, TaskLifecycleStatus.WAITING_APPROVAL),
        (TaskLifecycleStatus.RUNNING, TaskLifecycleStatus.COMPLETED),
    ],
)
def test_lifecycle_accepts_business_transitions(current, next_status) -> None:
    validate_lifecycle_transition(current, next_status)


@pytest.mark.parametrize(
    "terminal",
    [
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.FAILED,
        TaskLifecycleStatus.CANCELLED,
    ],
)
def test_terminal_lifecycle_rejects_outgoing_transition(terminal) -> None:
    with pytest.raises(TaskLifecycleConflict, match="illegal task lifecycle transition"):
        validate_lifecycle_transition(terminal, TaskLifecycleStatus.RUNNING)


def test_adjudication_stores_only_supporting_ids_not_evidence_bodies() -> None:
    decision = AdjudicationDecision(
        decision_id="decision-1",
        task_id="task-1",
        outcome="fixed",
        evidence_ids=["evidence-1"],
        operation_ids=["operation-1"],
        decided_at="2026-08-29T00:00:00+00:00",
    )

    assert decision.evidence_ids == ["evidence-1"]
    assert decision.operation_ids == ["operation-1"]
    assert not hasattr(decision, "evidence")
    assert not hasattr(decision, "test_results")
