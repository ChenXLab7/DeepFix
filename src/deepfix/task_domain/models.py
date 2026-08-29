from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field

from deepfix.compaction.models import StrictModel


class TaskDefinitionConflict(RuntimeError):
    pass


class TaskLifecycleConflict(RuntimeError):
    pass


class AdjudicationDecisionConflict(RuntimeError):
    pass


class TaskLifecycleStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_LIFECYCLE_TRANSITIONS: dict[
    TaskLifecycleStatus,
    frozenset[TaskLifecycleStatus],
] = {
    TaskLifecycleStatus.CREATED: frozenset(
        {
            TaskLifecycleStatus.RUNNING,
            TaskLifecycleStatus.PAUSED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
    TaskLifecycleStatus.RUNNING: frozenset(
        {
            TaskLifecycleStatus.WAITING_APPROVAL,
            TaskLifecycleStatus.PAUSED,
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
    TaskLifecycleStatus.WAITING_APPROVAL: frozenset(
        {
            TaskLifecycleStatus.RUNNING,
            TaskLifecycleStatus.PAUSED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
    TaskLifecycleStatus.PAUSED: frozenset(
        {
            TaskLifecycleStatus.RUNNING,
            TaskLifecycleStatus.WAITING_APPROVAL,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
    TaskLifecycleStatus.COMPLETED: frozenset(),
    TaskLifecycleStatus.FAILED: frozenset(),
    TaskLifecycleStatus.CANCELLED: frozenset(),
}


def validate_lifecycle_transition(
    current: TaskLifecycleStatus,
    next_status: TaskLifecycleStatus,
) -> None:
    if next_status not in _ALLOWED_LIFECYCLE_TRANSITIONS[current]:
        raise TaskLifecycleConflict(
            f"illegal task lifecycle transition: {current} -> {next_status}"
        )


class TaskDefinition(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    original_message_id: str = Field(min_length=1)
    original_problem: str = Field(min_length=1)
    approval_mode: str = Field(min_length=1)
    source_project_root: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    workspace_baseline_id: str | None = None
    project_python: str = Field(min_length=1)
    confinement_level: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class TaskLifecycle(StrictModel):
    task_id: str = Field(min_length=1)
    status: TaskLifecycleStatus
    version: int = Field(ge=1)
    paused_from: TaskLifecycleStatus | None = None
    reason: str | None = None
    updated_at: str = Field(min_length=1)


class AdjudicationDecision(StrictModel):
    decision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    outcome: Literal[
        "fixed",
        "not_reproduced",
        "paused",
        "failed",
        "cancelled",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    decided_at: str = Field(min_length=1)
