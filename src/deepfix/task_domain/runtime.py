from __future__ import annotations

from typing import Any

from pydantic import Field

from deepfix.compaction.models import StrictModel
from deepfix.task_domain.models import TaskLifecycleStatus


class TaskRuntime(StrictModel):
    """Request-local service result; never a persistence authority."""

    task_id: str = Field(min_length=1)
    lifecycle: TaskLifecycleStatus
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)
    latest_decision_id: str | None = None
    pause_reason: str | None = None
