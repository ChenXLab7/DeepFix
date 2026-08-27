from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.compaction.models import ContextRecoveryMetadata
from deepfix.config import ApprovalMode
from deepfix.investigation.models import InvestigationRecoveryMetadata


class TaskStatus(StrEnum):
    CREATED = "created"
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    WAITING_APPROVAL = "waiting_approval"
    EDITING = "editing"
    TESTING = "testing"
    REVIEWING = "reviewing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {
            TaskStatus.CLARIFYING,
            TaskStatus.INVESTIGATING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.CLARIFYING: frozenset(
        {
            TaskStatus.INVESTIGATING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.INVESTIGATING: frozenset(
        {
            TaskStatus.CLARIFYING,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.EDITING,
            TaskStatus.TESTING,
            TaskStatus.REVIEWING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.WAITING_APPROVAL: frozenset(
        {
            TaskStatus.INVESTIGATING,
            TaskStatus.EDITING,
            TaskStatus.TESTING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.EDITING: frozenset(
        {
            TaskStatus.INVESTIGATING,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.TESTING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.TESTING: frozenset(
        {
            TaskStatus.INVESTIGATING,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.EDITING,
            TaskStatus.REVIEWING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.REVIEWING: frozenset(
        {
            TaskStatus.COMPLETED,
            TaskStatus.INVESTIGATING,
            TaskStatus.EDITING,
            TaskStatus.PAUSED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.PAUSED: frozenset(),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


@dataclass
class Evidence:
    source: str
    observation: str


@dataclass
class TestResult:
    command: str
    exit_code: int
    summary: str
    tool_call_id: str | None = None
    source_message_id: str | None = None


@dataclass
class ApprovalRecord:
    operation: str
    decision: str
    risk: str


@dataclass
class ContextMetrics:
    context_peak_tokens: int = 0
    context_overflow_count: int = 0
    active_compaction_count: int = 0
    working_memory_version: int = 0
    last_compaction_at: str | None = None
    latest_usage_ratio: float = 0.0
    latest_budget_zone: str | None = None
    normal_compaction_count: int = 0
    emergency_compaction_count: int = 0
    compaction_failure_count: int = 0
    normal_zone_passthrough_count: int = 0
    manual_compaction_error_count: int = 0
    overflow_retry_count: int = 0
    active_compaction_snapshot_version: int | None = None
    last_compaction_artifact: str | None = None
    last_compaction_error: str | None = None


class RepairOutcome(BaseModel):
    status: Literal["needs_input", "completed", "blocked"]
    resolution: Literal["fixed", "not_reproduced"] | None = None
    question: str | None = None
    diagnosis: str | None = None
    hypotheses: list[str] = Field(default_factory=list)
    repair_plan: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str
    review_summary: str | None = None
    residual_risks: list[str] = Field(default_factory=list)
    unverified_items: list[str] = Field(default_factory=list)


@dataclass
class TaskState:
    task_id: str
    project_root: str
    user_problem: str
    approval_mode: str
    source_project_root: str | None = None
    workspace_root: str | None = None
    workspace_baseline_id: str | None = None
    verification_policy_id: str | None = None
    verification_policy_version: int | None = None
    project_python: str = field(
        default_factory=lambda: str(Path(sys.executable).resolve())
    )
    status: TaskStatus = TaskStatus.CREATED
    resolution: Literal["fixed", "not_reproduced"] | None = None
    conversation: list[dict[str, str]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    diagnosis: str | None = None
    repair_plan: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    successful_changed_files: list[str] = field(default_factory=list)
    latest_change_verification: Literal[
        "not_applicable", "pending", "passed", "failed"
    ] = "not_applicable"
    test_results: list[TestResult] = field(default_factory=list)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    review: str | None = None
    final_summary: str | None = None
    pause_reason: str | None = None
    residual_risks: list[str] = field(default_factory=list)
    unverified_items: list[str] = field(default_factory=list)
    pending_question: str | None = None
    pending_actions: list[dict[str, object]] = field(default_factory=list)
    processed_tool_call_ids: list[str] = field(default_factory=list)
    paused_from: TaskStatus | None = None
    shell_calls: int = 0
    agent_invocations: int = 0
    consecutive_test_failures: int = 0
    working_memory_version: int = 0
    context_metrics: ContextMetrics = field(default_factory=ContextMetrics)
    offloaded_artifacts: list[str] = field(default_factory=list)
    external_evidence_ids: list[str] = field(default_factory=list)
    research_query_count: int = 0
    research_provider_errors: list[str] = field(default_factory=list)
    context_recovery: ContextRecoveryMetadata | None = None
    investigation_recovery: InvestigationRecoveryMetadata | None = None

    def __post_init__(self) -> None:
        self.project_root = str(Path(self.project_root).expanduser().resolve())
        self.source_project_root = str(
            Path(self.source_project_root or self.project_root).expanduser().resolve()
        )
        self.workspace_root = str(
            Path(self.workspace_root or self.project_root).expanduser().resolve()
        )

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        problem: str,
        approval_mode: ApprovalMode,
        project_python: str | Path | None = None,
        *,
        source_project_root: str | Path | None = None,
        workspace_baseline_id: str | None = None,
    ) -> TaskState:
        normalized_problem = problem.strip()
        if not normalized_problem:
            raise ValueError("问题描述不能为空")
        active_root = Path(project_root).expanduser().resolve()
        return cls(
            task_id=uuid4().hex,
            project_root=str(active_root),
            source_project_root=str(
                Path(source_project_root or active_root).expanduser().resolve()
            ),
            workspace_root=str(active_root),
            workspace_baseline_id=workspace_baseline_id,
            user_problem=normalized_problem,
            approval_mode=approval_mode.value,
            project_python=str(
                Path(project_python or sys.executable).expanduser().resolve()
            ),
        )

    def transition_to(self, next_status: TaskStatus) -> None:
        if next_status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"非法状态迁移: {self.status} -> {next_status}")
        if next_status is TaskStatus.PAUSED:
            self.paused_from = self.status
        self.status = next_status

    def resume(self) -> None:
        if self.status is not TaskStatus.PAUSED or self.paused_from is None:
            raise ValueError("只有已暂停任务才能恢复")
        self.status = self.paused_from
        self.paused_from = None
        self.pause_reason = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["paused_from"] = self.paused_from.value if self.paused_from else None
        payload["context_recovery"] = (
            self.context_recovery.model_dump(mode="json")
            if self.context_recovery
            else None
        )
        payload["investigation_recovery"] = (
            self.investigation_recovery.model_dump(mode="json")
            if self.investigation_recovery
            else None
        )
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TaskState:
        payload = dict(value)
        task_id = str(payload["task_id"])
        payload["conversation"] = [
            {
                **entry,
                "id": str(entry.get("id") or stable_conversation_message_id(
                    task_id,
                    ordinal,
                    str(entry.get("role", "user")),
                    entry.get("content", ""),
                )),
            }
            for ordinal, entry in enumerate(payload.get("conversation", []))
        ]
        payload["status"] = TaskStatus(str(payload["status"]))
        paused_from = payload.get("paused_from")
        payload["paused_from"] = TaskStatus(str(paused_from)) if paused_from else None
        payload["evidence"] = [
            item if isinstance(item, Evidence) else Evidence(**item)
            for item in payload.get("evidence", [])
        ]
        payload["test_results"] = [
            item if isinstance(item, TestResult) else TestResult(**item)
            for item in payload.get("test_results", [])
        ]
        payload["approvals"] = [
            item if isinstance(item, ApprovalRecord) else ApprovalRecord(**item)
            for item in payload.get("approvals", [])
        ]
        context_metrics = payload.get("context_metrics")
        if context_metrics is not None and not isinstance(context_metrics, ContextMetrics):
            payload["context_metrics"] = ContextMetrics(**context_metrics)
        context_recovery = payload.get("context_recovery")
        if context_recovery is not None and not isinstance(
            context_recovery, ContextRecoveryMetadata
        ):
            payload["context_recovery"] = ContextRecoveryMetadata.model_validate(
                context_recovery
            )
        investigation_recovery = payload.get("investigation_recovery")
        if investigation_recovery is not None and not isinstance(
            investigation_recovery, InvestigationRecoveryMetadata
        ):
            payload["investigation_recovery"] = (
                InvestigationRecoveryMetadata.model_validate(
                    investigation_recovery
                )
            )
        return cls(**payload)
