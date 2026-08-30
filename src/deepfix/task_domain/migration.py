from __future__ import annotations

from datetime import UTC, datetime

from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.models import TaskState, TaskStatus
from deepfix.task_domain.models import (
    TaskDefinition,
    TaskLifecycle,
    TaskLifecycleStatus,
)

TASK_DEFINITION_FIELDS = frozenset(
    {
        "task_id",
        "project_root",
        "user_problem",
        "approval_mode",
        "source_project_root",
        "workspace_root",
        "workspace_baseline_id",
        "project_python",
        "confinement_level",
    }
)

TASK_LIFECYCLE_FIELDS = frozenset({"status", "paused_from", "pause_reason"})

TASK_POLICY_REFERENCE_FIELDS = frozenset({"verification_policy_id", "verification_policy_version"})

TASK_ADJUDICATION_FIELDS = frozenset({"resolution"})

PLAN_2_LEGACY_FIELDS = frozenset(
    {
        "required_oracle_count",
        "passed_required_oracle_count",
        "supplemental_failure_count",
        "unresolved_operation_ids",
        "conversation",
        "evidence",
        "hypotheses",
        "diagnosis",
        "repair_plan",
        "changed_files",
        "successful_changed_files",
        "latest_change_verification",
        "test_results",
        "approvals",
        "review",
        "final_summary",
        "residual_risks",
        "unverified_items",
        "pending_question",
        "pending_actions",
        "processed_tool_call_ids",
        "shell_calls",
        "agent_invocations",
        "consecutive_test_failures",
        "working_memory_version",
        "context_metrics",
        "offloaded_artifacts",
        "external_evidence_ids",
        "research_query_count",
        "research_provider_errors",
        "context_recovery",
        "investigation_recovery",
    }
)

MIGRATED_LEGACY_FIELDS = {
    "evidence": frozenset(
        {
            "changed_files",
            "successful_changed_files",
            "latest_change_verification",
            "test_results",
        }
    ),
    "research": frozenset(
        {
            "external_evidence_ids",
            "research_query_count",
            "research_provider_errors",
        }
    ),
    "investigation": frozenset({"hypotheses"}),
    "execution": frozenset({"approvals", "unresolved_operation_ids"}),
    "history": frozenset(),
}

LEGACY_TO_LIFECYCLE = {
    TaskStatus.CREATED: TaskLifecycleStatus.CREATED,
    TaskStatus.CLARIFYING: TaskLifecycleStatus.PAUSED,
    TaskStatus.INVESTIGATING: TaskLifecycleStatus.RUNNING,
    TaskStatus.EDITING: TaskLifecycleStatus.RUNNING,
    TaskStatus.TESTING: TaskLifecycleStatus.RUNNING,
    TaskStatus.REVIEWING: TaskLifecycleStatus.RUNNING,
    TaskStatus.WAITING_APPROVAL: TaskLifecycleStatus.WAITING_APPROVAL,
    TaskStatus.PAUSED: TaskLifecycleStatus.PAUSED,
    TaskStatus.COMPLETED: TaskLifecycleStatus.COMPLETED,
    TaskStatus.FAILED: TaskLifecycleStatus.FAILED,
    TaskStatus.CANCELLED: TaskLifecycleStatus.CANCELLED,
}


def task_definition_from_legacy(
    task: TaskState,
    *,
    created_at: str | None = None,
) -> TaskDefinition:
    original_message_id = _original_message_id(task)
    return TaskDefinition(
        task_id=task.task_id,
        original_message_id=original_message_id,
        original_problem=task.user_problem,
        approval_mode=task.approval_mode,
        source_project_root=task.source_project_root or task.project_root,
        workspace_root=task.workspace_root or task.project_root,
        workspace_baseline_id=task.workspace_baseline_id,
        project_python=task.project_python,
        confinement_level=task.confinement_level,
        created_at=created_at or _now(),
    )


def legacy_payload_from_task(
    task: TaskState,
    *,
    switched_domains: set[str] | frozenset[str] = frozenset(),
) -> dict[str, object]:
    payload = task.to_dict()
    known_fields = (
        TASK_DEFINITION_FIELDS
        | TASK_LIFECYCLE_FIELDS
        | TASK_POLICY_REFERENCE_FIELDS
        | TASK_ADJUDICATION_FIELDS
        | PLAN_2_LEGACY_FIELDS
    )
    unknown_fields = set(payload) - known_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"TaskState fields lack an authority assignment: {names}")
    omitted = frozenset().union(*(MIGRATED_LEGACY_FIELDS[domain] for domain in switched_domains))
    return {key: payload[key] for key in PLAN_2_LEGACY_FIELDS if key not in omitted}


def reconstruct_task_state(
    definition: TaskDefinition,
    lifecycle: TaskLifecycle,
    legacy_payload: dict[str, object],
    *,
    legacy_phase_status: str | None,
    legacy_paused_from: str | None,
    verification_policy_id: str | None = None,
    verification_policy_version: int | None = None,
    resolution: str | None = None,
) -> TaskState:
    payload = dict(legacy_payload)
    payload.update(
        {
            "task_id": definition.task_id,
            "project_root": definition.workspace_root,
            "user_problem": definition.original_problem,
            "approval_mode": definition.approval_mode,
            "source_project_root": definition.source_project_root,
            "workspace_root": definition.workspace_root,
            "workspace_baseline_id": definition.workspace_baseline_id,
            "project_python": definition.project_python,
            "confinement_level": definition.confinement_level,
            "status": _legacy_status(lifecycle, legacy_phase_status).value,
            "paused_from": legacy_paused_from,
            "pause_reason": lifecycle.reason,
            "verification_policy_id": verification_policy_id,
            "verification_policy_version": verification_policy_version,
            "resolution": resolution,
        }
    )
    return TaskState.from_dict(payload)


def lifecycle_status_for_legacy(task: TaskState) -> TaskLifecycleStatus:
    return LEGACY_TO_LIFECYCLE[task.status]


def _legacy_status(
    lifecycle: TaskLifecycle,
    legacy_phase_status: str | None,
) -> TaskStatus:
    if lifecycle.status is TaskLifecycleStatus.RUNNING:
        candidate = TaskStatus(legacy_phase_status or TaskStatus.INVESTIGATING.value)
        if candidate in {
            TaskStatus.INVESTIGATING,
            TaskStatus.EDITING,
            TaskStatus.TESTING,
            TaskStatus.REVIEWING,
        }:
            return candidate
        return TaskStatus.INVESTIGATING
    if (
        lifecycle.status is TaskLifecycleStatus.PAUSED
        and legacy_phase_status == TaskStatus.CLARIFYING.value
    ):
        return TaskStatus.CLARIFYING
    return TaskStatus(lifecycle.status.value)


def _original_message_id(task: TaskState) -> str:
    for ordinal, entry in enumerate(task.conversation):
        if str(entry.get("role", "")) != "user":
            continue
        existing = entry.get("id")
        if existing:
            return str(existing)
        return stable_conversation_message_id(
            task.task_id,
            ordinal,
            "user",
            entry.get("content", ""),
        )
    return stable_conversation_message_id(
        task.task_id,
        0,
        "user",
        task.user_problem,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
