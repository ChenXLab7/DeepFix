"""One-way parser for historical giant task JSON payloads.

This module is migration-only. Runtime composition must consume the bounded
records returned here and must never expose or persist ``LegacyTaskPayload``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.task_domain.models import (
    AdjudicationDecision,
    TaskDefinition,
    TaskLifecycle,
    TaskLifecycleStatus,
)

_LIFECYCLE_BY_LEGACY_STATUS: dict[str, TaskLifecycleStatus] = {
    "created": TaskLifecycleStatus.CREATED,
    "clarifying": TaskLifecycleStatus.PAUSED,
    "investigating": TaskLifecycleStatus.RUNNING,
    "editing": TaskLifecycleStatus.RUNNING,
    "testing": TaskLifecycleStatus.RUNNING,
    "reviewing": TaskLifecycleStatus.RUNNING,
    "waiting_approval": TaskLifecycleStatus.WAITING_APPROVAL,
    "paused": TaskLifecycleStatus.PAUSED,
    "completed": TaskLifecycleStatus.COMPLETED,
    "failed": TaskLifecycleStatus.FAILED,
    "cancelled": TaskLifecycleStatus.CANCELLED,
}


@dataclass(frozen=True)
class LegacyTaskPayload:
    """Parsed migration inputs for the final bounded authorities."""

    definition: TaskDefinition
    lifecycle: TaskLifecycle
    adjudication: AdjudicationDecision | None
    graph_messages: tuple[dict[str, Any], ...]
    legacy_fields: Mapping[str, Any]
    context_recovery: Mapping[str, Any]
    investigation_recovery: Mapping[str, Any]
    artifact_references: tuple[str, ...]

    @classmethod
    def parse(
        cls,
        value: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> LegacyTaskPayload:
        payload = dict(value)
        task_id = _required(payload.get("task_id"), "task_id")
        problem = _required(payload.get("user_problem"), "user_problem")
        timestamp = created_at or _now()
        messages = tuple(_messages(task_id, payload.get("conversation", [])))
        original_message_id = next(
            (
                str(item["id"])
                for item in messages
                if str(item.get("role", "")) == "user"
            ),
            stable_conversation_message_id(task_id, 0, "user", problem),
        )
        project_root = _required(payload.get("project_root"), "project_root")
        definition = TaskDefinition(
            task_id=task_id,
            original_message_id=original_message_id,
            original_problem=problem,
            approval_mode=_required(payload.get("approval_mode"), "approval_mode"),
            source_project_root=str(payload.get("source_project_root") or project_root),
            workspace_root=str(payload.get("workspace_root") or project_root),
            workspace_baseline_id=_optional_text(payload.get("workspace_baseline_id")),
            project_python=_required(payload.get("project_python"), "project_python"),
            confinement_level=str(payload.get("confinement_level") or "legacy_local"),
            created_at=timestamp,
        )
        status_text = str(payload.get("status") or "created")
        try:
            status = _LIFECYCLE_BY_LEGACY_STATUS[status_text]
        except KeyError as exc:
            raise ValueError(f"unknown legacy task status: {status_text}") from exc
        paused_from = _paused_from(payload.get("paused_from"))
        lifecycle = TaskLifecycle(
            task_id=task_id,
            status=status,
            version=1,
            paused_from=paused_from,
            reason=_optional_text(payload.get("pause_reason")),
            updated_at=timestamp,
        )
        adjudication = _adjudication(payload, lifecycle, timestamp)
        return cls(
            definition=definition,
            lifecycle=lifecycle,
            adjudication=adjudication,
            graph_messages=messages,
            legacy_fields=payload,
            context_recovery=dict(payload.get("context_recovery") or {}),
            investigation_recovery=dict(payload.get("investigation_recovery") or {}),
            artifact_references=tuple(
                str(item) for item in payload.get("offloaded_artifacts", [])
            ),
        )

    @classmethod
    def parse_json(
        cls,
        value: str,
        *,
        created_at: str | None = None,
    ) -> LegacyTaskPayload:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("legacy task payload must be a JSON object")
        return cls.parse(decoded, created_at=created_at)


def _messages(task_id: str, raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise TypeError("legacy conversation must be a list")
    messages: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError("legacy conversation entries must be objects")
        item = dict(entry)
        item["id"] = str(
            item.get("id")
            or stable_conversation_message_id(
                task_id,
                ordinal,
                str(item.get("role") or "user"),
                item.get("content", ""),
            )
        )
        messages.append(item)
    return messages


def _paused_from(value: object) -> TaskLifecycleStatus | None:
    if value is None:
        return None
    return _LIFECYCLE_BY_LEGACY_STATUS.get(str(value), TaskLifecycleStatus.RUNNING)


def _adjudication(
    payload: Mapping[str, Any],
    lifecycle: TaskLifecycle,
    decided_at: str,
) -> AdjudicationDecision | None:
    resolution = payload.get("resolution")
    if resolution not in {"fixed", "not_reproduced"}:
        return None
    task_id = lifecycle.task_id
    material = f"{task_id}\0{lifecycle.version}\0{resolution}"
    return AdjudicationDecision(
        decision_id="legacy_decision_"
        + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32],
        task_id=task_id,
        outcome=resolution,
        evidence_ids=sorted(
            {str(item) for item in payload.get("external_evidence_ids", [])}
        ),
        operation_ids=[],
        decided_at=decided_at,
    )


def _required(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _optional_text(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
