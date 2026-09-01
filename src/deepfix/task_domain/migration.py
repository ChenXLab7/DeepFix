"""Migration-only field routing for historical task payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepfix.task_domain.legacy_payload import LegacyTaskPayload
from deepfix.task_domain.models import TaskDefinition, TaskLifecycleStatus

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


def parse_legacy_task(
    value: LegacyTaskPayload | Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> LegacyTaskPayload:
    if isinstance(value, LegacyTaskPayload):
        return value
    return LegacyTaskPayload.parse(value, created_at=created_at)


def task_definition_from_legacy(
    value: LegacyTaskPayload | Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> TaskDefinition:
    return parse_legacy_task(value, created_at=created_at).definition


def lifecycle_status_for_legacy(
    value: LegacyTaskPayload | Mapping[str, Any],
) -> TaskLifecycleStatus:
    return parse_legacy_task(value).lifecycle.status
