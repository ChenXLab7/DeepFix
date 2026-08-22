from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deepfix.compaction.models import WorkUnit

BudgetZone = Literal["normal", "observe", "normal_compaction", "emergency"]

_MODEL_INPUT_LIMITS = {
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}


class ContextBudgetConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ContextBudgetReport:
    request_tokens: int
    max_input_tokens: int
    output_reserve_tokens: int
    usage_ratio: float
    zone: BudgetZone
    target_ratio: float | None


@dataclass(frozen=True)
class RetentionPlan:
    retained_units: tuple[WorkUnit, ...]
    compressed_units: tuple[WorkUnit, ...]
    retained_message_ids: frozenset[str]
    estimated_ratio: float


class ContextBudgetMonitor:
    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] | None = None,
        output_reserve_tokens: int = 4096,
        model_input_limits: dict[str, int] | None = None,
    ) -> None:
        if output_reserve_tokens < 0:
            raise ValueError("output reserve 不能为负数")
        self._count = token_counter or _count_tokens_approximately
        self.output_reserve_tokens = output_reserve_tokens
        self.model_input_limits = dict(model_input_limits or _MODEL_INPUT_LIMITS)
        self._emitted_memory_hints: set[tuple[int, str]] = set()

    def measure(
        self,
        request: Any,
        protected_blocks: Sequence[Any],
    ) -> ContextBudgetReport:
        max_input_tokens = self._max_input_tokens(request.model)
        counted_values: list[str] = []
        if request.system_message is not None:
            counted_values.append(_content_text(request.system_message.content))
        counted_values.extend(_content_text(block) for block in protected_blocks)
        counted_values.extend(
            _content_text(message.content) for message in request.messages
        )
        counted_values.extend(_canonical_text(tool) for tool in request.tools)
        request_tokens = (
            sum(max(0, int(self._count(value))) for value in counted_values)
            + self.output_reserve_tokens
        )
        usage_ratio = request_tokens / max_input_tokens
        zone = classify_budget_zone(usage_ratio)
        return ContextBudgetReport(
            request_tokens=request_tokens,
            max_input_tokens=max_input_tokens,
            output_reserve_tokens=self.output_reserve_tokens,
            usage_ratio=usage_ratio,
            zone=zone,
            target_ratio=_target_ratio(zone),
        )

    def should_emit_memory_hint(
        self,
        working_memory_version: int,
        latest_work_unit_id: str,
    ) -> bool:
        key = (working_memory_version, latest_work_unit_id)
        if key in self._emitted_memory_hints:
            return False
        self._emitted_memory_hints.add(key)
        return True

    def _max_input_tokens(self, model: Any) -> int:
        profile = getattr(model, "profile", None)
        if profile:
            value = (
                profile.get("max_input_tokens")
                if isinstance(profile, dict)
                else getattr(profile, "max_input_tokens", None)
            )
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "name", None)
        )
        configured = self.model_input_limits.get(str(model_name))
        if configured is None or configured <= 0:
            raise ContextBudgetConfigurationError(
                "模型 profile 缺少 max_input_tokens，且模型名不在显式预算表中"
            )
        return configured


def classify_budget_zone(usage_ratio: float) -> BudgetZone:
    if usage_ratio < 0:
        raise ValueError("usage ratio 不能为负数")
    if usage_ratio <= 0.75:
        return "normal"
    if usage_ratio <= 0.82:
        return "observe"
    if usage_ratio <= 0.90:
        return "normal_compaction"
    return "emergency"


def select_retained_units(
    report: ContextBudgetReport,
    units: Sequence[WorkUnit],
    latest_user_message_id: str,
) -> RetentionPlan:
    ordered = tuple(units)
    if not ordered or report.target_ratio is None:
        retained_ids = frozenset(
            message_id for unit in ordered for message_id in unit.message_ids
        )
        return RetentionPlan(
            retained_units=ordered,
            compressed_units=(),
            retained_message_ids=retained_ids,
            estimated_ratio=report.usage_ratio,
        )

    total_messages = sum(len(unit.message_ids) for unit in ordered)
    unit_costs = {
        unit.unit_id: (
            report.request_tokens * len(unit.message_ids) / total_messages
        )
        for unit in ordered
    }
    target_tokens = report.max_input_tokens * report.target_ratio
    mandatory = {
        unit.unit_id
        for unit in ordered
        if unit.must_keep
        or unit.state != "complete"
        or latest_user_message_id in unit.message_ids
    }
    selected = set(mandatory)
    selected_cost = sum(unit_costs[unit_id] for unit_id in selected)

    candidates = sorted(
        (unit for unit in ordered if unit.unit_id not in selected),
        key=lambda unit: (_retention_rank(unit), -unit.end_index),
    )
    for unit in candidates:
        cost = unit_costs[unit.unit_id]
        if selected_cost + cost <= target_tokens:
            selected.add(unit.unit_id)
            selected_cost += cost

    retained = tuple(unit for unit in ordered if unit.unit_id in selected)
    compressed = tuple(unit for unit in ordered if unit.unit_id not in selected)
    retained_message_ids = frozenset(
        message_id for unit in retained for message_id in unit.message_ids
    )
    return RetentionPlan(
        retained_units=retained,
        compressed_units=compressed,
        retained_message_ids=retained_message_ids,
        estimated_ratio=selected_cost / report.max_input_tokens,
    )


def _retention_rank(unit: WorkUnit) -> int:
    if "modify" in unit.categories and unit.categories & {
        "verify_pass",
        "verify_fail",
    }:
        return 0
    if "verify_fail" in unit.categories:
        return 1
    return 2


def _target_ratio(zone: BudgetZone) -> float | None:
    if zone == "normal_compaction":
        return 0.75
    if zone == "emergency":
        return 0.65
    return None


def _content_text(value: Any) -> str:
    return value if isinstance(value, str) else _canonical_text(value)


def _canonical_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
    except (TypeError, ValueError):
        return repr(value)


def _count_tokens_approximately(value: str) -> int:
    if not value:
        return 0
    return max(1, math.ceil(len(value) / 4))
