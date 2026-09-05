from __future__ import annotations

from typing import Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel
from deepfix.evaluation.models import (
    EvaluationRun,
    EvaluationSummary,
    GateDefinition,
    GateThresholds,
)


class ABComparison(StrictModel):
    thresholds: GateThresholds
    paired_run_count: int = Field(ge=0)
    newly_solved_stable_failures: int = Field(ge=0)
    success_rate_delta: float
    legacy_false_fixed_per_fixed_claim: float = Field(ge=0)
    experiment_false_fixed_per_fixed_claim: float = Field(ge=0)
    legacy_false_fixed_per_all_tasks: float = Field(ge=0)
    experiment_false_fixed_per_all_tasks: float = Field(ge=0)
    false_fixed_rate_ratio_to_legacy: float = Field(ge=0)
    wrong_hypothesis_recovery_delta: float | None
    input_token_multiplier: float | None = Field(default=None, ge=0)
    output_token_multiplier: float | None = Field(default=None, ge=0)
    token_multiplier: float | None = Field(default=None, ge=0)
    model_call_multiplier: float | None = Field(default=None, ge=0)
    tool_call_multiplier: float | None = Field(default=None, ge=0)
    wall_time_multiplier: float | None = Field(default=None, ge=0)
    successes_per_100k_tokens_delta: float
    usage_estimated_runs: int = Field(ge=0)
    unknown_recovery_runs: int = Field(ge=0)
    fault_invariant_violations: int | None = Field(default=None, ge=0)
    integrity_errors: list[str] = Field(default_factory=list)


class GateDecision(StrictModel):
    status: Literal["pass", "fail", "inconclusive"]
    failed_thresholds: list[str] = Field(default_factory=list)
    inconclusive_reasons: list[str] = Field(default_factory=list)


def compare_summaries(
    gate: GateDefinition,
    legacy: EvaluationSummary,
    experiment: EvaluationSummary,
    *,
    fault_invariant_violations: int | None = None,
) -> ABComparison:
    integrity_errors = _integrity_errors(gate, legacy, experiment)
    legacy_by_case = _runs_by_case(legacy.runs)
    experiment_by_case = _runs_by_case(experiment.runs)
    newly_solved = sum(
        _success_count(legacy_by_case.get(case_id, [])) == 0
        and _success_count(experiment_by_case.get(case_id, []))
        == gate.runs_per_case
        for case_id in legacy.case_ids
    )
    legacy_recovery = _recovery_rate(legacy.runs)
    experiment_recovery = _recovery_rate(experiment.runs)
    recovery_delta = (
        experiment_recovery - legacy_recovery
        if legacy_recovery is not None and experiment_recovery is not None
        else None
    )
    legacy_tokens = (
        legacy.aggregate.total_input_tokens + legacy.aggregate.total_output_tokens
    )
    experiment_tokens = (
        experiment.aggregate.total_input_tokens
        + experiment.aggregate.total_output_tokens
    )
    false_fixed_ratio = _zero_safe_ratio(
        experiment.aggregate.false_fixed_per_all_tasks,
        legacy.aggregate.false_fixed_per_all_tasks,
    )
    return ABComparison(
        thresholds=gate.thresholds,
        paired_run_count=len(set(_run_ids(legacy)) & set(_run_ids(experiment))),
        newly_solved_stable_failures=newly_solved,
        success_rate_delta=(
            experiment.aggregate.success_rate - legacy.aggregate.success_rate
        ),
        legacy_false_fixed_per_fixed_claim=(
            legacy.aggregate.false_fixed_per_fixed_claim
        ),
        experiment_false_fixed_per_fixed_claim=(
            experiment.aggregate.false_fixed_per_fixed_claim
        ),
        legacy_false_fixed_per_all_tasks=(
            legacy.aggregate.false_fixed_per_all_tasks
        ),
        experiment_false_fixed_per_all_tasks=(
            experiment.aggregate.false_fixed_per_all_tasks
        ),
        false_fixed_rate_ratio_to_legacy=false_fixed_ratio,
        wrong_hypothesis_recovery_delta=recovery_delta,
        input_token_multiplier=_ratio(
            experiment.aggregate.total_input_tokens,
            legacy.aggregate.total_input_tokens,
        ),
        output_token_multiplier=_ratio(
            experiment.aggregate.total_output_tokens,
            legacy.aggregate.total_output_tokens,
        ),
        token_multiplier=_ratio(experiment_tokens, legacy_tokens),
        model_call_multiplier=_ratio(
            experiment.aggregate.total_model_calls,
            legacy.aggregate.total_model_calls,
        ),
        tool_call_multiplier=_ratio(
            experiment.aggregate.total_tool_calls,
            legacy.aggregate.total_tool_calls,
        ),
        wall_time_multiplier=_ratio(
            experiment.aggregate.total_wall_seconds,
            legacy.aggregate.total_wall_seconds,
        ),
        successes_per_100k_tokens_delta=(
            experiment.aggregate.successes_per_100k_tokens
            - legacy.aggregate.successes_per_100k_tokens
        ),
        usage_estimated_runs=sum(
            run.usage.usage_estimated for run in [*legacy.runs, *experiment.runs]
        ),
        unknown_recovery_runs=sum(
            run.wrong_hypothesis_recovery == "unknown"
            for run in [*legacy.runs, *experiment.runs]
        ),
        fault_invariant_violations=fault_invariant_violations,
        integrity_errors=integrity_errors,
    )


def decide_migration(comparison: ABComparison) -> GateDecision:
    inconclusive = []
    if comparison.integrity_errors:
        inconclusive.append("summary_integrity")
    if comparison.usage_estimated_runs:
        inconclusive.append("provider_usage")
    if comparison.unknown_recovery_runs or (
        comparison.wrong_hypothesis_recovery_delta is None
    ):
        inconclusive.append("wrong_hypothesis_recovery")
    if comparison.fault_invariant_violations is None:
        inconclusive.append("fault_invariants")
    ratios = {
        "token_usage": comparison.token_multiplier,
        "model_calls": comparison.model_call_multiplier,
        "tool_calls": comparison.tool_call_multiplier,
        "wall_time": comparison.wall_time_multiplier,
    }
    inconclusive.extend(name for name, value in ratios.items() if value is None)
    if inconclusive:
        return GateDecision(
            status="inconclusive",
            inconclusive_reasons=list(dict.fromkeys(inconclusive)),
        )

    thresholds = comparison.thresholds
    failed = []
    checks = {
        "minimum_newly_solved_stable_failures": (
            comparison.newly_solved_stable_failures
            >= thresholds.minimum_newly_solved_stable_failures
        ),
        "minimum_success_rate_delta": (
            comparison.success_rate_delta >= thresholds.minimum_success_rate_delta
        ),
        "maximum_false_fixed_rate": (
            comparison.experiment_false_fixed_per_all_tasks
            <= thresholds.maximum_false_fixed_rate
        ),
        "maximum_false_fixed_rate_ratio_to_legacy": (
            comparison.false_fixed_rate_ratio_to_legacy
            <= thresholds.maximum_false_fixed_rate_ratio_to_legacy
        ),
        "minimum_wrong_hypothesis_recovery_delta": (
            comparison.wrong_hypothesis_recovery_delta
            >= thresholds.minimum_wrong_hypothesis_recovery_delta
        ),
        "maximum_token_multiplier": (
            comparison.token_multiplier <= thresholds.maximum_token_multiplier
        ),
        "fault_invariant_violations": (
            comparison.fault_invariant_violations
            <= thresholds.fault_invariant_violations
        ),
    }
    failed.extend(name for name, passed in checks.items() if not passed)
    return GateDecision(
        status="fail" if failed else "pass",
        failed_thresholds=failed,
    )


def _integrity_errors(
    gate: GateDefinition,
    legacy: EvaluationSummary,
    experiment: EvaluationSummary,
) -> list[str]:
    errors = []
    if legacy.loop != "legacy" or experiment.loop != "experiment":
        errors.append("loop_identity")
    if legacy.status != "complete" or experiment.status != "complete":
        errors.append("incomplete_summary")
    if legacy.manifest_sha256 != gate.case_manifest_sha256 or (
        experiment.manifest_sha256 != gate.case_manifest_sha256
    ):
        errors.append("manifest")
    if legacy.budget != gate.budget or experiment.budget != gate.budget:
        errors.append("budget")
    if legacy.expected_runs_per_case != gate.runs_per_case or (
        experiment.expected_runs_per_case != gate.runs_per_case
    ):
        errors.append("run_count")
    if legacy.case_ids != experiment.case_ids or (
        legacy.case_expectations != experiment.case_expectations
    ):
        errors.append("case_pairing")
    if set(_run_ids(legacy)) != set(_run_ids(experiment)):
        errors.append("run_pairing")
    expected_total = len(legacy.case_ids) * gate.runs_per_case
    if len(legacy.runs) != expected_total or len(experiment.runs) != expected_total:
        errors.append("coverage")
    for summary in (legacy, experiment):
        if any(item.model_accounting != "all_model_roles" for item in summary.provenance):
            errors.append("model_accounting")
        if any(
            item.budget_enforcement != "pre_call_reservation"
            for item in summary.provenance
        ):
            errors.append("budget_enforcement")
    return list(dict.fromkeys(errors))


def _runs_by_case(runs: list[EvaluationRun]) -> dict[str, list[EvaluationRun]]:
    grouped: dict[str, list[EvaluationRun]] = {}
    for run in runs:
        grouped.setdefault(run.case_id, []).append(run)
    return grouped


def _run_ids(summary: EvaluationSummary) -> list[str]:
    return [run.run_id for run in summary.runs]


def _success_count(runs: list[EvaluationRun]) -> int:
    return sum(bool(run.verdict and run.verdict.success) for run in runs)


def _recovery_rate(runs: list[EvaluationRun]) -> float | None:
    attempted = [
        run
        for run in runs
        if run.wrong_hypothesis_recovery in {"recovered", "not_recovered"}
    ]
    if not attempted:
        return None
    return sum(
        run.wrong_hypothesis_recovery == "recovered" for run in attempted
    ) / len(attempted)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _zero_safe_ratio(numerator: float, denominator: float) -> float:
    if denominator:
        return numerator / denominator
    return 0.0 if numerator == 0 else 1e308
