from __future__ import annotations

import pytest

from deepfix.evaluation.compare import (
    ABComparison,
    compare_summaries,
    decide_migration,
)
from deepfix.evaluation.metrics import aggregate_runs
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationProvenanceBatch,
    EvaluationRun,
    EvaluationSummary,
    EvaluationVerdict,
    GateDefinition,
    GateThresholds,
    RunUsage,
)


def _thresholds() -> GateThresholds:
    return GateThresholds(
        minimum_newly_solved_stable_failures=1,
        minimum_success_rate_delta=0.15,
        maximum_false_fixed_rate=0.05,
        maximum_false_fixed_rate_ratio_to_legacy=0.5,
        minimum_wrong_hypothesis_recovery_delta=0.15,
        maximum_token_multiplier=1.5,
        fault_invariant_violations=0,
    )


def _valid_comparison(**updates) -> ABComparison:
    values = {
        "thresholds": _thresholds(),
        "paired_run_count": 6,
        "newly_solved_stable_failures": 1,
        "success_rate_delta": 0.34,
        "legacy_false_fixed_per_fixed_claim": 0.2,
        "experiment_false_fixed_per_fixed_claim": 0.0,
        "legacy_false_fixed_per_all_tasks": 0.1,
        "experiment_false_fixed_per_all_tasks": 0.0,
        "false_fixed_rate_ratio_to_legacy": 0.0,
        "wrong_hypothesis_recovery_delta": 0.5,
        "token_multiplier": 1.2,
        "model_call_multiplier": 1.1,
        "tool_call_multiplier": 0.9,
        "wall_time_multiplier": 1.1,
        "successes_per_100k_tokens_delta": 0.4,
        "usage_estimated_runs": 0,
        "unknown_recovery_runs": 0,
        "fault_invariant_violations": 0,
        "integrity_errors": [],
    }
    values.update(updates)
    return ABComparison(**values)


@pytest.mark.parametrize(
    ("updates", "failed_threshold"),
    [
        ({"newly_solved_stable_failures": 0}, "minimum_newly_solved_stable_failures"),
        ({"success_rate_delta": 0.14}, "minimum_success_rate_delta"),
        ({"experiment_false_fixed_per_all_tasks": 0.06}, "maximum_false_fixed_rate"),
        (
            {"false_fixed_rate_ratio_to_legacy": 0.51},
            "maximum_false_fixed_rate_ratio_to_legacy",
        ),
        (
            {"wrong_hypothesis_recovery_delta": 0.14},
            "minimum_wrong_hypothesis_recovery_delta",
        ),
        ({"token_multiplier": 1.51}, "maximum_token_multiplier"),
        ({"fault_invariant_violations": 1}, "fault_invariant_violations"),
    ],
)
def test_one_failed_threshold_fails_entire_gate(updates, failed_threshold) -> None:
    comparison = _valid_comparison(**updates)

    decision = decide_migration(comparison)

    assert decision.status == "fail"
    assert failed_threshold in decision.failed_thresholds


def test_missing_provider_usage_is_inconclusive() -> None:
    decision = decide_migration(_valid_comparison(usage_estimated_runs=1))

    assert decision.status == "inconclusive"
    assert "provider_usage" in decision.inconclusive_reasons


def test_missing_fault_gate_result_is_inconclusive() -> None:
    decision = decide_migration(
        _valid_comparison(fault_invariant_violations=None)
    )

    assert decision.status == "inconclusive"
    assert "fault_invariants" in decision.inconclusive_reasons


def _run(
    run_id: str,
    loop: str,
    *,
    success: bool,
    recovery: str,
    tokens: int,
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        case_id="case-1",
        loop=loop,
        task_id=f"task-{loop}-{run_id}",
        conclusion="fixed" if success else "failed",
        oracle_exit_code=0 if success else 1,
        scope_violations=[],
        usage=RunUsage(
            input_tokens=tokens,
            output_tokens=0,
            model_calls=1,
            tool_calls=1,
            wall_seconds=1,
        ),
        verdict=EvaluationVerdict(success=success, false_fixed=False),
        wrong_hypothesis_recovery=recovery,
    )


def _summary(loop: str, runs: list[EvaluationRun]) -> EvaluationSummary:
    return EvaluationSummary(
        loop=loop,
        status="complete",
        manifest_sha256="a" * 64,
        budget=EvaluationBudget(
            max_input_tokens=100,
            max_output_tokens=100,
            max_wall_seconds=10,
            max_tool_calls=10,
            max_side_effects=2,
        ),
        case_ids=["case-1"],
        case_expectations={"case-1": "fixed"},
        expected_runs_per_case=3,
        runs=runs,
        provenance=[
            EvaluationProvenanceBatch(
                run_ids=[run.run_id for run in runs],
                capture_timing="runtime",
                source_repository="test-source",
                source_revision="source-revision",
                source_tree_sha256="c" * 64,
                source_dirty=False,
                runner_revision="d" * 40,
                runner_dirty=False,
                main_model_name="test-main",
                compaction_model_name="test-compaction",
                endpoint_fingerprint="e" * 64,
                python_version="3.12",
                python_executable_sha256="f" * 64,
                budget_enforcement="pre_call_reservation",
                model_accounting="all_model_roles",
            )
        ],
        aggregate=aggregate_runs(runs),
    )


def test_compare_summaries_calculates_stable_new_solution_and_recovery_delta() -> None:
    legacy_runs = [
        _run(f"case-1-{index:03d}", "legacy", success=False, recovery="not_recovered", tokens=10)
        for index in range(1, 4)
    ]
    experiment_runs = [
        _run(f"case-1-{index:03d}", "experiment", success=True, recovery="recovered", tokens=12)
        for index in range(1, 4)
    ]
    gate = GateDefinition(
        case_manifest_sha256="a" * 64,
        historical_legacy_summary_sha256="b" * 64,
        runs_per_case=3,
        budget=_summary("legacy", legacy_runs).budget,
        thresholds=_thresholds(),
    )

    comparison = compare_summaries(
        gate,
        _summary("legacy", legacy_runs),
        _summary("experiment", experiment_runs),
        fault_invariant_violations=0,
    )

    assert comparison.newly_solved_stable_failures == 1
    assert comparison.success_rate_delta == 1.0
    assert comparison.wrong_hypothesis_recovery_delta == 1.0
    assert comparison.token_multiplier == 1.2


def test_missing_paired_run_makes_decision_inconclusive() -> None:
    legacy_runs = [
        _run(f"case-1-{index:03d}", "legacy", success=False, recovery="not_recovered", tokens=10)
        for index in range(1, 4)
    ]
    experiment_runs = [
        _run(f"case-1-{index:03d}", "experiment", success=True, recovery="recovered", tokens=12)
        for index in range(1, 3)
    ]
    gate = GateDefinition(
        case_manifest_sha256="a" * 64,
        historical_legacy_summary_sha256="b" * 64,
        runs_per_case=3,
        budget=_summary("legacy", legacy_runs).budget,
        thresholds=_thresholds(),
    )
    comparison = compare_summaries(
        gate,
        _summary("legacy", legacy_runs),
        _summary("experiment", experiment_runs),
        fault_invariant_violations=0,
    )

    decision = decide_migration(comparison)

    assert decision.status == "inconclusive"
    assert {"run_pairing", "coverage"} <= set(comparison.integrity_errors)
