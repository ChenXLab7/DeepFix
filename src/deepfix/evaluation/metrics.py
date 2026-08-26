from __future__ import annotations

import statistics

from deepfix.evaluation.models import (
    AggregateMetrics,
    EvaluationCase,
    EvaluationRun,
    EvaluationVerdict,
)


def grade_run(case: EvaluationCase, run: EvaluationRun) -> EvaluationVerdict:
    return grade_run_for_expected_outcome(case.expected_outcome, run)


def grade_run_for_expected_outcome(
    expected_outcome: str,
    run: EvaluationRun,
) -> EvaluationVerdict:
    false_fixed = run.conclusion == "fixed" and (
        expected_outcome != "fixed"
        or run.oracle_exit_code != 0
        or bool(run.scope_violations)
    )
    success = (
        not false_fixed
        and run.oracle_exit_code == 0
        and not run.scope_violations
        and run.conclusion == expected_outcome
    )
    return EvaluationVerdict(success=success, false_fixed=false_fixed)


def aggregate_runs(runs: list[EvaluationRun]) -> AggregateMetrics:
    total_input_tokens = sum(run.usage.input_tokens for run in runs)
    total_output_tokens = sum(run.usage.output_tokens for run in runs)
    total_tokens = total_input_tokens + total_output_tokens
    success_count = sum(bool(run.verdict and run.verdict.success) for run in runs)
    false_fixed_count = sum(
        bool(run.verdict and run.verdict.false_fixed) for run in runs
    )
    fixed_claim_count = sum(run.conclusion == "fixed" for run in runs)
    not_reproduced_claim_count = sum(
        run.conclusion == "not_reproduced" for run in runs
    )
    correct_not_reproduced_count = sum(
        run.conclusion == "not_reproduced"
        and bool(run.verdict and run.verdict.success)
        for run in runs
    )
    successful_token_counts = [
        run.usage.input_tokens + run.usage.output_tokens
        for run in runs
        if run.verdict and run.verdict.success
    ]

    return AggregateMetrics(
        run_count=len(runs),
        success_count=success_count,
        success_rate=(success_count / len(runs)) if runs else 0.0,
        false_fixed_count=false_fixed_count,
        false_fixed_per_fixed_claim=(
            false_fixed_count / fixed_claim_count if fixed_claim_count else 0.0
        ),
        false_fixed_per_all_tasks=(
            false_fixed_count / len(runs) if runs else 0.0
        ),
        not_reproduced_claim_count=not_reproduced_claim_count,
        correct_not_reproduced_count=correct_not_reproduced_count,
        not_reproduced_accuracy=(
            correct_not_reproduced_count / not_reproduced_claim_count
            if not_reproduced_claim_count
            else 0.0
        ),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_model_calls=sum(run.usage.model_calls for run in runs),
        total_tool_calls=sum(run.usage.tool_calls for run in runs),
        total_wall_seconds=sum(run.usage.wall_seconds for run in runs),
        successes_per_100k_tokens=(
            success_count * 100_000 / total_tokens if total_tokens else 0.0
        ),
        median_tokens_per_success=(
            statistics.median(successful_token_counts)
            if successful_token_counts
            else 0.0
        ),
    )
