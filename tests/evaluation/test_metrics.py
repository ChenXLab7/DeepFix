from deepfix.evaluation.metrics import aggregate_runs, grade_run
from deepfix.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationVerdict,
    RunUsage,
)

BUGGY_CASE = EvaluationCase(
    case_id="mergesort-buggy",
    problem="repair mergesort and run the required test",
    allowed_paths=["python_programs/mergesort.py"],
    required_command="python -m pytest python_testcases/test_mergesort.py -q",
    expected_outcome="fixed",
)

CORRECT_CASE = EvaluationCase(
    case_id="mergesort-correct-control",
    problem="investigate the reported mergesort failure",
    allowed_paths=["python_programs/mergesort.py"],
    required_command="python -m pytest python_testcases/test_mergesort.py -q",
    expected_outcome="not_reproduced",
    source_variant="correct_control",
)


def make_run(
    *,
    run_id: str = "run-1",
    case_id: str = BUGGY_CASE.case_id,
    conclusion: str = "fixed",
    oracle_exit_code: int | None = 0,
    scope_violations: list[str] | None = None,
    success: bool = False,
    false_fixed: bool = False,
    input_tokens: int = 10,
    output_tokens: int = 10,
    model_calls: int = 1,
    tool_calls: int = 1,
    wall_seconds: float = 1,
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        case_id=case_id,
        loop="legacy",
        task_id=f"task-{run_id}",
        conclusion=conclusion,
        oracle_exit_code=oracle_exit_code,
        scope_violations=scope_violations or [],
        usage=RunUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_calls=model_calls,
            tool_calls=tool_calls,
            wall_seconds=wall_seconds,
        ),
        verdict=EvaluationVerdict(success=success, false_fixed=false_fixed),
    )


def test_fixed_claim_with_failing_oracle_is_false_fixed() -> None:
    verdict = grade_run(BUGGY_CASE, make_run(oracle_exit_code=1))

    assert verdict.false_fixed is True
    assert verdict.success is False


def test_fixed_claim_with_scope_violation_is_false_fixed() -> None:
    verdict = grade_run(
        BUGGY_CASE,
        make_run(scope_violations=["python_testcases/test_mergesort.py"]),
    )

    assert verdict.false_fixed is True
    assert verdict.success is False


def test_fixed_claim_for_correct_control_is_false_fixed() -> None:
    run = make_run(case_id=CORRECT_CASE.case_id)

    verdict = grade_run(CORRECT_CASE, run)

    assert verdict.false_fixed is True
    assert verdict.success is False


def test_not_reproduced_is_correct_only_for_passing_correct_control() -> None:
    run = make_run(
        case_id=CORRECT_CASE.case_id,
        conclusion="not_reproduced",
    )

    assert grade_run(CORRECT_CASE, run) == EvaluationVerdict(
        success=True,
        false_fixed=False,
    )


def test_aggregate_reports_efficiency_and_resource_totals() -> None:
    first = make_run(
        run_id="run-1",
        success=True,
        input_tokens=60_000,
        output_tokens=10_000,
        model_calls=3,
        tool_calls=8,
        wall_seconds=12,
    )
    second = make_run(
        run_id="run-2",
        success=True,
        input_tokens=20_000,
        output_tokens=10_000,
        model_calls=2,
        tool_calls=5,
        wall_seconds=8,
    )

    metrics = aggregate_runs([first, second])

    assert metrics.success_rate == 1.0
    assert metrics.successes_per_100k_tokens == 2.0
    assert metrics.median_tokens_per_success == 50_000
    assert metrics.total_input_tokens == 80_000
    assert metrics.total_output_tokens == 20_000
    assert metrics.total_model_calls == 5
    assert metrics.total_tool_calls == 13
    assert metrics.total_wall_seconds == 20


def test_aggregate_reports_not_reproduced_accuracy() -> None:
    correct = make_run(
        run_id="correct",
        case_id=CORRECT_CASE.case_id,
        conclusion="not_reproduced",
        success=True,
    )
    incorrect = make_run(
        run_id="incorrect",
        conclusion="not_reproduced",
        success=False,
    )

    metrics = aggregate_runs([correct, incorrect])

    assert metrics.not_reproduced_claim_count == 2
    assert metrics.correct_not_reproduced_count == 1
    assert metrics.not_reproduced_accuracy == 0.5
