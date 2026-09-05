from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from deepfix.evaluation.legacy import _run_oracle
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
    RunUsage,
)
from deepfix.investigation.evaluation import RepairLoopOutcome
from deepfix.investigation.loop import RepairLoopResult

CaseExperimentExecutor = Callable[
    [EvaluationCase, Path, Path, EvaluationBudget],
    tuple[RepairLoopResult, RunUsage],
]
OracleRunner = Callable[[str, Path, int, Path], int]


class ExperimentLoopRunner:
    """Evaluation-harness adapter for the opt-in Experiment Loop.

    The trusted-corpus grant and concrete online composition belong to the A/B gate
    plan. Without an injected executor this runner fails closed instead of silently
    falling back to the legacy loop or bypassing approval.
    """

    budget_enforcement = "pre_call_reservation"
    model_accounting = "all_model_roles"

    def __init__(
        self,
        *,
        project_python: Path,
        case_executor: CaseExperimentExecutor | None = None,
        oracle_runner: OracleRunner | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.project_python = project_python.expanduser().resolve()
        self.case_executor = case_executor
        self.oracle_runner = oracle_runner or _run_oracle
        self.clock = clock

    def run(
        self,
        case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        started_at = self.clock()
        loop_result: RepairLoopResult | None = None
        sanitized_error_code: str | None = None
        usage = RunUsage(
            input_tokens=0,
            output_tokens=0,
            model_calls=0,
            tool_calls=0,
            wall_seconds=0,
        )
        if self.case_executor is None:
            sanitized_error_code = "trusted_evaluation_grant_required"
        else:
            try:
                loop_result, usage = self.case_executor(
                    case, workspace, run_dir, budget
                )
            except Exception:  # noqa: BLE001 - persist only a stable error code
                sanitized_error_code = "experiment_runner_failed"

        try:
            oracle_exit_code = int(
                self.oracle_runner(
                    case.required_command,
                    workspace,
                    budget.max_wall_seconds,
                    self.project_python,
                )
            )
        except Exception:  # noqa: BLE001 - persist only a stable error code
            oracle_exit_code = None
            sanitized_error_code = sanitized_error_code or "oracle_execution_failed"

        conclusion = _evaluation_conclusion(loop_result)
        elapsed = max(0.0, self.clock() - started_at)
        return EvaluationRun(
            run_id=f"{case.case_id}-{run_dir.name}",
            case_id=case.case_id,
            loop="experiment",
            task_id=(
                loop_result.task_id
                if loop_result is not None
                else f"unavailable-{case.case_id}-{run_dir.name}"
            ),
            conclusion=conclusion,
            oracle_exit_code=oracle_exit_code,
            scope_violations=[],
            usage=usage.model_copy(update={"wall_seconds": elapsed}),
            sanitized_error_code=sanitized_error_code,
        )


def _evaluation_conclusion(
    result: RepairLoopResult | None,
) -> str:
    if result is None:
        return "failed"
    return {
        RepairLoopOutcome.FIXED: "fixed",
        RepairLoopOutcome.NOT_REPRODUCED: "not_reproduced",
        RepairLoopOutcome.BLOCKED: "blocked",
        RepairLoopOutcome.REVIEW: "blocked",
        RepairLoopOutcome.CONTINUE: "failed",
    }[result.outcome]
