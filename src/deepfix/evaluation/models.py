from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from deepfix.compaction.models import StrictModel


class EvaluationBudget(StrictModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_wall_seconds: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_side_effects: int = Field(ge=0)


class EvaluationCase(StrictModel):
    case_id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    problem: str = Field(min_length=1)
    allowed_paths: list[str] = Field(min_length=1)
    required_command: str = Field(min_length=1)
    expected_outcome: Literal["fixed", "not_reproduced"]
    source_variant: Literal["buggy", "correct_control"] = "buggy"
    source_subdir: str = "."


class EvaluationManifest(StrictModel):
    schema_version: int = 1
    budget: EvaluationBudget
    cases: list[EvaluationCase] = Field(min_length=1)


class RunUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    usage_estimated: bool = False


class EvaluationVerdict(StrictModel):
    success: bool
    false_fixed: bool


class EvaluationRun(StrictModel):
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    loop: Literal["legacy", "experiment"]
    task_id: str = Field(min_length=1)
    conclusion: Literal["fixed", "not_reproduced", "failed", "blocked"]
    oracle_exit_code: int | None
    scope_violations: list[str]
    usage: RunUsage
    verdict: EvaluationVerdict | None = None
    sanitized_error_code: str | None = None
    budget_violations: list[str] = Field(default_factory=list)


class AggregateMetrics(StrictModel):
    run_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    false_fixed_count: int = Field(ge=0)
    false_fixed_per_fixed_claim: float = Field(ge=0)
    false_fixed_per_all_tasks: float = Field(ge=0)
    not_reproduced_claim_count: int = Field(ge=0)
    correct_not_reproduced_count: int = Field(ge=0)
    not_reproduced_accuracy: float = Field(ge=0, le=1)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_model_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_wall_seconds: float = Field(ge=0)
    successes_per_100k_tokens: float = Field(ge=0)
    median_tokens_per_success: float = Field(ge=0)


class EvaluationProvenanceBatch(StrictModel):
    run_ids: list[str] = Field(min_length=1)
    capture_timing: Literal["runtime", "historical_backfill"]
    source_repository: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dirty: bool
    runner_revision: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    runner_dirty: bool
    main_model_name: str = Field(min_length=1)
    compaction_model_name: str = Field(min_length=1)
    endpoint_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version: str = Field(min_length=1)
    python_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_enforcement: Literal[
        "post_run_observation",
        "pre_call_reservation",
    ]
    model_accounting: Literal[
        "main_model_trace_only",
        "all_model_roles",
    ]


class EvaluationSummary(StrictModel):
    schema_version: int = 1
    loop: Literal["legacy", "experiment"]
    status: Literal["complete", "stopped_budget"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget: EvaluationBudget
    case_ids: list[str] = Field(min_length=1)
    case_expectations: dict[str, Literal["fixed", "not_reproduced"]]
    expected_runs_per_case: int = Field(gt=0)
    runs: list[EvaluationRun]
    provenance: list[EvaluationProvenanceBatch] = Field(min_length=1)
    aggregate: AggregateMetrics

    @model_validator(mode="after")
    def validate_run_summary(self) -> EvaluationSummary:
        if self.aggregate.run_count != len(self.runs):
            raise ValueError("aggregate run_count does not match runs")
        if any(run.verdict is None for run in self.runs):
            raise ValueError("summary runs must be graded")
        if any(run.case_id not in self.case_ids for run in self.runs):
            raise ValueError("summary run references an unknown case_id")
        return self


class GateThresholds(StrictModel):
    minimum_newly_solved_stable_failures: int = Field(ge=0)
    minimum_success_rate_delta: float = Field(ge=-1, le=1)
    maximum_false_fixed_rate: float = Field(ge=0, le=1)
    maximum_false_fixed_rate_ratio_to_legacy: float = Field(ge=0)
    minimum_wrong_hypothesis_recovery_delta: float = Field(ge=-1, le=1)
    maximum_token_multiplier: float = Field(gt=0)
    fault_invariant_violations: int = Field(ge=0)


class GateDefinition(StrictModel):
    schema_version: Literal[1] = 1
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_legacy_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runs_per_case: int = Field(gt=0)
    budget: EvaluationBudget
    thresholds: GateThresholds
