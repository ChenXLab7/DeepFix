from __future__ import annotations

from typing import Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel


class EvaluationBudget(StrictModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_wall_seconds: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_side_effects: int = Field(ge=0)


class EvaluationCase(StrictModel):
    case_id: str = Field(min_length=1)
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
