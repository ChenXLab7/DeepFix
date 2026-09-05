import pytest
from pydantic import ValidationError

from deepfix.evaluation.models import EvaluationBudget, EvaluationCase


def test_case_requires_oracle_and_allowed_paths() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({"case_id": "mergesort", "problem": "broken"})


def test_budget_rejects_zero_token_cap() -> None:
    with pytest.raises(ValidationError):
        EvaluationBudget(
            max_input_tokens=0,
            max_output_tokens=10_000,
            max_wall_seconds=600,
            max_tool_calls=40,
            max_side_effects=10,
        )


def test_case_id_cannot_escape_the_runs_directory() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            case_id="../outside",
            problem="broken",
            allowed_paths=["python_programs/x.py"],
            required_command="python -m pytest python_testcases/test_x.py -q",
            expected_outcome="fixed",
        )
