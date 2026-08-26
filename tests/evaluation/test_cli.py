import json
import sys
from pathlib import Path

import pytest

from deepfix.evaluation.__main__ import main, validate_summary
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
    RunUsage,
)


def _write_manifest(path: Path, *, cases: list[dict] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "budget": {
                    "max_input_tokens": 100_000,
                    "max_output_tokens": 20_000,
                    "max_wall_seconds": 600,
                    "max_tool_calls": 40,
                    "max_side_effects": 5,
                },
                "cases": cases
                or [
                    {
                        "case_id": "sample-buggy",
                        "problem": "repair the sample",
                        "allowed_paths": ["value.py"],
                        "required_command": "python -m pytest -q",
                        "expected_outcome": "fixed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class FixedRunner:
    def run(
        self,
        case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        return EvaluationRun(
            run_id=f"{case.case_id}-{run_dir.name}",
            case_id=case.case_id,
            loop="legacy",
            task_id="task-fixture",
            conclusion="fixed",
            oracle_exit_code=0,
            scope_violations=[],
            usage=RunUsage(
                input_tokens=80,
                output_tokens=20,
                model_calls=1,
                tool_calls=2,
                wall_seconds=1,
            ),
        )


def test_cli_dry_run_lists_cases_without_constructing_runner(
    tmp_path,
    capsys,
) -> None:
    manifest = _write_manifest(tmp_path / "cases.json")

    def forbidden_runner_factory(_python: Path):
        raise AssertionError("dry-run must not construct a runner")

    code = main(
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--project",
            str(tmp_path),
            "--dry-run",
        ],
        runner_factory=forbidden_runner_factory,
    )

    assert code == 0
    assert "sample-buggy" in capsys.readouterr().out


def test_cli_refuses_model_execution_without_online_opt_in(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("DEEPFIX_RUN_ONLINE", raising=False)
    manifest = _write_manifest(tmp_path / "cases.json")

    code = main(
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--project",
            str(tmp_path),
            "--runs",
            "1",
            "--output",
            str(tmp_path / "runs"),
            "--summary",
            str(tmp_path / "summary.json"),
        ],
        runner_factory=lambda _python: pytest.fail("runner must not be constructed"),
    )

    assert code == 2
    assert "DEEPFIX_RUN_ONLINE=1" in capsys.readouterr().err


def test_cli_runs_real_harness_with_injected_runner_and_writes_sanitized_summary(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("DEEPFIX_RUN_ONLINE", "1")
    manifest = _write_manifest(tmp_path / "cases.json")
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    code = main(
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--project",
            str(source),
            "--python",
            sys.executable,
            "--runs",
            "1",
            "--output",
            str(tmp_path / "runs"),
            "--summary",
            str(summary_path),
        ],
        runner_factory=lambda _python: FixedRunner(),
    )

    assert code == 0
    summary = validate_summary(summary_path)
    assert summary.status == "complete"
    assert summary.case_ids == ["sample-buggy"]
    assert summary.aggregate.success_count == 1
    assert summary.runs[0].verdict is not None
    assert summary.runs[0].verdict.success is True
    assert (tmp_path / "runs" / "sample-buggy" / "001" / "result.json").is_file()
    assert "completed baseline" in capsys.readouterr().out


def test_validate_summary_rejects_forbidden_prompt_field(tmp_path) -> None:
    path = tmp_path / "unsafe-summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "loop": "legacy",
                "status": "complete",
                "manifest_sha256": "a" * 64,
                "budget": {
                    "max_input_tokens": 100,
                    "max_output_tokens": 20,
                    "max_wall_seconds": 10,
                    "max_tool_calls": 5,
                    "max_side_effects": 1,
                },
                "case_ids": ["sample-buggy"],
                "runs": [],
                "aggregate": {
                    "run_count": 0,
                    "success_count": 0,
                    "success_rate": 0,
                    "false_fixed_count": 0,
                    "false_fixed_per_fixed_claim": 0,
                    "false_fixed_per_all_tasks": 0,
                    "not_reproduced_claim_count": 0,
                    "correct_not_reproduced_count": 0,
                    "not_reproduced_accuracy": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_model_calls": 0,
                    "total_tool_calls": 0,
                    "total_wall_seconds": 0,
                    "successes_per_100k_tokens": 0,
                    "median_tokens_per_success": 0,
                },
                "prompt": "must never be committed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden summary field"):
        validate_summary(path)


def test_cli_stops_repetitions_after_observed_token_budget_exceeded(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPFIX_RUN_ONLINE", "1")
    manifest = _write_manifest(tmp_path / "cases.json")
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    class OverBudgetRunner(FixedRunner):
        def __init__(self) -> None:
            self.calls = 0

        def run(self, *args, **kwargs) -> EvaluationRun:
            self.calls += 1
            run = super().run(*args, **kwargs)
            return run.model_copy(
                update={
                    "usage": run.usage.model_copy(
                        update={"input_tokens": 100_001}
                    )
                }
            )

    runner = OverBudgetRunner()
    code = main(
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--project",
            str(source),
            "--runs",
            "2",
            "--output",
            str(tmp_path / "runs"),
            "--summary",
            str(summary_path),
        ],
        runner_factory=lambda _python: runner,
    )

    summary = validate_summary(summary_path)
    assert code == 3
    assert runner.calls == 1
    assert summary.status == "stopped_budget"
    assert summary.runs[0].budget_violations == ["max_input_tokens"]
