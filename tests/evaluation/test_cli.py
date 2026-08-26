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
            task_id=f"task-{run_dir.name}",
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
    assert summary.case_expectations == {"sample-buggy": "fixed"}
    assert summary.expected_runs_per_case == 1
    assert len(summary.provenance) == 1
    assert summary.provenance[0].run_ids == ["sample-buggy-001"]
    assert summary.provenance[0].budget_enforcement == "post_run_observation"
    assert summary.provenance[0].model_accounting == "main_model_trace_only"
    assert len(summary.provenance[0].endpoint_fingerprint) == 64
    assert summary.runs[0].verdict is not None
    assert summary.runs[0].verdict.success is True
    assert (tmp_path / "runs" / "sample-buggy" / "001" / "result.json").is_file()
    assert "completed baseline" in capsys.readouterr().out


def test_cli_run_start_assigns_non_overlapping_run_ids(
    tmp_path,
    monkeypatch,
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
            "--runs",
            "2",
            "--run-start",
            "2",
            "--output",
            str(tmp_path / "runs"),
            "--summary",
            str(summary_path),
        ],
        runner_factory=lambda _python: FixedRunner(),
    )

    summary = validate_summary(summary_path)
    assert code == 0
    assert [run.run_id for run in summary.runs] == [
        "sample-buggy-002",
        "sample-buggy-003",
    ]
    assert (tmp_path / "runs" / "sample-buggy" / "002" / "result.json").is_file()
    assert (tmp_path / "runs" / "sample-buggy" / "003" / "result.json").is_file()


def test_cli_rejects_non_positive_run_start(tmp_path) -> None:
    manifest = _write_manifest(tmp_path / "cases.json")

    with pytest.raises(SystemExit):
        main(
            [
                "baseline",
                "--manifest",
                str(manifest),
                "--project",
                str(tmp_path),
                "--run-start",
                "0",
                "--dry-run",
            ]
        )


def test_cli_merges_compatible_summaries_and_recomputes_aggregate(
    tmp_path,
    monkeypatch,
) -> None:
    first = _write_batch_summary(tmp_path, monkeypatch, run_start=1)
    second = _write_batch_summary(tmp_path, monkeypatch, run_start=2)
    merged_path = tmp_path / "merged.json"

    code = main(
        [
            "merge-summaries",
            "--output",
            str(merged_path),
            str(second),
            str(first),
        ]
    )

    merged = validate_summary(merged_path)
    assert code == 0
    assert [run.run_id for run in merged.runs] == [
        "sample-buggy-001",
        "sample-buggy-002",
    ]
    assert merged.aggregate.run_count == 2
    assert merged.aggregate.total_input_tokens == 160
    assert merged.expected_runs_per_case == 2
    assert len(merged.provenance) == 2


def test_cli_merge_rejects_mismatched_budget(tmp_path, monkeypatch, capsys) -> None:
    first = _write_batch_summary(tmp_path, monkeypatch, run_start=1)
    second = _write_batch_summary(tmp_path, monkeypatch, run_start=2)
    raw = json.loads(second.read_text(encoding="utf-8"))
    raw["budget"]["max_input_tokens"] = 99_999
    second.write_text(json.dumps(raw), encoding="utf-8")

    code = main(
        [
            "merge-summaries",
            "--output",
            str(tmp_path / "merged.json"),
            str(first),
            str(second),
        ]
    )

    assert code == 2
    assert "budget" in capsys.readouterr().err
    assert not (tmp_path / "merged.json").exists()


def test_cli_merge_rejects_duplicate_run_ids(tmp_path, monkeypatch, capsys) -> None:
    summary = _write_batch_summary(tmp_path, monkeypatch, run_start=1)

    code = main(
        [
            "merge-summaries",
            "--output",
            str(tmp_path / "merged.json"),
            str(summary),
            str(summary),
        ]
    )

    assert code == 2
    assert "duplicate run" in capsys.readouterr().err
    assert not (tmp_path / "merged.json").exists()


def test_cli_merge_upgrades_complete_legacy_batches_with_explicit_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    first = _write_batch_summary(tmp_path, monkeypatch, run_start=1)
    second = _write_batch_summary(tmp_path, monkeypatch, run_start=2)
    for path in (first, second):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("case_expectations")
        raw.pop("expected_runs_per_case")
        raw.pop("provenance")
        path.write_text(json.dumps(raw), encoding="utf-8")
    merged_path = tmp_path / "merged-legacy.json"

    code = main(
        [
            "merge-summaries",
            "--manifest",
            str(first.parent / "cases.json"),
            "--project",
            str(first.parent / "source"),
            "--python",
            sys.executable,
            "--runner-revision",
            "a" * 40,
            "--runner-revision",
            "b" * 40,
            "--output",
            str(merged_path),
            str(first),
            str(second),
        ]
    )

    merged = validate_summary(merged_path)
    assert code == 0
    assert merged.expected_runs_per_case == 2
    assert [item.runner_revision for item in merged.provenance] == [
        "a" * 40,
        "b" * 40,
    ]
    assert all(item.runner_dirty is False for item in merged.provenance)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda raw: raw["aggregate"].update({"success_count": 0}), "aggregate"),
        (
            lambda raw: raw["runs"][0]["verdict"].update({"success": False}),
            "verdict",
        ),
        (lambda raw: raw["runs"][0].update({"loop": "experiment"}), "loop"),
    ],
)
def test_validate_summary_rejects_tampered_derived_facts(
    tmp_path,
    monkeypatch,
    mutation,
    error,
) -> None:
    summary = _write_batch_summary(tmp_path, monkeypatch, run_start=1)
    raw = json.loads(summary.read_text(encoding="utf-8"))
    mutation(raw)
    summary.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_summary(summary)


def test_validate_summary_rejects_duplicate_task_ids(
    tmp_path,
    monkeypatch,
) -> None:
    summary = _write_batch_summary(
        tmp_path,
        monkeypatch,
        run_start=1,
        runs=2,
    )
    raw = json.loads(summary.read_text(encoding="utf-8"))
    raw["runs"][1]["task_id"] = raw["runs"][0]["task_id"]
    summary.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task"):
        validate_summary(summary)


def test_validate_summary_rejects_incomplete_case_coverage(
    tmp_path,
    monkeypatch,
) -> None:
    summary = _write_batch_summary(
        tmp_path,
        monkeypatch,
        run_start=1,
        runs=2,
    )
    raw = json.loads(summary.read_text(encoding="utf-8"))
    raw["runs"] = raw["runs"][:1]
    raw["aggregate"]["run_count"] = 1
    raw["provenance"][0]["run_ids"] = [raw["runs"][0]["run_id"]]
    summary.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage"):
        validate_summary(summary)


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


def _write_batch_summary(
    tmp_path,
    monkeypatch,
    *,
    run_start: int,
    runs: int = 1,
) -> Path:
    monkeypatch.setenv("DEEPFIX_RUN_ONLINE", "1")
    batch = tmp_path / f"batch-{run_start}"
    batch.mkdir()
    manifest = _write_manifest(batch / "cases.json")
    source = batch / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    summary = batch / "summary.json"
    code = main(
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--project",
            str(source),
            "--runs",
            str(runs),
            "--run-start",
            str(run_start),
            "--output",
            str(batch / "runs"),
            "--summary",
            str(summary),
        ],
        runner_factory=lambda _python: FixedRunner(),
    )
    assert code == 0
    return summary
