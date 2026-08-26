from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepfix.evaluation.harness import CaseRunner, EvaluationHarness
from deepfix.evaluation.legacy import LegacyLoopRunner
from deepfix.evaluation.manifest import load_manifest
from deepfix.evaluation.metrics import (
    aggregate_runs,
    grade_run,
    grade_run_for_expected_outcome,
)
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationProvenanceBatch,
    EvaluationRun,
    EvaluationSummary,
)
from deepfix.evaluation.provenance import build_provenance_batch

RunnerFactory = Callable[[Path], CaseRunner]

_FORBIDDEN_SUMMARY_FIELDS = frozenset(
    {
        "api_key",
        "api_keys",
        "environment",
        "messages",
        "prompt",
        "prompts",
        "secret",
        "secrets",
        "source_code",
        "stderr",
        "stdout",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m deepfix.evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--project", required=True)
    baseline.add_argument("--python", default=sys.executable)
    baseline.add_argument("--runs", type=_positive_int, default=3)
    baseline.add_argument("--run-start", type=_positive_int, default=1)
    baseline.add_argument("--case", dest="selected_cases", action="append")
    baseline.add_argument("--output")
    baseline.add_argument("--summary")
    baseline.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate-summary")
    validate.add_argument("path")

    merge = subparsers.add_parser("merge-summaries")
    merge.add_argument("summaries", nargs="+")
    merge.add_argument("--output", required=True)
    merge.add_argument("--manifest")
    merge.add_argument("--historical-provenance", action="append")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner_factory: RunnerFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-summary":
        try:
            summary = validate_summary(Path(args.path))
        except (OSError, ValueError) as error:
            print(f"invalid summary: {error}", file=sys.stderr)
            return 2
        print(
            f"valid summary: loop={summary.loop} "
            f"cases={len(summary.case_ids)} runs={len(summary.runs)}"
        )
        return 0
    if args.command == "merge-summaries":
        return _merge_summaries(
            [Path(path) for path in args.summaries],
            Path(args.output),
            manifest_path=Path(args.manifest) if args.manifest else None,
            historical_provenance_paths=(
                [Path(path) for path in args.historical_provenance]
                if args.historical_provenance
                else None
            ),
        )

    manifest_path = Path(args.manifest).expanduser().resolve()
    try:
        manifest = load_manifest(manifest_path)
        cases = _select_cases(manifest.cases, args.selected_cases)
    except (OSError, TypeError, ValueError) as error:
        print(f"invalid manifest: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"dry-run: cases={len(cases)} runs_each={args.runs}")
        for case in cases:
            print(f"- {case.case_id}: expected={case.expected_outcome}")
        return 0

    if os.environ.get("DEEPFIX_RUN_ONLINE") != "1":
        print(
            "online evaluation disabled; set DEEPFIX_RUN_ONLINE=1 explicitly",
            file=sys.stderr,
        )
        return 2
    if not args.output or not args.summary:
        print("--output and --summary are required for online evaluation", file=sys.stderr)
        return 2

    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        print(f"project directory does not exist: {project}", file=sys.stderr)
        return 2
    project_python = Path(args.python).expanduser().resolve()
    if not project_python.is_file():
        print(f"python executable does not exist: {project_python}", file=sys.stderr)
        return 2

    resolved_runner_factory = runner_factory or (
        lambda python: LegacyLoopRunner(project_python=python)
    )
    harness = EvaluationHarness(
        Path(args.output),
        resolved_runner_factory(project_python),
    )
    graded_runs: list[EvaluationRun] = []
    stopped_budget = False
    for case in cases:
        for run_index in range(args.run_start, args.run_start + args.runs):
            execution = harness.run_case(
                case,
                project,
                run_index=run_index,
                budget=manifest.budget,
            )
            violations = _budget_violations(execution.run, manifest.budget)
            graded = execution.run.model_copy(
                update={
                    "verdict": grade_run(case, execution.run),
                    "budget_violations": violations,
                }
            )
            (execution.run_dir / "result.json").write_text(
                graded.model_dump_json(indent=2),
                encoding="utf-8",
            )
            graded_runs.append(graded)
            if violations:
                stopped_budget = True
                break
        if stopped_budget:
            break

    canonical_runs = _canonical_runs(
        graded_runs,
        [case.case_id for case in cases],
    )
    summary = EvaluationSummary(
        loop="legacy",
        status="stopped_budget" if stopped_budget else "complete",
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        budget=manifest.budget,
        case_ids=[case.case_id for case in cases],
        case_expectations={case.case_id: case.expected_outcome for case in cases},
        expected_runs_per_case=args.runs,
        runs=canonical_runs,
        provenance=[
            build_provenance_batch(
                project,
                project_python,
                [run.run_id for run in canonical_runs],
            )
        ],
        aggregate=aggregate_runs(canonical_runs),
    )
    summary_path = Path(args.summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    validate_summary(summary_path)

    if stopped_budget:
        print(f"stopped baseline after budget violation; summary={summary_path}")
        return 3
    print(f"completed baseline: runs={len(graded_runs)} summary={summary_path}")
    return 0


def validate_summary(path: Path) -> EvaluationSummary:
    raw = json.loads(path.read_text(encoding="utf-8"))
    forbidden = _find_forbidden_field(raw)
    if forbidden is not None:
        raise ValueError(f"forbidden summary field: {forbidden}")
    summary = EvaluationSummary.model_validate(raw)
    _validate_summary_integrity(summary)
    return summary


def _merge_summaries(
    paths: list[Path],
    output: Path,
    *,
    manifest_path: Path | None,
    historical_provenance_paths: list[Path] | None,
) -> int:
    if len(paths) < 2:
        print("merge requires at least two summaries", file=sys.stderr)
        return 2
    try:
        if (
            historical_provenance_paths is not None
            and len(historical_provenance_paths) != len(paths)
        ):
            raise ValueError("historical provenance count must match summary count")
        manifest = load_manifest(manifest_path) if manifest_path is not None else None
        resolved_manifest_path = (
            manifest_path.expanduser().resolve()
            if manifest_path is not None
            else None
        )
        summaries = [
            _load_summary_for_merge(
                path,
                manifest=manifest,
                manifest_path=resolved_manifest_path,
                historical_provenance=(
                    _load_historical_provenance(
                        historical_provenance_paths[index]
                    )
                    if historical_provenance_paths
                    else None
                ),
            )
            for index, path in enumerate(paths)
        ]
        first = summaries[0]
        runs: list[EvaluationRun] = []
        seen_runs: set[tuple[str, str]] = set()
        for summary in summaries:
            _require_compatible_summary(first, summary)
            for run in summary.runs:
                identity = (run.case_id, run.run_id)
                if identity in seen_runs:
                    raise ValueError(
                        f"duplicate run: case_id={run.case_id} run_id={run.run_id}"
                    )
                seen_runs.add(identity)
                runs.append(run)
    except (OSError, TypeError, ValueError) as error:
        print(f"cannot merge summaries: {error}", file=sys.stderr)
        return 2

    merged_runs = _canonical_runs(runs, first.case_ids)
    merged = EvaluationSummary(
        loop=first.loop,
        status=(
            "stopped_budget"
            if any(summary.status == "stopped_budget" for summary in summaries)
            else "complete"
        ),
        manifest_sha256=first.manifest_sha256,
        budget=first.budget,
        case_ids=first.case_ids,
        case_expectations=first.case_expectations,
        expected_runs_per_case=sum(
            summary.expected_runs_per_case for summary in summaries
        ),
        runs=merged_runs,
        provenance=[
            provenance
            for summary in summaries
            for provenance in summary.provenance
        ],
        aggregate=aggregate_runs(merged_runs),
    )
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    print(f"merged summaries: runs={len(runs)} output={output_path}")
    return 0


def _load_summary_for_merge(
    path: Path,
    *,
    manifest,
    manifest_path: Path | None,
    historical_provenance: EvaluationProvenanceBatch | None,
) -> EvaluationSummary:
    resolved_path = path.expanduser().resolve()
    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    new_fields = {"case_expectations", "expected_runs_per_case", "provenance"}
    present_new_fields = new_fields.intersection(raw)
    if present_new_fields:
        if present_new_fields != new_fields:
            missing = ", ".join(sorted(new_fields - present_new_fields))
            raise ValueError(f"partially upgraded summary is missing: {missing}")
        return validate_summary(resolved_path)

    if (
        manifest is None
        or manifest_path is None
        or historical_provenance is None
    ):
        raise ValueError(
            "legacy summary requires manifest and per-batch historical provenance"
        )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if raw.get("manifest_sha256") != manifest_hash:
        raise ValueError("legacy summary manifest hash does not match")
    case_by_id = {case.case_id: case for case in manifest.cases}
    case_ids = raw.get("case_ids")
    if not isinstance(case_ids, list) or any(
        case_id not in case_by_id for case_id in case_ids
    ):
        raise ValueError("legacy summary contains unknown cases")
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise TypeError("legacy summary runs are invalid")
    coverage = {
        case_id: sum(run.get("case_id") == case_id for run in runs)
        for case_id in case_ids
    }
    coverage_counts = set(coverage.values())
    if raw.get("status") != "complete" or len(coverage_counts) != 1:
        raise ValueError("legacy summary must have complete uniform coverage")
    expected_runs_per_case = coverage_counts.pop()
    if expected_runs_per_case <= 0:
        raise ValueError("legacy summary contains no runs")
    run_ids = [run.get("run_id") for run in runs]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise ValueError("legacy summary run IDs are invalid")

    raw["case_expectations"] = {
        case_id: case_by_id[case_id].expected_outcome for case_id in case_ids
    }
    raw["expected_runs_per_case"] = expected_runs_per_case
    if historical_provenance.capture_timing != "historical_backfill":
        raise ValueError("historical provenance must be marked as backfilled")
    if set(historical_provenance.run_ids) != set(run_ids):
        raise ValueError("historical provenance does not match summary runs")
    raw["provenance"] = [historical_provenance.model_dump(mode="json")]
    summary = EvaluationSummary.model_validate(raw)
    _validate_summary_integrity(summary)
    return summary


def _load_historical_provenance(path: Path) -> EvaluationProvenanceBatch:
    return EvaluationProvenanceBatch.model_validate_json(
        path.expanduser().resolve().read_text(encoding="utf-8")
    )


def _require_compatible_summary(
    expected: EvaluationSummary,
    actual: EvaluationSummary,
) -> None:
    checks = (
        ("loop", expected.loop, actual.loop),
        ("manifest_sha256", expected.manifest_sha256, actual.manifest_sha256),
        ("budget", expected.budget, actual.budget),
        ("case_ids", expected.case_ids, actual.case_ids),
        (
            "case_expectations",
            expected.case_expectations,
            actual.case_expectations,
        ),
    )
    for name, expected_value, actual_value in checks:
        if actual_value != expected_value:
            raise ValueError(f"incompatible {name}")


def _canonical_runs(
    runs: list[EvaluationRun],
    case_ids: list[str],
) -> list[EvaluationRun]:
    case_order = {case_id: index for index, case_id in enumerate(case_ids)}
    return sorted(runs, key=lambda run: (case_order[run.case_id], run.run_id))


def _validate_summary_integrity(summary: EvaluationSummary) -> None:
    if set(summary.case_expectations) != set(summary.case_ids):
        raise ValueError("case expectations do not match case_ids")

    run_ids: set[str] = set()
    task_ids: set[str] = set()
    coverage = {case_id: 0 for case_id in summary.case_ids}
    for run in summary.runs:
        if run.loop != summary.loop:
            raise ValueError(f"run loop does not match summary: {run.run_id}")
        if run.run_id in run_ids:
            raise ValueError(f"duplicate run: {run.run_id}")
        run_ids.add(run.run_id)
        if run.task_id in task_ids:
            raise ValueError(f"duplicate task: {run.task_id}")
        task_ids.add(run.task_id)
        coverage[run.case_id] += 1
        expected_verdict = grade_run_for_expected_outcome(
            summary.case_expectations[run.case_id],
            run,
        )
        if run.verdict != expected_verdict:
            raise ValueError(f"run verdict does not match evidence: {run.run_id}")

    if summary.status == "complete" and any(
        count != summary.expected_runs_per_case for count in coverage.values()
    ):
        raise ValueError("summary case coverage is incomplete")
    if summary.status == "stopped_budget" and any(
        count > summary.expected_runs_per_case for count in coverage.values()
    ):
        raise ValueError("summary case coverage exceeds expected runs")

    provenance_run_ids = [
        run_id
        for provenance in summary.provenance
        for run_id in provenance.run_ids
    ]
    if len(provenance_run_ids) != len(set(provenance_run_ids)):
        raise ValueError("duplicate provenance run coverage")
    if set(provenance_run_ids) != run_ids:
        raise ValueError("provenance coverage does not match runs")

    expected_aggregate = aggregate_runs(summary.runs)
    if summary.aggregate != expected_aggregate:
        raise ValueError("aggregate does not match runs")


def _find_forbidden_field(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_SUMMARY_FIELDS:
                return str(key)
            nested = _find_forbidden_field(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_forbidden_field(item)
            if nested is not None:
                return nested
    return None


def _select_cases(
    cases: list[EvaluationCase],
    selected_case_ids: list[str] | None,
) -> list[EvaluationCase]:
    if not selected_case_ids:
        return cases
    selected = set(selected_case_ids)
    by_id = {case.case_id: case for case in cases}
    missing = sorted(selected - by_id.keys())
    if missing:
        raise ValueError(f"unknown case_id: {', '.join(missing)}")
    return [case for case in cases if case.case_id in selected]


def _budget_violations(
    run: EvaluationRun,
    budget: EvaluationBudget,
) -> list[str]:
    checks = (
        ("max_input_tokens", run.usage.input_tokens, budget.max_input_tokens),
        ("max_output_tokens", run.usage.output_tokens, budget.max_output_tokens),
        ("max_wall_seconds", run.usage.wall_seconds, budget.max_wall_seconds),
        ("max_tool_calls", run.usage.tool_calls, budget.max_tool_calls),
    )
    return [name for name, actual, maximum in checks if actual > maximum]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
