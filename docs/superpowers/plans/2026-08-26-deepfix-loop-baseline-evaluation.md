# DeepFix Legacy Loop Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible evaluation harness and freeze the current Tool-Call Loop baseline before implementing the Experiment Loop.

**Architecture:** Add an evaluation-only package with strict case/result models, deterministic outcome grading, LLM trace metrics, a legacy runner adapter, and a PowerShell-compatible module CLI. Each run uses a fresh copied project and produces sanitized JSON; real-model runs are never part of the default offline pytest suite.

**Tech Stack:** Python 3.11+, Pydantic 2, pathlib/shutil, JSON/JSONL, existing BugfixService and LLMTraceMiddleware, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-26-deepfix-capability-oriented-agent-loop-design.md`

## Global Constraints

- This plan measures the existing loop only; it does not add Experiment behavior.
- Every case starts from an identical source tree and receives its own Task/database/artifact directory.
- Gold labels and grading commands are not injected into the Agent prompt beyond the user-visible required verification command.
- API keys, full prompts, source code, stdout bodies, and environment variables never enter committed result summaries.
- Online benchmark execution requires `DEEPFIX_RUN_ONLINE=1`; default pytest remains offline.
- Baseline thresholds are frozen before the new-loop A/B run.

## File Structure

### New production/evaluation files

- `src/deepfix/evaluation/__init__.py` — public evaluation contracts.
- `src/deepfix/evaluation/models.py` — case, budget, run, verdict, and aggregate Pydantic models.
- `src/deepfix/evaluation/manifest.py` — strict JSON manifest loader and validation.
- `src/deepfix/evaluation/metrics.py` — deterministic grading and aggregate calculations.
- `src/deepfix/evaluation/traces.py` — sanitized `llm_calls.jsonl` usage extraction.
- `src/deepfix/evaluation/harness.py` — clean-copy orchestration and injectable case runner protocol.
- `src/deepfix/evaluation/legacy.py` — current BugfixService runner adapter.
- `src/deepfix/evaluation/__main__.py` — `python -m deepfix.evaluation` entry point.
- `evaluations/quixbugs-python.json` — fixed Python case definitions and resource envelope.
- `evaluations/results/.gitkeep` — result directory without secrets/raw traces.

### Tests

- `tests/evaluation/__init__.py`
- `tests/evaluation/test_models.py`
- `tests/evaluation/test_manifest.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_traces.py`
- `tests/evaluation/test_harness.py`
- `tests/evaluation/test_cli.py`

---

### Task 1: Evaluation Contracts and Manifest

**Files:**
- Create: `src/deepfix/evaluation/__init__.py`
- Create: `src/deepfix/evaluation/models.py`
- Create: `src/deepfix/evaluation/manifest.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_models.py`
- Create: `tests/evaluation/test_manifest.py`

**Interfaces:**
- Consumes: `deepfix.compaction.models.StrictModel`.
- Produces: `EvaluationBudget`, `EvaluationCase`, `EvaluationManifest`, `RunUsage`, `EvaluationVerdict`, `EvaluationRun`, `load_manifest(path: Path) -> EvaluationManifest`.

- [ ] **Step 1: Write failing strict-model tests**

```python
# tests/evaluation/test_models.py
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
```

- [ ] **Step 2: Run model tests and verify RED**

Run: `python -m pytest tests/evaluation/test_models.py -q`

Expected: FAIL because `deepfix.evaluation.models` does not exist.

- [ ] **Step 3: Implement strict contracts**

```python
# src/deepfix/evaluation/models.py
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
    run_id: str
    case_id: str
    loop: Literal["legacy", "experiment"]
    task_id: str
    conclusion: Literal["fixed", "not_reproduced", "failed", "blocked"]
    oracle_exit_code: int | None
    scope_violations: list[str]
    usage: RunUsage
    verdict: EvaluationVerdict | None = None
    sanitized_error_code: str | None = None


class AggregateMetrics(StrictModel):
    run_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    false_fixed_count: int = Field(ge=0)
    false_fixed_per_fixed_claim: float = Field(ge=0)
    false_fixed_per_all_tasks: float = Field(ge=0)
    successes_per_100k_tokens: float = Field(ge=0)
    median_tokens_per_success: float = Field(ge=0)
```

- [ ] **Step 4: Write and run manifest validation tests**

```python
# tests/evaluation/test_manifest.py
import json

import pytest

from deepfix.evaluation.manifest import load_manifest


def test_manifest_rejects_duplicate_case_ids(tmp_path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "case_id": "same",
        "problem": "run the required test",
        "allowed_paths": ["python_programs/x.py"],
        "required_command": "python -m pytest python_testcases/test_x.py -q",
        "expected_outcome": "fixed",
    }
    path.write_text(json.dumps({
        "schema_version": 1,
        "budget": {
            "max_input_tokens": 100000,
            "max_output_tokens": 20000,
            "max_wall_seconds": 600,
            "max_tool_calls": 40,
            "max_side_effects": 5,
        },
        "cases": [case, case],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_manifest(path)
```

Run: `python -m pytest tests/evaluation/test_models.py tests/evaluation/test_manifest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit contracts**

```bash
git add src/deepfix/evaluation tests/evaluation
git commit -m "feat: add evaluation case contracts"
```

### Task 2: Deterministic Grading and Aggregate Metrics

**Files:**
- Create: `src/deepfix/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: `EvaluationCase`, `EvaluationRun`.
- Produces: `grade_run(case: EvaluationCase, run: EvaluationRun) -> EvaluationVerdict`, `aggregate_runs(runs: list[EvaluationRun]) -> AggregateMetrics`.

- [ ] **Step 1: Write failing false-FIXED and efficiency tests**

```python
# tests/evaluation/test_metrics.py
from deepfix.evaluation.metrics import aggregate_runs, grade_run
from deepfix.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationVerdict,
    RunUsage,
)


CASE = EvaluationCase(
    case_id="mergesort",
    problem="repair mergesort and run the required test",
    allowed_paths=["python_programs/mergesort.py"],
    required_command="python -m pytest python_testcases/test_mergesort.py -q",
    expected_outcome="fixed",
)


def make_run(
    *,
    conclusion="fixed",
    oracle_exit_code=0,
    success=False,
    input_tokens=10,
    output_tokens=10,
) -> EvaluationRun:
    return EvaluationRun(
        run_id="run-1",
        case_id=CASE.case_id,
        loop="legacy",
        task_id="task-1",
        conclusion=conclusion,
        oracle_exit_code=oracle_exit_code,
        scope_violations=[],
        usage=RunUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_calls=1,
            tool_calls=1,
            wall_seconds=1,
        ),
        verdict=EvaluationVerdict(success=success, false_fixed=False),
    )


def test_fixed_claim_with_failing_oracle_is_false_fixed() -> None:
    run = make_run(conclusion="fixed", oracle_exit_code=1)
    verdict = grade_run(CASE, run)
    assert verdict.false_fixed is True
    assert verdict.success is False


def test_successes_per_100k_tokens() -> None:
    runs = [
        make_run(success=True, input_tokens=60_000, output_tokens=10_000),
        make_run(success=True, input_tokens=20_000, output_tokens=10_000),
    ]
    assert aggregate_runs(runs).successes_per_100k_tokens == 2.0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/evaluation/test_metrics.py -q`

Expected: FAIL because metric functions do not exist.

- [ ] **Step 3: Implement deterministic formulas**

```python
# src/deepfix/evaluation/metrics.py
import statistics


def grade_run(case: EvaluationCase, run: EvaluationRun) -> EvaluationVerdict:
    false_fixed = (
        run.conclusion == "fixed"
        and (run.oracle_exit_code != 0 or bool(run.scope_violations))
    )
    success = (
        not false_fixed
        and run.oracle_exit_code == 0
        and run.conclusion == case.expected_outcome
    )
    return EvaluationVerdict(success=success, false_fixed=false_fixed)


def aggregate_runs(runs: list[EvaluationRun]) -> AggregateMetrics:
    total_tokens = sum(r.usage.input_tokens + r.usage.output_tokens for r in runs)
    successes = sum(bool(r.verdict and r.verdict.success) for r in runs)
    false_fixed = sum(bool(r.verdict and r.verdict.false_fixed) for r in runs)
    fixed_claims = sum(r.conclusion == "fixed" for r in runs)
    successful_token_counts = [
        r.usage.input_tokens + r.usage.output_tokens
        for r in runs
        if r.verdict and r.verdict.success
    ]
    return AggregateMetrics(
        run_count=len(runs),
        success_count=successes,
        false_fixed_count=false_fixed,
        false_fixed_per_fixed_claim=(false_fixed / fixed_claims) if fixed_claims else 0.0,
        false_fixed_per_all_tasks=(false_fixed / len(runs)) if runs else 0.0,
        successes_per_100k_tokens=(successes * 100_000 / total_tokens) if total_tokens else 0.0,
        median_tokens_per_success=(
            statistics.median(successful_token_counts)
            if successful_token_counts
            else 0.0
        ),
    )
```

Add both false-FIXED denominators, NOT_REPRODUCED accuracy, median tokens per success, tool/model-call totals, and wall-time totals.

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/evaluation/test_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit metrics**

```bash
git add src/deepfix/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: grade repair evaluation runs"
```

### Task 3: Trace Usage Extraction

**Files:**
- Create: `src/deepfix/evaluation/traces.py`
- Test: `tests/evaluation/test_traces.py`

**Interfaces:**
- Consumes: sanitized JSONL records written by `LLMTraceMiddleware`.
- Produces: `summarize_llm_trace(path: Path, task_id: str) -> RunUsage`.

- [ ] **Step 1: Write a failing trace parser test**

```python
def test_trace_summary_counts_only_matching_task(tmp_path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(
        '{"event":"response","task_id":"t1","input_tokens":30,"output_tokens":5,"duration_seconds":1.2}\n'
        '{"event":"response","task_id":"t2","input_tokens":999,"output_tokens":999,"duration_seconds":9}\n',
        encoding="utf-8",
    )
    usage = summarize_llm_trace(path, "t1")
    assert (usage.input_tokens, usage.output_tokens, usage.model_calls) == (30, 5, 1)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/evaluation/test_traces.py -q`

Expected: FAIL because `summarize_llm_trace` is missing.

- [ ] **Step 3: Implement bounded sanitized parsing**

Read one line at a time, ignore malformed/unrelated records, sum only `event=response`, and record `usage_estimated=True` when total token usage is zero or absent. Never include prompt/messages in the returned model.

- [ ] **Step 4: Verify PASS and commit**

Run: `python -m pytest tests/evaluation/test_traces.py -q`

Expected: PASS.

```bash
git add src/deepfix/evaluation/traces.py tests/evaluation/test_traces.py
git commit -m "feat: summarize evaluation token usage"
```

### Task 4: Clean-Copy Harness and Legacy Runner

**Files:**
- Create: `src/deepfix/evaluation/harness.py`
- Create: `src/deepfix/evaluation/legacy.py`
- Test: `tests/evaluation/test_harness.py`

**Interfaces:**
- Consumes: `EvaluationManifest`, `BugfixService` factory, `render_report`-independent TaskState facts.
- Produces: `CaseRunner` protocol, `EvaluationHarness.run_case(case, source, run_index)`, `LegacyLoopRunner.run(case, workspace, run_dir, budget)`.

- [ ] **Step 1: Write failing isolation and repeatability tests**

```python
def test_each_run_gets_a_fresh_copy(tmp_path, manifest, fake_runner) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    harness = EvaluationHarness(tmp_path / "runs", fake_runner)

    first = harness.run_case(manifest.cases[0], source, run_index=1)
    (first.workspace / "value.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = harness.run_case(manifest.cases[0], source, run_index=2)

    assert (second.workspace / "value.py").read_text(encoding="utf-8") == "VALUE = 1\n"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/evaluation/test_harness.py -q`

Expected: FAIL because harness classes do not exist.

- [ ] **Step 3: Implement the injectable runner boundary**

```python
class CaseRunner(Protocol):
    def run(
        self,
        case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        raise NotImplementedError
```

Use `shutil.copytree` into `runs/{case_id}/{run_index}/workspace`; reject a run directory that already exists. Before exposing the Workspace to the Agent, remove `correct_python_programs` and every gold patch/answer file. For `source_variant=correct_control`, first copy only the named algorithm from `correct_python_programs` over its `python_programs` counterpart, then remove the entire correct directory. Record preparation hashes outside the Workspace so the Agent cannot read the answer source. The legacy adapter constructs config/database/artifacts under `run_dir`, invokes the current service, automatically approves only policy action `allow`, and stops/persists a deterministic error if manual approval is required.

- [ ] **Step 4: Run harness tests**

Run: `python -m pytest tests/evaluation/test_harness.py -q`

Expected: PASS with no model/network calls.

- [ ] **Step 5: Commit harness**

```bash
git add src/deepfix/evaluation/harness.py src/deepfix/evaluation/legacy.py tests/evaluation/test_harness.py
git commit -m "feat: run isolated legacy loop evaluations"
```

### Task 5: Evaluation CLI, QuixBugs Manifest, and Frozen Baseline

**Files:**
- Create: `src/deepfix/evaluation/__main__.py`
- Create: `evaluations/quixbugs-python.json`
- Create: `evaluations/results/.gitkeep`
- Create: `tests/evaluation/test_cli.py`
- Modify after online run: `evaluations/results/legacy-baseline-summary.json`

**Interfaces:**
- Consumes: `load_manifest`, `EvaluationHarness`, `LegacyLoopRunner`, `aggregate_runs`.
- Produces: `python -m deepfix.evaluation baseline --manifest PATH --project PATH --runs N --output PATH --summary PATH`, sanitized aggregate JSON.

- [ ] **Step 1: Write failing CLI dry-run tests**

```python
def test_cli_dry_run_lists_cases_without_api_calls(tmp_path, capsys) -> None:
    code = main(["baseline", "--manifest", str(MANIFEST), "--project", str(tmp_path), "--dry-run"])
    assert code == 0
    assert "mergesort" in capsys.readouterr().out
```

- [ ] **Step 2: Verify RED, implement argparse CLI, and verify PASS**

Run: `python -m pytest tests/evaluation/test_cli.py -q`

Expected before implementation: FAIL. Expected after implementation: PASS.

The real `baseline` command must refuse to call a model unless `DEEPFIX_RUN_ONLINE=1`, write raw per-run artifacts below the user-supplied output directory, and write only the sanitized aggregate summary to the requested summary path.

- [ ] **Step 3: Add the initial fixed case manifest**

```json
{
  "schema_version": 1,
  "budget": {
    "max_input_tokens": 400000,
    "max_output_tokens": 60000,
    "max_wall_seconds": 900,
    "max_tool_calls": 80,
    "max_side_effects": 10
  },
  "cases": [
    {
      "case_id": "mergesort-buggy",
      "problem": "mergesort 的相关测试失败。只允许修改 python_programs/mergesort.py，并运行 python -m pytest python_testcases/test_mergesort.py -q --timeout=5 验证。",
      "allowed_paths": ["python_programs/mergesort.py"],
      "required_command": "python -m pytest python_testcases/test_mergesort.py -q --timeout=5",
      "expected_outcome": "fixed",
      "source_variant": "buggy"
    },
    {
      "case_id": "find-first-buggy",
      "problem": "find_first_in_sorted 的相关测试失败。只允许修改 python_programs/find_first_in_sorted.py，并运行 python -m pytest python_testcases/test_find_first_in_sorted.py -q --timeout=5 验证。",
      "allowed_paths": ["python_programs/find_first_in_sorted.py"],
      "required_command": "python -m pytest python_testcases/test_find_first_in_sorted.py -q --timeout=5",
      "expected_outcome": "fixed",
      "source_variant": "buggy"
    },
    {
      "case_id": "breadth-first-search-buggy",
      "problem": "breadth_first_search 的相关测试失败。只允许修改 python_programs/breadth_first_search.py，并运行 python -m pytest python_testcases/test_breadth_first_search.py -q --timeout=5 验证。",
      "allowed_paths": ["python_programs/breadth_first_search.py"],
      "required_command": "python -m pytest python_testcases/test_breadth_first_search.py -q --timeout=5",
      "expected_outcome": "fixed",
      "source_variant": "buggy"
    },
    {
      "case_id": "quicksort-timeout-buggy",
      "problem": "quicksort 的相关测试失败或超时。只允许修改 python_programs/quicksort.py，并运行 python -m pytest python_testcases/test_quicksort.py -q --timeout=5 验证。",
      "allowed_paths": ["python_programs/quicksort.py"],
      "required_command": "python -m pytest python_testcases/test_quicksort.py -q --timeout=5",
      "expected_outcome": "fixed",
      "source_variant": "buggy"
    },
    {
      "case_id": "find-first-correct-control",
      "problem": "find_first_in_sorted 的相关测试失败。只允许修改 python_programs/find_first_in_sorted.py，并运行 python -m pytest python_testcases/test_find_first_in_sorted.py -q --timeout=5 验证。",
      "allowed_paths": ["python_programs/find_first_in_sorted.py"],
      "required_command": "python -m pytest python_testcases/test_find_first_in_sorted.py -q --timeout=5",
      "expected_outcome": "not_reproduced",
      "source_variant": "correct_control"
    }
  ]
}
```

- [ ] **Step 4: Run all offline evaluation tests and Ruff**

Run: `python -m pytest tests/evaluation -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/evaluation tests/evaluation`

Expected: PASS.

- [ ] **Step 5: Commit the harness before spending API budget**

```bash
git add src/deepfix/evaluation tests/evaluation evaluations/quixbugs-python.json evaluations/results/.gitkeep
git commit -m "feat: add legacy loop evaluation CLI"
```

- [ ] **Step 6: Run and freeze the online legacy baseline**

PowerShell prerequisites:

```powershell
$env:DEEPFIX_RUN_ONLINE = "1"
$env:QUIXBUGS_PYTHON_ROOT = "C:\Users\17823\AppData\Local\Temp\deepfix-quixbugs-ee5e7feb-a7ec-4cee-8ced-cab7dd1a1f2d"
```

Run:

```powershell
python -m deepfix.evaluation baseline `
  --manifest evaluations/quixbugs-python.json `
  --project $env:QUIXBUGS_PYTHON_ROOT `
  --runs 3 `
  --output .deepfix-evaluation/legacy `
  --summary evaluations/results/legacy-baseline-summary.json
```

Expected: fifteen sanitized run records plus an aggregate containing `success_rate`, `false_fixed_rate`, `input_tokens`, `output_tokens`, `model_calls`, `tool_calls`, `wall_seconds`, and `successes_per_100k_tokens`.

- [ ] **Step 7: Verify and commit only the sanitized baseline**

Run: `python -m deepfix.evaluation validate-summary evaluations/results/legacy-baseline-summary.json`

Expected: `valid summary: loop=legacy cases=5 runs=15`, with no secret/prompt/source fields.

```bash
git add evaluations/results/legacy-baseline-summary.json
git commit -m "test: freeze legacy loop baseline"
```

## Plan 1 Completion Checkpoint

Stop for review. Do not start the trusted-execution plan until the manifest, hard resource envelope, outcome labels, and sanitized legacy summary have been reviewed.
