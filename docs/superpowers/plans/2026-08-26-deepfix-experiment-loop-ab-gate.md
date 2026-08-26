# DeepFix Experiment Loop A/B Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide with fixed-resource evidence whether the Experiment Loop should replace the production Tool-Call Loop.

**Architecture:** Pre-register metric thresholds and the exact resource envelope before any new-loop online run, execute deterministic fault injection separately from stochastic model benchmarks, run paired repeated QuixBugs cases, and generate a sanitized comparison plus an explicit pass/fail decision. This plan never switches production behavior.

**Tech Stack:** Python 3.11+, existing evaluation harness, JSON, pytest fault injection, Provider usage metadata, deterministic statistics, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-26-deepfix-capability-oriented-agent-loop-design.md`

## Global Constraints

- Legacy and Experiment runs use the same model name, API endpoint, model parameters, case source, prompt-visible task, Python executable, and pre-registered hard resource caps.
- Both runners use the same system-created trusted-corpus approval grant bound to the committed QuixBugs manifest/source hashes; this grant is evaluation-only and cannot be requested by model output.
- Equal Model Call counts are not required. Input/output tokens, calls, tools, wall time, and success/100k Token are all reported.
- Fault-injection invariants require zero violations in covered deterministic cases. Real-model reliability uses rates, not an impossible lifetime zero-error claim.
- Thresholds are committed before the first online Experiment run and cannot be edited after observing its results.
- A pass writes a recommendation only. Production migration requires a new reviewed plan.
- A fail or inconclusive result stops work; it does not trigger more abstractions, models, or Subagents.

## File Structure

### New files

- `evaluations/experiment-loop-gate.json` — pre-registered cases, resource envelope, metric formulas, and thresholds.
- `src/deepfix/evaluation/compare.py` — paired comparison and decision function.
- `tests/evaluation/test_compare.py` — deterministic gate calculations.
- `tests/evaluation/test_fault_injection.py` — journal, confinement, Receipt, Oracle, and budget invariants.
- Generated after runs: `evaluations/results/experiment-loop-summary.json`.
- Generated after runs: `evaluations/results/legacy-control-summary.json`.
- Generated after comparison: `evaluations/results/experiment-loop-ab-report.json`.
- Generated after comparison: `evaluations/results/experiment-loop-decision.md`.

### Existing files modified

- `src/deepfix/evaluation/__main__.py` — `compare` and `validate-gate` commands.
- `tests/evaluation/test_cli.py` — command coverage and immutable-threshold checks.

---

### Task 1: Pre-Register the Gate

**Files:**
- Create: `evaluations/experiment-loop-gate.json`
- Modify: `src/deepfix/evaluation/__main__.py`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**
- Consumes: frozen historical `evaluations/results/legacy-baseline-summary.json`, existing case manifest.
- Produces: validated immutable GateDefinition and its SHA-256 fingerprint.

- [ ] **Step 1: Write a failing gate validation test**

```python
def test_gate_rejects_resource_caps_different_from_baseline(tmp_path, baseline) -> None:
    gate = valid_gate_dict(baseline)
    gate["budget"]["max_input_tokens"] += 1
    path = write_json(tmp_path / "gate.json", gate)
    assert main(["validate-gate", str(path), "--baseline", str(baseline.path)]) == 2
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/evaluation/test_cli.py -q -k gate`

Expected: FAIL because gate validation is absent.

- [ ] **Step 3: Implement strict GateDefinition, preregistration, and validation**

The `preregister` command constructs the committed JSON with real file hashes:

```python
gate = GateDefinition(
    schema_version=1,
    case_manifest_sha256=sha256_file(Path("evaluations/quixbugs-python.json")),
    historical_legacy_summary_sha256=sha256_file(
        Path("evaluations/results/legacy-baseline-summary.json")
    ),
    runs_per_case=3,
    budget=EvaluationBudget(
        max_input_tokens=300_000,
        max_output_tokens=60_000,
        max_wall_seconds=900,
        max_tool_calls=80,
        max_side_effects=10,
    ),
    thresholds=GateThresholds(
        minimum_newly_solved_stable_failures=3,
        minimum_success_rate_delta=0.15,
        maximum_false_fixed_rate=0.05,
        maximum_false_fixed_rate_ratio_to_legacy=0.5,
        minimum_wrong_hypothesis_recovery_delta=0.15,
        maximum_token_multiplier=1.5,
        fault_invariant_violations=0,
    ),
)
write_json_atomic(Path("evaluations/experiment-loop-gate.json"), gate)
```

Validation recomputes both hashes and rejects mismatch. The historical baseline anchors pre-registration but is not the primary control after Plan 2 changes shared infrastructure.

- [ ] **Step 4: Run validation tests and commit before online execution**

Run: `python -m pytest tests/evaluation/test_cli.py -q -k gate`

Expected: PASS.

Run: `python -m deepfix.evaluation preregister --manifest evaluations/quixbugs-python.json --baseline evaluations/results/legacy-baseline-summary.json --output evaluations/experiment-loop-gate.json`

Expected: `wrote immutable gate: evaluations/experiment-loop-gate.json`.

Run: `python -m deepfix.evaluation validate-gate evaluations/experiment-loop-gate.json --baseline evaluations/results/legacy-baseline-summary.json`

Expected: `valid immutable gate: runs=3 cases=5`.

```bash
git add evaluations/experiment-loop-gate.json src/deepfix/evaluation/__main__.py tests/evaluation/test_cli.py
git commit -m "test: preregister experiment loop gate"
```

### Task 2: Deterministic Fault-Injection Gate

**Files:**
- Create: `tests/evaluation/test_fault_injection.py`

**Interfaces:**
- Consumes: TaskWorkspace, WorkspacePathPolicy, WorkspaceCommandPolicy, OperationJournalStore/Reconciler, VerificationPolicy, TokenBudgetStore.
- Produces: deterministic zero-violation test matrix independent of model behavior.

- [ ] **Step 1: Add the fault matrix**

```python
@pytest.mark.parametrize("fault", [
    "crash_after_file_edit_before_receipt",
    "crash_after_receipt_before_state_commit",
    "parallel_identical_side_effect_replay",
    "unknown_command_exit",
])
def test_side_effect_fault_never_replays(fault, fault_harness) -> None:
    result = fault_harness.run(fault)
    assert result.side_effect_handler_calls == 1
    assert result.duplicate_side_effects == 0


@pytest.mark.parametrize("escape", [
    "../outside.txt",
    "absolute-outside-path",
    "symlink-outside",
    "junction-outside",
    "git-config-global",
    "pip-install-global",
])
def test_workspace_escape_is_denied(escape, confinement_harness) -> None:
    result = confinement_harness.attempt(escape)
    assert result.allowed is False
    assert result.outside_hash_before == result.outside_hash_after
```

Also cover required-Oracle downgrade, targeted-pass/relevant-full-fail, unknown operation used as success evidence, parallel Token reservation oversell, and weak-Claim progress reset.

- [ ] **Step 2: Run the fault gate**

Run: `python -m pytest tests/evaluation/test_fault_injection.py -q`

Expected: PASS with zero duplicate side effects, outside writes, Oracle bypasses, budget oversells, and false progress resets.

- [ ] **Step 3: Commit fault gate**

```bash
git add tests/evaluation/test_fault_injection.py
git commit -m "test: enforce experiment loop fault invariants"
```

### Task 3: Paired Comparison and Decision Logic

**Files:**
- Create: `src/deepfix/evaluation/compare.py`
- Create: `tests/evaluation/test_compare.py`

**Interfaces:**
- Produces: `compare_summaries(gate, legacy, experiment) -> ABComparison`, `decide_migration(comparison) -> GateDecision`.

- [ ] **Step 1: Write failing all-thresholds-must-pass tests**

```python
def test_one_failed_threshold_fails_entire_gate(valid_comparison) -> None:
    comparison = valid_comparison.model_copy(update={"token_multiplier": 1.51})
    decision = decide_migration(comparison)
    assert decision.status == "fail"
    assert "maximum_token_multiplier" in decision.failed_thresholds


def test_missing_provider_usage_is_inconclusive(valid_comparison) -> None:
    comparison = valid_comparison.model_copy(update={"usage_estimated_runs": 1})
    decision = decide_migration(comparison)
    assert decision.status == "inconclusive"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/evaluation/test_compare.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement metric comparison without hidden weights**

Calculate success-rate delta, both false-FIXED denominators, newly solved stable failures, wrong-hypothesis recovery delta, total input/output token ratio, model/tool-call ratio, wall-time ratio, and success/100k Token delta. Gate status is `pass` only when every required threshold passes and all runs use real Provider usage; missing/corrupt runs produce `inconclusive`.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/evaluation/test_compare.py -q`

Expected: PASS.

```bash
git add src/deepfix/evaluation/compare.py tests/evaluation/test_compare.py
git commit -m "feat: compare loop evaluation results"
```

### Task 4: Run Paired Legacy Control and Experiment Treatment

**Files:**
- Generate: `evaluations/results/experiment-loop-summary.json`
- Generate: `evaluations/results/legacy-control-summary.json`
- Generate: `evaluations/results/experiment-loop-ab-report.json`
- Generate: `evaluations/results/experiment-loop-decision.md`
- Modify: `src/deepfix/evaluation/__main__.py`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**
- Consumes: immutable gate, historical baseline, current LegacyLoopRunner, ExperimentLoopRunner.
- Produces: sanitized new summary, comparison, and decision.

- [ ] **Step 1: Add and test the compare CLI**

Run before implementation: `python -m pytest tests/evaluation/test_cli.py -q -k compare`

Expected: FAIL.

Implement:

```text
python -m deepfix.evaluation compare
  --gate evaluations/experiment-loop-gate.json
  --legacy evaluations/results/legacy-control-summary.json
  --experiment evaluations/results/experiment-loop-summary.json
  --report evaluations/results/experiment-loop-ab-report.json
  --decision evaluations/results/experiment-loop-decision.md
```

Run after implementation: `python -m pytest tests/evaluation/test_cli.py -q -k compare`

Expected: PASS.

- [ ] **Step 2: Run the current legacy loop as the paired control**

Plan 2 changed shared Workspace/Journal/Test Evidence behavior, so rerun the legacy loop on the current code before the treatment. Use the same trusted-corpus grant, source hashes, cases, model, and caps:

```powershell
$env:DEEPFIX_RUN_ONLINE = "1"
$env:QUIXBUGS_PYTHON_ROOT = "C:\Users\17823\AppData\Local\Temp\deepfix-quixbugs-ee5e7feb-a7ec-4cee-8ced-cab7dd1a1f2d"
python -m deepfix.evaluation baseline `
  --manifest evaluations/quixbugs-python.json `
  --gate evaluations/experiment-loop-gate.json `
  --project $env:QUIXBUGS_PYTHON_ROOT `
  --runs 3 `
  --output .deepfix-evaluation/legacy-control `
  --summary evaluations/results/legacy-control-summary.json
```

Expected: the same five cases and fifteen runs. The historical Plan 1 baseline remains in the report as a secondary reference; `legacy-control-summary.json` is the primary A/B control.

- [ ] **Step 3: Run the online Experiment benchmark**

PowerShell:

```powershell
$env:DEEPFIX_RUN_ONLINE = "1"
$env:QUIXBUGS_PYTHON_ROOT = "C:\Users\17823\AppData\Local\Temp\deepfix-quixbugs-ee5e7feb-a7ec-4cee-8ced-cab7dd1a1f2d"
python -m deepfix.evaluation experiment `
  --manifest evaluations/quixbugs-python.json `
  --gate evaluations/experiment-loop-gate.json `
  --project $env:QUIXBUGS_PYTHON_ROOT `
  --runs 3 `
  --output .deepfix-evaluation/experiment `
  --summary evaluations/results/experiment-loop-summary.json
```

Expected: exactly the same case/run count and hard caps as the gate; any cap breach marks that run invalid rather than silently extending it.

The summary must record `confinement_level=guarded_local` and the trusted-corpus grant fingerprint. A passing A/B result does not claim OS-level strict confinement; production migration must either provide a strict runner or retain explicit approval for project-code execution.

- [ ] **Step 4: Generate the comparison and decision**

Run:

```powershell
python -m deepfix.evaluation compare `
  --gate evaluations/experiment-loop-gate.json `
  --legacy evaluations/results/legacy-control-summary.json `
  --experiment evaluations/results/experiment-loop-summary.json `
  --report evaluations/results/experiment-loop-ab-report.json `
  --decision evaluations/results/experiment-loop-decision.md
```

Expected: exit code 0 for `pass`, 2 for `fail`, or 3 for `inconclusive`; all three statuses still write the report and decision file.

- [ ] **Step 5: Verify sanitized outputs and full regression**

Run: `python -m deepfix.evaluation validate-summary evaluations/results/experiment-loop-summary.json`

Expected: valid, no prompt/source/secret fields.

Run: `python -m deepfix.evaluation validate-summary evaluations/results/legacy-control-summary.json`

Expected: valid, same cases, runs, grant fingerprint, model, and hard caps as the Experiment summary.

Run: `python -m pytest -q`

Expected: full offline suite PASS.

Run: `python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Commit evidence only**

```bash
git add src/deepfix/evaluation/__main__.py tests/evaluation/test_cli.py evaluations/results/legacy-control-summary.json evaluations/results/experiment-loop-summary.json evaluations/results/experiment-loop-ab-report.json evaluations/results/experiment-loop-decision.md
git commit -m "test: evaluate experiment loop migration gate"
```

## Plan 4 Completion Checkpoint

Stop and review `experiment-loop-decision.md`.

- If `pass`: create a new spec-aligned production migration plan that removes replaced Tool-level gates after switching the entry.
- If `fail`: keep production on the legacy loop, inspect paired traces, and remove or revise only the demonstrated ineffective mechanism.
- If `inconclusive`: fix measurement integrity or add runs under the unchanged pre-registered gate; do not change thresholds.
