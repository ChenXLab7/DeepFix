# DeepFix Minimal Experiment Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation-only, single-main-model Experiment Loop that improves strategy and evidence use without changing the production `deepfix new` entry.

**Architecture:** Add minimal investigation-domain contracts and pure components for Blackboard projection, one-best-strategy planning, adaptive Experiment execution, deterministic-first criteria evaluation, reducer-only state commits, evidence-gated outcomes, progress fingerprints, and token reservation. Wrap the existing Deep Agents graph as the local Executor. The evaluation harness injects this loop directly; the legacy production path remains unchanged until A/B passes.

**Tech Stack:** Python 3.11+, Pydantic 2, SQLite/WAL, LangChain structured output, Deep Agents 0.7.x, existing Working Memory/Compaction/Research/Artifact/Receipt stores, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-26-deepfix-capability-oriented-agent-loop-design.md`

## Global Constraints

- Experiment intent labels are composable; no REPRODUCE→INVESTIGATE→EDIT→VERIFY phase state machine.
- Normal Planner output contains one best StrategyDecision. Alternative comparison is required only in REFLECT.
- Model output contains criterion claims and semantic candidates only. ResultBuilder attaches status, Receipt, Observation, file-change evidence, and test evidence.
- Deterministic Criteria never call an LLM. Semantic Criteria require validated independent provenance roots.
- Executor recommendation has no authority in OutcomeAdjudicator.
- Artifact Retrieval is auxiliary by default and does not increment Experiment count/stagnation.
- All changes in this plan are unreachable from production CLI except through the evaluation runner adapter.

## File Structure

### New production files

- `src/deepfix/investigation/experiments.py` — Experiment/criterion/strategy/result/assessment models and stable IDs.
- `src/deepfix/investigation/blackboard.py` — unique current-state projection from existing authority stores.
- `src/deepfix/investigation/evaluation.py` — ResultBuilder, deterministic criterion checks, semantic fallback, and OutcomeAdjudicator pure rules.
- `src/deepfix/investigation/reducer.py` — sole Experiment-state writer and provenance-root propagation.
- `src/deepfix/investigation/token_budget.py` — atomic reservation/settlement ledger.
- `src/deepfix/investigation/loop.py` — StrategyPlanner protocol, local Executor adapter, and DeepFixRepairLoop orchestration.
- `src/deepfix/evaluation/experiment.py` — evaluation-only runner adapter.

### Existing files modified

- `src/deepfix/investigation/models.py` — add Experiment references and progress fingerprint fields without duplicating facts.
- `src/deepfix/investigation/store.py` — add strategy/experiment/event tables and atomic reducer transaction.
- `src/deepfix/investigation/stagnation.py` — consume progress fingerprint and strategy signature.
- `src/deepfix/agent.py` — expose a local Experiment graph builder; do not switch the production builder.
- `src/deepfix/evaluation/legacy.py` — apply the same evaluation-only token reservation middleware used by Experiment runs.
- `src/deepfix/evaluation/__main__.py` — add `experiment` evaluation runner selection.

### Tests

- `tests/investigation/test_experiments.py`
- `tests/investigation/test_blackboard.py`
- `tests/investigation/test_experiment_evaluation.py`
- `tests/investigation/test_reducer.py`
- `tests/investigation/test_token_budget.py`
- `tests/investigation/test_loop.py`
- `tests/evaluation/test_experiment_runner.py`
- Modify: `tests/evaluation/test_harness.py`
- Modify: `tests/investigation/test_store.py`
- Modify: `tests/investigation/test_stagnation.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/evaluation/test_cli.py`

---

### Task 1: Experiment, Criterion, and Strategy Contracts

**Files:**
- Create: `src/deepfix/investigation/experiments.py`
- Create: `tests/investigation/test_experiments.py`

**Interfaces:**
- Consumes: `StrictModel`, `InvestigationCapability`, `ProvenanceRef`, stable SHA-256 identity conventions.
- Produces: `ExperimentIntent`, `ExperimentSpec`, deterministic/semantic SuccessCriterion, `StrategyDecision`, `ExecutorNarrativeResult`, `ExperimentResult`, `CriterionAssessment`, `ExperimentAssessment`, and stable ID builders.

- [ ] **Step 1: Write failing authority-boundary tests**

```python
def test_executor_claim_must_reference_declared_criterion(spec) -> None:
    narrative = ExecutorNarrativeResult(
        claimed_completed_criterion_ids=["unknown"],
        evidence_candidates=[],
        hypothesis_updates=[],
        remaining_questions=[],
        executor_recommendation="finish",
    )
    with pytest.raises(ValueError, match="unknown criterion"):
        validate_executor_narrative(spec, narrative)


def test_result_requires_system_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate({
            "experiment_id": "exp-1",
            "status": "completed",
            "executor_narrative": {},
            "changed_files": ["foo.py"],
        })
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/investigation/test_experiments.py -q`

Expected: FAIL because contracts do not exist.

- [ ] **Step 3: Implement discriminated criteria and adaptive intents**

```python
class DeterministicSuccessCriterion(StrictModel):
    criterion_id: str
    kind: Literal["deterministic"] = "deterministic"
    description: str
    check: DeterministicCriterionCheck
    required: bool = True


class SemanticSuccessCriterion(StrictModel):
    criterion_id: str
    kind: Literal["semantic"] = "semantic"
    description: str
    question: str
    required_evidence_types: set[EvidenceType]
    minimum_independent_root_count: int = Field(default=1, gt=0)
    required: bool = True


class ExperimentSpec(StrictModel):
    experiment_id: str
    task_id: str
    intents: set[ExperimentIntent] = Field(min_length=1)
    goal: str
    target_hypothesis_id: str | None = None
    evidence_gap_ids: list[str]
    success_criteria: list[SuccessCriterion] = Field(min_length=1)
    allowed_capabilities: set[InvestigationCapability]
    step_budget: int = Field(gt=0)
    model_call_budget: int = Field(gt=0)
    token_budget: int | None = Field(default=None, gt=0)
    time_budget_seconds: int = Field(gt=0)
    fallback: str
```

Define deterministic check variants for `test_passed`, `path_changed`, `path_unchanged`, `command_completed`, and `no_scope_violation`. Use Pydantic discriminators; never evaluate arbitrary expressions.

- [ ] **Step 4: Implement system/model envelope split and IDs**

`ExperimentResult` accepts only `executor_narrative`, system-derived `status`, `tool_receipt_ids`, `observation_ids`, `changed_file_evidence_ids`, and `test_evidence_ids`. Stable IDs hash task ID, StrategyDecision ID, normalized content, and Blackboard fingerprint.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/investigation/test_experiments.py -q`

Expected: PASS.

```bash
git add src/deepfix/investigation/experiments.py tests/investigation/test_experiments.py
git commit -m "feat: add experiment loop contracts"
```

### Task 2: Case Blackboard Projection

**Files:**
- Create: `src/deepfix/investigation/blackboard.py`
- Create: `tests/investigation/test_blackboard.py`

**Interfaces:**
- Consumes: TaskRepository, InvestigationStore, CompactionStore, WorkingMemoryStore, ResearchEvidenceStore, VerificationPolicy, OperationJournalStore.
- Produces: `CaseBlackboardView`, `CaseBlackboardBuilder.build(task_id) -> CaseBlackboardView`.

- [ ] **Step 1: Write failing authority/dedup tests**

```python
def test_blackboard_projects_same_evidence_once(blackboard_fixture) -> None:
    fixture = blackboard_fixture.with_duplicate_snapshot_reference("evidence-1")
    view = fixture.builder.build(fixture.task_id)
    assert [e.evidence_id for e in view.deterministic_evidence] == ["evidence-1"]


def test_working_memory_cannot_replace_test_exit_code(blackboard_fixture) -> None:
    fixture = blackboard_fixture.with_test(exit_code=1).with_memory_claim("pytest passed")
    view = fixture.builder.build(fixture.task_id)
    assert view.test_results[0].exit_code == 1
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/investigation/test_blackboard.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement bounded projection**

Build from authoritative stores by ID, project Snapshot only as history/reference, and expose task anchor, reproduction state, hypotheses, evidence gaps, last five Experiment summaries, code/test state, policy, conflicts, artifacts, and remaining budget. Reuse Protected Context dedup helpers rather than re-implementing ID precedence.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/investigation/test_blackboard.py tests/test_protected_context.py -q`

Expected: PASS.

```bash
git add src/deepfix/investigation/blackboard.py tests/investigation/test_blackboard.py
git commit -m "feat: project experiment case blackboard"
```

### Task 3: ResultBuilder and Deterministic-First Evidence Evaluation

**Files:**
- Create: `src/deepfix/investigation/evaluation.py`
- Create: `tests/investigation/test_experiment_evaluation.py`

**Interfaces:**
- Produces: `ExperimentResultBuilder.build`, `EvidenceEvaluator.evaluate`, `SemanticCriterionJudge` protocol, `adjudicate_outcome`.

- [ ] **Step 1: Write failing criterion authority tests**

```python
def test_executor_claim_does_not_complete_failed_pytest(evaluator_fixture) -> None:
    fixture = evaluator_fixture.test_criterion(exit_code=1, executor_claimed=True)
    assessment = fixture.evaluator.evaluate(fixture.spec, fixture.result)
    criterion = assessment.criterion_assessments[0]
    assert criterion.completed is False
    assert fixture.semantic_judge.calls == 0


def test_semantic_sources_count_independent_roots(evaluator_fixture) -> None:
    fixture = evaluator_fixture.semantic_criterion(min_roots=2)
    fixture.add_claim_and_summary_from_same_root("root-1")
    assessment = fixture.evaluator.evaluate(fixture.spec, fixture.result)
    assert assessment.criterion_assessments[0].completed is False
    assert fixture.semantic_judge.calls == 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/investigation/test_experiment_evaluation.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement system-built Result and criteria dispatch**

```python
class ExperimentResultBuilder:
    def build(self, spec, narrative, runtime_record) -> ExperimentResult:
        validate_executor_narrative(spec, narrative)
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status=runtime_record.status,
            executor_narrative=narrative,
            tool_receipt_ids=runtime_record.tool_receipt_ids,
            observation_ids=runtime_record.observation_ids,
            changed_file_evidence_ids=runtime_record.changed_file_evidence_ids,
            test_evidence_ids=runtime_record.test_evidence_ids,
        )
```

Dispatch each deterministic check to a pure registry function. For semantic criteria, validate evidence types and unique normalized root IDs before one structured main-model call. Record exact evidence IDs in CriterionAssessment.

- [ ] **Step 4: Implement evidence-gated outcomes**

`adjudicate_outcome` consumes VerificationPolicy and OracleEvaluation. FIXED requires every executable required Oracle, no higher-authority conflict, a real allowed change, and no unresolved/unknown side effect. Executor recommendation is absent from the function signature.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/investigation/test_experiment_evaluation.py tests/test_verification.py -q`

Expected: PASS.

```bash
git add src/deepfix/investigation/evaluation.py tests/investigation/test_experiment_evaluation.py
git commit -m "feat: evaluate experiment evidence deterministically"
```

### Task 4: Experiment Store and Reducer-Only Writes

**Files:**
- Create: `src/deepfix/investigation/reducer.py`
- Create: `tests/investigation/test_reducer.py`
- Modify: `src/deepfix/investigation/models.py`
- Modify: `src/deepfix/investigation/store.py`
- Modify: `tests/investigation/test_store.py`

**Interfaces:**
- Produces: strategy/experiment/event tables, `InvestigationStateReducer.commit(expected_version, result, assessment) -> InvestigationState`.

- [ ] **Step 1: Write failing idempotency and sole-writer tests**

```python
def test_reducer_replay_is_idempotent(reducer_fixture) -> None:
    first = reducer_fixture.commit()
    second = reducer_fixture.commit(expected_version=first.version)
    assert second.version == first.version
    assert reducer_fixture.store.count_experiment_events() == 1


def test_reducer_propagates_parent_roots(reducer_fixture) -> None:
    state = reducer_fixture.with_derived_claim(parents=["evidence-a", "evidence-b"]).commit()
    assert set(state.claims[-1].provenance_root_ids) == {"root-a", "root-b"}
```

- [ ] **Step 2: Verify RED, implement transaction, verify PASS**

Run before: `python -m pytest tests/investigation/test_reducer.py tests/investigation/test_store.py -q`

Expected: FAIL.

Add versioned tables and one transaction that inserts stable events, updates hypothesis/gap/claim projections, and advances materialized state only when expected version matches. Replay with identical event IDs returns current state; conflicting payload raises typed recovery error.

Run after: `python -m pytest tests/investigation/test_reducer.py tests/investigation/test_store.py -q`

Expected: PASS.

- [ ] **Step 3: Commit reducer**

```bash
git add src/deepfix/investigation/reducer.py src/deepfix/investigation/models.py src/deepfix/investigation/store.py tests/investigation/test_reducer.py tests/investigation/test_store.py
git commit -m "feat: commit experiment state through reducer"
```

### Task 5: Progress Fingerprint and Experiment-Level Stagnation

**Files:**
- Modify: `src/deepfix/investigation/stagnation.py`
- Modify: `src/deepfix/investigation/models.py`
- Modify: `tests/investigation/test_stagnation.py`

**Interfaces:**
- Produces: `progress_fingerprint(state) -> str`, `strategy_signature(decision) -> str`, Experiment-level stagnation ladder.

- [ ] **Step 1: Write failing weak-progress tests**

```python
def test_memory_and_audit_updates_do_not_change_progress_fingerprint(state) -> None:
    changed = state.model_copy(update={
        "version": state.version + 1,
        "memory_saved_generation": state.progress_generation,
    })
    assert progress_fingerprint(changed) == progress_fingerprint(state)


def test_closed_gap_changes_progress_fingerprint(state) -> None:
    changed = state.model_copy(update={"closed_evidence_gap_ids": ["gap-1"]})
    assert progress_fingerprint(changed) != progress_fingerprint(state)
```

- [ ] **Step 2: Verify RED, implement normalized fingerprint, verify PASS**

Include only closed gaps, independent roots, evidence-backed hypothesis transitions, code state hash, new test fingerprints, and new user-information IDs. Exclude time, sequence, memory saves, artifact reads, and weak claims.

Run: `python -m pytest tests/investigation/test_stagnation.py -q`

Expected: PASS after implementation.

- [ ] **Step 3: Commit stagnation update**

```bash
git add src/deepfix/investigation/stagnation.py src/deepfix/investigation/models.py tests/investigation/test_stagnation.py
git commit -m "feat: detect experiment strategy stagnation"
```

### Task 6: Atomic Token Reservation and Settlement

**Files:**
- Create: `src/deepfix/investigation/token_budget.py`
- Create: `tests/investigation/test_token_budget.py`
- Modify: `src/deepfix/evaluation/legacy.py`
- Modify: `tests/evaluation/test_harness.py`

**Interfaces:**
- Produces: `TokenBudgetStore.reserve`, `.settle`, `.charge_unknown`, `.available`; `TokenReservation`; evaluation-only `TokenBudgetMiddleware` shared by both loop runners.

- [ ] **Step 1: Write failing no-oversell tests**

```python
def test_reservation_blocks_call_that_cannot_fit(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    store.reserve("task-1", "call-1", input_tokens=80, output_tokens=20)
    with pytest.raises(TokenBudgetExhausted):
        store.reserve("task-1", "call-2", input_tokens=30, output_tokens=20)


def test_unknown_usage_charges_full_reservation(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    reservation = store.reserve("task-1", "call-1", input_tokens=80, output_tokens=20)
    store.charge_unknown(reservation.reservation_id)
    assert store.available("task-1").input_tokens == 20


def test_legacy_evaluation_runner_uses_same_reservation_policy(legacy_runner_factory) -> None:
    runner = legacy_runner_factory(input_cap=100, output_cap=40)
    runner.invoke_model(prompt_tokens=80, requested_output_tokens=20, actual=(70, 10))
    with pytest.raises(TokenBudgetExhausted):
        runner.invoke_model(prompt_tokens=30, requested_output_tokens=20)
```

- [ ] **Step 2: Verify RED, implement atomic ledger, verify PASS**

Use SQLite `BEGIN IMMEDIATE`; availability subtracts settled plus all outstanding reservations. Settle with Provider usage and release only positive difference. If actual usage exceeds reservation, persist breach and prevent subsequent reservations. `TokenBudgetMiddleware` estimates the final prompt after other prompt middleware, reserves before the handler, settles from AIMessage usage, and charges the reservation on unknown usage. The evaluation LegacyLoopRunner and ExperimentLoopRunner receive the same middleware factory and caps.

Run: `python -m pytest tests/investigation/test_token_budget.py tests/evaluation/test_harness.py -q`

Expected: PASS.

- [ ] **Step 3: Commit budget ledger**

```bash
git add src/deepfix/investigation/token_budget.py src/deepfix/evaluation/legacy.py tests/investigation/test_token_budget.py tests/evaluation/test_harness.py
git commit -m "feat: reserve experiment model tokens"
```

### Task 7: One-Best-Strategy Planner and Local Executor

**Files:**
- Create: `src/deepfix/investigation/loop.py`
- Create: `tests/investigation/test_loop.py`
- Modify: `src/deepfix/agent.py`
- Modify: `tests/test_agent.py`

**Interfaces:**
- Produces: `StrategyPlanner.plan(blackboard)`, `LocalAgentExperimentExecutor.execute(spec, context)`, `build_experiment_agent`; production `build_agent` unchanged.

- [ ] **Step 1: Write failing Planner/Executor contract tests**

```python
def test_normal_planner_returns_one_strategy(fake_model, blackboard) -> None:
    planner = StrategyPlanner(fake_model)
    decision = planner.plan(blackboard)
    assert decision.decision_type == "run_experiment"
    assert decision.reflection is None


def test_reflect_requires_two_alternatives(fake_model, stagnated_blackboard) -> None:
    planner = StrategyPlanner(fake_model)
    decision = planner.plan(stagnated_blackboard)
    assert decision.reflection is not None
    assert len(decision.reflection.alternatives) >= 2


def test_simple_experiment_can_read_edit_and_verify(local_executor_fixture) -> None:
    result = local_executor_fixture.run(intents={"investigate", "edit", "verify"})
    assert result.status == "completed"
    assert result.changed_file_evidence_ids
    assert result.test_evidence_ids
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/investigation/test_loop.py tests/test_agent.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement dynamic role prompts and bounded execution**

Planner prompt receives Blackboard and budget only. Executor prompt receives Task Anchor, ExperimentSpec, relevant evidence/artifacts, runtime feedback, and allowed capabilities. The same main model is used for both. Before each model call, reserve tokens; after response, settle actual usage. Approval resumes the same Experiment ID.

- [ ] **Step 4: Keep production builder unchanged and run tests**

Add a test asserting `build_agent` still creates the legacy graph and only `build_experiment_agent` constructs the new adapter.

Run: `python -m pytest tests/investigation/test_loop.py tests/test_agent.py -q`

Expected: PASS.

- [ ] **Step 5: Commit local loop building blocks**

```bash
git add src/deepfix/investigation/loop.py src/deepfix/agent.py tests/investigation/test_loop.py tests/test_agent.py
git commit -m "feat: add bounded local experiment executor"
```

### Task 8: Evaluation-Only DeepFixRepairLoop Vertical Workflow

**Files:**
- Modify: `src/deepfix/investigation/loop.py`
- Create: `src/deepfix/evaluation/experiment.py`
- Create: `tests/evaluation/test_experiment_runner.py`
- Modify: `src/deepfix/evaluation/__main__.py`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**
- Produces: `DeepFixRepairLoop.run(task_id) -> RepairLoopResult`, `ExperimentLoopRunner` for evaluation harness only.

- [ ] **Step 1: Write failing vertical workflows**

```python
def test_vertical_loop_repairs_and_verifies(loop_fixture) -> None:
    result = loop_fixture.simple_bug().run()
    assert result.outcome == "fixed"
    assert result.experiments[0].spec.intents == {"investigate", "edit", "verify"}
    assert result.oracle_evaluation.all_required_passed is True


def test_vertical_loop_reports_not_reproduced(loop_fixture) -> None:
    result = loop_fixture.correct_project().run()
    assert result.outcome == "not_reproduced"
    assert result.changed_file_evidence_ids == []


def test_vertical_loop_emits_high_level_events(loop_fixture) -> None:
    result = loop_fixture.simple_bug().run()
    assert [event.event_type for event in result.progress_events] == [
        "strategy_planned",
        "experiment_started",
        "experiment_completed",
        "experiment_assessed",
        "outcome_adjudicated",
    ]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/evaluation/test_experiment_runner.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement the outer loop**

For each iteration: build Blackboard, plan one decision, validate/run one adaptive Experiment, build/evaluate Result, reduce once, adjudicate, and either return or continue. Stop on hard budget, unresolved recovery, ask-user, blocked, or final outcome. Auxiliary Artifact reads attach to the current decision/experiment and do not increment Experiment count. Persist bounded `strategy_planned`, `experiment_started`, `experiment_tool_called`, `experiment_timed_out`, `experiment_completed`, `experiment_assessed`, `hypothesis_updated`, `evidence_gap_closed`, `strategy_changed`, `outcome_adjudicated`, and `task_paused` events using the existing Investigation event log and stable IDs.

- [ ] **Step 4: Add evaluation CLI selection without production switch**

`python -m deepfix.evaluation experiment` selects `ExperimentLoopRunner`. Do not add `--loop` to `deepfix new`, do not change `cli.main`, and add an assertion to `tests/evaluation/test_cli.py` that production parser has no experiment-loop flag.

- [ ] **Step 5: Run all offline vertical/regression tests**

Run: `python -m pytest tests/investigation tests/evaluation tests/test_agent.py tests/test_service.py tests/test_context.py tests/compaction -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/investigation src/deepfix/evaluation tests/investigation tests/evaluation`

Expected: PASS.

- [ ] **Step 6: Commit evaluation-only loop**

```bash
git add src/deepfix/investigation/loop.py src/deepfix/evaluation/experiment.py src/deepfix/evaluation/__main__.py tests/evaluation/test_experiment_runner.py tests/evaluation/test_cli.py
git commit -m "feat: add evaluation-only experiment repair loop"
```

## Plan 3 Completion Checkpoint

Stop for review. Demonstrate offline fake-model runs for FIXED, NOT_REPRODUCED, failed first hypothesis followed by REFLECT, timeout feedback, and required-Oracle conflict. Confirm `deepfix new` still resolves to the legacy loop.
