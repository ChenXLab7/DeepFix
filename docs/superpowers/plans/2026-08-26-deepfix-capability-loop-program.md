# DeepFix Capability-Oriented Loop Implementation Program

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this program plan-by-plan. Each child plan uses checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and deliver the frozen capability-oriented Experiment Loop without maintaining two production loops or adding deferred Multi-Agent/model-routing abstractions.

**Architecture:** Execute four independently reviewable plans in strict order. First freeze the legacy baseline; next build trustworthy workspace, side-effect, test-evidence, and oracle foundations; then build a minimal evaluation-only Experiment Loop; finally run the pre-registered A/B gate. Production migration is a separate future plan created only if the gate passes.

**Tech Stack:** Python 3.11+, Pydantic 2, SQLite/WAL, LangChain middleware, LangGraph/Deep Agents 0.7.x, pytest 8+, Ruff 0.12+, PowerShell-compatible CLI

**Spec:** `docs/superpowers/specs/2026-08-26-deepfix-capability-oriented-agent-loop-design.md`

## Global Constraints

- The frozen chain is `CaseBlackboard → StrategyPlanner → ExperimentExecutor → EvidenceEvaluator → InvestigationStateReducer → OutcomeAdjudicator`.
- Experiment is adaptive-granularity work, not a new reasoning-stage state machine.
- Tool Receipt and deterministic evidence own file, command, test, approval, and side-effect facts.
- Only the Reducer writes Investigation State; only the Evaluator completes SuccessCriterion.
- First phase uses the current main model and compaction model only; no Subagent, Model Router, separate Planner/Evaluator key, or precision Token scheduler beyond required reservation accounting.
- Preserve all unrelated dirty-worktree changes. Each task stages and commits only the files it lists.
- Every production change follows red-green-refactor and passes focused pytest plus Ruff on touched files.

## Ordered Plans and Review Gates

1. `2026-08-26-deepfix-loop-baseline-evaluation.md`
   - Deliverable: reproducible legacy-loop QuixBugs corpus, metric definitions, trace ingestion, and frozen baseline result.
   - Review gate: case manifest, oracle labels, resource envelope, and baseline summary are approved before safety infrastructure begins.
2. `2026-08-26-deepfix-trusted-execution-foundation.md`
   - Deliverable: isolated Task Workspace, canonical path policy, guarded command confinement, Operation Journal/recovery, enriched test evidence, and frozen VerificationPolicy.
   - Review gate: fault injection proves no duplicate side effect; required-oracle and path-escape tests pass before model autonomy increases.
3. `2026-08-26-deepfix-minimal-experiment-loop.md`
   - Deliverable: evaluation-only single-agent adaptive Experiment Loop with deterministic-first evaluation, provenance roots, progress fingerprint, token reservations, and evidence-gated outcomes.
   - Review gate: offline vertical workflows pass; production `deepfix new` still uses the legacy entry.
4. `2026-08-26-deepfix-experiment-loop-ab-gate.md`
   - Deliverable: pre-registered, fixed-resource paired legacy-control/experiment results and a written `pass` or `fail` migration decision. Plan 1 remains a historical pre-foundation baseline.
   - Review gate: on `pass`, write a new production-migration plan; on `fail`, stop and simplify based on traces.

## Explicit Stop Rule

Do not write or execute a production migration task in this program. The A/B gate is a real decision boundary, not a ceremonial final task. A failing or inconclusive gate ends this program without enabling the new loop in `deepfix new`.

## Frozen Spec Coverage

| Spec area | Implemented by |
|---|---|
| Legacy baseline and resource metrics | Plan 1 Tasks 1-5 |
| Workspace, canonical paths, Shell confinement | Plan 2 Tasks 1-2, 7 |
| Operation Journal and reconciliation | Plan 2 Tasks 3-5 |
| Test Evidence and VerificationPolicy | Plan 2 Task 6 |
| Strategy/Experiment/Criteria contracts | Plan 3 Task 1 |
| Blackboard authority and dedup | Plan 3 Task 2 |
| Deterministic-first evaluation and Outcome | Plan 3 Task 3 |
| Reducer-only state writes and provenance roots | Plan 3 Task 4 |
| Progress fingerprint and stagnation | Plan 3 Task 5 |
| Token reservation and settlement | Plan 3 Task 6 |
| Same-model Planner/Executor and dynamic prompts | Plan 3 Task 7 |
| Vertical loop, recovery, auxiliary Artifact access, events | Plan 3 Task 8 |
| Engineering invariants, A/B metrics, decision gate | Plan 4 Tasks 1-4 |
| Multi-Agent/model routing | Explicitly deferred; no implementation task |
| Production migration and old-gate deletion | New plan only after Plan 4 returns `pass` |
