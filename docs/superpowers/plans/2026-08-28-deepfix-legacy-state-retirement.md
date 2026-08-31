# DeepFix Legacy State Retirement and Pure Adjudication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task with the review checkpoints below. Use `superpowers:test-driven-development` for every production change and `superpowers:verification-before-completion` before claiming a task or checkpoint complete.

**Goal:** Finish the state-authority migration by making the five bounded repositories the only DeepFix business authorities, retaining DeepAgents/LangGraph as the Agent Loop and navigation runtime, and deleting the legacy Phase, Working Memory, giant `TaskState`, duplicated projections, and old Store write paths.

**Architecture:** Migrate readers before deleting compatibility code. First move context telemetry out of Working Memory, then create repository-native adjudication, reporting, context, and navigation projections. After those readers are green, remove the Phase control plane and public progress-writing tools, delete Working Memory, replace the mutable service aggregate with task-scoped repository reads plus request-local runtime values, and finally isolate legacy payload parsing as migration-only code. No new planner, task graph, manager, model router, persistence system, or second business-state aggregate is introduced.

**Tech Stack:** Python 3.11+, DeepAgents 0.7.x, LangChain 1.3.x, LangGraph 1.2.x, Pydantic 2, SQLite/WAL, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-27-deepfix-architecture-audit.md`

**Program:** `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md` Plan 4

**Prerequisite:** Plan 3 is complete at commit `1ad8885` with migration cutover hardening at `5521219`.

**Status:** READY FOR REVIEW — implementation has not started.

## Frozen Authority Rules

1. **DeepAgents/LangGraph own orchestration.** They own the Agent Loop, Graph Messages, native Todo, Checkpoint, interrupt, and resume.
2. **Todo owns navigation only.** Todo answers what the model intends to do next; it is not Evidence, permission, task lifecycle, or an outcome oracle.
3. **TaskRepository owns task-scoped definition and lifecycle.** `TaskDefinition.original_problem` is an immutable snapshot derived from `original_message_id`; the Agent cannot rewrite it.
4. **Domain repositories own current facts.** Investigation owns hypotheses/questions, Evidence owns accepted facts, Execution owns operations/receipts/approvals, and History owns compressed history plus context telemetry.
5. **Snapshot owns history, not truth.** Historical semantic items retain provenance and cannot override a current domain record with the same semantic ID.
6. **OutcomeAdjudicator is pure.** It reads repository views and code-state facts, returns a decision, and persists only outcome plus supporting Evidence/Operation IDs.
7. **Reports are projections.** `TaskReportView` is assembled at read time and is never stored as another aggregate.
8. **No compatibility dual-write after cutover.** Once a final reader is enabled and its migration gate passes, the corresponding legacy writer is removed in the same task.
9. **No authority hidden in prompts.** Prompts may advise the model but cannot determine task status, evidence trust, tool permission, or completion.
10. **No expensive online acceptance.** This plan ends at implementation complete plus focused/core offline tests; it does not run QuixBugs or paid A/B evaluation.

## Non-Goals

- Do not add Task Graph Harness, Multi-Agent dispatch, Planner/Executor roles, background worktrees, or another state machine.
- Do not create `WorkingMemory V2`, `TaskState V2`, a generic Store registry, a generic Repository base class, or a new database.
- Do not copy Evidence, Receipt, Journal, Investigation, Todo, or conversation bodies into TaskRepository.
- Do not replace DeepAgents native Todo or LangGraph Checkpoint persistence.
- Do not weaken approval, workspace confinement, Receipt idempotency, Operation recovery, required-oracle, compaction atomicity, stable message ID, or overflow retry behavior.
- Do not remove historical tables or artifacts until migrated restore tests pass. After cutover they remain read-only rollback inputs for the bounded migration window.

---

## Final Production Shape

```text
DeepAgents / LangGraph
├── Agent Loop + Graph Messages + Checkpoint
├── native Todo + write_todos
└── interrupt / resume
          │
          ▼
DeepFix middleware and tools
├── approval + workspace confinement
├── Receipt / Operation recovery
├── Evidence collection + research acquisition
├── repository-native Todo feedback
└── one Protected Context + Compaction coordinator
          │
          ▼
deepfix.db
├── TaskRepository          definition, lifecycle, verification, adjudication, budget
├── InvestigationRepository hypotheses, unresolved questions
├── EvidenceRepository      deterministic, research, semantic Evidence
├── ExecutionRepository     operations, receipts, approvals, integrity
└── HistoryRepository       snapshots, coverage, failures, context telemetry

Read-time only
├── OutcomeAdjudicationInput → OutcomeAdjudicator → AdjudicationDecision IDs
├── TaskReportView → Markdown report
└── ProtectedContext projection (current domain > historical snapshot)
```

The final design deliberately has no `AgentPhase`, `WorkingMemoryStore`, mutable fact-bearing `TaskState`, `LegacyNavigationFeedbackSource`, or independent old Store facade in a production construction path.

## File Structure

### New files

- `src/deepfix/task_domain/adjudication.py` — pure adjudication input builder and deterministic outcome rules; no persistence or model call.
- `src/deepfix/task_domain/runtime.py` — minimal non-authoritative service return model containing task identity, lifecycle, pending interrupt actions, and latest decision reference only. It must not contain conversation, Evidence, hypotheses, changes, tests, approvals, recovery bodies, or context history.
- `src/deepfix/task_domain/legacy_payload.py` — migration-only parser for historical giant `TaskState` JSON; never imported by Agent, service, middleware, reporting, or CLI runtime paths.
- `tests/task_domain/test_pure_adjudication.py` — final adjudicator authority and false-FIXED tests.
- `tests/task_domain/test_runtime_boundary.py` — proves the service result cannot become a second fact store.
- `tests/domain_repositories/test_context_telemetry.py` — History-owned context metric tests and Working Memory migration tests.
- `tests/test_repository_native_context.py` — current-domain projection, provenance, deduplication, and single-injection tests.
- `tests/test_repository_native_reporting.py` — read-time `TaskReportView` reconstruction tests.
- `tests/test_legacy_authority_retirement.py` — production import/construction/source-search retirement gate.

### Existing files modified

- `src/deepfix/domain_repositories/history.py` — own context telemetry and legacy metrics backfill alongside history-only Snapshot records.
- `src/deepfix/domain_repositories/evidence.py` — expose typed, immutable read views needed by verification, reporting, and context without copying persistence authority.
- `src/deepfix/domain_repositories/investigation.py` — expose current hypothesis/question view used by context and navigation.
- `src/deepfix/domain_repositories/execution.py` — expose approvals, receipts, incomplete operations, and integrity through a read-only report/adjudication view.
- `src/deepfix/domain_repositories/migration.py` — migrate Working Memory facts/metrics and old task payloads; validate before final cutover.
- `src/deepfix/task_domain/models.py` — keep immutable definition, lifecycle, adjudication, verification, and budget models only.
- `src/deepfix/task_domain/repository.py` — add no fact tables; persist only final decision and existing task-domain records.
- `src/deepfix/investigation/evaluation.py` — delegate final outcome authority to the pure task-domain adjudicator; retain only experiment/evidence assessment helpers that are still used.
- `src/deepfix/reporting.py` — define/build/render the ephemeral `TaskReportView` from the five repositories.
- `src/deepfix/protected_context.py` — build one ID-deduplicated current projection from final repositories and historical references.
- `src/deepfix/compaction/coordinator.py` — consume History telemetry/current repository inputs rather than Working Memory.
- `src/deepfix/compaction/middleware.py` — use History telemetry and remain the sole Protected Context injection point.
- `src/deepfix/compaction/snapshot.py` — merge old Snapshot plus current domain views, not Working Memory.
- `src/deepfix/compaction/models.py` — remove `working_memory_version` and obsolete Working Memory provenance kinds after migration tests cover old payload parsing.
- `src/deepfix/navigation/feedback.py` — replace `LegacyNavigationFeedbackSource` with repository-native projections.
- `src/deepfix/prompting.py` and `src/deepfix/prompts.py` — remove Phase prompts and store-dependent phase selection; retain one stable repair prompt plus bounded Todo feedback.
- `src/deepfix/investigation/middleware.py` — remove Phase tool visibility and navigation permits while preserving trusted tool boundaries.
- `src/deepfix/investigation/models.py`, `src/deepfix/investigation/store.py`, `src/deepfix/investigation/coordinator.py`, and `src/deepfix/investigation/tools.py` — remove Phase and `continue_investigation`; retain hypothesis/question and reliability behavior only.
- `src/deepfix/context.py` — remove `save_progress`, Working Memory rendering, and unused duplicate context middleware.
- `src/deepfix/agent.py` — construct only DeepAgents middleware, the five repositories, trusted tools, one context/compaction path, and repository-native Todo feedback.
- `src/deepfix/service.py` — operate by `task_id` and repository views rather than mutating giant `TaskState`.
- `src/deepfix/cli.py` — use minimal service results and `TaskReportView`; construct no old Store facade.
- `src/deepfix/models.py` — remove giant `TaskState`, `TaskStatus`, duplicated records, `ContextMetrics`, and fact-bearing `RepairOutcome`; keep only genuinely shared non-authority models, if any remain.
- `src/deepfix/persistence.py`, `src/deepfix/memory.py`, `src/deepfix/compaction/store.py`, `src/deepfix/research/store.py`, `src/deepfix/investigation/phase.py`, and old Receipt/Journal facade definitions — delete or reduce to migration-only modules after all production imports are gone.
- Existing focused tests under `tests/compaction`, `tests/investigation`, `tests/research`, `tests/artifact_retrieval`, `tests/navigation`, `tests/task_domain`, `tests/domain_repositories`, `tests/test_agent.py`, `tests/test_service.py`, `tests/test_cli.py`, and `tests/test_reporting.py` — migrate fixtures from giant state objects to repositories and Graph state.

## Stable Final Interfaces

These are the required public interface shapes. Private helpers may remain local to their owning module, but implementations must not add public fields that violate the authority rules.

```python
class ContextTelemetry(StrictModel):
    task_id: str
    context_peak_tokens: int = 0
    context_overflow_count: int = 0
    active_compaction_count: int = 0
    latest_usage_ratio: float = 0.0
    latest_budget_zone: str | None = None
    normal_compaction_count: int = 0
    emergency_compaction_count: int = 0
    compaction_failure_count: int = 0
    normal_zone_passthrough_count: int = 0
    manual_compaction_error_count: int = 0
    overflow_retry_count: int = 0
    active_snapshot_version: int | None = None
    last_compaction_artifact: str | None = None
    last_compaction_error: str | None = None
    last_compaction_at: str | None = None


class OutcomeAdjudicationInput(StrictModel):
    definition: TaskDefinition
    lifecycle: TaskLifecycle
    verification_policy: VerificationPolicy | None
    verification: VerificationEvidenceView
    execution_integrity: ExecutionIntegrity
    successful_change_evidence_ids: list[str]
    scope_violation_evidence_ids: list[str]
    reproduction_state: Literal["unknown", "reproduced", "not_reproduced"]


class OutcomeAdjudicator:
    def decide(self, value: OutcomeAdjudicationInput) -> OutcomeAssessment:
        """Return a pure assessment; only terminal outcomes carry an ID-only decision."""


class TaskRuntime(StrictModel):
    task_id: str
    lifecycle: TaskLifecycleStatus
    pending_actions: list[PendingAction] = []
    latest_decision_id: str | None = None
    pause_reason: str | None = None


class TaskReportView(StrictModel):
    definition: TaskDefinition
    lifecycle: TaskLifecycle
    decision: AdjudicationDecision | None
    hypotheses: list[InvestigationHypothesis]
    unresolved_questions: list[UnresolvedQuestion]
    evidence: list[EvidenceEnvelope]
    operations: list[OperationJournalEntry]
    approvals: list[ExecutionApproval]
    execution_integrity: ExecutionIntegrity
    verification_policy: VerificationPolicy | None
    verification: VerificationEvidenceView
    history: HistorySummaryView
    context_telemetry: ContextTelemetry
```

`TaskRuntime` is an API result, not a sixth Repository and not another task authority. Pending interrupt actions remain sourced from LangGraph interrupt/checkpoint state; if retaining them in the result is unnecessary after CLI migration, omit the field rather than persist it.

---

## Task 1: Move Context Telemetry to HistoryRepository

**Why first:** Working Memory cannot be deleted while compaction budget, overflow, and report metrics still live in its table.

**Files:**

- Create: `tests/domain_repositories/test_context_telemetry.py`
- Modify: `src/deepfix/domain_repositories/history.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/compaction/coordinator.py`
- Modify: `src/deepfix/compaction/middleware.py`
- Modify: `src/deepfix/compaction/migration.py`

- [x] **Step 1: Write failing History telemetry tests.**

Cover default reads, monotonic peak token updates, overflow/retry counters, normal/emergency compaction counters, failure details, active Snapshot version, last Artifact, and idempotent event replay. Also create a legacy `context_metrics` row and prove migration preserves every field exactly once.

- [x] **Step 2: Run the red test.**

```powershell
python -m pytest tests/domain_repositories/test_context_telemetry.py -q
```

Expected: FAIL because `HistoryRepository.context_telemetry()` and mutation methods do not exist.

- [x] **Step 3: Add the bounded History API.**

Implement `context_telemetry`, `record_budget_observation`, `record_overflow`, `record_overflow_retry`, `record_compaction_outcome`, and `record_compaction_failure` with explicit task/event arguments. Store telemetry in a History-owned table in the existing SQLite database. Use stable compaction event IDs to prevent retry double-counting.

- [x] **Step 4: Backfill legacy metrics before switching writers.**

Migration must compare the normalized payload/hash, write the final row, validate it, mark the metrics domain switched, and only then stop writes to `WorkingMemoryStore.context_metrics`.

- [x] **Step 5: Switch coordinator and middleware telemetry writes.**

Replace every `coordinator.memory_store.record_*` and `.metrics()` call with the History API. Do not change budget thresholds, one-overflow-retry behavior, or compaction failure propagation.

- [x] **Step 6: Verify focused regressions.**

```powershell
python -m pytest tests/domain_repositories/test_context_telemetry.py tests/compaction/test_budget.py tests/compaction/test_coordinator.py tests/compaction/test_middleware.py tests/compaction/test_overflow.py -q
python -m ruff check src/deepfix/domain_repositories/history.py src/deepfix/domain_repositories/migration.py src/deepfix/compaction
```

- [x] **Step 7: Commit.**

```powershell
git add src/deepfix/domain_repositories/history.py src/deepfix/domain_repositories/migration.py src/deepfix/compaction/coordinator.py src/deepfix/compaction/middleware.py src/deepfix/compaction/migration.py tests/domain_repositories/test_context_telemetry.py
git commit -m "refactor: move context telemetry to history authority"
```

## Task 2: Introduce Pure Repository-Native Outcome Adjudication

**Files:**

- Create: `src/deepfix/task_domain/adjudication.py`
- Create: `tests/task_domain/test_pure_adjudication.py`
- Modify: `src/deepfix/domain_repositories/evidence.py`
- Modify: `src/deepfix/investigation/evaluation.py`
- Modify: `src/deepfix/task_domain/__init__.py`
- Modify: `src/deepfix/task_domain/repository.py`

- [x] **Step 1: Write the decision table as failing tests.**

At minimum cover:

- all executable required oracles pass after the latest successful change, no stronger conflict, no scope violation, and no unknown Operation → `fixed`;
- a targeted oracle passes but a higher-authority required oracle conflicts → not `fixed`;
- model/Executor recommendation claims completion without supporting IDs → not `fixed`;
- unknown/incomplete Operation → `paused` with supporting operation IDs;
- scope violation → not `fixed`;
- baseline user oracle passes, no code change, and reproduction is `not_reproduced` → `not_reproduced`;
- self-generated supplemental tests alone → not `fixed`;
- decision contains only IDs, never copied Evidence payloads.

- [x] **Step 2: Run the red test.**

```powershell
python -m pytest tests/task_domain/test_pure_adjudication.py -q
```

- [x] **Step 3: Strengthen `VerificationEvidenceView`.**

Expose immutable typed test/change Evidence or stable query helpers sufficient for deterministic oracle evaluation. This is a read view over EvidenceRepository, not a new stored table and not a copied task aggregate.

- [x] **Step 4: Implement the pure adjudicator.**

The adjudicator must not call a model, parse model prose, inspect raw Journal transitions, mutate a Repository, or infer truth from Snapshot/Todo. It returns a new `AdjudicationDecision`; the service separately persists it through TaskRepository.

- [x] **Step 5: Remove the old final-outcome authority.**

Retain experiment assessment helpers in `investigation/evaluation.py`, but route final `fixed`/`not_reproduced` decisions through `OutcomeAdjudicator` only. `executor_recommendation` remains non-authoritative input to planning and is absent from adjudication input.

- [x] **Step 6: Verify.**

```powershell
python -m pytest tests/task_domain/test_pure_adjudication.py tests/test_verification.py tests/investigation/test_evaluation.py tests/task_domain/test_repository.py -q
python -m ruff check src/deepfix/task_domain src/deepfix/domain_repositories/evidence.py src/deepfix/investigation/evaluation.py
```

- [x] **Step 7: Commit.**

```powershell
git add src/deepfix/task_domain src/deepfix/domain_repositories/evidence.py src/deepfix/investigation/evaluation.py tests/task_domain/test_pure_adjudication.py
git commit -m "refactor: make outcome adjudication repository native"
```

## Task 3: Replace Persisted Report Facts with TaskReportView

**Files:**

- Create: `tests/test_repository_native_reporting.py`
- Modify: `src/deepfix/reporting.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/domain_repositories/history.py`
- Modify: `src/deepfix/domain_repositories/execution.py`
- Modify: `src/deepfix/domain_repositories/evidence.py`

- [ ] **Step 1: Write failing report-view tests.**

Build the same report solely from the five repositories. Prove stale values in a legacy task JSON cannot override current Evidence/Investigation/Execution records. Prove context metrics come from History and final outcome comes from the latest adjudication decision plus its supporting IDs.

- [ ] **Step 2: Run the red tests.**

```powershell
python -m pytest tests/test_repository_native_reporting.py -q
```

- [ ] **Step 3: Add read-only summary methods only where needed.**

Prefer existing `list_*`, `verification_view()`, and `integrity_view()` calls. Add bounded immutable summary views only when the report would otherwise need private SQL. Do not add report tables or a `ReportRepository`.

- [ ] **Step 4: Implement `build_task_report_view(repositories, task_id, artifact_root)`.**

Join definition/lifecycle/policy/decision, current hypotheses/questions, Evidence, approvals/operations/integrity, Snapshot summary, context telemetry, and task-owned Artifact references at read time. Historical Snapshot claims must retain provenance and must not replace current domain state.

- [ ] **Step 5: Make `render_report` accept only `TaskReportView`.**

Preserve the useful Chinese report sections and the correct conclusions for fixed, not reproduced, paused-after-change, and infrastructure failure cases. Remove “工作记忆版本”. Report the active Snapshot version and History-owned context metrics instead.

- [ ] **Step 6: Switch CLI report rendering.**

CLI must request a `TaskReportView` after start/continue/approval and must not reconstruct reports from `TaskState` or separate Research Store arguments.

- [ ] **Step 7: Verify.**

```powershell
python -m pytest tests/test_repository_native_reporting.py tests/test_reporting.py tests/test_cli.py tests/research/test_workflow.py -q
python -m ruff check src/deepfix/reporting.py src/deepfix/cli.py src/deepfix/domain_repositories
```

- [ ] **Step 8: Commit.**

```powershell
git add src/deepfix/reporting.py src/deepfix/cli.py src/deepfix/domain_repositories tests/test_repository_native_reporting.py tests/test_reporting.py tests/test_cli.py tests/research/test_workflow.py
git commit -m "refactor: build task reports from domain views"
```

### Review Checkpoint A

Pause and present:

- History telemetry migration evidence;
- pure adjudicator decision-table results;
- report reconstruction evidence;
- confirmation that no production writer was added outside the five repositories;
- `git diff --stat` and commits for Tasks 1–3.

Do not continue until approved.

## Task 4: Build One Repository-Native Protected Context Projection

**Files:**

- Create: `tests/test_repository_native_context.py`
- Modify: `src/deepfix/protected_context.py`
- Modify: `src/deepfix/compaction/snapshot.py`
- Modify: `src/deepfix/compaction/coordinator.py`
- Modify: `src/deepfix/compaction/middleware.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/context.py`

- [ ] **Step 1: Write failing projection tests.**

Cover task isolation, immutable original problem, current hypotheses/questions, deterministic Evidence, approval/operation integrity, active Snapshot history, and exact-once display of each `constraint_id`, `evidence_id`, and `hypothesis_id`. Add a stale Snapshot hypothesis that the current InvestigationRepository marks rejected and assert the stale claim is not displayed as current truth.

- [ ] **Step 2: Prove only one middleware injection.**

Add an Agent construction/request test that counts `<deepfix_protected_context>` blocks and requires exactly one. Snapshot text may expose historical references but not duplicate current records.

- [ ] **Step 3: Run the red tests.**

```powershell
python -m pytest tests/test_repository_native_context.py tests/test_protected_context.py -q
```

- [ ] **Step 4: Replace `ProtectedContextBuilder` dependencies.**

It must receive `DomainRepositories` plus request-local Graph Messages/config. It reads TaskDefinition/Lifecycle, Investigation, Evidence, Execution, and History directly. It must not accept legacy `TaskRepository`, `WorkingMemoryStore`, `CompactionStore`, `ResearchEvidenceStore`, or `InvestigationStore` facades.

- [ ] **Step 5: Replace Working Memory input to Snapshot merge.**

`CompactionSnapshotBuilder` receives old active Snapshot, new complete work units, current Investigation/Evidence views, immutable task definition/constraints, and Execution facts. Remove repeated natural-language Working Memory merging. Preserve Artifact-first → validate Snapshot → activate event → replace messages ordering.

- [ ] **Step 6: Make compaction middleware the single injection owner.**

Remove the separate `ProtectedContextMiddleware` from production Agent construction (or delete it if unused everywhere). Keep one request-local protected block; never write it into Graph Messages.

- [ ] **Step 7: Verify compaction safety.**

```powershell
python -m pytest tests/test_repository_native_context.py tests/test_protected_context.py tests/compaction/test_snapshot.py tests/compaction/test_coordinator.py tests/compaction/test_middleware.py tests/compaction/test_failure_atomicity.py tests/compaction/test_long_context_workflow.py -q
python -m ruff check src/deepfix/protected_context.py src/deepfix/compaction src/deepfix/agent.py src/deepfix/context.py
```

- [ ] **Step 8: Commit.**

```powershell
git add src/deepfix/protected_context.py src/deepfix/compaction src/deepfix/agent.py src/deepfix/context.py tests/test_repository_native_context.py tests/test_protected_context.py tests/compaction
git commit -m "refactor: project protected context from domain authorities"
```

## Task 5: Switch Todo Feedback and Remove the Phase Control Plane

**Files:**

- Modify: `src/deepfix/navigation/feedback.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/prompting.py`
- Modify: `src/deepfix/prompts.py`
- Modify: `src/deepfix/investigation/middleware.py`
- Modify: `src/deepfix/investigation/models.py`
- Modify: `src/deepfix/investigation/store.py`
- Modify: `src/deepfix/investigation/coordinator.py`
- Modify: `src/deepfix/investigation/tools.py`
- Delete: `src/deepfix/investigation/phase.py`
- Modify: `tests/navigation/test_feedback.py`
- Modify: `tests/investigation/test_middleware.py`
- Modify: `tests/test_prompting.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Write failing repository-native navigation tests.**

The existing Todo Graph State schema and three-Tool-Round reminder cadence must remain unchanged. Milestones must be derived from current Investigation/Evidence/Execution/Verification repository views, not legacy facades or Phase.

- [ ] **Step 2: Write failing phase-retirement tests.**

Assert all ordinary diagnostic/edit/test tools remain visible independent of Phase; blocked writes are rejected only by approval/workspace/execution policy; Todo status does not authorize tools; and no prompt contains a Phase matrix or `continue_investigation` instruction.

- [ ] **Step 3: Replace `LegacyNavigationFeedbackSource`.**

Rename it to `RepositoryNavigationFeedbackSource` and inject `DomainRepositories`. Keep milestone feedback advisory and request-local. A milestone may prompt “check whether the current Todo is complete” but never updates Todo itself.

- [ ] **Step 4: Collapse prompts to stable policy.**

`PromptPolicyMiddleware` must not read Investigation state. Retain stable repair/tool/research/reliability rules plus DeepAgents native Todo instructions. Remove `PHASE_PROMPTS` and any dynamic phase prompt selection.

- [ ] **Step 5: Remove Phase navigation and permits.**

Delete `AgentPhase`, `PhaseResolver`, phase-before/after event fields, paused phase, phase correction, phase-based capability maps, and `continue_investigation`. Preserve hypothesis/question/evidence-gap/experiment records only where they are facts or evaluation records—not tool gates.

- [ ] **Step 6: Verify.**

```powershell
python -m pytest tests/navigation tests/investigation/test_middleware.py tests/investigation/test_store.py tests/investigation/test_coordinator.py tests/test_prompting.py tests/test_agent.py -q
python -m ruff check src/deepfix/navigation src/deepfix/investigation src/deepfix/prompting.py src/deepfix/prompts.py src/deepfix/agent.py
rg -n "AgentPhase|PhaseResolver|PHASE_PROMPTS|continue_investigation|phase_based|phase permit" src/deepfix
```

Expected final `rg`: no production matches except an explicitly named migration-only legacy payload parser, if required for historical input.

- [ ] **Step 7: Commit.**

```powershell
git add src/deepfix/navigation src/deepfix/investigation src/deepfix/prompting.py src/deepfix/prompts.py src/deepfix/agent.py tests/navigation tests/investigation tests/test_prompting.py tests/test_agent.py
git commit -m "refactor: retire phase navigation authority"
```

### Review Checkpoint B

Pause and present:

- exact-once Protected Context evidence;
- stale Snapshot precedence test;
- unchanged Todo Checkpoint/reminder behavior;
- proof that Phase and `continue_investigation` no longer control production execution;
- focused compaction and Agent construction results.

Do not continue until approved.

## Task 6: Remove `save_progress` and WorkingMemoryStore

**Files:**

- Modify: `src/deepfix/context.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/compaction/models.py`
- Modify: `src/deepfix/compaction/snapshot.py`
- Modify: `src/deepfix/compaction/migration.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Delete: `src/deepfix/memory.py`
- Modify: tests currently importing `WorkingMemoryStore` or calling `save_progress`
- Add: retirement assertions in `tests/test_legacy_authority_retirement.py`

- [ ] **Step 1: Add failing retirement tests.**

Assert the Agent tool list excludes `save_progress`; no production constructor accepts Working Memory; snapshot/context correctness survives three compactions; user constraints, rejected hypotheses, experiments, and real test results survive because they come from current repositories/history provenance rather than a memory summary.

- [ ] **Step 2: Migrate unresolved legacy semantic fields.**

Before removal, backfill:

- hypotheses and rejection/reopen links → InvestigationRepository;
- unresolved questions → InvestigationRepository;
- deterministic evidence/tests/file changes → EvidenceRepository;
- experiments that are historical only → HistoryRepository;
- user constraints with message provenance → immutable task/history projection as already defined;
- next steps → no persistence migration; current navigation is native Todo;
- phase → no migration; it is deleted.

Validate counts, stable IDs, hashes, and provenance before setting the final Working Memory migration marker.

- [ ] **Step 3: Remove public `save_progress`.**

The model uses native `write_todos` for navigation and explicit domain tools such as `record_hypothesis` for semantic candidates. Deterministic Tool results continue to be collected from Receipts/Evidence automatically. Do not introduce a replacement catch-all tool.

- [ ] **Step 4: Remove Working Memory from Snapshot/context schemas.**

Delete `WorkingMemoryVersion`, `ProgressSnapshot`, renderers, `working_memory_version`, and Working Memory provenance categories from current models. Historical parser compatibility belongs only in migration code and cannot be imported by runtime modules.

- [ ] **Step 5: Delete `memory.py` after import search is clean.**

```powershell
rg -n "WorkingMemoryStore|WorkingMemoryVersion|ProgressSnapshot|save_progress|render_working_memory|working_memory_version" src/deepfix
```

Expected: no production runtime references. Migration-only literal field names are allowed only in `task_domain/legacy_payload.py` or `domain_repositories/migration.py` and must be documented inline.

- [ ] **Step 6: Verify.**

```powershell
python -m pytest tests/test_legacy_authority_retirement.py tests/test_context.py tests/test_protected_context.py tests/compaction tests/navigation tests/investigation -q
python -m ruff check src/deepfix tests/test_legacy_authority_retirement.py
```

- [ ] **Step 7: Commit.**

```powershell
git add -A src/deepfix/memory.py src/deepfix/context.py src/deepfix/agent.py src/deepfix/compaction src/deepfix/domain_repositories/migration.py tests
git commit -m "refactor: retire working memory authority"
```

## Task 7: Replace Giant TaskState in the Service Runtime

**Files:**

- Create: `src/deepfix/task_domain/runtime.py`
- Create: `tests/task_domain/test_runtime_boundary.py`
- Modify: `src/deepfix/service.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/artifact_retrieval/service.py`
- Modify: `src/deepfix/workspace.py`
- Modify: `src/deepfix/task_domain/repository.py`
- Modify: `src/deepfix/task_domain/models.py`
- Modify: service/CLI/artifact/workflow tests

- [ ] **Step 1: Write the minimal runtime boundary test.**

Assert `TaskRuntime` exposes task identity, lifecycle, optional pending interrupt actions, pause reason, and decision reference only. Explicitly reject fields named `conversation`, `evidence`, `hypotheses`, `changed_files`, `test_results`, `approvals`, `context_metrics`, `recovery`, `repair_plan`, or `final_summary`.

- [ ] **Step 2: Write service behavior tests against repositories and Graph state.**

Cover start, continue, pause, approval interrupt/resume, operation recovery, context/investigation recovery exceptions, recursion pause, structured response, required-oracle decision, and report generation. Verify:

- new user messages enter LangGraph with stable IDs and remain in Checkpoint/history, not TaskRepository;
- task lifecycle transitions through TaskRepository;
- pending approvals are reconstructed from the active interrupt and ExecutionRepository;
- tool results flow to Receipt/Evidence/Investigation repositories;
- final decision is produced by OutcomeAdjudicator and stored by ID only.

- [ ] **Step 3: Refactor `BugfixService` around `task_id`.**

Replace `_sync_context`, `_save(TaskState)`, `save_legacy_projection`, mutable phase/status transitions, and copied counters with direct bounded repository operations. Request-local variables may organize one method call but must not be persisted as a second aggregate.

- [ ] **Step 4: Keep infrastructure error propagation typed.**

Middleware/coordinator continue to throw typed exceptions with sanitized recovery metadata. Service catches them and transitions TaskLifecycle to `PAUSED`; middleware never changes business lifecycle. Persist recovery as History/debug event plus Artifact reference where appropriate, not as a TaskState field.

- [ ] **Step 5: Switch Workspace and Artifact Retrieval inputs.**

Pass immutable `TaskDefinition`/task ID and bounded views, never a giant mutable task object. Preserve canonical path, symlink/junction, shell confinement, and task-owned Artifact isolation checks.

- [ ] **Step 6: Verify.**

```powershell
python -m pytest tests/task_domain/test_runtime_boundary.py tests/test_service.py tests/test_cli.py tests/artifact_retrieval tests/research/test_workflow.py tests/task_domain/test_service_integration.py tests/domain_repositories/test_service_integration.py -q
python -m ruff check src/deepfix/service.py src/deepfix/cli.py src/deepfix/task_domain src/deepfix/artifact_retrieval src/deepfix/workspace.py
```

- [ ] **Step 7: Commit.**

```powershell
git add src/deepfix/service.py src/deepfix/cli.py src/deepfix/task_domain src/deepfix/artifact_retrieval src/deepfix/workspace.py tests
git commit -m "refactor: remove giant task state from runtime"
```

### Review Checkpoint C

Pause and present:

- absence of `save_progress` and Working Memory runtime imports;
- service start/continue/approval/recovery behavior using repository/Graph authorities;
- proof that `TaskRuntime` contains no fact copies;
- focused service, compaction, navigation, and recovery results.

Do not continue until approved.

## Task 8: Retire Old Store Facades and Isolate Legacy Payload Parsing

**Files:**

- Create: `src/deepfix/task_domain/legacy_payload.py`
- Modify: `src/deepfix/task_domain/migration.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/service.py`
- Modify/Delete: `src/deepfix/persistence.py`
- Modify/Delete: `src/deepfix/compaction/store.py`
- Modify/Delete: `src/deepfix/research/store.py`
- Modify/Delete: old `ToolExecutionReceiptStore`, `OperationJournalStore`, `VerificationPolicyStore`, and `InvestigationStore` facade definitions
- Modify: migration and restore tests

- [ ] **Step 1: Freeze historical fixtures.**

Store representative legacy JSON payloads in tests rather than constructing new runtime `TaskState` instances. Include created/running/approval/paused/completed tasks, legacy Working Memory, legacy Phase, compaction recovery, investigation recovery, approvals, research, and Artifact references.

- [ ] **Step 2: Implement a one-way migration-only parser.**

`LegacyTaskPayload` may understand historical field names, but it must return calls/records for the five repositories. It cannot be saved back, imported into production composition, or exposed by service/CLI APIs.

- [ ] **Step 3: Prove restore before deletion.**

For each fixture, migrate and compare definition hash, lifecycle, verification policy, decision IDs, Evidence hashes/provenance, Investigation state, Execution integrity, Snapshot/Artifact references, and context telemetry. Then restore the Graph thread through the normal LangGraph Checkpoint path.

- [ ] **Step 4: Remove production facade construction.**

Agent, CLI, service, context, navigation, research, compaction, investigation, operation recovery, and artifact retrieval must receive the bounded repositories directly. Remove old class-name exports once no production import remains.

- [ ] **Step 5: Delete giant state models.**

Remove `TaskState`, `TaskStatus`, duplicated `Evidence`, `TestResult`, `ApprovalRecord`, and fact-bearing `RepairOutcome` from `models.py` after callers use final domain types and Graph structured output candidates. If a model-output schema remains necessary, name it as a candidate/proposal and ensure it cannot carry authoritative fact fields.

- [ ] **Step 6: Run the retirement source gate.**

```powershell
rg -n "TaskState|TaskStatus|WorkingMemoryStore|AgentPhase|PhaseResolver|LegacyNavigationFeedbackSource|save_progress|continue_investigation|CompactionStore|ResearchEvidenceStore|VerificationPolicyStore|ToolExecutionReceiptStore|OperationJournalStore|InvestigationStore" src/deepfix
```

Expected: no production runtime dependency. Explicit legacy field-name handling is allowed only in migration/evaluation-fixture modules and must not instantiate old authorities.

- [ ] **Step 7: Verify.**

```powershell
python -m pytest tests/task_domain/test_migration.py tests/domain_repositories/test_migration_gate.py tests/test_legacy_authority_retirement.py tests/compaction/test_migration.py tests/investigation/test_migration.py -q
python -m ruff check src/deepfix tests
```

- [ ] **Step 8: Commit.**

```powershell
git add -A src/deepfix tests
git commit -m "refactor: retire legacy state and store facades"
```

## Task 9: Final Production Wiring and Offline Gate

**Files:**

- Modify: `docs/superpowers/plans/2026-08-28-deepfix-legacy-state-retirement.md` — completion evidence only
- Modify: `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md` — mark Plan 4 complete only after all gates pass
- Modify: any test-only import cleanup found by the final gate

- [ ] **Step 1: Inspect final Agent middleware/tool construction.**

Required production order/ownership:

1. stable Message identity;
2. DeepAgents native Todo;
3. lightweight Todo navigation reminder using final repository views;
4. migration adapters only when an unswitched historical task is detected;
5. trusted execution/approval/Receipt/Operation middleware;
6. stable prompt policy;
7. one protected-context/compaction middleware;
8. DeepAgents filesystem/backend behavior.

There must be no separate Phase middleware, duplicate context injector, Working Memory middleware, or public catch-all progress tool.

- [ ] **Step 2: Run focused final suites.**

```powershell
python -m pytest tests/task_domain tests/domain_repositories tests/navigation tests/compaction tests/investigation tests/artifact_retrieval tests/research tests/test_agent.py tests/test_service.py tests/test_cli.py tests/test_reporting.py tests/test_operations.py tests/test_verification.py -q
```

- [ ] **Step 3: Run the trusted offline core suite.**

Use the repository's established core selector and preserve the known Windows environment exclusions only when they reproduce the previously documented `_overlapped`/WinError 10106 infrastructure failure. Do not silently add new exclusions.

```powershell
python -m pytest -q
```

- [ ] **Step 4: Run formatting, source, and diff gates.**

```powershell
python -m ruff check src tests
git diff --check
rg -n "TaskState|WorkingMemoryStore|AgentPhase|PhaseResolver|PHASE_PROMPTS|LegacyNavigationFeedbackSource|save_progress|continue_investigation" src/deepfix
```

- [ ] **Step 5: Record exact evidence.**

Update this document with commands, pass/fail/skip counts, known environment-only failures, removal-search results, commit IDs, and remaining migration-only references. Do not write “all passed” without command output.

- [ ] **Step 6: Request code review.**

Use `superpowers:requesting-code-review` against the Plan 4 diff. Fix correctness/authority/recovery findings with TDD and rerun affected gates.

- [ ] **Step 7: Mark the program complete and commit docs.**

Only after review and verification:

```powershell
git add docs/superpowers/plans/2026-08-28-deepfix-legacy-state-retirement.md docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md
git commit -m "docs: record legacy authority retirement"
```

### Final Review Checkpoint

Present to the user before any merge/push:

- commits and diff summary;
- focused and core test evidence;
- source-search proof of retired production dependencies;
- remaining migration-only legacy references and why they cannot become runtime authorities;
- confirmation that no paid benchmark ran;
- confirmation that Gitee push/merge was not performed unless separately authorized.

---

## Failure and Rollback Rules

- If a migration validation fails, do not set the domain switch marker and do not stop the old reader/writer for that task.
- If a new authoritative read fails after a validated cutover, raise a typed recovery error with task ID, failed domain, sanitized underlying exception, and recovery action; BugfixService transitions lifecycle to `PAUSED`.
- Middleware and coordinator never directly mutate TaskLifecycle.
- A context Artifact/Snapshot failure before activation leaves Graph Messages intact. At 82–90% context usage, record failure and allow one original request pass-through; above 90% or on overflow, surface the typed recovery error according to the existing policy.
- Never replay an external side effect merely to rebuild state. Recover from Receipt, Operation Journal, Artifact hashes, and actual workspace hashes.
- Do not resurrect giant `TaskState` as a rollback mechanism. Rollback uses read-only historical tables/artifacts plus the migration-only parser.

## Plan Completion Criteria

Plan 4 is complete only when all of the following are true:

- [ ] DeepAgents/LangGraph remain the only Agent Loop, Message, Todo, Checkpoint, interrupt, and resume owners.
- [ ] Five bounded repositories are the only current business-state persistence authorities.
- [ ] OutcomeAdjudicator is deterministic/pure and persists only outcome plus supporting IDs.
- [ ] Required oracle conflicts, scope violations, unknown operations, and missing post-change verification prevent false `fixed`.
- [ ] `TaskReportView` is assembled at read time and is not persisted.
- [ ] Protected Context is injected exactly once and deduplicates constraint/evidence/hypothesis IDs.
- [ ] Current domain state overrides stale Snapshot semantics while provenance remains visible.
- [ ] Todo reminder behavior and Graph State schema remain unchanged.
- [ ] Phase, phase prompt/tool gating, `continue_investigation`, `save_progress`, and Working Memory are absent from production runtime.
- [ ] Service runtime contains no duplicate fact/conversation aggregate.
- [ ] Old Store class names are absent from production construction paths.
- [ ] Historical restore works through migration-only parsing and LangGraph Checkpoint.
- [ ] Approval, confinement, Receipt idempotency, Operation recovery, Artifact integrity, compaction atomicity, overflow single retry, and required-oracle tests remain green.
- [ ] Focused and core offline gates are recorded with exact evidence.
- [ ] No online QuixBugs or formal A/B benchmark was run.

## Self-Review Checklist

- [x] Every frozen spec authority has one final owner.
- [x] `unresolved_questions` remain in InvestigationRepository/History, not Todo.
- [x] Snapshot history cannot override current domain facts.
- [x] OutcomeAdjudicator does not persist copied Evidence.
- [x] Context telemetry has an owner before Working Memory deletion.
- [x] Readers switch and validate before legacy writers/classes are removed.
- [x] No new generic manager, planner, task graph, store registry, or database is introduced.
- [x] Each task has red-green-refactor tests, exact commands, and a commit boundary.
- [x] Expensive acceptance remains explicitly out of scope.
- [x] No unresolved interface or undefined authority remains.
