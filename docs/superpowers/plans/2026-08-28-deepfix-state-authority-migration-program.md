# DeepFix State Authority Migration Program

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this program plan-by-plan. Each child plan uses checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DeepFix's duplicated navigation and persistence authorities with framework-native Todo navigation and five bounded DeepFix repositories while preserving trusted execution, recovery, evidence provenance, compaction safety, and false-FIXED protection.

**Architecture:** Execute four independently reviewable plans in strict order. First add framework-native Todo navigation as a non-authoritative overlay; next establish the shared SQLite transaction boundary and bounded Task domain; then consolidate the remaining domain repositories; finally remove legacy Phase, Working Memory, giant TaskState, and duplicate projections after compatibility reads prove safe.

**Tech Stack:** Python 3.11+, DeepAgents 0.7.x, LangChain 1.3.x, LangGraph 1.2.x, Pydantic 2, SQLite/WAL, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-27-deepfix-architecture-audit.md`

## Global Constraints

- DeepAgents and LangGraph own the generic Agent Loop, Messages, native Todo schema, `write_todos`, Checkpoint, interrupt, and resume.
- Todo is the only navigation authority and never becomes Evidence, a tool-permission gate, or completion authority.
- Task Definition is an immutable snapshot tied to `original_message_id`; later constraints retain their source User Message IDs and never rewrite the original problem.
- Domain repositories own current facts; Snapshot owns history, not truth.
- Receipt and Operation remain distinct domain models even after persistence is consolidated.
- External research remains a capability and typed Evidence source, not an independent persistence authority.
- OutcomeAdjudicator consumes VerificationPolicy, Evidence view, ExecutionIntegrity, and code state; it does not interpret raw Journal lifecycle transitions.
- SQLite transactions never span model calls, tool execution, filesystem mutation, Shell commands, or network requests.
- Artifact content is atomically written and hash-verified before SQLite stores its reference.
- Preserve Workspace confinement, approval, Operation recovery, Receipt idempotency, deterministic Evidence, required-oracle checks, structured compaction, overflow recovery, and false-FIXED protection throughout the migration.
- No Multi-Agent routing, new Planner, new Phase model, TodoStore, WorkingMemory V2, generic Manager layer, or additional database is introduced.
- This program stops at implementation completion plus focused and core automated tests. It does not run the expensive formal QuixBugs A/B acceptance campaign.
- Preserve unrelated dirty-worktree changes; each implementation task stages only its listed files.
- Every production change follows red-green-refactor and pauses at its review gate.

## Target Persistence Boundary

```text
deepfix.db
├── LangGraph Checkpointer tables       # framework-owned
├── TaskRepository tables
├── InvestigationRepository tables
├── EvidenceRepository tables
├── ExecutionRepository tables
└── HistoryRepository tables

artifacts/
├── conversation_history/
├── large_tool_results/
├── operation_results/
├── research/
└── debug/
```

Five Repository classes are logical access boundaries over one DeepFix SQLite database. The Artifact root is the only second physical persistence subsystem. A shared connection/UnitOfWork enables atomic cross-table commits where no external side effect is in flight.

## Ordered Child Plans and Review Gates

### Plan 1: Framework-Native Todo Navigation Feedback

**Planned document:** `docs/superpowers/plans/2026-08-28-deepfix-todo-navigation-feedback.md`

**Deliverable:** Enable LangChain `TodoListMiddleware` and native `write_todos`; enforce at most one `in_progress`; add non-authoritative Graph State navigation metadata; count completed Tool Rounds; and dynamically inject a reminder after three rounds without a meaningful Todo update or once per new deterministic milestone.

**Required behavior:**

- One AIMessage containing one or more parallel Tool Calls plus all paired ToolMessages counts as one Tool Round.
- A Todo status transition, a changed `in_progress` item, or adding/removing an item resets the navigation counter; prose-only rewriting does not.
- Todo changes never reset decision-relevant progress or investigation stagnation.
- Milestone hints are derived read-only from existing Investigation, Evidence, Execution, and Verification views and are never persisted as facts.
- Reminder injection is request-local and does not create Conversation Messages or CompactionSnapshot facts.
- The Harness never automatically marks a Todo completed or uses Todo to expose/hide tools.
- Legacy Phase remains shadow-only during this plan so Todo behavior can be reviewed independently.

**Review gate:** Native Todo state survives Checkpoint resume and compaction; parallel tools count once; mechanical Todo rewrites do not count as real progress; three-round and one-shot milestone reminders are covered by deterministic tests; existing approval, Receipt, Journal, and adjudication tests remain green.

### Plan 2: Shared Persistence Foundation and Bounded Task Domain

**Status:** COMPLETE — implementation and offline migration gate recorded in
`2026-08-28-deepfix-task-persistence-foundation.md` on 2026-08-29; review approved
and Plan 3 completed.

**Planned document:** `docs/superpowers/plans/2026-08-28-deepfix-task-persistence-foundation.md`

**Deliverable:** Introduce the shared SQLite connection/UnitOfWork boundary; replace whole-object `TaskRepository.save(TaskState)` writes with immutable Task Definition, lifecycle, VerificationPolicy, AdjudicationDecision, and budget ledger operations; and provide a migration/read adapter for existing task JSON.

**Allowed TaskRepository ownership:**

```text
task_definitions
task_lifecycle
verification_policies
adjudication_decisions
token_budgets
token_reservations
```

**Explicit exclusions:**

```text
messages / todos
hypotheses / unresolved_questions
evidence / tests / changed files
operations / receipts / approvals
snapshots / artifacts
conversation summaries / report fields
```

**Review gate:** Repository code rejects mutation of the original Task Definition; lifecycle contains no investigation/edit/test phases; VerificationPolicy cannot be silently downgraded; budget reservation remains atomic; legacy tasks load through the adapter; no new whole-task JSON write is introduced.

### Plan 3: Domain Repository Consolidation

**Status:** COMPLETE — Tasks 1–8, migration cutover hardening, bounded repository
wiring, focused/trusted/core offline gates, and exact completion evidence recorded
in `2026-08-28-deepfix-domain-repository-consolidation.md` on 2026-08-30.
Awaiting user review before Plan 4 begins.

**Planned document:** `docs/superpowers/plans/2026-08-28-deepfix-domain-repository-consolidation.md`

**Deliverable:** Consolidate deterministic and external research Evidence into EvidenceRepository; Receipt and Operation persistence into ExecutionRepository; Hypothesis and UnresolvedQuestion into InvestigationRepository; and Compaction persistence into a history-only HistoryRepository.

**Required boundaries:**

- Evidence uses a common immutable envelope with typed payload, origin, authority, verification state, independent provenance roots, and Artifact references.
- A model cannot set or elevate system authority/trust fields.
- ResearchService owns acquisition behavior; EvidenceRepository owns accepted ExternalEvidence and research-attempt audit metadata.
- ExecutionRepository keeps separate `operations`, `receipts`, and approval records while sharing recovery, idempotency, query, and transaction infrastructure.
- Operation lifecycle remains `prepared`, `started`, `observed`, `committed`, or `unknown` because SQLite cannot roll back external side effects.
- InvestigationRepository stores Hypothesis and UnresolvedQuestion but never controls tool permissions or task lifecycle.
- HistoryRepository stores Snapshot lifecycle, input hash, message/work-unit coverage, historical semantic items, provenance, and Artifact references; current domain facts are removed from its authority.
- Snapshot `prepared/active/abandoned` activation and “do not remove original messages before Artifact and Snapshot verification” remain unchanged.

**Review gate:** Migration tests prove stable IDs, counts, hashes, provenance roots, and Artifact references survive; parallel Receipt writes remain idempotent; fault injection produces no duplicate side effect; current domain records override stale Snapshot projections; legacy Store writes can be disabled without losing restore or reporting data.

### Plan 4: Legacy Authority Retirement and Pure Adjudication

**Planned document:** `docs/superpowers/plans/2026-08-28-deepfix-legacy-state-retirement.md`

**Deliverable:** Switch all production reads to the five Repository views; make OutcomeAdjudicator consume high-level Evidence and ExecutionIntegrity; replace report persistence with a read-only TaskReportView; and remove Phase navigation, WorkingMemoryStore, giant TaskState fact copies, duplicate context middleware, and legacy Store write paths.

**Required removals:**

- `AgentPhase`, PhaseResolver, phase prompt matrix, phase-based tool visibility, and phase correction/permit navigation.
- `WorkingMemory.phase`, `WorkingMemory.next_steps`, and finally WorkingMemoryStore after compatibility reads end.
- `TaskState.conversation`, evidence, hypotheses, changed-files, tests, approvals, recovery copies, and other fields now owned by Domain Repositories.
- Independent ResearchStore, VerificationPolicyStore, ToolExecutionReceiptStore, OperationJournalStore, and over-broad CompactionStore access paths after their data has migrated.
- Public `save_progress` behavior that writes navigation, deterministic facts, or Phase.
- Repeated Protected Context/Compaction/Blackboard aggregation of the same current records.

**Required final behavior:**

- OutcomeAdjudicator reads Task Definition, VerificationPolicy, VerificationEvidenceView, ExecutionIntegrity, and code state only.
- TaskRepository persists the outcome plus supporting Evidence/Operation IDs, never copied Evidence bodies.
- TaskReportView joins authoritative repositories at read time and is never persisted as another TaskState.
- Todo navigation feedback switches from legacy adapters to the final Investigation/Evidence/Execution repositories without changing its Graph State schema.
- Context construction deduplicates each constraint, evidence, and hypothesis ID across Protected Context and Snapshot projection.

**Review gate:** Focused migration, recovery, compaction, adjudication, service, CLI-report, and Agent construction tests pass; the core offline suite passes; searches prove removed classes are no longer production dependencies; no expensive online benchmark is run.

## Transaction and Migration Rules

1. Add new schema and read adapters before changing any production authority.
2. Backfill one domain at a time using stable IDs and content hashes.
3. Validate row counts, referential links, Artifact existence, and hash equality before switching reads.
4. Switch the authoritative reader for one domain at a review gate.
5. Stop old writes immediately after the reader switch; do not maintain indefinite dual-write paths.
6. Keep old tables and Receipt files read-only for a bounded rollback window.
7. Remove compatibility code only in Plan 4 after restore tests pass against migrated tasks.

For external side effects, commits remain deliberately split:

```text
commit PREPARED
commit STARTED
execute external side effect
write and verify Artifact
commit Receipt metadata + OBSERVED + derived Evidence links
commit COMMITTED after authoritative state projection succeeds
```

A database failure may leave an orphan Artifact, which is safe to garbage-collect. The database must never reference an Artifact that was not written and verified.

## Test Strategy

Each child plan must include focused TDD tests plus the smallest relevant regression set. The final Plan 4 core gate covers:

- Agent middleware construction and ordering;
- native Todo creation, update, Checkpoint resume, compaction, and reminder behavior;
- immutable Task Definition and lifecycle transition rules;
- VerificationPolicy version and downgrade rejection;
- budget reservation and settlement;
- Evidence origin/authority/provenance invariants;
- parallel Receipt idempotency and Operation recovery fault injection;
- Snapshot activation/failure atomicity and current-domain-over-history projection;
- false-FIXED prevention through required-oracle conflicts and incomplete ExecutionIntegrity;
- report reconstruction without persisted fact copies;
- CLI task create/resume/report workflows.

Online QuixBugs runs and the formal Legacy-vs-Experiment A/B gate are explicitly excluded from this program's completion criteria.

## Architecture Coverage

| Frozen requirement | Implemented by |
|---|---|
| Native Todo and single navigation authority | Plan 1 |
| Three Tool Round reminder and milestone feedback | Plan 1 |
| Todo cannot become progress, permission, or outcome authority | Plans 1 and 4 |
| Immutable Task Definition and lifecycle-only status | Plan 2 |
| TaskRepository admission/exclusion boundary | Plan 2 |
| One SQLite database and shared UnitOfWork | Plan 2 |
| Evidence + external Research persistence | Plan 3 |
| Receipt + Operation persistence and recovery | Plan 3 |
| Hypothesis + UnresolvedQuestion persistence | Plan 3 |
| History-only Snapshot authority | Plan 3 |
| Pure OutcomeAdjudicator input boundary | Plan 4 |
| Delete WorkingMemoryStore and giant TaskState authority | Plan 4 |
| Remove Phase control plane and duplicate context projections | Plan 4 |
| Preserve DeepAgents-first framework ownership | All plans |

## Stop Rules

- Do not begin a child plan until the previous review gate is approved.
- If a migration requires permanent dual writes, stop and redesign the boundary instead of adding reconciliation complexity.
- If native Todo cannot survive the existing Checkpointer/compaction chain without a custom TodoStore, stop and inspect framework integration; do not create the Store as a workaround.
- If unified SQLite produces measured lock contention under core concurrency tests, report the evidence before proposing a second database.
- If a removed reliability invariant cannot be expressed through the target repositories, stop before deleting the legacy source.
- Do not introduce Multi-Agent, model routing, Experiment Planner production migration, or a formal online benchmark in this program.
