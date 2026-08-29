# DeepFix Task Persistence Foundation Implementation Plan

**Status:** COMPLETE — implemented on `codex/task-persistence-foundation`; final
offline evidence recorded below on 2026-08-29. Plan 3 remains blocked on the
review checkpoint at the end of this document.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one shared SQLite transaction boundary and a bounded Task domain whose immutable definition, lifecycle, VerificationPolicy, adjudication decisions, and token ledger are authoritative without prematurely removing unmigrated legacy task fields.

**Architecture:** Add a reusable SQLite connection/UnitOfWork infrastructure, then introduce narrow Task-domain contracts and repository operations. Existing `TaskState` callers continue through a field-level compatibility projection: fields whose authority has moved to normalized Task tables are reconstructed from those tables, while unmigrated Evidence, investigation, execution, history, reporting, and recovery fields remain in the legacy payload until Plans 3 and 4 migrate them. One SQLite database is shared physical infrastructure; it does not collapse domain ownership into one generic repository.

**Tech Stack:** Python 3.11+, DeepAgents 0.7.x, LangChain 1.3.x, LangGraph 1.2.x, Pydantic 2, SQLite/WAL, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-27-deepfix-architecture-audit.md`

**Program:** `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md` Plan 2

## Global Constraints

- DeepAgents/LangGraph continue to own the Agent Loop, Messages, Todo, Checkpoint, interrupt, and resume; this plan does not add a Loop, Planner, TodoStore, Checkpointer, or manager abstraction.
- `TaskDefinition.original_problem` is an immutable task-definition snapshot tied to `original_message_id`. A later User Message or constraint never mutates it.
- Task lifecycle contains only `created`, `running`, `waiting_approval`, `paused`, `completed`, `failed`, and `cancelled`. Investigation, diagnosis, editing, testing, and review remain temporary legacy phase projections until Plan 4 removes them.
- `TaskRepository` admits only Task Definition, lifecycle, VerificationPolicy, AdjudicationDecision, token budgets, and token reservations.
- Messages, todos, hypotheses, unresolved questions, Evidence, test results, changed files, operations, receipts, approvals, snapshots, artifacts, summaries, recovery bodies, and report fields are explicitly excluded from normalized Task tables.
- One SQLite file does not mean one Repository. This plan adds shared connection/transaction infrastructure, not a generic CRUD repository.
- `VerificationPolicy` does not own workspace mutation scope. Do not add `allowed_paths`; `VerificationOracle.relevant_paths` remains evidence relevance metadata, while canonical path and symlink/junction enforcement remain owned by `TaskWorkspace` and execution confinement.
- Legacy payload retirement is field-by-field. Only fields whose canonical writer and reader switch in this plan are removed from legacy authority; all other legacy fields continue to round-trip unchanged.
- Compatibility reads may expose a legacy phase-shaped `TaskState.status`, but normalized `TaskLifecycle.status` is the business lifecycle authority.
- SQLite transactions never span model calls, tool execution, filesystem mutation, Shell commands, network requests, or Artifact writes.
- VerificationPolicy required oracles cannot be removed or downgraded in a later version.
- Token reservation is atomic through `BEGIN IMMEDIATE`; calls reserve before invocation and settle from provider usage afterward.
- No production Evidence, Receipt, Journal, Investigation, Compaction, WorkingMemory, report, or CLI behavior is removed in this plan.
- Plan 3 remains responsible for `InvestigationRepository`, `EvidenceRepository`, `ExecutionRepository`, and `HistoryRepository`; Plan 2 neither creates their final tables nor switches their readers or writers.
- Every production change follows red-green-refactor. Commit only files listed by the current task and pause at the Plan 2 review checkpoint.

---

## File Structure

### New files

- `src/deepfix/database.py` — shared SQLite connection configuration and bounded UnitOfWork.
- `src/deepfix/task_domain/__init__.py` — public Task-domain exports only.
- `src/deepfix/task_domain/models.py` — immutable definition, lifecycle, adjudication, and compatibility contracts.
- `src/deepfix/task_domain/repository.py` — narrow Task-domain persistence operations over an injected database.
- `src/deepfix/task_domain/migration.py` — deterministic legacy payload decomposition and reconstruction.
- `tests/task_domain/test_database.py` — WAL, transaction, rollback, and connection ownership tests.
- `tests/task_domain/test_models.py` — Task-domain contract and transition tests.
- `tests/task_domain/test_repository.py` — definition, lifecycle, policy reference, and decision persistence tests.
- `tests/task_domain/test_migration.py` — legacy backfill, field ownership, reconstruction, and idempotency tests.
- `tests/task_domain/test_service_integration.py` — new-task, resume, list, and compatibility workflow tests.

### Existing files modified

- `src/deepfix/persistence.py` — retain import compatibility and `checkpoint_connection()`, delegate Task-domain work to the new repository, and expose a bounded legacy projection adapter instead of an unrestricted whole-object authority.
- `src/deepfix/models.py` — add no new authority; retain `TaskState` only as the compatibility/service projection used until Plan 4.
- `src/deepfix/verification.py` — make `VerificationPolicyStore` a compatibility facade over Task-domain policy operations.
- `src/deepfix/investigation/token_budget.py` — make `TokenBudgetStore` a compatibility facade over the Task budget ledger without changing middleware/callback interfaces.
- `src/deepfix/service.py` — create immutable definition after assigning the original User Message ID, use narrow lifecycle/policy/decision methods, and save only unmigrated legacy projection fields.
- `src/deepfix/cli.py` — construct one shared database object and inject it into Task-domain facades while preserving LangGraph `SqliteSaver` connection ownership.
- `tests/test_persistence.py` — replace whole-object-authority expectations with compatibility projection expectations.
- `tests/test_verification.py` — retain downgrade tests and add workspace-scope exclusion tests.
- `tests/investigation/test_token_budget.py` — retain existing budget behavior and add shared-transaction concurrency coverage.
- `tests/test_service.py` — preserve legacy service behavior while asserting normalized Task authority.
- `tests/test_cli.py` — verify list/new/resume continue to work through the compatibility adapter.

## Stable Interfaces

The following signatures are fixed for every task in this plan:

```python
class SQLiteDatabase:
    def __init__(self, path: str | Path) -> None: ...
    def connect(self, *, check_same_thread: bool = False) -> sqlite3.Connection: ...
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]: ...
    @contextmanager
    def unit_of_work(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]: ...


class TaskLifecycleStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRepository:
    def __init__(
        self,
        database: SQLiteDatabase | str | Path,
    ) -> None: ...
    def create_definition(self, definition: TaskDefinition) -> TaskDefinition: ...
    def get_definition(self, task_id: str) -> TaskDefinition: ...
    def get_lifecycle(self, task_id: str) -> TaskLifecycle: ...
    def transition_lifecycle(
        self,
        task_id: str,
        next_status: TaskLifecycleStatus,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> TaskLifecycle: ...
    def save_verification_policy(self, policy: VerificationPolicy) -> None: ...
    def load_verification_policy(
        self, task_id: str, version: int | None = None
    ) -> VerificationPolicy | None: ...
    def record_adjudication(self, decision: AdjudicationDecision) -> None: ...
    def latest_adjudication(self, task_id: str) -> AdjudicationDecision | None: ...
    def save_legacy_projection(self, task: TaskState) -> None: ...
    def get(self, task_id: str) -> TaskState: ...
    def list_recent(self, limit: int = 20) -> list[TaskState]: ...
```

`save(TaskState)` remains only as a deprecated compatibility alias for `save_legacy_projection()` during Plans 2 and 3. It must decompose the payload and must never issue an upsert of `TaskState.to_dict()` as an authoritative whole object.

---

### Task 1: Shared SQLite Database and UnitOfWork

**Files:**
- Create: `src/deepfix/database.py`
- Create: `tests/task_domain/test_database.py`
- Modify: `src/deepfix/persistence.py`

**Interfaces:**
- Produces: `SQLiteDatabase`, `open_sqlite_connection()` compatibility wrapper.
- Consumes: standard-library `sqlite3`, `Path`, and `contextmanager` only.

- [ ] **Step 1: Write failing connection and transaction tests**

```python
def test_database_connections_share_wal_and_busy_timeout(tmp_path):
    database = SQLiteDatabase(tmp_path / "state" / "deepfix.db")
    with database.connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_unit_of_work_commits_all_writes_together(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    with database.connection() as connection:
        connection.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
        connection.commit()
    with database.unit_of_work() as connection:
        connection.execute("INSERT INTO values_table VALUES ('a')")
        connection.execute("INSERT INTO values_table VALUES ('b')")
    with database.connection() as connection:
        assert connection.execute("SELECT value FROM values_table").fetchall() == [
            ("a",),
            ("b",),
        ]


def test_unit_of_work_rolls_back_every_write_on_error(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    with database.connection() as connection:
        connection.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
        connection.commit()
    with pytest.raises(RuntimeError, match="abort"):
        with database.unit_of_work(immediate=True) as connection:
            connection.execute("INSERT INTO values_table VALUES ('a')")
            raise RuntimeError("abort")
    with database.connection() as connection:
        assert connection.execute("SELECT value FROM values_table").fetchall() == []
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_database.py -q`

Expected: FAIL because `deepfix.database.SQLiteDatabase` does not exist.

- [ ] **Step 3: Implement connection ownership and bounded transactions**

```python
class SQLiteDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, *, check_same_thread: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=check_same_thread,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def unit_of_work(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
```

Keep `deepfix.persistence.open_sqlite_connection()` as:

```python
def open_sqlite_connection(
    database_path: str | Path,
    *,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    return SQLiteDatabase(database_path).connect(
        check_same_thread=check_same_thread
    )
```

- [ ] **Step 4: Run focused and current persistence tests**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_database.py tests/test_persistence.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/deepfix/database.py src/deepfix/persistence.py tests/task_domain/test_database.py
git commit -m "refactor: add shared sqlite unit of work"
```

---

### Task 2: Immutable Task Definition and Lifecycle Contracts

**Files:**
- Create: `src/deepfix/task_domain/__init__.py`
- Create: `src/deepfix/task_domain/models.py`
- Create: `tests/task_domain/test_models.py`

**Interfaces:**
- Produces: `TaskDefinition`, `TaskLifecycleStatus`, `TaskLifecycle`, `AdjudicationDecision`, `TaskDefinitionConflict`, `TaskLifecycleConflict`.
- Consumes: `StrictModel` from `deepfix.compaction.models` and Pydantic `Field`.

- [ ] **Step 1: Write failing immutable-definition and lifecycle tests**

```python
def test_task_definition_is_frozen_and_tracks_original_message():
    definition = TaskDefinition(
        task_id="task-1",
        original_message_id="message-1",
        original_problem="修复排序错误",
        approval_mode="manual",
        source_project_root="C:/repo",
        workspace_root="C:/workspaces/task-1",
        workspace_baseline_id="baseline-1",
        project_python="C:/Python/python.exe",
        confinement_level="guarded_local",
        created_at="2026-08-29T00:00:00+00:00",
    )
    with pytest.raises(ValidationError):
        definition.original_problem = "被模型改写"


def test_lifecycle_excludes_agent_phases():
    values = {item.value for item in TaskLifecycleStatus}
    assert values == {
        "created",
        "running",
        "waiting_approval",
        "paused",
        "completed",
        "failed",
        "cancelled",
    }
    assert "investigating" not in values
    assert "editing" not in values
    assert "testing" not in values


def test_adjudication_stores_only_ids_not_evidence_bodies():
    decision = AdjudicationDecision(
        decision_id="decision-1",
        task_id="task-1",
        outcome="fixed",
        evidence_ids=["evidence-1"],
        operation_ids=["operation-1"],
        decided_at="2026-08-29T00:00:00+00:00",
    )
    assert not hasattr(decision, "evidence")
    assert not hasattr(decision, "test_results")
```

- [ ] **Step 2: Run the model tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_models.py -q`

Expected: FAIL because `deepfix.task_domain.models` does not exist.

- [ ] **Step 3: Implement strict frozen contracts**

Use these exact fields:

```python
class TaskDefinition(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    original_message_id: str = Field(min_length=1)
    original_problem: str = Field(min_length=1)
    approval_mode: str = Field(min_length=1)
    source_project_root: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    workspace_baseline_id: str | None = None
    project_python: str = Field(min_length=1)
    confinement_level: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class TaskLifecycle(StrictModel):
    task_id: str = Field(min_length=1)
    status: TaskLifecycleStatus
    version: int = Field(ge=1)
    paused_from: TaskLifecycleStatus | None = None
    reason: str | None = None
    updated_at: str = Field(min_length=1)


class AdjudicationDecision(StrictModel):
    decision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    outcome: Literal["fixed", "not_reproduced", "paused", "failed", "cancelled"]
    evidence_ids: list[str] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    decided_at: str = Field(min_length=1)
```

Define the allowed lifecycle transition map in the same file. Terminal states have no outgoing transitions; `PAUSED` may return only to `RUNNING` or terminate.

- [ ] **Step 4: Run model tests and Ruff**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_models.py -q`

Run: `.venv\Scripts\ruff check src/deepfix/task_domain tests/task_domain/test_models.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/deepfix/task_domain tests/task_domain/test_models.py
git commit -m "feat: define bounded task domain contracts"
```

---

### Task 3: Narrow TaskRepository for Definition and Lifecycle

**Files:**
- Create: `src/deepfix/task_domain/repository.py`
- Create: `tests/task_domain/test_repository.py`
- Modify: `src/deepfix/task_domain/__init__.py`
- Modify: `src/deepfix/persistence.py`

**Interfaces:**
- Consumes: `SQLiteDatabase`, `TaskDefinition`, `TaskLifecycle`, `TaskLifecycleStatus`.
- Produces: `TaskRepository.create_definition()`, `get_definition()`, `get_lifecycle()`, and `transition_lifecycle()`.

- [ ] **Step 1: Write failing repository authority tests**

```python
def test_create_definition_is_idempotent_but_rejects_mutation(repository, definition):
    assert repository.create_definition(definition) == definition
    assert repository.create_definition(definition) == definition
    changed = definition.model_copy(update={"original_problem": "changed"})
    with pytest.raises(TaskDefinitionConflict):
        repository.create_definition(changed)


def test_definition_and_lifecycle_are_separate_rows(repository, definition):
    repository.create_definition(definition)
    lifecycle = repository.get_lifecycle(definition.task_id)
    assert lifecycle.status is TaskLifecycleStatus.CREATED
    assert lifecycle.version == 1


def test_lifecycle_uses_optimistic_version_and_transition_rules(repository, definition):
    repository.create_definition(definition)
    running = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
        expected_version=1,
    )
    assert running.version == 2
    with pytest.raises(TaskLifecycleConflict):
        repository.transition_lifecycle(
            definition.task_id,
            TaskLifecycleStatus.COMPLETED,
            expected_version=1,
        )
```

Also query `sqlite_master` and assert that Task tables do not contain columns named `messages`, `todos`, `evidence`, `receipts`, `operations`, `snapshots`, or `report`.

- [ ] **Step 2: Run repository tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_repository.py -q`

Expected: FAIL because the narrow repository is absent.

- [ ] **Step 3: Add normalized Task tables and narrow operations**

Create only these tables in this task:

```sql
CREATE TABLE IF NOT EXISTS task_definitions (
    task_id TEXT PRIMARY KEY,
    original_message_id TEXT NOT NULL UNIQUE,
    original_problem TEXT NOT NULL,
    approval_mode TEXT NOT NULL,
    source_project_root TEXT NOT NULL,
    workspace_root TEXT NOT NULL,
    workspace_baseline_id TEXT,
    project_python TEXT NOT NULL,
    confinement_level TEXT NOT NULL,
    created_at TEXT NOT NULL,
    definition_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_lifecycle (
    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    paused_from TEXT,
    reason TEXT,
    updated_at TEXT NOT NULL
);
```

`create_definition()` runs in one UnitOfWork, compares `definition_hash` on replay, and inserts the initial lifecycle row atomically. `transition_lifecycle()` uses:

```sql
UPDATE task_lifecycle
SET status = ?, version = version + 1, paused_from = ?, reason = ?, updated_at = ?
WHERE task_id = ? AND version = ?
```

Raise `TaskLifecycleConflict` when the row count is not exactly one.

- [ ] **Step 4: Preserve the public import path without creating a generic repository**

`deepfix.persistence.TaskRepository` becomes a compatibility subclass or re-export of `deepfix.task_domain.repository.TaskRepository`. Keep `checkpoint_connection()` delegating to `SQLiteDatabase.connection()` because the CLI uses the connection for LangGraph `SqliteSaver`.

- [ ] **Step 5: Run focused persistence tests**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_repository.py tests/task_domain/test_database.py tests/test_persistence.py -q`

Expected: new tests PASS; legacy persistence tests may remain RED only where Task 4 intentionally replaces whole-payload behavior.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/deepfix/task_domain src/deepfix/persistence.py tests/task_domain/test_repository.py
git commit -m "feat: add immutable task repository boundary"
```

---

### Task 4: Field-Level Legacy Migration and Read Projection

**Files:**
- Create: `src/deepfix/task_domain/migration.py`
- Create: `tests/task_domain/test_migration.py`
- Modify: `src/deepfix/task_domain/repository.py`
- Modify: `src/deepfix/persistence.py`
- Modify: `tests/test_persistence.py`

**Interfaces:**
- Consumes: legacy `TaskState.to_dict()`/`from_dict()`, stable message identity, normalized definition/lifecycle APIs.
- Produces: `LegacyTaskProjection`, `backfill_legacy_task()`, `reconstruct_task_state()`, `TaskRepository.save_legacy_projection()`, compatibility `save()`, `get()`, and `list_recent()`.

- [ ] **Step 1: Write failing field-level migration tests**

```python
def test_backfill_derives_original_message_id_and_is_idempotent(repository, task):
    repository.save_legacy_projection(task)
    first = repository.get_definition(task.task_id)
    repository.save_legacy_projection(task)
    second = repository.get_definition(task.task_id)
    assert first == second
    assert first.original_message_id == task.conversation[0]["id"]


def test_legacy_save_cannot_overwrite_immutable_problem(repository, task):
    repository.save_legacy_projection(task)
    task.user_problem = "模型改写后的目标"
    with pytest.raises(TaskDefinitionConflict):
        repository.save_legacy_projection(task)


def test_unmigrated_fields_continue_to_round_trip(repository, task):
    task.hypotheses = ["H1"]
    task.changed_files = ["src/value.py"]
    task.pending_actions = [{"name": "execute", "args": {"command": "pytest"}}]
    repository.save_legacy_projection(task)
    restored = repository.get(task.task_id)
    assert restored.hypotheses == ["H1"]
    assert restored.changed_files == ["src/value.py"]
    assert restored.pending_actions == task.pending_actions


def test_legacy_payload_does_not_own_migrated_definition_fields(repository, task):
    repository.save_legacy_projection(task)
    payload = repository.read_legacy_payload_for_test(task.task_id)
    for key in (
        "user_problem",
        "approval_mode",
        "source_project_root",
        "workspace_root",
        "workspace_baseline_id",
        "project_python",
        "confinement_level",
    ):
        assert key not in payload
```

Add a migration test for a historical `tasks(payload)` row with no message ID; assert the deterministic ID is stable across repeated backfills.

- [ ] **Step 2: Run migration tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_migration.py tests/test_persistence.py -q`

Expected: FAIL because the legacy decomposition adapter does not exist.

- [ ] **Step 3: Implement explicit field ownership sets**

```python
TASK_DEFINITION_FIELDS = frozenset({
    "task_id",
    "project_root",
    "user_problem",
    "approval_mode",
    "source_project_root",
    "workspace_root",
    "workspace_baseline_id",
    "project_python",
    "confinement_level",
})

TASK_LIFECYCLE_FIELDS = frozenset({"status", "paused_from", "pause_reason"})

TASK_POLICY_REFERENCE_FIELDS = frozenset({
    "verification_policy_id",
    "verification_policy_version",
})

PLAN_2_LEGACY_FIELDS = frozenset({
    "conversation",
    "evidence",
    "hypotheses",
    "diagnosis",
    "repair_plan",
    "changed_files",
    "successful_changed_files",
    "latest_change_verification",
    "test_results",
    "approvals",
    "review",
    "final_summary",
    "residual_risks",
    "unverified_items",
    "pending_question",
    "pending_actions",
    "processed_tool_call_ids",
    "shell_calls",
    "agent_invocations",
    "consecutive_test_failures",
    "working_memory_version",
    "context_metrics",
    "offloaded_artifacts",
    "external_evidence_ids",
    "research_query_count",
    "research_provider_errors",
    "context_recovery",
    "investigation_recovery",
    "unresolved_operation_ids",
    "required_oracle_count",
    "passed_required_oracle_count",
    "supplemental_failure_count",
    "resolution",
})
```

Persist legacy-only data in:

```sql
CREATE TABLE IF NOT EXISTS legacy_task_projection (
    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
    legacy_phase_status TEXT,
    payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

The compatibility writer rejects unknown TaskState fields rather than silently expanding TaskRepository ownership. It stores the temporary legacy phase separately so current Phase-dependent code can reconstruct `TaskState.status` while normalized lifecycle remains authoritative.

`reconstruct_task_state()` restores `project_root` from the immutable
`workspace_root`, restores the latest policy ID/version from
`verification_policies`, and obtains lifecycle terminal/waiting state from
`task_lifecycle`. It uses `legacy_phase_status` only while the canonical
lifecycle is `RUNNING`, or to preserve the temporary `CLARIFYING` compatibility
projection while canonical lifecycle is `PAUSED`.

- [ ] **Step 4: Map legacy phase statuses to business lifecycle**

Use this exact mapping:

```python
LEGACY_TO_LIFECYCLE = {
    TaskStatus.CREATED: TaskLifecycleStatus.CREATED,
    TaskStatus.CLARIFYING: TaskLifecycleStatus.PAUSED,
    TaskStatus.INVESTIGATING: TaskLifecycleStatus.RUNNING,
    TaskStatus.EDITING: TaskLifecycleStatus.RUNNING,
    TaskStatus.TESTING: TaskLifecycleStatus.RUNNING,
    TaskStatus.REVIEWING: TaskLifecycleStatus.RUNNING,
    TaskStatus.WAITING_APPROVAL: TaskLifecycleStatus.WAITING_APPROVAL,
    TaskStatus.PAUSED: TaskLifecycleStatus.PAUSED,
    TaskStatus.COMPLETED: TaskLifecycleStatus.COMPLETED,
    TaskStatus.FAILED: TaskLifecycleStatus.FAILED,
    TaskStatus.CANCELLED: TaskLifecycleStatus.CANCELLED,
}
```

Do not treat a switch among `INVESTIGATING`, `EDITING`, `TESTING`, and `REVIEWING` as a lifecycle transition or lifecycle version increment.

- [ ] **Step 5: Backfill existing `tasks` rows without deleting them**

On the first read or save of a legacy-only task:

1. parse the historical payload;
2. assign stable conversation IDs through `TaskState.from_dict()`;
3. derive `original_message_id` from the first original User Message;
4. insert definition, lifecycle, and legacy projection in one transaction;
5. validate definition hash and payload hash by rereading;
6. leave the old `tasks` row untouched as a rollback source.

Never dual-write the old `tasks.payload` after successful backfill.

If a historical payload has no conversation entry, derive the same deterministic
ordinal-zero User Message ID that `BugfixService.start()` would produce from
`task_id`, role `user`, and the immutable original problem. Do not generate a
random migration-only source ID.

- [ ] **Step 6: Run migration and compatibility tests**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_migration.py tests/test_persistence.py tests/compaction/test_migration.py tests/investigation/test_migration.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/deepfix/task_domain src/deepfix/persistence.py tests/task_domain/test_migration.py tests/test_persistence.py
git commit -m "feat: migrate task payload fields incrementally"
```

---

### Task 5: VerificationPolicy Ownership Without Workspace-Scope Duplication

**Files:**
- Modify: `src/deepfix/task_domain/repository.py`
- Modify: `src/deepfix/verification.py`
- Modify: `tests/task_domain/test_repository.py`
- Modify: `tests/test_verification.py`

**Interfaces:**
- Consumes: existing `VerificationPolicy`, `VerificationPolicyConflict`.
- Produces: `TaskRepository.save_verification_policy()` and `load_verification_policy()`; compatibility `VerificationPolicyStore` delegates to those methods.

- [ ] **Step 1: Write failing policy ownership and scope tests**

```python
def test_task_repository_rejects_required_oracle_downgrade(repository, policy):
    repository.save_verification_policy(policy)
    downgraded = policy.model_copy(update={
        "version": 2,
        "required_oracles": [],
        "supplemental_oracles": policy.required_oracles,
    })
    with pytest.raises(VerificationPolicyConflict):
        repository.save_verification_policy(downgraded)


def test_verification_policy_has_no_workspace_allowed_paths(policy):
    assert "allowed_paths" not in VerificationPolicy.model_fields
    assert "allowed_paths" not in policy.model_dump()


def test_relevant_paths_do_not_authorize_workspace_mutation(policy):
    oracle = policy.required_oracles[0].model_copy(
        update={"relevant_paths": ["tests/test_value.py"]}
    )
    assert oracle.relevant_paths == ["tests/test_value.py"]
    assert not hasattr(oracle, "can_modify")
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_repository.py tests/test_verification.py -q`

Expected: repository policy tests FAIL because the narrow methods are absent.

- [ ] **Step 3: Move policy persistence behind TaskRepository**

Reuse the existing `verification_policies` schema and downgrade algorithm. Accept an optional `connection` only as a private repository helper so callers inside an existing UnitOfWork do not open another connection. Keep the public methods narrow and model-based.

Avoid an import cycle: `task_domain.repository` imports `VerificationPolicy`
under `TYPE_CHECKING` and performs the runtime model import only inside
`load_verification_policy()`. `verification.py` may then import the repository
for its compatibility facade.

Change `VerificationPolicyStore` to:

```python
class VerificationPolicyStore:
    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        tasks: TaskRepository | None = None,
    ) -> None:
        if tasks is None and database_path is None:
            raise ValueError("database_path or tasks is required")
        self.tasks = tasks or TaskRepository(database_path)

    def save(self, policy: VerificationPolicy) -> None:
        self.tasks.save_verification_policy(policy)

    def load(self, task_id: str, version: int | None = None) -> VerificationPolicy | None:
        return self.tasks.load_verification_policy(task_id, version)
```

This facade remains until Plan 4 removes the independent Store name.

- [ ] **Step 4: Verify policy compatibility and Workspace ownership**

Run: `.venv\Scripts\python -m pytest tests/test_verification.py tests/test_workspace.py tests/test_backend.py -q`

Expected: PASS; no `allowed_paths` field is introduced and existing confinement tests remain green.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/deepfix/task_domain/repository.py src/deepfix/verification.py tests/task_domain/test_repository.py tests/test_verification.py
git commit -m "refactor: move verification policy behind task boundary"
```

---

### Task 6: Atomic Task Budget Ledger

**Files:**
- Modify: `src/deepfix/task_domain/repository.py`
- Modify: `src/deepfix/investigation/token_budget.py`
- Modify: `tests/task_domain/test_repository.py`
- Modify: `tests/investigation/test_token_budget.py`

**Interfaces:**
- Consumes: existing `TokenBalance`, `TokenUsage`, `TokenReservation`, `TokenBudgetConflict`, and `TokenBudgetExhausted`.
- Produces: TaskRepository-owned budget ledger helpers; compatibility `TokenBudgetStore` retains its current public API and middleware behavior.

- [ ] **Step 1: Add failing atomicity and identity tests**

```python
def test_parallel_reservations_cannot_oversubscribe_budget(tmp_path):
    store = TokenBudgetStore(tmp_path / "deepfix.db")
    store.initialize("task-1", input_cap=100, output_cap=50)
    barrier = Barrier(2)

    def reserve(call_id):
        barrier.wait()
        try:
            store.reserve(
                "task-1",
                call_id,
                input_tokens=70,
                output_tokens=30,
            )
            return "reserved"
        except TokenBudgetExhausted:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ["call-1", "call-2"]))
    assert sorted(results) == ["blocked", "reserved"]


def test_reservation_replay_is_idempotent_but_conflicting_payload_fails(tmp_path):
    store = TokenBudgetStore(tmp_path / "deepfix.db")
    store.initialize("task-1", input_cap=100, output_cap=50)
    first = store.reserve("task-1", "call-1", input_tokens=20, output_tokens=10)
    assert store.reserve(
        "task-1", "call-1", input_tokens=20, output_tokens=10
    ) == first
    with pytest.raises(TokenBudgetConflict):
        store.reserve("task-1", "call-1", input_tokens=21, output_tokens=10)
```

- [ ] **Step 2: Run budget tests before refactor**

Run: `.venv\Scripts\python -m pytest tests/investigation/test_token_budget.py -q`

Expected: existing tests PASS; new concurrency test demonstrates whether the current store already meets the invariant. If it passes, keep it as a characterization test and proceed with delegation without changing semantics.

- [ ] **Step 3: Move budget SQL behind the bounded Task repository infrastructure**

Retain the existing `token_budgets` and `token_reservations` schemas, stable reservation ID, `BEGIN IMMEDIATE`, settlement, unknown-usage charging, and breach behavior. `TokenBudgetStore` delegates to a Task budget ledger object created from the same `SQLiteDatabase`; `TokenBudgetMiddleware` and `TokenBudgetCallbackHandler` signatures do not change.

No model call may be placed inside a UnitOfWork. The only atomic sections are initialize, reserve, settle, and charge-unknown database operations.

- [ ] **Step 4: Run budget and evaluation regressions**

Run: `.venv\Scripts\python -m pytest tests/investigation/test_token_budget.py tests/evaluation/test_experiment_runner.py tests/evaluation/test_harness.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src/deepfix/task_domain/repository.py src/deepfix/investigation/token_budget.py tests/task_domain/test_repository.py tests/investigation/test_token_budget.py
git commit -m "refactor: consolidate atomic task budget ledger"
```

---

### Task 7: Adjudication Records and Service/CLI Wiring

**Files:**
- Modify: `src/deepfix/task_domain/repository.py`
- Modify: `src/deepfix/service.py`
- Modify: `src/deepfix/cli.py`
- Modify: `tests/task_domain/test_repository.py`
- Create: `tests/task_domain/test_service_integration.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 1–6 interfaces, existing `TaskState`, `BugfixService`, and CLI construction.
- Produces: canonical new-task creation, business lifecycle synchronization, adjudication references, and one shared `SQLiteDatabase` per CLI process.

- [ ] **Step 1: Write failing adjudication and service integration tests**

```python
def test_adjudication_persists_only_supporting_ids(repository, definition):
    repository.create_definition(definition)
    decision = AdjudicationDecision(
        decision_id="decision-1",
        task_id=definition.task_id,
        outcome="fixed",
        evidence_ids=["evidence-1"],
        operation_ids=["operation-1"],
        decided_at="2026-08-29T00:00:00+00:00",
    )
    repository.record_adjudication(decision)
    assert repository.latest_adjudication(definition.task_id) == decision


def test_service_creates_definition_from_original_user_message(service):
    task = service.start("原始修复问题")
    definition = service.repository.get_definition(task.task_id)
    assert definition.original_problem == "原始修复问题"
    assert definition.original_message_id == task.conversation[0]["id"]


def test_later_user_message_does_not_rewrite_definition(service):
    task = service.start("原始修复问题")
    service.continue_task(task.task_id, "补充约束")
    definition = service.repository.get_definition(task.task_id)
    assert definition.original_problem == "原始修复问题"
```

Add integration assertions that `TaskLifecycleStatus` is `RUNNING`, `WAITING_APPROVAL`, `PAUSED`, or terminal at the corresponding existing service boundaries, while reconstructed `TaskState.status` continues to satisfy current CLI and legacy Phase tests.

```python
def test_legacy_phase_changes_do_not_increment_business_lifecycle(service):
    task = service.start("修复失败测试")
    before = service.repository.get_lifecycle(task.task_id)
    task.status = TaskStatus.EDITING
    service.repository.save_legacy_projection(task)
    after = service.repository.get_lifecycle(task.task_id)
    assert after.status is TaskLifecycleStatus.RUNNING
    assert after.version == before.version
    assert service.repository.get(task.task_id).status is TaskStatus.EDITING
```

- [ ] **Step 2: Run integration tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_service_integration.py tests/task_domain/test_repository.py -q`

Expected: FAIL because service and adjudication wiring are absent.

- [ ] **Step 3: Add adjudication persistence**

Create:

```sql
CREATE TABLE IF NOT EXISTS adjudication_decisions (
    decision_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_definitions(task_id),
    outcome TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    operation_ids_json TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
```

Replay with the same ID and content is idempotent; changed content for an existing ID raises `AdjudicationDecisionConflict`. Do not store Evidence bodies, test output, Receipt bodies, or report text.

- [ ] **Step 4: Wire immutable task creation and field-level saves**

In `BugfixService.start()`:

1. create the workspace and VerificationPolicy as today;
2. generate and append the original HumanMessage ID;
3. build `TaskDefinition` from the final task/workspace configuration;
4. call `create_definition()`;
5. save the policy through the Task boundary;
6. transition lifecycle from `CREATED` to `RUNNING`;
7. call `save_legacy_projection()` for only unmigrated fields.

In `_save()`, keep `_sync_context()` during Plan 2, then call `save_legacy_projection()`. The compatibility adapter maps legacy Phase statuses to business lifecycle without treating phase-only changes as lifecycle progress.

- [ ] **Step 5: Construct one shared database in CLI**

```python
database = SQLiteDatabase(state_database_path())
repository = TaskRepository(database)
verification_policy_store = VerificationPolicyStore(tasks=repository)
token_budget_store = TokenBudgetStore(database=database)
with database.connection() as connection:
    checkpointer = SqliteSaver(connection)
```

The LangGraph connection stays open for the graph lifetime. Repository UnitOfWork connections are shorter-lived and never share an active transaction with the graph invocation.

- [ ] **Step 6: Run service and CLI regressions**

Run: `.venv\Scripts\python -m pytest tests/task_domain/test_service_integration.py tests/test_service.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/deepfix/task_domain/repository.py src/deepfix/service.py src/deepfix/cli.py tests/task_domain/test_repository.py tests/task_domain/test_service_integration.py tests/test_service.py tests/test_cli.py
git commit -m "feat: wire bounded task authority into service"
```

---

### Task 8: Plan 2 Migration Gate and Documentation

**Files:**
- Modify: `docs/superpowers/plans/2026-08-28-deepfix-task-persistence-foundation.md`
- Modify: `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md`
- Test: all Plan 2 focused and core offline suites.

**Interfaces:**
- Consumes: all Plan 2 outputs.
- Produces: reviewed Plan 2 completion evidence and a frozen handoff contract for Plan 3.

- [x] **Step 1: Add a schema-boundary regression**

In `tests/task_domain/test_repository.py`, enumerate normalized Task tables and assert their columns cannot hold excluded domains. Also scan production code and assert no new `TaskRepository.save(TaskState)` implementation writes `json.dumps(task.to_dict())` into the historical `tasks` table.

```python
def test_normalized_task_tables_exclude_other_domains(repository):
    forbidden = {
        "messages", "todos", "hypotheses", "evidence", "test_results",
        "changed_files", "operations", "receipts", "approvals",
        "snapshots", "artifacts", "summary", "report",
    }
    with repository.checkpoint_connection() as connection:
        for table in (
            "task_definitions",
            "task_lifecycle",
            "verification_policies",
            "adjudication_decisions",
            "token_budgets",
            "token_reservations",
        ):
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert columns.isdisjoint(forbidden)


def test_repository_never_rewrites_historical_whole_payload():
    source = Path("src/deepfix/task_domain/repository.py").read_text("utf-8")
    assert "json.dumps(task.to_dict())" not in source
    assert "UPDATE tasks" not in source
```

- [x] **Step 2: Add legacy rollback-window verification**

In `tests/task_domain/test_migration.py`, create a historical `tasks` row, backfill it, modify only an unmigrated legacy field, and assert:

- normalized definition hash is unchanged;
- normalized lifecycle remains authoritative;
- compatibility read reflects the unmigrated update;
- the original historical row still exists but was not updated after backfill.

```python
def test_backfill_keeps_historical_row_read_only(repository, legacy_task_row):
    before = repository.read_historical_row_for_test(legacy_task_row.task_id)
    task = repository.get(legacy_task_row.task_id)
    definition = repository.get_definition(task.task_id)
    task.hypotheses = ["H1"]
    repository.save_legacy_projection(task)
    after = repository.read_historical_row_for_test(legacy_task_row.task_id)
    assert after == before
    assert repository.get_definition(task.task_id) == definition
    assert repository.get(task.task_id).hypotheses == ["H1"]
```

- [x] **Step 3: Run the Plan 2 focused gate**

Run:

```powershell
.venv\Scripts\python -m pytest `
  tests/task_domain `
  tests/test_persistence.py `
  tests/test_verification.py `
  tests/investigation/test_token_budget.py `
  tests/test_service.py `
  tests/test_cli.py -q
```

Expected: PASS.

- [x] **Step 4: Run the trusted execution and navigation regression gate**

Run:

```powershell
.venv\Scripts\python -m pytest `
  tests/navigation `
  tests/test_workspace.py `
  tests/test_backend.py `
  tests/test_execution.py `
  tests/test_operations.py `
  tests/test_approval.py `
  tests/investigation/test_receipts.py `
  tests/compaction -q
```

Expected: PASS. This verifies that Plan 2 did not weaken Todo, Workspace, approval, Receipt, Journal, or compaction invariants.

- [x] **Step 5: Run the core offline suite and static checks**

Run: `.venv\Scripts\python -m pytest -q`

Run: `.venv\Scripts\ruff check src tests`

Run: `git diff --check`

Expected: all offline tests PASS; Ruff and diff checks exit 0. Do not run the online QuixBugs campaign.

- [x] **Step 6: Record exact completion evidence**

Update this document with:

- commit IDs for Tasks 1–7;
- exact focused/core test counts;
- confirmation that no `allowed_paths` field was added;
- confirmation that legacy retirement was field-by-field;
- confirmation that no Evidence/Execution/History tables were moved early;
- any rollback-window tables intentionally retained for Plan 4.

- [x] **Step 7: Mark Plan 2 complete in the parent program and commit**

```powershell
git add docs/superpowers/plans/2026-08-28-deepfix-task-persistence-foundation.md docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md
git commit -m "docs: record task persistence foundation completion"
```

## Plan 2 Completion Evidence — 2026-08-29

### Task commits

| Task | Commit | Result |
|---|---|---|
| 1 | `6f72211` | shared SQLite connection and UnitOfWork |
| 2 | `ef6f586` | immutable Task Definition and business lifecycle contracts |
| 3 | `4fb9b47` | bounded normalized TaskRepository |
| 4 | `755e001` | field-level legacy migration and read projection |
| 5 | `0ad7729` | VerificationPolicy authority behind Task boundary |
| 6 | `39e18f8` | atomic token budget/reservation ledger |
| 7 | `eb490ce` | Service/CLI wiring and Adjudication persistence |

### Normalized authority and compatibility boundary

The normalized Task boundary owns exactly:

```text
task_definitions
task_lifecycle
verification_policies
adjudication_decisions
token_budgets
token_reservations
```

`VerificationPolicy` has no `allowed_paths` field. Workspace mutation scope
remains owned by `TaskWorkspace` and execution confinement;
`VerificationOracle.relevant_paths` remains relevance metadata only.

Legacy retirement was field-by-field. Immutable definition fields, business
lifecycle fields, VerificationPolicy ID/version, and final `resolution` now
project from normalized authority. The temporary `legacy_task_projection`
continues to carry unmigrated conversation, hypotheses, evidence/test copies,
changed-file copies, approvals, investigation/recovery state, compaction
metrics, Artifact references, research metadata, and report fields. The
historical `tasks` table is retained read-only as the rollback source and is
never dual-written after backfill.

No Evidence, Investigation, Execution/Receipt, Message/Todo, Snapshot,
Artifact, or History table was moved into the Task domain. Those migrations
remain Plan 3 work.

### Migration and rollback evidence

- `tests/task_domain/test_migration.py`: **12 passed**.
- Historical whole-payload backfill preserves the original row, immutable
  definition, business lifecycle version, and unmigrated-field updates.
- Historical rows without a conversation receive the stable ordinal-zero User
  Message ID; historical completed rows backfill a canonical Adjudication.
- First backfill inserts definition, lifecycle, and projection in one
  transaction. Fault injection proves a projection failure rolls the entire
  migration back.
- Definition and legacy projection hashes are reread and compared before the
  migration transaction commits.

### Verification commands and exact results

- Plan 2 focused gate: **135 passed**.
- Trusted execution/navigation gate: **254 passed, 2 failed**. Both failures
  are the pre-existing Windows sandbox `_overlapped` / `WinError 10106`
  failures in the two long-context workflow tests; no new failure appeared.
- The repository's literal `pytest -q` command remains blocked at collection
  by duplicate `test_models.py` module names. Running the full offline suite
  with `PYTHONPATH=src;tests` and `--import-mode=importlib`, while explicitly
  deselecting the two known Windows environment failures, produced
  **969 passed, 2 skipped, 4 deselected**.
- The three initially exposed investigation middleware regressions were traced
  to a test fixture that mutated `workspace_baseline_id` after immutable task
  creation. Initializing that baseline before the first save made all three
  regression tests pass without weakening Task Definition immutability.
- `ruff check src tests`: PASS.
- `git diff --check`: PASS.
- No online QuixBugs campaign or other paid acceptance run was executed.

### Plan 3 handoff

Plan 3 must consolidate Evidence/Research, Investigation,
Execution/Receipt/Approval, and History/Compaction repositories before Plan 4
can remove `legacy_task_projection`, the historical `tasks` rollback table,
WorkingMemoryStore, Phase compatibility, and copied report fields.

## Plan 2 Review Checkpoint

Stop after Task 8. Report:

- normalized Task table list and ownership;
- compatibility fields intentionally retained;
- migration/backfill counts and hash validation results;
- focused and core test totals;
- remaining Plan 3 dependencies.

Do not start Plan 3 until the user reviews and approves this checkpoint.
