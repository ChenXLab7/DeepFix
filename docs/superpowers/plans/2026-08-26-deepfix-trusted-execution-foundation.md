# DeepFix Trusted Execution Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make side effects isolated, journaled, recoverable, scope-checked, and verifiable before increasing Agent autonomy.

**Architecture:** Introduce a per-task Workspace with a frozen baseline, canonical path/command policies, an append-only Operation Journal around existing Tool Receipts, and deterministic recovery reconciliation. Extend test evidence with origin/scope/timing and create a frozen VerificationPolicy used by outcome grading. Integrate these foundations into the existing legacy loop first so they are independently valuable and tested.

**Tech Stack:** Python 3.11+, pathlib/shutil/subprocess, SQLite/WAL, Pydantic 2, existing LocalShellBackend/InvestigationMiddleware/CompactionStore, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-26-deepfix-capability-oriented-agent-loop-design.md`

## Global Constraints

- New-loop side effects never target the source project directory; `TaskState` records source root, workspace root, and baseline ID separately.
- File paths are canonicalized after resolving `..`, symlink, and Windows reparse/junction behavior; an allowed lexical path with an outside canonical target is denied.
- `cwd` alone is not strong isolation. Local execution is labeled `guarded_local`; arbitrary unattended code requires a future/available `strict` runner.
- `guarded_local` pytest/project-code execution requires an existing approval grant. The evaluation harness may supply a system-created trusted-corpus grant for the fixed QuixBugs source; model text cannot create that grant.
- Journal order is `prepared → started → observed → committed`; an already-started side effect is never automatically replayed.
- Tool Receipt remains the result authority. Journal stores lifecycle, hashes, and references, not duplicate unbounded output.
- VerificationPolicy freezes before the first edit. Required Oracle cannot be silently removed or downgraded.
- No Experiment Loop code is introduced in this plan.

## File Structure

### New production files

- `src/deepfix/workspace.py` — workspace baseline, isolated copy/worktree creation, and canonical scope policy.
- `src/deepfix/execution.py` — confinement levels, guarded command policy, and workspace-bound runner.
- `src/deepfix/operations.py` — operation models, SQLite journal, and recovery reconciler.
- `src/deepfix/verification.py` — test evidence classifier, VerificationPolicy builder, and oracle evaluation.

### Existing production files modified

- `src/deepfix/models.py` — persist source/workspace/baseline and VerificationPolicy references on TaskState.
- `src/deepfix/config.py` — distinguish source root from active workspace root without adding model settings.
- `src/deepfix/backend.py` — execute through WorkspaceCommandRunner with task-specific environment.
- `src/deepfix/investigation/receipts.py` — allow Receipt reconstruction from observed journal artifacts.
- `src/deepfix/investigation/middleware.py` — journal side-effect lifecycle around the existing handler/Receipt boundary.
- `src/deepfix/compaction/models.py` — enrich SystemTestEvidence and FileChangeEvidence with workspace/code-state data.
- `src/deepfix/compaction/evidence.py` — deterministic test origin/scope/timing and file-hash collection.
- `src/deepfix/service.py` — create/load workspace and reconcile incomplete operations before Agent invocation.
- `src/deepfix/reporting.py` — show workspace, oracle coverage, and confinement level.

### Tests

- `tests/test_workspace.py`
- `tests/test_execution.py`
- `tests/test_operations.py`
- `tests/test_verification.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/investigation/test_receipts.py`
- Modify: `tests/investigation/test_middleware.py`
- Modify: `tests/compaction/test_evidence.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_reporting.py`

---

### Task 1: Task Workspace and Frozen Baseline

**Files:**
- Create: `src/deepfix/workspace.py`
- Create: `tests/test_workspace.py`
- Modify: `src/deepfix/models.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `TaskState.task_id`, original project root, DeepFix artifact root.
- Produces: `WorkspaceBaseline`, `TaskWorkspace`, `WorkspaceFactory.create(task_id, source_root) -> TaskWorkspace`, `compute_code_state_hash(workspace) -> str`.

- [ ] **Step 1: Write failing baseline/isolation tests**

```python
# tests/test_workspace.py
from deepfix.workspace import WorkspaceFactory, compute_code_state_hash


def test_workspace_copy_preserves_source_and_records_baseline(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
    factory = WorkspaceFactory(tmp_path / "deepfix-workspaces")

    workspace = factory.create("task-1", source)
    (workspace.root / "bug.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert (source / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert workspace.baseline.source_root == str(source.resolve())
    assert workspace.baseline.baseline_id
    assert compute_code_state_hash(workspace.root) != workspace.baseline.code_state_hash


def test_reopening_same_task_validates_baseline(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.py").write_text("x = 1\n", encoding="utf-8")
    factory = WorkspaceFactory(tmp_path / "deepfix-workspaces")
    first = factory.create("task-1", source)
    second = factory.load("task-1")
    assert second.baseline == first.baseline
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_workspace.py -q`

Expected: FAIL because `deepfix.workspace` does not exist.

- [ ] **Step 3: Implement baseline models and isolated-copy factory**

```python
# src/deepfix/workspace.py
class WorkspaceBaseline(StrictModel):
    baseline_id: str
    task_id: str
    source_root: str
    workspace_root: str
    git_head: str | None
    source_dirty_fingerprint: str | None
    code_state_hash: str
    managed_file_hashes: dict[str, str]
    python_fingerprint: str


class TaskWorkspace(StrictModel):
    task_id: str
    root: Path
    baseline: WorkspaceBaseline


class WorkspaceFactory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def create(self, task_id: str, source_root: str | Path) -> TaskWorkspace:
        source = Path(source_root).expanduser().resolve(strict=True)
        target = self.root / _task_segment(task_id)
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(source, target, ignore=_copy_ignore)
        hashes = _managed_file_hashes(target)
        baseline = WorkspaceBaseline(
            baseline_id=_baseline_id(task_id, source, hashes),
            task_id=task_id,
            source_root=str(source),
            workspace_root=str(target),
            git_head=_git_head(source),
            source_dirty_fingerprint=_git_dirty_fingerprint(source),
            code_state_hash=_hash_manifest(hashes),
            managed_file_hashes=hashes,
            python_fingerprint=_python_fingerprint(),
        )
        _atomic_write_json(target / ".deepfix-baseline.json", baseline.model_dump_json())
        return TaskWorkspace(task_id=task_id, root=target, baseline=baseline)

    def load(self, task_id: str) -> TaskWorkspace:
        target = (self.root / _task_segment(task_id)).resolve(strict=True)
        path = target / ".deepfix-baseline.json"
        baseline = WorkspaceBaseline.model_validate_json(path.read_text(encoding="utf-8"))
        if baseline.task_id != task_id or Path(baseline.workspace_root) != target:
            raise WorkspaceBaselineError("workspace baseline identity mismatch")
        return TaskWorkspace(task_id=task_id, root=target, baseline=baseline)
```

Implement the named private helpers `_task_segment`, `_managed_file_hashes`, `_baseline_id`, `_hash_manifest`, `_git_head`, `_git_dirty_fingerprint`, `_python_fingerprint`, `_copy_ignore`, and `_atomic_write_json`. Exclude `.git`, `.deepfix`, `__pycache__`, `.pytest_cache`, `.venv`, and DeepFix artifact directories from the copy/hash manifest. A clean Git source may use `git worktree add --detach`; dirty/non-Git sources use the isolated copy path.

- [ ] **Step 4: Persist workspace identity on TaskState**

Add `source_project_root`, `workspace_root`, and `workspace_baseline_id` fields. Keep `project_root` as a backward-compatible alias during this plan; new tasks set it to the workspace root. Add round-trip tests in `tests/test_models.py`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_workspace.py tests/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Commit workspace foundation**

```bash
git add src/deepfix/workspace.py src/deepfix/models.py tests/test_workspace.py tests/test_models.py
git commit -m "feat: isolate task workspaces"
```

### Task 2: Canonical Path Policy and Guarded Command Confinement

**Files:**
- Modify: `src/deepfix/workspace.py`
- Create: `src/deepfix/execution.py`
- Create: `tests/test_execution.py`
- Modify: `tests/test_workspace.py`

**Interfaces:**
- Produces: `WorkspacePathPolicy.resolve_allowed(path) -> Path`, `ConfinementLevel`, `CommandDecision`, `WorkspaceCommandPolicy.evaluate(command)`, `WorkspaceCommandRunner.execute(command, timeout) -> ExecuteResponse`.

- [ ] **Step 1: Write failing path escape tests**

```python
def test_parent_escape_is_rejected(workspace_policy, tmp_path) -> None:
    with pytest.raises(WorkspaceScopeError):
        workspace_policy.resolve_allowed("../outside.txt")


def test_symlink_to_outside_is_rejected(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    policy = WorkspacePathPolicy(workspace)
    with pytest.raises(WorkspaceScopeError):
        policy.resolve_allowed("link/value.py")
```

- [ ] **Step 2: Write failing command policy tests**

```python
@pytest.mark.parametrize("command", [
    "rm ../outside.txt",
    "python C:/outside/script.py",
    "git config --global user.name agent",
    "python -m pip install demo",
])
def test_guarded_policy_rejects_workspace_escape(command, workspace) -> None:
    decision = WorkspaceCommandPolicy(workspace).evaluate(command)
    assert decision.allowed is False


def test_guarded_policy_allows_scoped_pytest(workspace) -> None:
    decision = WorkspaceCommandPolicy(workspace).evaluate(
        "python -m pytest tests/test_value.py -q"
    )
    assert decision.allowed is True
    assert decision.confinement_level is ConfinementLevel.GUARDED_LOCAL
    assert decision.requires_approval is True
```

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/test_workspace.py tests/test_execution.py -q`

Expected: FAIL because policy types are missing.

- [ ] **Step 4: Implement path and command policies**

```python
class ConfinementLevel(StrEnum):
    GUARDED_LOCAL = "guarded_local"
    STRICT = "strict"


class CommandDecision(StrictModel):
    allowed: bool
    reason: str
    confinement_level: ConfinementLevel
    requires_approval: bool


class WorkspaceCommandRunner:
    def __init__(self, workspace: TaskWorkspace, policy: WorkspaceCommandPolicy) -> None:
        self.workspace = workspace
        self.policy = policy

    def execute(
        self,
        command: str,
        *,
        timeout: int,
        approval_grant: ApprovalGrant | None,
    ) -> ExecuteResponse:
        decision = self.policy.evaluate(command)
        if not decision.allowed:
            return ExecuteResponse(output=f"Denied: {decision.reason}", exit_code=126)
        if decision.requires_approval and not valid_grant(approval_grant, command):
            return ExecuteResponse(output="Denied: approval required", exit_code=126)
        env = task_scoped_environment(self.workspace)
        return run_managed_process(command, cwd=self.workspace.root, env=env, timeout=timeout)
```

The path policy must compare `Path.resolve(strict=False)` with the canonical allowed root using `os.path.commonpath`, reject symlink/reparse parents outside the root, and re-check immediately before file replacement. The environment redirects HOME/USERPROFILE, XDG cache/config, TEMP/TMP, PIP cache/config, and Git global config to task-owned paths. `ApprovalGrant` is system-created from the existing approval record and binds task ID, normalized command hash, and one execution; the fixed evaluation harness may create the same structure only after verifying the committed case-manifest/source hashes.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_workspace.py tests/test_execution.py -q`

Expected: PASS.

```bash
git add src/deepfix/workspace.py src/deepfix/execution.py tests/test_workspace.py tests/test_execution.py
git commit -m "feat: confine workspace paths and commands"
```

### Task 3: Operation Journal State Machine

**Files:**
- Create: `src/deepfix/operations.py`
- Create: `tests/test_operations.py`

**Interfaces:**
- Consumes: SQLite database path, WorkspaceBaseline, tool-call identity.
- Produces: `OperationJournalEntry`, `OperationJournalStore.prepare`, `.mark_started`, `.observe`, `.commit`, `.list_incomplete`.

- [ ] **Step 1: Write failing transition and idempotency tests**

```python
def test_journal_requires_prepared_before_started(tmp_path) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    with pytest.raises(OperationTransitionError):
        store.mark_started("missing")


def test_prepare_is_idempotent_for_same_call_hash(tmp_path, prepared_operation) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    first = store.prepare(prepared_operation)
    second = store.prepare(prepared_operation)
    assert second == first


def test_prepare_rejects_same_id_with_different_hash(tmp_path, prepared_operation) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    store.prepare(prepared_operation)
    changed = prepared_operation.model_copy(update={"call_hash": "different"})
    with pytest.raises(OperationConflictError):
        store.prepare(changed)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_operations.py -q`

Expected: FAIL because journal code does not exist.

- [ ] **Step 3: Implement strict models and transactional Store**

```python
class OperationStatus(StrEnum):
    PREPARED = "prepared"
    STARTED = "started"
    OBSERVED = "observed"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


class OperationJournalStore:
    def prepare(self, entry: NewOperationEntry) -> OperationJournalEntry:
        return self._insert_prepared(entry)

    def mark_started(self, operation_id: str) -> OperationJournalEntry:
        return self._transition(operation_id, OperationStatus.PREPARED, OperationStatus.STARTED)

    def observe(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt_id: str,
        artifact_references: list[str],
    ) -> OperationJournalEntry:
        return self._observe(operation_id, post_state, receipt_id, artifact_references)

    def commit(self, operation_id: str) -> OperationJournalEntry:
        return self._transition(operation_id, OperationStatus.OBSERVED, OperationStatus.COMMITTED)

    def mark_unknown(self, operation_id: str, reason: str) -> OperationJournalEntry:
        return self._mark_unknown(operation_id, reason)

    def list_incomplete(self, task_id: str) -> list[OperationJournalEntry]:
        return self._list_by_status(task_id, exclude={OperationStatus.COMMITTED})
```

Use one SQLite transaction per transition, compare expected current state in the UPDATE predicate, and return the already-written row when the exact transition is replayed.

- [ ] **Step 4: Run transition/concurrency tests**

Add a `ThreadPoolExecutor` test that races two identical `prepare` calls and asserts one row. Then run:

Run: `python -m pytest tests/test_operations.py -q`

Expected: PASS.

- [ ] **Step 5: Commit journal**

```bash
git add src/deepfix/operations.py tests/test_operations.py
git commit -m "feat: journal side effect operations"
```

### Task 4: Journal the Existing Tool/Receipt Boundary

**Files:**
- Modify: `src/deepfix/investigation/middleware.py`
- Modify: `src/deepfix/investigation/receipts.py`
- Modify: `tests/investigation/test_middleware.py`
- Modify: `tests/investigation/test_receipts.py`

**Interfaces:**
- Consumes: `OperationJournalStore`, existing `ToolExecutionReceiptStore`, canonical Workspace state snapshots.
- Produces: side-effect middleware order `prepare → start → handler once → artifact/receipt → observe → evidence/event → commit`.

- [ ] **Step 1: Add a failing crash-after-handler test**

```python
def test_receipt_failure_after_edit_does_not_reexecute_handler(
    middleware_factory,
    edit_request,
) -> None:
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        edit_request.target.write_text("fixed\n", encoding="utf-8")
        return success_tool_message(edit_request)

    middleware = middleware_factory(receipt_save_error=RuntimeError("disk full"))
    with pytest.raises(InvestigationCoordinationError):
        middleware.wrap_tool_call(edit_request.request, handler)
    with pytest.raises(InvestigationCoordinationError):
        middleware.wrap_tool_call(edit_request.request, handler)

    assert calls == 1
    assert middleware.journal.list_incomplete(edit_request.task_id)[0].status == "started"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/investigation/test_middleware.py -q -k journal`

Expected: FAIL because the handler is replayed or no journal exists.

- [ ] **Step 3: Implement journaling around side-effect tools**

Classify `write_file`, `edit_file`, `delete`, and `execute` as journaled. Write `prepared` before handler invocation and `started` immediately before it. Capture bounded command output to an internal Artifact before Receipt save. For file tools, capture canonical target and pre/post hashes. On replay:

```python
if entry.status is OperationStatus.COMMITTED:
    return receipts.load_required(task_id, call_id).tool_message
if entry.status in {OperationStatus.STARTED, OperationStatus.UNKNOWN}:
    raise recovery_error("operation_reconciliation_required", entry)
```

- [ ] **Step 4: Run receipt/middleware tests**

Run: `python -m pytest tests/investigation/test_receipts.py tests/investigation/test_middleware.py -q`

Expected: PASS, including existing two-parallel-read regression tests.

- [ ] **Step 5: Commit middleware integration**

```bash
git add src/deepfix/investigation/middleware.py src/deepfix/investigation/receipts.py tests/investigation/test_middleware.py tests/investigation/test_receipts.py
git commit -m "feat: journal tool side effects before execution"
```

### Task 5: Recovery Reconciliation

**Files:**
- Modify: `src/deepfix/operations.py`
- Modify: `tests/test_operations.py`
- Modify: `src/deepfix/service.py`
- Modify: `tests/test_service.py`

**Interfaces:**
- Produces: `OperationReconciler.reconcile_task(task_id, workspace) -> ReconciliationResult`; Service refuses Agent invocation while unresolved operations remain.

- [ ] **Step 1: Write failing reconstruction/conflict tests**

```python
def test_file_post_hash_reconstructs_observed_receipt(reconciler_fixture) -> None:
    fixture = reconciler_fixture.started_file_edit()
    fixture.target.write_text("fixed\n", encoding="utf-8")
    result = fixture.reconciler.reconcile_task(fixture.task_id, fixture.workspace)
    assert result.reconstructed_operation_ids == [fixture.operation_id]
    assert fixture.handler_calls == 0


def test_unexpected_file_hash_pauses_recovery(reconciler_fixture) -> None:
    fixture = reconciler_fixture.started_file_edit()
    fixture.target.write_text("third-party change\n", encoding="utf-8")
    result = fixture.reconciler.reconcile_task(fixture.task_id, fixture.workspace)
    assert result.conflict_operation_ids == [fixture.operation_id]
```

- [ ] **Step 2: Verify RED, implement reconciler, verify PASS**

Run before implementation: `python -m pytest tests/test_operations.py -q -k reconcile`

Expected: FAIL.

Implement file reconciliation from pre/post/current hashes; command reconstruction only when a durable output Artifact includes exit status; otherwise mark unknown and terminate the managed process group. Never invoke the original handler.

Run after implementation: `python -m pytest tests/test_operations.py tests/test_service.py -q`

Expected: PASS.

- [ ] **Step 3: Commit recovery**

```bash
git add src/deepfix/operations.py src/deepfix/service.py tests/test_operations.py tests/test_service.py
git commit -m "feat: reconcile interrupted side effects"
```

### Task 6: Test Evidence Trust and VerificationPolicy

**Files:**
- Create: `src/deepfix/verification.py`
- Create: `tests/test_verification.py`
- Modify: `src/deepfix/compaction/models.py`
- Modify: `src/deepfix/compaction/evidence.py`
- Modify: `tests/compaction/test_evidence.py`
- Modify: `src/deepfix/models.py`

**Interfaces:**
- Produces: enriched `SystemTestEvidence`, `VerificationOracle`, `VerificationPolicy`, `VerificationPolicyBuilder.build(task, project)`, `evaluate_required_oracles(policy, evidence) -> OracleEvaluation`.

- [ ] **Step 1: Write failing deterministic classification tests**

```python
def test_existing_user_test_is_required_targeted_oracle(project, task) -> None:
    policy = VerificationPolicyBuilder().build(task, project)
    oracle = policy.required_oracles[0]
    assert oracle.origin == "user_specified"
    assert oracle.scope == "targeted"


def test_agent_created_test_cannot_be_required(project, task) -> None:
    generated = project / "tests" / "test_generated.py"
    generated.write_text("def test_x(): assert True\n", encoding="utf-8")
    policy = VerificationPolicyBuilder().build(task, project)
    assert all(o.command.find("test_generated.py") < 0 for o in policy.required_oracles)


def test_policy_cannot_downgrade_failed_required_oracle(policy_store, policy) -> None:
    policy_store.save(policy)
    changed = policy.model_copy(update={
        "version": policy.version + 1,
        "required_oracles": [],
        "supplemental_oracles": policy.required_oracles,
    })
    with pytest.raises(VerificationPolicyConflict):
        policy_store.save(changed)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_verification.py tests/compaction/test_evidence.py -q`

Expected: FAIL because verification contracts/fields are missing.

- [ ] **Step 3: Extend deterministic test evidence**

Add `origin`, `scope`, `timing`, `workspace_baseline_id`, `code_state_hash`, `test_target_paths`, and `test_content_hashes` exactly as frozen in the spec. Classify origin from user-message provenance and baseline file hashes; classify timing by comparing code state with baseline/latest successful edit; classify scope from pytest targets/collection rather than model text.

- [ ] **Step 4: Implement and freeze VerificationPolicy**

```python
class VerificationPolicyBuilder:
    def build(self, task: TaskState, workspace: TaskWorkspace) -> VerificationPolicy:
        user = required_user_oracles(task, workspace)
        repository = repository_oracles(workspace)
        required = user + required_repository_oracles(task, repository)
        supplemental = supplemental_repository_oracles(task, repository)
        return VerificationPolicy(
            policy_id=stable_verification_id(task.task_id, required, supplemental),
            task_id=task.task_id,
            version=1,
            required_oracles=required,
            supplemental_oracles=supplemental,
            conflict_rules=default_oracle_conflict_rules(),
        )
```

Persist policy ID/version on TaskState before the first edit. Required-unavailable returns an evaluation that cannot produce FIXED. A related post-change module/full-suite failure blocks FIXED even when targeted required tests pass.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_verification.py tests/compaction/test_evidence.py tests/test_models.py -q`

Expected: PASS.

```bash
git add src/deepfix/verification.py src/deepfix/compaction/models.py src/deepfix/compaction/evidence.py src/deepfix/models.py tests/test_verification.py tests/compaction/test_evidence.py tests/test_models.py
git commit -m "feat: add trusted test oracles"
```

### Task 7: Backend, Service, and Reporting Integration

**Files:**
- Modify: `src/deepfix/config.py`
- Modify: `src/deepfix/backend.py`
- Modify: `src/deepfix/service.py`
- Modify: `src/deepfix/reporting.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: every new task runs in TaskWorkspace; existing loop uses guarded runner/journal/policy; report exposes workspace/confinement/oracle state.

- [ ] **Step 1: Add a failing end-to-end service test**

```python
def test_service_creates_workspace_before_agent_invocation(service_factory, project) -> None:
    service, agent = service_factory(source_project=project)
    task = service.start("fix value and run repository tests")
    assert Path(task.workspace_root).parent != project.parent
    assert agent.invocations[0].config["configurable"]["workspace_root"] == task.workspace_root
    assert task.verification_policy_id
```

- [ ] **Step 2: Verify RED, wire components, verify PASS**

Run before implementation: `python -m pytest tests/test_service.py -q -k workspace`

Expected: FAIL.

Update `build_backend` to receive TaskWorkspace/runner, Service to create/load/reconcile before invoke, and reporting to render source/workspace, baseline, confinement, required Oracle pass count, supplemental failures, and unresolved operations.

Run after implementation:

`python -m pytest tests/test_config.py tests/test_backend.py tests/test_service.py tests/test_reporting.py -q`

Expected: PASS.

- [ ] **Step 3: Run foundation regression and Ruff**

Run: `python -m pytest tests/test_workspace.py tests/test_execution.py tests/test_operations.py tests/test_verification.py tests/investigation tests/compaction/test_evidence.py tests/test_backend.py tests/test_service.py tests/test_reporting.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix tests/test_workspace.py tests/test_execution.py tests/test_operations.py tests/test_verification.py`

Expected: PASS.

- [ ] **Step 4: Commit integration**

```bash
git add src/deepfix/config.py src/deepfix/backend.py src/deepfix/service.py src/deepfix/reporting.py tests/test_config.py tests/test_backend.py tests/test_service.py tests/test_reporting.py
git commit -m "feat: use trusted task execution foundation"
```

## Plan 2 Completion Checkpoint

Stop for review. Demonstrate crash-after-edit recovery without a second handler call, required-oracle conflict blocking FIXED, parent/symlink/global-command rejection, and unchanged legacy-loop behavior on an ordinary targeted pytest repair.
