# DeepFix Evidence-Preserving Context Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DeepFix's fixed natural-language summarization path with a task-isolated, evidence-preserving compaction coordinator that keeps complete work units, stable identities, structured snapshots, adaptive budgets, and recoverable failure behavior.

**Architecture:** DeepFix owns message identity, WorkUnit partitioning, protected-context projection, budget selection, CompactionDelta validation, deterministic Snapshot merging, lifecycle persistence, event submission, and Overflow recovery. Deep Agents remains responsible for Backend/Artifact routing, LangChain XML history serialization primitives, large ToolMessage/media offload, Tool integration, and the existing model instance; no Deep Agents private summary, cutoff, or event method is used on the normal path.

**Tech Stack:** Python 3.11+, Deep Agents `>=0.7,<0.8`, LangChain/LangGraph middleware and messages, Pydantic 2, SQLite WAL, pytest 8+, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-22-deepfix-context-compaction-design.md`

## Global Constraints

- Work only in `C:\Users\17823\Documents\AI Agent\deepfix-agent` on branch `codex/deepfix-single-agent`.
- Do not add a second Agent or subagent; this remains the single Repair Agent architecture.
- Use red-green-refactor for every behavior: run the focused test and observe the expected failure before editing production code.
- Preserve `deepagents>=0.7,<0.8`, `pydantic>=2,<3`, Python `>=3.11`, SQLite WAL, and the existing `/.deepfix-artifacts/` route.
- Never call Deep Agents private summary/cutoff/event methods: `_create_summary`, `_acreate_summary`, `_determine_cutoff_index`, `_lc_helper`, or `_summarization_event` on the normal path.
- The only permitted legacy read is a migration adapter for already serialized `_summarization_event` state.
- Protected Context is request-local System content and must never be appended to Graph `messages`.
- User constraints, deterministic test/file/approval evidence, and Research Store state keep their type-specific authority; model output can only propose sourced semantic candidates.
- No message removal or `_deepfix_compaction_event` submission occurs until the full history artifact and prepared Snapshot are verified.
- Middleware and Coordinator never import or mutate `TaskStatus`; only `BugfixService` may transition a task to `PAUSED`.
- Default tests use fake models and controlled backends; they do not call DeepSeek or the network.
- Commit after every task with only that task's files staged.

---

## File Structure

Create a focused `deepfix.compaction` package rather than extending the already mixed `context.py`:

```text
src/deepfix/
├── compaction/
│   ├── __init__.py       # public compaction interfaces
│   ├── models.py         # immutable Pydantic domain records and events
│   ├── identity.py       # deterministic message/claim/hypothesis/work-unit IDs
│   ├── errors.py         # preparation errors and Service-facing recovery errors
│   ├── work_units.py     # pure WorkUnitPartitioner
│   ├── budget.py         # ContextBudgetMonitor and retention selector
│   ├── store.py          # Snapshot, deterministic-evidence, failure-ledger persistence
│   ├── snapshot.py       # Delta generation contract, validation, deterministic merge
│   ├── adapter.py        # Backend/history serialization compatibility boundary
│   ├── coordinator.py    # automatic/manual transaction and Overflow retry
│   └── middleware.py     # Graph identity and automatic coordination middleware
├── protected_context.py  # authoritative reads, deduplicated projection, System blocks
├── context.py            # structured save_progress tool and middleware assembly
├── memory.py             # structured Working Memory and coverage persistence
├── models.py             # TaskState recovery/evidence compatibility fields
├── service.py            # recovery exception boundary and deterministic ledger sync
└── agent.py              # replace Deep Agents summarization middleware registration
```

Tests mirror those responsibilities under `tests/compaction/`, while existing context, memory, Agent, Service, model, persistence, reporting, CLI, backend, and research tests remain regression gates.

---

### Task 1: Define compaction domain models and typed errors

**Files:**

- Create: `src/deepfix/compaction/__init__.py`
- Create: `src/deepfix/compaction/models.py`
- Create: `src/deepfix/compaction/errors.py`
- Create: `tests/compaction/__init__.py`
- Create: `tests/compaction/test_models.py`
- Create: `tests/compaction/test_errors.py`

**Interfaces:**

- Produces: `ProvenanceRef`, `FactCandidate`, `ProvenancedClaim`, `HypothesisProgressInput`, `HypothesisRecord`, `SnapshotCoverage`, `TaskAnchor`, `WorkUnit`, `CompactionDelta`, `CompactionSnapshot`, `DeepFixCompactionEvent`, `CompactionFailureRecord`, `ContextRecoveryMetadata`.
- Produces: `CompactionPreparationError` subclasses for internal handling and `ContextCoordinationError` subclasses for the Service boundary.

- [ ] **Step 1: Write model tests for strict candidate and lifecycle schemas**

```python
def test_delta_rejects_model_supplied_claim_id():
    with pytest.raises(ValidationError, match="claim_id"):
        CompactionDelta.model_validate({
            "confirmed_fact_candidates": [{
                "text": "pytest exits 1",
                "sources": [{"kind": "work_unit", "ref_id": "wu-1"}],
                "claim_id": "model-chosen",
            }],
            "user_constraint_candidates": [], "hypothesis_transitions": [],
            "experiments": [], "conflict_candidates": [],
            "unresolved_questions": [], "next_steps": [],
        })

def test_snapshot_lifecycle_timestamps_are_consistent(snapshot_payload):
    prepared = CompactionSnapshot.model_validate(snapshot_payload)
    assert prepared.lifecycle == "prepared"
    with pytest.raises(ValidationError):
        CompactionSnapshot.model_validate({**snapshot_payload, "lifecycle": "active", "activated_at": None})
```

- [ ] **Step 2: Run model tests and confirm import failure**

Run: `pytest tests/compaction/test_models.py -q`

Expected: FAIL because `deepfix.compaction.models` does not exist.

- [ ] **Step 3: Implement strict Pydantic models**

Use `ConfigDict(extra="forbid")` for all model-generated schemas. Define lifecycle validation explicitly:

```python
class CompactionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    version: int = Field(ge=1)
    previous_version: int | None = None
    lifecycle: Literal["prepared", "active", "abandoned"] = "prepared"
    created_at: str
    activated_at: str | None = None
    abandoned_at: str | None = None
    abandon_reason: str | None = None
    source_work_unit_ids: list[str]
    task_goal: str
    user_constraints: list[UserConstraint]
    confirmed_facts: list[ProvenancedClaim]
    deterministic_evidence: DeterministicEvidenceBlock
    active_hypotheses: list[HypothesisRecord]
    rejected_hypotheses: list[HypothesisRecord]
    confirmed_hypotheses: list[HypothesisRecord]
    changed_files: list[FileChangeEvidence]
    experiments: list[ExperimentRecord]
    test_results: list[SystemTestEvidence]
    conflicts: list[ConflictRecord]
    unresolved_questions: list[ProvenancedText]
    next_steps: list[ProvenancedText]
    artifact_references: list[ArtifactReference]
    content_hash: str
```

Make `FactCandidate` contain only `text` and `sources`; put `claim_id` only on `ProvenancedClaim`. Add `reopens_hypothesis_id` to hypothesis input, transition, and stored record.

Define the event and failure boundary fields exactly:

```python
class TaskAnchor(BaseModel):
    task_id: str
    task_goal: str
    user_constraints: list[UserConstraint]
    latest_user_message_id: str
    project_root: str
    project_python: str
    task_status: str

class DeepFixCompactionEvent(BaseModel):
    event_id: str
    task_id: str
    active_snapshot_version: int
    snapshot_message_id: str
    retained_message_ids: list[str]
    conversation_artifact: ArtifactReference
    input_hash: str

class CompactionFailureRecord(BaseModel):
    attempt_id: str
    task_id: str
    entrypoint: Literal["automatic", "manual_tool", "overflow_recovery"]
    budget_zone: Literal["normal", "observe", "normal_compaction", "emergency"]
    stage: str
    error_code: str
    input_hash: str
    original_messages_preserved: bool
    artifact_reference: str | None
    prepared_snapshot_version: int | None
    recorded_at: str

class ContextRecoveryMetadata(BaseModel):
    task_id: str
    stage: Literal[
        "protected_context", "artifact_write", "artifact_verify",
        "delta_generation", "snapshot_validate", "snapshot_write",
        "snapshot_verify", "compacted_model_call", "overflow_retry",
    ]
    error_code: str
    usage_ratio: float | None
    working_memory_version: int | None
    active_snapshot_version: int | None
    prepared_snapshot_version: int | None
    prepared_snapshot_lifecycle: Literal["prepared", "active", "abandoned"] | None
    conversation_artifact: str | None
    original_messages_preserved: bool
```

- [ ] **Step 4: Write and implement error-boundary tests**

```python
def test_preparation_error_is_not_service_recovery_error(failure_record):
    error = ArtifactPersistenceError(failure_record)
    assert isinstance(error, CompactionPreparationError)
    assert not isinstance(error, ContextCoordinationError)

def test_recovery_error_carries_safe_metadata(recovery_metadata):
    error = ContextRecoveryRequired(recovery_metadata)
    assert error.recovery.original_messages_preserved is True
```

Implement `ProtectedContextLoadError` and `ContextRecoveryRequired` as `ContextCoordinationError`; implement `ArtifactPersistenceError`, `SnapshotBuildError`, and `SnapshotPersistenceError` as `CompactionPreparationError`.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_models.py tests/compaction/test_errors.py -q
ruff check src/deepfix/compaction tests/compaction
git add src/deepfix/compaction tests/compaction
git commit -m "feat: define compaction domain models"
```

Expected: all focused tests PASS and Ruff exits 0.

### Task 2: Assign stable identities to every Graph Message and semantic entity

**Files:**

- Create: `src/deepfix/compaction/identity.py`
- Create: `tests/compaction/test_identity.py`
- Modify: `src/deepfix/compaction/models.py`

**Interfaces:**

- Produces: `ensure_message_ids(task_id: str, messages: Sequence[AnyMessage]) -> MessageIdentityResult`.
- Produces: `stable_claim_id`, `stable_hypothesis_id`, `stable_reopened_hypothesis_id`, `stable_work_unit_id`, `stable_generated_message_id`.
- Consumed by: every later WorkUnit, provenance, coverage, history, Snapshot, and event task.

- [ ] **Step 1: Write failing all-message identity tests**

```python
@pytest.mark.parametrize("message", [
    HumanMessage(content="bug"),
    AIMessage(content="inspect", tool_calls=[{"name": "read_file", "args": {"file_path": "a.py"}, "id": "c1"}]),
    ToolMessage(content="body", tool_call_id="c1"),
    AIMessage(content="result explanation"),
])
def test_missing_id_is_deterministic_for_every_message_type(message):
    first = ensure_message_ids("task-a", [message]).messages[0]
    second = ensure_message_ids("task-a", [message]).messages[0]
    assert first.id == second.id
    assert first.id.startswith("msg_")

def test_existing_duplicate_ids_are_reused_but_reported_ambiguous():
    result = ensure_message_ids("task-a", [HumanMessage(id="same", content="a"), AIMessage(id="same", content="b")])
    assert [m.id for m in result.messages] == ["same", "same"]
    assert result.conflicted_message_ids == {"same"}
```

- [ ] **Step 2: Run identity tests and confirm symbol failure**

Run: `pytest tests/compaction/test_identity.py -q`

Expected: FAIL because `ensure_message_ids` is undefined.

- [ ] **Step 3: Implement canonical serialization and IDs**

```python
def stable_message_id(task_id: str, ordinal: int, message: AnyMessage) -> str:
    tool_ids = sorted(_tool_call_ids(message))
    content_hash = _sha256(_canonical_content(message.content))
    material = "|".join((task_id, str(ordinal), message.type, ",".join(tool_ids), content_hash))
    return f"msg_{_sha256(material)[:32]}"
```

Normalize only CRLF/CR to LF; serialize structured content with sorted compact JSON. Store `_deepfix_original_ordinal` in `additional_kwargs`, copy rather than mutate caller-owned message objects, reuse all existing IDs, and return conflicts without rewriting them.

Return a frozen carrier with exact fields:

```python
@dataclass(frozen=True)
class MessageIdentityResult:
    messages: list[AnyMessage]
    assigned_message_ids: tuple[str, ...]
    conflicted_message_ids: frozenset[str]
```

- [ ] **Step 4: Add semantic ID tests and implementations**

```python
def test_claim_identity_does_not_change_when_sources_grow():
    assert stable_claim_id("task-a", " pytest exits 1 ") == stable_claim_id("task-a", "pytest exits 1")

def test_reopened_hypothesis_has_new_linked_identity():
    reopened = stable_reopened_hypothesis_id("task-a", "hyp-old", "ev-new", "cache stale")
    assert reopened != "hyp-old"
    assert reopened == stable_reopened_hypothesis_id("task-a", "hyp-old", "ev-new", "cache stale")
```

Use versioned normalization material (`claim:v1`, `hypothesis:v1`) so future normalization changes require explicit migration.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_identity.py -q
ruff check src/deepfix/compaction/identity.py tests/compaction/test_identity.py
git add src/deepfix/compaction/identity.py src/deepfix/compaction/models.py tests/compaction/test_identity.py
git commit -m "feat: assign stable context identities"
```

### Task 3: Upgrade Working Memory and `save_progress` to structured identities

**Files:**

- Modify: `src/deepfix/memory.py`
- Modify: `src/deepfix/context.py`
- Modify: `src/deepfix/service.py`
- Modify: `tests/test_memory.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_service.py`

**Interfaces:**

- Produces: structured `ProgressSnapshot.facts`, three hypothesis state projections, and `coverage: SnapshotCoverage`.
- Produces: `WorkingMemoryStore.save_progress(task_id, snapshot, valid_source_ids) -> WorkingMemoryVersion`.
- Preserves: `WorkingMemoryStore.save()` legacy input migration for old stored string snapshots.

- [ ] **Step 1: Write failing hypothesis transition and reopen tests**

```python
def test_save_progress_transitions_existing_hypothesis_without_text_matching(store):
    first = store.save_progress("task-a", progress(hypotheses=[new_hyp("cache stale", "msg-1")]), {"msg-1"})
    hyp_id = first.snapshot.active_hypotheses[0].hypothesis_id
    second = store.save_progress("task-a", progress(hypotheses=[transition(hyp_id, "rejected", "mtime changed", "msg-2")]), {"msg-2"})
    assert second.snapshot.rejected_hypotheses[0].hypothesis_id == hyp_id

def test_reopen_requires_new_evidence_and_keeps_old_rejection(store, rejected_version):
    with pytest.raises(ValueError, match="new evidence"):
        store.save_progress("task-a", progress(hypotheses=[reopen(rejected_version.id, sources=[])]), set())
```

- [ ] **Step 2: Run focused memory tests and confirm schema failures**

Run: `pytest tests/test_memory.py tests/test_context.py -q`

Expected: FAIL because snapshots still contain string facts/hypotheses and no coverage.

- [ ] **Step 3: Implement structured memory normalization**

Add Pydantic `model_validator(mode="before")` migration for persisted string lists only. Runtime `save_progress` must reject ambiguous combinations, validate every supplied `ProvenanceRef(kind="user_message"|"work_unit")` against `valid_source_ids`, generate IDs in the Store, and return resolved IDs:

```python
artifact={
    "version": saved.version,
    "hypothesis_ids": [item.hypothesis_id for item in saved.snapshot.all_hypotheses()],
    "claim_ids": [item.claim_id for item in saved.snapshot.facts],
}
```

Build coverage from stable runtime message IDs and completed WorkUnit IDs; do not accept task ID, claim ID, coverage, or canonical source paths as public Tool arguments.

- [ ] **Step 4: Render every Working Memory field category**

Extend `render_working_memory` with bounded sections for confirmed facts, active/rejected/confirmed hypotheses and reasons, evidence, checked files, experiments, next steps, unresolved questions, and coverage. Assert omitted entries emit `omitted_count`, content hash, and a Store version reference rather than silently disappearing.

- [ ] **Step 5: Preserve Service synchronization semantics**

Change `_sync_context` to read structured active hypotheses (`item.text`) without promoting rejected hypotheses to `TaskState.hypotheses` or any hypothesis to diagnosis. Add backward-compatibility round-trip tests for existing SQLite JSON.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/test_memory.py tests/test_context.py tests/test_service.py -q
ruff check src/deepfix/memory.py src/deepfix/context.py src/deepfix/service.py tests/test_memory.py tests/test_context.py tests/test_service.py
git add src/deepfix/memory.py src/deepfix/context.py src/deepfix/service.py tests/test_memory.py tests/test_context.py tests/test_service.py
git commit -m "feat: structure working memory identities"
```

### Task 4: Partition complete and parallel Tool work units

**Files:**

- Create: `src/deepfix/compaction/work_units.py`
- Create: `tests/compaction/test_work_units.py`

**Interfaces:**

- Produces: `partition_work_units(messages: Sequence[AnyMessage], conflicted_message_ids: set[str]) -> WorkUnitPartition`.
- Produces stable `WorkUnit.unit_id` from ordered message IDs and diagnostics for incomplete/ambiguous regions.

- [ ] **Step 1: Write failing single and parallel Tool unit tests**

```python
def test_parallel_calls_and_out_of_order_results_form_one_unit():
    messages = [
        AIMessage(id="m1", content="inspect", tool_calls=[call("read_file", "c1"), call("grep", "c2")]),
        ToolMessage(id="m2", content="grep result", tool_call_id="c2"),
        ToolMessage(id="m3", content="file result", tool_call_id="c1"),
        AIMessage(id="m4", content="both point to parser"),
    ]
    unit = partition_work_units(messages, set()).units[0]
    assert unit.message_ids == ["m1", "m2", "m3", "m4"]
    assert unit.tool_call_ids == ["c1", "c2"]
    assert unit.state == "complete"
```

- [ ] **Step 2: Write failing safety tests for malformed histories**

Cover missing Tool results, orphan ToolMessages, duplicate Tool Call IDs, duplicate message IDs, results crossing a later user turn, and a large-result artifact pointer. Assert each unsafe region is `incomplete` or `ambiguous` and `must_keep is True`.

- [ ] **Step 3: Run focused tests and confirm import failure**

Run: `pytest tests/compaction/test_work_units.py -q`

Expected: FAIL because `work_units.py` does not exist.

- [ ] **Step 4: Implement the pure partitioner**

Scan in original order, group an AIMessage's complete set of Tool Call IDs with all paired ToolMessages and consecutive Assistant explanation messages, stop at the next user message or Tool-calling AIMessage, and create conversational units for ordinary dialogue. Never modify messages and never infer a missing result.

Return `WorkUnitPartition(units: list[WorkUnit], unassigned_message_ids: list[str], diagnostics: list[str], safe_cut_indices: list[int])`; every `WorkUnit` includes `must_keep: bool` in addition to the spec fields.

- [ ] **Step 5: Add property test for indivisible cut points**

Generate deterministic permutations with `random.Random(0)` and assert every reported safe cutoff lies between units and never places a Tool call and its paired result on opposite sides.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_work_units.py -q
ruff check src/deepfix/compaction/work_units.py tests/compaction/test_work_units.py
git add src/deepfix/compaction/work_units.py tests/compaction/test_work_units.py
git commit -m "feat: partition complete tool work units"
```

### Task 5: Persist deterministic evidence, Snapshot lifecycle, and failure records

**Files:**

- Create: `src/deepfix/compaction/store.py`
- Create: `tests/compaction/test_store.py`
- Modify: `src/deepfix/models.py`
- Modify: `src/deepfix/memory.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_memory.py`
- Modify: `tests/test_persistence.py`

**Interfaces:**

- Produces: `CompactionStore.save_evidence`, `list_evidence`, `save_prepared_snapshot`, `get_snapshot`, `active_snapshot_from_event`, `activate_from_event`, `abandon_snapshot`, `record_failure`, `list_failures`.
- Produces additive `TaskState.context_recovery: ContextRecoveryMetadata | None` and expanded `ContextMetrics`.
- Preserves `WorkingMemoryStore` as the single owner of the existing `context_metrics` table; Coordinator and event reconciliation call its typed metric methods.

- [ ] **Step 1: Write failing additive-schema and isolation tests**

```python
def test_store_initialization_preserves_task_and_research_tables(database_with_task):
    store = CompactionStore(database_with_task)
    assert TaskRepository(database_with_task).list_recent()
    assert store.list_evidence("other-task") == []

def test_deterministic_evidence_is_idempotent_and_task_scoped(store, test_evidence):
    store.save_evidence(test_evidence)
    store.save_evidence(test_evidence)
    assert store.list_evidence("task-a") == [test_evidence]
    assert store.list_evidence("task-b") == []
```

- [ ] **Step 2: Write failing Snapshot lifecycle tests**

```python
def test_event_not_latest_row_selects_active_snapshot(store, snapshot_v1, snapshot_v2):
    store.save_prepared_snapshot(snapshot_v1)
    store.save_prepared_snapshot(snapshot_v2)
    event = DeepFixCompactionEvent(active_snapshot_version=1, **event_fields())
    assert store.active_snapshot_from_event("task-a", event).version == 1
    assert store.get_snapshot("task-a", 2).lifecycle == "prepared"
```

Also simulate event committed/SQLite lifecycle flag missing: `activate_from_event` changes prepared to active idempotently. Assert abandoned rows never participate in merge and semantic `content_hash` is unchanged by lifecycle transitions.

- [ ] **Step 3: Run focused Store tests and confirm import failure**

Run: `pytest tests/compaction/test_store.py tests/test_models.py tests/test_memory.py tests/test_persistence.py -q`

Expected: FAIL because `CompactionStore` and recovery fields do not exist.

- [ ] **Step 4: Implement tables and transactional methods**

Create additive tables with composite keys:

```sql
CREATE TABLE IF NOT EXISTS deterministic_evidence (
  task_id TEXT NOT NULL, evidence_id TEXT NOT NULL, kind TEXT NOT NULL,
  payload TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(task_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS compaction_snapshots (
  task_id TEXT NOT NULL, version INTEGER NOT NULL, lifecycle TEXT NOT NULL,
  input_hash TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
  activated_at TEXT, abandoned_at TEXT, abandon_reason TEXT,
  PRIMARY KEY(task_id, version), UNIQUE(task_id, input_hash)
);
CREATE TABLE IF NOT EXISTS compaction_failures (
  task_id TEXT NOT NULL, attempt_id TEXT NOT NULL, stage TEXT NOT NULL,
  payload TEXT NOT NULL, recorded_at TEXT NOT NULL,
  PRIMARY KEY(task_id, attempt_id, stage)
);
```

Use `BEGIN IMMEDIATE`, write/read hash verification, task ID on every query, and event-directed activation.

- [ ] **Step 5: Expand TaskState and metrics compatibly**

Add nullable recovery metadata plus budget zone, normal/emergency counts, failure, passthrough, manual error, overflow retry, active Snapshot version, last artifact, and last error fields. Update `TaskState.from_dict` so old payloads receive defaults. Extend `WorkingMemoryStore` with `record_budget`, `record_compaction_event`, `record_compaction_failure`, `record_passthrough`, `record_manual_error`, and `record_overflow_retry`; each performs one transactional update of the existing metrics row.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_store.py tests/test_models.py tests/test_memory.py tests/test_persistence.py -q
ruff check src/deepfix/compaction/store.py src/deepfix/models.py src/deepfix/memory.py tests/compaction/test_store.py tests/test_models.py tests/test_memory.py tests/test_persistence.py
git add src/deepfix/compaction/store.py src/deepfix/models.py src/deepfix/memory.py tests/compaction/test_store.py tests/test_models.py tests/test_memory.py tests/test_persistence.py
git commit -m "feat: persist compaction recovery state"
```

### Task 6: Collect current deterministic evidence from Graph and authoritative Stores

**Files:**

- Create: `src/deepfix/compaction/evidence.py`
- Create: `tests/compaction/test_evidence.py`
- Modify: `src/deepfix/models.py`
- Modify: `src/deepfix/service.py`
- Modify: `tests/test_service.py`

**Interfaces:**

- Produces: `EvidenceCollector.collect(task_id, messages, task_state) -> DeterministicEvidenceBlock`.
- Consumes: `CompactionStore`, `ResearchEvidenceStore`, paired stable message/Tool Call IDs, TaskState approval ledger.

- [ ] **Step 1: Write failing evidence-authority tests**

```python
def test_pytest_exit_code_comes_from_paired_tool_message(collector, task):
    block = collector.collect("task-a", execute_history(exit_code=1), task)
    assert block.tests[0].exit_code == 1
    assert block.tests[0].tool_call_id == "pytest-1"

def test_approved_target_is_not_reported_as_successful_file_change(collector, task):
    task.changed_files = ["src/calc.py"]
    block = collector.collect(task.task_id, [], task)
    assert block.files[0].status == "approved_target"
```

Add task-isolation tests for approvals and ResearchEvidenceStore. A model-authored memory claim that says “pytest passed” must not overwrite a nonzero system result.

- [ ] **Step 2: Run evidence tests and confirm import failure**

Run: `pytest tests/compaction/test_evidence.py -q`

Expected: FAIL because `EvidenceCollector` does not exist.

- [ ] **Step 3: Implement collection and idempotent ledger writes**

Parse only uniquely paired Tool calls/results. Record integer execute artifact `exit_code`; record successful write/edit/delete only when ToolMessage status/artifact confirms completion; merge TaskState approval records and Research Store verification states by their stable evidence IDs. Persist system records before returning the block.

- [ ] **Step 4: Extend Service's durable evidence provenance**

Add `tool_call_id` and `source_message_id` defaults to `TestResult`; update `_record_tool_results` to fill them from normalized Graph messages without duplicating prior IDs. Keep TaskState as a report projection; `CompactionStore` is the durable deterministic ledger.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_evidence.py tests/test_service.py tests/research/test_store.py -q
ruff check src/deepfix/compaction/evidence.py src/deepfix/models.py src/deepfix/service.py tests/compaction/test_evidence.py tests/test_service.py
git add src/deepfix/compaction/evidence.py src/deepfix/models.py src/deepfix/service.py tests/compaction/test_evidence.py tests/test_service.py
git commit -m "feat: collect deterministic repair evidence"
```

### Task 7: Implement adaptive request budgeting and whole-unit retention

**Files:**

- Create: `src/deepfix/compaction/budget.py`
- Create: `tests/compaction/test_budget.py`

**Interfaces:**

- Produces: `ContextBudgetMonitor.measure(request, protected_blocks) -> ContextBudgetReport`.
- Produces: `select_retained_units(report, units, latest_user_message_id) -> RetentionPlan`.

- [ ] **Step 1: Write exact threshold tests**

```python
@pytest.mark.parametrize((ratio, zone), [
    (0.75, "normal"), (0.75001, "observe"),
    (0.82, "observe"), (0.82001, "normal_compaction"),
    (0.90, "normal_compaction"), (0.90001, "emergency"),
])
def test_budget_boundaries(ratio, zone):
    assert classify_budget_zone(ratio) == zone
```

- [ ] **Step 2: Write request-total and retention-priority tests**

Use a deterministic fake token counter. Assert the total includes base System, all three protected blocks, effective messages, tools, and output reserve. Assert retention order: latest user, every incomplete/ambiguous unit, recent modify+verify, recent failed-test+analysis, then other complete units. Assert no partial unit is returned.

- [ ] **Step 3: Run budget tests and confirm import failure**

Run: `pytest tests/compaction/test_budget.py -q`

Expected: FAIL because `budget.py` does not exist.

- [ ] **Step 4: Implement budget resolution and observation hints**

Read `request.model.profile["max_input_tokens"]`; use an explicit, tested model-name table only if absent, and raise configuration error if neither source exists. Target `<=0.75` after normal compaction, `<=0.65` after emergency compaction, and `<=0.50` for Overflow recovery. Emit one stale-memory hint per `(working_memory_version, latest_work_unit_id)`.

Use exact immutable boundary records:

```python
@dataclass(frozen=True)
class ContextBudgetReport:
    request_tokens: int
    max_input_tokens: int
    output_reserve_tokens: int
    usage_ratio: float
    zone: BudgetZone
    target_ratio: float | None

@dataclass(frozen=True)
class RetentionPlan:
    retained_units: tuple[WorkUnit, ...]
    compressed_units: tuple[WorkUnit, ...]
    retained_message_ids: frozenset[str]
    estimated_ratio: float
```

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_budget.py -q
ruff check src/deepfix/compaction/budget.py tests/compaction/test_budget.py
git add src/deepfix/compaction/budget.py tests/compaction/test_budget.py
git commit -m "feat: budget context by complete work units"
```

### Task 8: Generate validated Deltas and merge structured Snapshots deterministically

**Files:**

- Create: `src/deepfix/compaction/snapshot.py`
- Create: `tests/compaction/test_snapshot.py`
- Create: `tests/compaction/test_snapshot_drift.py`

**Interfaces:**

- Produces: `CompactionDeltaGenerator.generate(model, work_units) -> CompactionDelta` using the Repair Agent's existing model instance.
- Produces: `CompactionSnapshotBuilder.build(BuildSnapshotInput) -> CompactionSnapshot`.
- Consumes: active Snapshot selected by event, latest Working Memory, Task Anchor, deterministic evidence, full compressed WorkUnits, and verified artifact references.

Define `BuildSnapshotInput` once in `snapshot.py` and use it unchanged in Coordinator:

```python
@dataclass(frozen=True)
class BuildSnapshotInput:
    task_id: str
    previous_snapshot: CompactionSnapshot | None
    compressed_units: tuple[WorkUnit, ...]
    latest_memory: WorkingMemoryVersion | None
    task_anchor: TaskAnchor
    deterministic_evidence: DeterministicEvidenceBlock
    delta: CompactionDelta
    artifact_reference: ArtifactReference
    input_hash: str
```

- [ ] **Step 1: Write failing candidate-authority tests**

```python
def test_builder_ignores_model_attempt_to_replace_system_test(builder, build_input):
    build_input.delta = delta_with_fact("pytest passed", source="wu-model")
    build_input.deterministic_evidence.tests = [system_test(exit_code=1)]
    snapshot = builder.build(build_input)
    assert snapshot.test_results[0].exit_code == 1
    assert any(item.information_type == "test" for item in snapshot.conflicts)

def test_user_constraint_requires_real_user_message_source(builder, build_input):
    build_input.delta.user_constraint_candidates = [constraint_candidate("never edit tests", "missing-message")]
    with pytest.raises(SnapshotBuildError, match="user message"):
        builder.build(build_input)
```

- [ ] **Step 2: Write failing identity and hypothesis merge tests**

Assert facts receive Builder-generated claim IDs, source growth preserves the ID, ordinary hypothesis transitions reuse IDs, reopen requires an old rejected hypothesis plus new evidence and creates a linked new ID, and semantic conflicts stay unresolved rather than overwriting either side.

- [ ] **Step 3: Run focused Snapshot tests and confirm import failure**

Run: `pytest tests/compaction/test_snapshot.py -q`

Expected: FAIL because `CompactionSnapshotBuilder` does not exist.

- [ ] **Step 4: Implement a narrow model Delta generator**

Use the same model object passed to `build_agent`:

```python
structured_model = model.with_structured_output(CompactionDelta)
delta = structured_model.invoke([
    SystemMessage(content=DELTA_EXTRACTION_PROMPT),
    HumanMessage(content=render_work_units_for_delta(work_units)),
])
return CompactionDelta.model_validate(delta)
```

The prompt must state that outputs are candidates, sources must refer to supplied IDs, and deterministic fields, task IDs, paths, versions, hashes, and canonical IDs are forbidden. Add an async twin using `ainvoke` without copying merge logic.

- [ ] **Step 5: Implement deterministic merge and validation**

Merge fields in this order: old active Snapshot history, Delta semantic candidates, latest Working Memory, then type-specific authoritative sources. Validate task IDs, WorkUnit coverage, source references, constraint source messages, evidence fingerprints, artifact hash, and content hash. Build `content_hash` from semantic payload only, excluding lifecycle/timestamps/hash.

- [ ] **Step 6: Prove three-plus compactions do not drift**

In `test_snapshot_drift.py`, feed three fake Deltas that paraphrase prior summaries. Assert the original user constraint text/source ID, rejected hypothesis reason/ID, and real pytest command/exit code are byte-for-byte unchanged; only provenance and new fields grow.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_snapshot.py tests/compaction/test_snapshot_drift.py -q
ruff check src/deepfix/compaction/snapshot.py tests/compaction/test_snapshot.py tests/compaction/test_snapshot_drift.py
git add src/deepfix/compaction/snapshot.py tests/compaction/test_snapshot.py tests/compaction/test_snapshot_drift.py
git commit -m "feat: merge structured compaction snapshots"
```

### Task 9: Add the public Backend/history adapter with write-before-remove verification

**Files:**

- Create: `src/deepfix/compaction/adapter.py`
- Create: `tests/compaction/test_adapter.py`
- Modify: `tests/test_backend.py`

**Interfaces:**

- Produces: `DeepAgentsArtifactAdapter.persist_history(task_id, attempt_id, messages, retained_ids) -> ArtifactReference` and equivalent `apersist_history`.
- Consumes only: Backend `download_files`, `write`, and `edit`; LangChain's public `get_buffer_string(messages, format="xml")` serializer.

- [ ] **Step 1: Write failing complete-history and idempotency tests**

```python
def test_history_contains_every_evicted_message_and_manifest(adapter, messages):
    ref = adapter.persist_history("task-a", "attempt-1", messages, {"m4"})
    body = adapter.read_verified(ref.path)
    assert all(message.id in body for message in messages)
    assert 'attempt_id="attempt-1"' in body
    assert '<retained_message id="m4"' in body

def test_same_attempt_does_not_append_twice(adapter, messages):
    first = adapter.persist_history("task-a", "attempt-1", messages, set())
    second = adapter.persist_history("task-a", "attempt-1", messages, set())
    assert first.content_hash == second.content_hash
    assert adapter.read_verified(first.path).count('attempt_id="attempt-1"') == 1
```

- [ ] **Step 2: Write failing backend error and verification tests**

Use controlled backends for `write` returning an error, `edit` throwing, read-after-write missing the event, and content hash mismatch. Every case must raise `ArtifactPersistenceError` and leave the input messages unchanged.

- [ ] **Step 3: Run adapter tests and confirm import failure**

Run: `pytest tests/compaction/test_adapter.py tests/test_backend.py -q`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 4: Implement append and verification without private summary helpers**

Append an XML-rendered section to `/.deepfix-artifacts/conversation_history/{task_id}.md`. Include attempt ID, ordered message IDs/content hashes, compressed and retained manifests, timestamp, and full serialized messages. Read raw bytes with `download_files`, use `write` for a missing artifact and `edit(old, old + section)` otherwise, then re-read and validate the exact event marker/hash.

Implement `apersist_history` with Backend async download/write/edit methods and the same pure serializer/hash verifier; do not run sync Backend I/O inside the async model wrapper.

- [ ] **Step 5: Add Deep Agents boundary sentinels**

In the adapter tests, monkeypatch `SummarizationMiddleware._create_summary`, `_acreate_summary`, `_determine_cutoff_index`, and `_offload_to_backend` to raise. Assert `persist_history` still passes. This proves the adapter reuses public Backend and XML serialization behavior without private summary/cutoff/event logic.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_adapter.py tests/test_backend.py -q
ruff check src/deepfix/compaction/adapter.py tests/compaction/test_adapter.py tests/test_backend.py
git add src/deepfix/compaction/adapter.py tests/compaction/test_adapter.py tests/test_backend.py
git commit -m "feat: verify recoverable conversation artifacts"
```

### Task 10: Build authoritative Protected Context with global deduplication

**Files:**

- Create: `src/deepfix/protected_context.py`
- Create: `tests/test_protected_context.py`
- Modify: `src/deepfix/context.py`
- Modify: `src/deepfix/models.py`
- Modify: `src/deepfix/service.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_models.py`

**Interfaces:**

- Produces: `ProtectedContextBuilder.build(task_id, request_messages, event) -> ProtectedContext`.
- Produces: `ProtectedContextProjector.project(context, snapshot) -> ProjectedContext` and `render_protected_context`.
- Consumes: `TaskRepository`, `WorkingMemoryStore`, `CompactionStore`, `ResearchEvidenceStore`, `EvidenceCollector`.

Define the protected boundary without Graph messages:

```python
@dataclass(frozen=True)
class ProtectedContext:
    task_anchor: TaskAnchor
    working_memory: WorkingMemoryVersion | None
    deterministic_evidence: DeterministicEvidenceBlock
    active_snapshot: CompactionSnapshot | None

@dataclass(frozen=True)
class ProjectedContext:
    task_anchor_xml: str
    working_memory_xml: str
    deterministic_evidence_xml: str
    visible_snapshot_xml: str | None
```

- [ ] **Step 1: Give every TaskState conversation entry a stable ID**

Write failing tests asserting `BugfixService.start` and `continue_task` persist dictionaries shaped as `{"id": stable_id, "role": "user", "content": text}` and pass a `HumanMessage(id=stable_id, content=text)` into the Graph. Add backward migration for old entries without IDs using task ID, original conversation ordinal, role, and normalized content hash.

- [ ] **Step 2: Write failing task-isolation and complete-field tests**

```python
def test_protected_context_is_task_local(builder, task_a, task_b):
    rendered = render_protected_context(builder.build(task_a.task_id, [], None))
    assert task_a.user_problem in rendered
    assert task_b.user_problem not in rendered

def test_working_memory_projection_contains_every_field(builder, complete_memory):
    rendered = render_protected_context(builder.build("task-a", [], None))
    for tag in ("rejected_hypotheses", "checked_files", "experiments", "unresolved_questions"):
        assert f"<{tag}>" in rendered
```

- [ ] **Step 3: Write failing authority-read and dedup tests**

Make each authoritative dependency throw in turn. Task, Working Memory, deterministic evidence, and Research Store read errors must become `ProtectedContextLoadError`; missing Working Memory is legal and renders `version="none"`. Assert the same `constraint_id`, `evidence_id`, `hypothesis_id`, and `claim_id` appears once across the complete projected request, with Snapshot contributing only additional provenance/artifact references.

- [ ] **Step 4: Run focused tests and confirm import failure**

Run: `pytest tests/test_protected_context.py tests/test_context.py tests/test_models.py -q`

Expected: FAIL because `ProtectedContextBuilder` and conversation IDs do not exist.

- [ ] **Step 5: Implement typed-source reads and projection**

Build Task Anchor from original problem, user conversation IDs, active constraints, project root, project Python, and TaskStatus. Build Working Memory from its latest complete structured version. Build deterministic evidence from the collector and Research Store. Deduplicate with fixed ownership: Anchor → deterministic evidence → Working Memory → Snapshot; merge later provenance into the first entity but never repeat current state/text.

- [ ] **Step 6: Inject exactly three request-local System blocks**

Render escaped, bounded `<deepfix_task_anchor>`, `<deepfix_working_memory>`, and `<deepfix_deterministic_evidence>` blocks. `ProtectedContextMiddleware.wrap_model_call` overrides only `system_message`; tests must assert neither `request.messages` nor the checkpoint state's `messages` gains a protected block.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
pytest tests/test_protected_context.py tests/test_context.py tests/test_models.py tests/test_service.py -q
ruff check src/deepfix/protected_context.py src/deepfix/context.py src/deepfix/models.py src/deepfix/service.py tests/test_protected_context.py tests/test_context.py tests/test_models.py tests/test_service.py
git add src/deepfix/protected_context.py src/deepfix/context.py src/deepfix/models.py src/deepfix/service.py tests/test_protected_context.py tests/test_context.py tests/test_models.py tests/test_service.py
git commit -m "feat: inject authoritative protected context"
```

### Task 11: Implement the transactional compaction coordinator

**Files:**

- Create: `src/deepfix/compaction/coordinator.py`
- Create: `tests/compaction/test_coordinator.py`

**Interfaces:**

- Produces: `CompactionCoordinator.prepare(CompactionRequest) -> PreparedCompaction`.
- Produces: `CompactionCoordinator.invoke_automatic(request, handler, protected) -> ModelResponse | ExtendedModelResponse`.
- Consumes: identity normalizer, partitioner, budget monitor, adapter, Delta generator, Snapshot builder/store.

Use these exact coordinator carriers:

```python
@dataclass(frozen=True)
class CompactionRequest:
    task_id: str
    entrypoint: Literal["automatic", "manual_tool", "overflow_recovery"]
    messages: tuple[AnyMessage, ...]
    active_event: DeepFixCompactionEvent | None
    protected_context: ProtectedContext
    budget: ContextBudgetReport
    model: BaseChatModel
    tool_call_id: str | None = None

@dataclass(frozen=True)
class PreparedCompaction:
    attempt_id: str
    input_hash: str
    artifact_reference: ArtifactReference
    snapshot: CompactionSnapshot
    snapshot_message: SystemMessage
    retention: RetentionPlan
    event: DeepFixCompactionEvent
```

- [ ] **Step 1: Write failing ordered-transaction test**

```python
def test_prepare_orders_artifact_before_snapshot_and_never_mutates_messages(coordinator, request, recorder):
    original = list(request.messages)
    prepared = coordinator.prepare(request)
    assert recorder.calls == [
        "partition", "artifact_write", "artifact_verify", "delta_generate",
        "snapshot_validate", "snapshot_write", "snapshot_verify",
    ]
    assert request.messages == original
    assert prepared.snapshot.lifecycle == "prepared"
```

- [ ] **Step 2: Write failing event-activation tests**

Assert automatic handler success returns `ExtendedModelResponse` with `_deepfix_compaction_event`, stable Snapshot Message ID, active version, retained message IDs, artifact reference, and input hash. Assert handler failure returns no event and leaves the Snapshot prepared. A later identical input reuses the same artifact/Snapshot/version.

- [ ] **Step 3: Write normal-zone preparation failure tests**

At each artifact/Delta/Snapshot failure stage with usage `0.85`, assert: original messages are passed to handler exactly once, no event is returned, one idempotent failure record is written, the same request does not attempt preparation again, and a prepared row is abandoned after the successful passthrough creates new conversation state.

- [ ] **Step 4: Write emergency failure tests**

At usage `0.91`, assert each preparation failure is wrapped as `ContextRecoveryRequired` with `original_messages_preserved=True`, stage/cause, active/prepared versions and artifact reference. Handler must not receive an unsafe uncompressed emergency request.

- [ ] **Step 5: Run coordinator tests and confirm import failure**

Run: `pytest tests/compaction/test_coordinator.py -q`

Expected: FAIL because `CompactionCoordinator` does not exist.

- [ ] **Step 6: Implement prepare and automatic entrypoint**

Generate `attempt_id` and `input_hash` from task ID, active event version, ordered stable message IDs, Working Memory version, and evidence fingerprint. Follow the exact transaction order and create the effective view as one bounded Snapshot Message plus whole retained units. For normal-zone preparation errors, catch `CompactionPreparationError`, record it, and call the original handler once; for emergency errors, wrap and raise recovery.

Record budget/failure/passthrough metrics through `WorkingMemoryStore`. Do not increment successful compaction metrics until event reconciliation proves Graph state contains the returned event.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_coordinator.py -q
ruff check src/deepfix/compaction/coordinator.py tests/compaction/test_coordinator.py
git add src/deepfix/compaction/coordinator.py tests/compaction/test_coordinator.py
git commit -m "feat: coordinate transactional context compaction"
```

### Task 12: Replace the manual compaction Tool without private Deep Agents summary logic

**Files:**

- Modify: `src/deepfix/compaction/coordinator.py`
- Create: `src/deepfix/compaction/tools.py`
- Create: `tests/compaction/test_tools.py`
- Modify: `tests/test_agent.py`

**Interfaces:**

- Produces: `build_compact_conversation_tool(coordinator) -> BaseTool`.
- Produces: `CompactionCoordinator.compact_manually(runtime) -> Command`.

- [ ] **Step 1: Write failing public Tool schema and no-op tests**

Assert the Tool exposes no model-supplied task ID, event, path, cutoff, or token fields. Below manual eligibility it returns a stable-ID success/no-op ToolMessage and no `_deepfix_compaction_event` update.

- [ ] **Step 2: Write failing success Command test**

```python
def test_manual_success_returns_event_and_stable_tool_message(tool, runtime):
    command = tool.invoke({}, config=runtime_config(runtime))
    assert command.update["_deepfix_compaction_event"]["active_snapshot_version"] == 1
    message = command.update["messages"][0]
    assert message.status == "success"
    assert message.id == stable_generated_message_id("task-a", "attempt-1", "manual_success")
```

- [ ] **Step 3: Write tiered manual failure tests**

At usage `<=0.90`, artifact/Delta/Snapshot failures return `status="error"` ToolMessage, preserve all messages, abandon unusable prepared rows, record the failure, and do not raise. At `>0.90`, the same failures raise `ContextRecoveryRequired` so Service can pause.

- [ ] **Step 4: Run focused Tool tests and confirm failure**

Run: `pytest tests/compaction/test_tools.py tests/test_agent.py -q`

Expected: FAIL because the DeepFix compact Tool is not registered.

- [ ] **Step 5: Implement ToolRuntime-derived identity and Command updates**

Read task ID, stable messages, and Tool Call ID from `ToolRuntime`; call the same coordinator preparation path as automatic compaction. Return only bounded error codes, never raw exception content. A success Command contains `_deepfix_compaction_event`, `_deepfix_compaction_session_id`, and exactly one paired ToolMessage. Register both sync and coroutine functions on the StructuredTool.

- [ ] **Step 6: Prove private summary APIs are unused**

Patch the four forbidden Deep Agents methods to raise, invoke manual compaction, and assert success. Keep the Tool name `compact_conversation` so the prompt and Agent surface remain stable.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_tools.py tests/test_agent.py -q
ruff check src/deepfix/compaction/tools.py src/deepfix/compaction/coordinator.py tests/compaction/test_tools.py tests/test_agent.py
git add src/deepfix/compaction/tools.py src/deepfix/compaction/coordinator.py tests/compaction/test_tools.py tests/test_agent.py
git commit -m "feat: add evidence-safe compact tool"
```

### Task 13: Add automatic middleware, event state, and single-retry Overflow recovery

**Files:**

- Create: `src/deepfix/compaction/middleware.py`
- Create: `tests/compaction/test_middleware.py`
- Create: `tests/compaction/test_overflow.py`
- Modify: `src/deepfix/context.py`

**Interfaces:**

- Produces: `MessageIdentityMiddleware` and `DeepFixCompactionMiddleware` with `_deepfix_compaction_event` private Graph state.
- Produces: effective-message reconstruction from the event and one minimal-safe-context retry.

Declare the state extension in `middleware.py` and assign it as both middleware classes' `state_schema`:

```python
class DeepFixCompactionState(AgentState):
    _deepfix_compaction_event: Annotated[
        NotRequired[dict[str, object] | None], PrivateStateAttr
    ]
    _deepfix_compaction_session_id: Annotated[
        NotRequired[str | None], PrivateStateAttr
    ]
```

- [ ] **Step 1: Write failing Graph state identity/event tests**

Assert `before_agent`/`before_model` fills missing IDs once and persists them in state. The one-time update replaces the list with `[RemoveMessage(id=REMOVE_ALL_MESSAGES), *normalized_messages]`; this is identity-equivalent replacement, not compaction deletion, and tests assert content/order are unchanged. Assert a prior event causes the next request to use event Snapshot + retained whole units while the checkpoint still retains recoverable original messages. An abandoned or merely latest prepared Snapshot must not affect the request.

- [ ] **Step 2: Write failing threshold behavior tests**

Assert `<=0.75` calls normally; observe zone only adds one stale-memory hint; normal and emergency zones call the coordinator with their distinct targets. Verify middleware wrapper order through a real nested handler trace rather than class-name list order.

- [ ] **Step 3: Write failing Overflow retry tests**

```python
def test_overflow_retries_handler_exactly_once_with_minimal_safe_context(middleware, request):
    calls = []
    def handler(received):
        calls.append(received)
        if len(calls) == 1:
            raise ContextOverflowError("too large")
        return ModelResponse(result=[AIMessage(content="ok")])
    middleware.wrap_model_call(request, handler)
    assert len(calls) == 2
    assert measured_ratio(calls[1]) <= 0.50
```

Also assert the retry retains all three protected blocks, latest user input, incomplete/ambiguous units, Snapshot and artifact reference. Preparation failure during recovery and a second Overflow both raise `ContextRecoveryRequired`; there is no third handler call.

- [ ] **Step 4: Run focused middleware tests and confirm import failure**

Run: `pytest tests/compaction/test_middleware.py tests/compaction/test_overflow.py -q`

Expected: FAIL because compaction middleware does not exist.

- [ ] **Step 5: Implement sync and async wrapper paths**

Compose protected context before measuring tokens. Track `overflow_retry_attempted` in invocation-local coordinator state, not persistent model memory. Async methods call async Delta generation and handler but reuse the same pure identity, partition, selection, validation, and merge functions.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_middleware.py tests/compaction/test_overflow.py -q
ruff check src/deepfix/compaction/middleware.py src/deepfix/context.py tests/compaction/test_middleware.py tests/compaction/test_overflow.py
git add src/deepfix/compaction/middleware.py src/deepfix/context.py tests/compaction/test_middleware.py tests/compaction/test_overflow.py
git commit -m "feat: compact and recover model context"
```

### Task 14: Wire the coordinator into Agent, CLI, and Service recovery boundaries

**Files:**

- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/context.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/service.py`
- Modify: `src/deepfix/models.py`
- Modify: `src/deepfix/extensions.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_extensions.py`

**Interfaces:**

- Changes: `build_agent(config, checkpointer, working_memory_store, task_repository, compaction_store, extensions=None, research_evidence_store=None, *, allowed_skill_roots=())`.
- Changes: `BugfixService(agent, repository, policy, config, working_memory_store, research_evidence_store, compaction_store)` and `_invoke` catches only Service-facing `ContextCoordinationError` for context recovery pause.

- [ ] **Step 1: Write failing Agent assembly tests**

Assert middleware contains `MessageIdentityMiddleware`, `PromptPolicyMiddleware`, `ProtectedContextMiddleware`, then `DeepFixCompactionMiddleware` in effective wrapper order, and contains no `SummarizationMiddleware`, `SummarizationToolMiddleware`, `ContextMemoryMiddleware`, or separately injecting `ResearchEvidenceMiddleware`. Assert `compact_conversation` still exists once, uses the same model instance as `CompactionDeltaGenerator`, and Deep Agents Filesystem/Backend tools remain available. Keep research tools; their current state is rendered through Protected Context.

- [ ] **Step 2: Write failing Service boundary tests**

```python
@pytest.mark.parametrize("error_type", [ProtectedContextLoadError, ContextRecoveryRequired])
def test_recovery_error_pauses_and_persists_metadata(error_type, service, recovery):
    service.agent = FakeAgent(error_type(recovery))
    task = service.start("long bug")
    assert task.status is TaskStatus.PAUSED
    assert task.context_recovery == recovery

def test_preparation_error_does_not_get_misclassified_by_service(service):
    service.agent = FakeAgent(ArtifactPersistenceError(failure_record()))
    assert service.start("bug").status is TaskStatus.FAILED
```

The second test proves preparation errors must be consumed by Coordinator in normal paths; an unexpected leak remains an ordinary Agent failure rather than silently invoking recovery semantics.

Add a resume test: recovery metadata remains persisted while PAUSED, and is cleared only after the resumed Agent invocation successfully returns or commits a replacement recovery event.

- [ ] **Step 3: Remove English ToolMessage success parsing**

Delete `message.text.startswith("Conversation compacted.")` logic. Coordinator writes attempt metrics and event reconciliation writes successful activation/version into `WorkingMemoryStore`; Service only synchronizes that typed metrics row. Add regression tests showing translated Tool text cannot change compaction counts.

- [ ] **Step 4: Wire stores through CLI without duplicate instances**

Construct one `CompactionStore(config.database_path)` and one `TaskRepository`; pass the same objects to Agent and Service. Update CLI fakes to capture identity and assert the shared instances. Preserve research client/store and backend sharing.

Add the three DeepFix middleware class names to `_PROTECTED_MIDDLEWARE_NAMES` so extensions cannot replace identity, protected-context, or compaction coordination behavior.

- [ ] **Step 5: Run focused integration tests and fix signatures**

Run: `pytest tests/test_agent.py tests/test_cli.py tests/test_service.py tests/test_context.py tests/test_extensions.py -q`

Expected before implementation: failures for old middleware types, missing constructor arguments, and missing recovery pause behavior. Implement only the necessary wiring and exception catch ordering (`ContextCoordinationError` before general `Exception`).

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/test_agent.py tests/test_cli.py tests/test_service.py tests/test_context.py tests/test_extensions.py -q
ruff check src/deepfix/agent.py src/deepfix/context.py src/deepfix/cli.py src/deepfix/service.py src/deepfix/models.py src/deepfix/extensions.py tests/test_agent.py tests/test_cli.py tests/test_service.py tests/test_context.py tests/test_extensions.py
git add src/deepfix/agent.py src/deepfix/context.py src/deepfix/cli.py src/deepfix/service.py src/deepfix/models.py src/deepfix/extensions.py tests/test_agent.py tests/test_cli.py tests/test_service.py tests/test_context.py tests/test_extensions.py
git commit -m "feat: activate DeepFix compaction coordination"
```

### Task 15: Migrate legacy state and expose auditable metrics/reporting

**Files:**

- Create: `src/deepfix/compaction/migration.py`
- Create: `tests/compaction/test_migration.py`
- Modify: `src/deepfix/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `src/deepfix/memory.py`
- Modify: `tests/test_memory.py`

**Interfaces:**

- Produces: `migrate_legacy_context_state(task_id, graph_state, stores, adapter) -> dict[str, object]`.
- Produces reports from structured event/Store metrics, not model prose.

- [ ] **Step 1: Write failing lazy migration tests**

Create old payloads with missing message IDs, string hypotheses/facts, old ContextMetrics, TestResult without Tool provenance, and serialized `_summarization_event`. Assert migration:

- preserves existing IDs and deterministically fills every missing Graph Message ID;
- generates legacy hypothesis/claim/evidence identities once;
- writes the old summary/history to a verified artifact reference;
- emits `_deepfix_compaction_event` and removes `_summarization_event` from migrated state;
- never lets legacy summary overwrite user constraints or deterministic evidence;
- is idempotent on a second run.

- [ ] **Step 2: Run migration tests and confirm import failure**

Run: `pytest tests/compaction/test_migration.py -q`

Expected: FAIL because the migration module does not exist.

- [ ] **Step 3: Implement task-lazy migration**

Invoke migration only when an old task first needs compaction or is resumed with legacy event state. Record a migration version in the technical Store. Use the narrow legacy adapter only to read already serialized `_summarization_event`; do not invoke any Deep Agents summary/cutoff helper.

- [ ] **Step 4: Update metrics and report tests**

Assert reports display latest budget zone, normal/emergency compaction counts, failure/passthrough/manual-error/Overflow retry counts, active Snapshot version, last artifact and bounded last error. Assert deterministic pytest exit code and research verification remain the evidence source and metrics never mark a repair complete.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/compaction/test_migration.py tests/test_memory.py tests/test_reporting.py -q
ruff check src/deepfix/compaction/migration.py src/deepfix/memory.py src/deepfix/reporting.py tests/compaction/test_migration.py tests/test_memory.py tests/test_reporting.py
git add src/deepfix/compaction/migration.py src/deepfix/memory.py src/deepfix/reporting.py tests/compaction/test_migration.py tests/test_memory.py tests/test_reporting.py
git commit -m "feat: migrate and report compaction state"
```

### Task 16: Prove the long-context one-shot Bug project end to end

**Files:**

- Create: `tests/compaction/test_long_context_workflow.py`
- Create: `tests/compaction/fixtures/bug_project/src/calculator.py`
- Create: `tests/compaction/fixtures/bug_project/tests/test_calculator.py`
- Modify: `README.md`

**Interfaces:**

- Verifies the complete single-Agent workflow with fake model outputs, real SQLite, controlled Backend, real LangChain messages, and a disposable copied Python project.

- [ ] **Step 1: Create the deterministic Bug fixture and failing baseline test**

Use a sign-normalization defect whose test initially fails with exit code 1. The workflow fixture must include parallel read-only investigation, a rejected hypothesis with reason, a failed pytest ToolMessage, approved edit, successful file ToolMessage, passing pytest ToolMessage, research evidence status, and at least three compactions.

- [ ] **Step 2: Write the long-context assertions**

```python
def test_long_bugfix_preserves_evidence_across_three_compactions(workflow):
    result = workflow.run()
    assert result.task.test_results[-1].exit_code == 0
    assert result.original_constraint == result.final_snapshot.user_constraints[0].text
    assert result.rejected_reason in result.final_protected_context
    assert result.history_contains_all_compressed_message_ids
    assert result.no_work_unit_was_split
    assert result.target_project_has_no_deepfix_artifacts
    assert result.task.status is TaskStatus.COMPLETED
```

- [ ] **Step 3: Add failure-path end-to-end cases**

Inject artifact failure at 0.85 and assert one uncompressed continuation with no pause; inject the same at 0.91 and assert PAUSED with recovery metadata; inject two Overflows and assert exactly two handler calls and PAUSED. Verify failure never removes checkpoint messages.

- [ ] **Step 4: Run the new end-to-end file and fix only integration defects**

Run: `pytest tests/compaction/test_long_context_workflow.py -q`

Expected before final integration fixes: FAIL at the first mismatched wiring or invariant; after fixes, all cases PASS without network or DeepSeek.

- [ ] **Step 5: Document the architecture and recovery behavior**

Update README's context section with the four budget zones, protected blocks, stable identity rule, Snapshot lifecycle, normal-zone passthrough, emergency pause, and `context_recovery` resume information. State that large result/media files remain under `DEEPFIX_HOME/artifacts`.

- [ ] **Step 6: Run the full verification gate**

Run:

```powershell
pytest -q
ruff check src tests
```

Expected: all offline tests PASS; online-marked tests remain deselected by the existing pytest configuration; Ruff exits 0.

- [ ] **Step 7: Inspect dependency-boundary and workspace diffs**

Run:

```powershell
rg -n "_create_summary|_acreate_summary|_determine_cutoff_index|_lc_helper|_summarization_event" src/deepfix
git status --short
git diff --check
```

Expected: the first command finds only the explicitly named legacy migration read/test guard, no normal runtime call; status contains only intended implementation/test/doc changes; `git diff --check` exits 0.

- [ ] **Step 8: Commit the end-to-end proof**

```powershell
git add tests/compaction/test_long_context_workflow.py tests/compaction/fixtures README.md
git commit -m "test: prove evidence-safe long context repair"
```

---

## Final Review Checklist

- Every ModelRequest receives task-isolated Task Anchor, full-field Working Memory, and deterministic evidence blocks.
- Stable IDs cover every Graph Message and drive WorkUnit, provenance, coverage, history, and event idempotency.
- Same constraint/evidence/hypothesis/claim identities render once; Snapshot only supplies compressed history and additional provenance.
- Tool calls, all parallel results, and Assistant explanation are retained or compacted as one unit.
- Delta output cannot supply canonical IDs or deterministic fields; Builder performs all canonicalization and authority checks.
- Snapshot lifecycle and `active_snapshot_version` follow the event-directed rules.
- Artifact and Snapshot verification precede event submission; failed preparation never removes messages.
- Normal-zone automatic failure continues once; non-emergency manual failure returns error ToolMessage; emergency/Overflow recovery failure pauses through Service.
- First Overflow creates one `<=0.50` minimal safe retry; a second Overflow never retries again.
- Deep Agents private summary/cutoff/event methods are absent from the normal path.
- Existing HITL, single-Agent, research isolation, real pytest completion gate, CLI resume, and report behavior remain green.
