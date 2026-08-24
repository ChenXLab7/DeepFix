# DeepFix Investigation Reliability Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic investigation coordination layer that observes Deep Agents' internal Tool loop, owns AgentPhase, gates editing behind supported hypotheses, detects stagnation, and pauses safely with recovery metadata.

**Architecture:** Add a focused `deepfix.investigation` package containing domain models, an append-only SQLite store, pure classifiers/evaluators, a coordinator, two structured tools, middleware, and legacy migration. `BugfixService` remains the only owner of `TaskStatus`; the middleware owns no business transitions and uses typed recovery exceptions. The existing compaction evidence ledger supplies authoritative test/file evidence, while Working Memory remains a non-authoritative semantic projection.

**Tech Stack:** Python 3.11+, Pydantic 2, SQLite/WAL, LangChain AgentMiddleware, LangGraph ToolNode/checkpoints, Deep Agents 0.7.x, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-24-deepfix-investigation-reliability-core-design.md`

## Global Constraints

- Implement only subproject 1, Investigation Reliability Core. Diagnostic Artifact Retrieval and Progress Events + CLI Renderer remain separate required follow-up specs and plans.
- Preserve the user's current uncommitted edits in `src/deepfix/agent.py`, `src/deepfix/prompts.py`, `src/deepfix/debug.py`, and `docs/debug/`; patch overlapping files surgically and never stage unrelated changes.
- `TaskStatus` is owned by `BugfixService`; `AgentPhase` is owned by `InvestigationStore`; Working Memory cannot override either.
- `phase_changed`, a newly read file/range, and a changed content fingerprint never create strong progress by themselves.
- Dependency scope requires a system-verifiable relation edge. Model-authored `reason` never changes scope classification.
- `EDITING → TESTING` requires observed execution of a deterministically classified pytest command after approval; approval, a proposed Tool Call, and arbitrary `execute` commands are insufficient.
- A legacy task with a successful edit but no paired post-edit pytest ToolMessage remains `EDITING`; old state has no trustworthy execution-start marker from which to infer `TESTING`.
- Structured validation errors return stable error ToolMessages. Store failures and level-2 stagnation raise typed recovery errors; middleware never changes TaskStatus.
- Every task follows red-green-refactor, runs its focused tests, and commits only its listed files.

## File Structure

### New production files

- `src/deepfix/investigation/__init__.py` — public investigation interfaces.
- `src/deepfix/investigation/models.py` — enums, Pydantic commands/events/state/recovery models.
- `src/deepfix/investigation/identity.py` — stable event, signature, relation, and permit IDs.
- `src/deepfix/investigation/errors.py` — typed recoverable investigation exceptions.
- `src/deepfix/investigation/store.py` — append-only event log and transactional materialized state.
- `src/deepfix/investigation/classification.py` — pytest command classification, result fingerprints, scope relations, bounded observations.
- `src/deepfix/investigation/progress.py` — strong-progress evaluation and phase-independent counter reset.
- `src/deepfix/investigation/phase.py` — deterministic AgentPhase transition rules.
- `src/deepfix/investigation/stagnation.py` — repeat/cycle/no-progress/exploration detection and permit gates.
- `src/deepfix/investigation/coordinator.py` — command validation, event derivation, and atomic state commits.
- `src/deepfix/investigation/tools.py` — `record_hypothesis` and `continue_investigation` Tool façades.
- `src/deepfix/investigation/middleware.py` — Deep Agents Model/Tool hook adapter and bounded prompt projection.
- `src/deepfix/investigation/receipts.py` — Backend-backed Tool execution receipts that prevent side-effect replay after event-commit failure.
- `src/deepfix/investigation/migration.py` — idempotent old-task reconstruction.

### Existing production files modified

- `src/deepfix/prompting.py` — read AgentPhase from InvestigationStore.
- `src/deepfix/prompts.py` — add diagnosing prompt and structured transition instructions while preserving user path rules.
- `src/deepfix/extensions.py` — require investigation capability metadata for extension tools.
- `src/deepfix/models.py` — persist investigation recovery metadata separately from context recovery.
- `src/deepfix/service.py` — catch investigation errors and emit lifecycle commands after business-state changes.
- `src/deepfix/agent.py` — construct tools/middleware and preserve optional LLM trace ordering.
- `src/deepfix/cli.py` — construct one shared Store/Coordinator for Agent and Service.
- `src/deepfix/compaction/evidence.py` — expose public paired-result collection so investigation and protected context share one deterministic evidence identity path.
- `src/deepfix/protected_context.py` — merge investigation hypothesis support into the existing unique protected projection.

### New tests

- `tests/investigation/test_models.py`
- `tests/investigation/test_identity.py`
- `tests/investigation/test_store.py`
- `tests/investigation/test_classification.py`
- `tests/investigation/test_progress.py`
- `tests/investigation/test_phase.py`
- `tests/investigation/test_stagnation.py`
- `tests/investigation/test_coordinator.py`
- `tests/investigation/test_tools.py`
- `tests/investigation/test_middleware.py`
- `tests/investigation/test_migration.py`
- `tests/investigation/test_workflow.py`
- `tests/investigation/helpers.py` — explicit offline factories shared by investigation tests; no model or network calls.

---

### Task 1: Domain Models, Stable Identities, and Recovery Errors

**Files:**
- Create: `src/deepfix/investigation/__init__.py`
- Create: `src/deepfix/investigation/models.py`
- Create: `src/deepfix/investigation/identity.py`
- Create: `src/deepfix/investigation/errors.py`
- Create: `tests/investigation/__init__.py`
- Create: `tests/investigation/test_models.py`
- Create: `tests/investigation/test_identity.py`

**Interfaces:**
- Consumes: `deepfix.compaction.models.StrictModel`, `ProvenanceRef` conventions, SHA-256 ID style from `deepfix.compaction.identity`.
- Produces: `AgentPhase`, `ProgressKind`, `ScopeKind`, `InvestigationCapability`, `NewInvestigationEvent`, `InvestigationEvent`, `InvestigationState`, `RecordHypothesisInput`, `ContinueInvestigationInput`, `InvestigationRecoveryMetadata`, `stable_investigation_id()`, and typed exceptions.

- [ ] **Step 1: Write failing model and identity tests**

```python
# tests/investigation/test_models.py
import pytest
from pydantic import ValidationError

from deepfix.investigation.models import (
    AgentPhase,
    ContinueInvestigationInput,
    InvestigationState,
    RecordHypothesisInput,
)


def test_new_state_has_no_false_progress_or_permit():
    state = InvestigationState.new("task-a")
    assert state.agent_phase is AgentPhase.INVESTIGATING
    assert state.progress_generation == 0
    assert state.stagnation_level == 0
    assert state.permit is None


def test_supported_hypothesis_requires_operational_fields():
    with pytest.raises(ValidationError):
        RecordHypothesisInput(
            statement="sign is inverted twice",
            evidence_ids=["evidence-1"],
            checked_locations=[],
            target_state="supported",
            reason="failure points here",
        )


def test_continue_intent_requires_a_tool_and_target():
    with pytest.raises(ValidationError):
        ContinueInvestigationInput(
            hypothesis_ids=["hyp-1"],
            unresolved_question="which branch flips sign?",
            expected_evidence="branch condition",
            tool_name="",
            target="src/sign.py",
            reason="inspect the branch",
        )
```

```python
# tests/investigation/test_identity.py
from deepfix.investigation.identity import stable_investigation_id


def test_investigation_ids_are_deterministic_and_scoped():
    first = stable_investigation_id("event", "task-a", "tool-1", "abc")
    assert first == stable_investigation_id("event", "task-a", "tool-1", "abc")
    assert first != stable_investigation_id("event", "task-b", "tool-1", "abc")
    assert first.startswith("event_")
```

- [ ] **Step 2: Run tests and verify missing-package failures**

Run: `pytest tests/investigation/test_models.py tests/investigation/test_identity.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'deepfix.investigation'`.

- [ ] **Step 3: Implement strict domain models, deterministic IDs, and errors**

```python
# src/deepfix/investigation/identity.py
import hashlib


def stable_investigation_id(prefix: str, task_id: str, *parts: str) -> str:
    values = [prefix.strip(), task_id.strip(), *(part.strip() for part in parts)]
    if any(not value for value in values):
        raise ValueError("investigation ID 输入不能为空")
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"
```

```python
# src/deepfix/investigation/models.py
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from deepfix.compaction.models import StrictModel


class AgentPhase(StrEnum):
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    EDITING = "editing"
    TESTING = "testing"
    REVIEWING = "reviewing"


class ProgressKind(StrEnum):
    TEST_EVIDENCE = "test_evidence"
    DECISION_EVIDENCE = "decision_evidence"
    HYPOTHESIS_TRANSITION = "hypothesis_transition"
    FILE_CHANGE = "file_change"
    POST_EDIT_TEST = "post_edit_test"
    USER_INFORMATION = "user_information"


class ScopeKind(StrEnum):
    DIRECT = "direct"
    DEPENDENCY = "dependency"
    EXPLORATORY = "exploratory"


class InvestigationCapability(StrEnum):
    READ = "read"
    SEARCH = "search"
    EXECUTE = "execute"
    MODIFY = "modify"
    RESEARCH = "research"
    MEMORY = "memory"
    COMPACTION = "compaction"
    META = "meta"


class InvestigationEventType(StrEnum):
    TASK_STARTED = "task_started"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    USER_INFORMATION_RECEIVED = "user_information_received"
    NEEDS_INPUT = "needs_input"
    TOOL_COMPLETED = "tool_completed"
    TEST_OBSERVED = "test_observed"
    FILE_CHECKED = "file_checked"
    FILE_CHANGED = "file_changed"
    FILE_CHANGE_FAILED = "file_change_failed"
    DECISION_EVIDENCE_OBSERVED = "decision_evidence_observed"
    HYPOTHESIS_RECORDED = "hypothesis_recorded"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    HYPOTHESIS_SUPPORTED = "hypothesis_supported"
    VERIFICATION_EXECUTION_OBSERVED = "verification_execution_observed"
    POST_EDIT_TEST_OBSERVED = "post_edit_test_observed"
    INVESTIGATION_INTENT_RECORDED = "investigation_intent_recorded"
    PHASE_CHANGED = "phase_changed"
    REEVALUATION_REQUIRED = "reevaluation_required"
    INVESTIGATION_PERMIT_GRANTED = "investigation_permit_granted"
    INVESTIGATION_STAGNATED = "investigation_stagnated"


class CheckedLocation(StrictModel):
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_lines(self):
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")
        return self


class ProposedChange(StrictModel):
    path: str = Field(min_length=1)
    description: str = Field(min_length=1)


class CheckedFile(StrictModel):
    path: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=1)
    ranges: list[CheckedLocation] = Field(default_factory=list)
    scope: ScopeKind
    first_event_id: str = Field(min_length=1)
    latest_event_id: str = Field(min_length=1)


class RelationEdge(StrictModel):
    relation_id: str = Field(min_length=1)
    relation: Literal["import", "call", "traceback", "grep_reference", "symbol_reference", "test_collection"]
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    source_message_id: str = Field(min_length=1)
    scope: Literal[ScopeKind.DEPENDENCY] = ScopeKind.DEPENDENCY


class InvestigationHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    state: Literal["candidate", "rejected", "supported"]
    evidence_ids: list[str]
    checked_locations: list[CheckedLocation]
    proposed_change: ProposedChange | None = None
    expected_effect: str | None = None
    reason: str = Field(min_length=1)


class InvestigationPermit(StrictModel):
    permit_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    target_hash: str = Field(min_length=1)
    granted_in_generation: int = Field(ge=0)
    consumed: bool = False


class RecordHypothesisInput(StrictModel):
    hypothesis_id: str | None = None
    statement: str = Field(min_length=1)
    evidence_ids: list[str]
    checked_locations: list[CheckedLocation]
    proposed_change: ProposedChange | None = None
    expected_effect: str | None = None
    target_state: Literal["candidate", "rejected", "supported"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_transition(self):
        if self.target_state == "supported" and (
            not self.evidence_ids
            or not self.checked_locations
            or self.proposed_change is None
            or not (self.expected_effect or "").strip()
        ):
            raise ValueError("supported 假设缺少证据、位置、修改目标或预期效果")
        if self.target_state == "rejected" and self.hypothesis_id is None:
            raise ValueError("rejected 迁移必须提供 hypothesis_id")
        return self


class ContinueInvestigationInput(StrictModel):
    hypothesis_ids: list[str]
    unresolved_question: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ToolObservation(StrictModel):
    event_type: InvestigationEventType
    tool_call_id: str | None = None
    source_message_id: str | None = None
    signature: str = ""
    result_fingerprint: str = ""
    scope: ScopeKind = ScopeKind.DIRECT
    progress_kind: ProgressKind | None = None
    evidence_id: str | None = None
    hypothesis_id: str | None = None
    path: str | None = None
    exit_code: int | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class NewInvestigationEvent(StrictModel):
    event_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    event_type: InvestigationEventType
    source_message_id: str | None = None
    tool_call_id: str | None = None
    phase_before: AgentPhase
    phase_after: AgentPhase
    progress_kind: ProgressKind | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def task_started(cls, task_id: str):
        from deepfix.investigation.identity import stable_investigation_id
        return cls(
            event_id=stable_investigation_id("event", task_id, "task_started"),
            task_id=task_id,
            event_type=InvestigationEventType.TASK_STARTED,
            phase_before=AgentPhase.INVESTIGATING,
            phase_after=AgentPhase.INVESTIGATING,
        )


class InvestigationEvent(NewInvestigationEvent):
    sequence: int = Field(ge=1)
    created_at: datetime


class InvestigationState(StrictModel):
    task_id: str
    version: int = 0
    migration_version: int = 0
    agent_phase: AgentPhase = AgentPhase.INVESTIGATING
    paused_agent_phase: AgentPhase | None = None
    checked_files: list[CheckedFile] = Field(default_factory=list, max_length=64)
    relation_edges: list[RelationEdge] = Field(default_factory=list, max_length=128)
    recent_tool_signatures: list[str] = Field(default_factory=list, max_length=32)
    test_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list, max_length=64)
    supported_hypothesis_ids: list[str] = Field(default_factory=list, max_length=64)
    seen_progress_fingerprints: list[str] = Field(default_factory=list, max_length=64)
    progress_generation: int = 0
    last_progress_event_id: str | None = None
    last_progress_at: datetime | None = None
    no_progress_count: int = 0
    exploratory_without_progress: int = 0
    reevaluation_required: bool = False
    stagnation_level: Literal[0, 1, 2] = 0
    permit: InvestigationPermit | None = None
    post_permit_review_pending: bool = False

    @classmethod
    def new(cls, task_id: str):
        return cls(task_id=task_id.strip())


class InvestigationRecoveryMetadata(StrictModel):
    task_id: str
    error_code: str
    agent_phase: AgentPhase
    state_version: int
    last_event_sequence: int
    tool_call_id: str | None = None
    permit_id: str | None = None
    checkpoint_available: bool
    recovery_action: str
```

```python
# src/deepfix/investigation/errors.py
from deepfix.investigation.models import InvestigationRecoveryMetadata


class InvestigationCoordinationError(RuntimeError):
    def __init__(self, recovery: InvestigationRecoveryMetadata) -> None:
        self.recovery = recovery
        super().__init__(recovery.error_code)


class InvestigationStateError(InvestigationCoordinationError):
    pass


class InvestigationStagnationError(InvestigationCoordinationError):
    pass
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/investigation/test_models.py tests/investigation/test_identity.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the domain foundation**

```bash
git add src/deepfix/investigation tests/investigation
git commit -m "feat: define investigation domain models"
```

### Task 2: Transactional Investigation Store

**Files:**
- Create: `src/deepfix/investigation/store.py`
- Create: `tests/investigation/test_store.py`
- Modify: `src/deepfix/investigation/models.py`

**Interfaces:**
- Consumes: `InvestigationState`, `NewInvestigationEvent`, `InvestigationEvent`.
- Produces: `InvestigationStore.load(task_id)`, `ensure_started(task_id)`, `commit(expected_version, events, next_state)`, `list_events(task_id)`, and `last_sequence(task_id)`.

- [ ] **Step 1: Write failing persistence, isolation, and idempotency tests**

```python
def new_event(task_id: str, event_id: str, event_type: str, phase: AgentPhase) -> NewInvestigationEvent:
    return NewInvestigationEvent(
        event_id=event_id,
        task_id=task_id,
        event_type=event_type,
        phase_before=phase,
        phase_after=phase,
    )


def test_store_commits_event_and_state_atomically(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    event = new_event("task-a", "event-1", "tool_completed", state.agent_phase)
    updated = state.model_copy(update={"no_progress_count": 1})
    committed = store.commit(state.version, [event], updated)
    assert committed.version == 2
    assert store.load("task-a").no_progress_count == 1
    assert [item.sequence for item in store.list_events("task-a")] == [1, 2]


def test_store_replay_is_idempotent_and_task_isolated(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    a = store.ensure_started("task-a")
    b = store.ensure_started("task-b")
    event = new_event("task-a", "event-a", "file_checked", a.agent_phase)
    committed = store.commit(a.version, [event], a)
    replayed = store.commit(a.version, [event], a)
    assert replayed == committed
    assert len(store.list_events("task-a")) == 2
    assert len(store.list_events("task-b")) == 1
    assert b.task_id == "task-b"


def test_store_rejects_stale_non_replay_commit(tmp_path):
    store = InvestigationStore(tmp_path / "deepfix.sqlite3")
    state = store.ensure_started("task-a")
    store.commit(state.version, [new_event("task-a", "e1", "file_checked", state.agent_phase)], state)
    with pytest.raises(InvestigationStateConflict):
        store.commit(state.version, [new_event("task-a", "e2", "file_checked", state.agent_phase)], state)
```

- [ ] **Step 2: Run store tests and verify import failure**

Run: `pytest tests/investigation/test_store.py -v`

Expected: fails because `InvestigationStore` is undefined.

- [ ] **Step 3: Implement WAL tables and compare-and-swap commit**

```python
class InvestigationStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self._initialize()

    def ensure_started(self, task_id: str) -> InvestigationState:
        existing = self.load(task_id)
        if existing is not None:
            return existing
        state = InvestigationState.new(task_id)
        event = NewInvestigationEvent.task_started(task_id)
        try:
            return self.commit(0, [event], state)
        except InvestigationStateConflict:
            loaded = self.load(task_id)
            if loaded is None:
                raise
            return loaded

    def commit(
        self,
        expected_version: int,
        events: list[NewInvestigationEvent],
        next_state: InvestigationState,
    ) -> InvestigationState:
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._load(connection, next_state.task_id)
            current_version = 0 if current is None else current.version
            existing = self._existing_event_ids(connection, next_state.task_id, events)
            if len(existing) == len(events):
                connection.rollback()
                if current is None:
                    raise InvestigationStateConflict("重放事件缺少物化状态")
                return current
            if existing or current_version != expected_version:
                connection.rollback()
                raise InvestigationStateConflict("investigation state 版本冲突")
            committed = next_state.model_copy(update={"version": current_version + 1})
            self._insert_events(connection, events, committed)
            self._upsert_state(connection, committed)
            connection.commit()
            return committed
```

Create tables with primary keys `(task_id, event_id)` and `(task_id)`, assign event `sequence` from `MAX(sequence)+1` inside the same transaction, compare an existing event's canonical payload before accepting replay, and retain only bounded state JSON while keeping the full bounded event log.

- [ ] **Step 4: Run focused store tests**

Run: `pytest tests/investigation/test_store.py -v`

Expected: all tests pass, including replay and cross-task isolation.

- [ ] **Step 5: Commit the store**

```bash
git add src/deepfix/investigation/models.py src/deepfix/investigation/store.py tests/investigation/test_store.py
git commit -m "feat: persist investigation events and state"
```

### Task 3: Deterministic Classification and Strong-Progress Evaluation

**Files:**
- Create: `src/deepfix/investigation/classification.py`
- Create: `src/deepfix/investigation/progress.py`
- Create: `tests/investigation/test_classification.py`
- Create: `tests/investigation/test_progress.py`

**Interfaces:**
- Consumes: Tool name/args, ToolMessage, current InvestigationState, current project Python.
- Produces: `is_pytest_verification()`, `tool_signature()`, `result_fingerprint()`, `RelationEdge`, `ToolObservation`, and `ProgressEvaluator.evaluate()`.

- [ ] **Step 1: Write failing command, scope, redaction, and progress tests**

```python
def observation(event_type: str, **updates) -> ToolObservation:
    base = ToolObservation(event_type=event_type)
    return base.model_copy(update=updates)


@pytest.mark.parametrize("command", ["pytest -q", "python -m pytest tests/test_sign.py", "python.exe -m pytest"])
def test_pytest_classifier_accepts_real_pytest_commands(command):
    assert is_pytest_verification(command, "C:/Python/python.exe")


@pytest.mark.parametrize("command", ["echo pytest", "python --version", "pip install pytest", "ruff check ."])
def test_pytest_classifier_rejects_non_verification_execute(command):
    assert not is_pytest_verification(command, "C:/Python/python.exe")


def test_model_reason_cannot_create_dependency_relation():
    assert relation_from_model_reason("src/random.py", "because it may matter") is None


def test_relation_requires_verifiable_source_message():
    edge = relation_from_tool_result(
        relation="import",
        source="src/api.py",
        target="src/sign.py",
        source_message_id="msg-1",
    )
    assert edge.scope is ScopeKind.DEPENDENCY


def test_new_file_and_phase_change_are_not_strong_progress():
    evaluator = ProgressEvaluator()
    assert evaluator.evaluate(observation("file_checked", path="src/new.py")) is None
    assert evaluator.evaluate(observation("phase_changed")) is None


def test_new_test_evidence_and_supported_hypothesis_are_strong_progress():
    evaluator = ProgressEvaluator()
    assert evaluator.evaluate(observation("test_observed", evidence_id="e1")) is ProgressKind.TEST_EVIDENCE
    assert evaluator.evaluate(observation("hypothesis_supported", hypothesis_id="h1")) is ProgressKind.HYPOTHESIS_TRANSITION


def test_same_test_command_and_result_in_same_generation_is_not_new_progress():
    evaluator = ProgressEvaluator()
    state = InvestigationState.new("task-a")
    first = observation("test_observed", signature="pytest-q", result_fingerprint="same-result", evidence_id="e1")
    second = observation("test_observed", signature="pytest-q", result_fingerprint="same-result", evidence_id="e2")
    state, progress = evaluator.apply(state, first)
    assert progress is ProgressKind.TEST_EVIDENCE
    state, progress = evaluator.apply(state, second)
    assert progress is None
```

- [ ] **Step 2: Run tests and verify classifier modules are missing**

Run: `pytest tests/investigation/test_classification.py tests/investigation/test_progress.py -v`

Expected: fails on missing imports.

- [ ] **Step 3: Implement canonical signatures, bounded observations, verified edges, and progress whitelist**

```python
_PYTEST_ENTRYPOINTS = {"pytest", "pytest.exe"}
_PYTHON_ENTRYPOINTS = {"python", "python.exe", "py", "py.exe"}


def is_pytest_verification(command: str, project_python: str) -> bool:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0].strip('"')).name.lower()
    if executable in _PYTEST_ENTRYPOINTS:
        return True
    return executable in _PYTHON_ENTRYPOINTS and len(tokens) >= 3 and tokens[1:3] == ["-m", "pytest"]


def result_fingerprint(message: ToolMessage) -> str:
    artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
    bounded = {
        "status": message.status,
        "exit_code": artifact.get("exit_code"),
        "operation": artifact.get("operation"),
        "path": artifact.get("path"),
        "content_hash": hashlib.sha256(message.text.encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha256(canonical_json(bounded).encode("utf-8")).hexdigest()[:32]
    return f"result_{digest}"


class ProgressEvaluator:
    _STRONG = {
        "test_observed": ProgressKind.TEST_EVIDENCE,
        "decision_evidence_observed": ProgressKind.DECISION_EVIDENCE,
        "hypothesis_supported": ProgressKind.HYPOTHESIS_TRANSITION,
        "hypothesis_rejected": ProgressKind.HYPOTHESIS_TRANSITION,
        "file_changed": ProgressKind.FILE_CHANGE,
        "post_edit_test_observed": ProgressKind.POST_EDIT_TEST,
        "user_information_received": ProgressKind.USER_INFORMATION,
    }

    def evaluate(self, observation: ToolObservation) -> ProgressKind | None:
        return self._STRONG.get(observation.event_type)

    def apply(
        self,
        state: InvestigationState,
        observation: ToolObservation,
    ) -> tuple[InvestigationState, ProgressKind | None]:
        progress = self.evaluate(observation)
        fingerprint = observation.result_fingerprint
        if progress in {ProgressKind.TEST_EVIDENCE, ProgressKind.POST_EDIT_TEST}:
            if fingerprint and fingerprint in state.seen_progress_fingerprints:
                return state, None
        if progress is not None and fingerprint:
            state = state.model_copy(update={"seen_progress_fingerprints": [fingerprint]})
        return state, progress
```

Add `ProgressEvaluator.apply(state, observation)` to suppress a test observation when the same normalized command signature and result fingerprint already occurred in the current progress generation. Event identity still includes tool_call_id, but result fingerprints deliberately do not. A newly read file, line range, code position, or changed content fingerprint is Tool Activity only; it becomes decision-relevant strong progress only through a whitelisted evidence event that changes a hypothesis, failure localization, scope edge, modification, test result, or user constraint.

Do not store raw ToolMessage text in `ToolObservation`; store a maximum 300-character safe summary only for deterministic test evidence and use content hashes elsewhere. Relation creation accepts only `import`, `call`, `traceback`, `grep_reference`, `symbol_reference`, or `test_collection` with a real source message ID. `continue_investigation.reason` 不能单独把 exploratory 目标提升为 dependency；only a Tool Result or existing system record containing one of those verified relation types may create a dependency edge.

- [ ] **Step 4: Run focused classification and progress tests**

Run: `pytest tests/investigation/test_classification.py tests/investigation/test_progress.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit deterministic classification**

```bash
git add src/deepfix/investigation/classification.py src/deepfix/investigation/progress.py tests/investigation/test_classification.py tests/investigation/test_progress.py
git commit -m "feat: classify investigation progress deterministically"
```

### Task 4: Phase Resolver and Evidence-Validated Hypotheses

**Files:**
- Create: `src/deepfix/investigation/phase.py`
- Create: `src/deepfix/investigation/coordinator.py`
- Modify: `src/deepfix/compaction/evidence.py`
- Create: `tests/investigation/test_phase.py`
- Create: `tests/investigation/test_coordinator.py`
- Create: `tests/investigation/helpers.py`
- Modify: `tests/compaction/test_evidence.py`

**Interfaces:**
- Consumes: InvestigationStore, TaskRepository, CompactionStore evidence, EvidenceCollector paired-result API, classified observations.
- Produces: `PhaseResolver.resolve()`, `InvestigationCoordinator.ensure_started()`, `record_tool_result()`, `record_observation()`, `record_hypothesis()`, `record_user_information()`, `state()`.

- [ ] **Step 1: Write failing phase and hypothesis-gate tests**

```python
# tests/investigation/helpers.py
def coordinator_fixture(tmp_path: Path) -> InvestigationCoordinator:
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    tasks.save(TaskState(
        task_id="task-a",
        project_root=str(tmp_path),
        project_python=sys.executable,
        user_problem="sign bug",
        approval_mode="manual",
        status=TaskStatus.INVESTIGATING,
    ))
    compaction = CompactionStore(database)
    return InvestigationCoordinator(
        store=InvestigationStore(database),
        tasks=tasks,
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, ResearchEvidenceStore(database)),
    )


def seed_evidence(store: CompactionStore, task_id: str, evidence_id: str) -> None:
    store.save_evidence(task_id, SystemTestEvidence(
        evidence_id=evidence_id,
        command="python -m pytest -q",
        exit_code=1,
        summary="1 failed",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"msg-{evidence_id}",
    ))


def seed_checked_location(
    store: InvestigationStore,
    task_id: str,
    path: str,
    start_line: int,
    end_line: int,
) -> None:
    state = store.ensure_started(task_id)
    event_id = stable_investigation_id("event", task_id, "checked", path)
    checked = CheckedFile(
        path=path,
        content_fingerprint="f" * 64,
        ranges=[CheckedLocation(path=path, start_line=start_line, end_line=end_line)],
        scope=ScopeKind.DIRECT,
        first_event_id=event_id,
        latest_event_id=event_id,
    )


def force_phase(
    store: InvestigationStore,
    state: InvestigationState,
    phase: AgentPhase,
) -> InvestigationState:
    event_id = stable_investigation_id("event", state.task_id, "force-phase", phase.value)
    return store.commit(
        state.version,
        [NewInvestigationEvent(
            event_id=event_id,
            task_id=state.task_id,
            event_type="phase_changed",
            phase_before=state.agent_phase,
            phase_after=phase,
        )],
        state.model_copy(update={"agent_phase": phase}),
    )
    store.commit(
        state.version,
        [NewInvestigationEvent(
            event_id=event_id,
            task_id=task_id,
            event_type="file_checked",
            phase_before=state.agent_phase,
            phase_after=state.agent_phase,
        )],
        state.model_copy(update={"checked_files": [checked]}),
    )


def supported_input(evidence_ids: list[str]) -> RecordHypothesisInput:
    return RecordHypothesisInput(
        statement="sign is inverted twice",
        evidence_ids=evidence_ids,
        checked_locations=[CheckedLocation(path="src/sign.py", start_line=1, end_line=20)],
        proposed_change=ProposedChange(path="src/sign.py", description="remove duplicate inversion"),
        expected_effect="sign=-1 returns the original negative value once",
        target_state="supported",
        reason="the failing branch and assertion agree",
    )


# tests/investigation/test_phase.py
def event(event_type: str, **updates) -> ToolObservation:
    return ToolObservation(event_type=event_type).model_copy(update=updates)


def test_pytest_failure_moves_investigating_to_diagnosing():
    assert PhaseResolver().resolve(AgentPhase.INVESTIGATING, event("test_observed", exit_code=1)) is AgentPhase.DIAGNOSING


def test_phase_changed_never_carries_progress_kind():
    result = PhaseResolver().transition("task-a", AgentPhase.DIAGNOSING, event("hypothesis_supported"))
    assert result.phase is AgentPhase.PLANNING
    assert result.phase_event.progress_kind is None


def test_supported_hypothesis_rejects_cross_task_evidence(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    seed_evidence(coordinator.compaction_store, "task-b", "evidence-b")
    seed_checked_location(coordinator.store, "task-a", "src/sign.py", 1, 20)
    with pytest.raises(ValueError, match="当前任务"):
        coordinator.record_hypothesis(
            "task-a", supported_input(evidence_ids=["evidence-b"]), source_id="tool-hyp-1"
        )


def test_supported_hypothesis_unlocks_planning_with_stable_id(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    seed_evidence(coordinator.compaction_store, "task-a", "evidence-a")
    seed_checked_location(coordinator.store, "task-a", "src/sign.py", 1, 20)
    first = coordinator.record_hypothesis(
        "task-a", supported_input(evidence_ids=["evidence-a"]), source_id="tool-hyp-1"
    )
    replay = coordinator.record_hypothesis(
        "task-a", supported_input(evidence_ids=["evidence-a"]), source_id="tool-hyp-1"
    )
    assert first.hypothesis_id == replay.hypothesis_id
    assert coordinator.state("task-a").agent_phase is AgentPhase.PLANNING


def test_state_read_failure_becomes_typed_recovery_error(tmp_path, monkeypatch):
    coordinator = coordinator_fixture(tmp_path)
    monkeypatch.setattr(coordinator.store, "load", lambda task_id: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    with pytest.raises(InvestigationStateError) as caught:
        coordinator.state("task-a")
    assert caught.value.recovery.error_code == "investigation_state_read_failed"
    assert caught.value.recovery.task_id == "task-a"


def test_tool_result_uses_shared_deterministic_evidence_identity(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    result = ToolMessage(
        id="msg-test-1",
        content="1 failed",
        tool_call_id="test-1",
        artifact={"exit_code": 1},
    )
    coordinator.record_tool_result(
        "task-a",
        {"name": "execute", "id": "test-1", "args": {"command": "python -m pytest -q"}},
        result,
    )
    evidence = coordinator.evidence_collector.store.list_evidence("task-a")
    assert coordinator.state("task-a").test_evidence_ids == [evidence[0].evidence_id]


def test_read_file_automatically_records_checked_range_without_strong_progress(tmp_path):
    coordinator = coordinator_fixture(tmp_path)
    before = coordinator.state("task-a")
    coordinator.record_tool_result(
        "task-a",
        {"name": "read_file", "id": "read-1", "args": {"file_path": "/src/sign.py", "offset": 10, "limit": 20}},
        ToolMessage(id="msg-read-1", content="return -value", tool_call_id="read-1"),
    )
    after = coordinator.state("task-a")
    assert after.checked_files[0].ranges[0].start_line == 11
    assert after.checked_files[0].ranges[0].end_line == 11
    assert after.progress_generation == before.progress_generation
```

- [ ] **Step 2: Run tests and verify resolver/coordinator failures**

Run: `pytest tests/investigation/test_phase.py tests/investigation/test_coordinator.py -v`

Expected: fails because phase and coordinator interfaces are absent.

- [ ] **Step 3: Implement deterministic transitions and coordinator validation**

```python
@dataclass(frozen=True)
class PhaseResolution:
    phase: AgentPhase
    phase_event: NewInvestigationEvent | None


class PhaseResolver:
    def resolve(self, phase: AgentPhase, event: ToolObservation) -> AgentPhase:
        if event.event_type == "needs_input":
            return AgentPhase.CLARIFYING
        if event.event_type == "test_observed" and event.exit_code != 0:
            return AgentPhase.DIAGNOSING
        if phase is AgentPhase.DIAGNOSING and event.event_type == "hypothesis_supported":
            return AgentPhase.PLANNING
        if phase is AgentPhase.PLANNING and event.event_type == "file_changed":
            return AgentPhase.EDITING
        if phase is AgentPhase.EDITING and event.event_type == "verification_execution_observed":
            return AgentPhase.TESTING
        if phase is AgentPhase.TESTING and event.event_type == "post_edit_test_observed":
            return AgentPhase.REVIEWING if event.exit_code == 0 else AgentPhase.DIAGNOSING
        if phase is AgentPhase.PLANNING and event.event_type == "file_change_failed":
            return AgentPhase.PLANNING
        if phase is AgentPhase.REVIEWING and event.event_type == "user_information_received":
            return AgentPhase.INVESTIGATING
        return phase

    def transition(self, task_id: str, phase: AgentPhase, event: ToolObservation) -> PhaseResolution:
        resolved = self.resolve(phase, event)
        if resolved is phase:
            return PhaseResolution(phase, None)
        phase_event = NewInvestigationEvent(
            event_id=stable_investigation_id("event", task_id, "phase", phase.value, resolved.value, event.result_fingerprint or event.event_type.value),
            task_id=task_id,
            event_type="phase_changed",
            source_message_id=event.source_message_id,
            tool_call_id=event.tool_call_id,
            phase_before=phase,
            phase_after=resolved,
            progress_kind=None,
            payload={"trigger": event.event_type.value},
        )
        return PhaseResolution(resolved, phase_event)
```

```python
class EvidenceCollector:
    def collect_pair(
        self,
        task_id: str,
        call: Mapping[str, object],
        result: ToolMessage,
        task_state: TaskState,
    ) -> DeterministicEvidence | None:
        block = self.collect(
            task_id,
            [AIMessage(content="", tool_calls=[call]), result],
            task_state,
        )
        call_id = str(call["id"])
        return next(
            (
                item
                for item in [*block.tests, *block.files]
                if item.tool_call_id == call_id
            ),
            None,
        )
```

Keep `_test_evidence`, `_file_evidence`, and `_evidence_id` as the single identity implementation. Add a compaction regression test proving `collect()` and `collect_pair()` produce the same evidence ID for the same pair.

```python
class InvestigationCoordinator:
    def state(self, task_id: str) -> InvestigationState:
        try:
            return self.store.load(task_id) or self.store.ensure_started(task_id)
        except InvestigationCoordinationError:
            raise
        except Exception as exc:
            raise InvestigationStateError(self.recovery(
                task_id,
                "investigation_state_read_failed",
                checkpoint_available=True,
                recovery_action="retry_state_read_without_model_call",
            )) from exc

    def record_hypothesis(
        self,
        task_id: str,
        command: RecordHypothesisInput,
        *,
        source_id: str,
    ) -> InvestigationHypothesis:
        state = self.store.ensure_started(task_id)
        evidence_ids = {item.evidence_id for item in self.compaction_store.list_evidence(task_id)}
        if not set(command.evidence_ids) <= evidence_ids:
            raise ValueError("假设证据不属于当前任务")
        if not all(self._location_was_checked(state, item) for item in command.checked_locations):
            raise ValueError("假设位置尚未被当前任务检查")
        first_source = command.evidence_ids[0] if command.evidence_ids else source_id
        hypothesis_id = command.hypothesis_id or stable_investigation_id(
            "hyp", task_id, first_source, command.statement
        )
        existing = self._hypothesis(state, hypothesis_id)
        if command.hypothesis_id and existing is None:
            raise ValueError("hypothesis_id 不存在于当前任务")
        observation = self._hypothesis_observation(hypothesis_id, command)
        updated = self.record_observation(task_id, observation)
        return next(item for item in updated.hypotheses if item.hypothesis_id == hypothesis_id)
```

When `_apply_observation` changes phase, commit the originating event and a separate derived `phase_changed` event in one transaction. Increment `progress_generation` once for the originating strong event; force `phase_changed.progress_kind=None`. Candidate hypotheses do not unlock tools; supported requires all Pydantic fields plus current-task evidence and checked locations.

- [ ] **Step 4: Run phase and coordinator tests**

Run: `pytest tests/investigation/test_phase.py tests/investigation/test_coordinator.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit phase authority and hypothesis gate**

```bash
git add src/deepfix/investigation/phase.py src/deepfix/investigation/coordinator.py src/deepfix/compaction/evidence.py tests/investigation/test_phase.py tests/investigation/test_coordinator.py tests/investigation/helpers.py tests/compaction/test_evidence.py
git commit -m "feat: gate repair phases on investigation evidence"
```

### Task 5: Stagnation Detector and One-Shot Permit

**Files:**
- Create: `src/deepfix/investigation/stagnation.py`
- Create: `tests/investigation/test_stagnation.py`
- Modify: `src/deepfix/investigation/coordinator.py`
- Modify: `tests/investigation/helpers.py`

**Interfaces:**
- Consumes: tool signatures, scope, progress result, ContinueInvestigationInput.
- Produces: `StagnationDetector.after_tool()`, `matching_cycle_size()`, `coordinator.authorize_tool()`, `grant_investigation_permit()`.

- [ ] **Step 1: Write failing threshold, reset, and permit tests**

```python
def stagnated_coordinator(tmp_path: Path) -> InvestigationCoordinator:
    coordinator = coordinator_fixture(tmp_path)
    state = coordinator.store.ensure_started("task-a")
    hypothesis = InvestigationHypothesis(
        hypothesis_id="hyp-1",
        statement="sign flips in helper",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate for one bounded follow-up",
    )
    event_id = stable_investigation_id("event", "task-a", "stagnated-fixture")
    coordinator.store.commit(
        state.version,
        [NewInvestigationEvent(
            event_id=event_id,
            task_id="task-a",
            event_type="reevaluation_required",
            phase_before=state.agent_phase,
            phase_after=state.agent_phase,
        )],
        state.model_copy(update={
            "hypotheses": [hypothesis],
            "reevaluation_required": True,
            "stagnation_level": 1,
        }),
    )
    return coordinator


def event(event_type: str, progress: str | None = None) -> ToolObservation:
    return ToolObservation(event_type=event_type, progress_kind=progress)


def no_progress(signature: str, scope: str = "direct") -> ToolObservation:
    return ToolObservation(
        event_type="tool_completed",
        signature=signature,
        result_fingerprint=f"result-{signature}",
        scope=scope,
    )


def state_with_no_progress_count(count: int) -> InvestigationState:
    return InvestigationState.new("task-a").model_copy(
        update={"no_progress_count": count, "recent_tool_signatures": [f"old-{index}" for index in range(count)]}
    )


detector = StagnationDetector()


def test_exact_repeat_triggers_on_third_result():
    state = InvestigationState.new("task-a")
    for index in range(2):
        state = detector.after_tool(state, no_progress("same-signature"))
        assert state.stagnation_level == 0
    state = detector.after_tool(state, no_progress("same-signature"))
    assert state.stagnation_level == 1


@pytest.mark.parametrize("size", [2, 8])
def test_short_cycle_repeated_twice_triggers(size):
    signatures = [f"s-{index}" for index in range(size)] * 2
    assert matching_cycle_size(signatures) == size


def test_phase_change_and_new_file_do_not_reset_stagnation():
    state = state_with_no_progress_count(5)
    state = detector.after_event(state, event("phase_changed"))
    state = detector.after_tool(state, no_progress("read-new-file", scope="exploratory"))
    assert state.stagnation_level == 1


def test_strong_progress_resets_generation_and_counters():
    state = state_with_no_progress_count(5)
    updated = detector.after_event(state, event("hypothesis_rejected", progress="hypothesis_transition"))
    assert updated.progress_generation == state.progress_generation + 1
    assert updated.no_progress_count == 0
    assert updated.stagnation_level == 0


def test_permit_allows_only_bound_tool_once_and_then_pauses(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)
    permit = coordinator.grant_investigation_permit(
        "task-a",
        ContinueInvestigationInput(
            hypothesis_ids=["hyp-1"],
            unresolved_question="which call flips sign?",
            expected_evidence="a verified call edge",
            tool_name="grep",
            target="src/sign.py|flip",
            reason="test evidence points to hyp-1",
        ),
    )
    assert coordinator.authorize_tool("task-a", "grep", {"pattern": "flip", "path": "src/sign.py"}).allowed
    coordinator.record_observation("task-a", no_progress("permitted-grep"))
    with pytest.raises(InvestigationStagnationError):
        coordinator.authorize_tool("task-a", "read_file", {"file_path": "src/other.py"})
```

- [ ] **Step 2: Run stagnation tests and verify failure**

Run: `pytest tests/investigation/test_stagnation.py -v`

Expected: fails because stagnation logic is absent.

- [ ] **Step 3: Implement exact thresholds and permit state machine**

```python
def matching_cycle_size(signatures: Sequence[str]) -> int | None:
    for size in range(2, min(8, len(signatures) // 2) + 1):
        if list(signatures[-size:]) == list(signatures[-2 * size : -size]):
            return size
    return None


class StagnationDetector:
    def after_event(self, state: InvestigationState, observation: ToolObservation) -> InvestigationState:
        if observation.event_type is InvestigationEventType.PHASE_CHANGED:
            return state
        if observation.progress_kind is None:
            return state
        return state.model_copy(update={
            "progress_generation": state.progress_generation + 1,
            "no_progress_count": 0,
            "exploratory_without_progress": 0,
            "recent_tool_signatures": [],
            "seen_progress_fingerprints": (
                [observation.result_fingerprint]
                if observation.result_fingerprint
                else []
            ),
            "reevaluation_required": False,
            "stagnation_level": 0,
            "permit": None,
            "post_permit_review_pending": False,
        })

    def after_tool(self, state: InvestigationState, observation: ToolObservation) -> InvestigationState:
        if observation.progress_kind is not None:
            return self.after_event(state, observation)
        signatures = [*state.recent_tool_signatures, observation.signature][-32:]
        no_progress = state.no_progress_count + 1
        exploratory = state.exploratory_without_progress + (observation.scope == "exploratory")
        repeated = signatures.count(observation.signature) >= 3
        stalled = repeated or matching_cycle_size(signatures) is not None or no_progress >= 6 or exploratory >= 4
        return state.model_copy(update={
            "recent_tool_signatures": signatures,
            "no_progress_count": no_progress,
            "exploratory_without_progress": exploratory,
            "reevaluation_required": stalled,
            "stagnation_level": 1 if stalled else state.stagnation_level,
        })
```

`grant_investigation_permit` validates current-task hypothesis IDs and stores a stable permit bound to canonical tool name plus target hash. `authorize_tool` consumes only the exact match. After the permitted result it sets `post_permit_review_pending=True`; `record_hypothesis` strong progress clears it, `save_progress` does not, and any further ordinary investigation request raises `InvestigationStagnationError` before handler execution.

- [ ] **Step 4: Run stagnation and coordinator regression tests**

Run: `pytest tests/investigation/test_stagnation.py tests/investigation/test_coordinator.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit stagnation control**

```bash
git add src/deepfix/investigation/stagnation.py src/deepfix/investigation/coordinator.py tests/investigation/test_stagnation.py
git commit -m "feat: stop stagnant investigation loops"
```

### Task 6: Structured Hypothesis and Continue Tools

**Files:**
- Create: `src/deepfix/investigation/tools.py`
- Create: `tests/investigation/test_tools.py`

**Interfaces:**
- Consumes: InvestigationCoordinator and LangChain ToolRuntime.
- Produces: `build_record_hypothesis_tool()` and `build_continue_investigation_tool()`.

- [ ] **Step 1: Write failing ToolNode tests**

```python
def invoke_tool(tool: BaseTool, call: dict[str, object], task_id: str) -> ToolMessage:
    node = ToolNode([tool])
    return node.invoke(
        {"messages": [AIMessage(content="", tool_calls=[call])]},
        {"configurable": {"thread_id": task_id}},
        runtime=Runtime(),
    )["messages"][0]


def invalid_supported_call(call_id: str) -> dict[str, object]:
    return {
        "name": "record_hypothesis",
        "id": call_id,
        "type": "tool_call",
        "args": {
            "statement": "maybe sign",
            "evidence_ids": [],
            "checked_locations": [],
            "target_state": "supported",
            "reason": "guess",
        },
    }


def valid_continue_call(call_id: str) -> dict[str, object]:
    return {
        "name": "continue_investigation",
        "id": call_id,
        "type": "tool_call",
        "args": {
            "hypothesis_ids": ["hyp-1"],
            "unresolved_question": "which call flips sign?",
            "expected_evidence": "a call edge",
            "tool_name": "grep",
            "target": "src/sign.py|flip",
            "reason": "bounded follow-up for hyp-1",
        },
    }


def test_record_hypothesis_schema_hides_task_authority(tmp_path):
    tool = build_record_hypothesis_tool(coordinator_fixture(tmp_path))
    assert "task_id" not in tool.args
    assert "statement" in tool.args
    assert "evidence_ids" in tool.args


def test_invalid_hypothesis_returns_stable_error_tool_message(tmp_path):
    tool = build_record_hypothesis_tool(coordinator_fixture(tmp_path))
    result = invoke_tool(tool, invalid_supported_call("call-1"), "task-a")
    assert result.status == "error"
    assert result.id == stable_generated_message_id("task-a", "call-1", "hypothesis_validation_error")


def test_continue_tool_returns_bound_permit(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)
    tool = build_continue_investigation_tool(coordinator)
    result = invoke_tool(tool, valid_continue_call("call-2"), "task-a")
    assert result.status == "success"
    assert result.artifact["tool_name"] == "grep"
    assert result.artifact["permit_id"]
```

- [ ] **Step 2: Run Tool tests and verify missing builders**

Run: `pytest tests/investigation/test_tools.py -v`

Expected: fails on missing tool builders.

- [ ] **Step 3: Implement runtime-scoped StructuredTools**

```python
def _task_id(runtime: ToolRuntime) -> str:
    return str(runtime.config.get("configurable", {}).get("thread_id", "")).strip()


def build_record_hypothesis_tool(coordinator: InvestigationCoordinator) -> BaseTool:
    def record_hypothesis(
        statement: str,
        evidence_ids: list[str],
        checked_locations: list[CheckedLocation],
        target_state: Literal["candidate", "rejected", "supported"],
        reason: str,
        runtime: ToolRuntime,
        hypothesis_id: str | None = None,
        proposed_change: ProposedChange | None = None,
        expected_effect: str | None = None,
    ) -> ToolMessage:
        task_id = _task_id(runtime)
        if not runtime.tool_call_id:
            return stable_error_message(task_id, "missing-call-id", "hypothesis_validation_error", "缺少 tool_call_id")
        try:
            record = coordinator.record_hypothesis(
                task_id,
                RecordHypothesisInput(
                    hypothesis_id=hypothesis_id,
                    statement=statement,
                    evidence_ids=evidence_ids,
                    checked_locations=checked_locations,
                    proposed_change=proposed_change,
                    expected_effect=expected_effect,
                    target_state=target_state,
                    reason=reason,
                ),
                source_id=runtime.tool_call_id,
            )
        except (ValidationError, ValueError) as exc:
            return stable_error_message(task_id, runtime.tool_call_id, "hypothesis_validation_error", str(exc))
        return stable_success_message(task_id, runtime.tool_call_id, "hypothesis_recorded", {
            "hypothesis_id": record.hypothesis_id,
            "state": record.state,
        })

    return StructuredTool.from_function(record_hypothesis, name="record_hypothesis", description=RECORD_DESCRIPTION)
```

Implement the continue tool with the same runtime-derived task ID rule. Catch only Pydantic/command validation errors; let `InvestigationCoordinationError` propagate to Service. Bound error text to 300 characters and never echo full model input.

- [ ] **Step 4: Run focused Tool tests**

Run: `pytest tests/investigation/test_tools.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit structured tools**

```bash
git add src/deepfix/investigation/tools.py tests/investigation/test_tools.py
git commit -m "feat: add investigation transition tools"
```

### Task 7: Investigation Middleware and Deep Agents Hook Contract

**Files:**
- Create: `src/deepfix/investigation/middleware.py`
- Create: `src/deepfix/investigation/receipts.py`
- Create: `tests/investigation/test_middleware.py`
- Modify: `src/deepfix/investigation/coordinator.py`
- Modify: `src/deepfix/protected_context.py`
- Modify: `tests/test_protected_context.py`

**Interfaces:**
- Consumes: ModelRequest, ToolCallRequest, InvestigationCoordinator, Backend, tool capability map.
- Produces: `ToolExecutionReceiptStore`, `InvestigationMiddleware.wrap_model_call()`, `wrap_tool_call()`, bounded `<deepfix_investigation_state>`, pre-execution gates, and replay-safe Tool Result recovery.

- [ ] **Step 1: Write failing model projection, visibility, HITL, and Tool Result tests**

```python
def named_tools(*names: str) -> list[BaseTool]:
    def run() -> str:
        return "ok"
    return [StructuredTool.from_function(run, name=name, description=f"{name} test") for name in names]


def model_request(*, tools: list[BaseTool]) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="continue")],
        system_message=SystemMessage(content="base"),
        tools=tools,
        state={"messages": []},
        runtime=Runtime(execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="model-node",
            thread_id="task-a",
        )),
    )


def capture_model_request(middleware: InvestigationMiddleware, request: ModelRequest) -> ModelRequest:
    captured: list[ModelRequest] = []
    middleware.wrap_model_call(
        request,
        lambda updated: captured.append(updated) or ModelResponse(result=[AIMessage(content="ok")]),
    )
    return captured[0]


def tool_request(name: str, call_id: str, args: dict[str, object]) -> ToolCallRequest:
    tool = named_tools(name)[0]
    runtime = ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": "task-a"}},
        stream_writer=lambda value: None,
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args, "type": "tool_call"},
        tool=tool,
        state={"messages": []},
        runtime=runtime,
    )


def pytest_tool_request() -> ToolCallRequest:
    return tool_request("execute", "pytest-1", {"command": "python -m pytest -q"})


def read_request() -> ToolCallRequest:
    return tool_request("read_file", "read-1", {"file_path": "/src/sign.py"})


def edit_request(call_id: str) -> ToolCallRequest:
    return tool_request("edit_file", call_id, {
        "file_path": "/src/sign.py",
        "old_string": "return -value",
        "new_string": "return value",
    })


def pytest_result(exit_code: int) -> ToolMessage:
    return ToolMessage(
        id="msg-pytest-1",
        content=f"pytest exit {exit_code}",
        tool_call_id="pytest-1",
        artifact={"exit_code": exit_code},
    )


def edit_result(call_id: str) -> ToolMessage:
    return ToolMessage(
        id=f"msg-{call_id}",
        content="edited",
        tool_call_id=call_id,
        artifact={"operation": "edit", "status": "succeeded", "path": "/src/sign.py"},
    )


def read_result() -> ToolMessage:
    return ToolMessage(id="msg-read-1", content="return -value", tool_call_id="read-1")


def all_test_tools() -> list[BaseTool]:
    return named_tools(
        "read_file", "grep", "execute", "edit_file", "record_hypothesis",
        "continue_investigation", "save_progress", "compact_conversation",
    )


def middleware_fixture(
    tmp_path: Path,
    *,
    phase: str = "investigating",
    stagnation_level: int = 0,
    fail_next_event_commit: bool = False,
) -> InvestigationMiddleware:
    coordinator = coordinator_fixture(tmp_path)
    state = coordinator.store.ensure_started("task-a")
    state = force_phase(coordinator.store, state, AgentPhase(phase))
    candidate = InvestigationHypothesis(
        hypothesis_id="hyp-1",
        statement="sign flips in helper",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate",
    )
    state = coordinator.store.commit(
        state.version,
        [NewInvestigationEvent(
            event_id=stable_investigation_id("event", "task-a", "middleware-fixture"),
            task_id="task-a",
            event_type="hypothesis_recorded",
            phase_before=state.agent_phase,
            phase_after=state.agent_phase,
        )],
        state.model_copy(update={
            "hypotheses": [candidate],
            "stagnation_level": stagnation_level,
            "reevaluation_required": stagnation_level > 0,
        }),
    )
    if fail_next_event_commit:
        original_record = coordinator.record_tool_result
        failed = False
        def fail_once(task_id, tool_call, result):
            nonlocal failed
            if not failed:
                failed = True
                current = coordinator.state(task_id)
                raise InvestigationStateError(InvestigationRecoveryMetadata(
                    task_id=task_id,
                    error_code="investigation_event_commit_failed",
                    agent_phase=current.agent_phase,
                    state_version=current.version,
                    last_event_sequence=coordinator.store.last_sequence(task_id),
                    tool_call_id=str(tool_call["id"]),
                    checkpoint_available=True,
                    recovery_action="replay_event_commit_without_rerunning_tool",
                ))
            return original_record(task_id, tool_call, result)
        coordinator.record_tool_result = fail_once
    capabilities = {
        "read_file": InvestigationCapability.READ,
        "grep": InvestigationCapability.SEARCH,
        "execute": InvestigationCapability.EXECUTE,
        "edit_file": InvestigationCapability.MODIFY,
        "record_hypothesis": InvestigationCapability.META,
        "continue_investigation": InvestigationCapability.META,
        "save_progress": InvestigationCapability.MEMORY,
        "compact_conversation": InvestigationCapability.COMPACTION,
    }
    backend = FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    return InvestigationMiddleware(
        coordinator,
        ToolExecutionReceiptStore(backend),
        capabilities,
    )


def test_diagnosing_request_hides_modify_tools_and_deduplicates_ids(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="diagnosing")
    request = model_request(tools=named_tools("read_file", "edit_file", "record_hypothesis"))
    captured = capture_model_request(middleware, request)
    assert [tool.name for tool in captured.tools] == ["read_file", "record_hypothesis"]
    assert captured.system_message.text.count("hyp-1") == 1


def test_protected_projection_owns_hypothesis_and_evidence_entities_once(tmp_path):
    protected, investigation = protected_and_investigation_fixture(tmp_path, shared_hypothesis_id="hyp-1", shared_evidence_id="evidence-1")
    rendered = protected.render_with_investigation(investigation.state("task-a"))
    assert rendered.count('hypothesis_id="hyp-1"') == 1
    assert rendered.count('evidence_id="evidence-1"') == 1
    assert 'investigation_support="supported"' in rendered


def test_level_one_exposes_only_three_meta_tools(tmp_path):
    middleware = middleware_fixture(tmp_path, stagnation_level=1)
    captured = capture_model_request(middleware, model_request(tools=all_test_tools()))
    assert {tool.name for tool in captured.tools} == {"record_hypothesis", "continue_investigation", "save_progress"}


def test_interrupt_command_does_not_mark_pytest_as_executed(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="editing")
    result = middleware.wrap_tool_call(pytest_tool_request(), lambda request: Command(goto="approval"))
    assert isinstance(result, Command)
    assert middleware.coordinator.state("task-a").agent_phase is AgentPhase.EDITING


def test_real_pytest_tool_message_records_testing_then_failure(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="editing")
    result = middleware.wrap_tool_call(pytest_tool_request(), lambda request: pytest_result(exit_code=1))
    events = middleware.coordinator.store.list_events("task-a")
    assert [item.event_type for item in events[-4:]] == [
        "verification_execution_observed", "phase_changed", "test_observed", "phase_changed"
    ]
    assert result.artifact["exit_code"] == 1
    assert middleware.coordinator.state("task-a").agent_phase is AgentPhase.DIAGNOSING


def test_level_two_blocks_before_handler_execution(tmp_path):
    middleware = middleware_fixture(tmp_path, stagnation_level=2)
    called = False
    def handler(request):
        nonlocal called
        called = True
        return read_result()
    with pytest.raises(InvestigationStagnationError):
        middleware.wrap_tool_call(read_request(), handler)
    assert called is False


def test_committed_tool_receipt_prevents_side_effect_reexecution(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="planning", fail_next_event_commit=True)
    executions = 0
    def handler(request):
        nonlocal executions
        executions += 1
        return edit_result(call_id="edit-1")
    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(edit_request(call_id="edit-1"), handler)
    middleware.wrap_tool_call(edit_request(call_id="edit-1"), handler)
    assert executions == 1
    assert middleware.receipts.load("task-a", "edit-1").tool_message.tool_call_id == "edit-1"
```

- [ ] **Step 2: Run middleware tests and verify missing adapter**

Run: `pytest tests/investigation/test_middleware.py -v`

Expected: fails because InvestigationMiddleware is absent.

- [ ] **Step 3: Implement model filtering and Tool hook sequencing**

```python
class ToolExecutionReceipt(StrictModel):
    task_id: str
    tool_call_id: str
    tool_name: str
    call_hash: str
    tool_message: dict[str, JsonValue]
    result_fingerprint: str


class ToolExecutionReceiptStore:
    def save(self, receipt: ToolExecutionReceipt) -> None:
        path = f"/investigation_receipts/{receipt.task_id}/{receipt.tool_call_id}.json"
        payload = receipt.model_dump_json(indent=2)
        written = self.backend.write(path, payload)
        if written.error:
            raise RuntimeError("tool execution receipt 写入失败")
        read = self.backend.read(path)
        content = None if read.file_data is None else read.file_data["content"]
        if read.error or content != payload:
            raise RuntimeError("tool execution receipt 校验失败")

    def load(self, task_id: str, tool_call_id: str) -> ToolExecutionReceipt | None:
        path = f"/investigation_receipts/{task_id}/{tool_call_id}.json"
        result = self.backend.read(path)
        if result.error:
            return None
        if result.file_data is None:
            raise RuntimeError("tool execution receipt 缺少内容")
        return ToolExecutionReceipt.model_validate_json(result.file_data["content"])


def receipt_from_result(task_id: str, tool_call: dict, result: ToolMessage) -> ToolExecutionReceipt:
    call_hash = stable_investigation_id("call", task_id, canonical_json(tool_call))
    return ToolExecutionReceipt(
        task_id=task_id,
        tool_call_id=str(tool_call["id"]),
        tool_name=str(tool_call["name"]),
        call_hash=call_hash,
        tool_message=message_to_dict(result),
        result_fingerprint=result_fingerprint(result),
    )
```

Add a focused receipt test using `FilesystemBackend`: assert `WriteResult.error is None`, `ReadResult.file_data["content"]` equals the serialized receipt, and task A cannot resolve task B's receipt path.

```python
_PHASE_CAPABILITIES = {
    AgentPhase.CLARIFYING: {InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
    AgentPhase.INVESTIGATING: {InvestigationCapability.READ, InvestigationCapability.SEARCH, InvestigationCapability.EXECUTE, InvestigationCapability.RESEARCH, InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
    AgentPhase.DIAGNOSING: {InvestigationCapability.READ, InvestigationCapability.SEARCH, InvestigationCapability.EXECUTE, InvestigationCapability.RESEARCH, InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
    AgentPhase.PLANNING: set(InvestigationCapability),
    AgentPhase.EDITING: {InvestigationCapability.READ, InvestigationCapability.SEARCH, InvestigationCapability.EXECUTE, InvestigationCapability.MODIFY, InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
    AgentPhase.TESTING: {InvestigationCapability.READ, InvestigationCapability.SEARCH, InvestigationCapability.EXECUTE, InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
    AgentPhase.REVIEWING: {InvestigationCapability.READ, InvestigationCapability.SEARCH, InvestigationCapability.EXECUTE, InvestigationCapability.META, InvestigationCapability.MEMORY, InvestigationCapability.COMPACTION},
}
_REEVALUATION_TOOL_NAMES = {"record_hypothesis", "continue_investigation", "save_progress"}


class InvestigationMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        task_id = model_request_task_id(request)
        state = self.coordinator.state(task_id)
        allowed = self.coordinator.allowed_tool_names(state, self.capabilities)
        tools = [tool for tool in request.tools or [] if tool.name in allowed]
        block = render_investigation_state(state, request.system_message.text if request.system_message else "")
        system = append_system_block(request.system_message, block)
        return handler(request.override(tools=tools, system_message=system))

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        task_id = runtime_task_id(request.runtime)
        name = str(request.tool_call["name"])
        args = request.tool_call.get("args", {})
        self.coordinator.authorize_tool(task_id, name, args)
        receipt = self.receipts.load(task_id, str(request.tool_call["id"]))
        if receipt is not None:
            expected_call_hash = stable_investigation_id("call", task_id, canonical_json(request.tool_call))
            if receipt.call_hash != expected_call_hash:
                raise InvestigationStateError(self.coordinator.recovery(
                    task_id,
                    "tool_receipt_call_mismatch",
                    tool_call_id=str(request.tool_call["id"]),
                ))
            restored = messages_from_dict([receipt.tool_message])[0]
            if not isinstance(restored, ToolMessage):
                raise RuntimeError("tool execution receipt 不是 ToolMessage")
            result = restored
        else:
            result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        if receipt is None:
            self.receipts.save(receipt_from_result(task_id, request.tool_call, result))
        if name == "execute" and is_pytest_verification(str(args.get("command", "")), self.coordinator.project_python(task_id)):
            self.coordinator.record_verification_execution(task_id, request.tool_call, result)
        self.coordinator.record_tool_result(task_id, request.tool_call, result)
        return result
```

Keep original ToolMessage identity, artifact, status, and content unchanged. The receipt is a recovery artifact, not an investigation event or diagnostic retrieval index. Save and verify it after actual handler execution but before event commit; on checkpoint replay return the receipt without calling the handler, then idempotently commit the event. Use result presence—not approval or Tool Call proposal—as proof execution occurred. In async mode implement the same order in `awrap_model_call` and `awrap_tool_call`; call the async handler at most once.

Extend ProtectedContextBuilder/renderer with the current InvestigationState. Protected Context owns unique hypothesis/evidence entities: merge `investigation_support` into a matching Working Memory hypothesis, add an Investigation hypothesis only when Working Memory lacks that ID, and keep deterministic evidence as the sole evidence-ID owner. The `<deepfix_investigation_state>` block renders phase, generation, gate/counts, at most 8 checked paths, and 8 recent signatures; it renders no hypothesis/evidence IDs. Tests must prove each `hypothesis_id` and `evidence_id` occurs once in the final ModelRequest.

- [ ] **Step 4: Run middleware and compaction pairing tests**

Run: `pytest tests/investigation/test_middleware.py tests/test_protected_context.py tests/compaction/test_work_units.py tests/compaction/test_middleware.py -v`

Expected: all tests pass and original ToolMessage pairing remains intact.

- [ ] **Step 5: Commit middleware adapter**

```bash
git add src/deepfix/investigation/middleware.py src/deepfix/investigation/receipts.py src/deepfix/investigation/coordinator.py src/deepfix/protected_context.py tests/investigation/test_middleware.py tests/test_protected_context.py
git commit -m "feat: observe and gate internal investigation tools"
```

### Task 8: Prompt Authority and Extension Capability Metadata

**Files:**
- Modify: `src/deepfix/prompting.py`
- Modify: `src/deepfix/prompts.py`
- Modify: `src/deepfix/extensions.py`
- Modify: `src/deepfix/context.py`
- Modify: `tests/test_prompting.py`
- Modify: `tests/test_extensions.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_approval.py`

**Interfaces:**
- Consumes: InvestigationStore AgentPhase and InvestigationCapability.
- Produces: `PromptPolicyMiddleware(InvestigationStore)`, diagnosing prompt, explicit `ToolRegistration.investigation_capability`.

- [ ] **Step 1: Replace Working Memory phase tests with InvestigationStore authority tests**

```python
def test_prompt_policy_uses_investigation_phase_not_working_memory(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    investigation = InvestigationStore(database)
    state = investigation.ensure_started("task-a")
    force_phase(investigation, state, AgentPhase.DIAGNOSING)
    memory = WorkingMemoryStore(database)
    memory.save("task-a", _snapshot("editing", "stale memory"))
    received = _capture(PromptPolicyMiddleware(investigation), _request("task-a"))
    assert '<deepfix_phase name="diagnosing">' in received.system_message.text
    assert '<deepfix_phase name="editing">' not in received.system_message.text


def test_extension_capability_is_required():
    registration = ToolRegistration(
        tool=_tool("inspect"),
        risk=RiskLevel.L0,
        policy_action=PolicyAction.ALLOW,
        network_access=False,
        investigation_capability=None,
    )
    with pytest.raises(ValueError, match="investigation_capability"):
        merge_extensions(AgentExtensions(tools=(registration,)))


def test_research_tools_declare_read_or_research_capability():
    extensions = build_research_extensions(
        inspect_dependency=_tool("inspect_dependency"),
        search_technical_sources=_tool("search_technical_sources"),
        fetch_external_evidence=_tool("fetch_external_evidence"),
        link_external_evidence=_tool("link_external_evidence"),
    )
    capabilities = {item.tool.name: item.investigation_capability for item in extensions.tools}
    assert capabilities["inspect_dependency"] is InvestigationCapability.READ
    assert capabilities["search_technical_sources"] is InvestigationCapability.RESEARCH
```

- [ ] **Step 2: Run prompt and extension tests and verify failures**

Run: `pytest tests/test_prompting.py tests/test_extensions.py -v`

Expected: old constructor/phase expectations fail.

- [ ] **Step 3: Switch PromptPolicy to AgentPhase and add explicit capability metadata**

```python
class PromptPolicyMiddleware(AgentMiddleware):
    def __init__(self, store: InvestigationStore) -> None:
        self.store = store

    def wrap_model_call(self, request: ModelRequest, handler):
        task_id = model_request_task_id(request)
        phase = self.store.ensure_started(task_id).agent_phase.value
        parts = compose_prompt_parts(request.system_message, PHASE_PROMPTS[phase])
        return handler(request.override(system_message=SystemMessage(content="\n\n".join(parts))))
```

Add a `diagnosing` phase prompt requiring `record_hypothesis` before planning. Update editing prompt to return to diagnosing, not Working Memory investigating. Extend `ToolRegistration` with `investigation_capability: InvestigationCapability | None`; reject `None` in `_validate_registration`. Assign research capabilities explicitly and add `InvestigationMiddleware`/`InvestigationMigrationMiddleware` to protected middleware names. Change the legacy `build_context_middleware` helper to accept an `InvestigationStore` parameter and pass it to PromptPolicy; update `tests/test_context.py`. Add explicit capabilities to the ToolRegistration factories in `tests/test_approval.py` and `tests/test_extensions.py`.

- [ ] **Step 4: Run prompt and extension tests**

Run: `pytest tests/test_prompting.py tests/test_extensions.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit prompt and extension authority changes**

```bash
git add src/deepfix/prompting.py src/deepfix/prompts.py src/deepfix/extensions.py src/deepfix/context.py tests/test_prompting.py tests/test_extensions.py tests/test_context.py tests/test_approval.py
git commit -m "feat: drive prompts from investigation phase"
```

### Task 9: Idempotent Legacy Investigation Migration

**Files:**
- Create: `src/deepfix/investigation/migration.py`
- Create: `tests/investigation/test_migration.py`

**Interfaces:**
- Consumes: TaskRepository, paired Graph messages with stable IDs, CompactionStore evidence, WorkingMemoryStore candidate hypotheses.
- Produces: `InvestigationMigrator.migrate(task_id, messages)` and `InvestigationMigrationMiddleware.before_agent()`.

- [ ] **Step 1: Write failing phase reconstruction and replay tests**

```python
def migrator_fixture(
    tmp_path: Path,
    status: TaskStatus,
    *,
    memory_phase: str | None = None,
    active_hypothesis: str | None = None,
) -> InvestigationMigrator:
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    task = TaskState(
        task_id="task-a",
        project_root=str(tmp_path),
        project_python=sys.executable,
        user_problem="legacy bug",
        approval_mode="manual",
        status=status,
    )
    tasks.save(task)
    memory = WorkingMemoryStore(database)
    if memory_phase is not None:
        memory.save("task-a", ProgressSnapshot(
            phase=memory_phase,
            summary="legacy memory",
            facts=[],
            evidence=[],
            active_hypotheses=[] if active_hypothesis is None else [active_hypothesis],
            rejected_hypotheses=[],
            checked_files=[],
            experiments=[],
            next_steps=[],
            unresolved_questions=[],
        ))
    return InvestigationMigrator(
        tasks=tasks,
        store=InvestigationStore(database),
        compaction_store=CompactionStore(database),
        memory=memory,
    )


def tool_pair(name: str, call_id: str, args: dict[str, object], artifact: dict[str, object]) -> list[AnyMessage]:
    return [
        AIMessage(
            id=f"ai-{call_id}",
            content="",
            tool_calls=[{"name": name, "id": call_id, "args": args, "type": "tool_call"}],
        ),
        ToolMessage(
            id=f"tool-{call_id}",
            content="result",
            tool_call_id=call_id,
            artifact=artifact,
        ),
    ]


def pytest_pair(exit_code: int) -> list[AnyMessage]:
    return tool_pair("execute", "pytest-1", {"command": "python -m pytest -q"}, {"exit_code": exit_code})


def pytest_failure_then_edit() -> list[AnyMessage]:
    return [
        *pytest_pair(1),
        *tool_pair("edit_file", "edit-1", {"file_path": "/src/sign.py"}, {
            "operation": "edit", "status": "succeeded", "path": "/src/sign.py"
        }),
    ]


def edit_then_pytest(exit_code: int) -> list[AnyMessage]:
    return [
        *tool_pair("edit_file", "edit-1", {"file_path": "/src/sign.py"}, {
            "operation": "edit", "status": "succeeded", "path": "/src/sign.py"
        }),
        *pytest_pair(exit_code),
    ]


@pytest.mark.parametrize(
    ("messages", "task_status", "expected"),
    [
        ([], TaskStatus.INVESTIGATING, AgentPhase.INVESTIGATING),
        (pytest_pair(exit_code=1), TaskStatus.TESTING, AgentPhase.DIAGNOSING),
        (pytest_failure_then_edit(), TaskStatus.EDITING, AgentPhase.EDITING),
        (edit_then_pytest(exit_code=0), TaskStatus.TESTING, AgentPhase.REVIEWING),
        ([], TaskStatus.CLARIFYING, AgentPhase.CLARIFYING),
    ],
)
def test_legacy_phase_is_reconstructed_from_durable_evidence(tmp_path, messages, task_status, expected):
    migrator = migrator_fixture(tmp_path, task_status)
    assert migrator.migrate("task-a", messages).agent_phase is expected


def test_working_memory_phase_and_unproven_hypothesis_do_not_unlock_editing(tmp_path):
    migrator = migrator_fixture(tmp_path, TaskStatus.INVESTIGATING, memory_phase="planning", active_hypothesis="maybe cache")
    state = migrator.migrate("task-a", [])
    assert state.agent_phase is AgentPhase.INVESTIGATING
    assert state.supported_hypothesis_ids == []


def test_migration_replay_is_idempotent(tmp_path):
    migrator = migrator_fixture(tmp_path, TaskStatus.TESTING)
    first = migrator.migrate("task-a", pytest_pair(exit_code=1))
    event_ids = [item.event_id for item in migrator.store.list_events("task-a")]
    second = migrator.migrate("task-a", pytest_pair(exit_code=1))
    assert second == first
    assert [item.event_id for item in migrator.store.list_events("task-a")] == event_ids
```

- [ ] **Step 2: Run migration tests and verify missing migrator**

Run: `pytest tests/investigation/test_migration.py -v`

Expected: fails on missing migration classes.

- [ ] **Step 3: Implement evidence-first migration version 1**

```python
class InvestigationMigrator:
    VERSION = 1

    def migrate(self, task_id: str, messages: Sequence[AnyMessage]) -> InvestigationState:
        existing = self.store.load(task_id)
        if existing is not None and existing.migration_version >= self.VERSION:
            return existing
        task = self.tasks.get(task_id)
        identified = ensure_message_ids(task_id, messages).messages
        observations = durable_observations(task_id, identified, self.compaction_store)
        phase = reconstruct_phase(task.status, observations)
        state = replay_observations(InvestigationState.new(task_id), observations)
        state = state.model_copy(update={
            "agent_phase": AgentPhase.CLARIFYING if task.status is TaskStatus.CLARIFYING else phase,
            "paused_agent_phase": phase if task.status is TaskStatus.CLARIFYING else None,
            "migration_version": self.VERSION,
        })
        return self.store.commit(0, migration_events(task_id, observations, state), state)
```

Order observations by original stable message ordinal. Treat approvals and incomplete execute Tool Calls as non-execution. A successful edit after the latest failure with no paired post-edit pytest remains EDITING. Import Working Memory hypotheses only as candidates unless all supported validation fields and current-task provenance are present.

- [ ] **Step 4: Run migration and identity tests**

Run: `pytest tests/investigation/test_migration.py tests/compaction/test_identity.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit legacy migration**

```bash
git add src/deepfix/investigation/migration.py tests/investigation/test_migration.py
git commit -m "feat: migrate legacy investigation state safely"
```

### Task 10: Service Lifecycle and Typed Pause Recovery

**Files:**
- Modify: `src/deepfix/models.py`
- Modify: `src/deepfix/service.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_service.py`
- Modify: `tests/research/test_workflow.py`
- Modify: `tests/compaction/test_long_context_workflow.py`

**Interfaces:**
- Consumes: InvestigationCoordinator lifecycle commands and InvestigationCoordinationError.
- Produces: persisted `TaskState.investigation_recovery`, safe PAUSED transition, resume lifecycle synchronization.

- [ ] **Step 1: Write failing persistence and Service recovery tests**

```python
def recovery_metadata(task_id: str) -> InvestigationRecoveryMetadata:
    return InvestigationRecoveryMetadata(
        task_id=task_id,
        error_code="investigation_stagnated",
        agent_phase=AgentPhase.DIAGNOSING,
        state_version=3,
        last_event_sequence=7,
        checkpoint_available=True,
        recovery_action="request_user_direction",
    )


class InvestigationErrorAgent:
    def invoke(self, value, config):
        task_id = config["configurable"]["thread_id"]
        raise InvestigationStagnationError(recovery_metadata(task_id))


class ForeignInvestigationErrorAgent:
    def invoke(self, value, config):
        raise InvestigationStagnationError(recovery_metadata("other-task"))


def service_coordinator(config: AppConfig) -> InvestigationCoordinator:
    database = config.database_path
    compaction = CompactionStore(database)
    return InvestigationCoordinator(
        store=InvestigationStore(database),
        tasks=TaskRepository(database),
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, ResearchEvidenceStore(database)),
    )


def test_task_round_trips_investigation_recovery():
    task = TaskState.create(Path.cwd(), "bug", ApprovalMode.MANUAL)
    task.investigation_recovery = recovery_metadata("task-a")
    restored = TaskState.from_dict(task.to_dict())
    assert restored.investigation_recovery == task.investigation_recovery


def test_investigation_error_pauses_and_persists_metadata(app_config):
    coordinator = service_coordinator(app_config)
    service, repository = make_service(app_config, InvestigationErrorAgent(), investigation=coordinator)
    task = service.start("repository scan loops")
    assert task.status is TaskStatus.PAUSED
    assert task.investigation_recovery.error_code == "investigation_stagnated"
    assert repository.get(task.task_id).investigation_recovery == task.investigation_recovery


def test_foreign_recovery_task_id_fails_closed(app_config):
    service, _ = make_service(app_config, ForeignInvestigationErrorAgent())
    task = service.start("bug")
    assert task.status is TaskStatus.FAILED
    assert "其他任务" in task.final_summary


def test_user_information_lifecycle_clears_recovery_after_success(app_config):
    coordinator = service_coordinator(app_config)
    service, _ = make_service(app_config, FakeAgent(outcome(), outcome(question="continue")), investigation=coordinator)
    task = service.start("bug")
    service.continue_task(task.task_id, "Python 3.12")
    assert coordinator.store.list_events(task.task_id)[-1].event_type == "user_information_received"


def test_needs_input_sets_business_status_and_agent_phase_to_clarifying(app_config):
    coordinator = service_coordinator(app_config)
    service, _ = make_service(
        app_config,
        FakeAgent(outcome(question="which Python version?")),
        investigation=coordinator,
    )
    task = service.start("version-sensitive bug")
    assert task.status is TaskStatus.CLARIFYING
    state = coordinator.state(task.task_id)
    assert state.agent_phase is AgentPhase.CLARIFYING
    assert state.paused_agent_phase is AgentPhase.INVESTIGATING
```

- [ ] **Step 2: Run model and Service tests and verify failures**

Run: `pytest tests/test_models.py tests/test_service.py -v`

Expected: fails because TaskState and Service lack investigation recovery integration.

- [ ] **Step 3: Persist recovery separately and catch typed errors**

```python
@dataclass
class TaskState:
    investigation_recovery: InvestigationRecoveryMetadata | None = None
```

```python
try:
    result = self.agent.invoke(value, graph_config)
except InvestigationCoordinationError as exc:
    if exc.recovery.task_id != task.task_id:
        task.final_summary = "Agent 返回了其他任务的调查恢复信息"
        task.transition_to(TaskStatus.FAILED)
        self._save(task)
        return task
    task.investigation_recovery = exc.recovery
    return self._pause(task, f"调查协调需要恢复：{exc.recovery.error_code}")
```

Inject `InvestigationCoordinator` into Service. Update the shared Service factories in `tests/test_service.py`, `tests/research/test_workflow.py`, and `tests/compaction/test_long_context_workflow.py` to construct it against the same database. After successfully persisting TaskStatus changes, call its public `record_paused`, `record_resumed`, `record_needs_input`, or `record_user_information` methods. `record_needs_input` stores the prior AgentPhase and sets CLARIFYING; user information restores that phase, except new information supplied after REVIEWING restarts INVESTIGATING. If lifecycle commit fails, preserve PAUSED and recovery metadata. Clear `investigation_recovery` only after a successful resumed Agent invocation, mirroring context recovery semantics.

- [ ] **Step 4: Run Service regression tests**

Run: `pytest tests/test_models.py tests/test_service.py tests/compaction/test_long_context_workflow.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Service recovery integration**

```bash
git add src/deepfix/models.py src/deepfix/service.py tests/test_models.py tests/test_service.py tests/research/test_workflow.py tests/compaction/test_long_context_workflow.py
git commit -m "feat: pause tasks on investigation recovery errors"
```

### Task 11: Agent and CLI Assembly with Middleware Contract Tests

**Files:**
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/investigation/__init__.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: shared InvestigationStore/Coordinator, two investigation tools, middleware, migration, capabilities.
- Produces: fully assembled DeepFix Agent and CLI Service using the same database-backed investigation authority.

- [ ] **Step 1: Write failing assembly and middleware-order contract tests**

```python
def capture_create_deep_agent(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"
    monkeypatch.setattr("deepfix.agent.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr("deepfix.agent.build_main_model", lambda config: object())
    monkeypatch.setattr("deepfix.agent.build_compaction_model", lambda config: object())
    return captured


class RecordingMiddleware(AgentMiddleware):
    def __init__(self, name: str, calls: list[str]) -> None:
        self.label = name
        self.calls = calls

    def wrap_model_call(self, request, handler):
        self.calls.append(f"{self.label}:before")
        response = handler(request)
        self.calls.append(f"{self.label}:after")
        return response


def invoke_minimal_agent(middleware: list[AgentMiddleware]) -> None:
    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        tools=[],
        middleware=middleware,
    )
    agent.invoke({"messages": [HumanMessage(content="run")]})


def test_agent_registers_investigation_tools_and_semantic_order(config, monkeypatch):
    captured = capture_create_deep_agent(monkeypatch)
    build_agent(config, InMemorySaver(), WorkingMemoryStore(config.database_path))
    names = [tool.name for tool in captured["tools"]]
    middleware = [type(item).__name__ for item in captured["middleware"]]
    assert {"record_hypothesis", "continue_investigation"} <= set(names)
    assert middleware[:7] == [
        "MessageIdentityMiddleware",
        "LegacyContextMigrationMiddleware",
        "InvestigationMigrationMiddleware",
        "InvestigationMiddleware",
        "PromptPolicyMiddleware",
        "ProtectedContextMiddleware",
        "DeepFixCompactionMiddleware",
    ]
    assert middleware[-1] == "LLMTraceMiddleware"


def test_model_middleware_list_is_outermost_first_for_installed_langchain():
    calls = []
    first = RecordingMiddleware("first", calls)
    second = RecordingMiddleware("second", calls)
    invoke_minimal_agent([first, second])
    assert calls == ["first:before", "second:before", "second:after", "first:after"]
```

- [ ] **Step 2: Run Agent/CLI tests and verify expected assembly failures**

Run: `pytest tests/test_agent.py tests/test_cli.py -v`

Expected: fails because investigation components are not assembled.

- [ ] **Step 3: Construct shared investigation runtime and preserve user trace changes**

```python
# src/deepfix/cli.py
investigation_store = InvestigationStore(config.database_path)
investigation = InvestigationCoordinator(
    store=investigation_store,
    tasks=repository,
    compaction_store=compaction_store,
    evidence_collector=EvidenceCollector(compaction_store, research_evidence_store),
)
agent = build_agent(
    config,
    checkpointer,
    working_memory_store,
    investigation=investigation,
    task_repository=repository,
    compaction_store=compaction_store,
    extensions=extensions,
    research_evidence_store=research_evidence_store,
    backend=artifact_backend,
)
service = BugfixService(
    agent,
    repository,
    ApprovalPolicy(config.approval_mode),
    config,
    working_memory_store,
    research_evidence_store,
    compaction_store,
    investigation,
)
```

In `build_agent`, add both tools to `core_tool_names`, pass explicit core/extension capabilities into InvestigationMiddleware, and insert migration/investigation before PromptPolicy. Do not remove the user's `extra_body={"thinking": {"type": "disabled"}}`, path prompt, or debug middleware. Keep LLMTrace last and optionalize it only in the later CLI/debug subproject.

- [ ] **Step 4: Run assembly, approval, and prompt suites**

Run: `pytest tests/test_agent.py tests/test_cli.py tests/test_approval.py tests/test_prompting.py tests/test_extensions.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Agent/CLI assembly**

```bash
git add src/deepfix/agent.py src/deepfix/cli.py src/deepfix/investigation/__init__.py tests/test_agent.py tests/test_cli.py
git commit -m "feat: assemble investigation reliability core"
```

### Task 12: Long-Loop End-to-End Regression and Final Verification

**Files:**
- Create: `tests/investigation/test_workflow.py`

**Interfaces:**
- Consumes: complete Agent/Service investigation integration.
- Produces: offline QuixBugs-style normal-flow, BFS-loop, GCD-repeat, permit-recovery, and fault-injection coverage.

- [ ] **Step 1: Write failing end-to-end workflow tests**

```python
@dataclass(frozen=True)
class ScriptStep:
    name: str
    call_id: str
    args: dict[str, object]
    result: ToolMessage | None = None
    hypothesis: RecordHypothesisInput | None = None
    continue_input: ContinueInvestigationInput | None = None


class OfflineRepairHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.investigation = coordinator_fixture(tmp_path)
        backend = FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        self.middleware = InvestigationMiddleware(
            self.investigation,
            ToolExecutionReceiptStore(backend),
            {
                "read_file": InvestigationCapability.READ,
                "grep": InvestigationCapability.SEARCH,
                "execute": InvestigationCapability.EXECUTE,
                "edit_file": InvestigationCapability.MODIFY,
                "record_hypothesis": InvestigationCapability.META,
                "continue_investigation": InvestigationCapability.META,
            },
        )
        self.tool_names: list[str] = []
        self.execution_counts: Counter[str] = Counter()

    @property
    def completed_read_calls(self) -> int:
        return sum(count for call_id, count in self.execution_counts.items() if call_id.startswith("read-"))

    def run(self, steps: list[ScriptStep]) -> InvestigationState:
        for step in steps:
            self.tool_names.append(step.name)
            if step.hypothesis is not None:
                self.investigation.record_hypothesis("task-a", step.hypothesis, source_id=step.call_id)
                continue
            if step.continue_input is not None:
                self.investigation.grant_investigation_permit("task-a", step.continue_input)
                continue
            request = tool_request(step.name, step.call_id, step.args)
            def handler(received, current=step):
                self.execution_counts[current.call_id] += 1
                assert current.result is not None
                return current.result
            self.middleware.wrap_tool_call(request, handler)
        return self.investigation.state("task-a")


def normal_flow() -> list[ScriptStep]:
    return [
        ScriptStep("execute", "test-fail", {"command": "python -m pytest -q"}, ToolMessage(
            id="msg-test-fail", content="1 failed", tool_call_id="test-fail", artifact={"exit_code": 1}
        )),
        ScriptStep("read_file", "read-test", {"file_path": "/tests/test_sign.py"}, ToolMessage(
            id="msg-read-test", content="assert apply_sign(-2, -1) == 2", tool_call_id="read-test"
        )),
        ScriptStep("read_file", "read-src", {"file_path": "/src/sign.py"}, ToolMessage(
            id="msg-read-src", content="return -value", tool_call_id="read-src"
        )),
        ScriptStep("record_hypothesis", "hyp-1", {}, hypothesis=supported_input([evidence_id_for("test-fail")])),
        ScriptStep("edit_file", "edit-1", {"file_path": "/src/sign.py"}, edit_result("edit-1")),
        ScriptStep("execute", "test-pass", {"command": "python -m pytest -q"}, ToolMessage(
            id="msg-test-pass", content="1 passed", tool_call_id="test-pass", artifact={"exit_code": 0}
        )),
    ]


def evidence_id_for(tool_call_id: str) -> str:
    return f"evidence-{tool_call_id}"


def exploratory_reads(count: int) -> list[ScriptStep]:
    return [
        ScriptStep("read_file", f"read-{index}", {"file_path": f"/unrelated/file_{index}.py"}, ToolMessage(
            id=f"msg-read-{index}", content=f"value_{index} = {index}", tool_call_id=f"read-{index}"
        ))
        for index in range(count)
    ]


def repeated_pytest_steps_with_one_valid_permit(hypothesis_id: str) -> list[ScriptStep]:
    failed = lambda call_id: ScriptStep(
        "execute",
        call_id,
        {"command": "python -m pytest -q"},
        ToolMessage(
            id=f"msg-{call_id}",
            content="same gcd failure",
            tool_call_id=call_id,
            artifact={"exit_code": 1},
        ),
    )
    permit = ContinueInvestigationInput(
        hypothesis_ids=[hypothesis_id],
        unresolved_question="does one bounded rerun reproduce the same negative-input failure?",
        expected_evidence="same deterministic pytest failure",
        tool_name="execute",
        target="python -m pytest -q",
        reason="one bounded reproduction for hyp-gcd",
    )
    return [
        failed("repeat-1"),
        failed("repeat-2"),
        failed("repeat-3"),
        failed("repeat-4"),
        ScriptStep("continue_investigation", "permit-1", {}, continue_input=permit),
        failed("repeat-5"),
        failed("repeat-6"),
    ]


def test_normal_bugfix_flow_reaches_reviewing_without_false_stagnation(tmp_path):
    harness = OfflineRepairHarness(tmp_path)
    seed_evidence(harness.investigation.compaction_store, "task-a", evidence_id_for("test-fail"))
    seed_checked_location(harness.investigation.store, "task-a", "src/sign.py", 1, 20)
    state = harness.run(normal_flow())
    assert state.agent_phase is AgentPhase.REVIEWING
    assert harness.tool_names == ["execute", "read_file", "read_file", "record_hypothesis", "edit_file", "execute"]


def test_bfs_repository_scan_is_stopped_before_context_growth(tmp_path):
    harness = OfflineRepairHarness(tmp_path)
    with pytest.raises(InvestigationStagnationError):
        harness.run(exploratory_reads(count=20))
    assert harness.completed_read_calls <= 5
    assert harness.investigation.state("task-a").stagnation_level == 2


def test_gcd_repeat_gets_one_permit_then_pauses(tmp_path):
    harness = OfflineRepairHarness(tmp_path)
    hypothesis = harness.investigation.record_hypothesis(
        "task-a",
        RecordHypothesisInput(
            statement="negative normalization repeats",
            evidence_ids=[],
            checked_locations=[],
            target_state="candidate",
            reason="candidate for bounded reproduction",
        ),
        source_id="hyp-gcd-call",
    )
    repeated = repeated_pytest_steps_with_one_valid_permit(hypothesis.hypothesis_id)
    with pytest.raises(InvestigationStagnationError):
        harness.run(repeated)
    assert sum(
        event.event_type == "investigation_permit_granted"
        for event in harness.investigation.store.list_events("task-a")
    ) == 1
    assert harness.execution_counts.total() <= 5


def test_store_failure_after_tool_execution_does_not_rerun_tool(tmp_path):
    middleware = middleware_fixture(tmp_path, phase="planning", fail_next_event_commit=True)
    executions = 0
    def handler(request):
        nonlocal executions
        executions += 1
        return edit_result("edit-1")
    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(edit_request("edit-1"), handler)
    middleware.wrap_tool_call(edit_request("edit-1"), handler)
    assert executions == 1
    event_ids = [item.event_id for item in middleware.coordinator.store.list_events("task-a")]
    assert len(event_ids) == len(set(event_ids))
```

- [ ] **Step 2: Run the end-to-end suite and verify initial failures**

Run: `pytest tests/investigation/test_workflow.py -v`

Expected: at least the long-loop or fault-injection scenarios fail until harness-facing edge handling is complete.

- [ ] **Step 3: Close only integration gaps exposed by the failing tests**

Implement the smallest changes in the owning investigation module. The acceptable changes are limited to: preserving the ToolMessage/checkpoint when commit fails, replaying the same deterministic event ID on resume, bounding state projection, and ensuring Service reports the typed pause reason. Do not add Artifact Retrieval, CLI streaming, or model-dependent semantic extraction in this task.

```python
try:
    committed = self.coordinator.record_tool_result(task_id, request.tool_call, result)
except InvestigationStateError as exc:
    exc.recovery = exc.recovery.model_copy(update={
        "tool_call_id": str(request.tool_call["id"]),
        "checkpoint_available": True,
        "recovery_action": "replay_event_commit_without_rerunning_tool",
    })
    raise
return result
```

- [ ] **Step 4: Run focused and complete offline verification**

Run: `pytest tests/investigation -v`

Expected: all investigation tests pass.

Run: `pytest`

Expected: the complete offline suite passes with zero failures.

Run: `ruff check src tests`

Expected: zero Ruff errors. If Ruff reports only the user's pre-existing uncommitted debug findings, do not silently edit them; report the exact files and request permission before changing user-owned debug code.

- [ ] **Step 5: Verify repository boundaries and commit the end-to-end coverage**

Run: `git status --short`

Expected: only intended investigation changes plus the user's preserved pre-existing debug/prompt changes are present; no QuixBugs fixture or target-project production code is modified.

```bash
git add tests/investigation/test_workflow.py src/deepfix/investigation src/deepfix/service.py src/deepfix/models.py
git commit -m "test: cover investigation reliability workflows"
```

Do not stage `src/deepfix/agent.py`, `src/deepfix/prompts.py`, `src/deepfix/debug.py`, or `docs/debug/` in this final commit unless a preceding task intentionally changed and already committed the exact reviewed hunks.

## Spec Coverage Map

| Design requirement | Implementation tasks |
|---|---|
| TaskStatus / AgentPhase / Working Memory authority split | Tasks 1, 4, 8, 10 |
| Event log, materialized state, stable IDs, task isolation | Tasks 1, 2 |
| Automatic checked files and deterministic test/file evidence | Tasks 3, 4 |
| Supported-hypothesis planning/edit gate | Tasks 4, 6, 7 |
| Phase transitions and true pytest execution boundary | Tasks 4, 7, 9 |
| Decision-relevant progress and verified dependency edges | Tasks 3, 4 |
| Repeat/cycle/no-progress/exploration stagnation | Task 5 |
| One-shot continue permit and level-2 typed pause | Tasks 5, 6, 10 |
| Protected projection deduplication and bounded prompt state | Tasks 7, 8 |
| Tool receipt replay after post-execution commit failure | Tasks 7, 12 |
| Legacy migration and Working Memory non-authority | Task 9 |
| Middleware order, Agent/CLI assembly, extension capability safety | Tasks 8, 11 |
| Normal flow, BFS/GCD loops, fault injection, full regressions | Task 12 |
| Artifact Retrieval and CLI Progress remain mandatory later deliverables | Global constraints and completion gate |

## Implementation Completion Gate

Before marking this plan complete, record fresh output for:

```bash
pytest tests/investigation -v
pytest
ruff check src tests
git status --short
```

Confirm that Investigation Reliability Core is complete while the overall reliability program remains open for the two required follow-up subprojects:

1. Diagnostic Artifact Retrieval
2. Progress Events + CLI Renderer
