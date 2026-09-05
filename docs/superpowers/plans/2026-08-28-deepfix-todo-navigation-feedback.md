# DeepFix Todo Navigation Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add framework-native Todo navigation and a lightweight, evidence-aware reminder loop that helps the Agent advance after enough investigation without making Todo a fact, permission, or outcome authority.

**Architecture:** Reuse LangChain `TodoListMiddleware`, `PlanningState`, native Todo schema, and `write_todos`. Add a thin DeepFix middleware that validates the single-`in_progress` invariant, counts complete Tool Rounds from stable Graph Messages, reads a transient navigation feedback projection, and injects request-local reminders. Legacy Phase and Working Memory remain unchanged and shadow-only in this plan.

**Status:** Completed and merged into `codex/deepfix-single-agent`. Final offline verification on 2026-08-29: `450 passed`; Ruff passed. The later State Authority Migration Plans 2–4 remain unimplemented.

**Tech Stack:** Python 3.11+, DeepAgents 0.7.x, LangChain 1.3.x, LangGraph 1.2.x, Pydantic 2, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-27-deepfix-architecture-audit.md`

**Program:** `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md` Plan 1

## Global Constraints

- Use framework-native `TodoListMiddleware`, `Todo`, `PlanningState`, and `write_todos`; do not create a TodoStore or custom Todo payload schema.
- Todo is navigation only. It cannot write Evidence, alter Investigation progress, expose/hide tools, satisfy VerificationPolicy, or affect OutcomeAdjudicator authority.
- At most one Todo is `in_progress`; if any Todo is not completed, exactly one is `in_progress`.
- One AIMessage containing one or more parallel Tool Calls plus all paired ToolMessages is one Tool Round.
- A `write_todos`-only round is navigation maintenance and does not increment the action-round counter.
- Todo status/current-item/list-membership changes reset the navigation counter; prose-only rewriting does not.
- Three completed action Tool Rounds without a meaningful Todo update trigger one request-local reminder and reset the scheduling counter.
- A new deterministic milestone may trigger one reminder before the three-round threshold; the same milestone fingerprint is not repeated.
- Reminder content is dynamically added to the model request and never appended to Graph Messages, Conversation Artifact, Snapshot, TaskState, or a Domain Store.
- This plan does not delete or weaken AgentPhase, WorkingMemoryStore, Investigation stagnation, Approval, Receipt, Journal, VerificationPolicy, or compaction recovery.
- Preserve unrelated dirty-worktree changes and stage only files listed by each task.

---

### Task 1: Native Todo State Contract and Validation

**Files:**
- Create: `src/deepfix/navigation/__init__.py`
- Create: `src/deepfix/navigation/models.py`
- Create: `tests/navigation/test_models.py`

**Interfaces:**
- Consumes: `langchain.agents.middleware.todo.PlanningState`, `Todo`; `PrivateStateAttr`.
- Produces: `TodoNavigationState`, `TodoValidation`, `validate_todos(todos)`, `todo_progress_fingerprint(todos)`.

- [x] **Step 1: Write failing tests for native schema reuse and the single-current-item rule**

```python
from deepfix.navigation.models import (
    TodoNavigationState,
    todo_progress_fingerprint,
    validate_todos,
)


def test_navigation_state_reuses_native_todos():
    annotations = TodoNavigationState.__annotations__
    assert "_deepfix_todo_rounds_since_update" in annotations
    assert "_deepfix_last_completed_tool_round_id" in annotations
    assert "_deepfix_last_todo_progress_fingerprint" in annotations
    assert "_deepfix_last_navigation_hint_fingerprint" in annotations
    assert "_deepfix_pending_navigation_reminder" in annotations


def test_noncompleted_plan_requires_exactly_one_in_progress():
    invalid = validate_todos([
        {"content": "reproduce", "status": "pending"},
        {"content": "diagnose", "status": "pending"},
    ])
    assert invalid.valid is False
    assert invalid.error_code == "todo_requires_one_in_progress"


def test_prose_only_rewrite_does_not_change_progress_fingerprint():
    before = [{"content": "investigate parser", "status": "in_progress"}]
    after = [{"content": "carefully investigate the parser", "status": "in_progress"}]
    assert todo_progress_fingerprint(before) == todo_progress_fingerprint(after)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/navigation/test_models.py -q`

Expected: FAIL because `deepfix.navigation.models` does not exist.

- [x] **Step 3: Define private Graph metadata without creating a custom Todo schema**

```python
from typing import Annotated, NotRequired

from langchain.agents.middleware.todo import PlanningState, Todo
from langchain.agents.middleware.types import PrivateStateAttr
from pydantic import BaseModel


class TodoNavigationState(PlanningState):
    _deepfix_todo_rounds_since_update: NotRequired[Annotated[int, PrivateStateAttr]]
    _deepfix_last_completed_tool_round_id: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_last_todo_progress_fingerprint: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_last_navigation_hint_fingerprint: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]
    _deepfix_pending_navigation_reminder: NotRequired[
        Annotated[str | None, PrivateStateAttr]
    ]


class TodoValidation(BaseModel):
    valid: bool
    error_code: str | None = None
    message: str | None = None
```

Implement `validate_todos()` against native `content/status` dictionaries. Empty/all-completed lists allow zero `in_progress`; every other list requires exactly one. Implement `todo_progress_fingerprint()` from list length, ordered status values, and the `in_progress` index only, so wording changes cannot reset the counter.

- [x] **Step 4: Run focused tests and Ruff**

Run: `python -m pytest tests/navigation/test_models.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/navigation/models.py tests/navigation/test_models.py`

Expected: PASS.

- [x] **Step 5: Commit the state contract**

```powershell
git add src/deepfix/navigation/__init__.py src/deepfix/navigation/models.py tests/navigation/test_models.py
git commit -m "feat: define native todo navigation state"
```

---

### Task 2: Stable Complete Tool Round Detection

**Files:**
- Create: `src/deepfix/navigation/rounds.py`
- Create: `tests/navigation/test_rounds.py`

**Interfaces:**
- Consumes: stable IDs assigned by `MessageIdentityMiddleware`; Graph `AIMessage` and paired `ToolMessage` objects.
- Produces: `ToolRoundDelta(count: int, latest_round_id: str | None)` and `completed_tool_rounds_after(messages, cursor_round_id)`.

- [x] **Step 1: Write failing tests for sequential, incomplete, and parallel rounds**

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepfix.navigation.rounds import completed_tool_rounds_after


def test_parallel_tool_calls_are_one_complete_round():
    messages = [
        HumanMessage(id="u1", content="fix it"),
        AIMessage(
            id="a1",
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_file", "args": {"file_path": "a.py"}},
                {"id": "c2", "name": "read_file", "args": {"file_path": "b.py"}},
            ],
        ),
        ToolMessage(id="t1", content="a", tool_call_id="c1"),
        ToolMessage(id="t2", content="b", tool_call_id="c2"),
    ]
    delta = completed_tool_rounds_after(messages, None)
    assert delta.count == 1
    assert delta.latest_round_id == "a1"


def test_missing_parallel_result_is_not_complete():
    messages = [
        AIMessage(
            id="a1",
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_file", "args": {}},
                {"id": "c2", "name": "grep", "args": {}},
            ],
        ),
        ToolMessage(id="t1", content="a", tool_call_id="c1"),
    ]
    assert completed_tool_rounds_after(messages, None).count == 0


def test_write_todos_only_round_is_not_an_action_round():
    messages = [
        AIMessage(
            id="a1",
            content="",
            tool_calls=[{"id": "c1", "name": "write_todos", "args": {}}],
        ),
        ToolMessage(id="t1", content="updated", tool_call_id="c1"),
    ]
    assert completed_tool_rounds_after(messages, None).count == 0
```

Add cases proving the cursor prevents recounting and that a missing cursor after compaction establishes a new baseline instead of recounting retained history.

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/navigation/test_rounds.py -q`

Expected: FAIL because `deepfix.navigation.rounds` does not exist.

- [x] **Step 3: Implement deterministic round detection**

```python
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


@dataclass(frozen=True)
class ToolRoundDelta:
    count: int
    latest_round_id: str | None


def completed_tool_rounds_after(
    messages: Sequence[AnyMessage],
    cursor_round_id: str | None,
) -> ToolRoundDelta:
    result_ids = {
        str(message.tool_call_id)
        for message in messages
        if isinstance(message, ToolMessage) and str(message.tool_call_id)
    }
    complete: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        round_id = str(message.id or "").strip()
        call_ids = [str(call.get("id", "")).strip() for call in message.tool_calls]
        names = [str(call.get("name", "")).strip() for call in message.tool_calls]
        if (
            round_id
            and all(call_ids)
            and all(call_id in result_ids for call_id in call_ids)
            and any(name != "write_todos" for name in names)
        ):
            complete.append(round_id)
    if not complete:
        return ToolRoundDelta(0, cursor_round_id)
    latest = complete[-1]
    if cursor_round_id is None:
        return ToolRoundDelta(len(complete), latest)
    if cursor_round_id not in complete:
        return ToolRoundDelta(0, latest)
    cursor_index = complete.index(cursor_round_id)
    return ToolRoundDelta(len(complete[cursor_index + 1 :]), latest)
```

Scan AI messages in order, require a stable AI message ID and exactly one paired ToolMessage for every non-empty tool-call ID, and count the group once. Ignore groups whose only tool is `write_todos`. If a non-null cursor is absent from retained messages, return zero new rounds and set the latest complete round as the new cursor; this avoids a false reminder after compaction.

- [x] **Step 4: Run focused tests and Ruff**

Run: `python -m pytest tests/navigation/test_rounds.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/navigation/rounds.py tests/navigation/test_rounds.py`

Expected: PASS.

- [x] **Step 5: Commit Tool Round detection**

```powershell
git add src/deepfix/navigation/rounds.py tests/navigation/test_rounds.py
git commit -m "feat: count stable todo action rounds"
```

---

### Task 3: Read-Only Navigation Feedback Projection

**Files:**
- Create: `src/deepfix/navigation/feedback.py`
- Create: `tests/navigation/test_feedback.py`

**Interfaces:**
- Consumes: `InvestigationStore.load(task_id)`, `CompactionStore.list_evidence(task_id)`, `VerificationPolicyStore.load(task_id)`, `evaluate_required_oracles()`.
- Produces: `NavigationFeedback`, `NavigationFeedbackSource` protocol, `LegacyNavigationFeedbackSource.build(task_id)`.

- [x] **Step 1: Write failing tests for deterministic milestone projection**

```python
from deepfix.navigation.feedback import LegacyNavigationFeedbackSource


def test_supported_hypothesis_is_a_navigation_hint(stores, supported_state):
    stores.investigation.commit(0, [], supported_state)
    feedback = LegacyNavigationFeedbackSource(
        stores.investigation,
        stores.evidence,
        stores.verification,
    ).build("task-a")
    assert "supported-hypothesis:h1" in feedback.milestone_ids
    assert any("h1" in line for line in feedback.lines)


def test_successful_change_without_post_change_oracle_requests_verification(stores):
    stores.evidence.save_evidence("task-a", stores.successful_file_change)
    feedback = stores.feedback_source.build("task-a")
    assert "verification-pending" in feedback.milestone_ids


def test_same_authoritative_state_has_stable_feedback_fingerprint(stores):
    first = stores.feedback_source.build("task-a")
    second = stores.feedback_source.build("task-a")
    assert first.fingerprint == second.fingerprint
```

Also cover: closed evidence-gap IDs; all required post-change oracles satisfied; baseline user-specified test passing before any successful code change; failed/unavailable required oracle must not emit completion wording.

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/navigation/test_feedback.py -q`

Expected: FAIL because the feedback module does not exist.

- [x] **Step 3: Implement a transient typed projection**

```python
from typing import Protocol

from pydantic import BaseModel


class NavigationFeedback(BaseModel):
    milestone_ids: tuple[str, ...] = ()
    lines: tuple[str, ...] = ()
    fingerprint: str


class NavigationFeedbackSource(Protocol):
    def build(self, task_id: str) -> NavigationFeedback:
        raise NotImplementedError
```

`LegacyNavigationFeedbackSource` performs read-only queries and sorts/deduplicates milestone IDs before hashing them. It may report supported hypotheses, closed evidence gaps, successful changes awaiting verification, required-oracle satisfaction, and baseline non-reproduction. It must not save data, mutate InvestigationState, or turn model prose into a milestone. Plan 3 will replace this adapter with final Repository views without changing the middleware protocol.

- [x] **Step 4: Run focused tests and Ruff**

Run: `python -m pytest tests/navigation/test_feedback.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/navigation/feedback.py tests/navigation/test_feedback.py`

Expected: PASS.

- [x] **Step 5: Commit the feedback projection**

```powershell
git add src/deepfix/navigation/feedback.py tests/navigation/test_feedback.py
git commit -m "feat: project trusted navigation milestones"
```

---

### Task 4: Todo Navigation Middleware

**Files:**
- Create: `src/deepfix/navigation/middleware.py`
- Create: `tests/navigation/test_middleware.py`

**Interfaces:**
- Consumes: `TodoNavigationState`, `NavigationFeedbackSource`, `completed_tool_rounds_after()`, LangChain `ModelRequest` and `ToolCallRequest`.
- Produces: `TodoNavigationMiddleware(feedback_source, reminder_rounds=3)` with sync/async model and tool hooks.

- [x] **Step 1: Write failing tests for validation, cadence, and request-local injection**

```python
def test_invalid_parallel_in_progress_todos_return_error_tool_message(middleware):
    request = write_todos_request([
        {"content": "reproduce", "status": "in_progress"},
        {"content": "diagnose", "status": "in_progress"},
    ])
    result = middleware.wrap_tool_call(request, lambda request: pytest.fail("called"))
    assert result.status == "error"
    assert "exactly one in_progress" in str(result.content)


def test_three_complete_rounds_inject_reminder_without_writing_messages(middleware):
    state = navigation_state_with_three_rounds()
    update = middleware.before_model(state, runtime("task-a"))
    assert update["_deepfix_todo_rounds_since_update"] == 0
    assert "是否已有足够证据停止继续调查" in (
        update["_deepfix_pending_navigation_reminder"]
    )
    captured = capture_model_request(middleware, state | update)
    assert "<todo_navigation_reminder>" in captured.system_message.text
    assert captured.messages == state["messages"]


def test_same_milestone_is_injected_once(middleware):
    first = middleware.before_model(base_state(), runtime("task-a"))
    resumed = base_state() | first
    second = middleware.before_model(resumed, runtime("task-a"))
    assert first["_deepfix_pending_navigation_reminder"] is not None
    assert second["_deepfix_pending_navigation_reminder"] is None
```

Add tests proving: a meaningful Todo status change resets the counter; prose-only rewriting does not; reminder scheduling does not mutate `InvestigationState.no_progress_count`; no Todo causes a “create a plan before more tools” advisory; HTML/XML-sensitive milestone content is escaped; async hooks match sync behavior.

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/navigation/test_middleware.py -q`

Expected: FAIL because `TodoNavigationMiddleware` does not exist.

- [x] **Step 3: Implement state updates and reminder rendering**

```python
class TodoNavigationMiddleware(AgentMiddleware):
    state_schema = TodoNavigationState

    def __init__(
        self,
        feedback_source: NavigationFeedbackSource,
        *,
        reminder_rounds: int = 3,
    ) -> None:
        if reminder_rounds <= 0:
            raise ValueError("reminder_rounds must be positive")
        self.feedback_source = feedback_source
        self.reminder_rounds = reminder_rounds

    def before_model(
        self,
        state: TodoNavigationState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        task_id = _runtime_task_id(runtime)
        todos = list(state.get("todos", ()))
        current_todo_fingerprint = todo_progress_fingerprint(todos)
        previous_todo_fingerprint = state.get(
            "_deepfix_last_todo_progress_fingerprint"
        )
        todo_changed = (
            previous_todo_fingerprint is not None
            and previous_todo_fingerprint != current_todo_fingerprint
        )
        delta = completed_tool_rounds_after(
            list(state.get("messages", ())),
            state.get("_deepfix_last_completed_tool_round_id"),
        )
        rounds = 0 if todo_changed else (
            int(state.get("_deepfix_todo_rounds_since_update", 0)) + delta.count
        )
        feedback = self.feedback_source.build(task_id)
        previous_hint = state.get("_deepfix_last_navigation_hint_fingerprint")
        new_milestone = bool(feedback.milestone_ids) and (
            feedback.fingerprint != previous_hint
        )
        cadence_due = rounds >= self.reminder_rounds
        missing_plan = not todos and delta.count > 0
        should_remind = new_milestone or cadence_due or missing_plan
        reminder = (
            render_navigation_reminder(todos, feedback)
            if should_remind
            else None
        )
        return {
            "_deepfix_todo_rounds_since_update": 0 if should_remind else rounds,
            "_deepfix_last_completed_tool_round_id": delta.latest_round_id,
            "_deepfix_last_todo_progress_fingerprint": current_todo_fingerprint,
            "_deepfix_last_navigation_hint_fingerprint": (
                feedback.fingerprint if new_milestone else previous_hint
            ),
            "_deepfix_pending_navigation_reminder": reminder,
        }
```

`before_model()` must compare the stored Todo progress fingerprint, add only newly completed action rounds, request a feedback projection, and schedule either a cadence reminder or a one-shot new-milestone reminder. It writes only private navigation metadata. `wrap_model_call()` appends the pending reminder to the request's SystemMessage and never changes `request.messages`. `wrap_tool_call()` validates `write_todos` arguments before delegating; all other tools pass through unchanged. Implement async wrappers with identical decisions.

- [x] **Step 4: Run focused tests and Ruff**

Run: `python -m pytest tests/navigation/test_middleware.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/navigation/middleware.py tests/navigation/test_middleware.py`

Expected: PASS.

- [x] **Step 5: Commit the middleware**

```powershell
git add src/deepfix/navigation/middleware.py tests/navigation/test_middleware.py
git commit -m "feat: add evidence-aware todo reminders"
```

---

### Task 5: Production Agent and CLI Wiring

**Files:**
- Create: `src/deepfix/navigation/prompts.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/investigation/coordinator.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/investigation/test_coordinator.py`

**Interfaces:**
- Consumes: `TodoListMiddleware`, `TodoNavigationMiddleware`, `LegacyNavigationFeedbackSource`, existing Store instances.
- Produces: `build_agent(..., verification_policy_store: VerificationPolicyStore | None = None)` with native `write_todos` available in every legacy Phase/stagnation condition.

- [x] **Step 1: Extend agent-construction tests before changing production wiring**

```python
assert middleware_names[:9] == [
    "MessageIdentityMiddleware",
    "TodoListMiddleware",
    "TodoNavigationMiddleware",
    "LegacyContextMigrationMiddleware",
    "InvestigationMigrationMiddleware",
    "InvestigationMiddleware",
    "PromptPolicyMiddleware",
    "ProtectedContextMiddleware",
    "DeepFixCompactionMiddleware",
]
assert "write_todos" in capabilities
assert capabilities["write_todos"] is InvestigationCapability.META
```

Add assertions that the CLI passes the same `VerificationPolicyStore` instance to `build_agent()` and `BugfixService`, and that an extension cannot register another `write_todos` tool.

- [x] **Step 2: Add a failing coordinator test proving Todo remains visible during reevaluation**

```python
def test_write_todos_remains_available_during_stagnation(coordinator):
    state = coordinator.state("task-a").model_copy(update={"stagnation_level": 1})
    allowed = coordinator.allowed_tool_names(
        state,
        {
            "write_todos": InvestigationCapability.META,
            "read_file": InvestigationCapability.READ,
        },
    )
    assert "write_todos" in allowed
```

- [x] **Step 3: Run the construction tests and verify RED**

Run: `python -m pytest tests/test_agent.py tests/test_cli.py tests/investigation/test_coordinator.py -q`

Expected: FAIL because Todo middleware and capability wiring are absent.

- [x] **Step 4: Wire native Todo before legacy investigation/context middleware**

Define the DeepFix-specific native Todo prompt in `navigation/prompts.py`:

```python
DEEPFIX_TODO_SYSTEM_PROMPT = """## Bug-repair task navigation

Use the native `write_todos` tool to keep a short plan for this bug-repair task.
Create the plan before starting a multi-step investigation. Keep at most one item
`in_progress`, mark it `completed` as soon as its work is actually done, and move the
next item to `in_progress`. Re-check the plan when new evidence changes the strategy.

Todo is navigation, not proof. A Todo status cannot establish a root cause, prove that
a file changed, prove that a test passed, authorize a tool, or declare the task fixed.
Use tool results and DeepFix's trusted evidence for those conclusions.
"""
```

In `build_agent()`:

```python
policy_store = verification_policy_store or VerificationPolicyStore(
    config.database_path
)
navigation_feedback = LegacyNavigationFeedbackSource(
    investigation_store,
    compaction,
    policy_store,
)

middleware=[
    MessageIdentityMiddleware(),
    TodoListMiddleware(system_prompt=DEEPFIX_TODO_SYSTEM_PROMPT),
    TodoNavigationMiddleware(navigation_feedback, reminder_rounds=3),
    # existing migration, investigation, context, compaction and debug middleware
]
```

Add `write_todos` to `core_tool_names` and classify it as `InvestigationCapability.META`. Add it to the legacy reevaluation allow-list so old Phase remains shadow state rather than blocking the new navigation tool. Update CLI wiring to reuse its existing policy-store instance.

- [x] **Step 5: Run focused tests and Ruff**

Run: `python -m pytest tests/test_agent.py tests/test_cli.py tests/investigation/test_coordinator.py -q`

Expected: PASS.

Run: `python -m ruff check src/deepfix/navigation/prompts.py src/deepfix/agent.py src/deepfix/cli.py src/deepfix/investigation/coordinator.py tests/test_agent.py tests/test_cli.py tests/investigation/test_coordinator.py`

Expected: PASS.

- [x] **Step 6: Commit agent wiring**

```powershell
git add src/deepfix/navigation/prompts.py src/deepfix/agent.py src/deepfix/cli.py src/deepfix/investigation/coordinator.py tests/test_agent.py tests/test_cli.py tests/investigation/test_coordinator.py
git commit -m "feat: enable native todo navigation"
```

---

### Task 6: Checkpoint, Compaction, and End-to-End Offline Regression

**Files:**
- Create: `tests/navigation/test_workflow.py`
- Modify: `tests/compaction/test_long_context_workflow.py`
- Modify: `tests/test_service.py`

**Interfaces:**
- Consumes: fully wired agent, `InMemorySaver`/SQLite Checkpointer, stable Message IDs, existing compaction coordinator.
- Produces: regression coverage proving navigation metadata survives restoration without entering protected facts, snapshots, reports, or task progress.

- [x] **Step 1: Write a checkpoint-resume workflow test**

Use a deterministic scripted chat model to emit: initial `write_todos`; one parallel read round; a second investigation round; Checkpoint resume; a third round; and then capture the next SystemMessage. Assert:

```python
assert result["todos"] == [
    {"content": "reproduce failure", "status": "completed"},
    {"content": "identify root cause", "status": "in_progress"},
    {"content": "apply minimal fix", "status": "pending"},
    {"content": "run required verification", "status": "pending"},
]
assert "<todo_navigation_reminder>" in captured_system_prompt
assert captured_system_prompt.count("<todo_navigation_reminder>") == 1
```

- [x] **Step 2: Write compaction isolation assertions**

After forcing compaction, assert the activated Snapshot JSON and Conversation Artifact contain no private navigation field names, while restored Graph State retains the Todo list and counter metadata through the Checkpointer.

```python
for forbidden in (
    "_deepfix_todo_rounds_since_update",
    "_deepfix_last_todo_progress_fingerprint",
    "_deepfix_pending_navigation_reminder",
):
    assert forbidden not in snapshot.model_dump_json()
    assert forbidden not in conversation_artifact_text
```

- [x] **Step 3: Add service/report isolation assertions**

Run an offline task through the report path and assert Todo navigation metadata is absent from `TaskState.to_dict()`, deterministic Evidence, ContextMetrics, and the rendered report. Todo itself remains available only from Graph State/Checkpoint.

- [x] **Step 4: Run the focused workflow tests**

Run: `python -m pytest tests/navigation tests/compaction/test_long_context_workflow.py tests/test_service.py -q`

Expected: PASS.

- [x] **Step 5: Run the Plan 1 core regression gate**

Run: `python -m pytest tests/test_agent.py tests/test_cli.py tests/test_service.py tests/test_verification.py tests/navigation tests/investigation tests/compaction -q`

Expected: PASS with no online tests executed.

Run: `python -m ruff check src/deepfix tests/navigation tests/test_agent.py tests/test_cli.py tests/test_service.py tests/investigation tests/compaction`

Expected: PASS.

- [x] **Step 6: Commit workflow coverage**

```powershell
git add tests/navigation/test_workflow.py tests/compaction/test_long_context_workflow.py tests/test_service.py
git commit -m "test: verify todo navigation recovery"
```

## Plan 1 Review Checkpoint

Stop after Task 6. Report:

- focused and core test command results;
- middleware order and native `write_todos` availability;
- parallel Tool Round counting evidence;
- checkpoint/compaction isolation evidence;
- reminder examples for cadence, supported hypothesis, pending verification, and required-oracle completion;
- confirmation that Phase, WorkingMemoryStore, trusted execution, and OutcomeAdjudicator behavior were not removed or weakened.

Do not begin `2026-08-28-deepfix-task-persistence-foundation.md` until this checkpoint is reviewed and explicitly approved.
