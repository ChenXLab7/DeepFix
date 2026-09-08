from types import SimpleNamespace

from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator, CompactionRequest
from deepfix.compaction.middleware import (
    DeepFixCompactionMiddleware,
    DeepFixCompactionState,
    MessageIdentityMiddleware,
)
from deepfix.compaction.models import (
    CompactionDelta,
    DeterministicEvidenceBlock,
    TaskAnchor,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.domain_repositories.history import HistoryRepository
from deepfix.protected_context import ProtectedContext


def _runtime(task_id="task-a"):
    return Runtime(
        execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="model-1",
            thread_id=task_id,
        )
    )


def _request(messages, task_id="task-a"):
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"], profile={"max_input_tokens": 100}),
        messages=list(messages),
        system_message=SystemMessage(content="base-system"),
        tools=[],
        state={"messages": list(messages)},
        runtime=_runtime(task_id),
    )


def _protected(latest="m-latest"):
    return ProtectedContext(
        task_anchor=TaskAnchor(
            task_id="task-a",
            task_goal="fix parser",
            user_constraints=[],
            latest_user_message_id=latest,
            project_root="C:/project",
            project_python="C:/python.exe",
            task_status="investigating",
        ),
        deterministic_evidence=DeterministicEvidenceBlock(),
        confirmed_facts=(),
        hypotheses=(),
        unresolved_questions=(),
        execution_integrity=ExecutionIntegrity(receipt_count=0, approval_count=0),
        external_evidence=(),
        active_snapshot=None,
    )


class _ProtectedBuilder:
    def __init__(self, latest="m-latest"):
        self.latest = latest

    def build(self, task_id, messages, event):
        return _protected(self.latest)


class _Budget:
    def __init__(self, reports):
        self.reports = list(reports)
        self.hints = set()

    def measure(self, request, protected_blocks):
        return self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]

    def should_emit_memory_hint(self, version, unit_id):
        key = (version, unit_id)
        if key in self.hints:
            return False
        self.hints.add(key)
        return True


def _report(zone, ratio, target=None):
    return ContextBudgetReport(
        request_tokens=int(ratio * 100),
        max_input_tokens=100,
        output_reserve_tokens=0,
        usage_ratio=ratio,
        zone=zone,
        target_ratio=target,
    )


class _Coordinator:
    def __init__(self, database):
        self.history_repository = HistoryRepository(database)
        self.snapshot_store = SimpleNamespace()
        self.requests = []

    def invoke_automatic(self, request, handler):
        self.requests.append(request)
        return handler(request.messages)


def _long_tool_history(count=12, size=12000):
    messages = [HumanMessage(id="start", content="fix; never change tests")]
    for i in range(count):
        messages.extend([AIMessage(id=f"ai-{i}", content="", tool_calls=[{
            "id": f"call-{i}", "name": "read_file", "args": {"file_path": "a.py"},
            "type": "tool_call"}]), ToolMessage(id=f"tool-{i}", tool_call_id=f"call-{i}",
                content=f"unique-{i}:" + "x" * size, artifact={"exit_code": 0})])
    messages.append(HumanMessage(id="latest", content="new direction and constraints"))
    return messages


def test_snip_is_view_only_restorable_and_skips_semantic_model(tmp_path):
    from deepfix.compaction.budget import ContextBudgetMonitor
    coordinator = _Coordinator(tmp_path / "state.db")
    coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder("latest"), ContextBudgetMonitor(output_reserve_tokens=0), coordinator)
    messages = _long_tool_history(size=20000)
    original = [m.model_dump() for m in messages]
    request = _request(messages).override(model=FakeListChatModel(responses=["ok"], profile={"max_input_tokens": 75000}))
    seen = []
    middleware.wrap_model_call(request, lambda r: seen.append(r) or ModelResponse(result=[AIMessage(content="ok")]))
    assert not coordinator.requests
    assert [m.model_dump() for m in request.state["messages"]] == original
    assert len(str(seen[0].messages[2].content)) < 4000
    assert seen[0].messages[-2].content == messages[-2].content
    assert seen[0].messages[-1].content == messages[-1].content
    assert [m.id for m in seen[0].messages] == [m.id for m in messages]
    body = coordinator.adapter.read_verified("/.deepfix-artifacts/conversation_history/task-a.md")
    assert "unique-0:" in body
    # A fresh middleware after Resume reconstructs the same view without checkpoint edits.
    resumed = DeepFixCompactionMiddleware(_ProtectedBuilder("latest"), ContextBudgetMonitor(output_reserve_tokens=0), coordinator)
    resumed.wrap_model_call(request, lambda r: seen.append(r) or ModelResponse(result=[AIMessage(content="ok")]))
    assert [m.model_dump() for m in seen[0].messages] == [m.model_dump() for m in seen[1].messages]
    assert coordinator.adapter.read_verified("/.deepfix-artifacts/conversation_history/task-a.md") == body


def test_message_count_snip_keeps_whole_pairs_and_users(tmp_path):
    from deepfix.compaction.budget import ContextBudgetMonitor
    coordinator = _Coordinator(tmp_path / "state.db")
    coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder("latest"), ContextBudgetMonitor(output_reserve_tokens=0), coordinator)
    messages = _long_tool_history(count=110, size=10)
    request = _request(messages).override(model=FakeListChatModel(responses=["ok"], profile={"max_input_tokens": 1000000}))
    seen = []
    middleware.wrap_model_call(request, lambda r: seen.extend(r.messages) or ModelResponse(result=[AIMessage(content="ok")]))
    assert len(seen) < 100
    assert len(request.state["messages"]) == 222
    assert {m.id for m in seen if isinstance(m, HumanMessage)} == {"start", "latest"}
    calls = {c["id"] for m in seen if isinstance(m, AIMessage) for c in m.tool_calls}
    assert calls == {m.tool_call_id for m in seen if isinstance(m, ToolMessage)}
    assert not coordinator.requests


def test_snip_view_survives_sqlite_checkpoint_reopen(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    from deepfix.compaction.budget import ContextBudgetMonitor

    messages = _long_tool_history(count=110, size=10)
    seen = []
    def graph_for(saver):
        coordinator = _Coordinator(tmp_path / "domain.db")
        coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
        middleware = DeepFixCompactionMiddleware(_ProtectedBuilder("latest"), ContextBudgetMonitor(output_reserve_tokens=0), coordinator)
        def node(state):
            request = _request(state["messages"]).override(model=FakeListChatModel(responses=["ok"], profile={"max_input_tokens": 1000000}))
            middleware.wrap_model_call(request, lambda r: seen.append(r.messages) or ModelResponse(result=[AIMessage(content="ok")]))
            return {}
        builder = StateGraph(MessagesState)
        builder.add_node("model", node)
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        return builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "task-a"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
        graph_for(saver).invoke({"messages": messages}, config)
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
        graph = graph_for(saver)
        assert [m.model_dump() for m in graph.get_state(config).values["messages"]] == [m.model_dump() for m in messages]
        graph.invoke({"messages": [HumanMessage(id="followup", content="additional user constraint")]}, config)
        assert len(graph.get_state(config).values["messages"]) == 223
    assert len(seen[0]) < 100 and len(seen[1]) < 100
    assert seen[1][-1].content == "additional user constraint"


def test_snip_archive_failure_keeps_original_view(tmp_path):
    from deepagents.backends.protocol import WriteResult

    from deepfix.compaction.budget import ContextBudgetMonitor
    coordinator = _Coordinator(tmp_path / "state.db")
    backend = FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    backend.write = lambda *args: WriteResult(error="storage unavailable")
    coordinator.adapter = DeepAgentsArtifactAdapter(backend)
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder("latest"), ContextBudgetMonitor(output_reserve_tokens=0), coordinator)
    messages = _long_tool_history(size=20000)
    request = _request(messages).override(model=FakeListChatModel(responses=["ok"], profile={"max_input_tokens": 75000}))
    seen = []
    middleware.wrap_model_call(request, lambda r: seen.extend(r.messages) or ModelResponse(result=[AIMessage(content="ok")]))
    assert [m.model_dump() for m in seen] == [m.model_dump() for m in messages]


def test_identity_middleware_replaces_messages_once_without_content_change():
    middleware = MessageIdentityMiddleware()
    messages = [HumanMessage(content="question"), AIMessage(content="answer")]
    state = {"messages": messages}

    update = middleware.before_model(state, _runtime())

    assert isinstance(update["messages"][0], RemoveMessage)
    assert update["messages"][0].id == REMOVE_ALL_MESSAGES
    normalized = update["messages"][1:]
    assert [item.content for item in normalized] == ["question", "answer"]
    assert all(item.id for item in normalized)
    assert middleware.before_model({"messages": normalized}, _runtime()) is None
    assert MessageIdentityMiddleware.state_schema is DeepFixCompactionState


def test_normal_observe_and_compaction_threshold_paths(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    reports = [
        _report("normal", 0.75),
        _report("observe", 0.80),
        _report("observe", 0.80),
        _report("normal_compaction", 0.85, 0.75),
        _report("emergency", 0.91, 0.65),
    ]
    budget = _Budget(reports)
    coordinator = _Coordinator(database)
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder("m-latest"), budget, coordinator)
    messages = [HumanMessage(id="m-latest", content="question")]
    received = []

    for _ in reports:
        middleware.wrap_model_call(
            _request(messages),
            lambda request: (
                received.append(request) or ModelResponse(result=[AIMessage(content="ok")])
            ),
        )

    assert "working memory" not in received[0].system_message.text.lower()
    assert "save_progress" not in received[1].system_message.text
    assert "save_progress" not in received[2].system_message.text
    assert all(
        request.system_message.text.count("<deepfix_protected_context>") == 1
        for request in received
    )
    assert [item.budget.target_ratio for item in coordinator.requests] == [0.75, 0.65]
    assert all(request.messages == tuple(messages) for request in coordinator.requests)


def test_protected_blocks_are_request_local_and_not_added_to_messages(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    middleware = DeepFixCompactionMiddleware(
        _ProtectedBuilder("m-latest"),
        _Budget([_report("normal", 0.70)]),
        _Coordinator(database),
    )
    messages = [HumanMessage(id="m-latest", content="question")]
    request = _request(messages)
    captured = []

    middleware.wrap_model_call(
        request,
        lambda updated: captured.append(updated) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert captured[0].system_message.text.count("<deepfix_task_anchor>") == 1
    assert captured[0].messages == messages
    assert request.state["messages"] == messages


class _EmptyDelta:
    def generate(self, model, units, messages=()):
        return CompactionDelta(
            user_constraint_candidates=[],
            confirmed_fact_candidates=[],
            hypothesis_transitions=[],
            experiments=[],
            conflict_candidates=[],
            unresolved_questions=[],
            next_steps=[],
        )


def test_committed_event_reconstructs_effective_view_without_deleting_checkpoint(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    repositories = DomainRepositories.create(database)
    coordinator = CompactionCoordinator(
        adapter=DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        ),
        delta_generator=_EmptyDelta(),
        snapshot_builder=CompactionSnapshotBuilder(),
        history_repository=repositories.history,
        evidence_repository=repositories.evidence,
        investigation_repository=repositories.investigation,
    )
    messages = [
        AIMessage(id="m1", content="old tool analysis " * 100),
        AIMessage(id="m2", content="old answer"),
        HumanMessage(id="m3", content="latest question"),
        AIMessage(id="m4", content="working"),
    ]
    report = _report("normal_compaction", 0.85, 0.75)
    prepared = coordinator.prepare(
        CompactionRequest(
            task_id="task-a",
            entrypoint="automatic",
            messages=tuple(messages),
            active_event=None,
            protected_context=_protected("m3"),
            budget=report,
            model=object(),
        )
    )
    middleware = DeepFixCompactionMiddleware(
        _ProtectedBuilder("m3"),
        _Budget([_report("normal", 0.70)]),
        coordinator,
    )
    request = _request(messages)
    request.state["_deepfix_compaction_event"] = prepared.event.model_dump(mode="json")
    captured = []

    middleware.wrap_model_call(
        request,
        lambda updated: captured.append(updated) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    effective_ids = [message.id for message in captured[0].messages]
    assert effective_ids[0] == prepared.snapshot_message.id
    assert "m1" not in effective_ids
    assert {"m3", "m4"} <= set(effective_ids)
    assert request.state["messages"] == messages
    assert repositories.history.get("task-a", 1).lifecycle == "active"
    assert coordinator.history_repository.context_telemetry("task-a").active_snapshot_version == 1


def test_micro_uses_character_budget_and_stops_at_target(tmp_path):
    import json
    coordinator = _Coordinator(tmp_path / "state.db")
    coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder(), None, coordinator)
    messages = _long_tool_history(count=12, size=4500)
    original = [m.model_dump() for m in messages]
    view = middleware._lightweight_view("task-a", messages, _report("normal", 0.01))
    assert view is not None
    assert len(json.dumps([m.model_dump(exclude={"artifact", "response_metadata"}) for m in view], ensure_ascii=False, default=str)) <= 40000
    results = [m for m in view if isinstance(m, ToolMessage)]
    assert len(results[0].content) < 1500
    assert results[-4:] == [m for m in messages if isinstance(m, ToolMessage)][-4:]
    assert any(len(m.content) > 2000 for m in results[:-4])
    assert [m.model_dump() for m in messages] == original


def test_snip_starts_above_fifty_messages_before_micro(tmp_path):
    coordinator = _Coordinator(tmp_path / "state.db")
    coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder(), None, coordinator)
    messages = _long_tool_history(count=25, size=1100)
    view = middleware._lightweight_view("task-a", messages, _report("normal", 0.01))
    assert view is not None
    assert 40 <= len(view) < 50
    assert all(m.content == next(old.content for old in messages if old.id == m.id)
               for m in view if isinstance(m, ToolMessage))
    assert middleware._lightweight_view("task-a", _long_tool_history(count=24, size=10), _report("normal", 0.01)) is None


def test_micro_keeps_recent_rounds_after_assistant_explanation(tmp_path):
    coordinator = _Coordinator(tmp_path / "state.db")
    coordinator.adapter = DeepAgentsArtifactAdapter(FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True))
    middleware = DeepFixCompactionMiddleware(_ProtectedBuilder(), None, coordinator)
    messages = _long_tool_history(count=12, size=4500)
    messages.insert(-1, AIMessage(id="explanation", content="I have read the results"))
    view = middleware._lightweight_view("task-a", messages, _report("normal", 0.01))
    assert view is not None
    assert [m for m in view if isinstance(m, ToolMessage)][-3:] == [m for m in messages if isinstance(m, ToolMessage)][-3:]
