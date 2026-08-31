import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import (
    CompactionCoordinator,
    CompactionRequest,
)
from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    ContextRecoveryRequired,
)
from deepfix.compaction.models import (
    CompactionDelta,
    CompactionFailureRecord,
    DeterministicEvidenceBlock,
    TaskAnchor,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import partition_work_units
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.domain_repositories.history import HistoryRepository
from deepfix.protected_context import ProtectedContext


def _delta():
    return CompactionDelta(
        user_constraint_candidates=[],
        confirmed_fact_candidates=[],
        hypothesis_transitions=[],
        experiments=[],
        conflict_candidates=[],
        unresolved_questions=[],
        next_steps=[],
    )


def _messages():
    return (
        HumanMessage(id="m1", content="old question"),
        AIMessage(id="m2", content="old answer"),
        HumanMessage(id="m3", content="latest question"),
        AIMessage(id="m4", content="working"),
    )


def _request(zone="normal_compaction", ratio=0.90):
    protected = ProtectedContext(
        task_anchor=TaskAnchor(
            task_id="task-a",
            task_goal="fix parser",
            user_constraints=[],
            latest_user_message_id="m3",
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
    return CompactionRequest(
        task_id="task-a",
        entrypoint="automatic",
        messages=_messages(),
        active_event=None,
        protected_context=protected,
        budget=ContextBudgetReport(
            request_tokens=int(ratio * 100),
            max_input_tokens=100,
            output_reserve_tokens=0,
            usage_ratio=ratio,
            zone=zone,
            target_ratio=0.75 if zone == "normal_compaction" else 0.65,
        ),
        model=object(),
    )


class _RecordingAdapter:
    def __init__(self, actual, calls):
        self.actual = actual
        self.calls = calls

    def persist_history(self, *args, **kwargs):
        self.calls.extend(["artifact_write", "artifact_verify"])
        return self.actual.persist_history(*args, **kwargs)


class _RecordingDelta:
    def __init__(self, calls):
        self.calls = calls

    def generate(self, model, units):
        self.calls.append("delta_generate")
        return _delta()


class _RecordingBuilder:
    def __init__(self, calls):
        self.calls = calls
        self.actual = CompactionSnapshotBuilder()

    def build(self, value):
        self.calls.append("snapshot_validate")
        return self.actual.build(value)


class _RecordingStore:
    def __init__(self, actual, calls):
        self.actual = actual
        self.calls = calls

    def save_prepared_snapshot(self, snapshot, input_hash):
        self.calls.append("snapshot_write")
        return self.actual.save_prepared_snapshot(snapshot, input_hash)

    def get_snapshot(self, task_id, version):
        self.calls.append("snapshot_verify")
        return self.actual.get_snapshot(task_id, version)

    def activate_from_event(self, task_id, event):
        self.calls.append("snapshot_activate")
        return self.actual.activate_from_event(task_id, event)

    def __getattr__(self, name):
        return getattr(self.actual, name)


def _coordinator(
    tmp_path,
    *,
    calls=None,
    adapter=None,
    store=None,
    delta_generator=None,
    model=None,
):
    database = tmp_path / "deepfix.sqlite3"
    actual_store = CompactionStore(database)
    history = HistoryRepository(database)
    calls = calls if calls is not None else []
    artifact = adapter or DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    )
    snapshot_store = store or actual_store
    return (
        CompactionCoordinator(
            adapter=(
                _RecordingAdapter(artifact, calls)
                if calls is not None and not adapter
                else artifact
            ),
            delta_generator=delta_generator or _RecordingDelta(calls),
            snapshot_builder=_RecordingBuilder(calls),
            snapshot_store=(
                _RecordingStore(snapshot_store, calls)
                if calls is not None and store is None
                else snapshot_store
            ),
            history_repository=history,
            model=model,
            partitioner=lambda messages, conflicts: _record_partition(calls, messages, conflicts),
        ),
        actual_store,
        history,
    )


def _record_partition(calls, messages, conflicts):
    calls.append("partition")
    return partition_work_units(messages, conflicts)


def test_prepare_orders_artifact_before_snapshot_and_never_mutates_messages(tmp_path):
    calls = []
    coordinator, _, _ = _coordinator(tmp_path, calls=calls)
    request = _request()
    original = list(request.messages)

    prepared = coordinator.prepare(request)

    assert calls == [
        "partition",
        "artifact_write",
        "artifact_verify",
        "delta_generate",
        "snapshot_validate",
        "snapshot_write",
        "snapshot_verify",
        "snapshot_activate",
    ]
    assert list(request.messages) == original
    assert prepared.snapshot.lifecycle == "active"
    assert {unit.unit_id for unit in prepared.retention.compressed_units}


def test_success_returns_event_and_identical_prepare_reuses_snapshot(tmp_path):
    coordinator, store, _ = _coordinator(tmp_path)
    request = _request()
    received = []

    result = coordinator.invoke_automatic(
        request,
        lambda messages: (
            received.append(messages) or ModelResponse(result=[AIMessage(content="ok")])
        ),
    )
    first = coordinator.prepare(request)
    second = coordinator.prepare(request)

    assert isinstance(result, ExtendedModelResponse)
    event = result.command.update["_deepfix_compaction_event"]
    assert event["active_snapshot_version"] == 1
    assert event["snapshot_message_id"] == first.snapshot_message.id
    assert event["retained_message_ids"] == sorted(first.retention.retained_message_ids)
    assert event["input_hash"] == first.input_hash
    assert first.snapshot.version == second.snapshot.version == 1
    assert store.get_snapshot("task-a", 1).lifecycle == "active"
    assert received[0][0].id == first.snapshot_message.id


def test_handler_failure_returns_no_event_but_activation_precedes_message_replacement(
    tmp_path,
):
    coordinator, store, _ = _coordinator(tmp_path)

    with pytest.raises(RuntimeError, match="model failed"):
        coordinator.invoke_automatic(
            _request(),
            lambda messages: (_ for _ in ()).throw(RuntimeError("model failed")),
        )

    assert store.get_snapshot("task-a", 1).lifecycle == "active"


def _failure(code="artifact_write_failed"):
    return CompactionFailureRecord(
        attempt_id="placeholder",
        task_id="task-a",
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="artifact_write",
        error_code=code,
        input_hash="f" * 64,
        original_messages_preserved=True,
        recorded_at="2026-08-22T00:00:00+00:00",
    )


class _FailingAdapter:
    def __init__(self):
        self.calls = 0

    def persist_history(self, *args, **kwargs):
        self.calls += 1
        raise ArtifactPersistenceError(_failure())


class _RuntimeFailingDelta:
    def generate(self, model, units):
        raise RuntimeError("delta failed")


class _ModelRecordingFailingDelta:
    def __init__(self):
        self.models = []

    def generate(self, model, units):
        self.models.append(model)
        raise RuntimeError("compaction model unavailable")


def test_delta_failure_never_falls_back_to_main_model(tmp_path):
    main_model = object()
    compaction_model = object()
    delta = _ModelRecordingFailingDelta()
    coordinator, store, _ = _coordinator(
        tmp_path,
        delta_generator=delta,
        model=compaction_model,
    )
    request = _request(ratio=0.85)
    request = CompactionRequest(
        task_id=request.task_id,
        entrypoint=request.entrypoint,
        messages=request.messages,
        active_event=request.active_event,
        protected_context=request.protected_context,
        budget=request.budget,
        model=main_model,
        tool_call_id=request.tool_call_id,
    )
    handler_calls = []

    coordinator.invoke_automatic(
        request,
        lambda messages: (
            handler_calls.append(messages) or ModelResponse(result=[AIMessage(content="continued")])
        ),
    )

    assert delta.models == [compaction_model]
    assert main_model not in delta.models
    assert len(handler_calls) == 1
    assert store.list_failures("task-a")[0].error_code == "delta_generation_failed"


class _RuntimeFailingBuilder:
    def build(self, value):
        raise RuntimeError("builder failed")


class _StageFailStore:
    def __init__(self, actual, stage):
        self.actual = actual
        self.stage = stage

    def save_prepared_snapshot(self, snapshot, input_hash):
        if self.stage == "snapshot_write":
            raise RuntimeError("write failed")
        return self.actual.save_prepared_snapshot(snapshot, input_hash)

    def get_snapshot(self, task_id, version):
        if self.stage == "snapshot_verify":
            raise RuntimeError("verify failed")
        return self.actual.get_snapshot(task_id, version)

    def activate_from_event(self, task_id, event):
        if self.stage == "snapshot_activate":
            raise RuntimeError("activation failed")
        if self.stage == "snapshot_activate_after_commit":
            self.actual.activate_from_event(task_id, event)
            raise RuntimeError("activation response lost")
        return self.actual.activate_from_event(task_id, event)

    def __getattr__(self, name):
        return getattr(self.actual, name)


def _stage_failure_coordinator(tmp_path, stage):
    database = tmp_path / "deepfix.sqlite3"
    store = CompactionStore(database)
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    )
    delta = _RecordingDelta([])
    builder = _RecordingBuilder([])
    snapshot_store = store
    if stage == "artifact_write":
        adapter = _FailingAdapter()
    elif stage == "delta_generation":
        delta = _RuntimeFailingDelta()
    elif stage == "snapshot_validate":
        builder = _RuntimeFailingBuilder()
    else:
        snapshot_store = _StageFailStore(store, stage)
    return (
        CompactionCoordinator(
            adapter=adapter,
            delta_generator=delta,
            snapshot_builder=builder,
            snapshot_store=snapshot_store,
        ),
        store,
    )


@pytest.mark.parametrize(
    "stage",
    [
        "artifact_write",
        "delta_generation",
        "snapshot_validate",
        "snapshot_write",
        "snapshot_verify",
        "snapshot_activate",
    ],
)
def test_each_normal_preparation_failure_calls_original_handler_once(tmp_path, stage):
    coordinator, store = _stage_failure_coordinator(tmp_path / stage, stage)
    calls = []

    result = coordinator.invoke_automatic(
        _request(ratio=0.85),
        lambda messages: calls.append(messages) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert isinstance(result, ModelResponse)
    assert calls == [_messages()]
    assert store.list_failures("task-a")[0].stage == stage


@pytest.mark.parametrize(
    "stage",
    [
        "artifact_write",
        "delta_generation",
        "snapshot_validate",
        "snapshot_write",
        "snapshot_verify",
        "snapshot_activate",
    ],
)
def test_each_emergency_preparation_failure_skips_unsafe_handler(tmp_path, stage):
    coordinator, _ = _stage_failure_coordinator(tmp_path / stage, stage)
    calls = []

    with pytest.raises(ContextRecoveryRequired) as captured:
        coordinator.invoke_automatic(
            _request(zone="emergency", ratio=0.91),
            lambda messages: calls.append(messages),
        )

    assert calls == []
    assert captured.value.recovery.stage == stage


def test_post_commit_activation_failure_does_not_mask_normal_passthrough(tmp_path):
    coordinator, store = _stage_failure_coordinator(
        tmp_path,
        "snapshot_activate_after_commit",
    )
    calls = []

    result = coordinator.invoke_automatic(
        _request(ratio=0.85),
        lambda messages: calls.append(messages) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert isinstance(result, ModelResponse)
    assert calls == [_messages()]
    assert store.get_snapshot("task-a", 1).lifecycle == "active"
    assert store.list_failures("task-a")[0].stage == "snapshot_activate"


def test_post_commit_activation_failure_does_not_mask_emergency_recovery(tmp_path):
    coordinator, store = _stage_failure_coordinator(
        tmp_path,
        "snapshot_activate_after_commit",
    )

    with pytest.raises(ContextRecoveryRequired) as captured:
        coordinator.invoke_automatic(
            _request(zone="emergency", ratio=0.91),
            lambda messages: pytest.fail("unsafe handler must not run"),
        )

    assert captured.value.recovery.stage == "snapshot_activate"
    assert store.get_snapshot("task-a", 1).lifecycle == "active"


def test_normal_failure_passthrough_once_and_same_input_skips_reprepare(tmp_path):
    adapter = _FailingAdapter()
    coordinator, store, history = _coordinator(tmp_path, adapter=adapter)
    handler_calls = []
    handler = lambda messages: (
        handler_calls.append(messages) or ModelResponse(result=[AIMessage(content="ok")])
    )

    first = coordinator.invoke_automatic(_request(ratio=0.85), handler)
    second = coordinator.invoke_automatic(_request(ratio=0.85), handler)

    assert isinstance(first, ModelResponse)
    assert isinstance(second, ModelResponse)
    assert adapter.calls == 1
    assert len(handler_calls) == 2
    assert handler_calls[0] == _messages()
    assert len(store.list_failures("task-a")) == 1
    assert history.context_telemetry("task-a").normal_zone_passthrough_count == 2


class _VerifyFailStore:
    def __init__(self, actual):
        self.actual = actual
        self.fail_verify = True
        self.abandoned = []

    def save_prepared_snapshot(self, snapshot, input_hash):
        return self.actual.save_prepared_snapshot(snapshot, input_hash)

    def get_snapshot(self, task_id, version):
        if self.fail_verify:
            self.fail_verify = False
            raise RuntimeError("verify failed")
        return self.actual.get_snapshot(task_id, version)

    def abandon_snapshot(self, task_id, version, reason):
        self.abandoned.append(version)
        return self.actual.abandon_snapshot(task_id, version, reason)

    def __getattr__(self, name):
        return getattr(self.actual, name)


class _CoverageCorruptStore:
    def __init__(self, actual):
        self.actual = actual

    def save_prepared_snapshot(self, snapshot, input_hash):
        return self.actual.save_prepared_snapshot(snapshot, input_hash)

    def get_snapshot(self, task_id, version):
        snapshot = self.actual.get_snapshot(task_id, version)
        return snapshot.model_copy(
            update={"coverage": snapshot.coverage.model_copy(update={"covered_message_ids": []})}
        )

    def __getattr__(self, name):
        return getattr(self.actual, name)


def test_invalid_history_coverage_preserves_original_messages(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    actual = CompactionStore(database)
    coordinator, _, _ = _coordinator(
        tmp_path,
        store=_CoverageCorruptStore(actual),
    )
    calls = []

    result = coordinator.invoke_automatic(
        _request(ratio=0.85),
        lambda messages: calls.append(messages) or ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert isinstance(result, ModelResponse)
    assert calls == [_messages()]
    assert actual.list_failures("task-a")[0].stage == "snapshot_verify"


def test_passthrough_abandons_snapshot_prepared_before_failure(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    actual = CompactionStore(database)
    failing = _VerifyFailStore(actual)
    coordinator, _, _ = _coordinator(tmp_path, store=failing)

    coordinator.invoke_automatic(
        _request(ratio=0.85),
        lambda messages: ModelResponse(result=[AIMessage(content="ok")]),
    )

    assert failing.abandoned == [1]
    assert actual.get_snapshot("task-a", 1).lifecycle == "abandoned"


def test_emergency_preparation_failure_requires_recovery_without_handler(tmp_path):
    adapter = _FailingAdapter()
    coordinator, _, _ = _coordinator(tmp_path, adapter=adapter)
    handler_calls = []

    with pytest.raises(ContextRecoveryRequired) as captured:
        coordinator.invoke_automatic(
            _request(zone="emergency", ratio=0.91),
            lambda messages: handler_calls.append(messages),
        )

    assert handler_calls == []
    assert captured.value.recovery.original_messages_preserved is True
    assert captured.value.recovery.usage_ratio == 0.91
