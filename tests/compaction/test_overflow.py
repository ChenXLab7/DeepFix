
import asyncio

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator
from deepfix.compaction.errors import ContextRecoveryRequired
from deepfix.compaction.middleware import DeepFixCompactionMiddleware
from deepfix.compaction.models import CompactionDelta, DeterministicEvidenceBlock, TaskAnchor
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.compaction.store import CompactionStore
from deepfix.domain_repositories.history import HistoryRepository
from deepfix.memory import WorkingMemoryStore
from deepfix.protected_context import ProtectedContext

from .test_middleware import _request


def _messages():
    return [
        HumanMessage(id="m1", content="old question"),
        AIMessage(id="m2", content="old answer"),
        HumanMessage(id="m3", content="latest question"),
        AIMessage(
            id="m4",
            content="running tool",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "src/parser.py"},
                    "id": "call-incomplete",
                    "type": "tool_call",
                }
            ],
        ),
    ]


def _protected():
    return ProtectedContext(
        task_anchor=TaskAnchor(
            task_id="task-a",
            task_goal="fix parser",
            user_constraints=[],
            latest_user_message_id="m3",
            project_root="C:/project",
            project_python="C:/python.exe",
            task_status="investigating",
        ),
        working_memory=None,
        deterministic_evidence=DeterministicEvidenceBlock(),
        active_snapshot=None,
    )


class _ProtectedBuilder:
    def build(self, task_id, messages, event):
        return _protected()


class _Budget:
    def measure(self, request, blocks):
        return ContextBudgetReport(
            request_tokens=70,
            max_input_tokens=100,
            output_reserve_tokens=0,
            usage_ratio=0.70,
            zone="normal",
            target_ratio=None,
        )

    def should_emit_memory_hint(self, version, unit_id):
        return False


class _Delta:
    def generate(self, model, units):
        return _empty_delta()

    async def agenerate(self, model, units):
        return _empty_delta()


def _empty_delta():
    return CompactionDelta(
        user_constraint_candidates=[],
        confirmed_fact_candidates=[],
        hypothesis_transitions=[],
        experiments=[],
        conflict_candidates=[],
        unresolved_questions=[],
        next_steps=[],
    )


def _middleware(tmp_path, adapter=None):
    database = tmp_path / "deepfix.sqlite3"
    memory = WorkingMemoryStore(database)
    history = HistoryRepository(database)
    coordinator = CompactionCoordinator(
        adapter=adapter
        or DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        ),
        delta_generator=_Delta(),
        snapshot_builder=CompactionSnapshotBuilder(),
        snapshot_store=CompactionStore(database),
        memory_store=memory,
        history_repository=history,
    )
    return DeepFixCompactionMiddleware(_ProtectedBuilder(), _Budget(), coordinator), history


def test_overflow_retries_handler_exactly_once_with_minimal_safe_context(tmp_path):
    middleware, history = _middleware(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise ContextOverflowError("too large")
        return ModelResponse(result=[AIMessage(content="ok")])

    result = middleware.wrap_model_call(_request(_messages()), handler)

    assert result is not None
    assert len(calls) == 2
    retry_ids = {message.id for message in calls[1].messages}
    assert {"m3", "m4"} <= retry_ids
    assert "m1" not in retry_ids
    assert isinstance(calls[1].messages[0], SystemMessage)
    assert "<deepfix_task_anchor>" in calls[1].system_message.text
    assert history.context_telemetry("task-a").overflow_retry_count == 1


def test_second_overflow_raises_recovery_without_third_call(tmp_path):
    middleware, _ = _middleware(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        raise ContextOverflowError("still too large")

    with pytest.raises(ContextRecoveryRequired) as captured:
        middleware.wrap_model_call(_request(_messages()), handler)

    assert len(calls) == 2
    assert captured.value.recovery.stage == "overflow_retry"
    assert captured.value.recovery.prepared_snapshot_version == 1


class _FailingAdapter:
    def persist_history(self, *args, **kwargs):
        raise RuntimeError("artifact failed")


def test_overflow_preparation_failure_pauses_without_retrying_handler(tmp_path):
    middleware, _ = _middleware(tmp_path, adapter=_FailingAdapter())
    calls = []

    def handler(request):
        calls.append(request)
        raise ContextOverflowError("too large")

    with pytest.raises(ContextRecoveryRequired):
        middleware.wrap_model_call(_request(_messages()), handler)

    assert len(calls) == 1


def test_async_overflow_path_uses_one_async_retry(tmp_path):
    middleware, _ = _middleware(tmp_path)
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise ContextOverflowError("too large")
        return ModelResponse(result=[AIMessage(content="ok")])

    result = asyncio.run(
        middleware.awrap_model_call(_request(_messages()), handler)
    )

    assert result is not None
    assert len(calls) == 2
