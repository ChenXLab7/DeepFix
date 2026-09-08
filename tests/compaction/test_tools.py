import asyncio
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetReport
from deepfix.compaction.coordinator import CompactionCoordinator
from deepfix.compaction.errors import (
    ArtifactPersistenceError,
    ContextRecoveryRequired,
)
from deepfix.compaction.identity import stable_generated_message_id
from deepfix.compaction.models import (
    CompactionDelta,
    CompactionFailureRecord,
    DeterministicEvidenceBlock,
    TaskAnchor,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder
from deepfix.compaction.tools import build_compact_conversation_tool
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.protected_context import ProtectedContext


def _messages():
    return [
        HumanMessage(id="m1", content="old question"),
        AIMessage(id="m2", content="old answer"),
        HumanMessage(id="m3", content="latest question"),
        AIMessage(id="m4", content="working"),
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
        deterministic_evidence=DeterministicEvidenceBlock(),
        confirmed_facts=(),
        hypotheses=(),
        unresolved_questions=(),
        execution_integrity=ExecutionIntegrity(receipt_count=0, approval_count=0),
        external_evidence=(),
        active_snapshot=None,
    )


class _ProtectedBuilder:
    def build(self, task_id, messages, event):
        assert task_id == "task-a"
        return _protected()


class _Budget:
    def __init__(self, zone, ratio):
        self.report = ContextBudgetReport(
            request_tokens=int(ratio * 100),
            max_input_tokens=100,
            output_reserve_tokens=0,
            usage_ratio=ratio,
            zone=zone,
            target_ratio=(
                0.75 if zone == "normal_compaction" else 0.65 if zone == "emergency" else None
            ),
        )

    def measure(self, request, blocks):
        assert len(blocks) == 4
        return self.report


class _Delta:
    def generate(self, model, units, messages=()):
        return _empty_delta()

    async def agenerate(self, model, units, messages=()):
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


def _runtime():
    return SimpleNamespace(
        state={"messages": _messages()},
        config={"configurable": {"thread_id": "task-a"}},
        execution_info=None,
        tool_call_id="compact-call-1",
        tools=[],
    )


def _coordinator(tmp_path, zone="normal_compaction", ratio=0.85, adapter=None):
    database = tmp_path / "deepfix.sqlite3"
    repositories = DomainRepositories.create(database)
    return CompactionCoordinator(
        adapter=adapter
        or DeepAgentsArtifactAdapter(
            FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
        ),
        delta_generator=_Delta(),
        snapshot_builder=CompactionSnapshotBuilder(),
        history_repository=repositories.history,
        evidence_repository=repositories.evidence,
        investigation_repository=repositories.investigation,
        budget_monitor=_Budget(zone, ratio),
        protected_builder=_ProtectedBuilder(),
        model=object(),
    )


def test_tool_schema_exposes_no_runtime_authority_fields(tmp_path):
    tool = build_compact_conversation_tool(_coordinator(tmp_path))

    assert tool.name == "compact_conversation"
    assert not {
        "task_id",
        "event",
        "path",
        "cutoff",
        "token",
    } & set(tool.args)
    assert tool.coroutine is not None


def test_manual_below_threshold_returns_stable_noop_without_event(tmp_path):
    coordinator = _coordinator(tmp_path, zone="normal", ratio=0.70)
    runtime = _runtime()

    command = coordinator.compact_manually(runtime)

    assert "_deepfix_compaction_event" not in command.update
    message = command.update["messages"][0]
    assert message.status == "success"
    assert message.id == stable_generated_message_id("task-a", "compact-call-1", "manual_noop")
    assert runtime.state["messages"] == _messages()


def test_manual_success_returns_event_session_and_stable_tool_message(tmp_path):
    coordinator = _coordinator(tmp_path)

    command = coordinator.compact_manually(_runtime())

    assert command.update["_deepfix_compaction_event"]["active_snapshot_version"] == 1
    attempt_id = command.update["_deepfix_compaction_session_id"]
    message = command.update["messages"][0]
    assert message.status == "success"
    assert message.id == stable_generated_message_id("task-a", attempt_id, "manual_success")


def test_async_manual_success_uses_async_artifact_and_delta_paths(tmp_path):
    command = asyncio.run(_coordinator(tmp_path).acompact_manually(_runtime()))

    assert command.update["_deepfix_compaction_event"]["active_snapshot_version"] == 1
    assert command.update["messages"][0].status == "success"


class _FailingAdapter:
    def persist_history(self, *args, **kwargs):
        raise ArtifactPersistenceError(
            CompactionFailureRecord(
                attempt_id="placeholder",
                task_id="task-a",
                entrypoint="manual_tool",
                budget_zone="normal_compaction",
                stage="artifact_write",
                error_code="artifact_write_failed",
                input_hash="f" * 64,
                original_messages_preserved=True,
                recorded_at="2026-08-22T00:00:00+00:00",
            )
        )


def test_manual_normal_failure_returns_bounded_error_and_preserves_messages(tmp_path):
    runtime = _runtime()
    coordinator = _coordinator(tmp_path, adapter=_FailingAdapter())

    command = coordinator.compact_manually(runtime)

    assert "_deepfix_compaction_event" not in command.update
    assert command.update["messages"][0].status == "error"
    assert "artifact_write_failed" in command.update["messages"][0].text
    assert runtime.state["messages"] == _messages()


def test_manual_emergency_failure_raises_recovery(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        zone="emergency",
        ratio=0.91,
        adapter=_FailingAdapter(),
    )

    with pytest.raises(ContextRecoveryRequired):
        coordinator.compact_manually(_runtime())


def test_manual_tool_never_calls_private_summary_methods(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("private summary helper called")

    for name in (
        "_create_summary",
        "_acreate_summary",
        "_determine_cutoff_index",
        "_offload_to_backend",
    ):
        monkeypatch.setattr(SummarizationMiddleware, name, forbidden)

    command = _coordinator(tmp_path).compact_manually(_runtime())

    assert command.update["messages"][0].status == "success"
