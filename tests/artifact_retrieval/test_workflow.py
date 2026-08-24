from __future__ import annotations

import hashlib
import re

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from artifact_retrieval.helpers import snapshot
from deepfix.artifact_retrieval.collector import ArtifactReferenceCollector
from deepfix.artifact_retrieval.service import DiagnosticArtifactService
from deepfix.artifact_retrieval.tools import (
    build_read_diagnostic_artifact_tool,
    build_search_diagnostic_artifacts_tool,
)
from deepfix.backend import build_backend
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.models import DeepFixCompactionEvent
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode, load_config
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.store import InvestigationStore
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository
from deepfix.research.store import ResearchEvidenceStore


def _runtime(task_id: str, call_id: str) -> ToolRuntime:
    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": task_id}},
        stream_writer=lambda value: None,
        tool_call_id=call_id,
        store=None,
    )


def _tool_request(task_id: str, call_id: str) -> ToolCallRequest:
    def run() -> str:
        return "unused"

    tool = StructuredTool.from_function(
        run,
        name="execute",
        description="test producer",
    )
    return ToolCallRequest(
        tool_call={
            "name": "execute",
            "id": call_id,
            "args": {"command": "python -m pytest -q"},
            "type": "tool_call",
        },
        tool=tool,
        state={"messages": []},
        runtime=_runtime(task_id, call_id),
    )


def _offload(
    backend,
    task_id: str,
    call_id: str,
    content: str,
) -> ToolMessage:
    middleware = FilesystemMiddleware(
        backend=backend,
        tool_token_limit_before_evict=20,
        tools=["read_file"],
    )
    result = middleware.wrap_tool_call(
        _tool_request(task_id, call_id),
        lambda _: ToolMessage(
            id=f"tool-{call_id}",
            content=content,
            name="execute",
            tool_call_id=call_id,
        ),
    )
    assert isinstance(result, ToolMessage)
    return result


def _invoke(tool, task_id: str, messages, call_id: str, args) -> ToolMessage:
    call = {
        "name": tool.name,
        "id": call_id,
        "args": args,
        "type": "tool_call",
    }
    result = ToolNode([tool]).invoke(
        {
            "messages": [
                *messages,
                AIMessage(id=f"ai-{call_id}", content="", tool_calls=[call]),
            ]
        },
        {"configurable": {"thread_id": task_id}},
        runtime=Runtime(),
    )
    return result["messages"][0]


def _environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(project, ApprovalMode.MANUAL)
    backend = build_backend(config)
    tasks = TaskRepository(config.database_path)
    for task_id in ("task-a", "task-b"):
        tasks.save(
            TaskState(
                task_id=task_id,
                project_root=str(config.project_root),
                project_python=str(config.project_python),
                user_problem="bug",
                approval_mode="manual",
                status=TaskStatus.INVESTIGATING,
            )
        )
    compaction = CompactionStore(config.database_path)
    research = ResearchEvidenceStore(config.database_path)
    coordinator = InvestigationCoordinator(
        store=InvestigationStore(config.database_path),
        tasks=tasks,
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(compaction, research),
    )
    service = DiagnosticArtifactService(
        tasks,
        ArtifactReferenceCollector(compaction, backend),
        backend,
    )
    return (
        backend,
        compaction,
        build_search_diagnostic_artifacts_tool(service, coordinator),
        build_read_diagnostic_artifact_tool(service, coordinator),
    )


def _activate_history(compaction, task_id: str, reference) -> None:
    prepared = snapshot(
        task_id=task_id,
        lifecycle="prepared",
        references=[reference],
    )
    prepared = compaction.save_prepared_snapshot(prepared, input_hash="d" * 64)
    compaction.activate_from_event(
        task_id,
        DeepFixCompactionEvent(
            event_id=f"event-{task_id}",
            task_id=task_id,
            active_snapshot_version=prepared.version,
            snapshot_message_id=f"snapshot-{task_id}",
            retained_message_ids=[],
            conversation_artifact=reference,
            input_hash="d" * 64,
        ),
    )


def test_public_filesystem_offload_is_searchable_by_diagnostic_tool(
    tmp_path,
    monkeypatch,
):
    backend, _, search_tool, _ = _environment(tmp_path, monkeypatch)
    full_output = "\n".join(
        ["pytest failure"] * 40
        + ["AssertionError: sign=-1 returned the wrong value"]
        + ["traceback detail"] * 40
    )
    offloaded = _offload(backend, "task-a", "pytest-current", full_output)
    messages = [
        HumanMessage(id="user-current", content="fix sign"),
        AIMessage(
            id="ai-current",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "id": "pytest-current",
                    "args": {"command": "python -m pytest -q"},
                    "type": "tool_call",
                }
            ],
        ),
        offloaded,
    ]

    result = _invoke(
        search_tool,
        "task-a",
        messages,
        "search-current",
        {"query": "AssertionError", "max_matches": 5},
    )

    assert "/.deepfix-artifacts/large_tool_results/pytest-current" in str(
        offloaded.content
    )
    assert "AssertionError" not in str(offloaded.content)
    assert result.status == "success"
    assert "AssertionError: sign=-1" in str(result.content)


def test_compacted_history_authorizes_old_offload_for_search_and_read(
    tmp_path,
    monkeypatch,
):
    backend, compaction, search_tool, read_tool = _environment(tmp_path, monkeypatch)
    full_output = "header\nArchivedAssertion: sign branch failed\nfooter\n" * 30
    offloaded = _offload(backend, "task-a", "pytest-old", full_output)
    old_messages = ensure_message_ids(
        "task-a",
        [
            AIMessage(
                id="ai-old",
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "id": "pytest-old",
                        "args": {"command": "python -m pytest -q"},
                        "type": "tool_call",
                    }
                ],
            ),
            offloaded,
        ],
    ).messages
    reference = DeepAgentsArtifactAdapter(backend).persist_history(
        "task-a",
        "attempt-old",
        old_messages,
        retained_ids=set(),
        work_unit_ids={"wu-old"},
    )
    _activate_history(compaction, "task-a", reference)
    current_messages = [HumanMessage(id="user-new", content="continue")]

    searched = _invoke(
        search_tool,
        "task-a",
        current_messages,
        "search-old",
        {"query": "ArchivedAssertion", "max_matches": 5},
    )
    large_match = re.search(
        r"\[(artifact_[a-f0-9]{32}) large_tool_result ",
        str(searched.content),
    )
    assert large_match is not None
    artifact_id = large_match.group(1)
    read = _invoke(
        read_tool,
        "task-a",
        [*current_messages, searched],
        "read-old",
        {"artifact_id": artifact_id, "start_line": 1, "line_count": 5},
    )

    assert searched.status == "success"
    assert "ArchivedAssertion" in str(searched.content)
    assert read.status == "success"
    assert "ArchivedAssertion: sign branch failed" in str(read.content)


def test_cross_task_and_non_diagnostic_artifacts_remain_unreachable(
    tmp_path,
    monkeypatch,
):
    backend, compaction, search_tool, read_tool = _environment(tmp_path, monkeypatch)
    secret = "TASK_B_PRIVATE_TRACEBACK"
    offloaded = _offload(backend, "task-b", "pytest-private", (secret + "\n") * 80)
    old_messages = ensure_message_ids(
        "task-b",
        [
            AIMessage(
                id="ai-private",
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "id": "pytest-private",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            offloaded,
        ],
    ).messages
    reference = DeepAgentsArtifactAdapter(backend).persist_history(
        "task-b",
        "attempt-private",
        old_messages,
        retained_ids=set(),
    )
    _activate_history(compaction, "task-b", reference)
    backend.write("/.deepfix-artifacts/research/task-a.md", secret)
    backend.write("/.deepfix-artifacts/receipts/task-a.json", secret)
    messages = [HumanMessage(id="user-a", content="continue")]

    searched = _invoke(
        search_tool,
        "task-a",
        messages,
        "search-private",
        {"query": secret},
    )
    forged_id = "artifact_" + hashlib.sha256(secret.encode()).hexdigest()[:32]
    read = _invoke(
        read_tool,
        "task-a",
        messages,
        "read-private",
        {"artifact_id": forged_id, "start_line": 1, "line_count": 20},
    )

    assert searched.status == "error"
    assert read.status == "error"
    assert secret not in str(searched.content)
    assert secret not in str(read.content)
