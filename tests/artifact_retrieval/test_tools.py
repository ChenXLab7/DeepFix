from __future__ import annotations

import hashlib
import json

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from deepfix.artifact_retrieval.errors import (
    DiagnosticArtifactSystemError,
    DiagnosticArtifactToolError,
)
from deepfix.artifact_retrieval.models import (
    DiagnosticMatch,
    DiagnosticReadResult,
    DiagnosticSearchResult,
)
from deepfix.artifact_retrieval.tools import (
    build_read_diagnostic_artifact_tool,
    build_search_diagnostic_artifacts_tool,
)
from deepfix.compaction.identity import stable_generated_message_id
from deepfix.investigation.errors import InvestigationStateError

ARTIFACT_ID = "artifact_" + "a" * 32
CONTENT_HASH = "b" * 64


class ServiceStub:
    def __init__(self):
        self.calls = []
        self.error: Exception | None = None

    def search(
        self,
        task_id,
        messages,
        query,
        artifact_kinds,
        max_matches,
    ):
        self.calls.append(
            (
                "search",
                task_id,
                list(messages),
                query,
                artifact_kinds,
                max_matches,
            )
        )
        if self.error is not None:
            raise self.error
        return DiagnosticSearchResult(
            query_terms=["failure"],
            matches=[
                DiagnosticMatch(
                    artifact_id=ARTIFACT_ID,
                    kind="large_tool_result",
                    start_line=10,
                    end_line=12,
                    excerpt="10: failure\n11: trace\n12: assertion",
                    content_hash=CONTENT_HASH,
                )
            ],
            searched_artifact_count=2,
            omitted_artifact_count=0,
            truncated=False,
        )

    def read(
        self,
        task_id,
        messages,
        artifact_id,
        start_line,
        line_count,
    ):
        self.calls.append(
            (
                "read",
                task_id,
                list(messages),
                artifact_id,
                start_line,
                line_count,
            )
        )
        if self.error is not None:
            raise self.error
        return DiagnosticReadResult(
            artifact_id=ARTIFACT_ID,
            kind="large_tool_result",
            start_line=10,
            end_line=12,
            total_lines=20,
            content="10: failure\n11: trace\n12: assertion",
            content_hash=CONTENT_HASH,
            truncated=True,
        )


def invoke_tool(
    tool: BaseTool,
    call: dict[str, object],
    task_id: str = "task-a",
) -> ToolMessage:
    node = ToolNode([tool])
    result = node.invoke(
        {
            "messages": [
                HumanMessage(id="user-1", content="continue"),
                AIMessage(id="ai-1", content="", tool_calls=[call]),
            ]
        },
        {"configurable": {"thread_id": task_id}},
        runtime=Runtime(),
    )
    return result["messages"][0]


def search_call(call_id: str = "search-1"):
    return {
        "name": "search_diagnostic_artifacts",
        "id": call_id,
        "type": "tool_call",
        "args": {
            "query": "Failure",
            "artifact_kinds": ["large_tool_result"],
            "max_matches": 5,
        },
    }


def read_call(call_id: str = "read-1"):
    return {
        "name": "read_diagnostic_artifact",
        "id": call_id,
        "type": "tool_call",
        "args": {
            "artifact_id": ARTIFACT_ID,
            "start_line": 10,
            "line_count": 3,
        },
    }


def test_tool_schemas_hide_runtime_task_messages_and_backend_path(tmp_path):
    service = ServiceStub()
    search = build_search_diagnostic_artifacts_tool(service)
    read = build_read_diagnostic_artifact_tool(service)

    assert set(search.args) == {"query", "artifact_kinds", "max_matches"}
    assert set(read.args) == {"artifact_id", "start_line", "line_count"}


def test_search_returns_stable_safe_tool_message(tmp_path):
    service = ServiceStub()
    tool = build_search_diagnostic_artifacts_tool(
        service,
    )

    result = invoke_tool(tool, search_call())

    assert result.status == "success"
    assert result.id == stable_generated_message_id(
        "task-a",
        "search-1",
        "diagnostic_artifact_search",
    )
    assert result.name == "search_diagnostic_artifacts"
    assert "10: failure" in result.text
    assert result.artifact == {
        "result_type": "diagnostic_artifact_search",
        "artifact_ids": [ARTIFACT_ID],
        "match_count": 1,
        "searched_artifact_count": 2,
        "omitted_artifact_count": 0,
        "truncated": False,
        "content_hashes": [CONTENT_HASH],
        "query_terms_hash": hashlib.sha256(
            json.dumps(
                ["failure"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    assert "backend_path" not in str(result.artifact)
    assert service.calls[0][1] == "task-a"
    assert len(service.calls[0][2]) == 2


def test_read_returns_stable_safe_tool_message(tmp_path):
    service = ServiceStub()
    tool = build_read_diagnostic_artifact_tool(
        service,
    )

    result = invoke_tool(tool, read_call())

    assert result.status == "success"
    assert result.id == stable_generated_message_id(
        "task-a",
        "read-1",
        "diagnostic_artifact_read",
    )
    assert result.artifact == {
        "result_type": "diagnostic_artifact_read",
        "artifact_id": ARTIFACT_ID,
        "kind": "large_tool_result",
        "start_line": 10,
        "end_line": 12,
        "total_lines": 20,
        "content_hash": CONTENT_HASH,
        "truncated": True,
    }


def test_final_tool_content_never_exceeds_character_budget(tmp_path):
    class LongService(ServiceStub):
        def search(self, *args):
            result = super().search(*args)
            match = result.matches[0].model_copy(update={"excerpt": "界" * 12_000})
            return result.model_copy(update={"matches": [match]})

        def read(self, *args):
            result = super().read(*args)
            return result.model_copy(update={"content": "界" * 12_000})

    service = LongService()

    search_result = invoke_tool(
        build_search_diagnostic_artifacts_tool(service),
        search_call(),
    )
    read_result = invoke_tool(
        build_read_diagnostic_artifact_tool(service),
        read_call(),
    )

    assert len(search_result.text) == 12_000
    assert len(read_result.text) == 12_000
    assert search_result.artifact["truncated"] is True
    assert read_result.artifact["truncated"] is True


def test_recoverable_service_error_returns_bounded_stable_error(tmp_path):
    service = ServiceStub()
    service.error = DiagnosticArtifactToolError(
        "artifact_no_matches",
        "secret-value-" * 100,
    )
    tool = build_search_diagnostic_artifacts_tool(
        service,
    )

    result = invoke_tool(tool, search_call("search-error"))

    assert result.status == "error"
    assert result.id == stable_generated_message_id(
        "task-a",
        "search-error",
        "artifact_no_matches",
    )
    assert len(result.text) <= 300
    assert ("secret-value-" * 30) not in result.text
    assert result.artifact == {
        "result_type": "diagnostic_artifact_error",
        "error_code": "artifact_no_matches",
        "operation": "search_diagnostic_artifacts",
    }


def test_missing_runtime_task_returns_stable_error_without_service_call(tmp_path):
    service = ServiceStub()
    tool = build_search_diagnostic_artifacts_tool(
        service,
    )

    runtime = ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {}},
        stream_writer=lambda value: None,
        tool_call_id="runtime-error",
        store=None,
    )

    result = tool.func(
        query="Failure",
        artifact_kinds=["large_tool_result"],
        max_matches=5,
        runtime=runtime,
    )

    assert result.status == "error"
    assert result.artifact["error_code"] == "artifact_runtime_invalid"
    assert result.id == stable_generated_message_id(
        "unknown",
        "runtime-error",
        "artifact_runtime_invalid",
    )
    assert service.calls == []


@pytest.mark.parametrize(
    "builder,call",
    [
        (build_search_diagnostic_artifacts_tool, search_call("system-search")),
        (build_read_diagnostic_artifact_tool, read_call("system-read")),
    ],
)
def test_system_error_converts_to_investigation_recovery(
    tmp_path,
    builder,
    call,
):
    service = ServiceStub()
    service.error = DiagnosticArtifactSystemError(
        "diagnostic_artifact_backend_read_failed",
        "backend_read",
    )
    tool = builder(service)

    with pytest.raises(InvestigationStateError) as caught:
        invoke_tool(tool, call)

    assert caught.value.recovery.error_code == (
        "diagnostic_artifact_backend_read_failed"
    )
    assert caught.value.recovery.task_id == "task-a"
    assert caught.value.recovery.tool_call_id == call["id"]
    assert caught.value.recovery.recovery_action == (
        "pause_and_retry_diagnostic_artifact_read"
    )
