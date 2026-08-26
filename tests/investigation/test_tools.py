from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from deepfix.compaction.identity import stable_generated_message_id
from deepfix.investigation.tools import (
    build_continue_investigation_tool,
    build_record_hypothesis_tool,
)
from investigation.helpers import coordinator_fixture, stagnated_coordinator


def invoke_tool(
    tool: BaseTool,
    call: dict[str, object],
    task_id: str,
) -> ToolMessage:
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


def valid_candidate_call(call_id: str) -> dict[str, object]:
    return {
        "name": "record_hypothesis",
        "id": call_id,
        "type": "tool_call",
        "args": {
            "statement": "capacity=0 may expose a boundary bug",
            "evidence_ids": [],
            "checked_locations": [],
            "target_state": "candidate",
            "reason": "needs a targeted test",
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
    assert result.id == stable_generated_message_id(
        "task-a",
        "call-1",
        "hypothesis_validation_error",
    )


def test_candidate_receipt_exposes_id_and_next_action_to_model(tmp_path):
    tool = build_record_hypothesis_tool(coordinator_fixture(tmp_path))

    result = invoke_tool(tool, valid_candidate_call("candidate-call"), "task-a")

    hypothesis_id = result.artifact["hypothesis_id"]
    assert result.status == "success"
    assert hypothesis_id in result.text
    assert "continue_investigation" in result.text


def test_continue_tool_returns_bound_permit(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)
    tool = build_continue_investigation_tool(coordinator)

    result = invoke_tool(tool, valid_continue_call("call-2"), "task-a")

    assert result.status == "success"
    assert result.artifact["tool_name"] == "grep"
    assert result.artifact["permit_id"]


def test_continue_tool_schema_hides_task_and_runtime_authority(tmp_path):
    tool = build_continue_investigation_tool(stagnated_coordinator(tmp_path))

    assert "task_id" not in tool.args
    assert "runtime" not in tool.args
    assert "hypothesis_ids" in tool.args


def test_validation_error_is_bounded_and_does_not_echo_full_input(tmp_path):
    tool = build_record_hypothesis_tool(coordinator_fixture(tmp_path))
    call = invalid_supported_call("call-long")
    call["args"]["statement"] = "secret-value-" * 100

    result = invoke_tool(tool, call, "task-a")

    assert result.status == "error"
    assert len(result.text) <= 300
    assert ("secret-value-" * 20) not in result.text
