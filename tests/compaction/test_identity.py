import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_claim_id,
    stable_generated_message_id,
    stable_hypothesis_id,
    stable_reopened_hypothesis_id,
    stable_work_unit_id,
)


@pytest.mark.parametrize(
    "message",
    [
        HumanMessage(content="bug"),
        AIMessage(
            content="inspect",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "a.py"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="body", tool_call_id="c1"),
        AIMessage(content="result explanation"),
        SystemMessage(content="snapshot"),
    ],
)
def test_missing_id_is_deterministic_for_every_message_type(message):
    first = ensure_message_ids("task-a", [message]).messages[0]
    second = ensure_message_ids("task-a", [message]).messages[0]

    assert first.id == second.id
    assert first.id.startswith("msg_")
    assert first.additional_kwargs["_deepfix_original_ordinal"] == 0
    assert message.id is None


def test_existing_id_and_original_ordinal_are_reused_after_reordering():
    original = HumanMessage(
        id="framework-id",
        content="bug",
        additional_kwargs={"_deepfix_original_ordinal": 9},
    )

    result = ensure_message_ids("task-a", [AIMessage(content="new"), original])

    assert result.messages[1].id == "framework-id"
    assert result.messages[1].additional_kwargs["_deepfix_original_ordinal"] == 9
    assert result.assigned_message_ids == (result.messages[0].id,)


def test_existing_duplicate_ids_are_reused_but_reported_ambiguous():
    result = ensure_message_ids(
        "task-a",
        [HumanMessage(id="same", content="a"), AIMessage(id="same", content="b")],
    )

    assert [message.id for message in result.messages] == ["same", "same"]
    assert result.conflicted_message_ids == frozenset({"same"})


def test_line_endings_and_structured_key_order_are_canonical():
    left = HumanMessage(content=[{"text": "a\r\nb", "type": "text"}])
    right = HumanMessage(content=[{"type": "text", "text": "a\nb"}])

    assert ensure_message_ids("task-a", [left]).messages[0].id == (
        ensure_message_ids("task-a", [right]).messages[0].id
    )


def test_semantic_and_generated_identities_are_stable_and_scoped():
    assert stable_claim_id("task-a", " pytest exits 1 ") == stable_claim_id(
        "task-a", "pytest exits 1"
    )
    assert stable_claim_id("task-a", "pytest exits 1") != stable_claim_id(
        "task-b", "pytest exits 1"
    )
    assert stable_hypothesis_id("task-a", "msg-1", " cache stale ") == (
        stable_hypothesis_id("task-a", "msg-1", "cache stale")
    )
    reopened = stable_reopened_hypothesis_id(
        "task-a", "hyp-old", "ev-new", "cache stale"
    )
    assert reopened != "hyp-old"
    assert reopened == stable_reopened_hypothesis_id(
        "task-a", "hyp-old", "ev-new", "cache stale"
    )
    assert stable_work_unit_id("task-a", ["m1", "m2"]) != stable_work_unit_id(
        "task-a", ["m2", "m1"]
    )
    assert stable_generated_message_id("task-a", "attempt-1", "manual_error") == (
        stable_generated_message_id("task-a", "attempt-1", "manual_error")
    )
