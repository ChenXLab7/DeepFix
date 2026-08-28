from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepfix.navigation.rounds import completed_tool_rounds_after


def _round(
    message_id: str | None,
    calls: list[dict[str, object]],
    results: list[str] | None = None,
):
    messages = [AIMessage(id=message_id, content="", tool_calls=calls)]
    result_ids = (
        [call["id"] for call in calls]
        if results is None
        else results
    )
    for call_id in result_ids:
        messages.append(ToolMessage(id=f"result-{call_id}", content="ok", tool_call_id=call_id))
    return messages


def test_parallel_tool_calls_are_one_complete_round():
    messages = [
        HumanMessage(id="u1", content="fix it"),
        *_round(
            "a1",
            [
                {"id": "c1", "name": "read_file", "args": {"file_path": "a.py"}},
                {"id": "c2", "name": "read_file", "args": {"file_path": "b.py"}},
            ],
        ),
    ]
    delta = completed_tool_rounds_after(messages, None)
    assert delta.count == 1
    assert delta.latest_round_id == "a1"


def test_missing_parallel_result_is_not_complete():
    messages = _round(
        "a1",
        [{"id": "c1", "name": "read_file", "args": {}}, {"id": "c2", "name": "grep", "args": {}}],
        ["c1"],
    )
    assert completed_tool_rounds_after(messages, None).count == 0


def test_write_todos_only_round_is_not_an_action_round():
    messages = _round("a1", [{"id": "c1", "name": "write_todos", "args": {}}])
    assert completed_tool_rounds_after(messages, None).count == 0


def test_two_sequential_complete_rounds_count_in_order():
    messages = _round("a1", [{"id": "c1", "name": "read_file", "args": {}}]) + _round(
        "a2", [{"id": "c2", "name": "grep", "args": {}}]
    )
    delta = completed_tool_rounds_after(messages, None)
    assert (delta.count, delta.latest_round_id) == (2, "a2")


def test_cursor_prevents_recounting_and_counts_new_round():
    messages = _round("a1", [{"id": "c1", "name": "read_file", "args": {}}]) + _round(
        "a2", [{"id": "c2", "name": "grep", "args": {}}]
    )
    delta = completed_tool_rounds_after(messages, "a1")
    assert (delta.count, delta.latest_round_id) == (1, "a2")


def test_missing_cursor_after_compaction_establishes_zero_count_baseline():
    messages = _round("a2", [{"id": "c2", "name": "grep", "args": {}}])
    delta = completed_tool_rounds_after(messages, "a1")
    assert (delta.count, delta.latest_round_id) == (0, "a2")


def test_mixed_write_todos_and_action_round_counts_once():
    messages = _round(
        "a1",
        [
            {"id": "c1", "name": "write_todos", "args": {}},
            {"id": "c2", "name": "read_file", "args": {}},
        ],
    )
    delta = completed_tool_rounds_after(messages, None)
    assert (delta.count, delta.latest_round_id) == (1, "a1")


def test_missing_ai_id_is_ignored():
    messages = _round(None, [{"id": "c1", "name": "read_file", "args": {}}])
    delta = completed_tool_rounds_after(messages, None)
    assert (delta.count, delta.latest_round_id) == (0, None)


def test_missing_call_id_is_ignored():
    messages = _round("a1", [{"id": "", "name": "read_file", "args": {}}])
    delta = completed_tool_rounds_after(messages, None)
    assert (delta.count, delta.latest_round_id) == (0, None)


def test_no_messages_returns_empty_delta():
    delta = completed_tool_rounds_after([], None)
    assert (delta.count, delta.latest_round_id) == (0, None)


def test_no_complete_action_round_preserves_non_null_cursor():
    delta = completed_tool_rounds_after(
        _round("a1", [{"id": "c1", "name": "read_file", "args": {}}], []), "a0"
    )
    assert (delta.count, delta.latest_round_id) == (0, "a0")


def test_duplicate_tool_messages_do_not_multiply_round():
    messages = _round("a1", [{"id": "c1", "name": "read_file", "args": {}}])
    messages.append(ToolMessage(id="result-duplicate", content="again", tool_call_id="c1"))
    delta = completed_tool_rounds_after(messages, None)
    assert (delta.count, delta.latest_round_id) == (1, "a1")
