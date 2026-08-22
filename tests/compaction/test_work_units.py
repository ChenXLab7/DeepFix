from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepfix.compaction.work_units import partition_work_units


def _call(name: str, call_id: str, args=None):
    return {
        "name": name,
        "args": args or {},
        "id": call_id,
        "type": "tool_call",
    }


def test_parallel_calls_and_out_of_order_results_form_one_unit():
    messages = [
        AIMessage(
            id="m1",
            content="inspect",
            tool_calls=[_call("read_file", "c1"), _call("grep", "c2")],
        ),
        ToolMessage(id="m2", content="grep result", tool_call_id="c2"),
        ToolMessage(id="m3", content="file result", tool_call_id="c1"),
        AIMessage(id="m4", content="both point to parser"),
    ]

    partition = partition_work_units(messages, set())
    unit = partition.units[0]

    assert unit.message_ids == ["m1", "m2", "m3", "m4"]
    assert unit.tool_call_ids == ["c1", "c2"]
    assert unit.state == "complete"
    assert unit.must_keep is False
    assert unit.categories == {"read", "search"}
    assert partition.safe_cut_indices == [4]


def test_missing_result_keeps_entire_incomplete_unit():
    messages = [
        AIMessage(
            id="m1",
            content="inspect",
            tool_calls=[_call("read_file", "c1"), _call("grep", "c2")],
        ),
        ToolMessage(id="m2", content="file result", tool_call_id="c1"),
        AIMessage(id="m3", content="partial explanation"),
    ]

    unit = partition_work_units(messages, set()).units[0]

    assert unit.message_ids == ["m1", "m2", "m3"]
    assert unit.state == "incomplete"
    assert unit.must_keep is True


def test_orphan_result_and_duplicate_call_ids_are_ambiguous():
    messages = [
        ToolMessage(id="orphan", content="unknown", tool_call_id="c0"),
        AIMessage(id="m1", content="a", tool_calls=[_call("grep", "same")]),
        ToolMessage(id="m2", content="a result", tool_call_id="same"),
        AIMessage(id="m3", content="b", tool_calls=[_call("read_file", "same")]),
        ToolMessage(id="m4", content="b result", tool_call_id="same"),
    ]

    partition = partition_work_units(messages, set())

    assert all(unit.state == "ambiguous" for unit in partition.units)
    assert all(unit.must_keep for unit in partition.units)
    assert any("orphan" in diagnostic for diagnostic in partition.diagnostics)
    assert any("duplicate tool_call_id" in diagnostic for diagnostic in partition.diagnostics)


def test_result_crossing_user_turn_is_ambiguous_and_not_split():
    messages = [
        AIMessage(id="m1", content="inspect", tool_calls=[_call("grep", "c1")]),
        HumanMessage(id="m2", content="also check config"),
        ToolMessage(id="m3", content="late result", tool_call_id="c1"),
    ]

    partition = partition_work_units(messages, set())

    assert partition.units[0].state == "ambiguous"
    assert partition.units[0].message_ids == ["m1", "m2", "m3"]
    assert partition.units[0].must_keep is True


def test_conflicted_message_id_marks_affected_region_ambiguous():
    messages = [
        AIMessage(id="same", content="inspect", tool_calls=[_call("grep", "c1")]),
        ToolMessage(id="m2", content="result", tool_call_id="c1"),
        AIMessage(id="same", content="explain"),
    ]

    unit = partition_work_units(messages, {"same"}).units[0]

    assert unit.state == "ambiguous"
    assert unit.must_keep is True


def test_conversational_messages_have_stable_safe_boundaries():
    messages = [
        HumanMessage(id="u1", content="bug"),
        AIMessage(id="a1", content="I will inspect"),
        HumanMessage(id="u2", content="do not edit tests"),
    ]

    partition = partition_work_units(messages, set())

    assert [unit.message_ids for unit in partition.units] == [["u1", "a1"], ["u2"]]
    assert partition.safe_cut_indices == [2, 3]
    assert partition.unassigned_message_ids == []
