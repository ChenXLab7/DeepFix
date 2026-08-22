from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from deepfix.compaction.budget import (
    ContextBudgetConfigurationError,
    ContextBudgetMonitor,
    ContextBudgetReport,
    classify_budget_zone,
    select_retained_units,
)
from deepfix.compaction.models import WorkUnit


@pytest.mark.parametrize(
    ("ratio", "zone"),
    [
        (0.75, "normal"),
        (0.75001, "observe"),
        (0.82, "observe"),
        (0.82001, "normal_compaction"),
        (0.90, "normal_compaction"),
        (0.90001, "emergency"),
    ],
)
def test_budget_boundaries(ratio, zone):
    assert classify_budget_zone(ratio) == zone


def test_measure_counts_system_protection_messages_tools_and_reserve():
    counted = []

    def one_token(value: str) -> int:
        counted.append(value)
        return 1

    model = FakeListChatModel(
        responses=["ok"],
        profile={"max_input_tokens": 20},
    )
    request = ModelRequest(
        model=model,
        system_message=SystemMessage(content="base-system"),
        messages=[HumanMessage(content="first"), HumanMessage(content="second")],
        tools=[{"name": "read_file"}],
        state={"messages": []},
    )
    report = ContextBudgetMonitor(
        token_counter=one_token,
        output_reserve_tokens=2,
    ).measure(request, ["anchor", "memory", "evidence"])

    assert report.request_tokens == 9
    assert report.max_input_tokens == 20
    assert report.output_reserve_tokens == 2
    assert len(counted) == 7
    assert {"base-system", "anchor", "memory", "evidence", "first", "second"} <= set(
        counted
    )


def test_measure_requires_profile_or_explicit_model_table_entry():
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=[],
        state={"messages": []},
    )

    with pytest.raises(ContextBudgetConfigurationError, match="max_input_tokens"):
        ContextBudgetMonitor(output_reserve_tokens=0).measure(request, [])


@pytest.mark.parametrize("model_name", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_v4_models_have_explicit_one_million_token_fallback(model_name):
    monitor = ContextBudgetMonitor(
        token_counter=lambda value: 820_000,
        output_reserve_tokens=0,
    )
    request = SimpleNamespace(
        model=SimpleNamespace(model_name=model_name, profile=None),
        system_message=None,
        messages=[HumanMessage(content="large")],
        tools=[],
    )

    report = monitor.measure(request, [])

    assert report.max_input_tokens == 1_000_000
    assert report.usage_ratio == pytest.approx(0.82)
    assert report.zone == "observe"


def _unit(
    unit_id: str,
    index: int,
    *,
    categories=("other",),
    state="complete",
    message_ids=None,
):
    ids = message_ids or [f"m-{unit_id}"]
    return WorkUnit(
        unit_id=unit_id,
        purpose=unit_id,
        message_ids=ids,
        tool_call_ids=[],
        state=state,
        categories=set(categories),
        start_index=index,
        end_index=index + len(ids) - 1,
        must_keep=state != "complete",
    )


def test_retention_prioritizes_whole_units_and_mandatory_context():
    units = [
        _unit("old", 0),
        _unit("failed", 1, categories=("verify_fail",)),
        _unit("modify-verify", 2, categories=("modify", "verify_pass")),
        _unit("incomplete", 3, state="incomplete"),
        _unit("latest-user", 4, message_ids=["user-latest"]),
    ]
    report = ContextBudgetReport(
        request_tokens=90,
        max_input_tokens=100,
        output_reserve_tokens=0,
        usage_ratio=0.90,
        zone="normal_compaction",
        target_ratio=0.75,
    )

    plan = select_retained_units(report, units, "user-latest")

    assert {unit.unit_id for unit in plan.retained_units} == {
        "failed",
        "modify-verify",
        "incomplete",
        "latest-user",
    }
    assert [unit.unit_id for unit in plan.compressed_units] == ["old"]
    assert plan.retained_message_ids == frozenset(
        message_id
        for unit in plan.retained_units
        for message_id in unit.message_ids
    )
    assert plan.estimated_ratio <= 0.75


def test_mandatory_unit_is_retained_even_when_it_exceeds_target():
    units = [
        _unit("large-incomplete", 0, state="ambiguous", message_ids=["m1", "m2", "m3"]),
        _unit("latest", 3, message_ids=["user-latest"]),
    ]
    report = ContextBudgetReport(
        request_tokens=100,
        max_input_tokens=100,
        output_reserve_tokens=0,
        usage_ratio=1.0,
        zone="emergency",
        target_ratio=0.65,
    )

    plan = select_retained_units(report, units, "user-latest")

    assert {unit.unit_id for unit in plan.retained_units} == {
        "large-incomplete",
        "latest",
    }
    assert plan.compressed_units == ()


def test_observation_hint_is_once_per_memory_version_and_latest_unit():
    monitor = ContextBudgetMonitor(output_reserve_tokens=0)

    assert monitor.should_emit_memory_hint(2, "wu-3") is True
    assert monitor.should_emit_memory_hint(2, "wu-3") is False
    assert monitor.should_emit_memory_hint(2, "wu-4") is True
