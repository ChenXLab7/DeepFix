from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.database import SQLiteDatabase
from deepfix.investigation.token_budget import (
    TokenBudgetCallbackHandler,
    TokenBudgetConflict,
    TokenBudgetExhausted,
    TokenBudgetMiddleware,
    TokenBudgetStore,
)


def test_parallel_reservations_cannot_oversubscribe_budget(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=50)
    barrier = Barrier(2)

    def reserve(call_id):
        barrier.wait()
        try:
            store.reserve(
                "task-1",
                call_id,
                input_tokens=70,
                output_tokens=30,
            )
            return "reserved"
        except TokenBudgetExhausted:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ["call-1", "call-2"]))

    assert sorted(results) == ["blocked", "reserved"]


def test_reservation_replay_is_idempotent_but_conflicting_payload_fails(
    tmp_path,
) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=50)
    first = store.reserve(
        "task-1", "call-1", input_tokens=20, output_tokens=10
    )

    assert (
        store.reserve(
            "task-1", "call-1", input_tokens=20, output_tokens=10
        )
        == first
    )
    with pytest.raises(TokenBudgetConflict, match="identity"):
        store.reserve(
            "task-1", "call-1", input_tokens=21, output_tokens=10
        )


def test_store_accepts_shared_sqlite_database(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.db")
    first = TokenBudgetStore(database=database)
    second = TokenBudgetStore(database=database)
    first.initialize("task-1", input_cap=100, output_cap=50)
    first.reserve("task-1", "call-1", input_tokens=20, output_tokens=10)

    assert second.available("task-1").input_tokens == 80
    assert second.available("task-1").output_tokens == 40


def test_reservation_blocks_call_that_cannot_fit(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    store.reserve(
        "task-1",
        "call-1",
        input_tokens=80,
        output_tokens=20,
    )

    with pytest.raises(TokenBudgetExhausted):
        store.reserve(
            "task-1",
            "call-2",
            input_tokens=30,
            output_tokens=20,
        )


def test_unknown_usage_charges_full_reservation(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    reservation = store.reserve(
        "task-1",
        "call-1",
        input_tokens=80,
        output_tokens=20,
    )

    store.charge_unknown(reservation.reservation_id)

    available = store.available("task-1")
    assert available.input_tokens == 20
    assert available.output_tokens == 20


def test_settlement_releases_unused_reservation(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    reservation = store.reserve(
        "task-1",
        "call-1",
        input_tokens=80,
        output_tokens=20,
    )

    store.settle(reservation.reservation_id, input_tokens=60, output_tokens=10)

    available = store.available("task-1")
    assert available.input_tokens == 40
    assert available.output_tokens == 30


def test_usage_above_reservation_blocks_future_calls(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    reservation = store.reserve(
        "task-1",
        "call-1",
        input_tokens=60,
        output_tokens=10,
    )
    store.settle(reservation.reservation_id, input_tokens=70, output_tokens=10)

    with pytest.raises(TokenBudgetExhausted, match="breach"):
        store.reserve(
            "task-1",
            "call-2",
            input_tokens=1,
            output_tokens=1,
        )


def test_middleware_reserves_before_call_and_settles_provider_usage(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    middleware = TokenBudgetMiddleware(
        store,
        input_cap=100,
        output_cap=40,
        input_estimator=lambda _request: 80,
        requested_output_tokens=20,
        input_safety_margin=0,
    )
    called = False

    def handler(_request):
        nonlocal called
        called = True
        return ModelResponse(
            result=[
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 70,
                        "output_tokens": 10,
                        "total_tokens": 80,
                    },
                )
            ]
        )

    middleware.wrap_model_call(_request(), handler)

    assert called is True
    assert store.available("task-1").input_tokens == 30
    assert store.available("task-1").output_tokens == 30


def test_direct_model_callback_charges_unknown_usage_reservation(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    callback = TokenBudgetCallbackHandler(
        store,
        input_cap=100,
        output_cap=40,
        role="compaction",
        task_id_resolver=lambda: "task-1",
        requested_output_tokens=20,
        input_safety_margin=0,
    )
    model = FakeListChatModel(responses=["summary"]).with_config(
        callbacks=[callback]
    )

    model.invoke([HumanMessage(content="compact this history")])

    available = store.available("task-1")
    assert available.input_tokens < 100
    assert available.output_tokens == 20


def test_usage_aggregates_settled_and_unknown_model_roles(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "state.db")
    store.initialize("task-1", input_cap=100, output_cap=40)
    main = store.reserve(
        "task-1", "main-1", input_tokens=50, output_tokens=10
    )
    compaction = store.reserve(
        "task-1", "compaction-1", input_tokens=30, output_tokens=10
    )
    store.settle(main.reservation_id, input_tokens=40, output_tokens=5)
    store.charge_unknown(compaction.reservation_id)

    usage = store.usage("task-1")

    assert usage.input_tokens == 70
    assert usage.output_tokens == 15
    assert usage.model_calls == 2
    assert usage.estimated is True


def _request() -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="repair")],
        system_message=SystemMessage(content="policy"),
        tools=[],
        state={"messages": []},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-node-1",
                thread_id="task-1",
            )
        ),
    )
