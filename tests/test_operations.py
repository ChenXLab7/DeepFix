from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from deepfix.operations import (
    NewOperationEntry,
    OperationConflictError,
    OperationJournalStore,
    OperationKind,
    OperationStateSnapshot,
    OperationStatus,
    OperationTransitionError,
)


@pytest.fixture
def prepared_operation() -> NewOperationEntry:
    return NewOperationEntry(
        operation_id="operation-1",
        task_id="task-1",
        experiment_id="legacy-task-1",
        tool_call_id="call-1",
        operation_kind=OperationKind.FILE_EDIT,
        call_hash="a" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(
            target_path="src/value.py",
            file_hash="b" * 64,
            code_state_hash="c" * 64,
        ),
        expected_post_state=OperationStateSnapshot(
            target_path="src/value.py",
            file_hash="d" * 64,
        ),
    )


def test_journal_requires_prepared_before_started(tmp_path) -> None:
    store = OperationJournalStore(tmp_path / "state.db")

    with pytest.raises(OperationTransitionError, match="missing"):
        store.mark_started("missing")


def test_prepare_is_idempotent_for_same_operation(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")

    first = store.prepare(prepared_operation)
    second = store.prepare(prepared_operation)

    assert second == first
    assert second.status is OperationStatus.PREPARED
    assert store.list_incomplete("task-1") == [first]


def test_prepare_rejects_same_id_with_different_hash(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    store.prepare(prepared_operation)
    changed = prepared_operation.model_copy(update={"call_hash": "e" * 64})

    with pytest.raises(OperationConflictError, match="conflict"):
        store.prepare(changed)


def test_transitions_are_ordered_and_exact_replays_are_idempotent(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    prepared = store.prepare(prepared_operation)

    with pytest.raises(OperationTransitionError, match="observed"):
        store.commit(prepared.operation_id)

    started = store.mark_started(prepared.operation_id)
    assert store.mark_started(prepared.operation_id) == started
    post_state = OperationStateSnapshot(
        target_path="src/value.py",
        file_hash="d" * 64,
        code_state_hash="f" * 64,
    )
    observed = store.observe(
        prepared.operation_id,
        post_state=post_state,
        receipt_id="receipt-1",
        artifact_references=["operations/output-1.json"],
    )
    assert store.observe(
        prepared.operation_id,
        post_state=post_state,
        receipt_id="receipt-1",
        artifact_references=["operations/output-1.json"],
    ) == observed
    committed = store.commit(prepared.operation_id)

    assert committed.status is OperationStatus.COMMITTED
    assert store.commit(prepared.operation_id) == committed
    assert store.list_incomplete("task-1") == []


def test_observe_replay_with_different_receipt_is_rejected(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    store.prepare(prepared_operation)
    store.mark_started(prepared_operation.operation_id)
    post_state = prepared_operation.expected_post_state
    assert post_state is not None
    store.observe(
        prepared_operation.operation_id,
        post_state=post_state,
        receipt_id="receipt-1",
        artifact_references=[],
    )

    with pytest.raises(OperationConflictError, match="observation conflict"):
        store.observe(
            prepared_operation.operation_id,
            post_state=post_state,
            receipt_id="receipt-2",
            artifact_references=[],
        )


def test_started_operation_can_be_marked_unknown_without_becoming_replayable(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")
    store.prepare(prepared_operation)
    store.mark_started(prepared_operation.operation_id)

    unknown = store.mark_unknown(
        prepared_operation.operation_id,
        "workspace state cannot be reconciled",
    )

    assert unknown.status is OperationStatus.UNKNOWN
    assert unknown.unknown_reason == "workspace state cannot be reconciled"
    with pytest.raises(OperationTransitionError):
        store.mark_started(prepared_operation.operation_id)


def test_parallel_identical_prepare_creates_one_row(
    tmp_path,
    prepared_operation,
) -> None:
    store = OperationJournalStore(tmp_path / "state.db")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: store.prepare(prepared_operation), range(8))
        )

    assert all(result == results[0] for result in results)
    assert store.list_incomplete("task-1") == [results[0]]
