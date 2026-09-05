from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import ToolMessage

from deepfix.domain_repositories.execution import ExecutionRepository
from deepfix.investigation.receipts import ToolResultArtifactStorage, receipt_from_result
from deepfix.operations import (
    NewOperationEntry,
    OperationConflictError,
    OperationKind,
    OperationReconciler,
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


def _receipt(call_id: str, content: str = "ok"):
    return receipt_from_result(
        "task-1",
        {
            "name": "execute",
            "id": call_id,
            "args": {"command": "python -m pytest -q"},
            "type": "tool_call",
        },
        ToolMessage(content=content, name="execute", tool_call_id=call_id),
    )


def test_journal_requires_prepared_before_started(tmp_path) -> None:
    store = ExecutionRepository(tmp_path / "state.db")

    with pytest.raises(OperationTransitionError, match="missing"):
        store.mark_started("missing")


def test_prepare_is_idempotent_for_same_operation(
    tmp_path,
    prepared_operation,
) -> None:
    store = ExecutionRepository(tmp_path / "state.db")

    first = store.prepare(prepared_operation)
    second = store.prepare(prepared_operation)

    assert second == first
    assert second.status is OperationStatus.PREPARED
    assert store.list_incomplete("task-1") == [first]


def test_prepare_rejects_same_id_with_different_hash(
    tmp_path,
    prepared_operation,
) -> None:
    store = ExecutionRepository(tmp_path / "state.db")
    store.prepare(prepared_operation)
    changed = prepared_operation.model_copy(update={"call_hash": "e" * 64})

    with pytest.raises(OperationConflictError, match="conflict"):
        store.prepare(changed)


def test_transitions_are_ordered_and_exact_replays_are_idempotent(
    tmp_path,
    prepared_operation,
) -> None:
    store = ExecutionRepository(tmp_path / "state.db")
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
    receipt = _receipt(prepared.tool_call_id).model_copy(
        update={"call_hash": prepared.call_hash}
    )
    observed = store.observe_with_receipt(
        prepared.operation_id,
        post_state=post_state,
        receipt=receipt,
        artifact_references=[],
    )
    assert store.observe_with_receipt(
        prepared.operation_id,
        post_state=post_state,
        receipt=receipt,
        artifact_references=[],
    ) == observed
    committed = store.commit(prepared.operation_id)

    assert committed.status is OperationStatus.COMMITTED
    assert store.commit(prepared.operation_id) == committed
    assert store.list_incomplete("task-1") == []


def test_observe_replay_with_different_receipt_is_rejected(
    tmp_path,
    prepared_operation,
) -> None:
    store = ExecutionRepository(tmp_path / "state.db")
    store.prepare(prepared_operation)
    store.mark_started(prepared_operation.operation_id)
    post_state = prepared_operation.expected_post_state
    assert post_state is not None
    store.observe_with_receipt(
        prepared_operation.operation_id,
        post_state=post_state,
        receipt=_receipt(prepared_operation.tool_call_id).model_copy(
            update={"call_hash": prepared_operation.call_hash}
        ),
        artifact_references=[],
    )

    with pytest.raises(OperationConflictError, match="observation conflict"):
        store.observe_with_receipt(
            prepared_operation.operation_id,
            post_state=post_state,
            receipt=_receipt(prepared_operation.tool_call_id, "different").model_copy(
                update={"call_hash": prepared_operation.call_hash}
            ),
            artifact_references=[],
        )


def test_started_operation_can_be_marked_unknown_without_becoming_replayable(
    tmp_path,
    prepared_operation,
) -> None:
    store = ExecutionRepository(tmp_path / "state.db")
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
    store = ExecutionRepository(tmp_path / "state.db")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: store.prepare(prepared_operation), range(8))
        )

    assert all(result == results[0] for result in results)
    assert store.list_incomplete("task-1") == [results[0]]


def test_file_post_hash_reconstructs_observed_receipt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src" / "value.py"
    target.parent.mkdir(parents=True)
    target.write_text("broken\n", encoding="utf-8")
    store = ExecutionRepository(tmp_path / "state.db")
    artifacts = ToolResultArtifactStorage(tmp_path / "artifacts" / "receipts")
    prepared = NewOperationEntry(
        operation_id="operation-recover",
        task_id="task-1",
        experiment_id="legacy-task-1",
        tool_call_id="call-recover",
        operation_kind=OperationKind.FILE_EDIT,
        call_hash="a" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(
            target_path="src/value.py",
            target_exists=True,
            file_hash=_hash(target),
        ),
        expected_post_state=OperationStateSnapshot(
            target_path="src/value.py",
            target_exists=True,
            file_hash=_text_hash("fixed\n"),
        ),
    )
    store.prepare(prepared)
    store.mark_started(prepared.operation_id)
    target.write_text("fixed\n", encoding="utf-8")

    result = OperationReconciler(store, artifacts).reconcile_task(
        "task-1", workspace
    )

    assert result.reconstructed_operation_ids == [prepared.operation_id]
    assert result.conflict_operation_ids == []
    assert store.load_receipt("task-1", "call-recover") is not None
    assert store.load_operation(prepared.operation_id).status is OperationStatus.OBSERVED


def test_unexpected_file_hash_pauses_recovery(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src" / "value.py"
    target.parent.mkdir(parents=True)
    target.write_text("broken\n", encoding="utf-8")
    store = ExecutionRepository(tmp_path / "state.db")
    prepared = NewOperationEntry(
        operation_id="operation-conflict",
        task_id="task-1",
        experiment_id="legacy-task-1",
        tool_call_id="call-conflict",
        operation_kind=OperationKind.FILE_EDIT,
        call_hash="b" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(
            target_path="src/value.py",
            target_exists=True,
            file_hash=_hash(target),
        ),
        expected_post_state=OperationStateSnapshot(
            target_path="src/value.py",
            target_exists=True,
            file_hash=_text_hash("fixed\n"),
        ),
    )
    store.prepare(prepared)
    store.mark_started(prepared.operation_id)
    target.write_text("third-party change\n", encoding="utf-8")

    result = OperationReconciler(
        store,
        ToolResultArtifactStorage(tmp_path / "artifacts" / "receipts"),
    ).reconcile_task("task-1", workspace)

    assert result.conflict_operation_ids == [prepared.operation_id]
    assert store.load_operation(prepared.operation_id).status is OperationStatus.UNKNOWN


def test_command_artifact_with_exit_status_reconstructs_receipt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ExecutionRepository(tmp_path / "state.db")
    artifacts = ToolResultArtifactStorage(tmp_path / "artifacts" / "receipts")
    prepared = NewOperationEntry(
        operation_id="operation-command",
        task_id="task-1",
        experiment_id="legacy-task-1",
        tool_call_id="execute-1",
        operation_kind=OperationKind.COMMAND,
        call_hash="d" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(command_hash="e" * 64),
    )
    store.prepare(prepared)
    store.mark_started(prepared.operation_id)
    artifacts.save_result_artifact(
        "task-1",
        "execute-1",
        "execute",
        ToolMessage(
            content="1 failed",
            name="execute",
            tool_call_id="execute-1",
            artifact={"exit_code": 1},
        ),
    )

    result = OperationReconciler(store, artifacts).reconcile_task(
        "task-1", workspace
    )

    assert result.reconstructed_operation_ids == [prepared.operation_id]
    assert store.load_receipt("task-1", "execute-1").tool_message.artifact[
        "exit_code"
    ] == 1
    assert store.load_operation(prepared.operation_id).status is OperationStatus.OBSERVED


def test_command_without_verified_exit_status_is_unknown_and_terminated(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ExecutionRepository(tmp_path / "state.db")
    artifacts = ToolResultArtifactStorage(tmp_path / "artifacts" / "receipts")
    prepared = NewOperationEntry(
        operation_id="operation-command-unknown",
        task_id="task-1",
        experiment_id="legacy-task-1",
        tool_call_id="execute-unknown",
        operation_kind=OperationKind.COMMAND,
        call_hash="f" * 64,
        workspace_baseline_id="baseline-1",
        pre_state=OperationStateSnapshot(command_hash="1" * 64),
    )
    store.prepare(prepared)
    store.mark_started(prepared.operation_id)
    terminated: list[str] = []

    result = OperationReconciler(
        store,
        artifacts,
        terminate_process_group=lambda entry: terminated.append(entry.operation_id),
    ).reconcile_task("task-1", workspace)

    assert result.unknown_operation_ids == [prepared.operation_id]
    assert terminated == [prepared.operation_id]
    assert store.load_operation(prepared.operation_id).status is OperationStatus.UNKNOWN


def _hash(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_hash(content: str) -> str:
    import hashlib
    import os

    return hashlib.sha256(content.replace("\n", os.linesep).encode("utf-8")).hexdigest()
