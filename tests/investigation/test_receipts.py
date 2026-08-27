import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import ToolMessage

from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
    ToolResultArtifact,
    receipt_from_result,
)


def receipt(task_id: str, call_id: str, content: str) -> ToolExecutionReceipt:
    return receipt_from_result(
        task_id,
        {
            "name": "read_file",
            "id": call_id,
            "args": {"file_path": f"/{content}.py"},
            "type": "tool_call",
        },
        ToolMessage(
            content=content,
            name="read_file",
            tool_call_id=call_id,
        ),
    )


def test_receipt_store_writes_only_under_internal_artifact_root(tmp_path):
    project_root = tmp_path / "project"
    artifact_root = tmp_path / "deepfix-artifacts" / "investigation_receipts"
    project_root.mkdir()
    store = ToolExecutionReceiptStore(artifact_root)

    store.save(receipt("task-a", "read-1", "source"))

    assert store.load("task-a", "read-1") is not None
    assert list(artifact_root.rglob("*.json"))
    assert not (project_root / "investigation_receipts").exists()


def test_parallel_receipt_saves_remain_complete_and_isolated(tmp_path):
    store = ToolExecutionReceiptStore(tmp_path / "investigation_receipts")
    first = receipt("task-a", "read-source", "source")
    second = receipt("task-a", "read-test", "test")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(store.save, item) for item in (first, second)]
        for future in futures:
            future.result()

    assert store.load("task-a", "read-source") == first
    assert store.load("task-a", "read-test") == second
    assert len(list((tmp_path / "investigation_receipts").rglob("*.json"))) == 2


def test_identical_receipt_save_is_idempotent_but_conflict_is_rejected(tmp_path):
    store = ToolExecutionReceiptStore(tmp_path / "investigation_receipts")
    original = receipt("task-a", "read-1", "source")
    conflicting = receipt("task-a", "read-1", "different")

    store.save(original)
    store.save(original)

    with pytest.raises(RuntimeError, match="回执冲突"):
        store.save(conflicting)

    assert store.load("task-a", "read-1") == original


def test_failed_atomic_replace_leaves_no_partial_receipt(tmp_path, monkeypatch):
    root = tmp_path / "investigation_receipts"
    store = ToolExecutionReceiptStore(root)

    def fail_replace(source, destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        store.save(receipt("task-a", "read-1", "source"))

    assert store.load("task-a", "read-1") is None
    assert not list(root.rglob("*.tmp"))


def test_receipt_save_uses_a_windows_safe_temporary_path(tmp_path):
    target_root_length = 164
    padding_length = max(1, target_root_length - len(str(tmp_path.resolve())) - 1)
    root = tmp_path / ("r" * padding_length)
    store = ToolExecutionReceiptStore(root)
    expected = receipt("task-a", "read-1", "source")

    store.save(expected)

    assert store.load("task-a", "read-1") == expected
    assert not list(root.rglob("*.tmp"))


def test_command_result_artifact_is_bounded_durable_and_idempotent(tmp_path):
    root = tmp_path / "deepfix-artifacts" / "investigation_receipts"
    store = ToolExecutionReceiptStore(root)
    result = ToolMessage(
        content="x" * 200,
        name="execute",
        tool_call_id="execute-1",
        artifact={"exit_code": 7},
    )

    first = store.save_result_artifact(
        "task-a", "execute-1", "execute", result, max_output_bytes=32
    )
    second = store.save_result_artifact(
        "task-a", "execute-1", "execute", result, max_output_bytes=32
    )

    assert first == second
    path = root.parent / first
    artifact = ToolResultArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    assert artifact.exit_code == 7
    assert len(artifact.output.encode("utf-8")) <= 32
    assert not list(root.parent.rglob("*.tmp"))
