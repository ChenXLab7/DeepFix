import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import ToolMessage

from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
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
