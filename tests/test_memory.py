from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from deepfix.memory import ProgressSnapshot, WorkingMemoryStore


def snapshot(summary: str = "已复现失败") -> ProgressSnapshot:
    return ProgressSnapshot(
        phase="investigating",
        summary=summary,
        facts=["失败测试可以稳定复现"],
        evidence=[
            {
                "source": "tests/test_calc.py:18",
                "observation": "期望 2，实际 -2",
            }
        ],
        active_hypotheses=["符号处理重复取反"],
        rejected_hypotheses=[],
        checked_files=["src/calc.py"],
        experiments=["pytest 单测退出码为 1"],
        next_steps=["直接验证 normalize_sign"],
        unresolved_questions=[],
    )


def test_store_versions_are_monotonic_and_task_local(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    first = store.save("task-a", snapshot("第一版"))
    second = store.save("task-a", snapshot("第二版"))
    other = store.save("task-b", snapshot("另一个任务"))

    assert (first.version, second.version, other.version) == (1, 2, 1)
    assert store.latest("task-a").snapshot.summary == "第二版"
    assert [item.version for item in store.list_versions("task-a")] == [1, 2]
    assert store.latest("missing") is None


def test_snapshot_rejects_more_than_ten_active_hypotheses():
    values = snapshot().model_dump()
    values["active_hypotheses"] = [f"假设 {index}" for index in range(11)]

    with pytest.raises(ValidationError):
        ProgressSnapshot.model_validate(values)


def test_concurrent_writers_allocate_unique_versions(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    with ThreadPoolExecutor(max_workers=4) as executor:
        saved = list(
            executor.map(
                lambda index: store.save("task-a", snapshot(f"并发版本 {index}")),
                range(4),
            )
        )

    assert sorted(item.version for item in saved) == [1, 2, 3, 4]
    assert [item.version for item in store.list_versions("task-a")] == [1, 2, 3, 4]


def test_context_metrics_keep_peak_and_count_events(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    store.record_peak_tokens("task-a", 120)
    store.record_peak_tokens("task-a", 80)
    store.record_overflow("task-a")
    store.record_compaction("task-a")

    metrics = store.metrics("task-a")

    assert metrics.context_peak_tokens == 120
    assert metrics.context_overflow_count == 1
    assert metrics.active_compaction_count == 1
    assert metrics.last_compaction_at is not None


def test_store_rejects_blank_task_id_and_negative_token_estimate(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    with pytest.raises(ValueError, match="task_id"):
        store.save("  ", snapshot())
    with pytest.raises(ValueError, match="token"):
        store.record_peak_tokens("task-a", -1)

    assert store.latest("task-a") is None
