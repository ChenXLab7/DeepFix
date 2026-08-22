from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from deepfix.compaction.models import (
    FactCandidate,
    HypothesisProgressInput,
    ProvenanceRef,
    SnapshotCoverage,
)
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


def test_context_metrics_record_budget_failures_and_event_versions(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    store.record_budget("task-a", 0.85, "normal_compaction")
    store.record_compaction_failure("task-a", "artifact_write_failed")
    store.record_passthrough("task-a")
    store.record_manual_error("task-a")
    store.record_overflow_retry("task-a")
    store.record_compaction_event(
        "task-a",
        snapshot_version=3,
        artifact_path="conversation_history/task-a.md",
        emergency=False,
    )

    metrics = store.metrics("task-a")
    assert metrics.latest_usage_ratio == 0.85
    assert metrics.latest_budget_zone == "normal_compaction"
    assert metrics.compaction_failure_count == 1
    assert metrics.normal_zone_passthrough_count == 1
    assert metrics.manual_compaction_error_count == 1
    assert metrics.overflow_retry_count == 1
    assert metrics.normal_compaction_count == 1
    assert metrics.active_compaction_snapshot_version == 3
    assert metrics.last_compaction_artifact == "conversation_history/task-a.md"
    assert metrics.last_compaction_error == "artifact_write_failed"


def test_context_metrics_bound_last_error_without_changing_failure_count(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    metrics = store.record_compaction_failure("task-a", "x" * 500)

    assert metrics.compaction_failure_count == 1
    assert len(metrics.last_compaction_error) == 300
    assert metrics.last_compaction_error.endswith("…")


def test_store_rejects_blank_task_id_and_negative_token_estimate(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    with pytest.raises(ValueError, match="task_id"):
        store.save("  ", snapshot())
    with pytest.raises(ValueError, match="token"):
        store.record_peak_tokens("task-a", -1)

    assert store.latest("task-a") is None


def test_save_progress_transitions_existing_hypothesis_by_id(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    source = ProvenanceRef(kind="work_unit", ref_id="wu-1")
    first = store.save_progress(
        "task-a",
        phase="investigating",
        summary="检查缓存",
        facts=[FactCandidate(text="失败可复现", sources=[source])],
        evidence=[],
        hypotheses=[
            HypothesisProgressInput(
                text="缓存过期",
                target_state="active",
                sources=[source],
            )
        ],
        checked_files=["src/cache.py"],
        experiments=[],
        next_steps=["验证缓存键"],
        unresolved_questions=[],
        coverage=SnapshotCoverage(covered_message_ids=["m1"]),
        valid_source_ids={"wu-1"},
    )
    hypothesis_id = first.snapshot.active_hypotheses[0].hypothesis_id

    second = store.save_progress(
        "task-a",
        phase="investigating",
        summary="已排除缓存",
        facts=[],
        evidence=[],
        hypotheses=[
            HypothesisProgressInput(
                hypothesis_id=hypothesis_id,
                text="缓存过期",
                target_state="rejected",
                reason="关闭缓存后仍失败",
                sources=[source],
            )
        ],
        checked_files=["src/cache.py"],
        experiments=["禁用缓存后退出码仍为 1"],
        next_steps=["检查符号逻辑"],
        unresolved_questions=[],
        coverage=SnapshotCoverage(covered_message_ids=["m1", "m2"]),
        valid_source_ids={"wu-1"},
    )

    assert second.snapshot.active_hypotheses == []
    assert second.snapshot.rejected_hypotheses[0].hypothesis_id == hypothesis_id
    assert second.snapshot.rejected_hypotheses[0].reason == "关闭缓存后仍失败"


def test_save_progress_reopens_rejected_hypothesis_with_new_identity(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    old_source = ProvenanceRef(kind="work_unit", ref_id="wu-old")
    first = store.save_progress(
        "task-a",
        phase="investigating",
        summary="排除旧假设",
        facts=[],
        evidence=[],
        hypotheses=[
            HypothesisProgressInput(
                text="缓存过期",
                target_state="active",
                sources=[old_source],
            )
        ],
        checked_files=[],
        experiments=[],
        next_steps=[],
        unresolved_questions=[],
        coverage=SnapshotCoverage(),
        valid_source_ids={"wu-old"},
    )
    old_id = first.snapshot.active_hypotheses[0].hypothesis_id
    store.save_progress(
        "task-a",
        phase="investigating",
        summary="确认排除",
        facts=[],
        evidence=[],
        hypotheses=[
            HypothesisProgressInput(
                hypothesis_id=old_id,
                text="缓存过期",
                target_state="rejected",
                reason="禁用缓存仍失败",
                sources=[old_source],
            )
        ],
        checked_files=[],
        experiments=[],
        next_steps=[],
        unresolved_questions=[],
        coverage=SnapshotCoverage(),
        valid_source_ids={"wu-old"},
    )
    new_source = ProvenanceRef(kind="system_evidence", ref_id="ev-new")

    reopened = store.save_progress(
        "task-a",
        phase="investigating",
        summary="新证据要求重开",
        facts=[],
        evidence=[],
        hypotheses=[
            HypothesisProgressInput(
                text="缓存过期",
                target_state="active",
                reason="新日志显示缓存键冲突",
                reopens_hypothesis_id=old_id,
                sources=[new_source],
            )
        ],
        checked_files=[],
        experiments=[],
        next_steps=[],
        unresolved_questions=[],
        coverage=SnapshotCoverage(),
        valid_source_ids={"ev-new"},
    )

    assert reopened.snapshot.rejected_hypotheses[0].hypothesis_id == old_id
    assert reopened.snapshot.active_hypotheses[0].hypothesis_id != old_id
    assert reopened.snapshot.active_hypotheses[0].reopens_hypothesis_id == old_id


def test_save_progress_rejects_text_only_transition_and_unknown_source(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    source = ProvenanceRef(kind="work_unit", ref_id="fabricated")

    with pytest.raises(ValueError, match="source"):
        store.save_progress(
            "task-a",
            phase="investigating",
            summary="非法来源",
            facts=[],
            evidence=[],
            hypotheses=[
                HypothesisProgressInput(
                    text="缓存过期",
                    target_state="rejected",
                    reason="仅文本匹配",
                    sources=[source],
                )
            ],
            checked_files=[],
            experiments=[],
            next_steps=[],
            unresolved_questions=[],
            coverage=SnapshotCoverage(),
            valid_source_ids=set(),
        )
