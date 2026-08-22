import pytest

from deepfix.config import ApprovalMode
from deepfix.models import (
    ApprovalRecord,
    ContextMetrics,
    Evidence,
    RepairOutcome,
    TaskState,
    TaskStatus,
)
from deepfix.models import TestResult as RepairTestResult


def test_create_normalizes_task_input(tmp_path):
    task = TaskState.create(tmp_path / ".", "  除法结果错误  ", ApprovalMode.MANUAL)

    assert task.project_root == str(tmp_path.resolve())
    assert task.user_problem == "除法结果错误"
    assert task.approval_mode == "manual"
    assert task.status is TaskStatus.CREATED


def test_create_rejects_empty_problem(tmp_path):
    with pytest.raises(ValueError, match="问题描述不能为空"):
        TaskState.create(tmp_path, "   ", ApprovalMode.MANUAL)


def test_new_task_can_begin_investigation(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.MANUAL)

    task.transition_to(TaskStatus.INVESTIGATING)

    assert task.status is TaskStatus.INVESTIGATING


def test_completed_task_cannot_return_to_editing(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.MANUAL)
    task.status = TaskStatus.COMPLETED

    with pytest.raises(ValueError, match="非法状态迁移"):
        task.transition_to(TaskStatus.EDITING)


def test_paused_task_resumes_previous_status(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)
    task.transition_to(TaskStatus.PAUSED)

    task.resume()

    assert task.status is TaskStatus.INVESTIGATING
    assert task.paused_from is None


def test_task_round_trip_preserves_pending_actions_and_processed_tool_calls(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.pending_actions.append(
        {"name": "execute", "args": {"command": "pytest -q"}}
    )
    task.processed_tool_call_ids.append("call-1")

    restored = TaskState.from_dict(task.to_dict())

    assert restored.pending_actions == task.pending_actions
    assert restored.processed_tool_call_ids == ["call-1"]


def test_task_round_trip_preserves_context_state(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.working_memory_version = 3
    task.context_metrics.context_peak_tokens = 4200
    task.offloaded_artifacts = ["conversation_history/a.md"]

    restored = TaskState.from_dict(task.to_dict())

    assert restored.working_memory_version == 3
    assert restored.context_metrics.context_peak_tokens == 4200
    assert restored.offloaded_artifacts == ["conversation_history/a.md"]
    assert isinstance(restored.context_metrics, ContextMetrics)


def test_task_from_old_payload_uses_context_defaults(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    payload = task.to_dict()
    payload.pop("working_memory_version")
    payload.pop("context_metrics")
    payload.pop("offloaded_artifacts")

    restored = TaskState.from_dict(payload)

    assert restored.working_memory_version == 0
    assert restored.context_metrics == ContextMetrics()
    assert restored.offloaded_artifacts == []
    assert restored.context_recovery is None


def test_task_round_trip_preserves_research_summary(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.external_evidence_ids = ["evidence-2", "evidence-1"]
    task.research_query_count = 3
    task.research_provider_errors = ["github: rate limited"]

    restored = TaskState.from_dict(task.to_dict())

    assert restored.external_evidence_ids == ["evidence-2", "evidence-1"]
    assert restored.research_query_count == 3
    assert restored.research_provider_errors == ["github: rate limited"]


def test_task_from_old_payload_uses_research_defaults(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    payload = task.to_dict()
    payload.pop("external_evidence_ids")
    payload.pop("research_query_count")
    payload.pop("research_provider_errors")

    restored = TaskState.from_dict(payload)

    assert restored.external_evidence_ids == []
    assert restored.research_query_count == 0
    assert restored.research_provider_errors == []


def test_active_task_cannot_resume(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.MANUAL)
    task.transition_to(TaskStatus.INVESTIGATING)

    with pytest.raises(ValueError, match="只有已暂停任务才能恢复"):
        task.resume()


def test_task_round_trip_preserves_nested_types(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.GUARDED)
    task.evidence.append(Evidence("tests/test_calc.py:8", "期望 2，实际为 3"))
    task.test_results.append(RepairTestResult("pytest -q", 1, "1 failed"))
    task.approvals.append(ApprovalRecord("execute: pytest -q", "approve", "L1"))
    task.transition_to(TaskStatus.INVESTIGATING)
    task.transition_to(TaskStatus.PAUSED)

    restored = TaskState.from_dict(task.to_dict())

    assert restored == task
    assert isinstance(restored.evidence[0], Evidence)
    assert isinstance(restored.test_results[0], RepairTestResult)
    assert isinstance(restored.approvals[0], ApprovalRecord)


def test_repair_outcome_converts_evidence_payload():
    outcome = RepairOutcome(
        status="completed",
        diagnosis="除法运算符使用错误",
        evidence=[{"source": "src/calc.py:4", "observation": "使用了加法运算符"}],
        summary="已修复并通过测试",
    )

    assert outcome.evidence == [Evidence("src/calc.py:4", "使用了加法运算符")]
