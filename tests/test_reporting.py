from deepfix.config import ApprovalMode
from deepfix.models import ApprovalRecord, ContextMetrics, Evidence, TaskState
from deepfix.models import TestResult as RepairTestResult
from deepfix.reporting import render_report


def test_report_contains_evidence_changes_tests_approvals_and_risks(tmp_path):
    task = TaskState.create(tmp_path, "除法结果错误", ApprovalMode.MANUAL)
    task.diagnosis = "除数为负数时选择了错误分支"
    task.evidence.append(Evidence("tests/test_calc.py:8", "期望 2，实际为 3"))
    task.changed_files.append("src/calc.py")
    task.test_results.append(RepairTestResult("pytest -q", 0, "12 passed"))
    task.approvals.append(ApprovalRecord("edit_file", "approve", "L1"))
    task.final_summary = "整数除法错误已修复"
    task.residual_risks.append("尚未进行性能压测")
    task.unverified_items.append("Windows 3.11")

    report = render_report(task)

    assert "除数为负数时选择了错误分支" in report
    assert "tests/test_calc.py:8" in report
    assert "src/calc.py" in report
    assert "[通过] `pytest -q` (exit_code=0)：12 passed" in report
    assert "[L1] edit_file：approve" in report
    assert "尚未进行性能压测" in report
    assert "Windows 3.11" in report


def test_report_sections_have_stable_order_and_empty_collections_are_explicit(tmp_path):
    task = TaskState.create(tmp_path, "排序结果不稳定", ApprovalMode.GUARDED)

    report = render_report(task)

    headings = [
        "# DeepFix 修复报告",
        "## 用户问题",
        "## 根因与证据",
        "## 修改文件",
        "## 测试结果",
        "## 审批记录",
        "## 上下文管理",
        "## 结论、风险与未验证项",
    ]
    assert [report.index(heading) for heading in headings] == sorted(
        report.index(heading) for heading in headings
    )
    assert report.count("无") >= 4


def test_report_uses_exit_code_not_success_wording_to_mark_test_result(tmp_path):
    task = TaskState.create(tmp_path, "测试仍然失败", ApprovalMode.MANUAL)
    task.test_results.append(RepairTestResult("pytest -q", 1, "看起来像 12 passed"))

    report = render_report(task)

    assert "[失败] `pytest -q` (exit_code=1)" in report
    assert "[通过] `pytest -q`" not in report


def test_report_renders_deterministic_context_metrics_and_artifacts(tmp_path):
    task = TaskState.create(tmp_path, "超长修复任务", ApprovalMode.MANUAL)
    task.working_memory_version = 3
    task.context_metrics = ContextMetrics(
        context_peak_tokens=4200,
        context_overflow_count=0,
        active_compaction_count=1,
        working_memory_version=3,
        last_compaction_at="2026-08-21T10:00:00+00:00",
    )
    task.offloaded_artifacts = ["conversation_history/task.md"]

    report = render_report(task)

    assert "工作记忆版本：3" in report
    assert "上下文峰值估算：4200 tokens" in report
    assert "主动压缩次数：1" in report
    assert "上下文溢出次数：0" in report
    assert "conversation_history/task.md" in report
