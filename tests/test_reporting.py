from deepfix.config import ApprovalMode
from deepfix.models import ApprovalRecord, ContextMetrics, Evidence, TaskState
from deepfix.models import TestResult as RepairTestResult
from deepfix.reporting import render_report
from deepfix.research.models import ExternalEvidence


def external_evidence(**overrides):
    values = {
        "evidence_id": "evidence-1",
        "task_id": "task-1",
        "candidate_id": "candidate-1",
        "source_type": "official_docs",
        "evidence_level": "E1",
        "title": "Pydantic model_copy",
        "url": "https://docs.pydantic.dev/models/",
        "query": "pydantic model_copy",
        "relevant_excerpt": "model_copy accepts update.",
        "retrieved_at": "2026-08-22T00:00:00+00:00",
        "dependency_name": "pydantic",
        "documented_version": "2.8",
        "project_version": "2.8.4",
        "local_verification": "verified",
        "local_evidence": [Evidence("tests/test_models.py:10", "目标回归测试通过")],
        "linked_test_tool_call_ids": ["pytest-call-1"],
        "verification_explanation": "本地回归确认官方结论适用",
        "artifact_path": "/.deepfix-artifacts/research/task-1/evidence-1.md",
    }
    values.update(overrides)
    return ExternalEvidence.model_validate(values)


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


def test_report_exposes_trusted_execution_and_oracle_state(tmp_path):
    task = TaskState.create(tmp_path, "修复失败测试", ApprovalMode.MANUAL)
    task.source_project_root = str(tmp_path / "source")
    task.workspace_root = str(tmp_path / "workspaces" / task.task_id)
    task.workspace_baseline_id = "baseline-a"
    task.confinement_level = "guarded_local"
    task.verification_policy_id = "policy-a"
    task.verification_policy_version = 1
    task.required_oracle_count = 2
    task.passed_required_oracle_count = 1
    task.supplemental_failure_count = 1
    task.unresolved_operation_ids = ["operation-a"]

    report = render_report(task)

    assert "## 可信执行" in report
    assert f"源项目：{task.source_project_root}" in report
    assert f"任务 Workspace：{task.workspace_root}" in report
    assert "隔离级别：guarded_local" in report
    assert "Required Oracle：1/2 通过" in report
    assert "Supplemental 失败：1" in report
    assert "未完成 Operation：operation-a" in report


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


def test_report_labels_not_reproduced_resolution(tmp_path):
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.resolution = "not_reproduced"
    task.final_summary = "未复现用户描述的问题：当前环境中所运行的 pytest 测试通过，且未修改代码。"

    report = render_report(task)

    assert "处理结果：未复现（not_reproduced）" in report
    assert "结论：未复现用户描述的问题" in report


def test_paused_report_preserves_successful_unverified_change(tmp_path):
    task = TaskState.create(tmp_path, "查找首个元素失败", ApprovalMode.MANUAL)
    task.changed_files.append("python_programs/find_first_in_sorted.py")
    task.successful_changed_files.append("python_programs/find_first_in_sorted.py")
    task.latest_change_verification = "pending"
    task.pause_reason = "调查协调需要恢复：investigation_state_commit_failed"
    task.final_summary = task.pause_reason

    report = render_report(task)

    assert "代码修改已成功写入：python_programs/find_first_in_sorted.py" in report
    assert "最新修改尚未经过测试验证" in report
    assert "暂停原因：调查协调需要恢复：investigation_state_commit_failed" in report


def test_approved_change_without_success_evidence_is_not_reported_as_written(tmp_path):
    task = TaskState.create(tmp_path, "查找首个元素失败", ApprovalMode.MANUAL)
    task.changed_files.append("python_programs/find_first_in_sorted.py")
    task.pause_reason = "编辑前暂停"
    task.final_summary = task.pause_reason

    report = render_report(task)

    assert "代码修改已成功写入" not in report


def test_report_renders_deterministic_context_metrics_and_artifacts(tmp_path):
    task = TaskState.create(tmp_path, "超长修复任务", ApprovalMode.MANUAL)
    task.context_metrics = ContextMetrics(
        context_peak_tokens=4200,
        context_overflow_count=0,
        active_compaction_count=1,
        last_compaction_at="2026-08-21T10:00:00+00:00",
        latest_usage_ratio=0.91,
        latest_budget_zone="emergency",
        normal_compaction_count=2,
        emergency_compaction_count=1,
        compaction_failure_count=3,
        normal_zone_passthrough_count=1,
        manual_compaction_error_count=1,
        overflow_retry_count=2,
        active_compaction_snapshot_version=4,
        last_compaction_artifact="conversation_history/task.md",
        last_compaction_error="snapshot_verify_failed",
    )
    task.offloaded_artifacts = ["conversation_history/task.md"]

    report = render_report(task)

    assert "上下文峰值估算：4200 tokens" in report
    assert "主动压缩次数：1" in report
    assert "上下文溢出次数：0" in report
    assert "最近预算区域：emergency (91.0%)" in report
    assert "普通/紧急压缩次数：2/1" in report
    assert "失败/直通/主动错误次数：3/1/1" in report
    assert "Overflow 重试次数：2" in report
    assert "生效 Snapshot 版本：4" in report
    assert "最后压缩 Artifact：conversation_history/task.md" in report
    assert "最后压缩错误：snapshot_verify_failed" in report
    assert "conversation_history/task.md" in report


def test_report_renders_verified_external_evidence_with_true_local_linkage(tmp_path):
    task = TaskState.create(tmp_path, "模型复制失败", ApprovalMode.MANUAL)
    evidence = external_evidence(task_id=task.task_id)

    report = render_report(task, [evidence])

    assert "## 外部研究证据" in report
    assert "[E1/verified] Pydantic model_copy" in report
    assert "资料版本：2.8" in report
    assert "项目版本：2.8.4" in report
    assert "外部结论：model_copy accepts update." in report
    assert "tests/test_models.py:10：目标回归测试通过" in report
    assert "pytest-call-1" in report
    assert evidence.url in report
    assert evidence.artifact_path in report


def test_report_marks_unverified_e3_as_external_clue(tmp_path):
    task = TaskState.create(tmp_path, "模型复制失败", ApprovalMode.MANUAL)
    evidence = external_evidence(
        task_id=task.task_id,
        evidence_level="E3",
        source_type="github_issue",
        local_verification="unverified",
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
    )

    report = render_report(task, [evidence])

    assert "[E3/unverified]" in report
    assert "仅为外部线索" in report


def test_report_keeps_contradiction_and_version_mismatch_visible(tmp_path):
    task = TaskState.create(tmp_path, "模型复制失败", ApprovalMode.MANUAL)
    evidence = external_evidence(
        task_id=task.task_id,
        documented_version="3.0",
        project_version="2.8.4",
        local_verification="contradicted",
        verification_explanation="当前版本测试失败，官方结论不适用",
    )

    report = render_report(task, [evidence])

    assert "已被本地证据推翻" in report
    assert "当前版本测试失败，官方结论不适用" in report
    assert "版本不一致" in report


def test_report_uses_store_records_instead_of_task_model_prose(tmp_path):
    task = TaskState.create(tmp_path, "模型复制失败", ApprovalMode.MANUAL)
    task.evidence.append(Evidence("agent", "模型声称官方资料已验证"))
    stored = external_evidence(
        task_id=task.task_id,
        local_verification="unverified",
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
    )

    report = render_report(task, [stored])

    assert "[E1/unverified]" in report
    assert "仅为外部线索" in report
    assert "[E1/verified]" not in report
