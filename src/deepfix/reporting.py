from __future__ import annotations

from collections.abc import Sequence

from deepfix.models import TaskState
from deepfix.research.models import ExternalEvidence
from deepfix.research.reporting import render_external_evidence


def render_report(
    task: TaskState,
    external_evidence: Sequence[ExternalEvidence] = (),
) -> str:
    current_evidence = [
        item for item in external_evidence if item.task_id == task.task_id
    ]
    sections = [
        "# DeepFix 修复报告",
        "",
        "## 用户问题",
        "",
        task.user_problem,
        "",
        "## 根因与证据",
        "",
        f"根因：{task.diagnosis or '无'}",
        "证据：",
        *_evidence_lines(task),
        "",
        "## 外部研究证据",
        "",
        *render_external_evidence(current_evidence),
        "",
        "## 修改文件",
        "",
        *_list_or_none(task.changed_files),
        "",
        "## 测试结果",
        "",
        *_test_lines(task),
        "",
        "## 审批记录",
        "",
        *_approval_lines(task),
        "",
        "## 可信执行",
        "",
        f"源项目：{task.source_project_root or task.project_root}",
        f"任务 Workspace：{task.workspace_root or task.project_root}",
        f"Workspace Baseline：{task.workspace_baseline_id or '无'}",
        f"隔离级别：{task.confinement_level}",
        (
            "Verification Policy："
            f"{task.verification_policy_id or '无'}"
            + (
                f" (v{task.verification_policy_version})"
                if task.verification_policy_version is not None
                else ""
            )
        ),
        (
            "Required Oracle："
            f"{task.passed_required_oracle_count}/"
            f"{task.required_oracle_count} 通过"
        ),
        f"Supplemental 失败：{task.supplemental_failure_count}",
        "未完成 Operation："
        + ("、".join(task.unresolved_operation_ids) or "无"),
        "",
        "## 上下文管理",
        "",
        f"工作记忆版本：{task.working_memory_version}",
        f"上下文峰值估算：{task.context_metrics.context_peak_tokens} tokens",
        f"主动压缩次数：{task.context_metrics.active_compaction_count}",
        f"上下文溢出次数：{task.context_metrics.context_overflow_count}",
        (
            f"最近预算区域：{task.context_metrics.latest_budget_zone or '无'} "
            f"({task.context_metrics.latest_usage_ratio:.1%})"
        ),
        (
            "普通/紧急压缩次数："
            f"{task.context_metrics.normal_compaction_count}/"
            f"{task.context_metrics.emergency_compaction_count}"
        ),
        (
            "失败/直通/主动错误次数："
            f"{task.context_metrics.compaction_failure_count}/"
            f"{task.context_metrics.normal_zone_passthrough_count}/"
            f"{task.context_metrics.manual_compaction_error_count}"
        ),
        f"Overflow 重试次数：{task.context_metrics.overflow_retry_count}",
        (
            "生效 Snapshot 版本："
            f"{task.context_metrics.active_compaction_snapshot_version or '无'}"
        ),
        (
            "最后压缩 Artifact："
            f"{task.context_metrics.last_compaction_artifact or '无'}"
        ),
        f"最后压缩错误：{task.context_metrics.last_compaction_error or '无'}",
        f"最后主动压缩时间：{task.context_metrics.last_compaction_at or '无'}",
        "卸载文件：",
        *_list_or_none(task.offloaded_artifacts),
        "",
        "## 结论、风险与未验证项",
        "",
        f"处理结果：{_resolution_label(task)}",
        f"结论：{_conclusion(task)}",
        f"暂停原因：{task.pause_reason or '无'}",
        f"复核：{task.review or '无'}",
        "残余风险：",
        *_list_or_none(task.residual_risks),
        "未验证项：",
        *_unverified_lines(task),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _evidence_lines(task: TaskState) -> list[str]:
    if not task.evidence:
        return ["无"]
    return [
        f"- {item.source}：{item.observation}"
        for item in task.evidence
    ]


def _conclusion(task: TaskState) -> str:
    if task.pause_reason and task.successful_changed_files:
        paths = "、".join(task.successful_changed_files)
        verification = {
            "pending": "最新修改尚未经过测试验证",
            "passed": "最新修改已经通过测试验证",
            "failed": "最新修改后的测试仍然失败",
            "not_applicable": "最新修改缺少验证状态",
        }[task.latest_change_verification]
        return (
            f"代码修改已成功写入：{paths}；{verification}；"
            f"Agent 流程已暂停"
        )
    return task.final_summary or "无"


def _resolution_label(task: TaskState) -> str:
    return {
        "fixed": "已修复（fixed）",
        "not_reproduced": "未复现（not_reproduced）",
    }.get(task.resolution, "无")


def _unverified_lines(task: TaskState) -> list[str]:
    items = list(task.unverified_items)
    if task.latest_change_verification == "pending":
        pending = "最新成功修改尚未运行后续验证"
        if pending not in items:
            items.append(pending)
    return _list_or_none(items)


def _test_lines(task: TaskState) -> list[str]:
    if not task.test_results:
        return ["无"]
    return [
        f"- [{'通过' if result.exit_code == 0 else '失败'}] "
        f"`{result.command}` (exit_code={result.exit_code})：{result.summary}"
        for result in task.test_results
    ]


def _approval_lines(task: TaskState) -> list[str]:
    if not task.approvals:
        return ["无"]
    return [
        f"- [{record.risk}] {record.operation}：{record.decision}"
        for record in task.approvals
    ]


def _list_or_none(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["无"]
