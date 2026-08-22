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
        "## 上下文管理",
        "",
        f"工作记忆版本：{task.working_memory_version}",
        f"上下文峰值估算：{task.context_metrics.context_peak_tokens} tokens",
        f"主动压缩次数：{task.context_metrics.active_compaction_count}",
        f"上下文溢出次数：{task.context_metrics.context_overflow_count}",
        f"最后主动压缩时间：{task.context_metrics.last_compaction_at or '无'}",
        "卸载文件：",
        *_list_or_none(task.offloaded_artifacts),
        "",
        "## 结论、风险与未验证项",
        "",
        f"结论：{task.final_summary or '无'}",
        f"复核：{task.review or '无'}",
        "残余风险：",
        *_list_or_none(task.residual_risks),
        "未验证项：",
        *_list_or_none(task.unverified_items),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _evidence_lines(task: TaskState) -> list[str]:
    if not task.evidence:
        return ["无"]
    return [
        f"- {item.source}：{item.observation}"
        for item in task.evidence
    ]


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
