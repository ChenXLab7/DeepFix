from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from deepfix.compaction.models import (
    ArtifactReference,
    FileChangeEvidence,
    StrictModel,
    SystemTestEvidence,
)
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.evidence import (
    EvidenceEnvelope,
    EvidenceKind,
    VerificationEvidenceView,
    restore_external_evidence,
)
from deepfix.domain_repositories.execution import ExecutionApproval, ExecutionIntegrity
from deepfix.domain_repositories.history import ContextTelemetry
from deepfix.investigation.models import InvestigationHypothesis, UnresolvedQuestion
from deepfix.models import TaskState
from deepfix.operations import OperationJournalEntry
from deepfix.research.models import ExternalEvidence
from deepfix.research.reporting import render_external_evidence
from deepfix.task_domain.models import AdjudicationDecision, TaskDefinition, TaskLifecycle
from deepfix.verification import VerificationPolicy, evaluate_required_oracles


class HistorySummaryView(StrictModel):
    active_snapshot_version: int | None = None
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    compaction_failure_count: int = 0


class TaskReportView(StrictModel):
    definition: TaskDefinition
    lifecycle: TaskLifecycle
    decision: AdjudicationDecision | None = None
    hypotheses: list[InvestigationHypothesis] = Field(default_factory=list)
    unresolved_questions: list[UnresolvedQuestion] = Field(default_factory=list)
    evidence: list[EvidenceEnvelope] = Field(default_factory=list)
    operations: list[OperationJournalEntry] = Field(default_factory=list)
    approvals: list[ExecutionApproval] = Field(default_factory=list)
    execution_integrity: ExecutionIntegrity
    verification_policy: VerificationPolicy | None = None
    verification: VerificationEvidenceView
    history: HistorySummaryView
    context_telemetry: ContextTelemetry
    external_evidence: list[ExternalEvidence] = Field(default_factory=list)


def build_task_report_view(
    repositories: DomainRepositories,
    task_id: str,
) -> TaskReportView:
    records = repositories.history.list_for_task(task_id)
    active = next(
        (item for item in reversed(records) if item.lifecycle == "active"),
        None,
    )
    evidence = repositories.evidence.list_for_task(task_id)
    external = [
        restore_external_evidence(item)
        for item in evidence
        if item.kind is EvidenceKind.EXTERNAL_RESEARCH
    ]
    return TaskReportView(
        definition=repositories.tasks.get_definition(task_id),
        lifecycle=repositories.tasks.get_lifecycle(task_id),
        decision=repositories.tasks.latest_adjudication(task_id),
        hypotheses=repositories.investigation.list_hypotheses(task_id),
        unresolved_questions=repositories.investigation.list_questions(task_id),
        evidence=evidence,
        operations=repositories.execution.list_operations(task_id),
        approvals=repositories.execution.list_approvals(task_id),
        execution_integrity=repositories.execution.integrity_view(task_id),
        verification_policy=repositories.tasks.load_verification_policy(task_id),
        verification=repositories.evidence.verification_view(task_id),
        history=HistorySummaryView(
            active_snapshot_version=active.version if active is not None else None,
            artifact_references=(active.artifact_references if active is not None else []),
            compaction_failure_count=len(repositories.history.list_failures(task_id)),
        ),
        context_telemetry=repositories.history.context_telemetry(task_id),
        external_evidence=external,
    )


def render_report(
    task: TaskState | TaskReportView,
    external_evidence: Sequence[ExternalEvidence] = (),
) -> str:
    if isinstance(task, TaskReportView):
        return _render_repository_report(task)
    return _render_legacy_report(task, external_evidence)


def _render_legacy_report(
    task: TaskState,
    external_evidence: Sequence[ExternalEvidence] = (),
) -> str:
    current_evidence = [item for item in external_evidence if item.task_id == task.task_id]
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
        (f"Required Oracle：{task.passed_required_oracle_count}/{task.required_oracle_count} 通过"),
        f"Supplemental 失败：{task.supplemental_failure_count}",
        "未完成 Operation：" + ("、".join(task.unresolved_operation_ids) or "无"),
        "",
        "## 上下文管理",
        "",
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
        (f"生效 Snapshot 版本：{task.context_metrics.active_compaction_snapshot_version or '无'}"),
        (f"最后压缩 Artifact：{task.context_metrics.last_compaction_artifact or '无'}"),
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


def _render_repository_report(view: TaskReportView) -> str:
    tests = list(view.verification.test_evidence)
    changes = [
        item for item in view.verification.file_change_evidence if item.status == "succeeded"
    ]
    supported = [item for item in view.hypotheses if item.state == "supported"]
    policy = view.verification_policy
    oracle = evaluate_required_oracles(policy, tests) if policy is not None else None
    metrics = view.context_telemetry
    sections = [
        "# DeepFix 修复报告",
        "",
        "## 用户问题",
        "",
        view.definition.original_problem,
        "",
        "## 根因与证据",
        "",
        *(
            [
                f"- {item.statement}（Evidence: " + (", ".join(item.evidence_ids) or "无") + "）"
                for item in supported
            ]
            or ["无"]
        ),
        "",
        "## 外部研究证据",
        "",
        *render_external_evidence(view.external_evidence),
        "",
        "## 修改文件",
        "",
        *_list_or_none([item.path for item in changes]),
        "",
        "## 测试结果",
        "",
        *_repository_test_lines(tests),
        "",
        "## 审批记录",
        "",
        *_repository_approval_lines(view.approvals),
        "",
        "## 可信执行",
        "",
        f"源项目：{view.definition.source_project_root}",
        f"任务 Workspace：{view.definition.workspace_root}",
        f"Workspace Baseline：{view.definition.workspace_baseline_id or '无'}",
        f"隔离级别：{view.definition.confinement_level}",
        "Verification Policy："
        + (f"{policy.policy_id} (v{policy.version})" if policy is not None else "无"),
        (
            "Required Oracle："
            + (
                f"{len(oracle.passed_required_oracle_ids)}/{len(policy.required_oracles)} 通过"
                if oracle is not None and policy is not None
                else "0/0 通过"
            )
        ),
        "Supplemental/冲突失败："
        + (str(len(oracle.conflicting_evidence_ids)) if oracle is not None else "0"),
        "未完成 Operation："
        + (", ".join(view.execution_integrity.incomplete_operation_ids) or "无"),
        "",
        "## 上下文管理",
        "",
        f"上下文峰值估算：{metrics.context_peak_tokens} tokens",
        f"主动压缩次数：{metrics.active_compaction_count}",
        f"上下文溢出次数：{metrics.context_overflow_count}",
        (f"最近预算区域：{metrics.latest_budget_zone or '无'} ({metrics.latest_usage_ratio:.1%})"),
        (
            "普通/紧急压缩次数："
            f"{metrics.normal_compaction_count}/{metrics.emergency_compaction_count}"
        ),
        (
            "失败/直通/主动错误次数："
            f"{metrics.compaction_failure_count}/"
            f"{metrics.normal_zone_passthrough_count}/"
            f"{metrics.manual_compaction_error_count}"
        ),
        f"Overflow 重试次数：{metrics.overflow_retry_count}",
        "生效 Snapshot 版本："
        + (
            str(view.history.active_snapshot_version)
            if view.history.active_snapshot_version is not None
            else "无"
        ),
        f"最后压缩 Artifact：{metrics.last_compaction_artifact or '无'}",
        f"最后压缩错误：{metrics.last_compaction_error or '无'}",
        "历史 Artifact：",
        *_list_or_none([item.path for item in view.history.artifact_references]),
        "",
        "## 结论、风险与未验证项",
        "",
        f"处理结果：{_repository_resolution_label(view)}",
        f"结论：{_repository_conclusion(view, changes, tests)}",
        f"暂停原因：{view.lifecycle.reason or '无'}",
        "残余风险：",
        *_list_or_none([item.text for item in view.unresolved_questions if item.status == "open"]),
        "未验证项：",
        *_repository_unverified_lines(view, changes, tests),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _repository_resolution_label(view: TaskReportView) -> str:
    if view.decision is None:
        return "尚无可信终态裁决"
    return {
        "fixed": "已修复（fixed）",
        "not_reproduced": "未复现（not_reproduced）",
        "paused": "已暂停（paused）",
        "failed": "失败（failed）",
        "cancelled": "已取消（cancelled）",
    }[view.decision.outcome]


def _repository_conclusion(
    view: TaskReportView,
    changes: list[FileChangeEvidence],
    tests: list[SystemTestEvidence],
) -> str:
    if view.decision is not None and view.decision.outcome == "fixed":
        return "修复已由可信裁决确认；依据：" + ", ".join(view.decision.evidence_ids)
    if view.decision is not None and view.decision.outcome == "not_reproduced":
        return "未复现用户描述的问题；依据：" + ", ".join(view.decision.evidence_ids)
    if view.lifecycle.reason and changes:
        latest_test = tests[-1] if tests else None
        verification = (
            "最新测试通过"
            if latest_test is not None and latest_test.exit_code == 0
            else "最新修改尚未获得通过的验证"
        )
        return f"代码修改已成功写入：{', '.join(item.path for item in changes)}；{verification}"
    return view.lifecycle.reason or "任务尚未形成可信终态裁决"


def _repository_unverified_lines(
    view: TaskReportView,
    changes: list[FileChangeEvidence],
    tests: list[SystemTestEvidence],
) -> list[str]:
    values = [item.text for item in view.unresolved_questions if item.status == "open"]
    if changes and not any(item.exit_code == 0 for item in tests):
        values.append("最新成功修改尚未运行通过的后续验证")
    if view.execution_integrity.incomplete_operation_ids:
        values.append("存在未完成的副作用 Operation")
    return _list_or_none(list(dict.fromkeys(values)))


def _repository_test_lines(items: list[SystemTestEvidence]) -> list[str]:
    if not items:
        return ["无"]
    return [
        f"- [{'通过' if item.exit_code == 0 else '失败'}] "
        f"`{item.command}` (exit_code={item.exit_code})：{item.summary}"
        for item in items
    ]


def _repository_approval_lines(items: list[ExecutionApproval]) -> list[str]:
    if not items:
        return ["无"]
    return [f"- [{item.risk}] {item.operation}：{item.decision}" for item in items]


def _evidence_lines(task: TaskState) -> list[str]:
    if not task.evidence:
        return ["无"]
    return [f"- {item.source}：{item.observation}" for item in task.evidence]


def _conclusion(task: TaskState) -> str:
    if task.pause_reason and task.successful_changed_files:
        paths = "、".join(task.successful_changed_files)
        verification = {
            "pending": "最新修改尚未经过测试验证",
            "passed": "最新修改已经通过测试验证",
            "failed": "最新修改后的测试仍然失败",
            "not_applicable": "最新修改缺少验证状态",
        }[task.latest_change_verification]
        return f"代码修改已成功写入：{paths}；{verification}；Agent 流程已暂停"
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
    return [f"- [{record.risk}] {record.operation}：{record.decision}" for record in task.approvals]


def _list_or_none(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["无"]
