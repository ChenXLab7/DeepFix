from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from deepfix.compaction.identity import stable_generated_message_id
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.models import (
    CheckedLocation,
    ContinueInvestigationInput,
    ProposedChange,
    RecordHypothesisInput,
)

_RECORD_DESCRIPTION = (
    "记录或迁移当前调查假设。supported 假设必须引用系统证据、已检查位置、"
    "拟修改目标和预期效果；任务 ID 由运行时自动提供。"
)
_CONTINUE_DESCRIPTION = (
    "停滞门禁触发后，为一个与现有假设绑定的明确调查动作申请一次性许可。"
    "工具、目标和任务 ID 都会被系统校验。"
)


def build_record_hypothesis_tool(
    coordinator: InvestigationCoordinator,
) -> BaseTool:
    def record_hypothesis(
        statement: str,
        evidence_ids: list[str],
        checked_locations: list[CheckedLocation],
        target_state: Literal["candidate", "rejected", "supported"],
        reason: str,
        runtime: ToolRuntime,
        hypothesis_id: str | None = None,
        proposed_change: ProposedChange | None = None,
        expected_effect: str | None = None,
    ) -> ToolMessage:
        task_id = _task_id(runtime)
        call_id = str(runtime.tool_call_id or "").strip()
        if not task_id:
            return _error_message(
                "unknown",
                call_id or "record-hypothesis",
                "hypothesis_validation_error",
                "记录假设失败：运行配置缺少 thread_id",
                "record_hypothesis",
            )
        if not call_id:
            return _error_message(
                task_id,
                "missing-call-id",
                "hypothesis_validation_error",
                "记录假设失败：缺少 tool_call_id",
                "record_hypothesis",
            )
        try:
            record = coordinator.record_hypothesis(
                task_id,
                RecordHypothesisInput(
                    hypothesis_id=hypothesis_id,
                    statement=statement,
                    evidence_ids=evidence_ids,
                    checked_locations=checked_locations,
                    proposed_change=proposed_change,
                    expected_effect=expected_effect,
                    target_state=target_state,
                    reason=reason,
                ),
                source_id=call_id,
            )
        except (ValidationError, ValueError) as exc:
            return _error_message(
                task_id,
                call_id,
                "hypothesis_validation_error",
                f"记录假设失败：{_safe_validation_error(exc)}",
                "record_hypothesis",
            )
        next_action = (
            "；需要收集新证据时，调用 continue_investigation 并引用该 hypothesis_id"
            if record.state == "candidate"
            else ""
        )
        return _success_message(
            task_id,
            call_id,
            "hypothesis_recorded",
            (
                f"调查假设已记录：hypothesis_id={record.hypothesis_id}，"
                f"state={record.state}{next_action}"
            ),
            "record_hypothesis",
            {
                "hypothesis_id": record.hypothesis_id,
                "state": record.state,
            },
        )

    return StructuredTool.from_function(
        func=record_hypothesis,
        name="record_hypothesis",
        description=_RECORD_DESCRIPTION,
    )


def build_continue_investigation_tool(
    coordinator: InvestigationCoordinator,
) -> BaseTool:
    def continue_investigation(
        hypothesis_ids: list[str],
        unresolved_question: str,
        expected_evidence: str,
        tool_name: str,
        target: str,
        reason: str,
        runtime: ToolRuntime,
    ) -> ToolMessage:
        task_id = _task_id(runtime)
        call_id = str(runtime.tool_call_id or "").strip()
        if not task_id:
            return _error_message(
                "unknown",
                call_id or "continue-investigation",
                "continue_validation_error",
                "申请调查许可失败：运行配置缺少 thread_id",
                "continue_investigation",
            )
        if not call_id:
            return _error_message(
                task_id,
                "missing-call-id",
                "continue_validation_error",
                "申请调查许可失败：缺少 tool_call_id",
                "continue_investigation",
            )
        try:
            permit = coordinator.grant_investigation_permit(
                task_id,
                ContinueInvestigationInput(
                    hypothesis_ids=hypothesis_ids,
                    unresolved_question=unresolved_question,
                    expected_evidence=expected_evidence,
                    tool_name=tool_name,
                    target=target,
                    reason=reason,
                ),
            )
        except (ValidationError, ValueError) as exc:
            return _error_message(
                task_id,
                call_id,
                "continue_validation_error",
                f"申请调查许可失败：{_safe_validation_error(exc)}",
                "continue_investigation",
            )
        return _success_message(
            task_id,
            call_id,
            "continue_permit_granted",
            "一次性调查许可已发放",
            "continue_investigation",
            {
                "permit_id": permit.permit_id,
                "tool_name": permit.tool_name,
                "target_hash": permit.target_hash,
                "granted_in_generation": permit.granted_in_generation,
            },
        )

    return StructuredTool.from_function(
        func=continue_investigation,
        name="continue_investigation",
        description=_CONTINUE_DESCRIPTION,
    )


def _task_id(runtime: ToolRuntime) -> str:
    return str(
        runtime.config.get("configurable", {}).get("thread_id", "")
    ).strip()


def _error_message(
    task_id: str,
    call_id: str,
    result_type: str,
    content: str,
    name: str,
) -> ToolMessage:
    return ToolMessage(
        content=_bounded(content),
        name=name,
        tool_call_id=call_id,
        status="error",
        id=stable_generated_message_id(task_id, call_id, result_type),
    )


def _success_message(
    task_id: str,
    call_id: str,
    result_type: str,
    content: str,
    name: str,
    artifact: dict[str, object],
) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=name,
        tool_call_id=call_id,
        status="success",
        artifact=artifact,
        id=stable_generated_message_id(task_id, call_id, result_type),
    )


def _safe_validation_error(exc: ValidationError | ValueError) -> str:
    if isinstance(exc, ValidationError):
        items = [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_input=False, include_url=False)
        ]
        return _bounded("; ".join(items) or "输入不符合结构化约束")
    return _bounded(str(exc))


def _bounded(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"
