from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from html import escape
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_generated_message_id,
)
from deepfix.compaction.work_units import partition_work_units
from deepfix.investigation.classification import is_pytest_verification
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import InvestigationStateError
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    InvestigationCapability,
    InvestigationState,
)
from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    ToolExecutionReceiptStore,
    receipt_from_result,
    tool_call_hash,
)
from deepfix.prompting import model_request_task_id

_PLATFORM_NAME = os.name


class InvestigationMiddleware(AgentMiddleware):
    def __init__(
        self,
        coordinator: InvestigationCoordinator,
        receipts: ToolExecutionReceiptStore,
        capabilities: Mapping[str, InvestigationCapability],
    ) -> None:
        self.coordinator = coordinator
        self.receipts = receipts
        self.capabilities = dict(capabilities)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._model_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._model_request(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        task_id, receipt, correction = self._prepare_tool(request)
        if correction is not None:
            result = correction
        elif receipt is None:
            result = _diagnostic_artifact_redirect(task_id, request)
            if result is None:
                result = handler(request)
        else:
            result = receipt.tool_message
        return self._finish_tool(task_id, request, result, receipt)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        task_id, receipt, correction = self._prepare_tool(request)
        if correction is not None:
            result = correction
        elif receipt is None:
            result = _diagnostic_artifact_redirect(task_id, request)
            if result is None:
                result = await handler(request)
        else:
            result = receipt.tool_message
        return self._finish_tool(task_id, request, result, receipt)

    def _model_request(self, request: ModelRequest) -> ModelRequest:
        task_id = model_request_task_id(request)
        if not task_id:
            return request
        state = self.coordinator.state(task_id)
        allowed = self.coordinator.allowed_tool_names(state, self.capabilities)
        tools = [
            tool
            for tool in request.tools or []
            if str(getattr(tool, "name", "")) in allowed
        ]
        block = render_investigation_state(
            state,
            _available_work_unit_refs(task_id, request.messages),
        )
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{block}" if original else block
        return request.override(
            tools=tools,
            system_message=SystemMessage(content=content),
        )

    def _prepare_tool(
        self,
        request: ToolCallRequest,
    ) -> tuple[str, ToolExecutionReceipt | None, ToolMessage | None]:
        task_id = _runtime_task_id(request)
        call_id = str(request.tool_call.get("id", "")).strip()
        name = str(request.tool_call.get("name", "")).strip()
        if not task_id or not call_id or not name:
            raise ValueError("Tool hook 缺少 task_id、tool_call_id 或 tool name")
        try:
            receipt = self.receipts.load(task_id, call_id)
        except Exception as exc:
            raise self._state_error(
                task_id,
                "tool_receipt_read_failed",
                call_id,
                "pause_before_tool_reexecution",
                cause=exc,
            ) from exc
        if receipt is not None:
            expected = tool_call_hash(task_id, request.tool_call)
            if receipt.call_hash != expected:
                raise self._state_error(
                    task_id,
                    "tool_receipt_call_mismatch",
                    call_id,
                    "pause_and_inspect_tool_receipt",
                )
            return task_id, receipt, None

        shell_correction = _windows_shell_correction(
            task_id,
            request,
            self.coordinator.project_python(task_id),
        )
        if shell_correction is not None:
            return task_id, None, shell_correction

        authorization = self.coordinator.authorize_tool(
            task_id,
            name,
            _tool_arguments(request.tool_call),
            tool_call_id=call_id,
        )
        if authorization.correction_required:
            if authorization.correction_kind == "duplicate_execute":
                return task_id, None, _duplicate_execute_message(task_id, call_id)
            if authorization.correction_kind == "duplicate_hypothesis":
                return task_id, None, _duplicate_hypothesis_message(
                    task_id,
                    call_id,
                    authorization.correction_ref_id or "unknown",
                )
            return task_id, None, _decision_correction_message(task_id, call_id)
        state = self.coordinator.state(task_id)
        allowed = self.coordinator.allowed_tool_names(state, self.capabilities)
        if name not in allowed and authorization.permit_id is None:
            raise self._state_error(
                task_id,
                "tool_not_allowed_in_agent_phase",
                call_id,
                "return_to_a_phase_that_allows_the_tool",
            )
        return task_id, None, None

    def _finish_tool(
        self,
        task_id: str,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
        receipt: ToolExecutionReceipt | None,
    ) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage):
            return result
        call_id = str(request.tool_call.get("id", "")).strip()
        if str(result.tool_call_id) != call_id:
            raise self._state_error(
                task_id,
                "tool_result_call_id_mismatch",
                call_id,
                "pause_and_inspect_tool_result",
            )
        if receipt is None:
            receipt = receipt_from_result(task_id, request.tool_call, result)
            try:
                self.receipts.save(receipt)
            except Exception as exc:
                raise self._state_error(
                    task_id,
                    "tool_receipt_persistence_failed",
                    call_id,
                    "do_not_retry_tool_without_manual_recovery",
                    cause=exc,
                ) from exc
        self.coordinator.record_tool_result(task_id, request.tool_call, result)
        return result

    def _state_error(
        self,
        task_id: str,
        error_code: str,
        tool_call_id: str,
        recovery_action: str,
        *,
        cause: Exception | None = None,
    ) -> InvestigationStateError:
        error_type = None
        error_detail = None
        error_fingerprint = None
        if cause is not None:
            error_type = type(cause).__name__[:120]
            error_detail = " ".join(str(cause).split())[:500] or error_type
            error_fingerprint = stable_investigation_id(
                "error",
                error_type,
                error_detail,
            )
        return InvestigationStateError(
            self.coordinator.recovery(
                task_id,
                error_code,
                tool_call_id=tool_call_id,
                checkpoint_available=True,
                recovery_action=recovery_action,
                error_type=error_type,
                error_detail=error_detail,
                error_fingerprint=error_fingerprint,
            )
        )


def render_investigation_state(
    state: InvestigationState,
    work_unit_refs: Sequence[tuple[str, str, str]] = (),
) -> str:
    lines = [
        "<deepfix_investigation_state>",
        f"<phase>{state.agent_phase.value}</phase>",
        f"<progress_generation>{state.progress_generation}</progress_generation>",
        f"<stagnation_level>{state.stagnation_level}</stagnation_level>",
        f"<reevaluation_required>{str(state.reevaluation_required).lower()}</reevaluation_required>",
        f"<no_progress_count>{state.no_progress_count}</no_progress_count>",
        f"<exploratory_without_progress>{state.exploratory_without_progress}</exploratory_without_progress>",
        "<checked_paths>",
    ]
    lines.extend(
        f"<path>{escape(item.path[:500])}</path>"
        for item in state.checked_files[-8:]
    )
    lines.extend(("</checked_paths>", "<recent_tool_signatures>"))
    lines.extend(
        f"<signature>{escape(item[:160])}</signature>"
        for item in state.recent_tool_signatures[-8:]
    )
    lines.append("</recent_tool_signatures>")
    if state.repair_reevaluation_required:
        lines.extend(
            (
                "<repair_reevaluation_checkpoint>",
                "修改后验证失败，原 supported 假设和修复计划已被测试反证。",
                "先用 record_hypothesis 排除或修正旧假设；证据不足时记录 candidate，",
                "再用 continue_investigation 申请一次定向调查。不要直接重复测试或继续修改。",
                "</repair_reevaluation_checkpoint>",
            )
        )
    elif state.diagnostic_decision_required:
        lines.extend(
            (
                "<diagnostic_decision_checkpoint>",
                "当前必须进行诊断决策。",
                "证据充分时立即调用 record_hypothesis，不要继续读取相邻 traceback 片段。",
                "证据不足时先记录 candidate 假设，再调用 continue_investigation 说明未解决问题和预期证据。",
                "</diagnostic_decision_checkpoint>",
            )
        )
    if state.memory_save_blocked_generation == state.progress_generation:
        lines.extend(
            (
                "<memory_save_checkpoint>",
                "本进展代次内不再调用 save_progress；确定性证据仍由系统保留。",
                "若任务已经完成，请直接返回 RepairOutcome；否则继续产生新的有效证据。",
                "</memory_save_checkpoint>",
            )
        )
    elif state.memory_saved_generation == state.progress_generation:
        lines.extend(
            (
                "<memory_save_checkpoint>",
                "当前进展代次的 Working Memory 已保存，不要重复调用 save_progress。",
                "完成复核后直接返回 RepairOutcome。",
                "</memory_save_checkpoint>",
            )
        )
    lines.append("<current_hypotheses>")
    lines.extend(
        (
            f'<hypothesis id="{escape(item.hypothesis_id, quote=True)}" '
            f'state="{escape(item.state, quote=True)}">'
            f"{escape(item.statement[:500])}</hypothesis>"
        )
        for item in state.hypotheses[-8:]
    )
    lines.append("</current_hypotheses>")
    lines.extend(
        (
            "<available_work_unit_refs>",
            "save_progress 中 kind=work_unit 的 ref_id 必须使用下列稳定 ID，不能使用文件路径。",
        )
    )
    lines.extend(
        (
            f'<work_unit ref_id="{escape(ref_id, quote=True)}" '
            f'path="{escape(path, quote=True)}" '
            f'purpose="{escape(purpose[:300], quote=True)}" />'
        )
        for ref_id, path, purpose in work_unit_refs[-8:]
    )
    lines.append("</available_work_unit_refs>")
    lines.append("</deepfix_investigation_state>")
    return "\n".join(lines)


def _available_work_unit_refs(
    task_id: str,
    messages: Sequence[AnyMessage],
) -> list[tuple[str, str, str]]:
    identified = ensure_message_ids(task_id, messages).messages
    units = partition_work_units(identified, set()).units
    refs: list[tuple[str, str, str]] = []
    for unit in units:
        paths: list[str] = []
        for message in identified[unit.start_index : unit.end_index + 1]:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                arguments = call.get("args", {})
                if not isinstance(arguments, Mapping):
                    continue
                path = str(
                    arguments.get("file_path", arguments.get("path", ""))
                ).strip()
                if path and path not in paths:
                    paths.append(path)
        refs.append((unit.unit_id, ", ".join(paths), unit.purpose))
    return refs


def _runtime_task_id(request: ToolCallRequest) -> str:
    return str(
        request.runtime.config.get("configurable", {}).get("thread_id", "")
    ).strip()


def _tool_arguments(tool_call: Mapping[str, Any]) -> Mapping[str, object]:
    value = tool_call.get("args", {})
    return value if isinstance(value, Mapping) else {}


def _diagnostic_artifact_redirect(
    task_id: str,
    request: ToolCallRequest,
) -> ToolMessage | None:
    name = str(request.tool_call.get("name", "")).strip()
    arguments = _tool_arguments(request.tool_call)
    path = str(arguments.get("file_path", "")).strip().replace("\\", "/")
    normalized = f"/{path.lstrip('/')}"
    if name != "read_file" or not normalized.startswith(
        "/.deepfix-artifacts/large_tool_results/"
    ):
        return None
    call_id = str(request.tool_call.get("id", "")).strip()
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "diagnostic_artifact_redirect",
        ),
        content=(
            "该路径是已卸载的诊断结果，不能使用 read_file 顺序翻页。"
            "请先调用 search_diagnostic_artifacts 搜索异常类型、末尾错误或关键栈帧，"
            "再使用返回的 artifact_id 调用 read_diagnostic_artifact 读取必要片段。"
        ),
        tool_call_id=call_id,
        name="read_file",
        status="error",
        artifact={
            "result_type": "diagnostic_artifact_redirect",
            "error_code": "use_diagnostic_artifact_tools",
            "operation": "read_file",
        },
    )


def _decision_correction_message(task_id: str, call_id: str) -> ToolMessage:
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "diagnostic_decision_correction",
        ),
        content=(
            "当前处于诊断决策检查点，不能继续读取或执行。"
            "请根据现有证据调用 record_hypothesis；"
            "若证据不足，先记录 candidate，再调用 continue_investigation 申请一次定向调查。"
        ),
        tool_call_id=call_id,
        status="error",
        artifact={
            "result_type": "diagnostic_decision_correction",
            "error_code": "diagnostic_decision_required",
        },
    )


def _duplicate_execute_message(task_id: str, call_id: str) -> ToolMessage:
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "duplicate_execute_correction",
        ),
        content=(
            "该 execute 命令已在当前进展代次执行，重复执行不会产生新证据。"
            "请分析已有结果、更新假设，或改用能区分候选根因的定向命令。"
        ),
        tool_call_id=call_id,
        name="execute",
        status="error",
        artifact={
            "result_type": "duplicate_execute_correction",
            "error_code": "duplicate_execute_in_progress_generation",
        },
    )


def _duplicate_hypothesis_message(
    task_id: str,
    call_id: str,
    hypothesis_id: str,
) -> ToolMessage:
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "duplicate_hypothesis_correction",
        ),
        content=(
            f"相同 candidate 已存在：hypothesis_id={hypothesis_id}。"
            "不要再次创建该假设；若需要验证它，请调用 continue_investigation "
            "并引用这个 hypothesis_id。"
        ),
        tool_call_id=call_id,
        name="record_hypothesis",
        status="error",
        artifact={
            "result_type": "duplicate_hypothesis_correction",
            "error_code": "duplicate_hypothesis_candidate",
            "hypothesis_id": hypothesis_id,
        },
    )


def _windows_shell_correction(
    task_id: str,
    request: ToolCallRequest,
    project_python: str,
) -> ToolMessage | None:
    if _PLATFORM_NAME != "nt":
        return None
    if str(request.tool_call.get("name", "")).strip() != "execute":
        return None
    command = str(_tool_arguments(request.tool_call).get("command", ""))
    normalized = command.lower()
    has_unix_pipeline = any(
        marker in normalized
        for marker in ("| head", "| tail", "| grep", "| sed")
    )
    if not has_unix_pipeline:
        clean_pytest = _pytest_prefix_before_shell_control(command, project_python)
        if clean_pytest:
            call_id = str(request.tool_call.get("id", "")).strip()
            return ToolMessage(
                id=stable_generated_message_id(
                    task_id,
                    call_id,
                    "pytest_shell_correction",
                ),
                content=(
                    "pytest 验证命令不能附加 shell 重定向、管道或命令连接符。"
                    f"请直接执行：{clean_pytest}"
                ),
                tool_call_id=call_id,
                name="execute",
                status="error",
                artifact={
                    "result_type": "pytest_shell_correction",
                    "error_code": "pytest_shell_composition_not_allowed",
                    "suggested_command": clean_pytest,
                },
            )
        return None
    call_id = str(request.tool_call.get("id", "")).strip()
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "windows_shell_correction",
        ),
        content=(
            "当前项目在 Windows Shell 中运行，head/tail/grep/sed 管道不可用。"
            "pytest 请直接使用 -x、--tb=short、-q 等参数缩小输出，例如 "
            "python -m pytest -q -x --tb=short；大型结果会由系统自动卸载。"
        ),
        tool_call_id=call_id,
        name="execute",
        status="error",
        artifact={
            "result_type": "windows_shell_correction",
            "error_code": "nonportable_windows_shell_pipeline",
        },
    )


def _pytest_prefix_before_shell_control(
    command: str,
    project_python: str,
) -> str | None:
    positions = [
        index
        for marker in ("2>&1", "1>&2", "|", "&&", ";", ">", "<")
        if (index := command.find(marker)) >= 0
    ]
    if not positions:
        return None
    prefix = command[: min(positions)].strip()
    if not prefix or not is_pytest_verification(prefix, project_python):
        return None
    return prefix
