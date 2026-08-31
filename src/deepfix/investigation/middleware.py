from __future__ import annotations

import hashlib
import os
from collections.abc import Awaitable, Callable, Mapping
from html import escape
from pathlib import Path
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from deepfix.compaction.identity import (
    stable_generated_message_id,
)
from deepfix.compaction.models import ArtifactReference
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
from deepfix.operations import (
    NewOperationEntry,
    OperationJournalEntry,
    OperationJournalStore,
    OperationKind,
    OperationStateSnapshot,
    OperationStatus,
)
from deepfix.prompting import model_request_task_id
from deepfix.workspace import WorkspacePathPolicy, compute_code_state_hash

_PLATFORM_NAME = os.name
_SIDE_EFFECT_KINDS = {
    "write_file": OperationKind.FILE_WRITE,
    "edit_file": OperationKind.FILE_EDIT,
    "delete": OperationKind.FILE_DELETE,
    "execute": OperationKind.COMMAND,
}


class InvestigationMiddleware(AgentMiddleware):
    def __init__(
        self,
        coordinator: InvestigationCoordinator,
        receipts: ToolExecutionReceiptStore,
        capabilities: Mapping[str, InvestigationCapability],
        operation_journal: OperationJournalStore | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.receipts = receipts
        self.capabilities = dict(capabilities)
        self.operation_journal = operation_journal
        if operation_journal is not None and receipts.repository is None:
            receipts.repository = operation_journal.repository

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
        task_id, receipt, correction, operation = self._prepare_tool(request)
        if correction is not None:
            result = correction
        elif receipt is None:
            result = _diagnostic_artifact_redirect(task_id, request)
            if result is None:
                if operation is not None:
                    self.operation_journal.mark_started(operation.operation_id)
                result = handler(request)
        else:
            result = receipt.tool_message
        return self._finish_tool(task_id, request, result, receipt, operation)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        task_id, receipt, correction, operation = self._prepare_tool(request)
        if correction is not None:
            result = correction
        elif receipt is None:
            result = _diagnostic_artifact_redirect(task_id, request)
            if result is None:
                if operation is not None:
                    self.operation_journal.mark_started(operation.operation_id)
                result = await handler(request)
        else:
            result = receipt.tool_message
        return self._finish_tool(task_id, request, result, receipt, operation)

    def _model_request(self, request: ModelRequest) -> ModelRequest:
        task_id = model_request_task_id(request)
        if not task_id:
            return request
        state = self.coordinator.state(task_id)
        block = render_investigation_state(state)
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{block}" if original else block
        return request.override(
            system_message=SystemMessage(content=content),
        )

    def _prepare_tool(
        self,
        request: ToolCallRequest,
    ) -> tuple[
        str,
        ToolExecutionReceipt | None,
        ToolMessage | None,
        OperationJournalEntry | None,
    ]:
        task_id = _runtime_task_id(request)
        call_id = str(request.tool_call.get("id", "")).strip()
        name = str(request.tool_call.get("name", "")).strip()
        if not task_id or not call_id or not name:
            raise ValueError("Tool hook 缺少 task_id、tool_call_id 或 tool name")
        operation = self._load_operation(task_id, call_id, name)
        if operation is not None and operation.call_hash != tool_call_hash(
            task_id, request.tool_call
        ):
            raise self._state_error(
                task_id,
                "operation_call_mismatch",
                call_id,
                "pause_and_inspect_operation_journal",
            )
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
        if receipt is None and operation is not None:
            receipt = self.operation_journal.load_receipt(task_id, call_id)
        if receipt is not None:
            expected = tool_call_hash(task_id, request.tool_call)
            if receipt.call_hash != expected:
                raise self._state_error(
                    task_id,
                    "tool_receipt_call_mismatch",
                    call_id,
                    "pause_and_inspect_tool_receipt",
                )
            return task_id, receipt, None, operation
        if operation is not None and operation.status is OperationStatus.COMMITTED:
            raise self._state_error(
                task_id,
                "committed_operation_receipt_missing",
                call_id,
                "restore_receipt_from_internal_artifact_without_reexecution",
            )
        if operation is not None and operation.status in {
            OperationStatus.STARTED,
            OperationStatus.OBSERVED,
            OperationStatus.UNKNOWN,
        }:
            raise self._state_error(
                task_id,
                "operation_reconciliation_required",
                call_id,
                "inspect_workspace_and_reconcile_operation_without_reexecution",
            )

        experiment_allowed = _experiment_allowed_capabilities(request)
        capability = self.capabilities.get(name)
        if (
            experiment_allowed is not None
            and capability is not None
            and capability not in experiment_allowed
        ):
            return (
                task_id,
                None,
                ToolMessage(
                    content=(
                        f"Tool '{name}' is not allowed by this experiment; "
                        "return this constraint to the Experiment Planner."
                    ),
                    name=name,
                    tool_call_id=call_id,
                    status="error",
                ),
                None,
            )

        shell_correction = _windows_shell_correction(
            task_id,
            request,
            self.coordinator.project_python(task_id),
        )
        if shell_correction is not None:
            return task_id, None, shell_correction, None

        authorization = self.coordinator.authorize_tool(
            task_id,
            name,
            _tool_arguments(request.tool_call),
            tool_call_id=call_id,
        )
        if authorization.correction_required:
            if authorization.correction_kind == "duplicate_execute":
                return task_id, None, _duplicate_execute_message(task_id, call_id), None
            if authorization.correction_kind == "duplicate_hypothesis":
                return (
                    task_id,
                    None,
                    _duplicate_hypothesis_message(
                        task_id,
                        call_id,
                        authorization.correction_ref_id or "unknown",
                    ),
                    None,
                )
            return task_id, None, _decision_correction_message(task_id, call_id), None
        if operation is None:
            operation = self._prepare_operation(task_id, request)
        return task_id, None, None, operation

    def _finish_tool(
        self,
        task_id: str,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
        receipt: ToolExecutionReceipt | None,
        operation: OperationJournalEntry | None,
    ) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage):
            if operation is not None:
                raise self._state_error(
                    task_id,
                    "operation_result_not_durable",
                    str(request.tool_call.get("id", "")),
                    "pause_and_reconcile_operation",
                )
            return result
        call_id = str(request.tool_call.get("id", "")).strip()
        if str(result.tool_call_id) != call_id:
            raise self._state_error(
                task_id,
                "tool_result_call_id_mismatch",
                call_id,
                "pause_and_inspect_tool_result",
            )
        artifact_references: list[ArtifactReference] = (
            self.operation_journal.artifact_references(operation.operation_id)
            if operation is not None
            else []
        )
        if receipt is None:
            receipt = receipt_from_result(task_id, request.tool_call, result)
            if operation is not None and operation.operation_kind is OperationKind.COMMAND:
                try:
                    artifact_references.append(
                        self.receipts.save_result_artifact_reference(
                            task_id,
                            call_id,
                            str(request.tool_call.get("name", "")),
                            result,
                        )
                    )
                except Exception as exc:
                    raise self._state_error(
                        task_id,
                        "tool_result_artifact_persistence_failed",
                        call_id,
                        "do_not_retry_tool_without_operation_reconciliation",
                        cause=exc,
                    ) from exc
            if operation is None:
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
        if operation is not None and operation.status is OperationStatus.COMMITTED:
            return result
        if operation is not None:
            post_state = self._operation_snapshot(task_id, request, result=result)
            try:
                operation = self.operation_journal.observe_with_receipt(
                    operation.operation_id,
                    post_state=post_state,
                    receipt=receipt,
                    artifact_references=artifact_references,
                )
            except Exception as exc:
                raise self._state_error(
                    task_id,
                    "tool_receipt_persistence_failed",
                    call_id,
                    "do_not_retry_tool_without_operation_reconciliation",
                    cause=exc,
                ) from exc
        self.coordinator.record_tool_result(task_id, request.tool_call, result)
        if operation is not None:
            self.operation_journal.commit(operation.operation_id)
        return result

    def _load_operation(
        self,
        task_id: str,
        call_id: str,
        name: str,
    ) -> OperationJournalEntry | None:
        if self.operation_journal is None or name not in _SIDE_EFFECT_KINDS:
            return None
        return self.operation_journal.load(stable_investigation_id("operation", task_id, call_id))

    def _prepare_operation(
        self,
        task_id: str,
        request: ToolCallRequest,
    ) -> OperationJournalEntry | None:
        name = str(request.tool_call.get("name", "")).strip()
        if self.operation_journal is None or name not in _SIDE_EFFECT_KINDS:
            return None
        call_id = str(request.tool_call.get("id", "")).strip()
        task = self.coordinator.tasks.get(task_id)
        if not task.workspace_root or not task.workspace_baseline_id:
            raise self._state_error(
                task_id,
                "operation_workspace_identity_missing",
                call_id,
                "pause_and_restore_workspace_identity",
            )
        pre_state = self._operation_snapshot(task_id, request)
        return self.operation_journal.prepare(
            NewOperationEntry(
                operation_id=stable_investigation_id("operation", task_id, call_id),
                task_id=task_id,
                experiment_id=stable_investigation_id("experiment", task_id, call_id),
                tool_call_id=call_id,
                operation_kind=_SIDE_EFFECT_KINDS[name],
                call_hash=tool_call_hash(task_id, request.tool_call),
                workspace_baseline_id=task.workspace_baseline_id,
                pre_state=pre_state,
                expected_post_state=self._expected_post_state(task_id, request, pre_state),
            )
        )

    def _expected_post_state(
        self,
        task_id: str,
        request: ToolCallRequest,
        pre_state: OperationStateSnapshot,
    ) -> OperationStateSnapshot | None:
        name = str(request.tool_call.get("name", "")).strip()
        if name == "execute":
            return None
        arguments = _tool_arguments(request.tool_call)
        if name == "delete":
            return pre_state.model_copy(
                update={"target_exists": False, "file_hash": None, "code_state_hash": None}
            )
        content: str | None = None
        if name == "write_file":
            value = arguments.get("content")
            if isinstance(value, str):
                content = value
        elif name == "edit_file" and pre_state.target_path:
            task = self.coordinator.tasks.get(task_id)
            target = Path(task.workspace_root or task.project_root) / pre_state.target_path
            old = arguments.get("old_string")
            new = arguments.get("new_string")
            if target.is_file() and isinstance(old, str) and isinstance(new, str):
                original = target.read_text(encoding="utf-8")
                if old in original:
                    content = original.replace(old, new)
        if content is None:
            return None
        return pre_state.model_copy(
            update={
                "target_exists": True,
                "file_hash": hashlib.sha256(
                    content.replace("\n", os.linesep).encode("utf-8")
                ).hexdigest(),
                "code_state_hash": None,
            }
        )

    def _operation_snapshot(
        self,
        task_id: str,
        request: ToolCallRequest,
        *,
        result: ToolMessage | None = None,
    ) -> OperationStateSnapshot:
        task = self.coordinator.tasks.get(task_id)
        workspace = Path(task.workspace_root or task.project_root)
        name = str(request.tool_call.get("name", "")).strip()
        arguments = _tool_arguments(request.tool_call)
        if name == "execute":
            artifact = result.artifact if result is not None else None
            exit_code = artifact.get("exit_code") if isinstance(artifact, Mapping) else None
            return OperationStateSnapshot(
                code_state_hash=compute_code_state_hash(workspace),
                command_hash=tool_call_hash(task_id, request.tool_call),
                exit_code=exit_code if isinstance(exit_code, int) else None,
            )
        raw_path = str(arguments.get("file_path", arguments.get("path", ""))).strip()
        relative_path = raw_path.replace("\\", "/").lstrip("/")
        target = WorkspacePathPolicy(workspace).resolve_allowed(relative_path)
        exists = target.is_file()
        return OperationStateSnapshot(
            target_path=target.relative_to(workspace.resolve()).as_posix(),
            target_exists=exists,
            file_hash=_sha256_file(target) if exists else None,
            code_state_hash=compute_code_state_hash(workspace),
        )

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
) -> str:
    lines = [
        "<deepfix_investigation_state>",
        f"<progress_generation>{state.progress_generation}</progress_generation>",
        "<checked_paths>",
    ]
    lines.extend(f"<path>{escape(item.path[:500])}</path>" for item in state.checked_files[-8:])
    lines.extend(("</checked_paths>", "<recent_tool_signatures>"))
    lines.extend(
        f"<signature>{escape(item[:160])}</signature>" for item in state.recent_tool_signatures[-8:]
    )
    lines.append("</recent_tool_signatures>")
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
    lines.append("</deepfix_investigation_state>")
    return "\n".join(lines)


def _runtime_task_id(request: ToolCallRequest) -> str:
    return str(request.runtime.config.get("configurable", {}).get("thread_id", "")).strip()


def _experiment_allowed_capabilities(
    request: ToolCallRequest,
) -> set[InvestigationCapability] | None:
    raw = request.runtime.config.get("configurable", {}).get("allowed_capabilities")
    if raw is None:
        return None
    return {InvestigationCapability(item) for item in raw}


def _tool_arguments(tool_call: Mapping[str, Any]) -> Mapping[str, object]:
    value = tool_call.get("args", {})
    return value if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic_artifact_redirect(
    task_id: str,
    request: ToolCallRequest,
) -> ToolMessage | None:
    name = str(request.tool_call.get("name", "")).strip()
    arguments = _tool_arguments(request.tool_call)
    path = str(arguments.get("file_path", "")).strip().replace("\\", "/")
    normalized = f"/{path.lstrip('/')}"
    if name != "read_file" or not normalized.startswith("/.deepfix-artifacts/large_tool_results/"):
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
            "若证据不足，先记录 candidate，并更新 Todo 选择一次定向调查。"
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
            "不要再次创建该假设；若需要验证它，请更新 Todo 并引用这个 hypothesis_id。"
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
        marker in normalized for marker in ("| head", "| tail", "| grep", "| sed")
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
