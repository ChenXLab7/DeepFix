from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from html import escape
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import InvestigationStateError
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
        task_id, receipt = self._prepare_tool(request)
        if receipt is None:
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
        task_id, receipt = self._prepare_tool(request)
        if receipt is None:
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
        block = render_investigation_state(state)
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{block}" if original else block
        return request.override(
            tools=tools,
            system_message=SystemMessage(content=content),
        )

    def _prepare_tool(
        self,
        request: ToolCallRequest,
    ) -> tuple[str, ToolExecutionReceipt | None]:
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
            return task_id, receipt

        authorization = self.coordinator.authorize_tool(
            task_id,
            name,
            _tool_arguments(request.tool_call),
        )
        state = self.coordinator.state(task_id)
        allowed = self.coordinator.allowed_tool_names(state, self.capabilities)
        if name not in allowed and authorization.permit_id is None:
            raise self._state_error(
                task_id,
                "tool_not_allowed_in_agent_phase",
                call_id,
                "return_to_a_phase_that_allows_the_tool",
            )
        return task_id, None

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
                ) from exc
        self.coordinator.record_tool_result(task_id, request.tool_call, result)
        return result

    def _state_error(
        self,
        task_id: str,
        error_code: str,
        tool_call_id: str,
        recovery_action: str,
    ) -> InvestigationStateError:
        return InvestigationStateError(
            self.coordinator.recovery(
                task_id,
                error_code,
                tool_call_id=tool_call_id,
                checkpoint_available=True,
                recovery_action=recovery_action,
            )
        )


def render_investigation_state(state: InvestigationState) -> str:
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
    lines.extend(("</recent_tool_signatures>", "</deepfix_investigation_state>"))
    return "\n".join(lines)


def _runtime_task_id(request: ToolCallRequest) -> str:
    return str(
        request.runtime.config.get("configurable", {}).get("thread_id", "")
    ).strip()


def _tool_arguments(tool_call: Mapping[str, Any]) -> Mapping[str, object]:
    value = tool_call.get("args", {})
    return value if isinstance(value, Mapping) else {}
