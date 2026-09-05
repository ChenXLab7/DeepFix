from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_DEBUG_LOG_LOCK = threading.Lock()


def append_debug_record(
    log_path: str | Path,
    record: dict[str, Any],
) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _DEBUG_LOG_LOCK, path.open("a", encoding="utf-8") as file:
        file.write(line)


class LLMTraceMiddleware(AgentMiddleware):
    def __init__(
        self,
        log_path: str | Path,
        role: str = "main",
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.role = role
        self._counter = 0
        self._lock = threading.Lock()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        call_id = self._next_call_id()
        started_at = time.perf_counter()

        messages = list(request.messages or [])
        system_text = request.system_message.text if request.system_message else ""

        task_id = self._task_id(request)
        trigger = self._trigger(messages)

        serialized_messages = [
            self._serialize_message(message)
            for message in messages
        ]

        messages_chars = sum(
            len(str(message.get("content", "")))
            for message in serialized_messages
        )

        fingerprint = self._fingerprint(
            system_text,
            serialized_messages,
        )

        request_record = {
            "event": "request",
            "call_id": call_id,
            "model_role": self.role,
            "task_id": task_id,
            "trigger": trigger,
            "model": self._model_name(request),
            "system_prompt_chars": len(system_text),
            "messages_chars": messages_chars,
            "message_count": len(messages),
            "total_context_chars": len(system_text) + messages_chars,
            "prompt_fingerprint": fingerprint,

            # middleware 全部处理后的最终 system prompt
            "system_prompt": system_text,

            # middleware 全部处理后的最终 messages
            "messages": serialized_messages,

            "tools": [
                getattr(tool, "name", type(tool).__name__)
                for tool in request.tools
            ],

            "state_keys": (
                list(request.state.keys())
                if isinstance(request.state, dict)
                else []
            ),

            # messages 已经单独完整保存，不重复保存
            "state_without_messages": self._state_without_messages(
                request.state
            ),
        }

        self._write(request_record)

        print()
        print("=" * 90)
        print(
            f"[LLM #{call_id}] "
            f"task={task_id or '-'} "
            f"trigger={trigger}"
        )
        print(
            f"context={request_record['total_context_chars']:,} chars | "
            f"messages={len(messages)} | "
            f"fingerprint={fingerprint}"
        )
        print(">>> calling LLM")

        try:
            response = handler(request)
        except Exception as exc:
            duration = time.perf_counter() - started_at

            self._write(
                {
                    "event": "error",
                    "call_id": call_id,
                    "model_role": self.role,
                    "task_id": task_id,
                    "duration_seconds": duration,
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )

            print(
                f"<<< LLM #{call_id} ERROR: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        duration = time.perf_counter() - started_at

        result_messages = getattr(response, "result", None)

        if result_messages is None:
            result_messages = [response]

        serialized_response = [
            self._serialize_message(message)
            for message in result_messages
            if isinstance(message, BaseMessage)
        ]

        tool_calls: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for message in result_messages:
            if not isinstance(message, AIMessage):
                continue

            tool_calls.extend(message.tool_calls)

            usage = getattr(message, "usage_metadata", None)

            if isinstance(usage, dict):
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
                total_tokens += int(usage.get("total_tokens", 0) or 0)

        response_record = {
            "event": "response",
            "call_id": call_id,
            "model_role": self.role,
            "task_id": task_id,
            "duration_seconds": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_messages": serialized_response,
            "tool_calls": self._safe(tool_calls),
            "structured_response": self._safe(
                getattr(response, "structured_response", None)
            ),
        }

        self._write(response_record)

        tool_names = [
            str(call.get("name", ""))
            for call in tool_calls
        ]

        print(
            f"<<< LLM #{call_id} "
            f"{duration:.2f}s | "
            f"tokens={input_tokens}/{output_tokens}/{total_tokens}"
        )
        print(
            f"decision={'tools=' + str(tool_names) if tool_names else 'no tool call'}"
        )
        print("=" * 90)

        return response

    def _next_call_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            append_debug_record(self.log_path, record)

    @staticmethod
    def _serialize_message(message: BaseMessage) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": type(message).__name__,
            "id": getattr(message, "id", None),
            "content": LLMTraceMiddleware._safe(
                getattr(message, "content", "")
            ),
        }

        if isinstance(message, AIMessage):
            result["tool_calls"] = LLMTraceMiddleware._safe(
                message.tool_calls
            )
            result["usage_metadata"] = LLMTraceMiddleware._safe(
                getattr(message, "usage_metadata", None)
            )
            result["response_metadata"] = LLMTraceMiddleware._safe(
                getattr(message, "response_metadata", None)
            )

        if isinstance(message, ToolMessage):
            result["tool_name"] = message.name
            result["tool_call_id"] = message.tool_call_id
            result["artifact"] = LLMTraceMiddleware._safe(
                getattr(message, "artifact", None)
            )

        return result

    @staticmethod
    def _trigger(messages: list[BaseMessage]) -> str:
        if not messages:
            return "no_messages"

        last = messages[-1]

        if isinstance(last, ToolMessage):
            return f"tool_result:{last.name or 'unknown'}"

        return type(last).__name__

    @staticmethod
    def _fingerprint(
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> str:
        payload = json.dumps(
            {
                "system": system_prompt,
                "messages": messages,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _task_id(request: ModelRequest) -> str:
        runtime = request.runtime
        execution_info = (
            runtime.execution_info
            if runtime is not None
            else None
        )

        if execution_info is None:
            return ""

        return str(execution_info.thread_id or "").strip()

    @staticmethod
    def _model_name(request: ModelRequest) -> str:
        model = request.model

        return str(
            getattr(
                model,
                "model_name",
                getattr(model, "model", type(model).__name__),
            )
        )

    @staticmethod
    def _state_without_messages(state: Any) -> Any:
        if not isinstance(state, dict):
            return LLMTraceMiddleware._safe(state)

        return {
            key: LLMTraceMiddleware._safe(value)
            for key, value in state.items()
            if key != "messages"
        }

    @staticmethod
    def _safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False, default=str)
            return value
        except Exception:  # noqa: BLE001 - tracing must not fail on unknown values
            return repr(value)
