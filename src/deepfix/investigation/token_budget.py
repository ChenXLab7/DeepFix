from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langgraph.config import get_config
from pydantic import Field

from deepfix.compaction.models import StrictModel
from deepfix.database import SQLiteDatabase
from deepfix.prompting import model_request_task_id
from deepfix.task_domain.repository import TaskRepository


class TokenBudgetExhausted(RuntimeError):
    pass


class TokenBudgetConflict(RuntimeError):
    pass


class TokenBalance(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    breached: bool = False


class TokenUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    estimated: bool = False


class TokenReservation(StrictModel):
    reservation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    reserved_input_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    status: Literal["reserved", "settled", "unknown"]
    actual_input_tokens: int | None = Field(default=None, ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)


class TokenBudgetStore:
    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        database: SQLiteDatabase | None = None,
    ) -> None:
        if database is not None and database_path is not None:
            raise ValueError("provide database_path or database, not both")
        if database is None and database_path is None:
            raise ValueError("database_path or database is required")
        self.database = database or SQLiteDatabase(database_path)
        self.database_path = self.database.path
        self.tasks = TaskRepository(self.database)

    def initialize(self, task_id: str, *, input_cap: int, output_cap: int) -> None:
        self.tasks.initialize_token_budget(
            task_id,
            input_cap=input_cap,
            output_cap=output_cap,
        )

    def reserve(
        self,
        task_id: str,
        call_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        return self.tasks.reserve_tokens(
            task_id,
            call_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        return self.tasks.settle_tokens(
            reservation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def charge_unknown(self, reservation_id: str) -> TokenReservation:
        return self.tasks.charge_unknown_tokens(reservation_id)

    def available(self, task_id: str) -> TokenBalance:
        return self.tasks.available_tokens(task_id)

    def usage(self, task_id: str) -> TokenUsage:
        return self.tasks.token_usage(task_id)


class TokenBudgetMiddleware(AgentMiddleware):
    def __init__(
        self,
        store: TokenBudgetStore,
        *,
        input_cap: int,
        output_cap: int,
        input_estimator: Callable[[ModelRequest], int] | None = None,
        requested_output_tokens: int = 4096,
        input_safety_margin: int = 32,
    ) -> None:
        if requested_output_tokens <= 0 or input_safety_margin < 0:
            raise ValueError("invalid model token reservation configuration")
        self.store = store
        self.input_cap = input_cap
        self.output_cap = output_cap
        self.input_estimator = input_estimator or _estimate_request_tokens
        self.requested_output_tokens = requested_output_tokens
        self.input_safety_margin = input_safety_margin
        self._counter = 0
        self._lock = threading.Lock()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        task_id = model_request_task_id(request)
        if not task_id:
            return handler(request)
        self.store.initialize(
            task_id,
            input_cap=self.input_cap,
            output_cap=self.output_cap,
        )
        estimated_input = self.input_estimator(request) + self.input_safety_margin
        requested_output = min(self.requested_output_tokens, self.output_cap)
        reservation = self.store.reserve(
            task_id,
            self._next_call_id(task_id),
            input_tokens=estimated_input,
            output_tokens=requested_output,
        )
        try:
            response = handler(request)
        except Exception:
            self.store.charge_unknown(reservation.reservation_id)
            raise
        usage = _response_usage(response)
        if usage is None:
            self.store.charge_unknown(reservation.reservation_id)
        else:
            self.store.settle(
                reservation.reservation_id,
                input_tokens=usage[0],
                output_tokens=usage[1],
            )
        return response

    def _next_call_id(self, task_id: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{task_id}-model-call-{self._counter}"


class TokenBudgetCallbackHandler(BaseCallbackHandler):
    raise_error = True

    def __init__(
        self,
        store: TokenBudgetStore,
        *,
        input_cap: int,
        output_cap: int,
        role: str,
        task_id_resolver: Callable[[], str] | None = None,
        requested_output_tokens: int = 4096,
        input_safety_margin: int = 32,
    ) -> None:
        self.store = store
        self.input_cap = input_cap
        self.output_cap = output_cap
        self.role = role.strip()
        self.task_id_resolver = task_id_resolver or _configured_task_id
        self.requested_output_tokens = requested_output_tokens
        self.input_safety_margin = input_safety_margin
        self._reservations: dict[str, str] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        task_id = self.task_id_resolver().strip()
        if not task_id:
            return
        self.store.initialize(
            task_id,
            input_cap=self.input_cap,
            output_cap=self.output_cap,
        )
        message_chars = sum(
            len(str(getattr(message, "content", ""))) for batch in messages for message in batch
        )
        estimated_input = max(1, math.ceil(message_chars / 4))
        estimated_input += self.input_safety_margin
        reservation = self.store.reserve(
            task_id,
            f"{self.role}-{run_id}",
            input_tokens=estimated_input,
            output_tokens=min(self.requested_output_tokens, self.output_cap),
        )
        with self._lock:
            self._reservations[str(run_id)] = reservation.reservation_id

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        reservation_id = self._pop_reservation(run_id)
        if reservation_id is None:
            return
        usage = _llm_result_usage(response)
        if usage is None:
            self.store.charge_unknown(reservation_id)
            return
        self.store.settle(
            reservation_id,
            input_tokens=usage[0],
            output_tokens=usage[1],
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del error, kwargs
        reservation_id = self._pop_reservation(run_id)
        if reservation_id is not None:
            self.store.charge_unknown(reservation_id)

    def _pop_reservation(self, run_id: UUID) -> str | None:
        with self._lock:
            return self._reservations.pop(str(run_id), None)


def _estimate_request_tokens(request: ModelRequest) -> int:
    system_text = request.system_message.text if request.system_message else ""
    content = (
        system_text
        + "\n"
        + "\n".join(str(getattr(message, "content", "")) for message in request.messages)
    )
    return max(1, math.ceil(len(content) / 4))


def _response_usage(response: ModelResponse) -> tuple[int, int] | None:
    result = getattr(response, "result", None)
    messages = result if isinstance(result, list) else [response]
    input_tokens = 0
    output_tokens = 0
    usage_seen = False
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        usage_seen = True
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return (input_tokens, output_tokens) if usage_seen else None


def _llm_result_usage(response: LLMResult) -> tuple[int, int] | None:
    input_tokens = 0
    output_tokens = 0
    usage_seen = False
    for generation_batch in response.generations:
        for generation in generation_batch:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if not isinstance(usage, dict):
                continue
            usage_seen = True
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
    return (input_tokens, output_tokens) if usage_seen else None


def _configured_task_id() -> str:
    try:
        config = get_config()
    except RuntimeError:
        return ""
    return str(config.get("configurable", {}).get("thread_id", "")).strip()
