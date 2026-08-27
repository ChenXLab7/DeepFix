from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import UTC, datetime
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
from deepfix.investigation.identity import stable_investigation_id
from deepfix.persistence import open_sqlite_connection
from deepfix.prompting import model_request_task_id


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
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def initialize(self, task_id: str, *, input_cap: int, output_cap: int) -> None:
        if input_cap <= 0 or output_cap <= 0:
            raise ValueError("token caps must be positive")
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT input_cap, output_cap FROM token_budgets
                    WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO token_budgets(
                            task_id, input_cap, output_cap,
                            settled_input, settled_output, breached, updated_at
                        ) VALUES (?, ?, ?, 0, 0, 0, ?)
                        """,
                        (task_id, input_cap, output_cap, _now()),
                    )
                    connection.commit()
                    return
                if (int(row[0]), int(row[1])) != (input_cap, output_cap):
                    raise TokenBudgetConflict("token budget caps cannot change")
                connection.rollback()
            except Exception:
                connection.rollback()
                raise

    def reserve(
        self,
        task_id: str,
        call_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token reservation cannot be negative")
        reservation_id = stable_investigation_id(
            "token-reservation", task_id, call_id
        )
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = self._load_reservation(connection, reservation_id)
                if existing is not None:
                    if (
                        existing.task_id != task_id
                        or existing.call_id != call_id
                        or existing.reserved_input_tokens != input_tokens
                        or existing.reserved_output_tokens != output_tokens
                    ):
                        raise TokenBudgetConflict("token reservation identity conflict")
                    connection.rollback()
                    return existing
                budget = self._budget_row(connection, task_id)
                if bool(budget[4]):
                    raise TokenBudgetExhausted("token budget breach blocks new calls")
                outstanding = connection.execute(
                    """
                    SELECT COALESCE(SUM(reserved_input), 0),
                           COALESCE(SUM(reserved_output), 0)
                    FROM token_reservations
                    WHERE task_id = ? AND status = 'reserved'
                    """,
                    (task_id,),
                ).fetchone()
                available_input = int(budget[0]) - int(budget[2]) - int(outstanding[0])
                available_output = int(budget[1]) - int(budget[3]) - int(outstanding[1])
                if input_tokens > available_input or output_tokens > available_output:
                    raise TokenBudgetExhausted("token reservation exceeds available budget")
                reservation = TokenReservation(
                    reservation_id=reservation_id,
                    task_id=task_id,
                    call_id=call_id,
                    reserved_input_tokens=input_tokens,
                    reserved_output_tokens=output_tokens,
                    status="reserved",
                )
                connection.execute(
                    """
                    INSERT INTO token_reservations(
                        reservation_id, task_id, call_id,
                        reserved_input, reserved_output, status,
                        actual_input, actual_output, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'reserved', NULL, NULL, ?)
                    """,
                    (
                        reservation_id,
                        task_id,
                        call_id,
                        input_tokens,
                        output_tokens,
                        _now(),
                    ),
                )
                connection.commit()
                return reservation
            except Exception:
                connection.rollback()
                raise

    def settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("actual token usage cannot be negative")
        return self._settle(
            reservation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="settled",
        )

    def charge_unknown(self, reservation_id: str) -> TokenReservation:
        with open_sqlite_connection(self.database_path) as connection:
            reservation = self._load_reservation(connection, reservation_id)
        if reservation is None:
            raise KeyError(reservation_id)
        return self._settle(
            reservation_id,
            input_tokens=reservation.reserved_input_tokens,
            output_tokens=reservation.reserved_output_tokens,
            status="unknown",
        )

    def available(self, task_id: str) -> TokenBalance:
        with open_sqlite_connection(self.database_path) as connection:
            budget = self._budget_row(connection, task_id)
            outstanding = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_input), 0),
                       COALESCE(SUM(reserved_output), 0)
                FROM token_reservations
                WHERE task_id = ? AND status = 'reserved'
                """,
                (task_id,),
            ).fetchone()
        return TokenBalance(
            input_tokens=max(
                0,
                int(budget[0]) - int(budget[2]) - int(outstanding[0]),
            ),
            output_tokens=max(
                0,
                int(budget[1]) - int(budget[3]) - int(outstanding[1]),
            ),
            breached=bool(budget[4]),
        )

    def usage(self, task_id: str) -> TokenUsage:
        with open_sqlite_connection(self.database_path) as connection:
            budget = self._budget_row(connection, task_id)
            row = connection.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(CASE WHEN status = 'unknown' THEN 1 ELSE 0 END), 0)
                FROM token_reservations
                WHERE task_id = ? AND status IN ('settled', 'unknown')
                """,
                (task_id,),
            ).fetchone()
        return TokenUsage(
            input_tokens=int(budget[2]),
            output_tokens=int(budget[3]),
            model_calls=int(row[0]),
            estimated=bool(row[1]),
        )

    def _settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        status: Literal["settled", "unknown"],
    ) -> TokenReservation:
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load_reservation(connection, reservation_id)
                if current is None:
                    raise KeyError(reservation_id)
                if current.status != "reserved":
                    if (
                        current.actual_input_tokens == input_tokens
                        and current.actual_output_tokens == output_tokens
                        and current.status == status
                    ):
                        connection.rollback()
                        return current
                    raise TokenBudgetConflict("token reservation settlement conflict")
                breached = (
                    input_tokens > current.reserved_input_tokens
                    or output_tokens > current.reserved_output_tokens
                )
                connection.execute(
                    """
                    UPDATE token_budgets SET
                        settled_input = settled_input + ?,
                        settled_output = settled_output + ?,
                        breached = CASE WHEN ? THEN 1 ELSE breached END,
                        updated_at = ?
                    WHERE task_id = ?
                    """,
                    (
                        input_tokens,
                        output_tokens,
                        int(breached),
                        _now(),
                        current.task_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE token_reservations SET
                        status = ?, actual_input = ?, actual_output = ?, updated_at = ?
                    WHERE reservation_id = ? AND status = 'reserved'
                    """,
                    (
                        status,
                        input_tokens,
                        output_tokens,
                        _now(),
                        reservation_id,
                    ),
                )
                connection.commit()
                return current.model_copy(
                    update={
                        "status": status,
                        "actual_input_tokens": input_tokens,
                        "actual_output_tokens": output_tokens,
                    }
                )
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _budget_row(connection, task_id: str):
        row = connection.execute(
            """
            SELECT input_cap, output_cap, settled_input, settled_output, breached
            FROM token_budgets WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    @staticmethod
    def _load_reservation(connection, reservation_id: str) -> TokenReservation | None:
        row = connection.execute(
            """
            SELECT task_id, call_id, reserved_input, reserved_output,
                   status, actual_input, actual_output
            FROM token_reservations WHERE reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            return None
        return TokenReservation(
            reservation_id=reservation_id,
            task_id=str(row[0]),
            call_id=str(row[1]),
            reserved_input_tokens=int(row[2]),
            reserved_output_tokens=int(row[3]),
            status=str(row[4]),
            actual_input_tokens=(int(row[5]) if row[5] is not None else None),
            actual_output_tokens=(int(row[6]) if row[6] is not None else None),
        )

    def _initialize_schema(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS token_budgets (
                    task_id TEXT PRIMARY KEY,
                    input_cap INTEGER NOT NULL,
                    output_cap INTEGER NOT NULL,
                    settled_input INTEGER NOT NULL,
                    settled_output INTEGER NOT NULL,
                    breached INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS token_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    reserved_input INTEGER NOT NULL,
                    reserved_output INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    actual_input INTEGER,
                    actual_output INTEGER,
                    updated_at TEXT NOT NULL,
                    UNIQUE(task_id, call_id)
                );
                """
            )
            connection.commit()


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
            len(str(getattr(message, "content", "")))
            for batch in messages
            for message in batch
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
    content = system_text + "\n" + "\n".join(
        str(getattr(message, "content", "")) for message in request.messages
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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
