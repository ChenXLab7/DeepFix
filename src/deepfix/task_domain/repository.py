from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepfix.database import SQLiteDatabase
from deepfix.task_domain.models import (
    AdjudicationDecision,
    AdjudicationDecisionConflict,
    TaskDefinition,
    TaskDefinitionConflict,
    TaskLifecycle,
    TaskLifecycleConflict,
    TaskLifecycleStatus,
    validate_lifecycle_transition,
)

if TYPE_CHECKING:
    from deepfix.investigation.token_budget import (
        TokenBalance,
        TokenReservation,
        TokenUsage,
    )
    from deepfix.verification import VerificationPolicy


class TaskRepository:
    """Persist only bounded Task-domain state."""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.database_path = self.database.path
        self._initialize_schema()

    @contextmanager
    def checkpoint_connection(self) -> Iterator[sqlite3.Connection]:
        with self.database.connection() as connection:
            yield connection

    def create_definition(self, definition: TaskDefinition) -> TaskDefinition:
        try:
            with self.database.unit_of_work(immediate=True) as connection:
                result = self._create_definition(connection, definition)
        except sqlite3.IntegrityError as exc:
            raise TaskDefinitionConflict(
                "task definition identity conflict"
            ) from exc
        return result

    def get_definition(self, task_id: str) -> TaskDefinition:
        with self.database.connection() as connection:
            definition = self._get_definition(connection, task_id)
        if definition is None:
            raise KeyError(task_id)
        return definition

    def get_lifecycle(self, task_id: str) -> TaskLifecycle:
        with self.database.connection() as connection:
            lifecycle = self._get_lifecycle(connection, task_id)
        if lifecycle is None:
            raise KeyError(task_id)
        return lifecycle

    def transition_lifecycle(
        self,
        task_id: str,
        next_status: TaskLifecycleStatus,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> TaskLifecycle:
        with self.database.unit_of_work(immediate=True) as connection:
            return self._transition_lifecycle(
                connection,
                task_id,
                next_status,
                reason=reason,
                expected_version=expected_version,
            )

    def save_verification_policy(self, policy: VerificationPolicy) -> None:
        from deepfix.verification import VerificationPolicyConflict

        try:
            with self.database.unit_of_work(immediate=True) as connection:
                self._save_verification_policy(connection, policy)
        except sqlite3.IntegrityError as exc:
            raise VerificationPolicyConflict(
                "verification policy identity conflict"
            ) from exc

    def load_verification_policy(
        self,
        task_id: str,
        version: int | None = None,
    ) -> VerificationPolicy | None:
        with self.database.connection() as connection:
            return self._load_verification_policy(connection, task_id, version)

    def record_adjudication(self, decision: AdjudicationDecision) -> None:
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT task_id, outcome, evidence_ids_json,
                       operation_ids_json, decided_at
                FROM adjudication_decisions WHERE decision_id = ?
                """,
                (decision.decision_id,),
            ).fetchone()
            if row is not None:
                existing = AdjudicationDecision(
                    decision_id=decision.decision_id,
                    task_id=str(row[0]),
                    outcome=str(row[1]),
                    evidence_ids=json.loads(str(row[2])),
                    operation_ids=json.loads(str(row[3])),
                    decided_at=str(row[4]),
                )
                if existing == decision:
                    return
                raise AdjudicationDecisionConflict(
                    "adjudication decision identity conflict"
                )
            connection.execute(
                """
                INSERT INTO adjudication_decisions(
                    decision_id, task_id, outcome, evidence_ids_json,
                    operation_ids_json, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.task_id,
                    decision.outcome,
                    _canonical_json(decision.evidence_ids),
                    _canonical_json(decision.operation_ids),
                    decision.decided_at,
                ),
            )

    def latest_adjudication(self, task_id: str) -> AdjudicationDecision | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT decision_id, outcome, evidence_ids_json,
                       operation_ids_json, decided_at
                FROM adjudication_decisions
                WHERE task_id = ?
                ORDER BY decided_at DESC, rowid DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return AdjudicationDecision(
            decision_id=str(row[0]),
            task_id=task_id,
            outcome=str(row[1]),
            evidence_ids=json.loads(str(row[2])),
            operation_ids=json.loads(str(row[3])),
            decided_at=str(row[4]),
        )

    def initialize_token_budget(
        self,
        task_id: str,
        *,
        input_cap: int,
        output_cap: int,
    ) -> None:
        from deepfix.investigation.token_budget import TokenBudgetConflict

        if input_cap <= 0 or output_cap <= 0:
            raise ValueError("token caps must be positive")
        with self.database.unit_of_work(immediate=True) as connection:
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
                return
            if (int(row[0]), int(row[1])) != (input_cap, output_cap):
                raise TokenBudgetConflict("token budget caps cannot change")

    def reserve_tokens(
        self,
        task_id: str,
        call_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        from deepfix.investigation.token_budget import (
            TokenBudgetConflict,
            TokenBudgetExhausted,
            TokenReservation,
        )

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token reservation cannot be negative")
        reservation_id = _token_reservation_id(task_id, call_id)
        with self.database.unit_of_work(immediate=True) as connection:
            existing = self._load_token_reservation(connection, reservation_id)
            if existing is not None:
                if (
                    existing.task_id != task_id
                    or existing.call_id != call_id
                    or existing.reserved_input_tokens != input_tokens
                    or existing.reserved_output_tokens != output_tokens
                ):
                    raise TokenBudgetConflict("token reservation identity conflict")
                return existing
            budget = self._token_budget_row(connection, task_id)
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
                raise TokenBudgetExhausted(
                    "token reservation exceeds available budget"
                )
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
            return reservation

    def settle_tokens(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenReservation:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("actual token usage cannot be negative")
        with self.database.unit_of_work(immediate=True) as connection:
            return self._settle_tokens(
                connection,
                reservation_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="settled",
            )

    def charge_unknown_tokens(self, reservation_id: str) -> TokenReservation:
        with self.database.unit_of_work(immediate=True) as connection:
            reservation = self._load_token_reservation(connection, reservation_id)
            if reservation is None:
                raise KeyError(reservation_id)
            return self._settle_tokens(
                connection,
                reservation_id,
                input_tokens=reservation.reserved_input_tokens,
                output_tokens=reservation.reserved_output_tokens,
                status="unknown",
            )

    def available_tokens(self, task_id: str) -> TokenBalance:
        from deepfix.investigation.token_budget import TokenBalance

        with self.database.connection() as connection:
            budget = self._token_budget_row(connection, task_id)
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

    def token_usage(self, task_id: str) -> TokenUsage:
        from deepfix.investigation.token_budget import TokenUsage

        with self.database.connection() as connection:
            budget = self._token_budget_row(connection, task_id)
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

    def save_legacy_projection(self, task) -> None:
        from deepfix.domain_repositories.migration import (
            ensure_no_domain_migration_fence,
            switched_domains_for_task,
        )
        from deepfix.task_domain.migration import (
            MIGRATED_LEGACY_FIELDS,
            legacy_payload_from_task,
            lifecycle_status_for_legacy,
            task_definition_from_legacy,
        )

        with self.database.unit_of_work(immediate=True) as connection:
            migration_domains = (
                "evidence",
                "research",
                "investigation",
                "execution",
                "history",
            )
            ensure_no_domain_migration_fence(
                connection,
                task.task_id,
                migration_domains,
            )
            switched_domains = switched_domains_for_task(
                connection,
                task.task_id,
                migration_domains,
            )
            payload = legacy_payload_from_task(
                task,
                switched_domains=switched_domains,
            )
            frozen_fields = frozenset().union(
                *(MIGRATED_LEGACY_FIELDS[domain] for domain in switched_domains)
            )
            existing_projection = connection.execute(
                "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if existing_projection is not None:
                previous_payload = json.loads(str(existing_projection[0]))
                payload.update(
                    {
                        field: previous_payload[field]
                        for field in frozen_fields
                        if field in previous_payload
                    }
                )
            serialized = _canonical_json(payload)
            payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            updated_at = _now()
            existing_definition = self._get_definition(connection, task.task_id)
            definition = task_definition_from_legacy(
                task,
                created_at=(
                    existing_definition.created_at
                    if existing_definition is not None
                    else None
                ),
            )
            if existing_definition is not None:
                definition = definition.model_copy(
                    update={
                        "original_message_id": existing_definition.original_message_id,
                    }
                )
            self._create_definition(connection, definition)
            self._synchronize_legacy_lifecycle(
                connection,
                task.task_id,
                lifecycle_status_for_legacy(task),
                reason=task.pause_reason,
            )
            connection.execute(
                """
                INSERT INTO legacy_task_projection(
                    task_id, legacy_phase_status, legacy_paused_from,
                    payload, payload_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    legacy_phase_status = excluded.legacy_phase_status,
                    legacy_paused_from = excluded.legacy_paused_from,
                    payload = excluded.payload,
                    payload_hash = excluded.payload_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    task.task_id,
                    task.status.value,
                    task.paused_from.value if task.paused_from is not None else None,
                    serialized,
                    payload_hash,
                    updated_at,
                ),
            )
            stored_definition = self._get_definition(connection, task.task_id)
            stored_projection = connection.execute(
                "SELECT payload_hash FROM legacy_task_projection WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if (
                stored_definition is None
                or _definition_hash(stored_definition) != _definition_hash(definition)
                or stored_projection is None
                or str(stored_projection[0]) != payload_hash
            ):
                raise sqlite3.IntegrityError("task migration verification failed")

    def save(self, task) -> None:
        """Compatibility alias retained until Plan 4 removes TaskState writes."""
        self.save_legacy_projection(task)

    def get(self, task_id: str):
        from deepfix.task_domain.migration import reconstruct_task_state

        try:
            definition = self.get_definition(task_id)
        except KeyError:
            self._backfill_historical_task(task_id)
            definition = self.get_definition(task_id)
        lifecycle = self.get_lifecycle(task_id)
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT legacy_phase_status, legacy_paused_from, payload
                FROM legacy_task_projection WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        policy = self.load_verification_policy(task_id)
        adjudication = self.latest_adjudication(task_id)
        resolution = (
            adjudication.outcome
            if adjudication is not None
            and adjudication.outcome in {"fixed", "not_reproduced"}
            else None
        )
        return reconstruct_task_state(
            definition,
            lifecycle,
            json.loads(str(row[2])),
            legacy_phase_status=str(row[0]) if row[0] is not None else None,
            legacy_paused_from=str(row[1]) if row[1] is not None else None,
            verification_policy_id=(policy.policy_id if policy is not None else None),
            verification_policy_version=(
                policy.version if policy is not None else None
            ),
            resolution=resolution,
        )

    def list_recent(self, limit: int = 20) -> list:
        self._backfill_all_historical_tasks()
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT task_id FROM legacy_task_projection
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self.get(str(row[0])) for row in rows]

    def list_recent_definitions(self, limit: int = 20) -> list[TaskDefinition]:
        """Return immutable task definitions without loading legacy projections."""
        self._backfill_all_historical_tasks()
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT definition.task_id
                FROM task_definitions AS definition
                JOIN task_lifecycle AS lifecycle USING (task_id)
                ORDER BY lifecycle.updated_at DESC, definition.rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self.get_definition(str(row[0])) for row in rows]

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_definitions (
                    task_id TEXT PRIMARY KEY,
                    original_message_id TEXT NOT NULL UNIQUE,
                    original_problem TEXT NOT NULL,
                    approval_mode TEXT NOT NULL,
                    source_project_root TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    workspace_baseline_id TEXT,
                    project_python TEXT NOT NULL,
                    confinement_level TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    definition_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_lifecycle (
                    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    paused_from TEXT,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legacy_task_projection (
                    task_id TEXT PRIMARY KEY REFERENCES task_definitions(task_id),
                    legacy_phase_status TEXT,
                    legacy_paused_from TEXT,
                    payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_policies (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    policy_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(task_id, version),
                    UNIQUE(policy_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS token_budgets (
                    task_id TEXT PRIMARY KEY,
                    input_cap INTEGER NOT NULL,
                    output_cap INTEGER NOT NULL,
                    settled_input INTEGER NOT NULL,
                    settled_output INTEGER NOT NULL,
                    breached INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS adjudication_decisions (
                    decision_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES task_definitions(task_id),
                    outcome TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    operation_ids_json TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
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
                )
                """
            )

    def _synchronize_legacy_lifecycle(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        desired: TaskLifecycleStatus,
        *,
        reason: str | None,
    ) -> TaskLifecycle:
        current = self._get_lifecycle(connection, task_id)
        if current is None:
            raise KeyError(task_id)
        if current.status is desired:
            return current
        if current.status is TaskLifecycleStatus.CREATED and desired in {
            TaskLifecycleStatus.WAITING_APPROVAL,
            TaskLifecycleStatus.COMPLETED,
        }:
            current = self._transition_lifecycle(
                connection,
                task_id,
                TaskLifecycleStatus.RUNNING,
                expected_version=current.version,
            )
        return self._transition_lifecycle(
            connection,
            task_id,
            desired,
            reason=reason,
            expected_version=current.version,
        )

    def _create_definition(
        self,
        connection: sqlite3.Connection,
        definition: TaskDefinition,
    ) -> TaskDefinition:
        definition_hash = _definition_hash(definition)
        existing = self._get_definition(connection, definition.task_id)
        if existing is not None:
            if _definition_hash(existing) != definition_hash:
                raise TaskDefinitionConflict("task definition is immutable")
            return existing
        message_owner = connection.execute(
            "SELECT task_id FROM task_definitions WHERE original_message_id = ?",
            (definition.original_message_id,),
        ).fetchone()
        if message_owner is not None:
            raise TaskDefinitionConflict("original message already defines another task")
        connection.execute(
            """
            INSERT INTO task_definitions(
                task_id, original_message_id, original_problem,
                approval_mode, source_project_root, workspace_root,
                workspace_baseline_id, project_python, confinement_level,
                created_at, definition_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                definition.task_id,
                definition.original_message_id,
                definition.original_problem,
                definition.approval_mode,
                definition.source_project_root,
                definition.workspace_root,
                definition.workspace_baseline_id,
                definition.project_python,
                definition.confinement_level,
                definition.created_at,
                definition_hash,
            ),
        )
        connection.execute(
            """
            INSERT INTO task_lifecycle(
                task_id, status, version, paused_from, reason, updated_at
            ) VALUES (?, ?, 1, NULL, NULL, ?)
            """,
            (
                definition.task_id,
                TaskLifecycleStatus.CREATED.value,
                definition.created_at,
            ),
        )
        return definition

    def _transition_lifecycle(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        next_status: TaskLifecycleStatus,
        *,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> TaskLifecycle:
        current = self._get_lifecycle(connection, task_id)
        if current is None:
            raise KeyError(task_id)
        if expected_version is not None and current.version != expected_version:
            raise TaskLifecycleConflict("task lifecycle version conflict")
        if next_status is current.status:
            return current
        validate_lifecycle_transition(current.status, next_status)
        paused_from = current.status if next_status is TaskLifecycleStatus.PAUSED else None
        next_reason = (
            reason
            if next_status
            in {TaskLifecycleStatus.PAUSED, TaskLifecycleStatus.FAILED}
            else None
        )
        updated_at = _now()
        cursor = connection.execute(
            """
            UPDATE task_lifecycle
            SET status = ?, version = version + 1,
                paused_from = ?, reason = ?, updated_at = ?
            WHERE task_id = ? AND version = ?
            """,
            (
                next_status.value,
                paused_from.value if paused_from is not None else None,
                next_reason,
                updated_at,
                task_id,
                current.version,
            ),
        )
        if cursor.rowcount != 1:
            raise TaskLifecycleConflict("task lifecycle version conflict")
        return TaskLifecycle(
            task_id=task_id,
            status=next_status,
            version=current.version + 1,
            paused_from=paused_from,
            reason=next_reason,
            updated_at=updated_at,
        )

    @staticmethod
    def _save_verification_policy(
        connection: sqlite3.Connection,
        policy: VerificationPolicy,
    ) -> None:
        from deepfix.verification import (
            VerificationPolicy,
            VerificationPolicyConflict,
        )

        latest_row = connection.execute(
            """
            SELECT payload FROM verification_policies
            WHERE task_id = ? ORDER BY version DESC LIMIT 1
            """,
            (policy.task_id,),
        ).fetchone()
        if latest_row is not None:
            latest = VerificationPolicy.model_validate_json(latest_row[0])
            if policy.version <= latest.version:
                if policy == latest:
                    return
                raise VerificationPolicyConflict("policy version must increase")
            old_required = {item.oracle_id for item in latest.required_oracles}
            new_required = {item.oracle_id for item in policy.required_oracles}
            if not old_required.issubset(new_required):
                raise VerificationPolicyConflict(
                    "required oracle cannot be removed or downgraded"
                )
        connection.execute(
            """
            INSERT INTO verification_policies(task_id, version, policy_id, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                policy.task_id,
                policy.version,
                policy.policy_id,
                policy.model_dump_json(),
            ),
        )

    @staticmethod
    def _load_verification_policy(
        connection: sqlite3.Connection,
        task_id: str,
        version: int | None,
    ) -> VerificationPolicy | None:
        from deepfix.verification import VerificationPolicy

        query = (
            "SELECT payload FROM verification_policies WHERE task_id = ? "
            + (
                "AND version = ?"
                if version is not None
                else "ORDER BY version DESC LIMIT 1"
            )
        )
        parameters = (task_id, version) if version is not None else (task_id,)
        row = connection.execute(query, parameters).fetchone()
        return VerificationPolicy.model_validate_json(row[0]) if row else None

    def _settle_tokens(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        status: str,
    ) -> TokenReservation:
        from deepfix.investigation.token_budget import (
            TokenBudgetConflict,
        )

        current = self._load_token_reservation(connection, reservation_id)
        if current is None:
            raise KeyError(reservation_id)
        if current.status != "reserved":
            if (
                current.actual_input_tokens == input_tokens
                and current.actual_output_tokens == output_tokens
                and current.status == status
            ):
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
        return current.model_copy(
            update={
                "status": status,
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
            }
        )

    @staticmethod
    def _token_budget_row(connection: sqlite3.Connection, task_id: str):
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
    def _load_token_reservation(
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> TokenReservation | None:
        from deepfix.investigation.token_budget import TokenReservation

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

    def _backfill_historical_task(self, task_id: str) -> None:
        from deepfix.models import TaskState

        with self.database.connection() as connection:
            if not _table_exists(connection, "tasks"):
                raise KeyError(task_id)
            row = connection.execute(
                "SELECT payload FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        task = TaskState.from_dict(json.loads(str(row[0])))
        self.save_legacy_projection(task)
        if task.resolution in {"fixed", "not_reproduced"}:
            lifecycle = self.get_lifecycle(task.task_id)
            self.record_adjudication(
                AdjudicationDecision(
                    decision_id=_adjudication_id(
                        task.task_id,
                        lifecycle.version,
                        task.resolution,
                    ),
                    task_id=task.task_id,
                    outcome=task.resolution,
                    evidence_ids=sorted(set(task.external_evidence_ids)),
                    operation_ids=[],
                    decided_at=lifecycle.updated_at,
                )
            )

    def _backfill_all_historical_tasks(self) -> None:
        with self.database.connection() as connection:
            if not _table_exists(connection, "tasks"):
                return
            rows = connection.execute(
                """
                SELECT task_id FROM tasks
                WHERE task_id NOT IN (SELECT task_id FROM task_definitions)
                ORDER BY updated_at ASC, rowid ASC
                """
            ).fetchall()
        for row in rows:
            self._backfill_historical_task(str(row[0]))

    @staticmethod
    def _get_definition(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> TaskDefinition | None:
        row = connection.execute(
            """
            SELECT task_id, original_message_id, original_problem, approval_mode,
                   source_project_root, workspace_root, workspace_baseline_id,
                   project_python, confinement_level, created_at
            FROM task_definitions WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return TaskDefinition(
            task_id=str(row[0]),
            original_message_id=str(row[1]),
            original_problem=str(row[2]),
            approval_mode=str(row[3]),
            source_project_root=str(row[4]),
            workspace_root=str(row[5]),
            workspace_baseline_id=str(row[6]) if row[6] is not None else None,
            project_python=str(row[7]),
            confinement_level=str(row[8]),
            created_at=str(row[9]),
        )

    @staticmethod
    def _get_lifecycle(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> TaskLifecycle | None:
        row = connection.execute(
            """
            SELECT task_id, status, version, paused_from, reason, updated_at
            FROM task_lifecycle WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return TaskLifecycle(
            task_id=str(row[0]),
            status=TaskLifecycleStatus(str(row[1])),
            version=int(row[2]),
            paused_from=(
                TaskLifecycleStatus(str(row[3])) if row[3] is not None else None
            ),
            reason=str(row[4]) if row[4] is not None else None,
            updated_at=str(row[5]),
        )


def _definition_hash(definition: TaskDefinition) -> str:
    payload = _canonical_json(
        definition.model_dump(mode="json"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_reservation_id(task_id: str, call_id: str) -> str:
    values = ["token-reservation", task_id.strip(), call_id.strip()]
    if any(not value for value in values):
        raise ValueError("token reservation identity cannot be empty")
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:32]
    return f"token-reservation_{digest}"


def _adjudication_id(task_id: str, version: int, outcome: str) -> str:
    values = ["adjudication", task_id.strip(), str(version), outcome.strip()]
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:32]
    return f"adjudication_{digest}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None
