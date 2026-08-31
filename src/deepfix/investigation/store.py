from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.investigation import InvestigationRepository
from deepfix.domain_repositories.migration import domain_is_switched
from deepfix.investigation.experiments import (
    ExperimentAssessment,
    ExperimentResult,
    StrategyDecision,
)
from deepfix.investigation.models import (
    InvestigationEvent,
    InvestigationState,
    NewInvestigationEvent,
)
from deepfix.persistence import open_sqlite_connection


class InvestigationStateConflict(RuntimeError):
    pass


class InvestigationStore:
    def __init__(
        self,
        database_path: SQLiteDatabase | str | Path,
        *,
        repositories=None,
    ) -> None:
        self.database = (
            repositories.database
            if repositories is not None
            else (
                database_path
                if isinstance(database_path, SQLiteDatabase)
                else SQLiteDatabase(database_path)
            )
        )
        self.database_path = self.database.path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.repository = (
            repositories.investigation
            if repositories is not None
            else InvestigationRepository(self.database)
        )
        self._initialize()

    def load(self, task_id: str) -> InvestigationState | None:
        with open_sqlite_connection(self.database_path) as connection:
            legacy = self._load(connection, task_id)
        if legacy is None:
            return None
        current = {
            item.hypothesis_id: item
            for item in self.repository.list_hypotheses(task_id)
        }
        if not current:
            return legacy
        merged = {item.hypothesis_id: item for item in legacy.hypotheses}
        merged.update(current)
        hypotheses = list(merged.values())[-64:]
        return legacy.model_copy(
            update={
                "hypotheses": hypotheses,
                "supported_hypothesis_ids": [
                    item.hypothesis_id
                    for item in hypotheses
                    if item.state == "supported"
                ][-64:],
            }
        )

    def ensure_started(self, task_id: str) -> InvestigationState:
        existing = self.load(task_id)
        if existing is not None:
            return existing
        state = InvestigationState.new(task_id)
        event = NewInvestigationEvent.task_started(task_id)
        try:
            return self.commit(0, [event], state)
        except InvestigationStateConflict:
            loaded = self.load(task_id)
            if loaded is None:
                raise
            return loaded

    def commit(
        self,
        expected_version: int,
        events: list[NewInvestigationEvent],
        next_state: InvestigationState,
    ) -> InvestigationState:
        if any(event.task_id != next_state.task_id for event in events):
            raise InvestigationStateConflict("事件与物化状态的任务不一致")

        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load(connection, next_state.task_id)
                current_version = 0 if current is None else current.version
                existing = self._existing_events(connection, next_state.task_id, events)
                if len(existing) == len(events):
                    self._validate_replay(events, existing)
                    connection.rollback()
                    if current is None:
                        raise InvestigationStateConflict("重放事件缺少物化状态")
                    replayed = self.load(next_state.task_id)
                    if replayed is None:
                        raise InvestigationStateConflict("重放事件缺少物化状态")
                    return replayed
                if existing or current_version != expected_version:
                    raise InvestigationStateConflict("investigation state 版本冲突")

                committed = next_state.model_copy(update={"version": current_version + 1})
                self._insert_events(connection, events)
                self.repository.project_hypotheses(
                    connection,
                    committed.task_id,
                    committed.hypotheses,
                )
                self._upsert_state(connection, committed)
                connection.commit()
                return committed
            except Exception:
                connection.rollback()
                raise

    def list_events(self, task_id: str) -> list[InvestigationEvent]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload, sequence, created_at
                FROM investigation_events
                WHERE task_id = ?
                ORDER BY sequence
                """,
                (task_id,),
            ).fetchall()
        events = []
        for row in rows:
            payload = _legacy_event_payload(cast(dict[str, Any], json.loads(row[0])))
            if payload is None:
                continue
            events.append(
                InvestigationEvent.model_validate(
                    {**payload, "sequence": row[1], "created_at": row[2]}
                )
            )
        return events

    def commit_experiment(
        self,
        expected_version: int,
        *,
        event_id: str,
        event_payload: str,
        result: ExperimentResult,
        assessment: ExperimentAssessment,
        next_state: InvestigationState,
    ) -> InvestigationState:
        if result.experiment_id != assessment.experiment_id:
            raise InvestigationStateConflict("experiment result/assessment 不一致")
        if next_state.task_id.strip() == "":
            raise InvestigationStateConflict("experiment state 缺少 task_id")
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                current = self._load(connection, next_state.task_id)
                current_version = 0 if current is None else current.version
                existing = connection.execute(
                    """
                    SELECT payload FROM experiment_events
                    WHERE task_id = ? AND event_id = ?
                    """,
                    (next_state.task_id, event_id),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != event_payload:
                        raise InvestigationStateConflict(
                            "相同 experiment event_id 的内容冲突"
                        )
                    connection.rollback()
                    if current is None:
                        raise InvestigationStateConflict("重放缺少物化状态")
                    replayed = self.load(next_state.task_id)
                    if replayed is None:
                        raise InvestigationStateConflict("重放缺少物化状态")
                    return replayed
                if current_version != expected_version:
                    raise InvestigationStateConflict("investigation state 版本冲突")
                committed = next_state.model_copy(update={"version": current_version + 1})
                created_at = datetime.now(UTC).isoformat(timespec="microseconds")
                connection.execute(
                    """
                    INSERT INTO experiment_results(
                        task_id, experiment_id, result_payload,
                        assessment_payload, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(task_id, experiment_id) DO UPDATE SET
                        result_payload = excluded.result_payload,
                        assessment_payload = excluded.assessment_payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        committed.task_id,
                        result.experiment_id,
                        result.model_dump_json(),
                        assessment.model_dump_json(),
                        created_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO experiment_events(
                        task_id, event_id, experiment_id, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        committed.task_id,
                        event_id,
                        result.experiment_id,
                        event_payload,
                        created_at,
                    ),
                )
                self.repository.project_hypotheses(
                    connection,
                    committed.task_id,
                    committed.hypotheses,
                )
                self._upsert_state(connection, committed)
                connection.commit()
                return committed
            except Exception:
                connection.rollback()
                raise

    def count_experiment_events(self, task_id: str) -> int:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM experiment_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row[0])

    def save_strategy_decision(self, decision: StrategyDecision) -> None:
        payload = decision.model_dump_json()
        created_at = datetime.now(UTC).isoformat(timespec="microseconds")
        with open_sqlite_connection(self.database_path) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO strategy_decisions(
                        task_id, decision_id, payload, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        decision.task_id,
                        decision.decision_id,
                        payload,
                        created_at,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                row = connection.execute(
                    """
                    SELECT payload FROM strategy_decisions
                    WHERE task_id = ? AND decision_id = ?
                    """,
                    (decision.task_id, decision.decision_id),
                ).fetchone()
                if row is None or str(row[0]) != payload:
                    raise InvestigationStateConflict(
                        "strategy decision identity conflict"
                    ) from exc

    def count_strategy_decisions(self, task_id: str) -> int:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM strategy_decisions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row[0])

    def last_sequence(self, task_id: str) -> int:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM investigation_events
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        return int(row[0])

    def has_event(self, task_id: str, event_id: str) -> bool:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM investigation_events
                WHERE task_id = ? AND event_id = ?
                """,
                (task_id, event_id),
            ).fetchone()
        return row is not None

    def _initialize(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_state (
                    task_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def _load(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> InvestigationState | None:
        row = connection.execute(
            "SELECT payload FROM investigation_state WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return InvestigationState.model_validate(
            _legacy_state_payload(cast(dict[str, Any], json.loads(row[0])))
        )

    @staticmethod
    def _existing_events(
        connection: sqlite3.Connection,
        task_id: str,
        events: list[NewInvestigationEvent],
    ) -> dict[str, str]:
        if not events:
            return {}
        event_ids = [event.event_id for event in events]
        placeholders = ",".join("?" for _ in event_ids)
        rows = connection.execute(
            f"""
            SELECT event_id, payload
            FROM investigation_events
            WHERE task_id = ? AND event_id IN ({placeholders})
            """,
            (task_id, *event_ids),
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    @staticmethod
    def _validate_replay(
        events: list[NewInvestigationEvent],
        existing: dict[str, str],
    ) -> None:
        for event in events:
            stored = _legacy_event_payload(
                cast(dict[str, Any], json.loads(existing[event.event_id]))
            )
            normalized = (
                json.dumps(stored, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if stored is not None
                else ""
            )
            if normalized != _canonical_event(event):
                raise InvestigationStateConflict("相同 event_id 的事件内容冲突")

    @staticmethod
    def _insert_events(
        connection: sqlite3.Connection,
        events: list[NewInvestigationEvent],
    ) -> None:
        if not events:
            return
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0)
            FROM investigation_events
            WHERE task_id = ?
            """,
            (events[0].task_id,),
        ).fetchone()
        first_sequence = int(row[0]) + 1
        created_at = datetime.now(UTC).isoformat(timespec="microseconds")
        connection.executemany(
            """
            INSERT INTO investigation_events(
                task_id, event_id, sequence, event_type, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event.task_id,
                    event.event_id,
                    first_sequence + offset,
                    event.event_type.value,
                    _canonical_event(event),
                    created_at,
                )
                for offset, event in enumerate(events)
            ],
        )

    def _upsert_state(
        self,
        connection: sqlite3.Connection,
        state: InvestigationState,
    ) -> None:
        updated_at = datetime.now(UTC).isoformat(timespec="microseconds")
        legacy_state = state
        if domain_is_switched(self.database, "investigation", state.task_id):
            legacy_state = state.model_copy(
                update={"hypotheses": [], "supported_hypothesis_ids": []}
            )
        connection.execute(
            """
            INSERT INTO investigation_state(task_id, version, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                version = excluded.version,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                legacy_state.task_id,
                legacy_state.version,
                legacy_state.model_dump_json(),
                updated_at,
            ),
        )


def _canonical_event(event: NewInvestigationEvent) -> str:
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migration-only parser for retired phase/permit state fields."""
    normalized = dict(payload)
    for field in (
        "agent_phase",
        "paused_agent_phase",
        "permit",
        "post_permit_review_pending",
    ):
        normalized.pop(field, None)
    return normalized


def _legacy_event_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Strip retired event projections and omit phase-only historical events."""
    if payload.get("event_type") in {
        "phase_changed",
        "investigation_permit_granted",
    }:
        return None
    normalized = dict(payload)
    normalized.pop("phase_before", None)
    normalized.pop("phase_after", None)
    return normalized
