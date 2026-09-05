from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from deepfix.database import SQLiteDatabase
from deepfix.investigation.experiments import (
    ExperimentAssessment,
    ExperimentResult,
    StrategyDecision,
)
from deepfix.investigation.models import (
    InvestigationEvent,
    InvestigationHypothesis,
    InvestigationState,
    NewInvestigationEvent,
    UnresolvedQuestion,
)


class HypothesisIdentityConflict(RuntimeError):
    pass


class HypothesisTransitionError(RuntimeError):
    pass


class InvestigationEvidenceMissing(RuntimeError):
    pass


class QuestionIdentityConflict(RuntimeError):
    pass


class InvestigationStateConflict(RuntimeError):
    pass


class InvestigationRepository:
    """Own current investigation beliefs, never execution or lifecycle policy."""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self._initialize_schema()

    def record_hypothesis(
        self,
        task_id: str,
        hypothesis: InvestigationHypothesis,
    ) -> InvestigationHypothesis:
        normalized_task_id = _required(task_id, "task_id")
        with self.database.unit_of_work(immediate=True) as connection:
            return self._record_hypothesis(
                connection,
                normalized_task_id,
                hypothesis,
            )

    def load(self, task_id: str) -> InvestigationState | None:
        with self.database.connection() as connection:
            state = self._load_state(connection, task_id)
        if state is None:
            return None
        current = {item.hypothesis_id: item for item in self.list_hypotheses(task_id)}
        if not current:
            return state
        merged = {item.hypothesis_id: item for item in state.hypotheses}
        merged.update(current)
        hypotheses = list(merged.values())[-64:]
        return state.model_copy(
            update={
                "hypotheses": hypotheses,
                "supported_hypothesis_ids": [
                    item.hypothesis_id for item in hypotheses if item.state == "supported"
                ][-64:],
            }
        )

    def ensure_started(self, task_id: str) -> InvestigationState:
        existing = self.load(task_id)
        if existing is not None:
            return existing
        state = InvestigationState.new(task_id)
        try:
            return self.commit(0, [NewInvestigationEvent.task_started(task_id)], state)
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
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._load_state(connection, next_state.task_id)
            current_version = 0 if current is None else current.version
            existing = self._existing_events(connection, next_state.task_id, events)
            if len(existing) == len(events):
                self._validate_replay(events, existing)
                if current is None:
                    raise InvestigationStateConflict("重放事件缺少物化状态")
                return current
            if existing or current_version != expected_version:
                raise InvestigationStateConflict("investigation state 版本冲突")
            committed = next_state.model_copy(update={"version": current_version + 1})
            self._insert_events(connection, events)
            self.project_hypotheses(connection, committed.task_id, committed.hypotheses)
            self._upsert_state(connection, committed)
            return committed

    def list_events(self, task_id: str) -> list[InvestigationEvent]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload, sequence, created_at FROM investigation_events
                WHERE task_id = ? ORDER BY sequence
                """,
                (task_id,),
            ).fetchall()
        events: list[InvestigationEvent] = []
        for row in rows:
            payload = _legacy_event_payload(cast(dict[str, Any], json.loads(row[0])))
            if payload is not None:
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
        with self.database.unit_of_work(immediate=True) as connection:
            current = self._load_state(connection, next_state.task_id)
            current_version = 0 if current is None else current.version
            existing = connection.execute(
                "SELECT payload FROM experiment_events WHERE task_id = ? AND event_id = ?",
                (next_state.task_id, event_id),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != event_payload:
                    raise InvestigationStateConflict("相同 experiment event_id 的内容冲突")
                if current is None:
                    raise InvestigationStateConflict("重放缺少物化状态")
                return current
            if current_version != expected_version:
                raise InvestigationStateConflict("investigation state 版本冲突")
            committed = next_state.model_copy(update={"version": current_version + 1})
            created_at = _utc_now()
            connection.execute(
                """
                INSERT INTO experiment_results(
                    task_id, experiment_id, result_payload, assessment_payload, updated_at
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
            self.project_hypotheses(connection, committed.task_id, committed.hypotheses)
            self._upsert_state(connection, committed)
            return committed

    def count_experiment_events(self, task_id: str) -> int:
        return self._count(task_id, "experiment_events")

    def save_strategy_decision(self, decision: StrategyDecision) -> None:
        payload = decision.model_dump_json()
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload FROM strategy_decisions WHERE task_id = ? AND decision_id = ?",
                (decision.task_id, decision.decision_id),
            ).fetchone()
            if row is not None:
                if str(row[0]) != payload:
                    raise InvestigationStateConflict("strategy decision identity conflict")
                return
            connection.execute(
                """
                INSERT INTO strategy_decisions(task_id, decision_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (decision.task_id, decision.decision_id, payload, _utc_now()),
            )

    def count_strategy_decisions(self, task_id: str) -> int:
        return self._count(task_id, "strategy_decisions")

    def last_sequence(self, task_id: str) -> int:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM investigation_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row[0])

    def has_event(self, task_id: str, event_id: str) -> bool:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM investigation_events WHERE task_id = ? AND event_id = ?",
                (task_id, event_id),
            ).fetchone()
        return row is not None

    def backfill_current_hypothesis(
        self,
        task_id: str,
        hypothesis: InvestigationHypothesis,
    ) -> InvestigationHypothesis:
        """Import a legacy current record without treating history as authority."""
        normalized_task_id = _required(task_id, "task_id")
        with self.database.unit_of_work(immediate=True) as connection:
            existing = self._load_hypothesis(
                connection,
                normalized_task_id,
                hypothesis.hypothesis_id,
            )
            if existing is not None:
                if existing != hypothesis:
                    raise HypothesisIdentityConflict(
                        "legacy current hypothesis 与现有 domain record 冲突"
                    )
                return existing
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO hypotheses(
                    task_id, hypothesis_id, statement, state, payload,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_task_id,
                    hypothesis.hypothesis_id,
                    hypothesis.statement,
                    hypothesis.state,
                    hypothesis.model_dump_json(),
                    now,
                    now,
                ),
            )
        return hypothesis

    def get_hypothesis(
        self,
        task_id: str,
        hypothesis_id: str,
    ) -> InvestigationHypothesis:
        with self.database.connection() as connection:
            item = self._load_hypothesis(
                connection,
                _required(task_id, "task_id"),
                _required(hypothesis_id, "hypothesis_id"),
            )
        if item is None:
            raise KeyError((task_id, hypothesis_id))
        return item

    def list_hypotheses(self, task_id: str) -> list[InvestigationHypothesis]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM hypotheses
                WHERE task_id = ? ORDER BY created_at, hypothesis_id
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [
            InvestigationHypothesis.model_validate_json(str(row[0])) for row in rows
        ]

    def missing_current_evidence_ids(
        self,
        task_id: str,
        evidence_ids: list[str],
    ) -> list[str]:
        normalized_task_id = _required(task_id, "task_id")
        normalized_ids = list(
            dict.fromkeys(_required(item, "evidence_id") for item in evidence_ids)
        )
        with self.database.connection() as connection:
            return self._missing_current_evidence_ids(
                connection,
                normalized_task_id,
                normalized_ids,
            )

    def open_question(self, question: UnresolvedQuestion) -> UnresolvedQuestion:
        if question.status != "open":
            raise ValueError("open_question 只接受 open question")
        with self.database.unit_of_work(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT payload FROM unresolved_questions
                WHERE task_id = ? AND question_id = ?
                """,
                (question.task_id, question.question_id),
            ).fetchone()
            if row is not None:
                existing = UnresolvedQuestion.model_validate_json(str(row[0]))
                if existing != question:
                    raise QuestionIdentityConflict(
                        "相同 question_id 的内容不一致"
                    )
                return existing
            connection.execute(
                """
                INSERT INTO unresolved_questions(
                    task_id, question_id, status, payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question.task_id,
                    question.question_id,
                    question.status,
                    question.model_dump_json(),
                    question.created_at,
                    question.created_at,
                ),
            )
        return question

    def resolve_question(
        self,
        task_id: str,
        question_id: str,
        *,
        evidence_ids: list[str],
    ) -> UnresolvedQuestion:
        normalized_task_id = _required(task_id, "task_id")
        normalized_question_id = _required(question_id, "question_id")
        normalized_evidence_ids = list(
            dict.fromkeys(_required(item, "evidence_id") for item in evidence_ids)
        )
        if not normalized_evidence_ids:
            raise InvestigationEvidenceMissing(
                "解决 unresolved question 至少需要一个当前 Evidence"
            )
        with self.database.unit_of_work(immediate=True) as connection:
            self._require_current_evidence(
                connection,
                normalized_task_id,
                normalized_evidence_ids,
            )
            row = connection.execute(
                """
                SELECT payload FROM unresolved_questions
                WHERE task_id = ? AND question_id = ?
                """,
                (normalized_task_id, normalized_question_id),
            ).fetchone()
            if row is None:
                raise KeyError((normalized_task_id, normalized_question_id))
            current = UnresolvedQuestion.model_validate_json(str(row[0]))
            if current.status == "resolved":
                if current.resolution_evidence_ids != normalized_evidence_ids:
                    raise QuestionIdentityConflict("question 已由不同 Evidence 解决")
                return current
            resolved_at = _utc_now()
            resolved = current.model_copy(
                update={
                    "status": "resolved",
                    "resolution_evidence_ids": normalized_evidence_ids,
                    "resolved_at": resolved_at,
                }
            )
            connection.execute(
                """
                UPDATE unresolved_questions
                SET status = ?, payload = ?, updated_at = ?
                WHERE task_id = ? AND question_id = ?
                """,
                (
                    resolved.status,
                    resolved.model_dump_json(),
                    resolved_at,
                    normalized_task_id,
                    normalized_question_id,
                ),
            )
        return resolved

    def list_questions(
        self,
        task_id: str,
        *,
        status: Literal["open", "resolved"] | None = None,
    ) -> list[UnresolvedQuestion]:
        normalized_task_id = _required(task_id, "task_id")
        parameters: tuple[str, ...]
        if status is None:
            query = """
                SELECT payload FROM unresolved_questions
                WHERE task_id = ? ORDER BY created_at, question_id
            """
            parameters = (normalized_task_id,)
        else:
            query = """
                SELECT payload FROM unresolved_questions
                WHERE task_id = ? AND status = ? ORDER BY created_at, question_id
            """
            parameters = (normalized_task_id, status)
        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [UnresolvedQuestion.model_validate_json(str(row[0])) for row in rows]

    def project_hypotheses(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        hypotheses: list[InvestigationHypothesis],
    ) -> None:
        """Compatibility facade hook; caller owns the surrounding transaction."""
        for hypothesis in hypotheses:
            self._record_hypothesis(connection, task_id, hypothesis)

    def record_historical_conflict(
        self,
        task_id: str,
        hypothesis_id: str,
        *,
        current_statement: str,
        historical_statement: str,
        source_id: str,
    ) -> None:
        payload = json.dumps(
            {
                "current_statement": current_statement,
                "historical_statement": historical_statement,
                "source_id": source_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.database.unit_of_work(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO investigation_conflicts(
                    task_id, hypothesis_id, source_id, payload, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id, hypothesis_id, source_id) DO UPDATE SET
                    payload = excluded.payload
                """,
                (task_id, hypothesis_id, source_id, payload, _utc_now()),
            )

    def list_conflicts(self, task_id: str) -> list[dict[str, str]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT hypothesis_id, payload FROM investigation_conflicts
                WHERE task_id = ? ORDER BY hypothesis_id, source_id
                """,
                (_required(task_id, "task_id"),),
            ).fetchall()
        return [
            {
                "hypothesis_id": str(row[0]),
                **cast(dict[str, str], json.loads(str(row[1]))),
            }
            for row in rows
        ]

    def _record_hypothesis(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        hypothesis: InvestigationHypothesis,
    ) -> InvestigationHypothesis:
        existing = self._load_hypothesis(
            connection,
            task_id,
            hypothesis.hypothesis_id,
        )
        if existing is None:
            self._validate_new_hypothesis(connection, task_id, hypothesis)
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO hypotheses(
                    task_id, hypothesis_id, statement, state, payload,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    hypothesis.hypothesis_id,
                    hypothesis.statement,
                    hypothesis.state,
                    hypothesis.model_dump_json(),
                    now,
                    now,
                ),
            )
            return hypothesis
        if existing.statement != hypothesis.statement:
            raise HypothesisIdentityConflict(
                "相同 hypothesis_id 不能更换假设陈述"
            )
        if existing == hypothesis:
            return existing
        if existing.state != "candidate" or hypothesis.state not in {
            "supported",
            "rejected",
        }:
            raise HypothesisTransitionError(
                f"不允许的 hypothesis 状态迁移: {existing.state} -> {hypothesis.state}"
            )
        if hypothesis.reopens_hypothesis_id is not None:
            raise HypothesisTransitionError("普通状态迁移不能设置 reopens_hypothesis_id")
        self._require_current_evidence(
            connection,
            task_id,
            hypothesis.evidence_ids,
        )
        connection.execute(
            """
            UPDATE hypotheses SET state = ?, payload = ?, updated_at = ?
            WHERE task_id = ? AND hypothesis_id = ?
            """,
            (
                hypothesis.state,
                hypothesis.model_dump_json(),
                _utc_now(),
                task_id,
                hypothesis.hypothesis_id,
            ),
        )
        return hypothesis

    def _validate_new_hypothesis(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        hypothesis: InvestigationHypothesis,
    ) -> None:
        reopened_id = hypothesis.reopens_hypothesis_id
        if reopened_id is None:
            rejected_rows = connection.execute(
                """
                SELECT hypothesis_id, statement FROM hypotheses
                WHERE task_id = ? AND state = 'rejected'
                """,
                (task_id,),
            ).fetchall()
            if any(
                _normalize_statement(str(row[1]))
                == _normalize_statement(hypothesis.statement)
                for row in rejected_rows
            ):
                raise HypothesisTransitionError(
                    "重新开启已 rejected 的相同假设必须提供 reopens_hypothesis_id"
                )
            if hypothesis.state != "candidate":
                self._require_current_evidence(
                    connection,
                    task_id,
                    hypothesis.evidence_ids,
                )
            return
        if hypothesis.state != "candidate":
            raise HypothesisTransitionError("重新开启的假设必须以 candidate 创建")
        previous = self._load_hypothesis(connection, task_id, reopened_id)
        if previous is None or previous.state != "rejected":
            raise HypothesisTransitionError(
                "reopens_hypothesis_id 必须引用当前任务已 rejected 的假设"
            )
        self._require_current_evidence(
            connection,
            task_id,
            hypothesis.evidence_ids,
        )

    @staticmethod
    def _load_hypothesis(
        connection: sqlite3.Connection,
        task_id: str,
        hypothesis_id: str,
    ) -> InvestigationHypothesis | None:
        row = connection.execute(
            """
            SELECT payload FROM hypotheses
            WHERE task_id = ? AND hypothesis_id = ?
            """,
            (task_id, hypothesis_id),
        ).fetchone()
        return (
            None
            if row is None
            else InvestigationHypothesis.model_validate_json(str(row[0]))
        )

    @staticmethod
    def _require_current_evidence(
        connection: sqlite3.Connection,
        task_id: str,
        evidence_ids: list[str],
    ) -> None:
        normalized = list(dict.fromkeys(evidence_ids))
        if not normalized:
            raise InvestigationEvidenceMissing(
                "状态迁移至少需要一个当前 Evidence"
            )
        missing = InvestigationRepository._missing_current_evidence_ids(
            connection,
            task_id,
            normalized,
        )
        if missing:
            raise InvestigationEvidenceMissing(
                f"Evidence 不属于当前任务或不存在: {', '.join(missing)}"
            )

    @staticmethod
    def _missing_current_evidence_ids(
        connection: sqlite3.Connection,
        task_id: str,
        evidence_ids: list[str],
    ) -> list[str]:
        if not evidence_ids:
            return []
        placeholders = ",".join("?" for _ in evidence_ids)
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'evidence_records'
            """
        ).fetchone()
        found = (
            set()
            if table is None
            else {
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT evidence_id FROM evidence_records
                    WHERE task_id = ? AND evidence_id IN ({placeholders})
                    """,
                    (task_id, *evidence_ids),
                ).fetchall()
            }
        )
        return sorted(set(evidence_ids) - found)

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hypotheses (
                    task_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, hypothesis_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS unresolved_questions (
                    task_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, question_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_conflicts (
                    task_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, hypothesis_id, source_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_events (
                    task_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, event_id),
                    UNIQUE (task_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_decisions (
                    task_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, decision_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    task_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    result_payload TEXT NOT NULL,
                    assessment_payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, experiment_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_events (
                    task_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, event_id)
                )
                """
            )
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

    def _count(self, task_id: str, table: str) -> int:
        if table not in {"experiment_events", "strategy_decisions"}:
            raise ValueError("unsupported investigation table")
        with self.database.connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _load_state(
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
            SELECT event_id, payload FROM investigation_events
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
            "SELECT COALESCE(MAX(sequence), 0) FROM investigation_events WHERE task_id = ?",
            (events[0].task_id,),
        ).fetchone()
        first_sequence = int(row[0]) + 1
        created_at = _utc_now()
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

    @staticmethod
    def _upsert_state(
        connection: sqlite3.Connection,
        state: InvestigationState,
    ) -> None:
        connection.execute(
            """
            INSERT INTO investigation_state(task_id, version, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                version = excluded.version,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (state.task_id, state.version, state.model_dump_json(), _utc_now()),
        )


def stable_question_id(task_id: str, text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    digest = hashlib.sha256(f"{task_id}\0{normalized}".encode()).hexdigest()[:24]
    return f"question_{digest}"


def _canonical_event(event: NewInvestigationEvent) -> str:
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
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
    if payload.get("event_type") in {"phase_changed", "investigation_permit_granted"}:
        return None
    normalized = dict(payload)
    normalized.pop("phase_before", None)
    normalized.pop("phase_after", None)
    return normalized


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _normalize_statement(value: str) -> str:
    return " ".join(value.split()).casefold()
