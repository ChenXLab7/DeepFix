from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from deepfix.database import SQLiteDatabase
from deepfix.investigation.models import (
    InvestigationHypothesis,
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


def stable_question_id(task_id: str, text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    digest = hashlib.sha256(f"{task_id}\0{normalized}".encode()).hexdigest()[:24]
    return f"question_{digest}"


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _normalize_statement(value: str) -> str:
    return " ".join(value.split()).casefold()
