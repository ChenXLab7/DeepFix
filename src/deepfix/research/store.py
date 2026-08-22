from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from deepfix.models import Evidence
from deepfix.persistence import open_sqlite_connection
from deepfix.research.models import (
    ExternalEvidence,
    ResearchQuery,
    SearchCandidate,
    VerificationStatus,
)


class ResearchEvidenceStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_query(
        self,
        task_id: str,
        sanitized_query: str,
        providers: list[str],
        errors: list[str],
    ) -> ResearchQuery:
        query = ResearchQuery(
            query_id=uuid4().hex,
            task_id=task_id,
            sanitized_query=sanitized_query,
            providers=providers,
            provider_errors=errors,
            created_at=_utc_now(),
        )
        payload = _serialize(query)
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO research_queries(task_id, query_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (query.task_id, query.query_id, payload, query.created_at),
            )
            connection.commit()
        return query

    def save_candidates(
        self,
        task_id: str,
        query: str,
        candidates: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        created_at = _utc_now()
        saved = [
            candidate.model_copy(
                update={
                    "candidate_id": uuid4().hex,
                    "task_id": task_id,
                    "query": query,
                    "created_at": created_at,
                }
            )
            for candidate in candidates
        ]
        with open_sqlite_connection(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO search_candidates(
                    task_id,
                    candidate_id,
                    payload,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        candidate.task_id,
                        candidate.candidate_id,
                        _serialize(candidate),
                        candidate.created_at,
                    )
                    for candidate in saved
                ],
            )
            connection.commit()
        return saved

    def get_candidate(self, task_id: str, candidate_id: str) -> SearchCandidate:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM search_candidates
                WHERE task_id = ? AND candidate_id = ?
                """,
                (task_id, candidate_id),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return SearchCandidate.model_validate(json.loads(row[0]))

    def save_evidence(self, evidence: ExternalEvidence) -> None:
        _validate_uuid4(evidence.evidence_id, "evidence_id")
        self.get_candidate(evidence.task_id, evidence.candidate_id)
        updated_at = _utc_now()
        with open_sqlite_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO external_evidence(
                    task_id,
                    evidence_id,
                    candidate_id,
                    payload,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evidence.task_id,
                    evidence.evidence_id,
                    evidence.candidate_id,
                    _serialize(evidence),
                    updated_at,
                ),
            )
            connection.commit()

    def get_evidence(self, task_id: str, evidence_id: str) -> ExternalEvidence:
        with open_sqlite_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM external_evidence
                WHERE task_id = ? AND evidence_id = ?
                """,
                (task_id, evidence_id),
            ).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        return ExternalEvidence.model_validate(json.loads(row[0]))

    def list_evidence(self, task_id: str) -> list[ExternalEvidence]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM external_evidence
                WHERE task_id = ?
                ORDER BY updated_at, rowid
                """,
                (task_id,),
            ).fetchall()
        return [ExternalEvidence.model_validate(json.loads(row[0])) for row in rows]

    def update_verification(
        self,
        task_id: str,
        evidence_id: str,
        *,
        local_verification: VerificationStatus,
        local_evidence: list[Evidence],
        linked_test_tool_call_ids: list[str],
        verification_explanation: str | None,
    ) -> ExternalEvidence:
        current = self.get_evidence(task_id, evidence_id)
        payload = current.model_dump(mode="python")
        payload.update(
            {
                "local_verification": local_verification,
                "local_evidence": local_evidence,
                "linked_test_tool_call_ids": linked_test_tool_call_ids,
                "verification_explanation": verification_explanation,
            }
        )
        updated = ExternalEvidence.model_validate(payload)
        updated_at = _utc_now()
        with open_sqlite_connection(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE external_evidence
                SET payload = ?, updated_at = ?
                WHERE task_id = ? AND evidence_id = ?
                """,
                (_serialize(updated), updated_at, task_id, evidence_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(evidence_id)
            connection.commit()
        return updated

    def query_summary(self, task_id: str) -> tuple[int, list[str]]:
        with open_sqlite_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM research_queries
                WHERE task_id = ?
                ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        queries = [ResearchQuery.model_validate(json.loads(row[0])) for row in rows]
        errors = list(
            dict.fromkeys(
                error
                for query in queries
                for error in query.provider_errors
            )
        )
        return len(queries), errors

    def _initialize(self) -> None:
        with open_sqlite_connection(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_queries (
                    task_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, query_id)
                );

                CREATE TABLE IF NOT EXISTS search_candidates (
                    task_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, candidate_id)
                );

                CREATE TABLE IF NOT EXISTS external_evidence (
                    task_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, evidence_id)
                );
                """
            )
            connection.commit()


def _serialize(model: ResearchQuery | SearchCandidate | ExternalEvidence) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _validate_uuid4(value: str, field_name: Literal["evidence_id"]) -> None:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须是 UUID4") from exc
    if parsed.version != 4:
        raise ValueError(f"{field_name} 必须是 UUID4")
