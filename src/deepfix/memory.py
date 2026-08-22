from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field, StringConstraints, field_validator

from deepfix.compaction.identity import (
    stable_claim_id,
    stable_hypothesis_id,
    stable_reopened_hypothesis_id,
)
from deepfix.compaction.models import (
    FactCandidate,
    HypothesisProgressInput,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenanceRef,
    SnapshotCoverage,
)
from deepfix.models import ContextMetrics, Evidence
from deepfix.persistence import open_sqlite_connection

MemoryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
]
SummaryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class ProgressSnapshot(BaseModel):
    phase: Literal[
        "clarifying",
        "investigating",
        "planning",
        "editing",
        "testing",
        "reviewing",
    ]
    summary: SummaryText
    facts: list[ProvenancedClaim] = Field(default_factory=list, max_length=30)
    evidence: list[Evidence] = Field(default_factory=list, max_length=50)
    active_hypotheses: list[HypothesisRecord] = Field(default_factory=list, max_length=10)
    rejected_hypotheses: list[HypothesisRecord] = Field(default_factory=list, max_length=20)
    confirmed_hypotheses: list[HypothesisRecord] = Field(default_factory=list, max_length=20)
    checked_files: list[MemoryText] = Field(default_factory=list, max_length=50)
    experiments: list[MemoryText] = Field(default_factory=list, max_length=30)
    next_steps: list[MemoryText] = Field(default_factory=list, max_length=10)
    unresolved_questions: list[MemoryText] = Field(default_factory=list, max_length=10)
    coverage: SnapshotCoverage = Field(default_factory=SnapshotCoverage)

    @field_validator("facts", mode="before")
    @classmethod
    def migrate_legacy_facts(cls, value):
        return [
            {
                "claim_id": f"legacy_claim_{_text_hash(item)}",
                "text": item,
                "sources": [],
                "state": "confirmed",
            }
            if isinstance(item, str)
            else item
            for item in (value or [])
        ]

    @field_validator(
        "active_hypotheses",
        "rejected_hypotheses",
        "confirmed_hypotheses",
        mode="before",
    )
    @classmethod
    def migrate_legacy_hypotheses(cls, value, info):
        state = {
            "active_hypotheses": "active",
            "rejected_hypotheses": "rejected",
            "confirmed_hypotheses": "confirmed",
        }[info.field_name]
        return [
            {
                "hypothesis_id": f"legacy_hyp_{_text_hash(item)}",
                "text": item,
                "state": state,
                "reason": "legacy working memory" if state != "active" else None,
                "reopens_hypothesis_id": None,
                "sources": [],
                "updated_in_version": 1,
            }
            if isinstance(item, str)
            else item
            for item in (value or [])
        ]

    def all_hypotheses(self) -> list[HypothesisRecord]:
        return [
            *self.active_hypotheses,
            *self.rejected_hypotheses,
            *self.confirmed_hypotheses,
        ]


@dataclass(frozen=True)
class WorkingMemoryVersion:
    task_id: str
    version: int
    snapshot: ProgressSnapshot
    created_at: str


class WorkingMemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(
        self,
        task_id: str,
        snapshot: ProgressSnapshot,
    ) -> WorkingMemoryVersion:
        normalized_task_id = self._normalize_task_id(task_id)
        snapshot = self._canonicalize_legacy_ids(normalized_task_id, snapshot)
        created_at = self._now()
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM working_memory WHERE task_id = ?",
                    (normalized_task_id,),
                ).fetchone()
                version = int(row[0])
                connection.execute(
                    """
                    INSERT INTO working_memory(task_id, version, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        normalized_task_id,
                        version,
                        snapshot.model_dump_json(),
                        created_at,
                    ),
                )
                metrics = self._read_metrics(connection, normalized_task_id)
                metrics.working_memory_version = version
                self._write_metrics(connection, normalized_task_id, metrics, created_at)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return WorkingMemoryVersion(
            task_id=normalized_task_id,
            version=version,
            snapshot=snapshot,
            created_at=created_at,
        )

    def save_progress(
        self,
        task_id: str,
        *,
        phase: Literal[
            "clarifying", "investigating", "planning", "editing", "testing", "reviewing"
        ],
        summary: str,
        facts: list[FactCandidate],
        evidence: list[Evidence],
        hypotheses: list[HypothesisProgressInput],
        checked_files: list[str],
        experiments: list[str],
        next_steps: list[str],
        unresolved_questions: list[str],
        coverage: SnapshotCoverage,
        valid_source_ids: set[str],
    ) -> WorkingMemoryVersion:
        normalized_task_id = self._normalize_task_id(task_id)
        self._validate_sources(facts, hypotheses, valid_source_ids)
        latest = self.latest(normalized_task_id)
        next_version = (latest.version if latest else 0) + 1
        existing = {
            item.hypothesis_id: item
            for item in (latest.snapshot.all_hypotheses() if latest else [])
        }
        for update in hypotheses:
            self._apply_hypothesis_update(
                normalized_task_id,
                next_version,
                existing,
                update,
            )

        claims = [
            ProvenancedClaim(
                claim_id=stable_claim_id(normalized_task_id, item.text),
                text=item.text,
                sources=item.sources,
                state="confirmed",
            )
            for item in facts
        ]
        snapshot = ProgressSnapshot(
            phase=phase,
            summary=summary,
            facts=claims,
            evidence=evidence,
            active_hypotheses=[
                item for item in existing.values() if item.state == "active"
            ],
            rejected_hypotheses=[
                item for item in existing.values() if item.state == "rejected"
            ],
            confirmed_hypotheses=[
                item for item in existing.values() if item.state == "confirmed"
            ],
            checked_files=checked_files,
            experiments=experiments,
            next_steps=next_steps,
            unresolved_questions=unresolved_questions,
            coverage=coverage,
        )
        return self.save(normalized_task_id, snapshot)

    def latest(self, task_id: str) -> WorkingMemoryVersion | None:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT version, payload, created_at
                FROM working_memory
                WHERE task_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (normalized_task_id,),
            ).fetchone()
        if row is None:
            return None
        return self._version_from_row(normalized_task_id, row)

    def list_versions(self, task_id: str) -> list[WorkingMemoryVersion]:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT version, payload, created_at
                FROM working_memory
                WHERE task_id = ?
                ORDER BY version ASC
                """,
                (normalized_task_id,),
            ).fetchall()
        return [self._version_from_row(normalized_task_id, row) for row in rows]

    def metrics(self, task_id: str) -> ContextMetrics:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            return self._read_metrics(connection, normalized_task_id)

    def record_peak_tokens(self, task_id: str, estimate: int) -> ContextMetrics:
        if estimate < 0:
            raise ValueError("token estimate 不能为负数")

        def update(metrics: ContextMetrics) -> None:
            metrics.context_peak_tokens = max(metrics.context_peak_tokens, estimate)

        return self._update_metrics(task_id, update)

    def record_overflow(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.context_overflow_count += 1

        return self._update_metrics(task_id, update)

    def record_compaction(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.active_compaction_count += 1
            metrics.last_compaction_at = self._now()

        return self._update_metrics(task_id, update)

    def record_budget(
        self,
        task_id: str,
        usage_ratio: float,
        zone: str,
    ) -> ContextMetrics:
        if not 0 <= usage_ratio:
            raise ValueError("usage ratio 不能为负数")
        normalized_zone = zone.strip()
        if not normalized_zone:
            raise ValueError("budget zone 不能为空")

        def update(metrics: ContextMetrics) -> None:
            metrics.latest_usage_ratio = usage_ratio
            metrics.latest_budget_zone = normalized_zone

        return self._update_metrics(task_id, update)

    def record_compaction_failure(
        self,
        task_id: str,
        error_code: str,
    ) -> ContextMetrics:
        normalized_error = error_code.strip()
        if not normalized_error:
            raise ValueError("error code 不能为空")

        def update(metrics: ContextMetrics) -> None:
            metrics.compaction_failure_count += 1
            metrics.last_compaction_error = _bounded_metric_error(normalized_error)

        return self._update_metrics(task_id, update)

    def record_passthrough(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.normal_zone_passthrough_count += 1

        return self._update_metrics(task_id, update)

    def record_manual_error(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.manual_compaction_error_count += 1

        return self._update_metrics(task_id, update)

    def record_overflow_retry(self, task_id: str) -> ContextMetrics:
        def update(metrics: ContextMetrics) -> None:
            metrics.overflow_retry_count += 1

        return self._update_metrics(task_id, update)

    def record_compaction_event(
        self,
        task_id: str,
        *,
        snapshot_version: int,
        artifact_path: str,
        emergency: bool,
    ) -> ContextMetrics:
        if snapshot_version < 1:
            raise ValueError("snapshot version 必须为正整数")
        normalized_path = artifact_path.strip()
        if not normalized_path:
            raise ValueError("artifact path 不能为空")

        def update(metrics: ContextMetrics) -> None:
            metrics.active_compaction_count += 1
            metrics.last_compaction_at = self._now()
            metrics.active_compaction_snapshot_version = snapshot_version
            metrics.last_compaction_artifact = normalized_path
            if emergency:
                metrics.emergency_compaction_count += 1
            else:
                metrics.normal_compaction_count += 1

        return self._update_metrics(task_id, update)

    def _update_metrics(self, task_id: str, update) -> ContextMetrics:
        normalized_task_id = self._normalize_task_id(task_id)
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                metrics = self._read_metrics(connection, normalized_task_id)
                update(metrics)
                self._write_metrics(
                    connection,
                    normalized_task_id,
                    metrics,
                    self._now(),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return metrics

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memory (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS context_metrics (
                    task_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connection(self) -> sqlite3.Connection:
        return open_sqlite_connection(self.database_path)

    @staticmethod
    def _validate_sources(
        facts: list[FactCandidate],
        hypotheses: list[HypothesisProgressInput],
        valid_source_ids: set[str],
    ) -> None:
        for source in (
            source
            for item in [*facts, *hypotheses]
            for source in item.sources
        ):
            if source.ref_id not in valid_source_ids:
                raise ValueError(f"source ref_id 不属于当前任务: {source.ref_id}")

    @staticmethod
    def _apply_hypothesis_update(
        task_id: str,
        version: int,
        existing: dict[str, HypothesisRecord],
        update: HypothesisProgressInput,
    ) -> None:
        if update.hypothesis_id and update.reopens_hypothesis_id:
            raise ValueError("hypothesis_id 与 reopens_hypothesis_id 不能同时提供")
        if update.reopens_hypothesis_id:
            previous = existing.get(update.reopens_hypothesis_id)
            if previous is None or previous.state != "rejected":
                raise ValueError("reopens_hypothesis_id 必须引用已排除假设")
            if update.target_state != "active" or not update.reason or not update.sources:
                raise ValueError("重新开启假设需要 active、reason 和 new evidence source")
            first_source = update.sources[0].ref_id
            new_id = stable_reopened_hypothesis_id(
                task_id,
                previous.hypothesis_id,
                first_source,
                update.text,
            )
            existing[new_id] = HypothesisRecord(
                hypothesis_id=new_id,
                text=update.text,
                state="active",
                reason=update.reason,
                reopens_hypothesis_id=previous.hypothesis_id,
                sources=update.sources,
                updated_in_version=version,
            )
            return
        if update.hypothesis_id:
            previous = existing.get(update.hypothesis_id)
            if previous is None:
                raise ValueError("hypothesis_id 不属于当前任务")
            if previous.state != "active" and update.target_state != previous.state:
                raise ValueError("只有 active 假设可以进行普通状态迁移")
            if update.target_state in {"rejected", "confirmed"} and not update.reason:
                raise ValueError("迁移到 rejected/confirmed 必须提供 reason")
            existing[previous.hypothesis_id] = previous.model_copy(
                update={
                    "text": update.text,
                    "state": update.target_state,
                    "reason": update.reason,
                    "sources": _merge_sources(previous.sources, update.sources),
                    "updated_in_version": version,
                }
            )
            return
        if update.target_state != "active":
            raise ValueError("新假设只能以 active 创建，状态迁移必须提供 hypothesis_id")
        if not update.sources:
            raise ValueError("新假设必须提供 source")
        new_id = stable_hypothesis_id(task_id, update.sources[0].ref_id, update.text)
        existing[new_id] = HypothesisRecord(
            hypothesis_id=new_id,
            text=update.text,
            state="active",
            reason=update.reason,
            reopens_hypothesis_id=None,
            sources=update.sources,
            updated_in_version=version,
        )

    @staticmethod
    def _canonicalize_legacy_ids(
        task_id: str,
        snapshot: ProgressSnapshot,
    ) -> ProgressSnapshot:
        facts = [
            item.model_copy(update={"claim_id": stable_claim_id(task_id, item.text)})
            if item.claim_id.startswith("legacy_claim_")
            else item
            for item in snapshot.facts
        ]
        hypotheses = []
        for item in snapshot.all_hypotheses():
            if item.hypothesis_id.startswith("legacy_hyp_"):
                item = item.model_copy(
                    update={
                        "hypothesis_id": stable_hypothesis_id(
                            task_id,
                            "legacy-working-memory",
                            item.text,
                        )
                    }
                )
            hypotheses.append(item)
        return snapshot.model_copy(
            update={
                "facts": facts,
                "active_hypotheses": [
                    item for item in hypotheses if item.state == "active"
                ],
                "rejected_hypotheses": [
                    item for item in hypotheses if item.state == "rejected"
                ],
                "confirmed_hypotheses": [
                    item for item in hypotheses if item.state == "confirmed"
                ],
            }
        )

    @staticmethod
    def _read_metrics(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> ContextMetrics:
        row = connection.execute(
            "SELECT payload FROM context_metrics WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return ContextMetrics()
        payload = cast(dict[str, object], json.loads(row[0]))
        return ContextMetrics(**payload)

    @staticmethod
    def _write_metrics(
        connection: sqlite3.Connection,
        task_id: str,
        metrics: ContextMetrics,
        updated_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO context_metrics(task_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                json.dumps(asdict(metrics), ensure_ascii=False),
                updated_at,
            ),
        )

    @staticmethod
    def _version_from_row(
        task_id: str,
        row: sqlite3.Row | tuple[object, ...],
    ) -> WorkingMemoryVersion:
        return WorkingMemoryVersion(
            task_id=task_id,
            version=int(row[0]),
            snapshot=ProgressSnapshot.model_validate_json(str(row[1])),
            created_at=str(row[2]),
        )

    @staticmethod
    def _normalize_task_id(task_id: str) -> str:
        normalized = task_id.strip()
        if not normalized:
            raise ValueError("task_id 不能为空")
        return normalized

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds")


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:24]


def _bounded_metric_error(value: str) -> str:
    return value if len(value) <= 300 else value[:299] + "…"


def _merge_sources(
    previous: list[ProvenanceRef],
    current: list[ProvenanceRef],
) -> list[ProvenanceRef]:
    merged: dict[tuple[str, str], ProvenanceRef] = {}
    for source in [*previous, *current]:
        merged[(source.kind, source.ref_id)] = source
    return list(merged.values())
