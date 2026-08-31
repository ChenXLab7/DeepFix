from __future__ import annotations

import ast
import inspect
from pathlib import Path

from deepfix.agent import build_agent
from deepfix.compaction.models import CompactionSnapshot
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import EvidenceKind
from deepfix.domain_repositories.migration import DomainMigrator

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "deepfix"


def test_agent_constructor_and_tool_assembly_exclude_working_memory() -> None:
    parameters = inspect.signature(build_agent).parameters
    source = inspect.getsource(build_agent)

    assert "working_memory_store" not in parameters
    assert '"save_progress"' not in source
    assert "build_save_progress_tool" not in source


def test_runtime_modules_do_not_import_retired_memory_module() -> None:
    offenders: list[str] = []
    migration_files = {
        SOURCE_ROOT / "domain_repositories" / "migration.py",
    }
    for path in SOURCE_ROOT.rglob("*.py"):
        if path in migration_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "deepfix.memory":
                offenders.append(str(path.relative_to(SOURCE_ROOT)))

    assert offenders == []


def test_current_snapshot_schema_has_no_working_memory_projection() -> None:
    assert "working_memory_version" not in CompactionSnapshot.model_fields


def test_retired_memory_module_is_absent() -> None:
    assert not (SOURCE_ROOT / "memory.py").exists()


def test_legacy_memory_semantics_migrate_to_bounded_domain_owners(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.sqlite3")
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE working_memory (
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
            INSERT INTO working_memory(task_id, version, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "task-a",
                1,
                """{
                  "phase": "investigating",
                  "summary": "legacy",
                  "facts": [{"claim_id":"claim-1","text":"parser fails","sources":[]}],
                  "evidence": [],
                  "active_hypotheses": [{
                    "hypothesis_id":"hyp-1","text":"branch is wrong","state":"active",
                    "reason":"inspection","sources":[],"updated_in_version":1
                  }],
                  "rejected_hypotheses": [],
                  "confirmed_hypotheses": [],
                  "checked_files": [],
                  "experiments": ["reproduced with empty input"],
                  "next_steps": ["edit parser"],
                  "unresolved_questions": ["why only on Windows?"],
                  "coverage": {"covered_message_ids":[],"covered_work_unit_ids":[]}
                }""",
                "2026-08-31T00:00:00+00:00",
            ),
        )

    migrator = DomainMigrator(database)
    report = migrator.migrate_working_memory("task-a")

    assert report.ready_to_switch is True
    assert [item.kind for item in migrator.evidence.list_for_task("task-a")] == [
        EvidenceKind.SEMANTIC_CLAIM
    ]
    assert [item.hypothesis_id for item in migrator.investigation.list_hypotheses("task-a")] == [
        "hyp-1"
    ]
    assert [item.text for item in migrator.investigation.list_questions("task-a")] == [
        "why only on Windows?"
    ]
    history_kinds = {
        item.kind
        for record in migrator.history.list_for_task("task-a")
        for item in record.semantic_items
    }
    assert {"fact", "hypothesis", "experiment", "unresolved_question"} <= history_kinds
    assert "next_step" not in history_kinds


def test_legacy_memory_migration_refuses_missing_hypothesis_evidence(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.sqlite3")
    with database.unit_of_work() as connection:
        connection.execute(
            """
            CREATE TABLE working_memory (
                task_id TEXT NOT NULL, version INTEGER NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, version)
            )
            """
        )
        connection.execute(
            "INSERT INTO working_memory(task_id, version, payload, created_at) VALUES (?, ?, ?, ?)",
            (
                "task-a",
                1,
                """{
                  "summary":"legacy","facts":[],"evidence":[],
                  "active_hypotheses":[{
                    "hypothesis_id":"hyp-missing","text":"branch is wrong",
                    "state":"active","reason":"legacy",
                    "sources":[{"kind":"system_evidence","ref_id":"missing-evidence"}],
                    "updated_in_version":1
                  }],
                  "rejected_hypotheses":[],"confirmed_hypotheses":[],
                  "checked_files":[],"experiments":[],"next_steps":[],
                  "unresolved_questions":[],
                  "coverage":{"covered_message_ids":[],"covered_work_unit_ids":[]}
                }""",
                "2026-08-31T00:00:00+00:00",
            ),
        )

    report = DomainMigrator(database).migrate_working_memory("task-a")

    assert report.ready_to_switch is False
    assert report.missing_references == [
        "hypothesis:hyp-missing:evidence:missing-evidence"
    ]
