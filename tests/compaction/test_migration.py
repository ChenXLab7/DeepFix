from __future__ import annotations

import json

from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.migration import LegacyContextStores, migrate_legacy_context_state
from deepfix.compaction.models import DeepFixCompactionEvent
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.migration import DomainMigrator


def _fixture(tmp_path):
    database = SQLiteDatabase(tmp_path / "deepfix.sqlite3")
    payload = {
        "task_id": "legacy-context",
        "project_root": str(tmp_path / "project"),
        "user_problem": "修复符号归一化；不得修改公开 API",
        "approval_mode": "manual",
        "project_python": "python",
        "status": "investigating",
        "conversation": [{"role": "user", "content": "修复符号归一化；不得修改公开 API"}],
        "changed_files": ["src/calculator.py"],
        "test_results": [
            {
                "command": "pytest -q",
                "exit_code": 1,
                "summary": "1 failed",
                "tool_call_id": "call-test",
                "source_message_id": "message-test",
            }
        ],
        "approvals": [],
    }
    with database.unit_of_work() as connection:
        connection.execute(
            "CREATE TABLE tasks(task_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            (
                payload["task_id"],
                json.dumps(payload, ensure_ascii=False),
                "2026-08-31T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            CREATE TABLE working_memory (
                task_id TEXT, version INTEGER, payload TEXT, created_at TEXT,
                PRIMARY KEY(task_id, version)
            )
            """
        )
        connection.execute(
            "INSERT INTO working_memory VALUES (?, ?, ?, ?)",
            (
                payload["task_id"],
                1,
                json.dumps(
                    {
                        "phase": "investigating",
                        "summary": "正在定位",
                        "facts": [],
                        "evidence": [],
                        "active_hypotheses": [],
                        "rejected_hypotheses": [],
                        "confirmed_hypotheses": [],
                        "checked_files": [],
                        "experiments": [],
                        "next_steps": [],
                        "unresolved_questions": [],
                        "coverage": {
                            "covered_message_ids": [],
                            "covered_work_unit_ids": [],
                        },
                    }
                ),
                "2026-08-31T00:00:00+00:00",
            ),
        )
    repositories = DomainRepositories.create(database)
    repositories.tasks.get_definition(payload["task_id"])
    assert DomainMigrator(database).migrate_working_memory(payload["task_id"]).ready_to_switch
    state = {
        "messages": [
            HumanMessage(id="existing-id", content="原始约束"),
            AIMessage(content="我会检查实现"),
        ],
        "_summarization_event": {
            "summary_message": HumanMessage(content="旧摘要声称测试已经通过")
        },
    }
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    )
    return payload, repositories, state, adapter


def test_legacy_context_restores_evidence_and_history_without_legacy_store(tmp_path):
    payload, repositories, state, adapter = _fixture(tmp_path)
    stores = LegacyContextStores(repositories.tasks, repositories)

    migrated = migrate_legacy_context_state(payload["task_id"], state, stores, adapter)
    event = DeepFixCompactionEvent.model_validate(migrated["_deepfix_compaction_event"])
    snapshot = repositories.history.project_snapshot(
        payload["task_id"],
        event.active_snapshot_version,
        evidence=repositories.evidence,
        investigation=repositories.investigation,
    )

    assert snapshot.lifecycle == "active"
    assert snapshot.task_goal == payload["user_problem"]
    assert snapshot.test_results[0].exit_code == 1
    assert snapshot.changed_files[0].path == "src/calculator.py"
    assert "旧摘要声称测试已经通过" in adapter.read_verified(
        event.conversation_artifact.path
    )
    assert len(repositories.evidence.list_deterministic(payload["task_id"])) == 2


def test_legacy_context_migration_is_idempotent(tmp_path):
    payload, repositories, state, adapter = _fixture(tmp_path)
    stores = LegacyContextStores(repositories.tasks, repositories)
    first = migrate_legacy_context_state(payload["task_id"], state, stores, adapter)
    snapshot_count = len(repositories.history.list_for_task(payload["task_id"]))

    second = migrate_legacy_context_state(payload["task_id"], first, stores, adapter)

    assert second == first
    assert repositories.history.migration_version(payload["task_id"]) == 1
    assert len(repositories.history.list_for_task(payload["task_id"])) == snapshot_count
