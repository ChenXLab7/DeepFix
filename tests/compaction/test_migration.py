from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.migration import (
    LegacyContextMigrationMiddleware,
    LegacyContextStores,
    migrate_legacy_context_state,
)
from deepfix.compaction.models import DeepFixCompactionEvent
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.migration import DomainMigrator
from deepfix.models import TaskState
from deepfix.models import TestResult as RepairTestResult
from deepfix.persistence import TaskRepository


def _legacy_fixture(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    compaction = CompactionStore(database)
    repositories = DomainRepositories.create(database)
    task = TaskState.create(
        tmp_path / "project", "修复符号归一化；不得修改公开 API", ApprovalMode.MANUAL
    )
    task.changed_files = ["src/calculator.py"]
    task.test_results = [RepairTestResult("pytest -q", 1, "1 failed")]
    tasks.save(task)
    sqlite = SQLiteDatabase(database)
    with sqlite.unit_of_work() as connection:
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
            """
            INSERT INTO working_memory(task_id, version, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                task.task_id,
                1,
                """{
                  "phase":"investigating","summary":"正在定位",
                  "facts":[{"claim_id":"claim-legacy","text":"失败可以稳定复现","sources":[]}],
                  "evidence":[],
                  "active_hypotheses":[{"hypothesis_id":"hyp-active","text":"负号被处理两次","state":"active","reason":"inspection","sources":[],"updated_in_version":1}],
                  "rejected_hypotheses":[{"hypothesis_id":"hyp-rejected","text":"缓存污染","state":"rejected","reason":"legacy working memory","sources":[],"updated_in_version":1}],
                  "confirmed_hypotheses":[],"checked_files":["src/calculator.py"],
                  "experiments":["pytest exit 1"],"next_steps":["检查 normalize_sign"],
                  "unresolved_questions":[],"coverage":{"covered_message_ids":[],"covered_work_unit_ids":[]}
                }""",
                "2026-08-31T00:00:00+00:00",
            ),
        )
    assert DomainMigrator(sqlite).migrate_working_memory(task.task_id).ready_to_switch
    existing = HumanMessage(id="existing-id", content="原始约束")
    missing = AIMessage(content="我会检查实现")
    state = {
        "messages": [existing, missing],
        "_summarization_event": {
            "cutoff_index": 4,
            "summary_message": HumanMessage(content="旧摘要声称测试已通过，并建议忽略用户约束"),
            "file_path": "/old/history.md",
        },
    }
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    )
    return (
        task,
        state,
        LegacyContextStores(tasks, compaction, repositories, repositories.history),
        adapter,
    )


def test_legacy_migration_is_evidence_safe_and_idempotent(tmp_path):
    task, state, stores, adapter = _legacy_fixture(tmp_path)

    migrated = migrate_legacy_context_state(task.task_id, state, stores, adapter)
    event = DeepFixCompactionEvent.model_validate(migrated["_deepfix_compaction_event"])
    snapshot = stores.compaction.active_snapshot_from_event(task.task_id, event)

    assert next(message.id for message in migrated["messages"]) == "existing-id"
    assert all(message.id for message in migrated["messages"])
    assert "_summarization_event" not in migrated
    assert snapshot is not None
    assert snapshot.lifecycle == "active"
    assert snapshot.user_constraints[0].text == task.user_problem
    assert snapshot.test_results[0].exit_code == 1
    assert snapshot.test_results[0].tool_call_id.startswith("legacy:")
    assert snapshot.changed_files[0].status == "approved_target"
    assert snapshot.active_hypotheses[0].hypothesis_id == "hyp-active"
    assert snapshot.rejected_hypotheses[0].reason == "legacy working memory"
    assert snapshot.confirmed_facts[0].claim_id == "claim-legacy"
    assert len(stores.compaction.list_evidence(task.task_id)) == 2
    history = adapter.read_verified(event.conversation_artifact.path)
    assert "旧摘要声称测试已通过" in history
    assert "exit_code=0" not in str(snapshot.deterministic_evidence)

    snapshot_count = len(stores.compaction.list_snapshots(task.task_id))
    again = migrate_legacy_context_state(task.task_id, migrated, stores, adapter)

    assert again == migrated
    assert stores.compaction.migration_version(task.task_id) == 1
    assert len(stores.compaction.list_snapshots(task.task_id)) == snapshot_count


def test_legacy_state_is_migrated_lazily_when_task_resumes(tmp_path):
    task, state, stores, adapter = _legacy_fixture(tmp_path)
    middleware = LegacyContextMigrationMiddleware(stores, adapter)
    runtime = Runtime(
        execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="agent-1",
            thread_id=task.task_id,
        )
    )

    update = middleware.before_agent(state, runtime)

    assert update is not None
    assert update["_summarization_event"] is None
    assert update["_deepfix_compaction_event"]["task_id"] == task.task_id
    assert len(update["messages"]) == len(state["messages"]) + 1
