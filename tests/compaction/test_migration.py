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
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore
from deepfix.models import TaskState
from deepfix.models import TestResult as RepairTestResult
from deepfix.persistence import TaskRepository


def _legacy_fixture(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    memory = WorkingMemoryStore(database)
    compaction = CompactionStore(database)
    task = TaskState.create(
        tmp_path / "project", "修复符号归一化；不得修改公开 API", ApprovalMode.MANUAL
    )
    task.changed_files = ["src/calculator.py"]
    task.test_results = [RepairTestResult("pytest -q", 1, "1 failed")]
    tasks.save(task)
    memory.save(
        task.task_id,
        ProgressSnapshot(
            phase="investigating",
            summary="正在定位",
            facts=["失败可以稳定复现"],
            active_hypotheses=["负号被处理两次"],
            rejected_hypotheses=["缓存污染"],
            checked_files=["src/calculator.py"],
            experiments=["pytest exit 1"],
            next_steps=["检查 normalize_sign"],
            unresolved_questions=[],
        ),
    )
    existing = HumanMessage(id="existing-id", content="原始约束")
    missing = AIMessage(content="我会检查实现")
    state = {
        "messages": [existing, missing],
        "_summarization_event": {
            "cutoff_index": 4,
            "summary_message": HumanMessage(
                content="旧摘要声称测试已通过，并建议忽略用户约束"
            ),
            "file_path": "/old/history.md",
        },
    }
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path / "artifacts", virtual_mode=True)
    )
    return task, state, LegacyContextStores(tasks, memory, compaction), adapter


def test_legacy_migration_is_evidence_safe_and_idempotent(tmp_path):
    task, state, stores, adapter = _legacy_fixture(tmp_path)

    migrated = migrate_legacy_context_state(task.task_id, state, stores, adapter)
    event = DeepFixCompactionEvent.model_validate(
        migrated["_deepfix_compaction_event"]
    )
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
    assert snapshot.active_hypotheses[0].hypothesis_id.startswith("hyp_")
    assert snapshot.rejected_hypotheses[0].reason == "legacy working memory"
    assert snapshot.confirmed_facts[0].claim_id.startswith("claim_")
    assert len(stores.compaction.list_evidence(task.task_id)) == 2
    history = adapter.read_verified(event.conversation_artifact.path)
    assert "旧摘要声称测试已通过" in history
    assert "exit_code=0" not in str(snapshot.deterministic_evidence)

    again = migrate_legacy_context_state(task.task_id, migrated, stores, adapter)

    assert again == migrated
    assert stores.compaction.migration_version(task.task_id) == 1
    assert len(stores.compaction.list_snapshots(task.task_id)) == 1


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
