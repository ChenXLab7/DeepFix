from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.models import ApprovalRecord, TaskState
from deepfix.research.models import ExternalEvidence, SearchCandidate
from deepfix.research.store import ResearchEvidenceStore


def _collector(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    return EvidenceCollector(
        CompactionStore(database),
        ResearchEvidenceStore(database),
    )


def _task(tmp_path, task_id="task-a"):
    task = TaskState.create(tmp_path, "修复失败测试", ApprovalMode.MANUAL)
    task.task_id = task_id
    return task


def _execute_history(exit_code: int):
    return [
        AIMessage(
            id="m1",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "pytest -q"},
                    "id": "pytest-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            id="m2",
            content="1 failed" if exit_code else "1 passed",
            name="execute",
            tool_call_id="pytest-1",
            artifact={"exit_code": exit_code},
        ),
    ]


def test_pytest_exit_code_comes_from_paired_tool_message(tmp_path):
    collector = _collector(tmp_path)
    task = _task(tmp_path)

    block = collector.collect("task-a", _execute_history(exit_code=1), task)
    collector.collect("task-a", _execute_history(exit_code=1), task)

    assert block.tests[0].exit_code == 1
    assert block.tests[0].tool_call_id == "pytest-1"
    assert block.tests[0].source_message_id == "m2"
    assert collector.store.list_evidence("task-a") == [block.tests[0]]


def test_approved_target_is_not_reported_as_successful_file_change(tmp_path):
    collector = _collector(tmp_path)
    task = _task(tmp_path)
    task.changed_files = ["src/calc.py"]

    block = collector.collect(task.task_id, [], task)

    assert block.files[0].path == "src/calc.py"
    assert block.files[0].operation == "approved_target"
    assert block.files[0].status == "approved_target"


def test_file_success_requires_paired_system_artifact(tmp_path):
    collector = _collector(tmp_path)
    task = _task(tmp_path)
    messages = [
        AIMessage(
            id="m1",
            content="",
            tool_calls=[
                {
                    "name": "edit_file",
                    "args": {"file_path": "src/calc.py"},
                    "id": "edit-1",
                    "type": "tool_call",
                },
                {
                    "name": "write_file",
                    "args": {"file_path": "src/new.py"},
                    "id": "write-1",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            id="m2",
            content="edited",
            name="edit_file",
            tool_call_id="edit-1",
            artifact={
                "operation": "edit",
                "path": "src/calc.py",
                "status": "succeeded",
            },
        ),
        ToolMessage(
            id="m3",
            content="model says written",
            name="write_file",
            tool_call_id="write-1",
        ),
    ]

    block = collector.collect(task.task_id, messages, task)

    assert [(item.path, item.status) for item in block.files] == [
        ("src/calc.py", "succeeded")
    ]


def test_approvals_and_research_are_task_scoped_and_store_authoritative(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    research = ResearchEvidenceStore(database)
    collector = EvidenceCollector(CompactionStore(database), research)
    task = _task(tmp_path)
    task.approvals.append(ApprovalRecord("execute", "approve", "L2"))
    other_candidate = research.save_candidates(
        "task-b",
        "other",
        [
            SearchCandidate(
                candidate_id="draft",
                task_id="draft",
                source_type="official_docs",
                evidence_level="E1",
                title="Other",
                url="https://example.com/other",
                query="draft",
                repository=None,
                created_at="draft",
            )
        ],
    )[0]
    research.save_evidence(
        ExternalEvidence(
            evidence_id=uuid4().hex,
            task_id="task-b",
            candidate_id=other_candidate.candidate_id,
            source_type="official_docs",
            evidence_level="E1",
            title="Other",
            url="https://example.com/other",
            query="other",
            relevant_excerpt="other task only",
            retrieved_at="2026-08-22T00:00:00+00:00",
            dependency_name=None,
            documented_version=None,
            project_version=None,
            local_verification="verified",
            linked_test_tool_call_ids=[],
            verification_explanation="verified elsewhere",
            artifact_path="/.deepfix-artifacts/research/task-b/evidence.md",
        )
    )

    block = collector.collect("task-a", [], task)

    assert [(item.operation, item.decision) for item in block.approvals] == [
        ("execute", "approve")
    ]
    assert block.research == []


def test_ambiguous_duplicate_tool_call_is_not_deterministic_evidence(tmp_path):
    collector = _collector(tmp_path)
    task = _task(tmp_path)
    messages = [
        *_execute_history(exit_code=1),
        AIMessage(
            id="m3",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "pytest -q"},
                    "id": "pytest-1",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    block = collector.collect(task.task_id, messages, task)

    assert block.tests == []
