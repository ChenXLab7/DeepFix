import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.compaction.errors import ProtectedContextLoadError
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    CompactionSnapshot,
    DeterministicEvidenceBlock,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenanceRef,
    TaskAnchor,
    UserConstraint,
)
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.investigation.models import (
    InvestigationHypothesis,
    InvestigationState,
)
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore, WorkingMemoryVersion
from deepfix.models import TaskState
from deepfix.persistence import TaskRepository
from deepfix.protected_context import (
    ProtectedContext,
    ProtectedContextBuilder,
    ProtectedContextMiddleware,
    ProtectedContextProjector,
    render_protected_context,
)
from deepfix.research.store import ResearchEvidenceStore


def _task(tmp_path, task_id, problem):
    task = TaskState.create(tmp_path, problem, ApprovalMode.MANUAL)
    task.task_id = task_id
    task.conversation = [
        {"id": f"user-{task_id}", "role": "user", "content": problem}
    ]
    return task


def _real_builder(tmp_path, *tasks):
    database = tmp_path / "deepfix.sqlite3"
    repository = TaskRepository(database)
    for task in tasks:
        repository.save(task)
    memory = WorkingMemoryStore(database)
    compaction = CompactionStore(database)
    research = ResearchEvidenceStore(database)
    return (
        ProtectedContextBuilder(
            repository,
            memory,
            compaction,
            research,
            EvidenceCollector(compaction, research),
        ),
        memory,
    )


def test_protected_context_is_task_local(tmp_path):
    task_a = _task(tmp_path, "task-a", "修复 A 的解析错误")
    task_b = _task(tmp_path, "task-b", "B TASK SECRET")
    builder, _ = _real_builder(tmp_path, task_a, task_b)

    rendered = render_protected_context(builder.build("task-a", [], None))

    assert task_a.user_problem in rendered
    assert task_b.user_problem not in rendered


def test_missing_memory_is_legal_and_complete_memory_has_every_field(tmp_path):
    task = _task(tmp_path, "task-a", "修复错误")
    builder, memory = _real_builder(tmp_path, task)

    missing = render_protected_context(builder.build("task-a", [], None))
    assert '<deepfix_working_memory version="none">' in missing

    memory.save(
        "task-a",
        ProgressSnapshot(
            phase="investigating",
            summary="调查中",
            facts=[],
            evidence=[],
            active_hypotheses=[],
            rejected_hypotheses=["路径问题已排除"],
            checked_files=["src/calc.py"],
            experiments=["pytest exit_code=1"],
            next_steps=["检查 parser"],
            unresolved_questions=["为何只在 Windows 失败"],
        ),
    )

    rendered = render_protected_context(builder.build("task-a", [], None))
    for tag in (
        "rejected_hypotheses",
        "checked_files",
        "experiments",
        "unresolved_questions",
    ):
        assert f"<{tag}>" in rendered


def _snapshot_with_duplicates(anchor, memory, evidence):
    return CompactionSnapshot(
        task_id="task-a",
        version=1,
        lifecycle="active",
        created_at="2026-08-22T00:00:00+00:00",
        activated_at="2026-08-22T00:00:01+00:00",
        source_work_unit_ids=["wu-1"],
        task_goal=anchor.task_goal,
        user_constraints=anchor.user_constraints,
        confirmed_facts=memory.snapshot.facts,
        deterministic_evidence=evidence,
        active_hypotheses=memory.snapshot.active_hypotheses,
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[
            ArtifactReference(
                path="/.deepfix-artifacts/conversation_history/task-a.md",
                kind="conversation_history",
                content_hash="a" * 64,
                work_unit_ids=["wu-1"],
            )
        ],
        content_hash="b" * 64,
    )


def test_projection_globally_deduplicates_current_entities():
    constraint = UserConstraint(
        constraint_id="constraint-1",
        text="不要修改测试",
        source_user_message_id="user-1",
    )
    claim = ProvenancedClaim(
        claim_id="claim-1",
        text="失败可复现",
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
    )
    hypothesis = HypothesisRecord(
        hypothesis_id="hyp-1",
        text="边界分支错误",
        state="active",
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
        updated_in_version=1,
    )
    memory = WorkingMemoryVersion(
        task_id="task-a",
        version=1,
        snapshot=ProgressSnapshot(
            phase="investigating",
            summary="调查",
            facts=[claim],
            active_hypotheses=[hypothesis],
        ),
        created_at="2026-08-22T00:00:00+00:00",
    )
    evidence = DeterministicEvidenceBlock(
        approvals=[
            ApprovalEvidence(
                evidence_id="evidence-1",
                operation="execute",
                decision="approve",
                risk="L2",
            )
        ]
    )
    anchor = TaskAnchor(
        task_id="task-a",
        task_goal="修复错误",
        user_constraints=[constraint],
        latest_user_message_id="user-1",
        project_root="C:/project",
        project_python="C:/python.exe",
        task_status="investigating",
    )
    snapshot = _snapshot_with_duplicates(anchor, memory, evidence)
    context = ProtectedContext(anchor, memory, evidence, snapshot)

    rendered = render_protected_context(
        context,
        projector=ProtectedContextProjector(),
    )

    assert rendered.count('constraint_id="constraint-1"') == 1
    assert rendered.count('evidence_id="evidence-1"') == 1
    assert rendered.count('hypothesis_id="hyp-1"') == 1
    assert rendered.count('claim_id="claim-1"') == 1
    assert "conversation_history/task-a.md" in rendered


def test_investigation_support_merges_into_existing_hypothesis_once():
    hypothesis = HypothesisRecord(
        hypothesis_id="hyp-1",
        text="边界分支错误",
        state="active",
        updated_in_version=1,
    )
    memory = WorkingMemoryVersion(
        task_id="task-a",
        version=1,
        snapshot=ProgressSnapshot(
            phase="investigating",
            summary="调查",
            active_hypotheses=[hypothesis],
        ),
        created_at="2026-08-22T00:00:00+00:00",
    )
    investigation = InvestigationState.new("task-a").model_copy(
        update={
            "hypotheses": [
                InvestigationHypothesis(
                    hypothesis_id="hyp-1",
                    statement="边界分支错误",
                    state="supported",
                    evidence_ids=["evidence-1"],
                    checked_locations=[],
                    reason="system evidence supports it",
                )
            ],
            "supported_hypothesis_ids": ["hyp-1"],
        }
    )
    anchor = TaskAnchor(
        task_id="task-a",
        task_goal="修复错误",
        latest_user_message_id="user-1",
        project_root="C:/project",
        project_python="C:/python.exe",
        task_status="investigating",
    )
    evidence = DeterministicEvidenceBlock(
        approvals=[
            ApprovalEvidence(
                evidence_id="evidence-1",
                operation="execute",
                decision="approve",
                risk="L2",
            )
        ]
    )
    context = ProtectedContext(
        anchor,
        memory,
        evidence,
        None,
        investigation_state=investigation,
    )

    rendered = render_protected_context(context)

    assert rendered.count('hypothesis_id="hyp-1"') == 1
    assert rendered.count('evidence_id="evidence-1"') == 1
    assert 'investigation_support="supported"' in rendered


class _Throwing:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise RuntimeError(f"{name} failed")

        return fail


@pytest.mark.parametrize("dependency", ["tasks", "memory", "compaction", "research"])
def test_authoritative_read_failure_is_typed(tmp_path, dependency):
    task = _task(tmp_path, "task-a", "修复错误")
    builder, _ = _real_builder(tmp_path, task)
    if dependency == "tasks":
        builder.tasks = _Throwing()
    elif dependency == "memory":
        builder.memory = _Throwing()
    elif dependency == "compaction":
        builder.compaction = _Throwing()
    else:
        builder.research = _Throwing()
        builder.collector.research_store = builder.research

    with pytest.raises(ProtectedContextLoadError) as captured:
        builder.build("task-a", [], None)

    assert captured.value.recovery.stage == "protected_context"
    assert captured.value.recovery.original_messages_preserved is True


def test_middleware_injects_three_request_local_blocks_only(tmp_path):
    task = _task(tmp_path, "task-a", "修复错误")
    builder, _ = _real_builder(tmp_path, task)
    middleware = ProtectedContextMiddleware(builder)
    graph_messages = [HumanMessage(id="graph-user", content="继续")]
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=list(graph_messages),
        system_message=SystemMessage(content="base"),
        tools=[],
        state={"messages": list(graph_messages)},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-1",
                thread_id="task-a",
            )
        ),
    )
    captured = []

    def handler(updated):
        captured.append(updated)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)
    received = captured[0]

    assert received.system_message.text.count("<deepfix_task_anchor>") == 1
    assert received.system_message.text.count("<deepfix_working_memory") == 1
    assert received.system_message.text.count("<deepfix_deterministic_evidence>") == 1
    assert received.messages == graph_messages
    assert request.state["messages"] == graph_messages
