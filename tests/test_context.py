import pytest
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.agent import build_main_model
from deepfix.backend import build_backend
from deepfix.compaction.models import SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import partition_work_units
from deepfix.config import ApprovalMode, load_config
from deepfix.context import (
    ContextMemoryMiddleware,
    build_context_middleware,
    build_save_progress_tool,
    render_working_memory,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import InvestigationHypothesis, NewInvestigationEvent
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore, WorkingMemoryVersion
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.research.middleware import ResearchEvidenceMiddleware
from deepfix.research.store import ResearchEvidenceStore


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    return load_config(tmp_path, ApprovalMode.MANUAL)


@pytest.fixture
def store(config):
    return WorkingMemoryStore(config.database_path)


def snapshot(summary: str = "已复现失败") -> ProgressSnapshot:
    return ProgressSnapshot(
        phase="investigating",
        summary=summary,
        facts=["失败可复现"],
        evidence=[],
        active_hypotheses=[],
        rejected_hypotheses=[],
        checked_files=[],
        experiments=[],
        next_steps=["检查调用链"],
        unresolved_questions=[],
    )


def progress_call(call_id: str, **overrides):
    args = {
        "phase": "investigating",
        "summary": "已复现失败",
        "facts": [{"text": "失败可复现", "sources": []}],
        "evidence": [],
        "hypotheses": [],
        "checked_files": [],
        "experiments": [],
        "next_steps": ["检查调用链"],
        "unresolved_questions": [],
    }
    args.update(overrides)
    return {
        "name": "save_progress",
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def invoke_tool(tool, call, thread_id: str):
    node = ToolNode([tool])
    return node.invoke(
        {"messages": [AIMessage(content="", tool_calls=[call])]},
        {"configurable": {"thread_id": thread_id}},
        runtime=Runtime(),
    )["messages"][0]


def model_request(thread_id: str) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="继续调查")],
        system_message=SystemMessage(content="基础系统提示"),
        tools=[],
        state={"messages": []},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-node-1",
                thread_id=thread_id,
            )
        ),
    )


def test_save_progress_uses_thread_id_and_does_not_expose_task_id(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    tool = build_save_progress_tool(store)

    result = invoke_tool(tool, progress_call("save-1"), "task-a")

    assert "task_id" not in tool.args
    assert "claim_id" not in str(tool.args)
    assert "active_hypotheses" not in tool.args
    assert "rejected_hypotheses" not in tool.args
    assert "hypotheses" in tool.args
    assert store.latest("task-a").version == 1
    assert store.latest("task-a").snapshot.summary == "已复现失败"
    assert result.status == "success"
    assert result.artifact["version"] == 1
    assert len(result.artifact["claim_ids"]) == 1
    assert result.artifact["hypothesis_ids"] == []


def test_save_progress_rejects_invalid_snapshot_without_writing_version(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    tool = build_save_progress_tool(store)
    call = progress_call(
        "save-invalid",
        hypotheses=[
            {
                "hypothesis_id": "hyp-old",
                "text": "缓存过期",
                "target_state": "active",
                "reason": "新证据",
                "reopens_hypothesis_id": "hyp-rejected",
                "sources": [],
            }
        ],
    )

    result = invoke_tool(tool, call, "task-invalid")

    assert result.status == "error"
    assert "hypothesis_id" in result.text
    assert store.latest("task-invalid") is None


def test_save_progress_accepts_current_task_evidence_and_investigation_hypothesis(
    tmp_path,
):
    database = tmp_path / "deepfix.sqlite3"
    memory = WorkingMemoryStore(database)
    compaction = CompactionStore(database)
    investigation = InvestigationStore(database)
    task_id = "task-a"
    evidence_id = "evidence-test-failure"
    hypothesis_id = "hyp-supported-root-cause"
    compaction.save_evidence(
        task_id,
        SystemTestEvidence(
            evidence_id=evidence_id,
            command="python -m pytest test_mergesort.py -q",
            exit_code=1,
            summary="RecursionError",
            tool_call_id="pytest-before",
            source_message_id="pytest-result-before",
        ),
    )
    state = investigation.ensure_started(task_id)
    supported = InvestigationHypothesis(
        hypothesis_id=hypothesis_id,
        statement="单元素数组缺少递归终止条件",
        state="supported",
        evidence_ids=[evidence_id],
        checked_locations=[],
        reason="失败测试和源码共同支持",
    )
    investigation.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id=stable_investigation_id(
                    "event", task_id, "supported-hypothesis-fixture"
                ),
                task_id=task_id,
                event_type="hypothesis_supported",
                phase_before=state.agent_phase,
                phase_after=state.agent_phase,
            )
        ],
        state.model_copy(
            update={
                "hypotheses": [supported],
                "supported_hypothesis_ids": [hypothesis_id],
            }
        ),
    )
    tool = build_save_progress_tool(
        memory,
        compaction_store=compaction,
        investigation_store=investigation,
    )
    call = progress_call(
        "save-supported",
        facts=[
            {
                "text": "单元素输入会无限递归",
                "sources": [
                    {"kind": "system_evidence", "ref_id": evidence_id}
                ],
            }
        ],
        hypotheses=[
            {
                "hypothesis_id": hypothesis_id,
                "text": "单元素数组缺少递归终止条件",
                "target_state": "confirmed",
                "reason": "最小修复后测试通过",
                "sources": [
                    {"kind": "system_evidence", "ref_id": evidence_id}
                ],
            }
        ],
    )

    result = invoke_tool(tool, call, task_id)

    assert result.status == "success"
    saved = memory.latest(task_id).snapshot
    assert saved.facts[0].sources[0].ref_id == evidence_id
    assert saved.confirmed_hypotheses[0].hypothesis_id == hypothesis_id


def test_save_progress_validates_work_unit_sources_by_work_unit_id(tmp_path):
    task_id = "task-work-unit"
    memory = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    tool = build_save_progress_tool(memory)
    call = progress_call("save-work-unit")
    work_unit_seed = "pending-work-unit-id"
    call["args"]["facts"] = [
        {
            "text": "已形成完整工作单元",
            "sources": [{"kind": "work_unit", "ref_id": work_unit_seed}],
        }
    ]
    ai_message = AIMessage(
        id="message-save-work-unit",
        content="",
        tool_calls=[call],
    )
    messages = [HumanMessage(id="message-user", content="修复问题"), ai_message]
    work_unit_id = partition_work_units(messages, set()).units[-1].unit_id
    ai_message.tool_calls[0]["args"]["facts"][0]["sources"][0]["ref_id"] = (
        work_unit_id
    )
    node = ToolNode([tool])

    result = node.invoke(
        {"messages": messages},
        {"configurable": {"thread_id": task_id}},
        runtime=Runtime(),
    )["messages"][-1]

    assert result.status == "success"
    assert memory.latest(task_id).snapshot.facts[0].sources[0].ref_id == work_unit_id


def test_save_progress_accepts_supported_as_investigation_alias(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    memory = WorkingMemoryStore(database)
    investigation = InvestigationStore(database)
    task_id = "task-supported-alias"
    hypothesis_id = "hyp-supported-alias"
    evidence_id = "evidence-supported-alias"
    CompactionStore(database).save_evidence(
        task_id,
        SystemTestEvidence(
            evidence_id=evidence_id,
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="call-supported-alias",
            source_message_id="message-supported-alias",
        ),
    )
    state = investigation.ensure_started(task_id)
    hypothesis = InvestigationHypothesis(
        hypothesis_id=hypothesis_id,
        statement="边界条件错误",
        state="supported",
        evidence_ids=[evidence_id],
        checked_locations=[],
        reason="已由调查工具验证",
    )
    investigation.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id=stable_investigation_id(
                    "event", task_id, "supported-alias-fixture"
                ),
                task_id=task_id,
                event_type="hypothesis_supported",
                phase_before=state.agent_phase,
                phase_after=state.agent_phase,
            )
        ],
        state.model_copy(update={"hypotheses": [hypothesis]}),
    )
    tool = build_save_progress_tool(memory, investigation_store=investigation)
    call = progress_call(
        "save-supported-alias",
        hypotheses=[
            {
                "hypothesis_id": hypothesis_id,
                "text": "边界条件错误",
                "target_state": "supported",
                "reason": "沿用 Investigation 的操作状态",
                "sources": [],
            }
        ],
    )

    result = invoke_tool(tool, call, task_id)

    assert result.status == "success"
    saved = memory.latest(task_id).snapshot
    assert saved.active_hypotheses[0].hypothesis_id == hypothesis_id


def test_context_middleware_injects_only_latest_task_memory_and_records_peak(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    store.save("task-a", snapshot("旧调查摘要"))
    store.save("task-a", snapshot("最新调查摘要"))
    middleware = ContextMemoryMiddleware(store)
    captured = []

    def handler(received):
        captured.append(received)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(model_request("task-a"), handler)

    injected = captured[0].system_message.text
    assert "<deepfix_working_memory" in injected
    assert "最新调查摘要" in injected
    assert "检查调用链" in injected
    assert "旧调查摘要" not in injected
    assert store.metrics("task-a").context_peak_tokens > 0

    captured.clear()
    middleware.wrap_model_call(model_request("task-b"), handler)
    assert captured[0].system_message.text == "基础系统提示"
    assert store.metrics("task-b").context_peak_tokens == 0


def test_render_working_memory_has_a_hard_injection_bound():
    bounded_snapshot = snapshot("摘要" * 1000).model_copy(
        update={"facts": [f"事实-{index}-" + "x" * 1000 for index in range(30)]}
    )
    rendered = render_working_memory(
        WorkingMemoryVersion(
            task_id="task-a",
            version=7,
            snapshot=bounded_snapshot,
            created_at="2026-08-21T00:00:00+00:00",
        )
    )

    assert "…[truncated]" in rendered
    assert "事实-0-" in rendered
    assert "事实-8-" not in rendered
    assert len(rendered) <= 12_000


def test_render_working_memory_includes_every_field_category():
    complete = snapshot().model_copy(
        update={
            "rejected_hypotheses": ["路径问题已排除"],
            "checked_files": ["src/calc.py"],
            "experiments": ["pytest 退出码为 1"],
        }
    )

    rendered = render_working_memory(
        WorkingMemoryVersion(
            task_id="task-a",
            version=2,
            snapshot=complete,
            created_at="2026-08-21T00:00:00+00:00",
        )
    )

    assert "<rejected_hypotheses>" in rendered
    assert "路径问题已排除" in rendered
    assert "<checked_files>" in rendered
    assert "src/calc.py" in rendered
    assert "<experiments>" in rendered
    assert "pytest 退出码为 1" in rendered


def test_context_middleware_shares_one_summarization_engine(config, store):
    middleware = build_context_middleware(
        build_main_model(config),
        build_backend(config),
        store,
        InvestigationStore(config.database_path),
    )
    summarization = next(
        item for item in middleware if isinstance(item, SummarizationMiddleware)
    )
    tool_layer = next(
        item for item in middleware if isinstance(item, SummarizationToolMiddleware)
    )

    assert tool_layer._summarization is summarization
    assert summarization._lc_helper.trigger == ("fraction", 0.70)
    assert summarization._lc_helper.keep == ("fraction", 0.15)


def test_context_middleware_has_explicit_prompt_memory_research_order(config, store):
    research_store = ResearchEvidenceStore(config.database_path)

    middleware = build_context_middleware(
        build_main_model(config),
        build_backend(config),
        store,
        InvestigationStore(config.database_path),
        research_store,
    )
    names = [type(item) for item in middleware]

    assert names[-3:] == [
        PromptPolicyMiddleware,
        ContextMemoryMiddleware,
        ResearchEvidenceMiddleware,
    ]
