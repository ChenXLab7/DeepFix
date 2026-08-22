from __future__ import annotations

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import ExecutionInfo, Runtime

from deepfix.memory import ProgressSnapshot, WorkingMemoryStore
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.prompts import CORE_REPAIR_PROMPT, PHASE_PROMPTS, RESEARCH_POLICY_PROMPT


def _snapshot(phase: str, summary: str) -> ProgressSnapshot:
    return ProgressSnapshot(
        phase=phase,
        summary=summary,
        facts=[],
        evidence=[],
        active_hypotheses=[],
        rejected_hypotheses=[],
        checked_files=[],
        experiments=[],
        next_steps=[],
        unresolved_questions=[],
    )


def _request(task_id: str) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="继续")],
        system_message=SystemMessage(content=CORE_REPAIR_PROMPT),
        tools=[],
        state={"messages": []},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="checkpoint-1",
                checkpoint_ns="",
                task_id="model-node-1",
                thread_id=task_id,
            )
        ),
    )


def _capture(middleware, request):
    received = []

    def handler(updated):
        received.append(updated)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)
    return received[0]


def test_prompt_policy_defaults_to_investigating_without_memory(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    received = _capture(PromptPolicyMiddleware(store), _request("task-a"))

    assert '<deepfix_phase name="investigating">' in received.system_message.text


def test_prompt_policy_uses_latest_snapshot_from_current_task(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    store.save("task-a", _snapshot("planning", "旧阶段"))
    store.save("task-a", _snapshot("editing", "最新阶段"))
    store.save("task-b", _snapshot("testing", "其他任务"))

    received = _capture(PromptPolicyMiddleware(store), _request("task-a"))
    text = received.system_message.text

    assert '<deepfix_phase name="editing">' in text
    assert '<deepfix_phase name="planning">' not in text
    assert '<deepfix_phase name="testing">' not in text


def test_prompt_policy_composes_core_one_phase_and_research_policy(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    received = _capture(PromptPolicyMiddleware(store), _request("task-a"))
    text = received.system_message.text

    assert text.count(CORE_REPAIR_PROMPT) == 1
    assert text.count(RESEARCH_POLICY_PROMPT) == 1
    assert sum(text.count(prompt) for prompt in PHASE_PROMPTS.values()) == 1


def test_prompt_policy_does_not_write_dynamic_prompts_to_message_history(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")
    request = _request("task-a")
    original_messages = list(request.messages)

    received = _capture(PromptPolicyMiddleware(store), request)

    assert received.messages == original_messages
    assert all("deepfix_phase" not in message.text for message in received.messages)


def test_research_policy_makes_local_evidence_authoritative(tmp_path):
    store = WorkingMemoryStore(tmp_path / "deepfix.sqlite3")

    received = _capture(PromptPolicyMiddleware(store), _request("task-a"))
    text = received.system_message.text

    assert "本地源码和真实测试结果优先于外部资料" in text
    assert "外部资料不能覆盖本地测试结果" in text
