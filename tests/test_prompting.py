from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepfix.prompting import PromptPolicyMiddleware
from deepfix.prompts import CORE_REPAIR_PROMPT, RESEARCH_POLICY_PROMPT


def _request():
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="继续")],
        system_message=SystemMessage(content="base"),
        tools=[],
        state={"messages": []},
    )


def _capture(request):
    received = []
    PromptPolicyMiddleware().wrap_model_call(
        request,
        lambda updated: received.append(updated)
        or ModelResponse(result=[AIMessage(content="ok")]),
    )
    return received[0]


def test_prompt_policy_is_stable_and_phase_free():
    received = _capture(_request())
    text = received.system_message.text

    assert text.count(CORE_REPAIR_PROMPT) == 1
    assert text.count(RESEARCH_POLICY_PROMPT) == 1
    assert "<deepfix_phase" not in text
    assert "continue_investigation" not in text


def test_prompt_policy_is_request_local_and_idempotent():
    request = _request()
    original_messages = list(request.messages)
    first = _capture(request)
    second = _capture(first)

    assert first.messages == original_messages
    assert second.system_message.text.count(CORE_REPAIR_PROMPT) == 1
    assert second.system_message.text.count(RESEARCH_POLICY_PROMPT) == 1


def test_research_policy_makes_local_evidence_authoritative():
    text = _capture(_request()).system_message.text
    assert "本地源码和真实测试结果优先于外部资料" in text
    assert "外部资料不能覆盖本地测试结果" in text


def test_core_prompt_requires_safe_diagnostic_artifact_retrieval():
    assert "search_diagnostic_artifacts" in CORE_REPAIR_PROMPT
    assert "read_diagnostic_artifact" in CORE_REPAIR_PROMPT
    assert "不要根据截断预览猜测完整结果" in CORE_REPAIR_PROMPT
    assert "不要用 grep 或 read_file 读取诊断 Artifact 根目录" in CORE_REPAIR_PROMPT
    assert "不能替代真实 pytest exit_code" in CORE_REPAIR_PROMPT
