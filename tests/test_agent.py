import pytest
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from deepfix.agent import build_agent, build_model
from deepfix.config import ApprovalMode, load_config


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    return load_config(tmp_path, ApprovalMode.MANUAL)


@pytest.fixture
def agent(config):
    return build_agent(config, checkpointer=InMemorySaver())


def test_agent_exposes_repair_tools_without_subagent_task_tool(agent):
    tools = agent.nodes["tools"].bound.tools_by_name

    assert "task" not in tools
    assert {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
    } <= tools.keys()


def test_agent_interrupts_every_side_effecting_tool(agent):
    middleware = (
        agent.nodes["HumanInTheLoopMiddleware.after_model"].bound.func.__self__
    )

    assert set(middleware.interrupt_on) == {
        "write_file",
        "edit_file",
        "delete",
        "execute",
    }


def test_agent_returns_structured_repair_outcome(agent):
    properties = agent.get_output_jsonschema()["properties"]

    assert "structured_response" in properties


def test_model_uses_configured_deepseek_chat_deterministically(config):
    model = build_model(config)

    assert isinstance(model, ChatDeepSeek)
    assert model.model_name == "deepseek-chat"
    assert model.temperature == 0
