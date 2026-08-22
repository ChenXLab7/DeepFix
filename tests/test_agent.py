import pytest
from langchain_core.tools import StructuredTool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from deepfix.agent import build_agent, build_compaction_model, build_main_model
from deepfix.compaction.middleware import (
    DeepFixCompactionMiddleware,
    MessageIdentityMiddleware,
)
from deepfix.config import ApprovalMode, load_config
from deepfix.extensions import build_research_extensions
from deepfix.memory import WorkingMemoryStore
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.protected_context import ProtectedContextMiddleware


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    return load_config(tmp_path, ApprovalMode.MANUAL)


@pytest.fixture
def agent(config):
    return build_agent(
        config,
        checkpointer=InMemorySaver(),
        working_memory_store=WorkingMemoryStore(config.database_path),
    )


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
        "save_progress",
        "compact_conversation",
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
    assert "save_progress" not in middleware.interrupt_on
    assert "compact_conversation" not in middleware.interrupt_on


def test_agent_returns_structured_repair_outcome(agent):
    properties = agent.get_output_jsonschema()["properties"]

    assert "structured_response" in properties


def test_role_models_use_independent_names_keys_and_shared_base_url(config):
    main = build_main_model(config)
    compaction = build_compaction_model(config)

    assert isinstance(main, ChatDeepSeek)
    assert isinstance(compaction, ChatDeepSeek)
    assert main is not compaction
    assert main.model_name == "deepseek-v4-pro"
    assert compaction.model_name == "deepseek-v4-flash"
    assert main.temperature == compaction.temperature == 0
    assert main.openai_api_base == compaction.openai_api_base == "https://api.deepseek.com"
    assert main.openai_api_key.get_secret_value() == "main-secret"
    assert compaction.openai_api_key.get_secret_value() == "compact-secret"


def test_both_role_models_disable_thinking_for_tool_choice_compatibility(config):
    main = build_main_model(config)
    compaction = build_compaction_model(config)

    expected = {"thinking": {"type": "disabled"}}
    assert main.extra_body == expected
    assert compaction.extra_body == expected


def _extension_tool(name):
    def run():
        return "ok"

    return StructuredTool.from_function(run, name=name, description="test tool")


def test_agent_assembles_research_extensions_without_changing_core_guards(
    config,
    monkeypatch,
):
    captured = {}
    registered = {}
    main_model = object()
    compaction_model = object()

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    def fake_register_harness_profile(key, profile):
        registered["key"] = key
        registered["profile"] = profile

    monkeypatch.setattr("deepfix.agent.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        "deepfix.agent.register_harness_profile",
        fake_register_harness_profile,
    )
    monkeypatch.setattr("deepfix.agent.build_main_model", lambda config: main_model)
    monkeypatch.setattr(
        "deepfix.agent.build_compaction_model",
        lambda config: compaction_model,
    )
    extensions = build_research_extensions(
        inspect_dependency=_extension_tool("inspect_dependency"),
        search_technical_sources=_extension_tool("search_technical_sources"),
        fetch_external_evidence=_extension_tool("fetch_external_evidence"),
        link_external_evidence=_extension_tool("link_external_evidence"),
    )

    result = build_agent(
        config,
        checkpointer=InMemorySaver(),
        working_memory_store=WorkingMemoryStore(config.database_path),
        extensions=extensions,
    )

    tool_names = [tool.name for tool in captured["tools"]]
    middleware = captured["middleware"]
    middleware_names = [type(item).__name__ for item in middleware]
    assert result == "compiled-agent"
    assert tool_names.count("save_progress") == 1
    assert tool_names.count("inspect_dependency") == 1
    assert tool_names.count("search_technical_sources") == 1
    assert tool_names.count("fetch_external_evidence") == 1
    assert tool_names.count("link_external_evidence") == 1
    assert middleware_names[:5] == [
        "MessageIdentityMiddleware",
        "LegacyContextMigrationMiddleware",
        "PromptPolicyMiddleware",
        "ProtectedContextMiddleware",
        "DeepFixCompactionMiddleware",
    ]
    assert not {
        "SummarizationMiddleware",
        "SummarizationToolMiddleware",
        "ContextMemoryMiddleware",
        "ResearchEvidenceMiddleware",
    } & set(middleware_names)
    identity, migration, prompt, protected, compaction = middleware[:5]
    assert isinstance(identity, MessageIdentityMiddleware)
    assert type(migration).__name__ == "LegacyContextMigrationMiddleware"
    assert isinstance(prompt, PromptPolicyMiddleware)
    assert isinstance(protected, ProtectedContextMiddleware)
    assert isinstance(compaction, DeepFixCompactionMiddleware)
    assert captured["model"] is main_model
    assert compaction.coordinator.model is compaction_model
    assert compaction.coordinator.model is not captured["model"]
    save_progress = next(tool for tool in captured["tools"] if tool.name == "save_progress")
    assert not any(
        hasattr(save_progress, name)
        for name in ("model", "main_model", "compaction_model")
    )
    assert tool_names.count("compact_conversation") == 1
    assert captured["subagents"] == []
    assert captured["skills"] == []
    assert captured["interrupt_on"] == {
        "write_file": True,
        "edit_file": True,
        "delete": True,
        "execute": True,
    }
    assert registered["key"] == f"deepseek:{config.main_model.model_name}"
    assert registered["profile"].excluded_middleware == frozenset(
        {"SummarizationMiddleware"}
    )
