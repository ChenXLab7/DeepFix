import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from deepfix.agent import (
    build_agent,
    build_compaction_model,
    build_experiment_agent,
    build_main_model,
)
from deepfix.approval import PolicyAction, RiskLevel
from deepfix.compaction.middleware import (
    DeepFixCompactionMiddleware,
    MessageIdentityMiddleware,
)
from deepfix.config import ApprovalMode, load_config
from deepfix.domain_repositories import DomainRepositories
from deepfix.extensions import (
    AgentExtensions,
    ToolRegistration,
    build_research_extensions,
)
from deepfix.investigation.models import InvestigationCapability
from deepfix.navigation.middleware import TodoNavigationMiddleware
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.task_domain.outcome import RepairOutcomeCandidate

EXPECTED_TODO_SYSTEM_PROMPT = """## Bug-repair task navigation

Use the native `write_todos` tool to keep a short plan for this bug-repair task.
Create the plan before starting a multi-step investigation. Keep at most one item
`in_progress`, mark it `completed` as soon as its work is actually done, and move the
next item to `in_progress`. Re-check the plan when new evidence changes the strategy.

Todo is navigation, not proof. A Todo status cannot establish a root cause, prove that
a file changed, prove that a test passed, authorize a tool, or declare the task fixed.
Use tool results and DeepFix's trusted evidence for those conclusions.
"""

EXPECTED_TODO_TOOL_DESCRIPTION = """Create or update a short Todo plan for this bug-repair task.

Keep at most one item `in_progress`. If any item is unfinished, exactly one item must
be `in_progress`; mark finished work `completed` promptly and advance the next item.

Todo is navigation only. Todo content or status cannot prove a root cause, file change,
test result, or repair outcome, and cannot authorize any tool."""


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
        "compact_conversation",
        "record_hypothesis",
        "search_diagnostic_artifacts",
        "read_diagnostic_artifact",
        "write_todos",
    } <= tools.keys()


def test_agent_interrupts_every_side_effecting_tool(agent):
    middleware = agent.nodes["HumanInTheLoopMiddleware.after_model"].bound.func.__self__

    assert set(middleware.interrupt_on) == {
        "write_file",
        "edit_file",
        "delete",
        "execute",
    }
    assert "save_progress" not in middleware.interrupt_on
    assert "compact_conversation" not in middleware.interrupt_on
    assert "search_diagnostic_artifacts" not in middleware.interrupt_on
    assert "read_diagnostic_artifact" not in middleware.interrupt_on
    assert "write_todos" not in middleware.interrupt_on


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
    assert main.request_timeout == 120
    assert compaction.request_timeout == 120
    assert main.max_retries == 0
    assert compaction.max_retries == 0


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
        extensions=extensions,
    )

    tool_names = [tool.name for tool in captured["tools"]]
    middleware = captured["middleware"]
    middleware_names = [type(item).__name__ for item in middleware]
    assert result == "compiled-agent"
    assert "save_progress" not in tool_names
    assert tool_names.count("inspect_dependency") == 1
    assert tool_names.count("search_technical_sources") == 1
    assert tool_names.count("fetch_external_evidence") == 1
    assert tool_names.count("link_external_evidence") == 1
    assert middleware_names[:9] == [
        "MessageIdentityMiddleware",
        "TodoListMiddleware",
        "TodoNavigationMiddleware",
        "LegacyContextMigrationMiddleware",
        "InvestigationMigrationMiddleware",
        "HumanInTheLoopMiddleware",
        "InvestigationMiddleware",
        "PromptPolicyMiddleware",
        "DeepFixCompactionMiddleware",
    ]
    assert not {
        "SummarizationMiddleware",
        "SummarizationToolMiddleware",
        "ContextMemoryMiddleware",
        "ResearchEvidenceMiddleware",
    } & set(middleware_names)
    identity, todo_list, todo_navigation, migration, _, approval, _, prompt, compaction = (
        middleware[:9]
    )
    assert isinstance(identity, MessageIdentityMiddleware)
    assert isinstance(todo_list, TodoListMiddleware)
    assert todo_list.system_prompt == EXPECTED_TODO_SYSTEM_PROMPT
    assert isinstance(todo_navigation, TodoNavigationMiddleware)
    assert todo_navigation.reminder_rounds == 3
    assert type(todo_navigation.feedback_source).__name__ == ("RepositoryNavigationFeedbackSource")
    assert type(migration).__name__ == "LegacyContextMigrationMiddleware"
    assert isinstance(approval, HumanInTheLoopMiddleware)
    assert isinstance(prompt, PromptPolicyMiddleware)
    assert isinstance(compaction, DeepFixCompactionMiddleware)
    assert middleware_names.count("DeepFixCompactionMiddleware") == 1
    assert "ProtectedContextMiddleware" not in middleware_names
    assert captured["model"] is main_model
    assert compaction.coordinator.model is compaction_model
    assert compaction.coordinator.model is not captured["model"]
    assert tool_names.count("compact_conversation") == 1
    assert tool_names.count("search_diagnostic_artifacts") == 1
    assert tool_names.count("read_diagnostic_artifact") == 1
    investigation_middleware = next(
        item for item in middleware if type(item).__name__ == "InvestigationMiddleware"
    )
    assert (
        investigation_middleware.artifacts.root_dir
        == (config.artifacts_path / "investigation_receipts").resolve()
    )
    assert investigation_middleware.capabilities["search_diagnostic_artifacts"] is (
        InvestigationCapability.READ
    )
    assert investigation_middleware.capabilities["read_diagnostic_artifact"] is (
        InvestigationCapability.READ
    )
    assert investigation_middleware.capabilities["write_todos"] is (InvestigationCapability.META)
    assert captured["subagents"] == []
    assert captured["skills"] == []
    assert captured["interrupt_on"] is None
    assert set(approval.interrupt_on) == {
        "write_file",
        "edit_file",
        "delete",
        "execute",
    }
    assert registered["key"] == f"deepseek:{config.main_model.model_name}"
    assert registered["profile"].excluded_middleware == frozenset({"SummarizationMiddleware"})


def test_agent_investigation_middleware_uses_shared_execution_repository(
    config,
    monkeypatch,
):
    captured = {}
    repositories = DomainRepositories.create(config.database_path)

    monkeypatch.setattr(
        "deepfix.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled-agent",
    )

    build_agent(
        config,
        checkpointer=InMemorySaver(),
        repositories=repositories,
    )

    middleware = next(
        item for item in captured["middleware"] if type(item).__name__ == "InvestigationMiddleware"
    )
    assert middleware.execution is repositories.execution


def test_agent_configures_write_todos_with_deepfix_navigation_contract(
    config,
    monkeypatch,
):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("deepfix.agent.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr("deepfix.agent.build_main_model", lambda config: object())
    monkeypatch.setattr("deepfix.agent.build_compaction_model", lambda config: object())

    build_agent(
        config,
        checkpointer=InMemorySaver(),
    )

    todo_list = next(
        item for item in captured["middleware"] if isinstance(item, TodoListMiddleware)
    )
    write_todos = next(tool for tool in todo_list.tools if tool.name == "write_todos")
    assert todo_list.tool_description == EXPECTED_TODO_TOOL_DESCRIPTION
    assert write_todos.description == EXPECTED_TODO_TOOL_DESCRIPTION


def test_experiment_builder_is_opt_in_and_legacy_builder_defaults_are_unchanged(
    config,
    monkeypatch,
):
    captured = []

    def fake_create_deep_agent(**kwargs):
        captured.append(kwargs)
        return kwargs["name"]

    monkeypatch.setattr("deepfix.agent.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr("deepfix.agent.build_main_model", lambda _config: object())
    monkeypatch.setattr("deepfix.agent.build_compaction_model", lambda _config: object())
    kwargs = {
        "config": config,
        "checkpointer": InMemorySaver(),
    }

    assert build_agent(**kwargs) == "deepfix_repair_agent"
    assert build_experiment_agent(**kwargs) == "deepfix_experiment_executor"

    legacy, experiment = captured
    assert legacy["response_format"] is RepairOutcomeCandidate
    assert any(isinstance(item, PromptPolicyMiddleware) for item in legacy["middleware"])
    assert experiment["response_format"].__name__ == "ExecutorNarrativeResult"
    assert not any(isinstance(item, PromptPolicyMiddleware) for item in experiment["middleware"])


@pytest.mark.parametrize(
    "name",
    ["search_diagnostic_artifacts", "read_diagnostic_artifact", "write_todos"],
)
def test_agent_rejects_extension_that_shadows_diagnostic_artifact_tool(
    config,
    name,
):
    extensions = AgentExtensions(
        tools=(
            ToolRegistration(
                tool=_extension_tool(name),
                risk=RiskLevel.L0,
                policy_action=PolicyAction.ALLOW,
                investigation_capability=InvestigationCapability.READ,
            ),
        )
    )

    with pytest.raises(ValueError, match="重复 Tool 名称"):
        build_agent(
            config,
            checkpointer=InMemorySaver(),
            extensions=extensions,
        )


class _RecordingMiddleware(AgentMiddleware):
    def __init__(self, label, calls):
        self.label = label
        self.calls = calls

    def wrap_model_call(self, request, handler):
        self.calls.append(f"{self.label}:before")
        response = handler(request)
        self.calls.append(f"{self.label}:after")
        return response


class _FirstRecordingMiddleware(_RecordingMiddleware):
    pass


class _SecondRecordingMiddleware(_RecordingMiddleware):
    pass


def test_model_middleware_list_is_outermost_first_for_installed_langchain():
    calls = []
    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        tools=[],
        middleware=[
            _FirstRecordingMiddleware("first", calls),
            _SecondRecordingMiddleware("second", calls),
        ],
    )

    agent.invoke({"messages": [HumanMessage(content="run")]})

    assert calls == [
        "first:before",
        "second:before",
        "second:after",
        "first:after",
    ]
