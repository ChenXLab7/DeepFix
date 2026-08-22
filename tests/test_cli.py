import pytest
from langchain_core.tools import StructuredTool

from deepfix import cli as cli_module
from deepfix.cli import build_parser, main, print_task_list, run_interaction
from deepfix.config import ApprovalMode
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository
from deepfix.research.store import ResearchEvidenceStore


class CliServiceStub:
    def __init__(self):
        self.decisions = []
        self.pause_calls = []

    def decide(self, task_id, decisions):
        self.decisions.append((task_id, decisions))
        task = self.task
        task.status = TaskStatus.PAUSED
        return task

    def pause_task(self, task_id, reason):
        self.pause_calls.append((task_id, reason))
        self.task.status = TaskStatus.PAUSED
        return self.task


def waiting_task(tmp_path, policy_action: str, risk: str = "L1") -> TaskState:
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.status = TaskStatus.WAITING_APPROVAL
    task.pending_actions = [
        {
            "name": "execute",
            "args": {"command": "pytest -q"},
            "description": "运行测试",
            "risk": risk,
            "policy_action": policy_action,
            "reason": "命令需要审批",
        }
    ]
    return task


def test_new_command_parses_project_problem_and_mode():
    args = build_parser().parse_args(
        [
            "new",
            "--project",
            "demo",
            "--mode",
            "manual",
            "除法结果错误",
        ]
    )

    assert args.command == "new"
    assert args.project == "demo"
    assert args.problem == "除法结果错误"
    assert args.mode == "manual"


def test_new_command_accepts_target_project_python():
    args = build_parser().parse_args(
        [
            "new",
            "--project",
            "demo",
            "--python",
            "demo/.venv/Scripts/python.exe",
            "测试失败",
        ]
    )

    assert args.python == "demo/.venv/Scripts/python.exe"


def test_resume_command_requires_task_id():
    args = build_parser().parse_args(["resume", "abc123"])

    assert args.task_id == "abc123"
    assert args.message is None


def test_missing_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_manual_action_prompts_and_approves(tmp_path):
    service = CliServiceStub()
    service.task = waiting_task(tmp_path, "ask")
    output = []

    result = run_interaction(
        service,
        service.task,
        input_fn=lambda prompt: output.append(prompt) or "a",
        output_fn=output.append,
    )

    assert service.decisions == [(service.task.task_id, ["approve"])]
    assert any("[L1] execute: pytest -q" in line for line in output)
    assert result.status is TaskStatus.PAUSED


def test_denied_l3_action_never_offers_approval(tmp_path):
    service = CliServiceStub()
    service.task = waiting_task(tmp_path, "deny", risk="L3")
    output = []

    run_interaction(
        service,
        service.task,
        input_fn=lambda prompt: pytest.fail(f"不应请求输入: {prompt}"),
        output_fn=output.append,
    )

    assert service.decisions == [(service.task.task_id, ["reject"])]
    assert any("强制拒绝" in line for line in output)


def test_q_pauses_same_task_without_deciding_pending_action(tmp_path):
    service = CliServiceStub()
    service.task = waiting_task(tmp_path, "ask")

    result = run_interaction(
        service,
        service.task,
        input_fn=lambda prompt: "q",
        output_fn=lambda line: None,
    )

    assert service.decisions == []
    assert service.pause_calls == [(service.task.task_id, "用户从终端暂停审批")]
    assert result.task_id == service.task.task_id
    assert result.pending_actions


def test_list_displays_task_identity_status_project_and_problem(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")
    task = TaskState.create(tmp_path, "排序结果不稳定", ApprovalMode.GUARDED)
    repository.save(task)
    output = []

    print_task_list(repository, output_fn=output.append)

    line = output[0]
    assert task.task_id in line
    assert task.status.value in line
    assert task.project_root in line
    assert task.user_problem in line


def test_main_list_does_not_require_model_api_key(tmp_path, monkeypatch):
    database_path = tmp_path / "deepfix.sqlite3"
    repository = TaskRepository(database_path)
    repository.save(TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "state_database_path", lambda: database_path)
    output = []

    exit_code = main(["list"], output_fn=output.append)

    assert exit_code == 0
    assert "测试失败" in output[0]


def test_new_command_shares_one_memory_store_between_agent_and_service(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "state" / "deepfix.sqlite3"
    captures = {}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(database_path.parent))
    monkeypatch.setattr(cli_module, "state_database_path", lambda: database_path)

    def fake_build_agent(
        config,
        checkpointer,
        working_memory_store,
        task_repository=None,
        compaction_store=None,
        extensions=None,
        research_evidence_store=None,
        backend=None,
    ):
        captures["agent_store"] = working_memory_store
        captures["agent_research_store"] = research_evidence_store
        captures["extensions"] = extensions
        captures["agent_repository"] = task_repository
        captures["agent_compaction_store"] = compaction_store
        captures["agent_backend"] = backend
        return object()

    class FakeService:
        def __init__(
            self,
            agent,
            repository,
            policy,
            config,
            working_memory_store,
            research_evidence_store,
            compaction_store,
        ):
            captures["service_store"] = working_memory_store
            captures["service_research_store"] = research_evidence_store
            captures["service_compaction_store"] = compaction_store
            captures["service_repository"] = repository
            self.config = config

        def start(self, problem):
            task = TaskState.create(
                self.config.project_root,
                problem,
                self.config.approval_mode,
            )
            task.status = TaskStatus.CLARIFYING
            task.pending_question = "请提供失败堆栈"
            return task

    monkeypatch.setattr(cli_module, "build_agent", fake_build_agent)
    monkeypatch.setattr(cli_module, "BugfixService", FakeService)

    exit_code = main(
        ["new", "--project", str(tmp_path), "测试失败"],
        output_fn=lambda line: None,
    )

    assert exit_code == 0
    assert captures["agent_store"] is captures["service_store"]
    assert captures["agent_research_store"] is captures["service_research_store"]
    assert isinstance(captures["agent_research_store"], ResearchEvidenceStore)
    assert captures["agent_repository"] is captures["service_repository"]
    assert captures["agent_compaction_store"] is captures["service_compaction_store"]
    assert captures["agent_backend"] is not None
    assert {item.tool.name for item in captures["extensions"].tools} == {
        "inspect_dependency",
        "search_technical_sources",
        "fetch_external_evidence",
        "link_external_evidence",
    }


def test_new_persists_explicit_project_python(tmp_path, monkeypatch):
    database_path = tmp_path / "state" / "deepfix.sqlite3"
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    project_python.parent.mkdir(parents=True)
    project_python.touch()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(database_path.parent))
    monkeypatch.setattr(cli_module, "state_database_path", lambda: database_path)

    class FakeAgent:
        def invoke(self, value, config):
            return {
                "structured_response": {
                    "status": "needs_input",
                    "question": "请提供失败堆栈",
                    "summary": "等待补充",
                },
                "messages": [],
            }

    monkeypatch.setattr(cli_module, "build_agent", lambda *args, **kwargs: FakeAgent())

    exit_code = main(
        [
            "new",
            "--project",
            str(tmp_path),
            "--python",
            str(project_python),
            "测试失败",
        ],
        output_fn=lambda line: None,
    )
    stored = TaskRepository(database_path).list_recent()[0]

    assert exit_code == 0
    assert stored.project_python == str(project_python.resolve())


def test_resume_reuses_persisted_project_python(tmp_path, monkeypatch):
    database_path = tmp_path / "state" / "deepfix.sqlite3"
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    project_python.parent.mkdir(parents=True)
    project_python.touch()
    repository = TaskRepository(database_path)
    task = TaskState.create(
        tmp_path,
        "测试失败",
        ApprovalMode.MANUAL,
        project_python,
    )
    task.status = TaskStatus.CLARIFYING
    task.pending_question = "请提供失败堆栈"
    repository.save(task)
    captures = {}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(database_path.parent))
    monkeypatch.setattr(cli_module, "state_database_path", lambda: database_path)

    def fake_build_agent(config, *args, **kwargs):
        captures["project_python"] = config.project_python
        return object()

    class FakeService:
        def __init__(self, *args):
            pass

    monkeypatch.setattr(cli_module, "build_agent", fake_build_agent)
    monkeypatch.setattr(cli_module, "BugfixService", FakeService)

    exit_code = main(
        ["resume", task.task_id],
        output_fn=lambda line: None,
    )

    assert exit_code == 0
    assert captures["project_python"] == project_python.resolve()


def test_research_http_client_uses_bounded_timeouts():
    with cli_module.build_research_client() as client:
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 15.0
        assert client.timeout.write == 15.0
        assert client.timeout.pool == 15.0


def _tool(name):
    def run():
        return "ok"

    return StructuredTool.from_function(run, name=name, description="test")


def test_research_factory_shares_client_store_and_fetcher_without_optional_keys(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "state" / "deepfix.sqlite3"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(database_path.parent))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEPFIX_SEARCH_PROVIDER", raising=False)
    config = cli_module.load_config(tmp_path, ApprovalMode.MANUAL)
    store = ResearchEvidenceStore(config.database_path)
    client = object()
    backend = object()
    captures = {"provider_clients": []}

    class Provider:
        enabled = True
        name = "fake"

        def __init__(self, received_client, *args, **kwargs):
            captures["provider_clients"].append(received_client)

    class Composite:
        def __init__(self, providers):
            captures["providers"] = providers

    class Fetcher:
        def __init__(self, received_client, *args, **kwargs):
            captures["fetch_client"] = received_client

    monkeypatch.setattr(cli_module, "PyPIProvider", Provider)
    monkeypatch.setattr(cli_module, "GitHubProvider", Provider)
    monkeypatch.setattr(cli_module, "TavilyProvider", Provider)
    monkeypatch.setattr(cli_module, "CompositeTechnicalSearchProvider", Composite)
    monkeypatch.setattr(cli_module, "SafeEvidenceFetcher", Fetcher)
    monkeypatch.setattr(
        cli_module,
        "build_inspect_dependency_tool",
        lambda inspector: _tool("inspect_dependency"),
    )

    def search_tool(sanitizer, inspector, provider, received_store):
        captures["search_store"] = received_store
        captures["search_provider"] = provider
        return _tool("search_technical_sources")

    def fetch_tool(fetcher, received_store, received_backend):
        captures["fetcher"] = fetcher
        captures["fetch_store"] = received_store
        captures["backend"] = received_backend
        return _tool("fetch_external_evidence")

    monkeypatch.setattr(cli_module, "build_search_technical_sources_tool", search_tool)
    monkeypatch.setattr(cli_module, "build_fetch_external_evidence_tool", fetch_tool)
    monkeypatch.setattr(
        cli_module,
        "build_link_external_evidence_tool",
        lambda received_store: _tool("link_external_evidence"),
    )

    extensions = cli_module.build_cli_research_extensions(
        config,
        store,
        client,
        backend,
    )

    assert captures["provider_clients"] == [client, client, client]
    assert captures["fetch_client"] is client
    assert captures["search_store"] is store
    assert captures["fetch_store"] is store
    assert captures["fetcher"] is not None
    assert captures["backend"] is backend
    assert len(extensions.tools) == 4


def test_run_interaction_queries_report_evidence_for_current_task_only(tmp_path):
    service = CliServiceStub()
    task = TaskState.create(tmp_path, "测试失败", ApprovalMode.MANUAL)
    task.status = TaskStatus.PAUSED
    service.task = task

    class Store:
        def __init__(self):
            self.calls = []

        def list_evidence(self, task_id):
            self.calls.append(task_id)
            return []

    store = Store()

    run_interaction(
        service,
        task,
        research_evidence_store=store,
        output_fn=lambda line: None,
    )

    assert store.calls == [task.task_id]
