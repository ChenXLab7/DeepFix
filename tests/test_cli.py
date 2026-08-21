import pytest

from deepfix import cli as cli_module
from deepfix.cli import build_parser, main, print_task_list, run_interaction
from deepfix.config import ApprovalMode
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository


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

    def fake_build_agent(config, checkpointer, working_memory_store):
        captures["agent_store"] = working_memory_store
        return object()

    class FakeService:
        def __init__(self, agent, repository, policy, config, working_memory_store):
            captures["service_store"] = working_memory_store
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
