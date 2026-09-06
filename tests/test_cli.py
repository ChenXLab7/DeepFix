import sys

import pytest

from deepfix import cli as cli_module
from deepfix.cli import build_parser, main, print_task_list, run_interaction
from deepfix.persistence import TaskRepository
from deepfix.task_domain.models import (
    TaskDefinition,
    TaskLifecycleStatus,
)
from deepfix.task_domain.runtime import TaskRuntime


class CliServiceStub:
    def __init__(self):
        self.decisions = []
        self.pause_calls = []

    def decide(self, task_id, decisions):
        self.decisions.append((task_id, decisions))
        self.task = self.task.model_copy(update={"lifecycle": TaskLifecycleStatus.PAUSED})
        return self.task

    def pause_task(self, task_id, reason):
        self.pause_calls.append((task_id, reason))
        self.task = self.task.model_copy(
            update={"lifecycle": TaskLifecycleStatus.PAUSED, "pause_reason": reason}
        )
        return self.task


def waiting_task(tmp_path, policy_action: str, risk: str = "L1") -> TaskRuntime:
    return TaskRuntime(
        task_id=f"task-{tmp_path.name}",
        lifecycle=TaskLifecycleStatus.WAITING_APPROVAL,
        pending_actions=[
            {
                "name": "execute",
                "args": {"command": "pytest -q"},
                "description": "运行测试",
                "risk": risk,
                "policy_action": policy_action,
                "reason": "命令需要审批",
            }
        ],
    )


def _create_task(repository: TaskRepository, tmp_path, problem: str, mode: str):
    task_id = f"task-{mode}"
    definition = TaskDefinition(
        task_id=task_id,
        original_message_id=f"message-{mode}",
        original_problem=problem,
        approval_mode=mode,
        source_project_root=str(tmp_path),
        workspace_root=str(tmp_path),
        workspace_baseline_id=None,
        project_python=sys.executable,
        confinement_level="legacy_local",
        created_at="2026-08-31T00:00:00+00:00",
    )
    repository.create_definition(definition)
    return definition


def test_new_command_parses_project_problem_and_mode():
    args = build_parser().parse_args(
        ["new", "--project", "demo", "--mode", "manual", "除法结果错误"]
    )
    assert (args.command, args.project, args.problem, args.mode) == (
        "new",
        "demo",
        "除法结果错误",
        "manual",
    )


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


def test_resume_command_accepts_structured_handoff_metadata():
    args = build_parser().parse_args(
        ["resume", "abc123", "环境仅为 Python 3.11", "--kind", "constraint", "--supersedes", "old"]
    )

    assert (args.message, args.kind, args.supersedes) == (
        "环境仅为 Python 3.11", "constraint", "old",
    )


def test_once_can_leave_waiting_input_as_a_noninteractive_handoff(tmp_path):
    service = CliServiceStub()
    service.task = TaskRuntime(
        task_id=f"task-{tmp_path.name}",
        lifecycle=TaskLifecycleStatus.WAITING_INPUT,
        pending_actions=[],
        pause_reason="请提供复现命令",
    )
    output = []

    result = run_interaction(
        service, service.task, input_fn=lambda _prompt: pytest.fail("不应读取输入"),
        output_fn=output.append, once=True,
    )

    assert result.lifecycle is TaskLifecycleStatus.WAITING_INPUT
    assert any(service.task.task_id in line for line in output)
    assert any("等待用户补充" in line for line in output)


def test_waiting_input_displays_question_and_continues_with_reply(tmp_path):
    class InputService(CliServiceStub):
        def __init__(self):
            super().__init__()
            self.continue_calls = []

        def continue_task(self, task_id, text, *, input_kind, supersedes_input_id=None):
            self.continue_calls.append((task_id, text, input_kind, supersedes_input_id))
            self.task = self.task.model_copy(update={"lifecycle": TaskLifecycleStatus.PAUSED})
            return self.task

    service = InputService()
    service.task = TaskRuntime(
        task_id=f"task-{tmp_path.name}", lifecycle=TaskLifecycleStatus.WAITING_INPUT,
        pending_actions=[], pause_reason="请提供复现命令",
    )
    output = []

    result = run_interaction(service, service.task, input_fn=lambda _prompt: "pytest tests/test_x.py", output_fn=output.append)

    assert service.continue_calls == [(service.task.task_id, "pytest tests/test_x.py", "information", None)]
    assert any("请提供复现命令" in line for line in output)
    assert result.lifecycle is TaskLifecycleStatus.PAUSED


def test_waiting_input_eof_preserves_task(tmp_path):
    service = CliServiceStub()
    service.task = TaskRuntime(
        task_id=f"task-{tmp_path.name}", lifecycle=TaskLifecycleStatus.WAITING_INPUT,
        pending_actions=[], pause_reason="请提供复现命令",
    )
    output = []

    result = run_interaction(
        service, service.task, input_fn=lambda _prompt: (_ for _ in ()).throw(EOFError), output_fn=output.append,
    )

    assert result.lifecycle is TaskLifecycleStatus.WAITING_INPUT
    assert any("已保留" in line for line in output)


def test_waiting_input_handles_many_replies_without_recursion(tmp_path):
    class ManyReplyService(CliServiceStub):
        def __init__(self):
            super().__init__()
            self.continues = 0

        def continue_task(self, task_id, text, *, input_kind, supersedes_input_id=None):
            self.continues += 1
            lifecycle = (
                TaskLifecycleStatus.PAUSED
                if self.continues == 1_100
                else TaskLifecycleStatus.WAITING_INPUT
            )
            self.task = self.task.model_copy(update={"lifecycle": lifecycle})
            return self.task

    service = ManyReplyService()
    service.task = TaskRuntime(
        task_id=f"task-{tmp_path.name}", lifecycle=TaskLifecycleStatus.WAITING_INPUT,
        pending_actions=[], pause_reason="请继续提供线索",
    )
    replies = iter(["下一条线索"] * 1_100)

    result = run_interaction(
        service, service.task, input_fn=lambda _prompt: next(replies), output_fn=lambda _line: None,
    )

    assert service.continues == 1_100
    assert result.lifecycle is TaskLifecycleStatus.PAUSED


def test_approval_prompt_eof_preserves_task(tmp_path):
    service = CliServiceStub()
    service.task = waiting_task(tmp_path, "ask")
    output = []

    result = run_interaction(
        service, service.task, input_fn=lambda _prompt: (_ for _ in ()).throw(EOFError), output_fn=output.append,
    )

    assert result.lifecycle is TaskLifecycleStatus.WAITING_APPROVAL
    assert service.decisions == []
    assert any("已保留" in line for line in output)


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
    assert result.lifecycle is TaskLifecycleStatus.PAUSED


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
        input_fn=lambda _prompt: "q",
        output_fn=lambda _line: None,
    )
    assert service.decisions == []
    assert service.pause_calls == [(service.task.task_id, "用户从终端暂停审批")]
    assert result.pending_actions


def test_list_displays_definition_lifecycle_project_and_problem(tmp_path):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")
    task = _create_task(repository, tmp_path, "排序结果不稳定", "guarded")
    output = []
    print_task_list(repository, output_fn=output.append)
    line = output[0]
    assert task.task_id in line
    assert TaskLifecycleStatus.CREATED.value in line
    assert task.source_project_root in line
    assert task.original_problem in line


def test_main_list_does_not_require_model_api_key(tmp_path, monkeypatch):
    database_path = tmp_path / "deepfix.sqlite3"
    repository = TaskRepository(database_path)
    _create_task(repository, tmp_path, "测试失败", "manual")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "state_database_path", lambda: database_path)
    output = []
    assert main(["list"], output_fn=output.append) == 0
    assert "测试失败" in output[0]
