from collections import deque
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt

from deepfix.approval import ApprovalPolicy
from deepfix.config import ApprovalMode, load_config
from deepfix.models import RepairOutcome, TaskStatus
from deepfix.persistence import TaskRepository
from deepfix.service import BugfixService


class FakeAgent:
    def __init__(self, *results):
        self.results = deque(results)
        self.invoke_calls = []

    def invoke(self, value, config):
        self.invoke_calls.append((value, config))
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def make_interrupt(name: str, args: dict[str, object]):
    return {
        "__interrupt__": (
            Interrupt(
                value={
                    "action_requests": [
                        {"name": name, "args": args, "description": "待审批操作"}
                    ],
                    "review_configs": [
                        {
                            "action_name": name,
                            "allowed_decisions": ["approve", "reject"],
                        }
                    ],
                },
                id="interrupt-1",
            ),
        )
    }


def outcome(status="needs_input", **overrides):
    values = {
        "status": status,
        "question": "请补充失败堆栈" if status == "needs_input" else None,
        "summary": "等待补充" if status == "needs_input" else "处理完成",
    }
    values.update(overrides)
    return {"structured_response": RepairOutcome(**values), "messages": []}


def passing_outcome(command="pytest -q"):
    return {
        "structured_response": RepairOutcome(
            status="completed",
            diagnosis="边界条件错误",
            hypotheses=["计算分支选择错误"],
            repair_plan=["修正条件判断"],
            summary="缺陷已修复",
            review_summary="修改范围最小",
        ),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": command},
                        "id": "test-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="1 passed",
                name="execute",
                tool_call_id="test-call-1",
                artifact={"exit_code": 0},
            ),
        ],
    }


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    return load_config(tmp_path, ApprovalMode.MANUAL)


def make_service(config, fake_agent):
    repository = TaskRepository(config.database_path)
    return (
        BugfixService(
            fake_agent,
            repository,
            ApprovalPolicy(config.approval_mode),
            config,
        ),
        repository,
    )


def test_start_passes_problem_and_stable_thread_id(app_config):
    fake_agent = FakeAgent(outcome())
    service, _ = make_service(app_config, fake_agent)

    task = service.start("除法结果错误")

    value, config = fake_agent.invoke_calls[0]
    assert value == {"messages": [{"role": "user", "content": "除法结果错误"}]}
    assert config["configurable"]["thread_id"] == task.task_id
    assert task.agent_invocations == 1


def test_interrupt_is_persisted_as_waiting_approval(app_config):
    fake_agent = FakeAgent(make_interrupt("execute", {"command": "pytest -q"}))
    service, repository = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.WAITING_APPROVAL
    assert task.shell_calls == 1
    assert service.pending_actions(task.task_id)[0]["name"] == "execute"
    assert repository.get(task.task_id).pending_actions == task.pending_actions


def test_l3_operation_is_rejected_even_when_user_requests_approval(app_config):
    fake_agent = FakeAgent(
        make_interrupt("execute", {"command": "git reset --hard"}),
        outcome(),
    )
    service, _ = make_service(app_config, fake_agent)
    task = service.start("清理项目")

    resumed = service.decide(task.task_id, ["approve"])

    resume_value, _ = fake_agent.invoke_calls[-1]
    assert isinstance(resume_value, Command)
    assert resume_value.resume["decisions"][0]["type"] == "reject"
    assert resumed.approvals[-1].decision == "reject"


def test_guarded_l1_operation_is_automatically_approved(app_config):
    guarded = replace(app_config, approval_mode=ApprovalMode.GUARDED)
    fake_agent = FakeAgent(
        make_interrupt("execute", {"command": "pytest -q"}),
        passing_outcome(),
    )
    service, _ = make_service(guarded, fake_agent)

    task = service.start("测试失败")

    resume_value, _ = fake_agent.invoke_calls[1]
    assert resume_value.resume["decisions"] == [{"type": "approve"}]
    assert task.status is TaskStatus.COMPLETED


def test_agent_exception_is_saved_as_failed(app_config):
    fake_agent = FakeAgent(RuntimeError("model unavailable"))
    service, repository = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.FAILED
    assert "model unavailable" in task.final_summary
    assert repository.get(task.task_id).status is TaskStatus.FAILED


def test_shell_budget_pauses_before_command_is_resumed(app_config):
    limited = replace(app_config, max_shell_calls=0)
    fake_agent = FakeAgent(make_interrupt("execute", {"command": "pytest -q"}))
    service, _ = make_service(limited, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert len(fake_agent.invoke_calls) == 1
    assert task.pending_actions


def test_agent_invocation_budget_pauses_before_calling_agent(app_config):
    limited = replace(app_config, max_agent_invocations=0)
    fake_agent = FakeAgent()
    service, _ = make_service(limited, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert fake_agent.invoke_calls == []


def test_changed_file_budget_pauses_before_approved_edit_is_resumed(app_config):
    limited = replace(app_config, max_changed_files=0)
    fake_agent = FakeAgent(
        make_interrupt(
            "edit_file",
            {"file_path": "/src/calc.py", "old_string": "x", "new_string": "y"},
        )
    )
    service, _ = make_service(limited, fake_agent)
    task = service.start("计算错误")

    paused = service.decide(task.task_id, ["approve"])

    assert paused.status is TaskStatus.PAUSED
    assert len(fake_agent.invoke_calls) == 1
    assert paused.pending_actions


def test_consecutive_test_failure_budget_pauses_task(app_config):
    limited = replace(app_config, max_consecutive_test_failures=1)
    failed_result = passing_outcome()
    failed_result["structured_response"] = outcome()["structured_response"]
    failed_result["messages"][-1].artifact = {"exit_code": 1}
    failed_result["messages"][-1].content = "1 failed"
    fake_agent = FakeAgent(failed_result)
    service, _ = make_service(limited, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert task.consecutive_test_failures == 1
    assert "连续测试失败" in task.final_summary


def test_rejected_action_records_approval_and_resumes_agent(app_config):
    fake_agent = FakeAgent(
        make_interrupt("write_file", {"file_path": "/src/calc.py", "content": "x"}),
        outcome(),
    )
    service, repository = make_service(app_config, fake_agent)
    task = service.start("计算错误")

    result = service.decide(task.task_id, ["reject"])

    assert result.approvals[-1].operation == "write_file"
    assert result.approvals[-1].decision == "reject"
    assert repository.get(task.task_id).approvals[-1].decision == "reject"


def test_continue_clarifying_task_appends_message_without_new_task(app_config):
    fake_agent = FakeAgent(outcome(), outcome(question="请提供 Python 版本"))
    service, _ = make_service(app_config, fake_agent)
    task = service.start("测试失败")

    continued = service.continue_task(task.task_id, "Python 3.12")

    value, config = fake_agent.invoke_calls[-1]
    assert continued.task_id == task.task_id
    assert value == {"messages": [{"role": "user", "content": "Python 3.12"}]}
    assert config["configurable"]["thread_id"] == task.task_id
    assert continued.conversation[-1]["content"] == "Python 3.12"


def test_completed_outcome_requires_and_records_passing_test(app_config):
    fake_agent = FakeAgent(passing_outcome())
    service, _ = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.COMPLETED
    assert task.test_results[0].command == "pytest -q"
    assert task.test_results[0].exit_code == 0
    assert task.final_summary == "缺陷已修复"


def test_completed_outcome_without_passing_test_is_paused(app_config):
    fake_agent = FakeAgent(outcome(status="completed", summary="声称已完成"))
    service, _ = make_service(app_config, fake_agent)

    task = service.start("测试失败")

    assert task.status is TaskStatus.PAUSED
    assert "通过的测试证据" in task.final_summary


def test_repeated_graph_message_history_does_not_duplicate_test_result(app_config):
    first = passing_outcome()
    first["structured_response"] = outcome()["structured_response"]
    second = passing_outcome()
    second["structured_response"] = outcome(question="还需要版本信息")[
        "structured_response"
    ]
    fake_agent = FakeAgent(first, second)
    service, _ = make_service(app_config, fake_agent)
    task = service.start("测试失败")

    continued = service.continue_task(task.task_id, "Python 3.12")

    assert len(continued.test_results) == 1
