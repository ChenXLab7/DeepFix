from dataclasses import replace

import pytest

from deepfix.domain_repositories import DomainRepositories
from deepfix.protected_context import ProtectedContextBuilder
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.outcome import RepairOutcomeCandidate
from task_domain.test_service_integration import FakeAgent, service_for
from task_domain.test_service_integration import config as config_fixture

config = config_fixture


def asks(question="哪个 Python 版本出现问题？"):
    return {
        "messages": [],
        "structured_response": RepairOutcomeCandidate(
            status="needs_input",
            summary="已调查入口，需要环境信息",
            question=question,
        ),
    }


def test_question_is_normal_handoff_and_inputs_survive_new_service(config):
    first = service_for(config, FakeAgent(asks()), workspace=True)
    task = first.start("修复解析错误")
    assert task.lifecycle is TaskLifecycleStatus.WAITING_INPUT
    assert task.pause_reason == "哪个 Python 版本出现问题？"
    assert (
        service_for(config, FakeAgent()).get_runtime(task.task_id).pause_reason == task.pause_reason
    )
    definition = first.repository.get_definition(task.task_id)
    second = service_for(config, FakeAgent(asks("请提供触发输入")), workspace=True)
    resumed = second.continue_task(task.task_id, "只在 Python 3.11 发生")
    assert resumed.task_id == task.task_id
    assert (
        second.repository.get_definition(task.task_id).workspace_root == definition.workspace_root
    )
    inputs = second.repository.list_inputs(task.task_id)
    assert [item.text for item in inputs] == ["修复解析错误", "只在 Python 3.11 发生"]
    assert len(second.repository.list_runs(task.task_id)) == 2
    context = ProtectedContextBuilder(DomainRepositories.create(config.database_path)).build(
        task.task_id,
        [],
        None,
    )
    assert not context.task_anchor.user_constraints
    assert context.user_inputs[-1].text == "只在 Python 3.11 发生"


def test_continue_control_is_not_new_evidence_or_constraint(config):
    service = service_for(config, FakeAgent(asks(), asks()))
    task = service.start("修复错误")
    service.continue_task(task.task_id, "继续")
    inputs = service.repository.list_inputs(task.task_id)
    assert inputs[-1].kind == "control"
    assert (
        not ProtectedContextBuilder(service.repositories)
        .build(
            task.task_id,
            [],
            None,
        )
        .task_anchor.user_constraints
    )


def test_unfinished_candidate_continues_autonomously(config):
    agent = FakeAgent(
        {
            "messages": [],
            "structured_response": RepairOutcomeCandidate(
                status="completed",
                summary="尚未实际运行测试",
            ),
        },
        asks(),
    )
    service = service_for(config, agent, workspace=True)
    task = service.start("修复错误")
    assert agent.invocations == 2
    assert task.lifecycle is TaskLifecycleStatus.WAITING_INPUT
    assert len(service.repository.list_runs(task.task_id)) == 1


def test_approval_resume_keeps_run_budget_across_service_restart(config):
    from task_domain.test_service_integration import approval_interrupt

    config = replace(config, max_agent_invocations=1)
    first = service_for(config, FakeAgent(approval_interrupt()))
    task = first.start("修复错误")
    second_agent = FakeAgent(asks())
    second = service_for(config, second_agent)
    second_agent.interrupts = first.agent.interrupts
    result = second.decide(task.task_id, ["approve"])
    assert result.lifecycle is TaskLifecycleStatus.PAUSED
    assert second_agent.invocations == 0
    assert "预算" in result.pause_reason


def test_constraint_supersession_survives_context_without_messages(config):
    service = service_for(config, FakeAgent(asks(), asks(), asks()))
    task = service.start("修复错误")
    service.continue_task(task.task_id, "不要修改测试", input_kind="constraint")
    old = service.repository.list_inputs(task.task_id)[-1]
    service.continue_task(
        task.task_id, "允许添加回归测试", input_kind="constraint", supersedes_input_id=old.input_id
    )
    context = ProtectedContextBuilder(service.repositories).build(task.task_id, [], None)
    assert [item.text for item in context.task_anchor.user_constraints] == ["允许添加回归测试"]


def test_workspace_denial_is_shown_before_approval_and_cannot_be_overridden(config):
    from task_domain.test_service_integration import approval_interrupt

    interrupt = approval_interrupt()
    interrupt["__interrupt__"][0].value["action_requests"][0]["args"]["command"] = (
        "/untrusted/python -m pytest -q"
    )
    agent = FakeAgent(interrupt, asks())
    service = service_for(config, agent, workspace=True)
    task = service.start("修复错误")
    assert task.pending_actions[0]["policy_action"] == "deny"
    service.decide(task.task_id, ["approve"])
    assert service.repositories.execution.list_approvals(task.task_id)[0].decision == "reject"


def test_run_cannot_reference_another_tasks_input(config):
    from deepfix.task_domain.models import TaskRun

    service = service_for(config, FakeAgent(asks(), asks()))
    first = service.start("first bug")
    second = service.start("second bug")
    with pytest.raises(ValueError, match="input"):
        service.repository.start_run(
            TaskRun(
                run_id="cross-task-run",
                task_id=second.task_id,
                input_id=service.repository.list_inputs(first.task_id)[0].input_id,
                created_at="2026-09-06T00:00:00Z",
            )
        )


def test_input_and_run_commit_atomically(config, monkeypatch):
    service = service_for(config, FakeAgent(asks()))
    task = service.start("repair bug")
    before = service.repository.list_inputs(task.task_id)

    def fail(*args):
        raise RuntimeError("simulated run insert failure")

    monkeypatch.setattr(service.repository, "_insert_run", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        service.continue_task(task.task_id, "new environment details")
    assert service.repository.list_inputs(task.task_id) == before


def test_second_service_cannot_start_overlapping_run(config):
    from deepfix.task_domain.models import TaskLifecycleConflict

    first = service_for(config, FakeAgent(asks()))
    task = first.start("repair bug")
    second = service_for(config, FakeAgent(asks()))
    before = first.repository.list_inputs(task.task_id)
    with (
        first.repository.execution_lock(task.task_id),
        pytest.raises(
            TaskLifecycleConflict,
            match="already running",
        ),
    ):
        second.continue_task(task.task_id, "continue")
    assert second.repository.list_inputs(task.task_id) == before
    assert (
        second.continue_task(task.task_id, "continue").lifecycle
        is TaskLifecycleStatus.WAITING_INPUT
    )
