from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping

from langgraph.checkpoint.sqlite import SqliteSaver

from deepfix.agent import build_agent
from deepfix.approval import ApprovalPolicy
from deepfix.config import ApprovalMode, load_config, state_database_path
from deepfix.memory import WorkingMemoryStore
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository
from deepfix.reporting import render_report
from deepfix.service import BugfixService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="创建新的代码修复任务")
    new_parser.add_argument("--project", required=True, help="目标 Python 项目路径")
    new_parser.add_argument(
        "--mode",
        choices=("manual", "guarded"),
        default="manual",
        help="审批模式",
    )
    new_parser.add_argument("problem", help="需要调查和修复的问题")

    resume_parser = subparsers.add_parser("resume", help="恢复已有任务")
    resume_parser.add_argument("task_id", help="任务 ID")
    resume_parser.add_argument("message", nargs="?", help="补充给 Agent 的信息")

    subparsers.add_parser("list", help="列出最近任务")
    return parser


def run_interaction(
    service,
    task: TaskState,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> TaskState:
    while task.status is TaskStatus.WAITING_APPROVAL and task.pending_actions:
        decisions: list[str] = []
        for action in task.pending_actions:
            risk = str(action.get("risk", "L2"))
            name = str(action.get("name", "unknown"))
            args = action.get("args", {})
            args = args if isinstance(args, Mapping) else {}
            summary = _action_summary(name, args)
            reason = str(action.get("reason", action.get("description", "")))
            policy_action = str(action.get("policy_action", "ask"))
            output_fn(f"[{risk}] {name}: {summary}")
            output_fn(f"原因: {reason or '未提供'}")

            if policy_action == "deny":
                output_fn("策略判定为 L3，强制拒绝；不会提供批准选项。")
                decisions.append("reject")
                continue
            if policy_action == "allow":
                output_fn("策略允许，自动批准。")
                decisions.append("approve")
                continue

            while True:
                choice = input_fn("选择 [a]批准 / [r]拒绝 / [q]暂停: ").strip().lower()
                if choice == "q":
                    return service.pause_task(task.task_id, "用户从终端暂停审批")
                if choice in {"a", "r"}:
                    decisions.append("approve" if choice == "a" else "reject")
                    break
                output_fn("无效选择，请输入 a、r 或 q。")
        task = service.decide(task.task_id, decisions)

    output_fn(f"任务 ID: {task.task_id}")
    if task.status is TaskStatus.CLARIFYING:
        output_fn(f"需要补充信息: {task.pending_question or '请提供更多信息'}")
    else:
        output_fn(render_report(task))
    return task


def print_task_list(
    repository: TaskRepository,
    *,
    output_fn: Callable[[str], None] = print,
) -> None:
    tasks = repository.list_recent()
    if not tasks:
        output_fn("暂无任务")
        return
    for task in tasks:
        output_fn(
            f"{task.task_id}\t{task.status.value}\t{task.project_root}\t{task.user_problem}"
        )


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    read_input = input_fn or input
    write_output = output_fn or print
    repository = TaskRepository(state_database_path())

    if args.command == "list":
        print_task_list(repository, output_fn=write_output)
        return 0

    if args.command == "new":
        config = load_config(args.project, ApprovalMode(args.mode))
        stored_task = None
    else:
        try:
            stored_task = repository.get(args.task_id)
        except KeyError:
            write_output(f"任务不存在: {args.task_id}")
            return 2
        config = load_config(
            stored_task.project_root,
            ApprovalMode(stored_task.approval_mode),
        )

    working_memory_store = WorkingMemoryStore(config.database_path)
    with repository.checkpoint_connection() as connection:
        checkpointer = SqliteSaver(connection)
        agent = build_agent(
            config,
            checkpointer,
            working_memory_store,
        )
        service = BugfixService(
            agent,
            repository,
            ApprovalPolicy(config.approval_mode),
            config,
            working_memory_store,
        )
        if args.command == "new":
            task = service.start(args.problem)
        else:
            assert stored_task is not None
            task = _resume_from_cli(service, stored_task, args.message, write_output)
            if task is None:
                return 2
        run_interaction(
            service,
            task,
            input_fn=read_input,
            output_fn=write_output,
        )
    return 0


def _resume_from_cli(
    service: BugfixService,
    task: TaskState,
    message: str | None,
    output_fn: Callable[[str], None],
) -> TaskState | None:
    if task.status is TaskStatus.CLARIFYING and not message:
        return task
    if task.status is TaskStatus.WAITING_APPROVAL and task.pending_actions:
        return task
    if (
        task.status is TaskStatus.PAUSED
        and task.paused_from is TaskStatus.WAITING_APPROVAL
        and task.pending_actions
        and not message
    ):
        return service.continue_task(task.task_id)
    if not message or not message.strip():
        output_fn("恢复该任务需要提供 MESSAGE。")
        return None
    return service.continue_task(task.task_id, message)


def _action_summary(name: str, args: Mapping[str, object]) -> str:
    if name == "execute":
        return str(args.get("command", ""))
    if name in {"write_file", "edit_file", "delete"}:
        return str(args.get("file_path", args.get("path", "")))
    return str(dict(args))


if __name__ == "__main__":
    raise SystemExit(main())
