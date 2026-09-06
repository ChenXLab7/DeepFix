from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver

from deepfix.agent import build_agent
from deepfix.approval import ApprovalPolicy
from deepfix.backend import build_backend
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.config import ApprovalMode, load_config, state_database_path
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.extensions import AgentExtensions, build_research_extensions
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.investigation.token_budget import TokenBudgetStore
from deepfix.operations import OperationReconciler
from deepfix.persistence import TaskRepository
from deepfix.reporting import build_task_report_view, render_report
from deepfix.research.dependency import DependencyInspector
from deepfix.research.fetcher import SafeEvidenceFetcher
from deepfix.research.providers import (
    CompositeTechnicalSearchProvider,
    GitHubProvider,
    PyPIProvider,
    TavilyProvider,
)
from deepfix.research.sanitizer import QuerySanitizer, UrlSafetyPolicy
from deepfix.research.tools import (
    build_fetch_external_evidence_tool,
    build_inspect_dependency_tool,
    build_link_external_evidence_tool,
    build_search_technical_sources_tool,
)
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.runtime import TaskRuntime
from deepfix.workspace import WorkspaceFactory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="创建新的代码修复任务")
    new_parser.add_argument("--project", required=True, help="目标 Python 项目路径")
    new_parser.add_argument(
        "--python",
        help="目标项目使用的 Python 解释器路径",
    )
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
    resume_parser.add_argument(
        "--kind",
        choices=("information", "control", "direction", "hypothesis", "constraint"),
        default="information",
        help="补充信息的类型",
    )
    resume_parser.add_argument("--supersedes", help="被本次补充替代的输入 ID")
    new_parser.add_argument("--once", action="store_true", help="单次运行后交还终端")
    resume_parser.add_argument("--once", action="store_true", help="单次运行后交还终端")

    subparsers.add_parser("list", help="列出最近任务")
    return parser


def run_interaction(
    service,
    task: TaskRuntime,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    once: bool = False,
) -> TaskRuntime:
    while True:
        while (
            task.lifecycle is TaskLifecycleStatus.WAITING_APPROVAL
            and task.pending_actions
        ):
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
                    try:
                        choice = input_fn("选择 [a]批准 / [r]拒绝 / [q]暂停: ").strip().lower()
                    except EOFError:
                        output_fn("输入结束；任务已保留，可稍后使用 resume 继续。")
                        return task
                    if choice == "q":
                        return service.pause_task(task.task_id, "用户从终端暂停审批")
                    if choice in {"a", "r"}:
                        decisions.append("approve" if choice == "a" else "reject")
                        break
                    output_fn("无效选择，请输入 a、r 或 q。")
            task = service.decide(task.task_id, decisions)

        if task.lifecycle is TaskLifecycleStatus.WAITING_INPUT:
            output_fn(f"任务 ID: {task.task_id}")
            output_fn(f"等待用户补充（handoff）：{task.pause_reason or '请提供更多信息'}")
            if once:
                return task
            try:
                reply = input_fn("回复（/pause 暂停任务）: ").strip()
            except EOFError:
                output_fn("输入结束；任务已保留，可稍后使用 resume 继续。")
                return task
            if reply.lower() == "/pause":
                return service.pause_task(task.task_id, "用户从终端暂停等待输入")
            if not reply:
                output_fn("未收到补充；任务已保留，可稍后使用 resume 继续。")
                return task
            task = service.continue_task(
                task.task_id,
                reply,
                input_kind="information",
            )
            continue

        output_fn(f"任务 ID: {task.task_id}")
        repositories = getattr(service, "repositories", None)
        if isinstance(repositories, DomainRepositories):
            output_fn(render_report(build_task_report_view(repositories, task.task_id)))
        elif task.lifecycle is TaskLifecycleStatus.PAUSED:
            output_fn(f"任务已暂停: {task.pause_reason or '请提供更多信息'}")
        return task


def print_task_list(
    repository: TaskRepository,
    *,
    output_fn: Callable[[str], None] = print,
) -> None:
    definitions = repository.list_recent_definitions()
    if not definitions:
        output_fn("暂无任务")
        return
    for definition in definitions:
        lifecycle = repository.get_lifecycle(definition.task_id)
        output_fn(
            f"{definition.task_id}\t{lifecycle.status.value}\t"
            f"{definition.source_project_root}\t{definition.original_problem}"
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
    database = SQLiteDatabase(state_database_path())
    repositories = DomainRepositories.create(database)
    repository = repositories.tasks

    if args.command == "list":
        print_task_list(repository, output_fn=write_output)
        return 0

    if args.command == "new":
        config = load_config(
            args.project,
            ApprovalMode(args.mode),
            project_python=args.python,
        )
        stored_task = None
    else:
        try:
            stored_task = repository.get_definition(args.task_id)
        except KeyError:
            write_output(f"任务不存在: {args.task_id}")
            return 2
        config = load_config(
            stored_task.source_project_root,
            ApprovalMode(stored_task.approval_mode),
            project_python=stored_task.project_python,
        )

    research_evidence_store = repositories.evidence
    investigation = InvestigationCoordinator(
        store=repositories.investigation,
        tasks=repository,
        evidence_repository=repositories.evidence,
        evidence_collector=EvidenceCollector(repositories.evidence),
    )
    artifact_backend = build_backend(config)
    result_artifacts = ToolResultArtifactStorage(
        config.artifacts_path / "investigation_receipts"
    )
    workspace_factory = WorkspaceFactory(
        config.workspaces_path or config.database_path.parent / "workspaces"
    )
    _token_budget_store = TokenBudgetStore(database=database)
    with build_research_client() as client:
        extensions = build_cli_research_extensions(
            config,
            research_evidence_store,
            client,
            artifact_backend,
        )
        with database.connection() as connection:
            checkpointer = SqliteSaver(connection)
            agent = build_agent(
                config,
                checkpointer,
                task_repository=repository,
                extensions=extensions,
                investigation=investigation,
                backend=artifact_backend,
                repositories=repositories,
            )
            service = BugfixService(
                agent,
                repository,
                ApprovalPolicy(config.approval_mode),
                config,
                investigation=investigation,
                operation_reconciler=OperationReconciler(
                    repositories.execution,
                    result_artifacts,
                ),
                workspace_factory=workspace_factory,
                execution_backend=artifact_backend,
                repositories=repositories,
            )
            if args.command == "new":
                task = service.start(args.problem)
            else:
                assert stored_task is not None
                runtime = service.get_runtime(stored_task.task_id)
                task = _resume_from_cli(
                    service, runtime, args.message, write_output,
                    input_kind=args.kind, supersedes_input_id=args.supersedes,
                )
                if task is None:
                    return 2
            run_interaction(
                service,
                task,
                input_fn=read_input,
                output_fn=write_output,
                once=args.once,
            )
    return 0


def build_research_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))


def build_cli_research_extensions(
    config,
    store: EvidenceRepository,
    client: httpx.Client,
    artifact_backend,
) -> AgentExtensions:
    url_policy = UrlSafetyPolicy()
    inspector = DependencyInspector(config.project_root, config.project_python)
    pypi = PyPIProvider(client, url_policy)
    github = GitHubProvider(
        client,
        url_policy,
        token=os.environ.get("GITHUB_TOKEN"),
    )
    tavily = TavilyProvider(
        client,
        url_policy,
        api_key=(os.environ.get("TAVILY_API_KEY") if config.search_provider == "tavily" else None),
    )
    provider = CompositeTechnicalSearchProvider((pypi, github, tavily))
    fetcher = SafeEvidenceFetcher(client, url_policy)
    return build_research_extensions(
        inspect_dependency=build_inspect_dependency_tool(inspector),
        search_technical_sources=build_search_technical_sources_tool(
            QuerySanitizer(),
            inspector,
            provider,
            store,
        ),
        fetch_external_evidence=build_fetch_external_evidence_tool(
            fetcher,
            store,
            artifact_backend,
        ),
        link_external_evidence=build_link_external_evidence_tool(store),
    )


def _resume_from_cli(
    service: BugfixService,
    task: TaskRuntime,
    message: str | None,
    output_fn: Callable[[str], None],
    *,
    input_kind: str = "information",
    supersedes_input_id: str | None = None,
) -> TaskRuntime | None:
    if (
        task.lifecycle is TaskLifecycleStatus.WAITING_APPROVAL
        and task.pending_actions
    ):
        return task
    if (
        task.lifecycle is TaskLifecycleStatus.PAUSED
        and task.pending_actions
        and not message
    ):
        return service.continue_task(task.task_id)
    if task.lifecycle in {TaskLifecycleStatus.PAUSED, TaskLifecycleStatus.WAITING_INPUT} and not message:
        return task
    if not message or not message.strip():
        output_fn("恢复该任务需要提供 MESSAGE。")
        return None
    return service.continue_task(
        task.task_id,
        message,
        input_kind=input_kind,
        supersedes_input_id=supersedes_input_id,
    )


def _action_summary(name: str, args: Mapping[str, object]) -> str:
    if name == "execute":
        return str(args.get("command", ""))
    if name in {"write_file", "edit_file", "delete"}:
        return str(args.get("file_path", args.get("path", "")))
    return str(dict(args))


if __name__ == "__main__":
    raise SystemExit(main())
