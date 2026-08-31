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
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode, load_config, state_database_path
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories import DomainRepositories
from deepfix.extensions import AgentExtensions, build_research_extensions
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.receipts import ToolExecutionReceiptStore
from deepfix.investigation.store import InvestigationStore
from deepfix.investigation.token_budget import TokenBudgetStore
from deepfix.models import TaskState, TaskStatus
from deepfix.operations import OperationJournalStore, OperationReconciler
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
from deepfix.research.store import ResearchEvidenceStore
from deepfix.research.tools import (
    build_fetch_external_evidence_tool,
    build_inspect_dependency_tool,
    build_link_external_evidence_tool,
    build_search_technical_sources_tool,
)
from deepfix.service import BugfixService
from deepfix.verification import VerificationPolicyStore
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

    subparsers.add_parser("list", help="列出最近任务")
    return parser


def run_interaction(
    service,
    task: TaskState,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    research_evidence_store: ResearchEvidenceStore | None = None,
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
        repositories = getattr(service, "repositories", None)
        if isinstance(repositories, DomainRepositories):
            output_fn(render_report(build_task_report_view(repositories, task.task_id)))
        else:
            external_evidence = (
                research_evidence_store.list_evidence(task.task_id)
                if research_evidence_store is not None
                else []
            )
            output_fn(render_report(task, external_evidence))
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
            f"{task.task_id}\t{task.status.value}\t"
            f"{task.source_project_root or task.project_root}\t{task.user_problem}"
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
            stored_task = repository.get(args.task_id)
        except KeyError:
            write_output(f"任务不存在: {args.task_id}")
            return 2
        config = load_config(
            stored_task.source_project_root or stored_task.project_root,
            ApprovalMode(stored_task.approval_mode),
            project_python=stored_task.project_python or None,
        )

    compaction_store = CompactionStore(
        config.database_path,
        repositories=repositories,
    )
    research_evidence_store = ResearchEvidenceStore(
        config.database_path,
        repositories=repositories,
    )
    investigation = InvestigationCoordinator(
        store=InvestigationStore(
            config.database_path,
            repositories=repositories,
        ),
        tasks=repository,
        compaction_store=compaction_store,
        evidence_collector=EvidenceCollector(compaction_store, research_evidence_store),
    )
    artifact_backend = build_backend(config)
    operation_journal = OperationJournalStore(
        config.database_path,
        repositories=repositories,
    )
    receipt_store = ToolExecutionReceiptStore(
        config.artifacts_path / "investigation_receipts",
        repository=repositories.execution,
    )
    workspace_factory = WorkspaceFactory(
        config.workspaces_path or config.database_path.parent / "workspaces"
    )
    verification_policy_store = VerificationPolicyStore(tasks=repository)
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
                compaction_store=compaction_store,
                extensions=extensions,
                research_evidence_store=research_evidence_store,
                investigation=investigation,
                verification_policy_store=verification_policy_store,
                backend=artifact_backend,
                repositories=repositories,
            )
            service = BugfixService(
                agent,
                repository,
                ApprovalPolicy(config.approval_mode),
                config,
                research_evidence_store,
                compaction_store,
                investigation,
                operation_reconciler=OperationReconciler(
                    operation_journal,
                    receipt_store,
                ),
                workspace_factory=workspace_factory,
                verification_policy_store=verification_policy_store,
                execution_backend=artifact_backend,
                repositories=repositories,
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
                research_evidence_store=research_evidence_store,
            )
    return 0


def build_research_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))


def build_cli_research_extensions(
    config,
    store: ResearchEvidenceStore,
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
