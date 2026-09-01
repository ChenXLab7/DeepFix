from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path

from deepfix.config import AppConfig, ApprovalMode, load_config
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
)
from deepfix.evaluation.traces import summarize_llm_trace
from deepfix.investigation.token_budget import (
    TokenBudgetCallbackHandler,
    TokenBudgetMiddleware,
    TokenBudgetStore,
)
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskLifecycleStatus
from deepfix.task_domain.runtime import TaskRuntime

ConfigFactory = Callable[
    [Path, Path, EvaluationBudget, Path],
    AppConfig,
]
ServiceFactory = Callable[[AppConfig], AbstractContextManager[BugfixService]]
OracleRunner = Callable[[str, Path, int, Path], int]

_CONFIG_ENV_LOCK = threading.Lock()


class LegacyLoopRunner:
    budget_enforcement = "pre_call_reservation"
    model_accounting = "all_model_roles"

    def __init__(
        self,
        *,
        project_python: Path,
        config_factory: ConfigFactory | None = None,
        service_factory: ServiceFactory | None = None,
        oracle_runner: OracleRunner | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.project_python = project_python.expanduser().resolve()
        self.config_factory = config_factory or _build_run_config
        self.service_factory = service_factory
        self.oracle_runner = oracle_runner or _run_oracle
        self.clock = clock

    def run(
        self,
        case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        config = self.config_factory(
            workspace,
            run_dir,
            budget,
            self.project_python,
        )
        started_at = self.clock()
        task: TaskRuntime | None = None
        decision_outcome: str | None = None
        tool_calls = 0
        sanitized_error_code: str | None = None
        middleware: TokenBudgetMiddleware | None = None

        try:
            middleware = build_evaluation_token_budget_middleware(config, budget)
            compaction_callbacks = build_evaluation_compaction_callbacks(
                config,
                budget,
            )
            service_context = (
                self.service_factory(config)
                if self.service_factory is not None
                else _default_service_factory(
                    config,
                    evaluation_middleware=middleware,
                    compaction_callbacks=compaction_callbacks,
                )
            )
            with service_context as service:
                task = service.start(case.problem)
                decision = service.repository.latest_adjudication(task.task_id)
                decision_outcome = decision.outcome if decision is not None else None
                tool_calls = service.repositories.execution.integrity_view(
                    task.task_id
                ).receipt_count
        except Exception:  # noqa: BLE001 - evaluation records only a stable code
            sanitized_error_code = "legacy_runner_failed"

        oracle_exit_code: int | None
        try:
            oracle_exit_code = int(
                self.oracle_runner(
                    case.required_command,
                    workspace,
                    budget.max_wall_seconds,
                    self.project_python,
                )
            )
        except Exception:  # noqa: BLE001 - never persist command exception details
            oracle_exit_code = None
            sanitized_error_code = sanitized_error_code or "oracle_execution_failed"

        elapsed = max(0.0, self.clock() - started_at)
        if task is None:
            conclusion = "failed"
            task_id = f"unavailable-{case.case_id}-{run_dir.name}"
        else:
            conclusion, task_error = _task_outcome(task, decision_outcome)
            sanitized_error_code = sanitized_error_code or task_error
            task_id = task.task_id

        trace_usage = summarize_llm_trace(
            config.artifacts_path / "debug" / "llm_calls.jsonl",
            task_id,
        )
        ledger_usage = None
        if middleware is not None:
            try:
                ledger_usage = middleware.store.usage(task_id)
            except KeyError:
                pass
        usage = trace_usage.model_copy(
            update={
                "input_tokens": (
                    ledger_usage.input_tokens if ledger_usage else trace_usage.input_tokens
                ),
                "output_tokens": (
                    ledger_usage.output_tokens if ledger_usage else trace_usage.output_tokens
                ),
                "model_calls": (
                    ledger_usage.model_calls if ledger_usage else trace_usage.model_calls
                ),
                "tool_calls": tool_calls,
                "wall_seconds": elapsed,
                "usage_estimated": (
                    ledger_usage.estimated if ledger_usage else trace_usage.usage_estimated
                ),
            }
        )
        return EvaluationRun(
            run_id=f"{case.case_id}-{run_dir.name}",
            case_id=case.case_id,
            loop="legacy",
            task_id=task_id,
            conclusion=conclusion,
            oracle_exit_code=oracle_exit_code,
            scope_violations=[],
            usage=usage,
            sanitized_error_code=sanitized_error_code,
        )


def _task_outcome(
    task: TaskRuntime,
    decision_outcome: str | None,
) -> tuple[str, str | None]:
    if task.lifecycle is TaskLifecycleStatus.COMPLETED:
        if decision_outcome in {"fixed", "not_reproduced"}:
            return decision_outcome, None
        return "failed", "completed_without_resolution"
    if task.lifecycle is TaskLifecycleStatus.WAITING_APPROVAL:
        actions = {str(item.get("policy_action", "ask")) for item in task.pending_actions}
        if "ask" in actions:
            return "blocked", "manual_approval_required"
        if "deny" in actions:
            return "blocked", "policy_denied"
        return "blocked", "pending_approval"
    if task.lifecycle is TaskLifecycleStatus.PAUSED:
        return "blocked", task.pause_reason or "agent_paused"
    if task.lifecycle is TaskLifecycleStatus.CANCELLED:
        return "failed", "agent_cancelled"
    if task.lifecycle is TaskLifecycleStatus.FAILED:
        return "failed", task.pause_reason or "agent_failed"
    return "failed", "incomplete_agent_response"


def _build_run_config(
    workspace: Path,
    run_dir: Path,
    budget: EvaluationBudget,
    project_python: Path,
) -> AppConfig:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    with _CONFIG_ENV_LOCK:
        previous = os.environ.get("DEEPFIX_HOME")
        os.environ["DEEPFIX_HOME"] = str(resolved_run_dir)
        try:
            config = load_config(
                workspace,
                ApprovalMode.GUARDED,
                project_python=project_python,
            )
        finally:
            if previous is None:
                os.environ.pop("DEEPFIX_HOME", None)
            else:
                os.environ["DEEPFIX_HOME"] = previous

    return replace(
        config,
        diagnostic_timeout_seconds=min(
            config.diagnostic_timeout_seconds,
            budget.max_wall_seconds,
        ),
        verification_timeout_seconds=min(
            config.verification_timeout_seconds,
            budget.max_wall_seconds,
        ),
        max_shell_calls=min(config.max_shell_calls, budget.max_tool_calls),
        max_changed_files=min(config.max_changed_files, budget.max_side_effects),
        max_agent_invocations=min(
            config.max_agent_invocations,
            budget.max_tool_calls,
        ),
    )


def _run_oracle(
    command: str,
    workspace: Path,
    timeout_seconds: int,
    project_python: Path,
) -> int:
    try:
        bound_command = _bind_python_command(command, project_python)
        completed = subprocess.run(
            bound_command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124
    except (OSError, ValueError):
        return 127
    return completed.returncode


def _bind_python_command(command: str, project_python: Path) -> str:
    stripped = command.lstrip()
    match = re.match(r"(?i)python(?:\.exe)?(?=\s|$)", stripped)
    if match is None:
        raise ValueError("oracle command must start with python")
    executable = subprocess.list2cmdline([str(project_python)])
    return f"{executable}{stripped[match.end() :]}"


@contextmanager
def _default_service_factory(
    config: AppConfig,
    *,
    evaluation_middleware: TokenBudgetMiddleware | None = None,
    compaction_callbacks: tuple[TokenBudgetCallbackHandler, ...] = (),
) -> Iterator[BugfixService]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from deepfix.agent import build_agent
    from deepfix.approval import ApprovalPolicy
    from deepfix.backend import build_backend
    from deepfix.cli import build_cli_research_extensions, build_research_client
    from deepfix.compaction.evidence import EvidenceCollector
    from deepfix.domain_repositories import DomainRepositories
    from deepfix.investigation.coordinator import InvestigationCoordinator

    repositories = DomainRepositories.create(config.database_path)
    repository = repositories.tasks
    research_evidence_store = repositories.evidence
    investigation = InvestigationCoordinator(
        store=repositories.investigation,
        tasks=repository,
        evidence_repository=repositories.evidence,
        evidence_collector=EvidenceCollector(repositories.evidence),
    )
    backend = build_backend(config)
    with build_research_client() as client:
        extensions = build_cli_research_extensions(
            config,
            research_evidence_store,
            client,
            backend,
        )
        if evaluation_middleware is not None:
            extensions = replace(
                extensions,
                middleware=(*extensions.middleware, evaluation_middleware),
            )
        with repository.checkpoint_connection() as connection:
            agent = build_agent(
                config,
                SqliteSaver(connection),
                task_repository=repository,
                extensions=extensions,
                investigation=investigation,
                backend=backend,
                compaction_model_callbacks=compaction_callbacks,
                repositories=repositories,
            )
            yield BugfixService(
                agent,
                repository,
                ApprovalPolicy(config.approval_mode),
                config,
                investigation=investigation,
                repositories=repositories,
            )


def build_evaluation_token_budget_middleware(
    config: AppConfig,
    budget: EvaluationBudget,
) -> TokenBudgetMiddleware:
    return TokenBudgetMiddleware(
        TokenBudgetStore(config.database_path),
        input_cap=budget.max_input_tokens,
        output_cap=budget.max_output_tokens,
        requested_output_tokens=min(4096, budget.max_output_tokens),
    )


def build_evaluation_compaction_callbacks(
    config: AppConfig,
    budget: EvaluationBudget,
) -> tuple[TokenBudgetCallbackHandler, ...]:
    return (
        TokenBudgetCallbackHandler(
            TokenBudgetStore(config.database_path),
            input_cap=budget.max_input_tokens,
            output_cap=budget.max_output_tokens,
            role="compaction",
            requested_output_tokens=min(4096, budget.max_output_tokens),
        ),
    )
