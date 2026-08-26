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
from deepfix.models import TaskState, TaskStatus
from deepfix.service import BugfixService

ConfigFactory = Callable[
    [Path, Path, EvaluationBudget, Path],
    AppConfig,
]
ServiceFactory = Callable[[AppConfig], AbstractContextManager[BugfixService]]
OracleRunner = Callable[[str, Path, int, Path], int]

_CONFIG_ENV_LOCK = threading.Lock()


class LegacyLoopRunner:
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
        self.service_factory = service_factory or _default_service_factory
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
        task: TaskState | None = None
        sanitized_error_code: str | None = None

        try:
            with self.service_factory(config) as service:
                task = service.start(case.problem)
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
            task_id = "unavailable"
            tool_calls = 0
        else:
            conclusion, task_error = _task_outcome(task)
            sanitized_error_code = sanitized_error_code or task_error
            task_id = task.task_id
            tool_calls = len(set(task.processed_tool_call_ids))

        trace_usage = summarize_llm_trace(
            config.artifacts_path / "debug" / "llm_calls.jsonl",
            task_id,
        )
        usage = trace_usage.model_copy(
            update={
                "tool_calls": tool_calls,
                "wall_seconds": elapsed,
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
    task: TaskState,
) -> tuple[str, str | None]:
    if task.status is TaskStatus.COMPLETED:
        if task.resolution in {"fixed", "not_reproduced"}:
            return task.resolution, None
        return "failed", "completed_without_resolution"
    if task.status is TaskStatus.WAITING_APPROVAL:
        actions = {
            str(item.get("policy_action", "ask")) for item in task.pending_actions
        }
        if "ask" in actions:
            return "blocked", "manual_approval_required"
        if "deny" in actions:
            return "blocked", "policy_denied"
        return "blocked", "pending_approval"
    if task.status is TaskStatus.CLARIFYING:
        return "blocked", "additional_input_required"
    if task.status is TaskStatus.PAUSED:
        if task.context_recovery is not None:
            return "blocked", task.context_recovery.error_code
        if task.investigation_recovery is not None:
            return "blocked", task.investigation_recovery.error_code
        return "blocked", "agent_paused"
    if task.status is TaskStatus.CANCELLED:
        return "failed", "agent_cancelled"
    if task.status is TaskStatus.FAILED:
        return "failed", "agent_failed"
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
    bound_command = _bind_python_command(command, project_python)
    try:
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
    except OSError:
        return 127
    return completed.returncode


def _bind_python_command(command: str, project_python: Path) -> str:
    stripped = command.lstrip()
    match = re.match(r"(?i)python(?:\.exe)?(?=\s|$)", stripped)
    if match is None:
        return command
    executable = subprocess.list2cmdline([str(project_python)])
    return f"{executable}{stripped[match.end():]}"


@contextmanager
def _default_service_factory(config: AppConfig) -> Iterator[BugfixService]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from deepfix.agent import build_agent
    from deepfix.approval import ApprovalPolicy
    from deepfix.backend import build_backend
    from deepfix.cli import build_cli_research_extensions, build_research_client
    from deepfix.compaction.evidence import EvidenceCollector
    from deepfix.compaction.store import CompactionStore
    from deepfix.investigation.coordinator import InvestigationCoordinator
    from deepfix.investigation.store import InvestigationStore
    from deepfix.memory import WorkingMemoryStore
    from deepfix.persistence import TaskRepository
    from deepfix.research.store import ResearchEvidenceStore

    repository = TaskRepository(config.database_path)
    working_memory_store = WorkingMemoryStore(config.database_path)
    compaction_store = CompactionStore(config.database_path)
    research_evidence_store = ResearchEvidenceStore(config.database_path)
    investigation = InvestigationCoordinator(
        store=InvestigationStore(config.database_path),
        tasks=repository,
        compaction_store=compaction_store,
        evidence_collector=EvidenceCollector(
            compaction_store,
            research_evidence_store,
        ),
    )
    backend = build_backend(config)
    with build_research_client() as client:
        extensions = build_cli_research_extensions(
            config,
            research_evidence_store,
            client,
            backend,
        )
        with repository.checkpoint_connection() as connection:
            agent = build_agent(
                config,
                SqliteSaver(connection),
                working_memory_store,
                task_repository=repository,
                compaction_store=compaction_store,
                extensions=extensions,
                research_evidence_store=research_evidence_store,
                investigation=investigation,
                backend=backend,
            )
            yield BugfixService(
                agent,
                repository,
                ApprovalPolicy(config.approval_mode),
                config,
                working_memory_store,
                research_evidence_store,
                compaction_store,
                investigation,
            )
