from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from langchain_deepseek import ChatDeepSeek

from deepfix.approval import merge_interrupt_on
from deepfix.artifact_retrieval.collector import ArtifactReferenceCollector
from deepfix.artifact_retrieval.service import DiagnosticArtifactService
from deepfix.artifact_retrieval.tools import (
    build_read_diagnostic_artifact_tool,
    build_search_diagnostic_artifacts_tool,
)
from deepfix.backend import build_backend
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.budget import ContextBudgetMonitor
from deepfix.compaction.coordinator import CompactionCoordinator
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.middleware import (
    DeepFixCompactionMiddleware,
    MessageIdentityMiddleware,
)
from deepfix.compaction.snapshot import (
    CompactionDeltaGenerator,
    CompactionSnapshotBuilder,
)
from deepfix.compaction.tools import build_compact_conversation_tool
from deepfix.config import AppConfig, ModelRoleConfig
from deepfix.debug import LLMTraceMiddleware
from deepfix.domain_repositories import DomainRepositories
from deepfix.extensions import AgentExtensions, merge_extensions
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.experiments import ExecutorNarrativeResult
from deepfix.investigation.loop import EXPERIMENT_EXECUTOR_SYSTEM_PROMPT
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.persistence import TaskRepository
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.prompts import CORE_REPAIR_PROMPT
from deepfix.protected_context import (
    ProtectedContextBuilder,
)
from deepfix.task_domain.outcome import RepairOutcomeCandidate


class _ThinkingDeepSeek(ChatDeepSeek):
    """Preserve DeepSeek reasoning across tool turns with the installed SDK."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for original, serialized in zip(messages, payload["messages"], strict=True):
            if serialized["role"] == "assistant":
                serialized["reasoning_content"] = original.additional_kwargs.get(
                    "reasoning_content", ""
                )
        # LangChain's structured-output tools force a choice, which thinking
        # endpoints reject. Keep the schema tools available for auto selection.
        choice = payload.get("tool_choice")
        if isinstance(choice, dict) or choice in {"any", "required"}:
            payload["tool_choice"] = "auto"
        return payload


def _build_deepseek_model(role: ModelRoleConfig, *, thinking: bool = False) -> ChatDeepSeek:
    profile = (
        {"max_input_tokens": 1_000_000}
        if role.model_name in {"deepseek-v4-flash", "deepseek-v4-pro"}
        else None
    )
    model_class = _ThinkingDeepSeek if thinking else ChatDeepSeek
    return model_class(
        model=role.model_name,
        api_key=role.api_key,
        base_url=role.base_url,
        temperature=0,
        timeout=role.request_timeout_seconds,
        max_retries=0,
        profile=profile,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


def build_main_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.main_model, thinking=True)


def build_compaction_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.compaction_model)


def build_agent(
    config: AppConfig,
    checkpointer,
    task_repository: TaskRepository | None = None,
    investigation: InvestigationCoordinator | None = None,
    extensions: AgentExtensions | None = None,
    repositories: DomainRepositories | None = None,
    *,
    allowed_skill_roots: tuple[str | Path, ...] = (),
    backend=None,
    compaction_model_callbacks: tuple[object, ...] = (),
    _experiment_mode: bool = False,
):
    register_harness_profile(
        f"deepseek:{config.main_model.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        ),
    )
    main_model = build_main_model(config)
    compaction_model = build_compaction_model(config)
    if compaction_model_callbacks:
        compaction_model = compaction_model.with_config(callbacks=list(compaction_model_callbacks))
    repositories = repositories or DomainRepositories.create(config.database_path)
    resolved_backend = backend or build_backend(config)
    tasks = task_repository or (
        repositories.tasks if repositories is not None else TaskRepository(config.database_path)
    )
    evidence_collector = EvidenceCollector(repositories.evidence)
    protected_builder = ProtectedContextBuilder(repositories)
    budget_monitor = ContextBudgetMonitor()
    coordinator = CompactionCoordinator(
        adapter=DeepAgentsArtifactAdapter(resolved_backend),
        delta_generator=CompactionDeltaGenerator(),
        snapshot_builder=CompactionSnapshotBuilder(),
        history_repository=repositories.history,
        evidence_repository=repositories.evidence,
        budget_monitor=budget_monitor,
        protected_builder=protected_builder,
        model=compaction_model,
    )
    compact_conversation = build_compact_conversation_tool(coordinator)
    artifact_collector = ArtifactReferenceCollector(repositories.history, resolved_backend)
    artifact_service = DiagnosticArtifactService(
        tasks,
        artifact_collector,
        resolved_backend,
    )
    search_diagnostic_artifacts = build_search_diagnostic_artifacts_tool(
        artifact_service,
    )
    read_diagnostic_artifact = build_read_diagnostic_artifact_tool(
        artifact_service,
    )
    core_tool_names = {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "compact_conversation",
        "search_diagnostic_artifacts",
        "read_diagnostic_artifact",
        "write_todos",
    }
    resolved = merge_extensions(
        extensions or AgentExtensions(),
        existing_tool_names=core_tool_names,
        allowed_skill_roots=tuple(allowed_skill_roots),
    )
    core_interrupts = {
        "write_file": True,
        "edit_file": True,
        "delete": True,
        "execute": True,
    }
    approval_interrupts = merge_interrupt_on(core_interrupts, resolved.tools)
    prompt_policy_middleware = [] if _experiment_mode else [PromptPolicyMiddleware()]
    return create_deep_agent(
        model=main_model,
        system_prompt=(
            EXPERIMENT_EXECUTOR_SYSTEM_PROMPT if _experiment_mode else CORE_REPAIR_PROMPT
        ),
        tools=[
            compact_conversation,
            search_diagnostic_artifacts,
            read_diagnostic_artifact,
            *(item.tool for item in resolved.tools),
        ],
        middleware=[
            MessageIdentityMiddleware(),
            TodoListMiddleware(),
            HumanInTheLoopMiddleware(interrupt_on=approval_interrupts),
            InvestigationMiddleware(
                tasks,
                repositories.execution,
                ToolResultArtifactStorage(config.artifacts_path / "investigation_receipts"),
                evidence_collector,
            ),
            *prompt_policy_middleware,
            DeepFixCompactionMiddleware(
                protected_builder,
                budget_monitor,
                coordinator,
            ),
            *resolved.middleware,
            LLMTraceMiddleware(
                log_path=config.artifacts_path / "debug" / "llm_calls.jsonl",
                role="main",
            ),
        ],
        backend=resolved_backend,
        subagents=[],
        skills=list(resolved.skill_sources),
        response_format=(ExecutorNarrativeResult if _experiment_mode else RepairOutcomeCandidate),
        interrupt_on=None,
        checkpointer=checkpointer,
        name=("deepfix_experiment_executor" if _experiment_mode else "deepfix_repair_agent"),
    )


def build_experiment_agent(*args, **kwargs):
    """Build the opt-in local executor without changing the production default."""
    kwargs["_experiment_mode"] = True
    return build_agent(*args, **kwargs)
