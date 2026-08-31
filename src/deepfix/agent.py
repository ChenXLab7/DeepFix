from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents.middleware import TodoListMiddleware
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
from deepfix.compaction.migration import (
    LegacyContextMigrationMiddleware,
    LegacyContextStores,
)
from deepfix.compaction.snapshot import (
    CompactionDeltaGenerator,
    CompactionSnapshotBuilder,
)
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.tools import build_compact_conversation_tool
from deepfix.config import AppConfig, ModelRoleConfig
from deepfix.context import build_save_progress_tool
from deepfix.debug import LLMTraceMiddleware
from deepfix.domain_repositories import DomainRepositories
from deepfix.extensions import AgentExtensions, merge_extensions
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.experiments import ExecutorNarrativeResult
from deepfix.investigation.loop import EXPERIMENT_EXECUTOR_SYSTEM_PROMPT
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.migration import (
    InvestigationMigrationMiddleware,
    InvestigationMigrator,
)
from deepfix.investigation.models import InvestigationCapability
from deepfix.investigation.receipts import ToolExecutionReceiptStore
from deepfix.investigation.store import InvestigationStore
from deepfix.investigation.tools import (
    build_record_hypothesis_tool,
)
from deepfix.memory import WorkingMemoryStore
from deepfix.models import RepairOutcome
from deepfix.navigation.feedback import RepositoryNavigationFeedbackSource
from deepfix.navigation.middleware import TodoNavigationMiddleware
from deepfix.navigation.prompts import (
    DEEPFIX_TODO_SYSTEM_PROMPT,
    DEEPFIX_TODO_TOOL_DESCRIPTION,
)
from deepfix.operations import OperationJournalStore
from deepfix.persistence import TaskRepository
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.prompts import CORE_REPAIR_PROMPT
from deepfix.protected_context import (
    ProtectedContextBuilder,
)
from deepfix.research.store import ResearchEvidenceStore
from deepfix.verification import VerificationPolicyStore


def _build_deepseek_model(role: ModelRoleConfig) -> ChatDeepSeek:
    profile = (
        {"max_input_tokens": 1_000_000}
        if role.model_name in {"deepseek-v4-flash", "deepseek-v4-pro"}
        else None
    )
    return ChatDeepSeek(
        model=role.model_name,
        api_key=role.api_key,
        base_url=role.base_url,
        temperature=0,
        profile=profile,
        extra_body={"thinking": {"type": "disabled"}},
    )


def build_main_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.main_model)


def build_compaction_model(config: AppConfig) -> ChatDeepSeek:
    return _build_deepseek_model(config.compaction_model)


def build_agent(
    config: AppConfig,
    checkpointer,
    working_memory_store: WorkingMemoryStore,
    task_repository: TaskRepository | None = None,
    compaction_store: CompactionStore | None = None,
    investigation: InvestigationCoordinator | None = None,
    extensions: AgentExtensions | None = None,
    research_evidence_store: ResearchEvidenceStore | None = None,
    verification_policy_store: VerificationPolicyStore | None = None,
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
    compaction = compaction_store or CompactionStore(
        config.database_path,
        repositories=repositories,
    )
    research = research_evidence_store or ResearchEvidenceStore(
        config.database_path,
        repositories=repositories,
    )
    evidence_collector = EvidenceCollector(compaction, research)
    investigation_store = (
        investigation.store
        if investigation is not None
        else InvestigationStore(
            config.database_path,
            repositories=repositories,
        )
    )
    investigation = investigation or InvestigationCoordinator(
        store=investigation_store,
        tasks=tasks,
        compaction_store=compaction,
        evidence_collector=evidence_collector,
    )
    navigation_feedback = RepositoryNavigationFeedbackSource(repositories)
    protected_builder = ProtectedContextBuilder(repositories)
    budget_monitor = ContextBudgetMonitor()
    coordinator = CompactionCoordinator(
        adapter=DeepAgentsArtifactAdapter(resolved_backend),
        delta_generator=CompactionDeltaGenerator(),
        snapshot_builder=CompactionSnapshotBuilder(),
        snapshot_store=compaction,
        memory_store=working_memory_store,
        history_repository=(repositories.history if repositories is not None else None),
        budget_monitor=budget_monitor,
        protected_builder=protected_builder,
        model=compaction_model,
    )
    save_progress = build_save_progress_tool(
        working_memory_store,
        compaction_store=compaction,
        investigation_store=investigation_store,
    )
    compact_conversation = build_compact_conversation_tool(coordinator)
    record_hypothesis = build_record_hypothesis_tool(investigation)
    artifact_collector = ArtifactReferenceCollector(compaction, resolved_backend)
    artifact_service = DiagnosticArtifactService(
        tasks,
        artifact_collector,
        resolved_backend,
    )
    search_diagnostic_artifacts = build_search_diagnostic_artifacts_tool(
        artifact_service,
        investigation,
    )
    read_diagnostic_artifact = build_read_diagnostic_artifact_tool(
        artifact_service,
        investigation,
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
        "save_progress",
        "compact_conversation",
        "record_hypothesis",
        "search_diagnostic_artifacts",
        "read_diagnostic_artifact",
        "write_todos",
    }
    resolved = merge_extensions(
        extensions or AgentExtensions(),
        existing_tool_names=core_tool_names,
        allowed_skill_roots=tuple(allowed_skill_roots),
    )
    capabilities = {
        "ls": InvestigationCapability.READ,
        "read_file": InvestigationCapability.READ,
        "write_file": InvestigationCapability.MODIFY,
        "edit_file": InvestigationCapability.MODIFY,
        "delete": InvestigationCapability.MODIFY,
        "glob": InvestigationCapability.SEARCH,
        "grep": InvestigationCapability.SEARCH,
        "execute": InvestigationCapability.EXECUTE,
        "save_progress": InvestigationCapability.MEMORY,
        "compact_conversation": InvestigationCapability.COMPACTION,
        "record_hypothesis": InvestigationCapability.META,
        "search_diagnostic_artifacts": InvestigationCapability.READ,
        "read_diagnostic_artifact": InvestigationCapability.READ,
        "write_todos": InvestigationCapability.META,
        **{
            item.tool.name: item.investigation_capability
            for item in resolved.tools
            if item.investigation_capability is not None
        },
    }
    core_interrupts = {
        "write_file": True,
        "edit_file": True,
        "delete": True,
        "execute": True,
    }
    prompt_policy_middleware = [] if _experiment_mode else [PromptPolicyMiddleware()]
    return create_deep_agent(
        model=main_model,
        system_prompt=(
            EXPERIMENT_EXECUTOR_SYSTEM_PROMPT if _experiment_mode else CORE_REPAIR_PROMPT
        ),
        tools=[
            save_progress,
            compact_conversation,
            record_hypothesis,
            search_diagnostic_artifacts,
            read_diagnostic_artifact,
            *(item.tool for item in resolved.tools),
        ],
        middleware=[
            MessageIdentityMiddleware(),
            TodoListMiddleware(
                system_prompt=DEEPFIX_TODO_SYSTEM_PROMPT,
                tool_description=DEEPFIX_TODO_TOOL_DESCRIPTION,
            ),
            TodoNavigationMiddleware(navigation_feedback, reminder_rounds=3),
            LegacyContextMigrationMiddleware(
                LegacyContextStores(
                    tasks,
                    working_memory_store,
                    compaction,
                    repositories.history if repositories is not None else None,
                ),
                DeepAgentsArtifactAdapter(resolved_backend),
            ),
            InvestigationMigrationMiddleware(
                InvestigationMigrator(
                    tasks=tasks,
                    store=investigation_store,
                    compaction_store=compaction,
                    memory=working_memory_store,
                )
            ),
            InvestigationMiddleware(
                investigation,
                ToolExecutionReceiptStore(
                    config.artifacts_path / "investigation_receipts",
                    repository=(repositories.execution if repositories is not None else None),
                ),
                capabilities,
                operation_journal=OperationJournalStore(
                    config.database_path,
                    repositories=repositories,
                ),
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
        response_format=(ExecutorNarrativeResult if _experiment_mode else RepairOutcome),
        interrupt_on=merge_interrupt_on(core_interrupts, resolved.tools),
        checkpointer=checkpointer,
        name=("deepfix_experiment_executor" if _experiment_mode else "deepfix_repair_agent"),
    )


def build_experiment_agent(*args, **kwargs):
    """Build the opt-in local executor without changing the production default."""
    kwargs["_experiment_mode"] = True
    return build_agent(*args, **kwargs)
