from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain_deepseek import ChatDeepSeek

from deepfix.approval import merge_interrupt_on
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
from deepfix.config import AppConfig
from deepfix.context import build_save_progress_tool
from deepfix.extensions import AgentExtensions, merge_extensions
from deepfix.memory import WorkingMemoryStore
from deepfix.models import RepairOutcome
from deepfix.persistence import TaskRepository
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.prompts import CORE_REPAIR_PROMPT
from deepfix.protected_context import (
    ProtectedContextBuilder,
    ProtectedContextMiddleware,
)
from deepfix.research.store import ResearchEvidenceStore


def build_model(config: AppConfig) -> ChatDeepSeek:
    return ChatDeepSeek(model=config.model_name, temperature=0)


def build_agent(
    config: AppConfig,
    checkpointer,
    working_memory_store: WorkingMemoryStore,
    task_repository: TaskRepository | None = None,
    compaction_store: CompactionStore | None = None,
    extensions: AgentExtensions | None = None,
    research_evidence_store: ResearchEvidenceStore | None = None,
    *,
    allowed_skill_roots: tuple[str | Path, ...] = (),
    backend=None,
):
    register_harness_profile(
        f"deepseek:{config.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
        ),
    )
    model = build_model(config)
    resolved_backend = backend or build_backend(config)
    tasks = task_repository or TaskRepository(config.database_path)
    compaction = compaction_store or CompactionStore(config.database_path)
    research = research_evidence_store or ResearchEvidenceStore(
        config.database_path
    )
    evidence_collector = EvidenceCollector(compaction, research)
    protected_builder = ProtectedContextBuilder(
        tasks,
        working_memory_store,
        compaction,
        research,
        evidence_collector,
    )
    budget_monitor = ContextBudgetMonitor()
    coordinator = CompactionCoordinator(
        adapter=DeepAgentsArtifactAdapter(resolved_backend),
        delta_generator=CompactionDeltaGenerator(),
        snapshot_builder=CompactionSnapshotBuilder(),
        snapshot_store=compaction,
        memory_store=working_memory_store,
        budget_monitor=budget_monitor,
        protected_builder=protected_builder,
        model=model,
    )
    save_progress = build_save_progress_tool(working_memory_store)
    compact_conversation = build_compact_conversation_tool(coordinator)
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
    return create_deep_agent(
        model=model,
        system_prompt=CORE_REPAIR_PROMPT,
        tools=[
            save_progress,
            compact_conversation,
            *(item.tool for item in resolved.tools),
        ],
        middleware=[
            MessageIdentityMiddleware(),
            LegacyContextMigrationMiddleware(
                LegacyContextStores(tasks, working_memory_store, compaction),
                DeepAgentsArtifactAdapter(resolved_backend),
            ),
            PromptPolicyMiddleware(working_memory_store),
            ProtectedContextMiddleware(protected_builder),
            DeepFixCompactionMiddleware(
                protected_builder,
                budget_monitor,
                coordinator,
            ),
            *resolved.middleware,
        ],
        backend=resolved_backend,
        subagents=[],
        skills=list(resolved.skill_sources),
        response_format=RepairOutcome,
        interrupt_on=merge_interrupt_on(core_interrupts, resolved.tools),
        checkpointer=checkpointer,
        name="deepfix_repair_agent",
    )
