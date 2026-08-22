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
from deepfix.config import AppConfig
from deepfix.context import build_context_middleware, build_save_progress_tool
from deepfix.extensions import AgentExtensions, merge_extensions
from deepfix.memory import WorkingMemoryStore
from deepfix.models import RepairOutcome
from deepfix.prompts import CORE_REPAIR_PROMPT
from deepfix.research.store import ResearchEvidenceStore


def build_model(config: AppConfig) -> ChatDeepSeek:
    return ChatDeepSeek(model=config.model_name, temperature=0)


def build_agent(
    config: AppConfig,
    checkpointer,
    working_memory_store: WorkingMemoryStore,
    extensions: AgentExtensions | None = None,
    research_evidence_store: ResearchEvidenceStore | None = None,
    *,
    allowed_skill_roots: tuple[str | Path, ...] = (),
):
    register_harness_profile(
        f"deepseek:{config.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    model = build_model(config)
    backend = build_backend(config)
    save_progress = build_save_progress_tool(working_memory_store)
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
        tools=[save_progress, *(item.tool for item in resolved.tools)],
        middleware=[
            *build_context_middleware(
                model,
                backend,
                working_memory_store,
                research_evidence_store,
            ),
            *resolved.middleware,
        ],
        backend=backend,
        subagents=[],
        skills=list(resolved.skill_sources),
        response_format=RepairOutcome,
        interrupt_on=merge_interrupt_on(core_interrupts, resolved.tools),
        checkpointer=checkpointer,
        name="deepfix_repair_agent",
    )
