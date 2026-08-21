from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain_deepseek import ChatDeepSeek

from deepfix.backend import build_backend
from deepfix.config import AppConfig
from deepfix.context import build_context_middleware, build_save_progress_tool
from deepfix.memory import WorkingMemoryStore
from deepfix.models import RepairOutcome
from deepfix.prompts import REPAIR_SYSTEM_PROMPT


def build_model(config: AppConfig) -> ChatDeepSeek:
    return ChatDeepSeek(model=config.model_name, temperature=0)


def build_agent(
    config: AppConfig,
    checkpointer,
    working_memory_store: WorkingMemoryStore,
):
    register_harness_profile(
        f"deepseek:{config.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    model = build_model(config)
    backend = build_backend(config)
    return create_deep_agent(
        model=model,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        tools=[build_save_progress_tool(working_memory_store)],
        middleware=build_context_middleware(
            model,
            backend,
            working_memory_store,
        ),
        backend=backend,
        subagents=[],
        response_format=RepairOutcome,
        interrupt_on={
            "write_file": True,
            "edit_file": True,
            "delete": True,
            "execute": True,
        },
        checkpointer=checkpointer,
        name="deepfix_repair_agent",
    )
