from __future__ import annotations

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)

from deepfix.prompting import PromptPolicyMiddleware
from deepfix.research.middleware import ResearchEvidenceMiddleware
from deepfix.research.store import ResearchEvidenceStore


def build_context_middleware(
    model,
    backend,
    research_store: ResearchEvidenceStore,
):
    """Build the legacy summarization stack without a second memory authority."""
    summarization = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("fraction", 0.70),
        keep=("fraction", 0.15),
        truncate_args_settings={
            "trigger": ("fraction", 0.70),
            "keep": ("fraction", 0.15),
            "max_length": 2000,
            "truncation_text": "...(argument truncated)",
        },
    )
    return [
        summarization,
        SummarizationToolMiddleware(
            summarization,
            system_prompt=(
                "使用 write_todos 维护当前导航；当上下文足够长时可调用 compact_conversation。"
            ),
        ),
        PromptPolicyMiddleware(),
        ResearchEvidenceMiddleware(research_store),
    ]
