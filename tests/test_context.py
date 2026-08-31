from __future__ import annotations

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepfix.context import build_context_middleware
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.research.middleware import ResearchEvidenceMiddleware
from deepfix.research.store import ResearchEvidenceStore


def test_context_stack_uses_native_summary_and_domain_research(tmp_path) -> None:
    model = FakeListChatModel(
        responses=["summary"], profile={"max_input_tokens": 10_000}
    )
    backend = object()
    research = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")

    middleware = build_context_middleware(model, backend, research)

    assert [type(item) for item in middleware] == [
        SummarizationMiddleware,
        SummarizationToolMiddleware,
        PromptPolicyMiddleware,
        ResearchEvidenceMiddleware,
    ]


def test_context_stack_has_no_catch_all_memory_middleware(tmp_path) -> None:
    middleware = build_context_middleware(
        FakeListChatModel(
            responses=["summary"], profile={"max_input_tokens": 10_000}
        ),
        object(),
        ResearchEvidenceStore(tmp_path / "deepfix.sqlite3"),
    )

    assert all("Memory" not in type(item).__name__ for item in middleware)
