from __future__ import annotations

from collections.abc import Callable
from html import escape

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.prompting import model_request_task_id
from deepfix.research.models import ExternalEvidence

_MAX_RECORDS = 5
_MAX_BLOCK_CHARACTERS = 8_000
_TRUNCATION_MARKER = "…[truncated]"


class ResearchEvidenceMiddleware(AgentMiddleware):
    def __init__(self, store: EvidenceRepository) -> None:
        self.store = store

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        task_id = model_request_task_id(request)
        evidence = self.store.list_external_evidence(task_id) if task_id else []
        if not evidence:
            return handler(request)
        block = render_research_evidence(evidence)
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{block}" if original else block
        return handler(
            request.override(system_message=SystemMessage(content=content))
        )


def render_research_evidence(evidence: list[ExternalEvidence]) -> str:
    ordered = sorted(evidence, key=_evidence_priority)
    opening = "<deepfix_external_evidence>"
    closing = "</deepfix_external_evidence>"
    entries: list[str] = []
    current_length = len(opening) + len(closing) + 2
    for item in ordered[:_MAX_RECORDS]:
        rendered = _render_evidence(item)
        addition = len(rendered) + 1
        if current_length + addition > _MAX_BLOCK_CHARACTERS:
            continue
        entries.append(rendered)
        current_length += addition
    return "\n".join((opening, *entries, closing))


def _evidence_priority(item: ExternalEvidence) -> int:
    if item.local_verification == "verified":
        return 0
    if item.local_verification == "contradicted":
        return 1
    if item.evidence_level in {"E1", "E2"}:
        return 2
    return 3


def _render_evidence(item: ExternalEvidence) -> str:
    warnings: list[str] = []
    if (
        item.documented_version
        and item.project_version
        and item.documented_version != item.project_version
    ):
        warnings.append("资料版本与项目版本不一致")
    if item.evidence_level == "E3":
        warnings.append("E3 仅为未经确认的外部线索")
    if item.local_verification == "unverified":
        warnings.append("尚未通过本地验证")
    elif item.local_verification == "contradicted":
        warnings.append("外部结论已被本地证据推翻")

    lines = [
        f'<evidence id="{_xml(item.evidence_id, 80)}">',
        f"<title>{_xml(item.title, 240)}</title>",
        f"<source_type>{_xml(item.source_type, 80)}</source_type>",
        f"<evidence_level>{item.evidence_level}</evidence_level>",
        f"<verification>{item.local_verification}</verification>",
        (
            "<documented_version>"
            f"{_xml(item.documented_version or 'unknown', 100)}"
            "</documented_version>"
        ),
        (
            "<project_version>"
            f"{_xml(item.project_version or 'unknown', 100)}"
            "</project_version>"
        ),
        f"<url>{_xml(item.url, 600)}</url>",
        f"<excerpt>{_xml(item.relevant_excerpt, 800)}</excerpt>",
        "<warnings>",
        *(f"<warning>{_xml(warning, 160)}</warning>" for warning in warnings),
        "</warnings>",
        "<local_evidence>",
    ]
    lines.extend(
        (
            "<item>"
            f"<source>{_xml(local.source, 120)}</source>"
            f"<observation>{_xml(local.observation, 240)}</observation>"
            "</item>"
        )
        for local in item.local_evidence[:3]
    )
    lines.extend(
        [
            "</local_evidence>",
            "<linked_test_tool_call_ids>",
            *(
                f"<tool_call_id>{_xml(call_id, 120)}</tool_call_id>"
                for call_id in item.linked_test_tool_call_ids[:5]
            ),
            "</linked_test_tool_call_ids>",
            (
                "<verification_explanation>"
                f"{_xml(item.verification_explanation or '', 400)}"
                "</verification_explanation>"
            ),
            f"<artifact_path>{_xml(item.artifact_path, 500)}</artifact_path>",
            "</evidence>",
        ]
    )
    return "\n".join(lines)


def _xml(value: str, limit: int) -> str:
    escaped = escape(value)
    if len(escaped) <= limit:
        return escaped
    marker = escape(_TRUNCATION_MARKER)
    available = limit - len(marker)
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(escape(value[:middle])) <= available:
            low = middle
        else:
            high = middle - 1
    return escape(value[:low]) + marker
