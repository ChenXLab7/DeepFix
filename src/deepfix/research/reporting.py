from __future__ import annotations

from collections.abc import Sequence

from deepfix.research.models import ExternalEvidence


def render_external_evidence(evidence: Sequence[ExternalEvidence]) -> list[str]:
    if not evidence:
        return ["无"]
    lines: list[str] = []
    for item in evidence:
        lines.extend(_render_item(item))
    return lines


def _render_item(item: ExternalEvidence) -> list[str]:
    has_test_link = bool(item.linked_test_tool_call_ids)
    is_verified = item.local_verification == "verified" and has_test_link
    is_contradicted = item.local_verification == "contradicted"
    lines = [
        f"- [{item.evidence_level}/{item.local_verification}] {item.title}",
        f"  - 来源：{item.source_type}",
        f"  - 资料版本：{item.documented_version or '未知'}",
        f"  - 项目版本：{item.project_version or '未知'}",
        f"  - 外部结论：{item.relevant_excerpt}",
    ]
    if is_verified:
        lines.append("  - 本地验证：已通过真实测试关联")
    elif is_contradicted:
        lines.append("  - 本地验证：外部结论已被本地证据推翻")
    else:
        lines.append("  - 本地验证：仅为外部线索")

    if item.verification_explanation:
        lines.append(f"  - 验证说明：{item.verification_explanation}")
    lines.append("  - 本地证据：")
    if item.local_evidence:
        lines.extend(
            f"    - {local.source}：{local.observation}"
            for local in item.local_evidence
        )
    else:
        lines.append("    - 无")
    lines.append(
        "  - 测试 Tool Call："
        + (", ".join(item.linked_test_tool_call_ids) or "无")
    )
    if (
        item.documented_version
        and item.project_version
        and item.documented_version != item.project_version
    ):
        lines.append("  - 警告：资料版本与项目版本不一致")
    if item.evidence_level == "E3" and not is_verified:
        lines.append("  - 警告：E3 仅为未经确认的外部线索")
    lines.extend(
        [
            f"  - URL：{item.url}",
            f"  - Artifact：{item.artifact_path}",
        ]
    )
    return lines
