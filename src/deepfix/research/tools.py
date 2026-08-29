from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from deepagents.backends.protocol import BackendProtocol
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepfix.compaction.models import ArtifactReference
from deepfix.models import Evidence
from deepfix.research.dependency import DependencyInspector
from deepfix.research.fetcher import EvidenceFetchError, SafeEvidenceFetcher
from deepfix.research.models import DependencyContext, ExternalEvidence, SearchCandidate
from deepfix.research.providers import (
    CompositeTechnicalSearchProvider,
    ProviderFailure,
)
from deepfix.research.sanitizer import QueryRejected, QuerySanitizer
from deepfix.research.store import ResearchEvidenceStore

_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_RETURNED_CANDIDATES = 10


def build_inspect_dependency_tool(inspector: DependencyInspector) -> BaseTool:
    def inspect_dependency(
        package_name: str,
        runtime: ToolRuntime,
    ) -> ToolMessage:
        task_id = _runtime_task_id(runtime)
        if task_id is None:
            return _error("inspect_dependency", runtime, "运行配置缺少有效的 thread_id")
        try:
            finding = inspector.inspect(package_name)
        except ValueError as exc:
            return _error("inspect_dependency", runtime, str(exc))
        except Exception as exc:  # noqa: BLE001 - Tool boundary must not crash the agent
            return _error(
                "inspect_dependency",
                runtime,
                f"依赖检查失败: {type(exc).__name__}",
            )
        return _success(
            "inspect_dependency",
            runtime,
            finding.model_dump(mode="json"),
        )

    return StructuredTool.from_function(
        func=inspect_dependency,
        name="inspect_dependency",
        description=(
            "读取当前 Python 项目的依赖声明，并用项目 Python 查询已安装版本。"
            "任务 ID 和 Python 环境由系统提供。"
        ),
    )


def build_search_technical_sources_tool(
    sanitizer: QuerySanitizer,
    inspector: DependencyInspector,
    provider: CompositeTechnicalSearchProvider,
    store: ResearchEvidenceStore,
) -> BaseTool:
    def search_technical_sources(
        query: str,
        package_name: str | None = None,
        runtime: ToolRuntime = None,
    ) -> ToolMessage:
        task_id = _runtime_task_id(runtime)
        if task_id is None:
            return _error(
                "search_technical_sources",
                runtime,
                "运行配置缺少有效的 thread_id",
            )
        try:
            sanitized = sanitizer.sanitize(query)
        except QueryRejected as exc:
            return _error(
                "search_technical_sources",
                runtime,
                f"技术查询被拒绝: {exc.rule}",
            )
        if package_name is None or not package_name.strip():
            return _error(
                "search_technical_sources",
                runtime,
                "Python 技术资料检索需要 package_name",
            )
        try:
            finding = inspector.inspect(package_name)
            result = provider.search(
                sanitized,
                DependencyContext(package=finding),
            )
        except (ValueError, ProviderFailure) as exc:
            store.save_query(task_id, sanitized.value, [], [str(exc)])
            return _error("search_technical_sources", runtime, str(exc))
        except Exception as exc:  # noqa: BLE001 - provider boundary is extensible
            diagnostic = f"技术资料检索失败: {type(exc).__name__}"
            store.save_query(task_id, sanitized.value, [], [diagnostic])
            return _error("search_technical_sources", runtime, diagnostic)

        store.save_query(
            task_id,
            sanitized.value,
            result.providers,
            result.errors,
        )
        drafts = [
            SearchCandidate(
                candidate_id="draft",
                task_id="draft",
                source_type=candidate.source_type,
                evidence_level=candidate.evidence_level,
                title=candidate.title,
                url=candidate.url,
                query="draft",
                repository=candidate.repository,
                created_at="draft",
            )
            for candidate in result.candidates
        ]
        saved = store.save_candidates(task_id, sanitized.value, drafts)
        return _success(
            "search_technical_sources",
            runtime,
            {
                "query": sanitized.value,
                "providers": result.providers,
                "provider_errors": result.errors,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "source_type": candidate.source_type,
                        "evidence_level": candidate.evidence_level,
                        "title": _bounded(candidate.title, 240),
                        "url": _bounded(candidate.url, 600),
                        "repository": candidate.repository,
                    }
                    for candidate in saved[:_MAX_RETURNED_CANDIDATES]
                ],
                "candidate_count": len(saved),
            },
        )

    return StructuredTool.from_function(
        func=search_technical_sources,
        name="search_technical_sources",
        description=(
            "在 PyPI、官方仓库和官方文档中检索 Python 依赖相关技术证据。"
            "先检查项目版本，再保存审计记录和仅属于当前任务的候选项。"
        ),
    )


def build_fetch_external_evidence_tool(
    fetcher: SafeEvidenceFetcher,
    store: ResearchEvidenceStore,
    artifact_backend: BackendProtocol,
) -> BaseTool:
    def fetch_external_evidence(
        candidate_id: str,
        runtime: ToolRuntime,
    ) -> ToolMessage:
        task_id = _runtime_task_id(runtime)
        if task_id is None:
            return _error(
                "fetch_external_evidence",
                runtime,
                "运行配置缺少有效的 thread_id",
            )
        try:
            candidate = store.get_candidate(task_id, candidate_id)
        except KeyError:
            return _error(
                "fetch_external_evidence",
                runtime,
                "当前任务中不存在该候选证据",
            )

        host = (urlsplit(candidate.url).hostname or "").lower()
        if not host:
            return _error("fetch_external_evidence", runtime, "候选证据 URL 无效")
        try:
            fetched = fetcher.fetch(candidate.url, allowed_domains=[host])
        except EvidenceFetchError as exc:
            return _error(
                "fetch_external_evidence",
                runtime,
                f"外部证据抓取失败: {exc.rule}",
            )
        except Exception as exc:  # noqa: BLE001 - fetcher boundary is extensible
            return _error(
                "fetch_external_evidence",
                runtime,
                f"外部证据抓取失败: {type(exc).__name__}",
            )

        evidence_id = uuid4().hex
        retrieved_at = datetime.now(UTC).isoformat(timespec="microseconds")
        artifact_path = (
            f"/.deepfix-artifacts/research/{task_id}/{evidence_id}.md"
        )
        artifact = _render_evidence_artifact(
            evidence_id=evidence_id,
            candidate=candidate,
            final_url=fetched.final_url,
            retrieved_at=retrieved_at,
            content=fetched.cleaned_text,
        )
        try:
            write_result = artifact_backend.write(artifact_path, artifact)
        except Exception as exc:  # noqa: BLE001 - backend implementations vary
            return _error(
                "fetch_external_evidence",
                runtime,
                f"证据 artifact 写入失败: {type(exc).__name__}",
            )
        if write_result.error is not None:
            return _error(
                "fetch_external_evidence",
                runtime,
                f"证据 artifact 写入失败: {write_result.error}",
            )
        try:
            read_result = artifact_backend.read(
                artifact_path,
                offset=0,
                limit=max(2000, artifact.count("\n") + 2),
            )
        except Exception as exc:  # noqa: BLE001 - backend implementations vary
            return _error(
                "fetch_external_evidence",
                runtime,
                f"证据 artifact 校验失败: {type(exc).__name__}",
            )
        file_data = read_result.file_data
        restored = (
            file_data.get("content")
            if read_result.error is None
            and file_data is not None
            and file_data.get("encoding") == "utf-8"
            else None
        )
        if restored != artifact:
            return _error(
                "fetch_external_evidence",
                runtime,
                "证据 artifact 校验失败: 回读内容不一致",
            )
        artifact_reference = ArtifactReference(
            path=artifact_path,
            kind="research",
            content_hash=hashlib.sha256(restored.encode("utf-8")).hexdigest(),
            work_unit_ids=[],
        )

        evidence = ExternalEvidence(
            evidence_id=evidence_id,
            task_id=task_id,
            candidate_id=candidate.candidate_id,
            source_type=candidate.source_type,
            evidence_level=candidate.evidence_level,
            title=candidate.title,
            url=fetched.final_url,
            query=candidate.query,
            relevant_excerpt=fetched.excerpt,
            retrieved_at=retrieved_at,
            dependency_name=None,
            documented_version=None,
            project_version=None,
            local_verification="unverified",
            local_evidence=[],
            linked_test_tool_call_ids=[],
            verification_explanation=None,
            artifact_path=artifact_path,
        )
        try:
            store.save_evidence(
                evidence,
                artifact_reference=artifact_reference,
                artifact_content=restored,
            )
        except Exception as exc:  # noqa: BLE001 - persistence becomes a Tool error
            return _error(
                "fetch_external_evidence",
                runtime,
                f"证据入库失败: {type(exc).__name__}",
            )
        return _success(
            "fetch_external_evidence",
            runtime,
            {
                "evidence_id": evidence.evidence_id,
                "candidate_id": evidence.candidate_id,
                "evidence_level": evidence.evidence_level,
                "relevant_excerpt": evidence.relevant_excerpt,
                "artifact_path": evidence.artifact_path,
            },
        )

    return StructuredTool.from_function(
        func=fetch_external_evidence,
        name="fetch_external_evidence",
        description=(
            "抓取当前任务已经检索并保存的候选证据。只接受 candidate_id；"
            "真实 URL 与 artifact 路径由系统查询和生成。"
        ),
    )


def build_link_external_evidence_tool(store: ResearchEvidenceStore) -> BaseTool:
    def link_external_evidence(
        evidence_id: str,
        status: Literal["verified", "contradicted"],
        test_tool_call_ids: list[str],
        local_evidence: list[Evidence],
        explanation: str,
        runtime: ToolRuntime,
    ) -> ToolMessage:
        task_id = _runtime_task_id(runtime)
        if task_id is None:
            return _error(
                "link_external_evidence",
                runtime,
                "运行配置缺少有效的 thread_id",
            )
        try:
            store.get_evidence(task_id, evidence_id)
        except KeyError:
            return _error(
                "link_external_evidence",
                runtime,
                "当前任务中不存在该外部证据",
            )

        explanation = explanation.strip()
        if not explanation:
            return _error(
                "link_external_evidence",
                runtime,
                "证据关联需要非空 explanation",
            )
        if any(
            not item.source.strip() or not item.observation.strip()
            for item in local_evidence
        ):
            return _error(
                "link_external_evidence",
                runtime,
                "本地源码证据的 source 和 observation 不能为空",
            )

        calls, results = _executed_tool_calls(runtime)
        linked_ids = list(dict.fromkeys(test_tool_call_ids))
        exit_codes: list[int] = []
        for call_id in linked_ids:
            call = calls.get(call_id)
            result = results.get(call_id)
            if call is None or result is None:
                return _error(
                    "link_external_evidence",
                    runtime,
                    f"找不到真实且配对的 Tool Call: {call_id}",
                )
            name, args = call
            command = str(args.get("command", ""))
            if (
                name != "execute"
                or result.name != "execute"
                or not _is_pytest_command(command)
                or result.status != "success"
            ):
                return _error(
                    "link_external_evidence",
                    runtime,
                    f"Tool Call 不是成功执行的 pytest: {call_id}",
                )
            artifact = result.artifact
            exit_code = (
                artifact.get("exit_code")
                if isinstance(artifact, Mapping)
                else None
            )
            if type(exit_code) is not int:
                return _error(
                    "link_external_evidence",
                    runtime,
                    f"pytest ToolMessage 缺少整数 exit_code: {call_id}",
                )
            exit_codes.append(exit_code)

        if status == "verified" and not any(code == 0 for code in exit_codes):
            return _error(
                "link_external_evidence",
                runtime,
                "标记 verified 至少需要一次真实通过的 pytest",
            )
        if (
            status == "contradicted"
            and not any(code != 0 for code in exit_codes)
            and not local_evidence
        ):
            return _error(
                "link_external_evidence",
                runtime,
                "标记 contradicted 需要失败的 pytest 或明确源码证据",
            )

        try:
            updated = store.update_verification(
                task_id,
                evidence_id,
                local_verification=status,
                local_evidence=local_evidence,
                linked_test_tool_call_ids=linked_ids,
                verification_explanation=explanation,
            )
        except Exception as exc:  # noqa: BLE001 - persistence becomes a Tool error
            return _error(
                "link_external_evidence",
                runtime,
                f"证据关联入库失败: {type(exc).__name__}",
            )
        payload = {
            "evidence_id": updated.evidence_id,
            "local_verification": updated.local_verification,
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            name="link_external_evidence",
            tool_call_id=runtime.tool_call_id or "",
            status="success",
            artifact=payload,
        )

    return StructuredTool.from_function(
        func=link_external_evidence,
        name="link_external_evidence",
        description=(
            "把当前任务的外部资料与真实本地 pytest 结果或明确源码证据关联。"
            "通过状态来自配对 ToolMessage 的整数 exit_code，不能由模型自行声明。"
        ),
    )


def _runtime_task_id(runtime: ToolRuntime | None) -> str | None:
    if runtime is None:
        return None
    task_id = str(runtime.config.get("configurable", {}).get("thread_id", "")).strip()
    return task_id if _TASK_ID.fullmatch(task_id) else None


def _executed_tool_calls(
    runtime: ToolRuntime,
) -> tuple[
    dict[str, tuple[str, Mapping[str, object]]],
    dict[str, ToolMessage],
]:
    calls: dict[str, tuple[str, Mapping[str, object]]] = {}
    results: dict[str, ToolMessage] = {}
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return calls, results
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                args = call.get("args")
                calls[str(call.get("id", ""))] = (
                    str(call.get("name", "")),
                    args if isinstance(args, Mapping) else {},
                )
        elif isinstance(message, ToolMessage):
            results[str(message.tool_call_id)] = message
    return calls, results


def _is_pytest_command(command: str) -> bool:
    normalized = command.strip().lower()
    return normalized == "pytest" or normalized.startswith(
        ("pytest ", "python -m pytest")
    )


def _error(
    name: str,
    runtime: ToolRuntime | None,
    content: str,
) -> ToolMessage:
    return ToolMessage(
        content=content,
        name=name,
        tool_call_id=(runtime.tool_call_id if runtime is not None else None) or "",
        status="error",
    )


def _success(
    name: str,
    runtime: ToolRuntime,
    payload: dict[str, object],
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=name,
        tool_call_id=runtime.tool_call_id or "",
        status="success",
    )


def _render_evidence_artifact(
    *,
    evidence_id: str,
    candidate: SearchCandidate,
    final_url: str,
    retrieved_at: str,
    content: str,
) -> str:
    return "\n".join(
        [
            "# DeepFix external evidence",
            "",
            f"- evidence_id: {evidence_id}",
            f"- candidate_id: {candidate.candidate_id}",
            f"- source_type: {candidate.source_type}",
            f"- evidence_level: {candidate.evidence_level}",
            f"- title: {candidate.title}",
            f"- url: {final_url}",
            f"- query: {candidate.query}",
            f"- retrieved_at: {retrieved_at}",
            "",
            content,
            "",
        ]
    )


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
