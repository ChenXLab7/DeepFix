from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage

from deepfix.approval import ApprovalPolicy
from deepfix.backend import build_backend
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode, load_config
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.store import InvestigationStore
from deepfix.models import Evidence, RepairOutcome
from deepfix.persistence import TaskRepository
from deepfix.reporting import build_task_report_view, render_report
from deepfix.research.dependency import DependencyInspector
from deepfix.research.fetcher import SafeEvidenceFetcher
from deepfix.research.providers import (
    CompositeTechnicalSearchProvider,
    GitHubProvider,
    PyPIProvider,
    TavilyProvider,
)
from deepfix.research.sanitizer import QuerySanitizer, UrlSafetyPolicy
from deepfix.research.store import ResearchEvidenceStore
from deepfix.research.tools import (
    build_fetch_external_evidence_tool,
    build_link_external_evidence_tool,
    build_search_technical_sources_tool,
)
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskLifecycleStatus

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC_ADDRESS = "93.184.216.34"


class _FakeAgent:
    def __init__(self, *results: dict[str, object]) -> None:
        self.results = deque(results)

    def invoke(self, value: object, config: dict[str, object]) -> dict[str, object]:
        del value, config
        return self.results.popleft()


def _outcome(status: str) -> dict[str, object]:
    return {
        "structured_response": RepairOutcome(
            status=status,
            question="请补充复现条件" if status == "needs_input" else None,
            diagnosis="外部资料给出了线索",
            summary="等待补充" if status == "needs_input" else "模型声称已经完成",
        ),
        "messages": [],
    }


def _runtime(
    task_id: str,
    *,
    messages: list[AIMessage | ToolMessage] | None = None,
    call_id: str = "research-call",
) -> ToolRuntime:
    return ToolRuntime(
        state={"messages": messages or []},
        context=None,
        config={"configurable": {"thread_id": task_id}},
        stream_writer=lambda _: None,
        tool_call_id=call_id,
        store=None,
    )


def _execute_messages(
    result,
    *,
    call_id: str,
    command: str,
) -> list[AIMessage | ToolMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": command},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=result.output,
            name="execute",
            tool_call_id=call_id,
            artifact={"exit_code": result.exit_code},
        ),
    ]


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_offline_research_evidence_workflow_is_task_local_and_cannot_bypass_tests(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "target-project"
    project.mkdir()
    requirements = project / "requirements.txt"
    smoke_test = project / "test_smoke.py"
    requirements.write_text("pydantic>=2,<3\n", encoding="utf-8")
    smoke_test.write_text("def test_smoke():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    original_project_files = {
        path.name: path.read_text(encoding="utf-8") for path in (requirements, smoke_test)
    }

    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "deepfix-home"))
    config = load_config(
        project,
        ApprovalMode.GUARDED,
        project_python=sys.executable,
    )
    repository = TaskRepository(config.database_path)
    research_store = ResearchEvidenceStore(config.database_path)
    compaction_store = CompactionStore(config.database_path)
    investigation = InvestigationCoordinator(
        store=InvestigationStore(config.database_path),
        tasks=repository,
        compaction_store=compaction_store,
        evidence_collector=EvidenceCollector(compaction_store, research_store),
    )
    backend = build_backend(config)
    service = BugfixService(
        _FakeAgent(_outcome("needs_input"), _outcome("completed")),
        repository,
        ApprovalPolicy(config.approval_mode),
        config,
        research_store,
        compaction_store,
        investigation,
    )
    task = service.start("验证 pydantic 版本相关问题")
    assert task.lifecycle is TaskLifecycleStatus.PAUSED

    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/pypi/pydantic/json":
            return httpx.Response(200, json=_fixture("pypi_project.json"))
        if request.url.path == "/search/issues":
            return httpx.Response(200, json=_fixture("github_search.json"))
        if request.url.path == "/repos/pydantic/pydantic/releases":
            return httpx.Response(200, json=_fixture("github_release.json"))
        if request.url.host in {"pypi.org", "docs.pydantic.dev", "github.com"}:
            marker = request.url.path.strip("/").replace("/", "-") or "home"
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=(
                    f"<main><h1>Evidence {marker}</h1><p>摘要可见 {marker} "
                    f"{'x' * 1_400} FULL-BODY-ONLY-{marker}</p></main>"
                ),
            )
        return httpx.Response(404, json={"message": "not found"})

    url_policy = UrlSafetyPolicy(resolver=lambda host: [PUBLIC_ADDRESS])
    inspector = DependencyInspector(config.project_root, config.project_python)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = CompositeTechnicalSearchProvider(
            (
                PyPIProvider(client, url_policy),
                GitHubProvider(client, url_policy),
                TavilyProvider(client, url_policy, api_key=None),
            )
        )
        search_tool = build_search_technical_sources_tool(
            QuerySanitizer(), inspector, provider, research_store
        )
        fetch_tool = build_fetch_external_evidence_tool(
            SafeEvidenceFetcher(client, url_policy), research_store, backend
        )
        link_tool = build_link_external_evidence_tool(research_store)

        before_sensitive = len(requests)
        rejected = search_tool.func(
            query="api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            package_name="pydantic",
            runtime=_runtime(task.task_id),
        )
        assert rejected.status == "error"
        assert len(requests) == before_sensitive
        assert research_store.query_summary(task.task_id) == (0, [])

        search = search_tool.func(
            query="pydantic model_copy validation error",
            package_name="pydantic",
            runtime=_runtime(task.task_id),
        )
        assert search.status == "success"
        search_payload = json.loads(search.content)
        by_level: dict[str, dict[str, object]] = {}
        for candidate in search_payload["candidates"]:
            by_level.setdefault(candidate["evidence_level"], candidate)
        assert set(by_level) == {"E1", "E2", "E3"}

        request_count = len(requests)
        cross_task = fetch_tool.func(
            candidate_id=by_level["E1"]["candidate_id"],
            runtime=_runtime("another-task"),
        )
        arbitrary_url = fetch_tool.func(
            candidate_id="https://attacker.example/private",
            runtime=_runtime(task.task_id),
        )
        assert cross_task.status == "error"
        assert arbitrary_url.status == "error"
        assert len(requests) == request_count
        assert set(fetch_tool.args) == {"candidate_id"}

        fetched_ids: dict[str, str] = {}
        for level in ("E1", "E2", "E3"):
            fetched = fetch_tool.func(
                candidate_id=by_level[level]["candidate_id"],
                runtime=_runtime(task.task_id, call_id=f"fetch-{level}"),
            )
            assert fetched.status == "success"
            fetched_ids[level] = json.loads(fetched.content)["evidence_id"]

    passing_command = "python -m pytest -q -p no:cacheprovider"
    passing_result = backend.execute(passing_command)
    assert passing_result.exit_code == 0
    passing_messages = _execute_messages(
        passing_result,
        call_id="real-passing-pytest",
        command=passing_command,
    )
    verified = link_tool.func(
        evidence_id=fetched_ids["E1"],
        status="verified",
        test_tool_call_ids=["real-passing-pytest"],
        local_evidence=[Evidence("test_smoke.py:2", "一次性项目的回归测试真实通过")],
        explanation="真实 pytest exit_code 为 0",
        runtime=_runtime(task.task_id, messages=passing_messages),
    )
    assert verified.status == "success"

    failing_command = "python -m pytest -q missing_test.py -p no:cacheprovider"
    failing_result = backend.execute(failing_command)
    assert failing_result.exit_code != 0
    failing_messages = _execute_messages(
        failing_result,
        call_id="real-failing-pytest",
        command=failing_command,
    )
    cannot_verify = link_tool.func(
        evidence_id=fetched_ids["E2"],
        status="verified",
        test_tool_call_ids=["real-failing-pytest"],
        local_evidence=[],
        explanation="失败测试不能证明外部结论",
        runtime=_runtime(task.task_id, messages=failing_messages),
    )
    fabricated = link_tool.func(
        evidence_id=fetched_ids["E3"],
        status="verified",
        test_tool_call_ids=["fabricated-call"],
        local_evidence=[],
        explanation="模型伪造的测试 ID",
        runtime=_runtime(task.task_id, messages=passing_messages),
    )
    assert cannot_verify.status == "error"
    assert fabricated.status == "error"

    contradicted = link_tool.func(
        evidence_id=fetched_ids["E2"],
        status="contradicted",
        test_tool_call_ids=["real-failing-pytest"],
        local_evidence=[],
        explanation="当前项目无法复现外部资料中的结论",
        runtime=_runtime(task.task_id, messages=failing_messages),
    )
    assert contradicted.status == "success"

    evidence = research_store.list_evidence(task.task_id)
    assert {item.evidence_level for item in evidence} == {"E1", "E2", "E3"}
    assert {item.local_verification for item in evidence} == {
        "verified",
        "contradicted",
        "unverified",
    }
    for item in evidence:
        artifact = config.artifacts_path / item.artifact_path.removeprefix("/.deepfix-artifacts/")
        assert artifact.is_file()
        marker = urlsplit(item.url).path.strip("/").replace("/", "-") or "home"
        assert f"FULL-BODY-ONLY-{marker}" in artifact.read_text(encoding="utf-8")
        assert f"FULL-BODY-ONLY-{marker}" not in item.model_dump_json()

    assert original_project_files == {
        path.name: path.read_text(encoding="utf-8") for path in (requirements, smoke_test)
    }
    assert all(
        "FULL-BODY-ONLY" not in path.read_text(encoding="utf-8")
        for path in (requirements, smoke_test)
    )

    continued = service.continue_task(task.task_id, "复现条件已经补充")
    assert continued.lifecycle is TaskLifecycleStatus.PAUSED
    assert continued.pause_reason == "缺少通过的测试证据，不能标记为完成"
    assert {item.evidence_id for item in research_store.list_evidence(task.task_id)} == set(
        fetched_ids.values()
    )
    assert research_store.query_summary(task.task_id) == (
        1,
        ["github_discussions: GITHUB_TOKEN 未配置，已跳过"],
    )

    report = render_report(build_task_report_view(service.repositories, task.task_id))
    assert "已通过真实测试关联" in report
    assert "外部结论已被本地证据推翻" in report
    assert "仅为外部线索" in report
    assert "缺少通过的测试证据，不能标记为完成" in report
