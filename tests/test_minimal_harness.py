
import pytest
from investigation.test_middleware import tool_request
from langchain_core.messages import ToolMessage

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.domain_repositories import DomainRepositories
from deepfix.investigation.errors import InvestigationStateError
from deepfix.investigation.middleware import InvestigationMiddleware
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.protected_context import ProtectedContextBuilder
from deepfix.reporting import build_task_report_view
from deepfix.task_domain.models import TaskDefinition


@pytest.fixture
def runtime(tmp_path):
    repos = DomainRepositories.create(tmp_path / "state.sqlite3")
    workspace = tmp_path / "project"
    workspace.mkdir()
    repos.tasks.create_definition(TaskDefinition(
        task_id="task-a", original_message_id="user-a", original_problem="repair bug",
        approval_mode="manual", source_project_root=str(workspace),
        workspace_root=str(workspace), workspace_baseline_id="baseline-a",
        project_python="python", confinement_level="workspace", created_at="2026-09-06",
    ))
    middleware = InvestigationMiddleware(
        tasks=repos.tasks, execution=repos.execution,
        artifacts=ToolResultArtifactStorage(tmp_path / "artifacts"),
        evidence_collector=EvidenceCollector(repos.evidence),
    )
    return repos, middleware


@pytest.mark.parametrize("name,args", [
    ("grep", {"pattern": "custom_repr"}),
    ("read_file", {"file_path": "/module.py"}),
    ("execute", {"command": "python -m pytest -q"}),
])
def test_new_calls_execute_and_same_identity_replays(runtime, name, args):
    repos, middleware = runtime
    executions = []

    def handler(request):
        executions.append(request.tool_call["id"])
        return ToolMessage(
            content=f"result {len(executions)}", name=name,
            tool_call_id=request.tool_call["id"],
            artifact={"exit_code": 1} if name == "execute" else None,
        )

    first = middleware.wrap_tool_call(tool_request(name, "a", args), handler)
    second = middleware.wrap_tool_call(tool_request(name, "b", args), handler)
    replay = middleware.wrap_tool_call(tool_request(name, "b", args), handler)
    assert executions == ["a", "b"]
    assert first.content != second.content
    assert replay.content == second.content
    assert repos.investigation.load("task-a") is None
    if name == "execute":
        assert len(repos.evidence.verification_view("task-a").test_evidence) == 2


def test_unknown_side_effect_is_not_replayed(runtime):
    _, middleware = runtime
    request = tool_request("execute", "crash", {"command": "python -m pytest -q"})
    calls = []

    def interrupted(request):
        calls.append(request.tool_call["id"])
        raise RuntimeError("process disappeared after starting")

    with pytest.raises(RuntimeError):
        middleware.wrap_tool_call(request, interrupted)
    with pytest.raises(InvestigationStateError):
        middleware.wrap_tool_call(request, interrupted)
    assert calls == ["crash"]


def test_context_and_report_do_not_read_investigation(runtime):
    repos, _ = runtime

    class Unavailable:
        def __getattr__(self, name):
            raise AssertionError(f"production read investigation.{name}")

    object.__setattr__(repos, "investigation", Unavailable())
    context = ProtectedContextBuilder(repos).build("task-a", [], None)
    report = build_task_report_view(repos, "task-a")
    assert context.task_anchor.task_id == "task-a"
    assert report.definition.task_id == "task-a"
