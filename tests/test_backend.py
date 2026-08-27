import sys
import time

from deepagents.backends import CompositeBackend, LocalShellBackend

from deepfix.backend import build_backend
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.config import ApprovalMode, load_config
from deepfix.workspace import WorkspaceFactory


def test_backend_is_rooted_at_target_project(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    marker.write_text("project marker", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_config(tmp_path, ApprovalMode.MANUAL)
    backend = build_backend(config)

    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, LocalShellBackend)
    assert backend.default.cwd == tmp_path.resolve()
    assert backend.artifacts_root == "/.deepfix-artifacts"
    assert "project marker" in backend.read("/marker.txt").file_data["content"]


def test_task_workspace_backend_is_confined_and_uses_workspace_root(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(project, ApprovalMode.MANUAL)
    workspace = WorkspaceFactory(tmp_path / "workspaces").create("task-a", project)

    backend = build_backend(config, workspace)
    denied = backend.execute('python -c "print(1)"')

    assert backend.default.cwd == workspace.root
    assert denied.exit_code == 126
    assert denied.output.startswith("Denied:")
    assert "VALUE = 1" in backend.read("/value.py").file_data["content"]


def test_task_workspace_backend_runs_scoped_pytest_with_project_python(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_value.py").write_text(
        "def test_value(): assert True\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(project, ApprovalMode.MANUAL)
    workspace = WorkspaceFactory(tmp_path / "workspaces").create("task-a", project)

    result = build_backend(config, workspace).execute(
        "python -m pytest test_value.py -q"
    )

    assert result.exit_code == 0
    assert "1 passed" in result.output


def test_backend_shell_uses_project_cwd_without_exposing_api_key(tmp_path, monkeypatch):
    secrets = {
        "DEEPSEEK_API_KEY": "common-secret",
        "DEEPFIX_MAIN_API_KEY": "main-secret",
        "DEEPFIX_COMPACTION_API_KEY": "compact-secret",
    }
    for name, secret in secrets.items():
        monkeypatch.setenv(name, secret)
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    result = backend.execute(
        "python -c \"import os; print(os.getcwd()); "
        "print(os.getenv('DEEPSEEK_API_KEY')); "
        "print(os.getenv('DEEPFIX_MAIN_API_KEY')); "
        "print(os.getenv('DEEPFIX_COMPACTION_API_KEY'))\""
    )

    assert result.exit_code == 0
    assert str(tmp_path.resolve()) in result.output
    assert all(secret not in result.output for secret in secrets.values())
    assert result.output.count("None") == 3


def test_backend_hard_caps_diagnostic_command_and_reports_timeout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "1")
    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    started_at = time.monotonic()
    result = backend.execute(
        f'"{sys.executable}" -c "while True: pass"',
        timeout=60,
    )
    elapsed = time.monotonic() - started_at

    assert result.exit_code == 124
    assert elapsed < 5
    assert "timed_out: true" in result.output
    assert "command_kind: diagnostic" in result.output
    assert "timeout_seconds: 1" in result.output
    assert "不要原样重试" in result.output


def test_backend_uses_separate_verification_timeout_for_pytest(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(1.25)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("DEEPFIX_VERIFICATION_TIMEOUT_SECONDS", "3")
    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    result = backend.execute(
        f'"{sys.executable}" -m pytest test_slow.py -q',
        timeout=60,
    )

    assert result.exit_code == 0
    assert "1 passed" in result.output


def test_backend_timeout_terminates_the_child_process_tree(tmp_path, monkeypatch):
    marker = tmp_path / "orphan-process-finished.txt"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "1")
    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    result = backend.execute(
        f'"{sys.executable}" -c "import time, pathlib; time.sleep(2); '
        "pathlib.Path('orphan-process-finished.txt').write_text('alive')\""
    )
    time.sleep(1.5)

    assert result.exit_code == 124
    assert not marker.exists()


def test_backend_routes_context_artifacts_outside_target_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    backend = build_backend(config)

    result = backend.write(
        "/.deepfix-artifacts/conversation_history/probe.md",
        "history",
    )

    assert result.error is None
    assert (
        config.artifacts_path / "conversation_history" / "probe.md"
    ).read_text(encoding="utf-8") == "history"
    assert not (config.project_root / ".deepfix-artifacts").exists()
    assert not (config.project_root / "conversation_history").exists()


def test_history_adapter_uses_internal_artifact_route(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)

    ref = DeepAgentsArtifactAdapter(build_backend(config)).persist_history(
        "task-a",
        "attempt-1",
        [],
        set(),
    )

    assert ref.path == "/.deepfix-artifacts/conversation_history/task-a.md"
    assert (config.artifacts_path / "conversation_history" / "task-a.md").exists()


def test_backend_routes_deep_agents_large_result_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    backend = build_backend(config)

    result = backend.write(
        "/.deepfix-artifacts/large_tool_results/call_1",
        "full diagnostic output",
    )

    assert result.error is None
    assert (
        config.artifacts_path / "large_tool_results" / "call_1"
    ).read_text(encoding="utf-8") == "full diagnostic output"
    assert not (config.project_root / ".deepfix-artifacts").exists()
