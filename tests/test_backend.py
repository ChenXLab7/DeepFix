from deepagents.backends import CompositeBackend, LocalShellBackend

from deepfix.backend import build_backend
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.config import ApprovalMode, load_config


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


def test_backend_shell_uses_project_cwd_without_exposing_api_key(tmp_path, monkeypatch):
    secret = "super-secret-for-test"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    result = backend.execute(
        "python -c \"import os; print(os.getcwd()); "
        "print(os.getenv('DEEPSEEK_API_KEY'))\""
    )

    assert result.exit_code == 0
    assert str(tmp_path.resolve()) in result.output
    assert secret not in result.output
    assert "None" in result.output


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
