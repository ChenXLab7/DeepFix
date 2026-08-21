from deepfix.backend import build_backend
from deepfix.config import ApprovalMode, load_config


def test_backend_is_rooted_at_target_project(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    marker.write_text("project marker", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    backend = build_backend(load_config(tmp_path, ApprovalMode.MANUAL))

    assert backend.cwd == tmp_path.resolve()
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
