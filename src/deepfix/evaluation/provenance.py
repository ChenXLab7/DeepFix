from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from deepfix.evaluation.models import EvaluationProvenanceBatch

_IGNORED_TREE_DIRECTORIES = frozenset(
    {".deepfix", ".deepfix-artifacts", ".git", ".pytest_cache", "__pycache__"}
)
_IGNORED_TREE_SUFFIXES = frozenset({".pyc", ".pyo"})


def build_provenance_batch(
    project: Path,
    project_python: Path,
    run_ids: list[str],
) -> EvaluationProvenanceBatch:
    source_root = project.expanduser().resolve()
    python_executable = project_python.expanduser().resolve()
    runner_root = _git_root(Path(__file__).resolve().parent)
    source_git_root = _git_root(source_root)
    source_revision_value = _git_value(source_root, "rev-parse", "HEAD")
    source_revision = source_revision_value or "unversioned"
    source_repository = _sanitized_repository(
        _git_value(source_root, "remote", "get-url", "origin")
    )
    resolved_runner_revision = _git_value(
        runner_root,
        "rev-parse",
        "HEAD",
    )
    if not resolved_runner_revision:
        raise ValueError("evaluation runner is not inside a Git revision")

    return EvaluationProvenanceBatch(
        run_ids=run_ids,
        capture_timing="runtime",
        source_repository=source_repository,
        source_revision=source_revision,
        source_tree_sha256=_source_tree_sha256(source_root),
        source_dirty=(
            _git_dirty(source_git_root)
            if source_revision_value is not None
            else True
        ),
        runner_revision=resolved_runner_revision,
        runner_dirty=_git_dirty(runner_root),
        main_model_name=os.environ.get("DEEPFIX_MAIN_MODEL", "deepseek-v4-pro"),
        compaction_model_name=os.environ.get(
            "DEEPFIX_COMPACTION_MODEL",
            "deepseek-v4-flash",
        ),
        endpoint_fingerprint=_endpoint_fingerprint(
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ),
        python_version=_python_version(python_executable),
        python_executable_sha256=_file_sha256(python_executable),
        budget_enforcement="post_run_observation",
        model_accounting="main_model_trace_only",
    )


def _git_root(path: Path) -> Path:
    value = _git_value(path, "rev-parse", "--show-toplevel")
    return Path(value).resolve() if value else path.resolve()


def _git_value(path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _git_dirty(root: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("cannot read Git working tree status") from error
    if completed.returncode != 0:
        raise ValueError("cannot read Git working tree status")
    return bool(completed.stdout.strip())


def _sanitized_repository(value: str | None) -> str:
    if not value:
        return "local-unversioned"
    if value.startswith("git@") and ":" in value:
        host, path = value.removeprefix("git@").split(":", 1)
        return f"ssh://{host}/{path}"
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https", "ssh"} and parsed.hostname:
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit(
            (parsed.scheme, f"{parsed.hostname}{port}", parsed.path, "", "")
        )
    return "local"


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_TREE_DIRECTORIES for part in relative.parts):
            continue
        if path.is_dir() or path.suffix.lower() in _IGNORED_TREE_SUFFIXES:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(f"symlink:{path.readlink()}".encode())
        elif path.is_file():
            digest.update(bytes.fromhex(_file_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoint_fingerprint(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    host = parsed.hostname or "invalid"
    port = f":{parsed.port}" if parsed.port is not None else ""
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            f"{host.lower()}{port}",
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _python_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("cannot identify evaluation Python") from error
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise ValueError("cannot identify evaluation Python")
    return version
