from __future__ import annotations

import os
import sys
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

from deepfix.config import AppConfig

_SAFE_ENVIRONMENT_VARIABLES = (
    "COMSPEC",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
)


def _build_project_backend(config: AppConfig) -> LocalShellBackend:
    shell_environment = {
        name: value
        for name in _SAFE_ENVIRONMENT_VARIABLES
        if (value := os.environ.get(name)) is not None
    }
    executable_directory = str(Path(sys.executable).parent)
    inherited_path = os.environ.get("PATH", "")
    shell_environment["PATH"] = os.pathsep.join(
        item for item in (executable_directory, inherited_path) if item
    )

    return LocalShellBackend(
        root_dir=config.project_root,
        virtual_mode=True,
        timeout=config.shell_timeout_seconds,
        max_output_bytes=100_000,
        env=shell_environment,
        inherit_env=False,
    )


def build_backend(config: AppConfig) -> CompositeBackend:
    """Build isolated project and internal-artifact filesystem routes."""
    return CompositeBackend(
        default=_build_project_backend(config),
        routes={
            "/.deepfix-artifacts/": FilesystemBackend(
                root_dir=config.artifacts_path,
                virtual_mode=True,
            )
        },
        artifacts_root="/.deepfix-artifacts",
    )
