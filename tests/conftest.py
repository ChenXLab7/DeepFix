from __future__ import annotations

import pytest

import deepfix.config


@pytest.fixture(autouse=True)
def isolate_package_env_file(tmp_path, monkeypatch):
    """Prevent offline tests from reading a developer's real src/deepfix/.env."""
    monkeypatch.setattr(
        deepfix.config,
        "_DEFAULT_ENV_FILE",
        tmp_path / "missing-deepfix.env",
    )
