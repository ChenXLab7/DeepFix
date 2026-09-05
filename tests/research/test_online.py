from __future__ import annotations

import os

import httpx
import pytest

RUN_ONLINE = os.environ.get("DEEPFIX_RUN_ONLINE") == "1"


@pytest.mark.online
@pytest.mark.skipif(
    not RUN_ONLINE,
    reason="设置 DEEPFIX_RUN_ONLINE=1 后才运行公共 PyPI 冒烟测试",
)
def test_public_pypi_metadata_endpoint_is_reachable():
    response = httpx.get(
        "https://pypi.org/pypi/pydantic/json",
        timeout=httpx.Timeout(15.0, connect=5.0),
    )

    response.raise_for_status()
    payload = response.json()
    assert payload["info"]["name"].lower() == "pydantic"
    assert payload["info"]["version"]


@pytest.mark.online
@pytest.mark.skipif(
    not RUN_ONLINE,
    reason="设置 DEEPFIX_RUN_ONLINE=1 后才运行公共 GitHub 冒烟测试",
)
def test_public_github_repository_endpoint_is_reachable():
    response = httpx.get(
        "https://api.github.com/repos/pydantic/pydantic",
        headers={"Accept": "application/vnd.github+json"},
        timeout=httpx.Timeout(15.0, connect=5.0),
    )

    response.raise_for_status()
    payload = response.json()
    assert payload["full_name"] == "pydantic/pydantic"
