from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from deepfix.research.models import DependencyContext, DependencyFinding
from deepfix.research.providers import (
    CompositeTechnicalSearchProvider,
    GitHubProvider,
    ProviderFailure,
    PyPIProvider,
    TavilyProvider,
)
from deepfix.research.sanitizer import SanitizedQuery, UrlSafetyPolicy

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC_ADDRESS = "93.184.216.34"


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _context(
    *,
    repository: str | None = None,
    domains: list[str] | None = None,
) -> DependencyContext:
    return DependencyContext(
        package=DependencyFinding(
            package_name="pydantic",
            declared_constraints=[">=2,<3"],
            installed_version="2.8.4",
            python_executable="C:/project/.venv/Scripts/python.exe",
            source_files=["pyproject.toml"],
            diagnostic=None,
        ),
        official_repository=repository,
        official_domains=domains or [],
    )


def _url_policy() -> UrlSafetyPolicy:
    return UrlSafetyPolicy(resolver=lambda host: [PUBLIC_ADDRESS])


def test_pypi_discovers_official_repository_domains_and_e1_candidates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/pypi/pydantic/json"
        return httpx.Response(200, json=_fixture("pypi_project.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = PyPIProvider(client, _url_policy()).search(
            SanitizedQuery("pydantic 2 model_copy"),
            _context(),
        )

    assert result.context.official_repository == "pydantic/pydantic"
    assert result.context.official_domains == ["docs.pydantic.dev"]
    assert {candidate.source_type for candidate in result.candidates} == {
        "pypi_metadata",
        "official_docs",
        "official_source",
        "release_note",
    }
    assert all(candidate.evidence_level == "E1" for candidate in result.candidates)


def test_pypi_normalizes_package_name_in_request():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json=_fixture("pypi_project.json"))

    context = _context()
    context.package.package_name = "Pydantic_Core"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        PyPIProvider(client, _url_policy()).search(SanitizedQuery("query"), context)

    assert seen_paths == ["/pypi/pydantic-core/json"]


def test_github_refuses_search_without_confirmed_repository():
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GitHubProvider(client, _url_policy())
        with pytest.raises(ProviderFailure, match="官方仓库"):
            provider.search(SanitizedQuery("model_copy"), _context())


@pytest.mark.parametrize(
    "query",
    [
        "model_copy repo:attacker/repository",
        "model_copy OR validation",
        "model_copy org:attacker",
        "model_copy type:pr",
    ],
)
def test_github_rejects_query_syntax_that_can_escape_repository_scope(query):
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GitHubProvider(client, _url_policy())
        with pytest.raises(ProviderFailure, match="查询语法"):
            provider.search(
                SanitizedQuery(query),
                _context(repository="pydantic/pydantic"),
            )


def test_github_search_is_repo_scoped_and_classifies_rest_and_graphql_results():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/search/issues":
            return httpx.Response(200, json=_fixture("github_search.json"))
        if request.url.path == "/repos/pydantic/pydantic/releases":
            return httpx.Response(200, json=_fixture("github_release.json"))
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_fixture("github_discussions.json"))
        return httpx.Response(404, json={"message": "not found"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = GitHubProvider(
            client,
            _url_policy(),
            token="github-secret-token",
        ).search(
            SanitizedQuery("model_copy validation"),
            _context(repository="pydantic/pydantic"),
        )

    issue_request = next(request for request in requests if request.url.path == "/search/issues")
    assert "repo:pydantic/pydantic" in issue_request.url.params["q"]
    assert all("/search/code" not in request.url.path for request in requests)
    graphql_request = next(request for request in requests if request.url.path == "/graphql")
    graphql_payload = json.loads(graphql_request.content)
    assert "repo:pydantic/pydantic" in graphql_payload["variables"]["query"]
    assert all(
        request.headers["authorization"] == "Bearer github-secret-token"
        for request in requests
    )

    by_title = {candidate.title: candidate for candidate in result.candidates}
    assert by_title["Open user report"].evidence_level == "E3"
    assert by_title["Closed validation issue"].evidence_level == "E2"
    assert by_title["Merged model_copy fix"].source_type == "github_pr"
    assert by_title["Merged model_copy fix"].evidence_level == "E2"
    assert by_title["Maintainer confirmation"].evidence_level == "E2"
    assert by_title["Pydantic v2.8.4"].evidence_level == "E1"
    assert by_title["Answered model_copy question"].evidence_level == "E2"
    assert by_title["Unconfirmed migration question"].evidence_level == "E3"
    assert "github-secret-token" not in repr(result)


def test_github_without_token_keeps_rest_and_skips_discussions():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/search/issues":
            return httpx.Response(200, json=_fixture("github_search.json"))
        if request.url.path == "/repos/pydantic/pydantic/releases":
            return httpx.Response(200, json=_fixture("github_release.json"))
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = GitHubProvider(client, _url_policy()).search(
            SanitizedQuery("model_copy"),
            _context(repository="pydantic/pydantic"),
        )

    assert "/graphql" not in paths
    assert any(candidate.source_type == "github_issue" for candidate in result.candidates)
    assert result.errors == ["github_discussions: GITHUB_TOKEN 未配置，已跳过"]


def test_github_rate_limit_error_exposes_reset_but_not_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-ratelimit-reset": "1780000000"},
            json={"message": "rate limit"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GitHubProvider(client, _url_policy(), token="github-secret-token")
        with pytest.raises(ProviderFailure) as captured:
            provider.search(
                SanitizedQuery("model_copy"),
                _context(repository="pydantic/pydantic"),
            )

    assert "1780000000" in str(captured.value)
    assert "github-secret-token" not in str(captured.value)


def test_tavily_without_key_is_disabled_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = TavilyProvider(client, _url_policy(), api_key=None)
        result = provider.search(
            SanitizedQuery("model_copy"),
            _context(domains=["docs.pydantic.dev"]),
        )

    assert provider.enabled is False
    assert result.candidates == []
    assert result.errors == []


def test_tavily_uses_bearer_header_and_filters_results_by_official_domains():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://api.tavily.com/search")
        assert request.headers["authorization"] == "Bearer tavily-secret"
        payload = json.loads(request.content)
        assert payload["include_domains"] == ["docs.pydantic.dev"]
        assert payload["include_raw_content"] is False
        return httpx.Response(200, json=_fixture("tavily_search.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = TavilyProvider(
            client,
            _url_policy(),
            api_key="tavily-secret",
        ).search(
            SanitizedQuery("model_copy"),
            _context(domains=["docs.pydantic.dev"]),
        )

    assert [candidate.url for candidate in result.candidates] == [
        "https://docs.pydantic.dev/latest/concepts/models/"
    ]
    assert "tavily-secret" not in repr(result)


def test_composite_keeps_pypi_results_when_github_times_out():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pypi.org":
            return httpx.Response(200, json=_fixture("pypi_project.json"))
        raise httpx.ConnectTimeout("github timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        composite = CompositeTechnicalSearchProvider(
            [
                PyPIProvider(client, _url_policy()),
                GitHubProvider(client, _url_policy()),
                TavilyProvider(client, _url_policy(), api_key=None),
            ]
        )
        result = composite.search(
            SanitizedQuery("model_copy"),
            _context(),
        )

    assert "pypi" in result.providers
    assert any(candidate.source_type == "pypi_metadata" for candidate in result.candidates)
    assert any(error.startswith("github:") for error in result.errors)
    assert result.context.official_repository == "pydantic/pydantic"
