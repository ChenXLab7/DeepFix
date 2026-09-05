from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from packaging.utils import canonicalize_name
from pydantic import BaseModel, Field

from deepfix.research.models import (
    DependencyContext,
    EvidenceLevel,
    SourceType,
)
from deepfix.research.sanitizer import (
    SanitizedQuery,
    UnsafeUrl,
    UrlSafetyPolicy,
)


class ProviderFailure(RuntimeError):
    pass


class SearchCandidateDraft(BaseModel):
    source_type: SourceType
    evidence_level: EvidenceLevel
    title: str
    url: str
    repository: str | None = None


class ProviderResult(BaseModel):
    candidates: list[SearchCandidateDraft] = Field(default_factory=list)
    context: DependencyContext
    errors: list[str] = Field(default_factory=list)


class TechnicalSearchResult(ProviderResult):
    providers: list[str] = Field(default_factory=list)


class TechnicalSearchProvider(Protocol):
    name: str
    enabled: bool

    def search(
        self,
        query: SanitizedQuery,
        context: DependencyContext,
    ) -> ProviderResult: ...


class PyPIProvider:
    name = "pypi"
    enabled = True

    def __init__(self, client: httpx.Client, url_policy: UrlSafetyPolicy) -> None:
        self.client = client
        self.url_policy = url_policy

    def search(
        self,
        query: SanitizedQuery,
        context: DependencyContext,
    ) -> ProviderResult:
        del query
        package_name = canonicalize_name(context.package.package_name)
        payload = _request_json(
            self.client,
            "GET",
            f"https://pypi.org/pypi/{package_name}/json",
            provider=self.name,
        )
        if not isinstance(payload, dict):
            raise ProviderFailure("PyPI 返回了无效响应")
        info = payload.get("info")
        if not isinstance(info, dict):
            raise ProviderFailure("PyPI 响应缺少项目信息")

        project_name = str(info.get("name") or package_name)
        version = str(info.get("version") or "unknown")
        candidates = [
            SearchCandidateDraft(
                source_type="pypi_metadata",
                evidence_level="E1",
                title=f"{project_name} {version} PyPI metadata",
                url=f"https://pypi.org/project/{package_name}/",
            )
        ]
        enriched = context.model_copy(deep=True)
        repository = enriched.official_repository
        domains = list(enriched.official_domains)
        errors: list[str] = []

        project_urls = info.get("project_urls")
        if isinstance(project_urls, dict):
            for raw_label, raw_url in project_urls.items():
                if not isinstance(raw_label, str) or not isinstance(raw_url, str):
                    continue
                try:
                    validated = self.url_policy.validate(raw_url)
                except UnsafeUrl as exc:
                    errors.append(f"pypi_project_url: {raw_label}: {exc.rule}")
                    continue
                label = raw_label.strip().lower()
                discovered_repository = _github_repository(validated.value)
                if discovered_repository and _is_source_label(label):
                    repository = repository or discovered_repository
                    candidates.append(
                        SearchCandidateDraft(
                            source_type="official_source",
                            evidence_level="E1",
                            title=f"{project_name} official source",
                            url=validated.value,
                            repository=discovered_repository,
                        )
                    )
                    continue
                if _is_documentation_label(label):
                    if validated.host != "github.com" and validated.host not in domains:
                        domains.append(validated.host)
                    source_type: SourceType = (
                        "release_note" if _is_release_label(label) else "official_docs"
                    )
                    candidates.append(
                        SearchCandidateDraft(
                            source_type=source_type,
                            evidence_level="E1",
                            title=f"{project_name} {raw_label}",
                            url=validated.value,
                            repository=repository,
                        )
                    )

        enriched.official_repository = repository
        enriched.official_domains = domains
        return ProviderResult(candidates=candidates, context=enriched, errors=errors)


class GitHubProvider:
    name = "github"
    enabled = True

    def __init__(
        self,
        client: httpx.Client,
        url_policy: UrlSafetyPolicy,
        *,
        token: str | None = None,
    ) -> None:
        self.client = client
        self.url_policy = url_policy
        self._token = token.strip() if token and token.strip() else None

    def search(
        self,
        query: SanitizedQuery,
        context: DependencyContext,
    ) -> ProviderResult:
        repository = context.official_repository
        if not repository or not _valid_repository(repository):
            raise ProviderFailure("缺少已确认的 GitHub 官方仓库")
        if _UNSAFE_GITHUB_QUALIFIER.search(query.value) or re.search(
            r"\bOR\b",
            query.value,
        ):
            raise ProviderFailure("查询语法包含不允许的 GitHub 限定符或布尔操作")

        scoped_query = f"{query.value} repo:{repository}"
        issue_payload = _request_json(
            self.client,
            "GET",
            "https://api.github.com/search/issues",
            provider=self.name,
            headers=self._headers(),
            params={"q": scoped_query, "per_page": "10"},
        )
        candidates = self._issue_candidates(issue_payload, repository)
        errors: list[str] = []

        try:
            release_payload = _request_json(
                self.client,
                "GET",
                f"https://api.github.com/repos/{repository}/releases",
                provider=self.name,
                headers=self._headers(),
                params={"per_page": "5"},
            )
            candidates.extend(self._release_candidates(release_payload, repository))
        except ProviderFailure as exc:
            errors.append(f"github_releases: {exc}")

        if self._token is None:
            errors.append("github_discussions: GITHUB_TOKEN 未配置，已跳过")
        else:
            try:
                discussion_payload = _request_json(
                    self.client,
                    "POST",
                    "https://api.github.com/graphql",
                    provider=self.name,
                    headers=self._headers(),
                    json_body={
                        "query": _DISCUSSION_QUERY,
                        "variables": {
                            "query": scoped_query,
                        },
                    },
                )
                candidates.extend(
                    self._discussion_candidates(discussion_payload, repository)
                )
            except ProviderFailure as exc:
                errors.append(f"github_discussions: {exc}")

        return ProviderResult(
            candidates=candidates,
            context=context.model_copy(deep=True),
            errors=errors,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "deepfix-agent",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _issue_candidates(
        self,
        payload: object,
        repository: str,
    ) -> list[SearchCandidateDraft]:
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ProviderFailure("GitHub Issue 搜索返回了无效响应")
        candidates: list[SearchCandidateDraft] = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("html_url")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            if not self._is_safe_github_url(url, repository):
                continue
            pull_request = item.get("pull_request")
            is_pull_request = isinstance(pull_request, dict)
            level = _github_item_level(item, is_pull_request)
            candidates.append(
                SearchCandidateDraft(
                    source_type="github_pr" if is_pull_request else "github_issue",
                    evidence_level=level,
                    title=title,
                    url=url,
                    repository=repository,
                )
            )
        return candidates

    def _release_candidates(
        self,
        payload: object,
        repository: str,
    ) -> list[SearchCandidateDraft]:
        if not isinstance(payload, list):
            raise ProviderFailure("GitHub Release 返回了无效响应")
        candidates: list[SearchCandidateDraft] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("draft") is True:
                continue
            title = item.get("name") or item.get("tag_name")
            url = item.get("html_url")
            if (
                isinstance(title, str)
                and isinstance(url, str)
                and self._is_safe_github_url(url, repository)
            ):
                candidates.append(
                    SearchCandidateDraft(
                        source_type="release_note",
                        evidence_level="E1",
                        title=title,
                        url=url,
                        repository=repository,
                    )
                )
        return candidates

    def _discussion_candidates(
        self,
        payload: object,
        repository: str,
    ) -> list[SearchCandidateDraft]:
        if not isinstance(payload, dict) or payload.get("errors"):
            raise ProviderFailure("GitHub Discussion GraphQL 返回错误")
        data = payload.get("data")
        search = data.get("search") if isinstance(data, dict) else None
        nodes = search.get("nodes") if isinstance(search, dict) else None
        if not isinstance(nodes, list):
            raise ProviderFailure("GitHub Discussion 返回了无效响应")
        candidates: list[SearchCandidateDraft] = []
        for item in nodes:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            if not self._is_safe_github_url(url, repository):
                continue
            association = str(item.get("authorAssociation") or "").upper()
            confirmed = (
                association in _MAINTAINER_ASSOCIATIONS
                or item.get("answerChosenAt") is not None
                or item.get("closed") is True
            )
            candidates.append(
                SearchCandidateDraft(
                    source_type="github_discussion",
                    evidence_level="E2" if confirmed else "E3",
                    title=title,
                    url=url,
                    repository=repository,
                )
            )
        return candidates

    def _is_safe_github_url(self, url: str, repository: str) -> bool:
        try:
            validated = self.url_policy.validate(url, allowed_domains=["github.com"])
        except UnsafeUrl:
            return False
        path = urlsplit(validated.value).path.rstrip("/")
        return path == f"/{repository}" or path.startswith(f"/{repository}/")


class TavilyProvider:
    name = "tavily"

    def __init__(
        self,
        client: httpx.Client,
        url_policy: UrlSafetyPolicy,
        *,
        api_key: str | None,
    ) -> None:
        self.client = client
        self.url_policy = url_policy
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.enabled = self._api_key is not None

    def search(
        self,
        query: SanitizedQuery,
        context: DependencyContext,
    ) -> ProviderResult:
        if not self.enabled or self._api_key is None:
            return ProviderResult(context=context.model_copy(deep=True))
        domains = sorted(
            domain
            for domain in context.official_domains
            if domain != "github.com"
        )
        if not domains:
            raise ProviderFailure("缺少可供 Tavily 使用的官方文档域名")
        payload = _request_json(
            self.client,
            "POST",
            "https://api.tavily.com/search",
            provider=self.name,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json_body={
                "query": query.value,
                "search_depth": "basic",
                "topic": "general",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "include_domains": domains,
                "max_results": 5,
            },
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ProviderFailure("Tavily 返回了无效响应")
        candidates: list[SearchCandidateDraft] = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            try:
                validated = self.url_policy.validate(url, allowed_domains=domains)
            except UnsafeUrl:
                continue
            candidates.append(
                SearchCandidateDraft(
                    source_type="official_docs",
                    evidence_level="E1",
                    title=title,
                    url=validated.value,
                    repository=context.official_repository,
                )
            )
        return ProviderResult(
            candidates=candidates,
            context=context.model_copy(deep=True),
        )


class CompositeTechnicalSearchProvider:
    def __init__(self, providers: Sequence[TechnicalSearchProvider]) -> None:
        self.providers = tuple(providers)

    def search(
        self,
        query: SanitizedQuery,
        context: DependencyContext,
    ) -> TechnicalSearchResult:
        current_context = context.model_copy(deep=True)
        candidates: list[SearchCandidateDraft] = []
        provider_names: list[str] = []
        errors: list[str] = []
        for provider in self.providers:
            if not provider.enabled:
                continue
            provider_names.append(provider.name)
            try:
                result = provider.search(query, current_context)
            except ProviderFailure as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            current_context = result.context
            candidates.extend(result.candidates)
            errors.extend(result.errors)
        return TechnicalSearchResult(
            candidates=candidates,
            context=current_context,
            errors=errors,
            providers=provider_names,
        )


_MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
_CONFIRMING_LABELS = {"accepted", "confirmed", "fixed", "resolved"}
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_UNSAFE_GITHUB_QUALIFIER = re.compile(
    r"\b(?:repo|org|user|type|is):",
    re.IGNORECASE,
)
_DISCUSSION_QUERY = """
query SearchDiscussions($query: String!) {
  search(type: DISCUSSION, query: $query, first: 5) {
    nodes {
      ... on Discussion {
        title
        url
        bodyText
        authorAssociation
        answerChosenAt
        closed
      }
    }
  }
}
""".strip()


def _github_item_level(item: dict[str, object], is_pull_request: bool) -> EvidenceLevel:
    association = str(item.get("author_association") or "").upper()
    if association in _MAINTAINER_ASSOCIATIONS:
        return "E2"
    pull_request = item.get("pull_request")
    if is_pull_request and isinstance(pull_request, dict):
        return "E2" if pull_request.get("merged_at") is not None else "E3"
    if item.get("state") == "closed":
        return "E2"
    labels = item.get("labels")
    if isinstance(labels, list):
        names = {
            str(label.get("name") or "").strip().lower()
            for label in labels
            if isinstance(label, dict)
        }
        if names & _CONFIRMING_LABELS:
            return "E2"
    return "E3"


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_body: dict[str, object] | None = None,
) -> object:
    try:
        response = client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
        )
    except httpx.HTTPError as exc:
        raise ProviderFailure(f"请求失败: {type(exc).__name__}") from exc
    if response.status_code in {403, 429}:
        reset = response.headers.get("x-ratelimit-reset", "unknown")
        raise ProviderFailure(f"请求受限，reset={reset}")
    if response.status_code == 404:
        raise ProviderFailure("资源不存在")
    if response.status_code >= 400:
        raise ProviderFailure(f"HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderFailure(f"{provider} 返回了无效 JSON") from exc


def _github_repository(url: str) -> str | None:
    parsed = urlsplit(url)
    if (parsed.hostname or "").lower() != "github.com":
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        return None
    repository = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return repository if _valid_repository(repository) else None


def _valid_repository(value: str) -> bool:
    return _REPOSITORY.fullmatch(value) is not None


def _is_source_label(value: str) -> bool:
    return any(word in value for word in ("source", "repository", "github", "code"))


def _is_release_label(value: str) -> bool:
    return any(word in value for word in ("changelog", "change log", "release"))


def _is_documentation_label(value: str) -> bool:
    return _is_release_label(value) or any(
        word in value
        for word in ("documentation", "docs", "homepage", "home page")
    )
