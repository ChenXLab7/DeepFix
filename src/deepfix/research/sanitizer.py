from __future__ import annotations

import math
import re
import socket
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SanitizedQuery:
    value: str


@dataclass(frozen=True)
class ValidatedUrl:
    value: str
    host: str


class QueryRejected(ValueError):
    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


class UnsafeUrl(ValueError):
    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*['\"]?\S{12,}",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:sk-(?:proj-)?|gh[pousr]_|AKIA)[A-Za-z0-9_-]{12,}"),
)
_CREDENTIAL_LIKE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{24,}(?![A-Za-z0-9])")
_WINDOWS_USER_PATH = re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE)
_UNC_PATH = re.compile(r"\\\\[^\\/\s]+[\\/]", re.IGNORECASE)
_UNIX_USER_PATH = re.compile(r"/(?:home|Users)/[^/\s]+/", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_DATABASE_URL = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|mssql)://\S+",
    re.IGNORECASE,
)
_INTERNAL_HOST = re.compile(
    r"\b(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.(?:internal|local|lan|corp))\b",
    re.IGNORECASE,
)
_PRIVATE_IPV4 = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|127(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_BUSINESS_IDENTIFIER = re.compile(
    r"\b(?:customer|order|account|tenant|user)[_-]?id\s*[:=]\s*[A-Za-z0-9_-]{8,}\b",
    re.IGNORECASE,
)


class QuerySanitizer:
    def sanitize(self, query: str) -> SanitizedQuery:
        value = query.strip()
        if not value:
            raise QueryRejected("empty", "技术查询不能为空")
        if len(value) > 2_000:
            raise QueryRejected("too_long", "技术查询不能超过 2,000 个字符")
        if len(value.splitlines()) > 20:
            raise QueryRejected("too_many_lines", "技术查询不能超过 20 行")
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            raise QueryRejected("secret", "技术查询疑似包含密钥或凭据")
        if any(
            pattern.search(value)
            for pattern in (_WINDOWS_USER_PATH, _UNC_PATH, _UNIX_USER_PATH)
        ):
            raise QueryRejected("user_path", "技术查询包含用户或共享目录路径")
        if any(len(line) > 400 for line in value.splitlines()):
            raise QueryRejected("long_block", "技术查询包含过长的连续源码或日志")
        if _DATABASE_URL.search(value):
            raise QueryRejected("database_url", "技术查询包含数据库连接串")
        if _EMAIL.search(value):
            raise QueryRejected("email", "技术查询包含邮箱地址")
        if _INTERNAL_HOST.search(value) or _PRIVATE_IPV4.search(value):
            raise QueryRejected("internal_host", "技术查询包含内部主机信息")
        if _BUSINESS_IDENTIFIER.search(value):
            raise QueryRejected("business_identifier", "技术查询包含业务标识")
        if self._contains_high_entropy_token(value):
            raise QueryRejected("secret", "技术查询疑似包含高熵凭据")
        return SanitizedQuery(value)

    @staticmethod
    def _contains_high_entropy_token(value: str) -> bool:
        for match in _CREDENTIAL_LIKE_TOKEN.finditer(value):
            token = match.group(0)
            character_classes = sum(
                (
                    any(character.islower() for character in token),
                    any(character.isupper() for character in token),
                    any(character.isdigit() for character in token),
                )
            )
            if character_classes >= 3 and _shannon_entropy(token) >= 3.5:
                return True
        return False


Resolver = Callable[[str], Iterable[str]]


class UrlSafetyPolicy:
    def __init__(self, resolver: Resolver | None = None) -> None:
        self._resolver = resolver or _resolve_public_addresses

    def validate(
        self,
        url: str,
        allowed_domains: Collection[str] = (),
    ) -> ValidatedUrl:
        if re.search(r"[\x00-\x20\x7f]", url):
            raise UnsafeUrl("malformed_url", "URL 包含空白或控制字符")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise UnsafeUrl("malformed_url", "URL 格式无效") from exc
        if parsed.scheme.lower() != "https":
            raise UnsafeUrl("https_required", "只允许 HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeUrl("userinfo", "URL 不能包含用户信息")
        if port not in {None, 443}:
            raise UnsafeUrl("nonstandard_port", "URL 不能使用非标准端口")
        if not parsed.hostname:
            raise UnsafeUrl("malformed_url", "URL 缺少主机名")

        host = _normalize_domain(parsed.hostname)
        if _is_ip_literal(host):
            raise UnsafeUrl("ip_literal", "URL 不能使用 IP 字面量")
        if _is_forbidden_hostname(host):
            raise UnsafeUrl("forbidden_host", "URL 指向受禁止的主机名")

        normalized_allowlist = {
            _normalize_domain(domain)
            for domain in allowed_domains
            if domain.strip()
        }
        if normalized_allowlist and not any(
            host == domain or host.endswith(f".{domain}")
            for domain in normalized_allowlist
        ):
            raise UnsafeUrl("domain_not_allowed", "URL 不在官方域名白名单中")

        try:
            addresses = list(self._resolver(host))
        except Exception as exc:
            raise UnsafeUrl("dns_failure", "无法安全解析 URL 主机名") from exc
        if not addresses:
            raise UnsafeUrl("dns_failure", "URL 主机名没有可验证的地址")
        for value in addresses:
            try:
                address = ip_address(value)
            except ValueError as exc:
                raise UnsafeUrl("dns_failure", "DNS 返回了无效地址") from exc
            if not address.is_global or address.is_multicast:
                raise UnsafeUrl("non_public_address", "URL 解析到了非公网地址")

        return ValidatedUrl(value=url, host=host)


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _normalize_domain(value: str) -> str:
    try:
        return value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise UnsafeUrl("malformed_url", "URL 主机名无效") from exc


def _is_ip_literal(host: str) -> bool:
    try:
        ip_address(host)
    except ValueError:
        return False
    return True


def _is_forbidden_hostname(host: str) -> bool:
    forbidden_names = {"localhost", "metadata.google.internal"}
    forbidden_suffixes = (".localhost", ".local", ".internal", ".lan")
    return host in forbidden_names or host.endswith(forbidden_suffixes)


def _resolve_public_addresses(host: str) -> list[str]:
    return list(
        dict.fromkeys(
            result[4][0]
            for result in socket.getaddrinfo(
                host,
                443,
                type=socket.SOCK_STREAM,
            )
        )
    )
