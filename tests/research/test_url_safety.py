from __future__ import annotations

import socket

import pytest

from deepfix.research.sanitizer import UnsafeUrl, UrlSafetyPolicy

PUBLIC_IPV4 = "93.184.216.34"
PUBLIC_IPV6 = "2606:4700:4700::1111"


def _resolver(addresses_by_host):
    def resolve(host):
        value = addresses_by_host[host]
        if isinstance(value, Exception):
            raise value
        return value

    return resolve


@pytest.mark.parametrize(
    ("url", "rule"),
    [
        ("http://docs.example.com/page", "https_required"),
        ("https://user:password@docs.example.com/page", "userinfo"),
        ("https://docs.example.com:8443/page", "nonstandard_port"),
        ("https://127.0.0.1/page", "ip_literal"),
        ("https://[::1]/page", "ip_literal"),
        ("https://localhost/page", "forbidden_host"),
        ("https://metadata.google.internal/latest", "forbidden_host"),
    ],
)
def test_url_structure_rejects_unsafe_targets_before_dns(url, rule):
    policy = UrlSafetyPolicy(resolver=lambda host: pytest.fail(f"DNS called for {host}"))

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate(url)

    assert captured.value.rule == rule


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.8",
        "172.16.0.8",
        "192.168.1.8",
        "169.254.169.254",
        "224.0.0.1",
        "192.0.2.10",
        "::1",
        "fe80::1",
    ],
)
def test_dns_rejects_non_public_address(address):
    policy = UrlSafetyPolicy(resolver=lambda host: [address])

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate("https://docs.example.com/page")

    assert captured.value.rule == "non_public_address"


def test_dns_rejects_host_when_any_answer_is_private():
    policy = UrlSafetyPolicy(resolver=lambda host: [PUBLIC_IPV4, "10.0.0.8"])

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate("https://docs.example.com/page")

    assert captured.value.rule == "non_public_address"


def test_dns_failure_is_rejected():
    policy = UrlSafetyPolicy(
        resolver=_resolver({"docs.example.com": socket.gaierror("not found")})
    )

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate("https://docs.example.com/page")

    assert captured.value.rule == "dns_failure"


def test_empty_dns_result_is_rejected():
    policy = UrlSafetyPolicy(resolver=lambda host: [])

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate("https://docs.example.com/page")

    assert captured.value.rule == "dns_failure"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/page",
        "https://docs.example.com/page",
        "https://api.docs.example.com/page",
    ],
)
def test_allowlist_accepts_exact_domain_and_subdomains(url):
    policy = UrlSafetyPolicy(resolver=lambda host: [PUBLIC_IPV4, PUBLIC_IPV6])

    validated = policy.validate(url, allowed_domains=["example.com"])

    assert validated.value == url
    assert validated.host in {
        "example.com",
        "docs.example.com",
        "api.docs.example.com",
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com.attacker.test/page",
        "https://notexample.com/page",
        "https://attacker.test/page",
    ],
)
def test_allowlist_rejects_domain_suffix_tricks(url):
    policy = UrlSafetyPolicy(resolver=lambda host: [PUBLIC_IPV4])

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate(url, allowed_domains=["example.com"])

    assert captured.value.rule == "domain_not_allowed"


def test_malformed_url_is_rejected():
    policy = UrlSafetyPolicy(resolver=lambda host: [PUBLIC_IPV4])

    with pytest.raises(UnsafeUrl) as captured:
        policy.validate("https://docs.example.com:invalid/page")

    assert captured.value.rule == "malformed_url"
