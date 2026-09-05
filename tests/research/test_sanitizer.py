from __future__ import annotations

import pytest

from deepfix.research.sanitizer import QueryRejected, QuerySanitizer


@pytest.mark.parametrize(
    ("query", "expected_rule"),
    [
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.secret.payload", "secret"),
        ("api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456", "secret"),
        ("-----BEGIN PRIVATE KEY-----", "secret"),
        ("token 4f9aB7cD2eF8gH1jK6mN3pQ5rS0tV4wX", "secret"),
        (r"错误位于 C:\Users\alice\project\src\app.py", "user_path"),
        (r"读取 \\server\share\private\config.toml", "user_path"),
        ("错误位于 /home/alice/project/app.py", "user_path"),
        ("错误位于 /Users/alice/project/app.py", "user_path"),
        ("\n".join(f"line {number}" for number in range(21)), "too_many_lines"),
        ("x" * 2001, "too_long"),
        ("x" * 401, "long_block"),
        ("contact alice@example.com about this traceback", "email"),
        ("service failed at orders.api.internal", "internal_host"),
        ("postgresql://user:password@db.example.com/orders", "database_url"),
        ("customer_id=ACME-CUSTOMER-2026-000001", "business_identifier"),
    ],
)
def test_sensitive_query_is_rejected_without_rewriting(query, expected_rule):
    sanitizer = QuerySanitizer()

    with pytest.raises(QueryRejected) as captured:
        sanitizer.sanitize(query)

    assert captured.value.rule == expected_rule
    assert not hasattr(captured.value, "sanitized_query")


@pytest.mark.parametrize(
    "query",
    [
        "pydantic 2.8.4 ValidationError model_copy update behavior",
        "PEP 517 build backend subprocess-exited-with-error",
        "httpx 0.27 ConnectTimeout retry API",
        "TypeError: BaseModel.model_dump() got an unexpected keyword argument",
        "pytest fixture ScopeMismatch short traceback",
    ],
)
def test_general_python_technical_query_is_accepted(query):
    result = QuerySanitizer().sanitize(f"  {query}  ")

    assert result.value == query


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query):
    with pytest.raises(QueryRejected) as captured:
        QuerySanitizer().sanitize(query)

    assert captured.value.rule == "empty"
