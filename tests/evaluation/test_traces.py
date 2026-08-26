import json

from deepfix.evaluation.traces import summarize_llm_trace


def test_trace_summary_counts_only_matching_response_events(tmp_path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    records = [
        {
            "event": "request",
            "task_id": "t1",
            "input_tokens": 500,
            "output_tokens": 500,
            "duration_seconds": 9,
        },
        {
            "event": "response",
            "task_id": "t1",
            "input_tokens": 30,
            "output_tokens": 5,
            "duration_seconds": 1.2,
        },
        {
            "event": "response",
            "task_id": "t2",
            "input_tokens": 999,
            "output_tokens": 999,
            "duration_seconds": 9,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    usage = summarize_llm_trace(path, "t1")

    assert (usage.input_tokens, usage.output_tokens, usage.model_calls) == (30, 5, 1)
    assert usage.wall_seconds == 1.2
    assert usage.usage_estimated is False


def test_trace_summary_ignores_malformed_records(tmp_path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(
        "not-json\n"
        '{"event":"response","task_id":"t1","input_tokens":"secret"}\n'
        '{"event":"response","task_id":"t1","input_tokens":7,"output_tokens":3}\n',
        encoding="utf-8",
    )

    usage = summarize_llm_trace(path, "t1")

    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.model_calls == 1


def test_trace_summary_marks_missing_usage_as_estimated(tmp_path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(
        '{"event":"response","task_id":"t1","duration_seconds":0.5}\n',
        encoding="utf-8",
    )

    usage = summarize_llm_trace(path, "t1")

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.model_calls == 1
    assert usage.usage_estimated is True


def test_missing_trace_returns_empty_estimated_usage(tmp_path) -> None:
    usage = summarize_llm_trace(tmp_path / "missing.jsonl", "t1")

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.model_calls == 0
    assert usage.usage_estimated is True
