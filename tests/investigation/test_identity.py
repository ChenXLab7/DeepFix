import pytest

from deepfix.investigation.identity import stable_investigation_id


def test_investigation_ids_are_deterministic_and_scoped():
    first = stable_investigation_id("event", "task-a", "tool-1", "abc")

    assert first == stable_investigation_id("event", "task-a", "tool-1", "abc")
    assert first != stable_investigation_id("event", "task-b", "tool-1", "abc")
    assert first.startswith("event_")


@pytest.mark.parametrize(
    ("prefix", "task_id", "parts"),
    [
        ("", "task-a", ("tool-1",)),
        ("event", "", ("tool-1",)),
        ("event", "task-a", ("",)),
    ],
)
def test_investigation_ids_reject_blank_identity_material(prefix, task_id, parts):
    with pytest.raises(ValueError, match="不能为空"):
        stable_investigation_id(prefix, task_id, *parts)
