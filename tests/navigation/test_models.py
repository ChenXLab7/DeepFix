from deepfix.navigation.models import (
    TodoNavigationState,
    todo_progress_fingerprint,
    validate_todos,
)


def test_navigation_state_reuses_native_todos():
    annotations = TodoNavigationState.__annotations__
    assert "_deepfix_todo_rounds_since_update" in annotations
    assert "_deepfix_last_completed_tool_round_id" in annotations
    assert "_deepfix_last_todo_progress_fingerprint" in annotations
    assert "_deepfix_last_navigation_hint_fingerprint" in annotations
    assert "_deepfix_pending_navigation_reminder" in annotations


def test_noncompleted_plan_requires_exactly_one_in_progress():
    invalid = validate_todos(
        [
            {"content": "reproduce", "status": "pending"},
            {"content": "diagnose", "status": "pending"},
        ]
    )
    assert invalid.valid is False
    assert invalid.error_code == "todo_requires_one_in_progress"


def test_empty_plan_is_valid():
    assert validate_todos([]).valid is True


def test_all_completed_plan_is_valid_without_current_item():
    result = validate_todos(
        [
            {"content": "reproduce", "status": "completed"},
            {"content": "diagnose", "status": "completed"},
        ]
    )
    assert result.valid is True


def test_exactly_one_current_item_is_valid():
    result = validate_todos(
        [
            {"content": "reproduce", "status": "completed"},
            {"content": "diagnose", "status": "in_progress"},
            {"content": "fix", "status": "pending"},
        ]
    )
    assert result.valid is True


def test_multiple_current_items_are_rejected():
    result = validate_todos(
        [
            {"content": "reproduce", "status": "in_progress"},
            {"content": "diagnose", "status": "in_progress"},
        ]
    )
    assert result.valid is False
    assert result.error_code == "todo_multiple_in_progress"


def test_invalid_native_status_is_rejected():
    result = validate_todos([{"content": "reproduce", "status": "blocked"}])
    assert result.valid is False
    assert result.error_code == "todo_invalid_status"


def test_prose_only_rewrite_does_not_change_progress_fingerprint():
    before = [{"content": "investigate parser", "status": "in_progress"}]
    after = [{"content": "carefully investigate the parser", "status": "in_progress"}]
    assert todo_progress_fingerprint(before) == todo_progress_fingerprint(after)


def test_status_transition_changes_progress_fingerprint():
    before = [{"content": "investigate parser", "status": "in_progress"}]
    after = [{"content": "investigate parser", "status": "completed"}]
    assert todo_progress_fingerprint(before) != todo_progress_fingerprint(after)


def test_current_item_movement_changes_progress_fingerprint():
    before = [
        {"content": "one", "status": "in_progress"},
        {"content": "two", "status": "pending"},
    ]
    after = [
        {"content": "one", "status": "pending"},
        {"content": "two", "status": "in_progress"},
    ]
    assert todo_progress_fingerprint(before) != todo_progress_fingerprint(after)


def test_list_cardinality_changes_progress_fingerprint():
    before = [{"content": "one", "status": "in_progress"}]
    after = [
        {"content": "one", "status": "in_progress"},
        {"content": "two", "status": "pending"},
    ]
    assert todo_progress_fingerprint(before) != todo_progress_fingerprint(after)


def test_progress_fingerprint_is_deterministic():
    todos = [{"content": "one", "status": "in_progress"}]
    assert todo_progress_fingerprint(todos) == todo_progress_fingerprint(todos)
