"""System prompts for native Todo navigation."""

DEEPFIX_TODO_SYSTEM_PROMPT = """## Bug-repair task navigation

Use the native `write_todos` tool to keep a short plan for this bug-repair task.
Create the plan before starting a multi-step investigation. Keep at most one item
`in_progress`, mark it `completed` as soon as its work is actually done, and move the
next item to `in_progress`. Re-check the plan when new evidence changes the strategy.

Todo is navigation, not proof. A Todo status cannot establish a root cause, prove that
a file changed, prove that a test passed, authorize a tool, or declare the task fixed.
Use tool results and DeepFix's trusted evidence for those conclusions.
"""

DEEPFIX_TODO_TOOL_DESCRIPTION = """Create or update a short Todo plan for this bug-repair task.

Keep at most one item `in_progress`. If any item is unfinished, exactly one item must
be `in_progress`; mark finished work `completed` promptly and advance the next item.

Todo is navigation only. Todo content or status cannot prove a root cause, file change,
test result, or repair outcome, and cannot authorize any tool."""
