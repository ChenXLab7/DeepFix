import json

import pytest

from deepfix.evaluation.manifest import load_manifest


def test_manifest_rejects_duplicate_case_ids(tmp_path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "case_id": "same",
        "problem": "run the required test",
        "allowed_paths": ["python_programs/x.py"],
        "required_command": "python -m pytest python_testcases/test_x.py -q",
        "expected_outcome": "fixed",
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "budget": {
                    "max_input_tokens": 100_000,
                    "max_output_tokens": 20_000,
                    "max_wall_seconds": 600,
                    "max_tool_calls": 40,
                    "max_side_effects": 5,
                },
                "cases": [case, case],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_manifest(path)
