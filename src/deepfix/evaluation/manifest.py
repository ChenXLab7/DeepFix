from __future__ import annotations

from pathlib import Path

from deepfix.evaluation.models import EvaluationManifest


def load_manifest(path: Path) -> EvaluationManifest:
    manifest = EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    case_ids = [case.case_id for case in manifest.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id")
    return manifest
