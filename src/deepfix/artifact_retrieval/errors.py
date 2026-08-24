from __future__ import annotations


class DiagnosticArtifactToolError(ValueError):
    def __init__(self, error_code: str, safe_message: str) -> None:
        self.error_code = _required(error_code, "error_code")
        self.safe_message = _required(safe_message, "safe_message")[:300]
        super().__init__(self.error_code)


class DiagnosticArtifactSystemError(RuntimeError):
    def __init__(self, error_code: str, stage: str) -> None:
        self.error_code = _required(error_code, "error_code")
        self.stage = _required(stage, "stage")
        super().__init__(self.error_code)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    return normalized
