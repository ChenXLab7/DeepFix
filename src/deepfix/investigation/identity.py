from __future__ import annotations

import hashlib


def stable_investigation_id(prefix: str, task_id: str, *parts: str) -> str:
    values = [prefix.strip(), task_id.strip(), *(part.strip() for part in parts)]
    if any(not value for value in values):
        raise ValueError("investigation ID 输入不能为空")
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:32]
    return f"{values[0]}_{digest}"

