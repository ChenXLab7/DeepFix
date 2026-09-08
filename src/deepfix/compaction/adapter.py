from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AnyMessage, messages_from_dict, messages_to_dict
from langchain_core.messages.utils import get_buffer_string

from deepfix.compaction.errors import ArtifactPersistenceError
from deepfix.compaction.models import ArtifactReference, CompactionFailureRecord

_TASK_ID = re.compile(r"[A-Za-z0-9_-]+")


class DeepAgentsArtifactAdapter:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def persist_history(
        self,
        task_id: str,
        attempt_id: str,
        messages: Sequence[AnyMessage],
        retained_ids: AbstractSet[str],
        *,
        work_unit_ids: AbstractSet[str] | None = None,
    ) -> ArtifactReference:
        prepared = _prepare(
            task_id,
            attempt_id,
            messages,
            retained_ids,
            work_unit_ids or set(),
        )
        existing = self._download(prepared.path, prepared)
        idempotent = _existing_event_hash(existing, prepared)
        if idempotent is not None:
            return prepared.reference(idempotent)
        updated = _append(existing, prepared.section)
        try:
            result = (
                self.backend.write(prepared.path, updated)
                if existing is None
                else self.backend.edit(prepared.path, existing, updated)
            )
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_write_failed") from exc
        if getattr(result, "error", None):
            raise _artifact_error(prepared, "artifact_write_failed")
        verified = self._download_after_write(prepared.path, prepared)
        _verify_event(verified, prepared)
        return prepared.reference(prepared.section_hash)

    async def apersist_history(
        self,
        task_id: str,
        attempt_id: str,
        messages: Sequence[AnyMessage],
        retained_ids: AbstractSet[str],
        *,
        work_unit_ids: AbstractSet[str] | None = None,
    ) -> ArtifactReference:
        prepared = _prepare(
            task_id,
            attempt_id,
            messages,
            retained_ids,
            work_unit_ids or set(),
        )
        existing = await self._adownload(prepared.path, prepared)
        idempotent = _existing_event_hash(existing, prepared)
        if idempotent is not None:
            return prepared.reference(idempotent)
        updated = _append(existing, prepared.section)
        try:
            result = (
                await self.backend.awrite(prepared.path, updated)
                if existing is None
                else await self.backend.aedit(prepared.path, existing, updated)
            )
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_write_failed") from exc
        if getattr(result, "error", None):
            raise _artifact_error(prepared, "artifact_write_failed")
        verified = await self._adownload_after_write(prepared.path, prepared)
        _verify_event(verified, prepared)
        return prepared.reference(prepared.section_hash)

    def read_verified(self, path: str) -> str:
        response = self.backend.download_files([path])[0]
        if response.error is not None or response.content is None:
            raise ValueError(f"artifact 不可读取: {path}")
        try:
            return response.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"artifact 不是 UTF-8: {path}") from exc

    def restore_history(self, path: str, attempt_id: str) -> list[AnyMessage]:
        text = self.read_verified(path)
        attempt = re.escape(html.escape(attempt_id, quote=True))
        event = re.search(
            rf'<deepfix_history_event attempt_id="{attempt}"[^>]*>(.*?)</deepfix_history_event>',
            text, re.DOTALL,
        )
        record = re.search(r'<message_objects sha256="([a-f0-9]+)">(.*?)</message_objects>',
                           event.group(1) if event else "", re.DOTALL)
        if record is None:
            raise ValueError("history has no restorable message objects")
        payload = html.unescape(record.group(2))
        if _sha256(payload) != record.group(1):
            raise ValueError("history message objects hash mismatch")
        return messages_from_dict(json.loads(payload))

    def _download(self, path: str, prepared: _PreparedHistory) -> str | None:
        try:
            response = self.backend.download_files([path])[0]
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_read_failed") from exc
        return _decode_download(response, prepared, allow_missing=True)

    def _download_after_write(self, path: str, prepared: _PreparedHistory) -> str:
        try:
            response = self.backend.download_files([path])[0]
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_verify_failed") from exc
        value = _decode_download(response, prepared, allow_missing=False)
        assert value is not None
        return value

    async def _adownload(
        self,
        path: str,
        prepared: _PreparedHistory,
    ) -> str | None:
        try:
            response = (await self.backend.adownload_files([path]))[0]
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_read_failed") from exc
        return _decode_download(response, prepared, allow_missing=True)

    async def _adownload_after_write(
        self,
        path: str,
        prepared: _PreparedHistory,
    ) -> str:
        try:
            response = (await self.backend.adownload_files([path]))[0]
        except Exception as exc:
            raise _artifact_error(prepared, "artifact_verify_failed") from exc
        value = _decode_download(response, prepared, allow_missing=False)
        assert value is not None
        return value


class _PreparedHistory:
    def __init__(
        self,
        *,
        task_id: str,
        attempt_id: str,
        path: str,
        section: str,
        section_hash: str,
        input_hash: str,
        work_unit_ids: list[str],
    ) -> None:
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.path = path
        self.section = section
        self.section_hash = section_hash
        self.input_hash = input_hash
        self.work_unit_ids = work_unit_ids

    def reference(self, content_hash: str) -> ArtifactReference:
        return ArtifactReference(
            path=self.path,
            kind="conversation_history",
            content_hash=content_hash,
            work_unit_ids=self.work_unit_ids,
        )


def _prepare(
    task_id: str,
    attempt_id: str,
    messages: Sequence[AnyMessage],
    retained_ids: AbstractSet[str],
    work_unit_ids: AbstractSet[str],
) -> _PreparedHistory:
    task_id = task_id.strip()
    attempt_id = attempt_id.strip()
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("task_id 只能包含字母、数字、下划线和连字符")
    if not attempt_id:
        raise ValueError("attempt_id 不能为空")
    message_ids = [str(message.id or "").strip() for message in messages]
    if any(not message_id for message_id in message_ids):
        raise ValueError("写 history 前所有 Message 必须具有稳定 ID")
    if len(set(message_ids)) != len(message_ids):
        raise ValueError("写 history 前 Message ID 必须唯一")
    unknown_retained = set(retained_ids) - set(message_ids)
    if unknown_retained:
        raise ValueError(f"retained_ids 不属于消息历史: {sorted(unknown_retained)}")

    manifests = []
    for message, message_id in zip(messages, message_ids, strict=True):
        kind = "retained_message" if message_id in retained_ids else "compressed_message"
        manifests.append(
            f'<{kind} id="{html.escape(message_id, quote=True)}" '
            f'content_hash="{_message_hash(message)}" />'
        )
    serialized = get_buffer_string(messages, format="xml") if messages else ""
    objects = json.dumps(messages_to_dict(list(messages)), ensure_ascii=False, sort_keys=True)
    body = "\n".join(
        [
            "<message_manifest>",
            *manifests,
            "</message_manifest>",
            "<serialized_messages>",
            serialized,
            "</serialized_messages>",
            f'<message_objects sha256="{_sha256(objects)}">{html.escape(objects)}</message_objects>',
        ]
    )
    input_hash = _sha256(body)
    section_hash = _sha256(
        "|".join((task_id, attempt_id, input_hash, *sorted(work_unit_ids)))
    )
    section = "\n".join(
        [
            (
                f'<deepfix_history_event attempt_id="{html.escape(attempt_id, quote=True)}" '
                f'event_hash="{section_hash}" created_at="{_utc_now()}">'
            ),
            body,
            "</deepfix_history_event>",
        ]
    )
    return _PreparedHistory(
        task_id=task_id,
        attempt_id=attempt_id,
        path=f"/.deepfix-artifacts/conversation_history/{task_id}.md",
        section=section,
        section_hash=section_hash,
        input_hash=input_hash,
        work_unit_ids=sorted(work_unit_ids),
    )


def _existing_event_hash(
    existing: str | None,
    prepared: _PreparedHistory,
) -> str | None:
    if existing is None:
        return None
    attempt = html.escape(prepared.attempt_id, quote=True)
    exact = f'attempt_id="{attempt}" event_hash="{prepared.section_hash}"'
    if exact in existing:
        return prepared.section_hash
    if f'attempt_id="{attempt}"' in existing:
        raise _artifact_error(prepared, "artifact_attempt_conflict")
    return None


def _append(existing: str | None, section: str) -> str:
    if not existing:
        return section + "\n"
    return existing.rstrip("\n") + "\n\n" + section + "\n"


def _decode_download(
    response: Any,
    prepared: _PreparedHistory,
    *,
    allow_missing: bool,
) -> str | None:
    if response.error == "file_not_found" and allow_missing:
        return None
    if response.error is not None or response.content is None:
        code = "artifact_read_failed" if allow_missing else "artifact_verify_failed"
        raise _artifact_error(prepared, code)
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _artifact_error(prepared, "artifact_verify_failed") from exc


def _verify_event(value: str, prepared: _PreparedHistory) -> None:
    marker = (
        f'attempt_id="{html.escape(prepared.attempt_id, quote=True)}" '
        f'event_hash="{prepared.section_hash}"'
    )
    if marker not in value:
        raise _artifact_error(prepared, "artifact_verify_failed")
    if prepared.section not in value:
        raise _artifact_error(prepared, "artifact_hash_mismatch")


def _message_hash(message: AnyMessage) -> str:
    payload = json.dumps(
        message.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return _sha256(payload)


def _artifact_error(
    prepared: _PreparedHistory,
    error_code: str,
) -> ArtifactPersistenceError:
    stage = "artifact_verify" if "verify" in error_code or "hash" in error_code else "artifact_write"
    return ArtifactPersistenceError(
        CompactionFailureRecord(
            attempt_id=prepared.attempt_id,
            task_id=prepared.task_id,
            entrypoint="automatic",
            budget_zone="normal_compaction",
            stage=stage,
            error_code=error_code,
            input_hash=prepared.input_hash,
            original_messages_preserved=True,
            artifact_reference=prepared.path,
            prepared_snapshot_version=None,
            recorded_at=_utc_now(),
        )
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
