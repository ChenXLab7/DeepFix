from __future__ import annotations

import sqlite3
from pathlib import Path

from deepfix.database import SQLiteDatabase
from deepfix.task_domain.repository import TaskRepository

__all__ = ["TaskRepository", "open_sqlite_connection"]


def open_sqlite_connection(
    database_path: str | Path,
    *,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    return SQLiteDatabase(database_path).connect(
        check_same_thread=check_same_thread,
    )
