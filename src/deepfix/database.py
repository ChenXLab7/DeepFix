from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SQLiteDatabase:
    """Own SQLite connection configuration and bounded transactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, *, check_same_thread: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=check_same_thread,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def unit_of_work(
        self,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
