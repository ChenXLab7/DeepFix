from __future__ import annotations

import sqlite3

import pytest

from deepfix.database import SQLiteDatabase


def test_database_connections_share_wal_and_busy_timeout(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state" / "deepfix.db")

    with database.connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_unit_of_work_commits_all_writes_together(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    with database.connection() as connection:
        connection.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
        connection.commit()

    with database.unit_of_work() as connection:
        connection.execute("INSERT INTO values_table VALUES ('a')")
        connection.execute("INSERT INTO values_table VALUES ('b')")

    with database.connection() as connection:
        assert connection.execute("SELECT value FROM values_table").fetchall() == [
            ("a",),
            ("b",),
        ]


def test_unit_of_work_rolls_back_every_write_on_error(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")
    with database.connection() as connection:
        connection.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
        connection.commit()

    with (
        pytest.raises(RuntimeError, match="abort"),
        database.unit_of_work(immediate=True) as connection,
    ):
        connection.execute("INSERT INTO values_table VALUES ('a')")
        raise RuntimeError("abort")

    with database.connection() as connection:
        assert connection.execute("SELECT value FROM values_table").fetchall() == []


def test_connection_context_closes_owned_connection(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "deepfix.db")

    with database.connection() as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
