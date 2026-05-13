#!/usr/bin/env python3
"""Export the local site SQLite artifact into D1-compatible SQL chunks."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("build/site_data/odds_genius.sqlite")
DEFAULT_OUTPUT_DIR = Path("build/site_data/d1_chunks")
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
TABLES = (
    "metadata",
    "fixtures",
    "fixture_decisions",
    "fixture_lineups",
    "fixture_h2h",
    "team_intelligence",
    "club_squads",
    "team_lineup_snapshots",
    "site_player_identity_map",
    "site_player_match_stats",
    "site_team_match_stats",
    "site_match_events",
    "site_lineup_slots",
    "site_formation_slots",
    "site_fixture_market_intelligence",
    "site_player_event_shortlists",
    "site_fixture_stats_payloads",
    "site_team_premium_payloads",
)


def sql_quote(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def schema_sql(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE sql IS NOT NULL
          AND type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
        """
    ).fetchall()
    statements = []
    for row in rows:
        if row["type"] == "index":
            statements.append(f"DROP INDEX IF EXISTS {row['name']};")
    for row in rows:
        if row["type"] == "table":
            statements.append(f"DROP TABLE IF EXISTS {row['name']};")
    statements.extend(f"{row['sql']};" for row in rows)
    return statements


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def write_chunk(output_dir: Path, index: int, statements: list[str]) -> Path:
    path = output_dir / f"{index:04d}.sql"
    path.write_text("\n".join(statements) + "\n", encoding="utf-8")
    return path


def export_chunks(db_path: Path, output_dir: Path, max_bytes: int) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    chunks: list[Path] = []
    counts: dict[str, int] = {}
    try:
        chunks.append(write_chunk(output_dir, 0, schema_sql(conn)))
        chunk_index = 1
        pending: list[str] = []
        pending_bytes = 0
        for table in TABLES:
            columns = table_columns(conn, table)
            column_sql = ", ".join(columns)
            counts[table] = 0
            for row in conn.execute(f"SELECT {column_sql} FROM {table}"):
                values_sql = ", ".join(sql_quote(row[column]) for column in columns)
                statement = f"INSERT INTO {table}({column_sql}) VALUES ({values_sql});"
                statement_bytes = len(statement.encode("utf-8")) + 1
                if pending and pending_bytes + statement_bytes > max_bytes:
                    chunks.append(write_chunk(output_dir, chunk_index, pending))
                    chunk_index += 1
                    pending = []
                    pending_bytes = 0
                pending.append(statement)
                pending_bytes += statement_bytes
                counts[table] += 1
        if pending:
            chunks.append(write_chunk(output_dir, chunk_index, pending))
    finally:
        conn.close()

    return {
        "chunks": len(chunks),
        "counts": counts,
        "output_dir": str(output_dir),
        "total_bytes": sum(path.stat().st_size for path in chunks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local site SQLite data to D1 SQL chunks.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_chunks(Path(args.db), Path(args.output_dir), args.max_bytes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
