#!/usr/bin/env python3
"""Compile compact, incremental website publish artifacts from local site SQLite.

This keeps calculation local. Cloudflare receives compact page payloads plus
changed index rows, not the raw source/evidence tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("build/site_data/odds_genius.sqlite")
DEFAULT_OUTPUT_DIR = Path("build/site_publish/current")
DEFAULT_RESULTS_ARCHIVE = Path("frontend/public/data/results_archive.json")
DEFAULT_WEEKLY_RESULTS = Path("frontend/public/data/weekly_results.json")
DEFAULT_R2_PREFIX = "site-data/v1"

D1_TABLES = (
    "metadata",
    "fixtures",
    "fixture_decisions",
    "fixture_lineups",
    "fixture_h2h",
    "team_intelligence",
    "club_squads",
    "team_lineup_snapshots",
    "site_external_sources",
    "site_fixture_external_content",
    "site_fixture_context_payloads",
    "site_fixture_stats_payloads",
    "site_team_premium_payloads",
)

SOURCE_DETAIL_TABLES = (
    "site_player_identity_map",
    "site_player_match_stats",
    "site_team_match_stats",
    "site_match_events",
    "site_lineup_slots",
    "site_formation_slots",
    "site_fixture_market_intelligence",
    "site_player_event_shortlists",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_payload(row: sqlite3.Row | None, field: str = "payload_json") -> Any:
    if row is None:
        return None
    try:
        return json.loads(row[field])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: Any) -> tuple[str, int]:
    text = pretty_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_text(canonical_json(payload)), len(text.encode("utf-8"))


def sql_quote(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def table_pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = list(conn.execute(f"PRAGMA table_info({table})"))
    return [row["name"] for row in sorted((row for row in rows if row["pk"]), key=lambda row: row["pk"])]


def row_identity(row: sqlite3.Row, pk_columns: list[str]) -> str:
    return "|".join(str(row[column]) for column in pk_columns)


def insert_statement(table: str, columns: list[str], row: sqlite3.Row) -> str:
    column_sql = ", ".join(columns)
    values_sql = ", ".join(sql_quote(row[column]) for column in columns)
    return f"INSERT OR REPLACE INTO {table}({column_sql}) VALUES ({values_sql});"


def delete_statement(table: str, pk_columns: list[str], identity: str) -> str:
    values = identity.split("|")
    where = " AND ".join(f"{column} = {sql_quote(value)}" for column, value in zip(pk_columns, values))
    return f"DELETE FROM {table} WHERE {where};"


def compact_table_row(row: sqlite3.Row, columns: list[str]) -> dict[str, Any]:
    return {column: row[column] for column in columns if row[column] is not None}


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def previous_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path, {})
    return manifest if isinstance(manifest, dict) else {}


def previous_hash_map(manifest: dict[str, Any], section: str) -> dict[str, str]:
    rows = manifest.get(section) if isinstance(manifest, dict) else []
    out: dict[str, str] = {}
    for row in rows if isinstance(rows, list) else []:
        key = row.get("object_key") or row.get("row_key")
        digest = row.get("sha256")
        if key and digest:
            out[str(key)] = str(digest)
    return out


def fixture_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM fixtures ORDER BY kickoff_time, fixture_key"))


def team_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT *
            FROM team_intelligence
            ORDER BY competition_key, team_slug, season DESC
            """
        )
    )


def one_by_key(conn: sqlite3.Connection, table: str, key_column: str, key_value: Any) -> sqlite3.Row | None:
    return conn.execute(f"SELECT * FROM {table} WHERE {key_column} = ? LIMIT 1", (key_value,)).fetchone()


def latest_team_sidecar(conn: sqlite3.Connection, table: str, competition_key: str, team_slug: str) -> sqlite3.Row | None:
    if table == "site_team_premium_payloads":
        return conn.execute(
            """
            SELECT *
            FROM site_team_premium_payloads
            WHERE competition_key = ? AND team_slug = ?
            LIMIT 1
            """,
            (competition_key, team_slug),
        ).fetchone()
    if table == "team_lineup_snapshots":
        return conn.execute(
            """
            SELECT *
            FROM team_lineup_snapshots
            WHERE competition_key = ? AND team_key = ?
            LIMIT 1
            """,
            (competition_key, team_slug),
        ).fetchone()
    return conn.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE competition_key = ? AND {"club_slug" if table == "club_squads" else "team_slug"} = ?
        ORDER BY season DESC
        LIMIT 1
        """,
        (competition_key, team_slug),
    ).fetchone()


def compile_fixture_objects(conn: sqlite3.Connection, output_dir: Path, r2_prefix: str, previous: dict[str, str]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    payload_root = output_dir / "payloads"
    for fixture in fixture_rows(conn):
        fixture_key = fixture["fixture_key"]
        decision = one_by_key(conn, "fixture_decisions", "fixture_key", fixture_key)
        lineup = one_by_key(conn, "fixture_lineups", "fixture_key", fixture_key)
        h2h = one_by_key(conn, "fixture_h2h", "fixture_key", fixture_key)
        stats = one_by_key(conn, "site_fixture_stats_payloads", "fixture_key", fixture_key)
        payload = {
            "schema": "fixture_page_payload_v1",
            "fixture_key": fixture_key,
            "fixture": parse_payload(fixture),
            "decision": parse_payload(decision),
            "lineup": parse_payload(lineup),
            "h2h": parse_payload(h2h),
            "stats": parse_payload(stats),
            "source_tables": [
                "fixtures",
                "fixture_decisions",
                "fixture_lineups",
                "fixture_h2h",
                "site_fixture_stats_payloads",
            ],
        }
        rel_path = Path("payloads") / "fixtures" / f"{fixture_key}.json"
        object_key = f"{r2_prefix}/{rel_path.as_posix()}"
        digest, byte_count = write_json(payload_root / "fixtures" / f"{fixture_key}.json", payload)
        objects.append(
            {
                "object_key": object_key,
                "relative_path": rel_path.as_posix(),
                "domain": "fixture",
                "entity_key": fixture_key,
                "sha256": digest,
                "bytes": byte_count,
                "changed": previous.get(object_key) != digest,
            }
        )
    return objects


def compile_team_objects(conn: sqlite3.Connection, output_dir: Path, r2_prefix: str, previous: dict[str, str]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    payload_root = output_dir / "payloads"
    for team in team_rows(conn):
        competition_key = team["competition_key"]
        team_slug = team["team_slug"]
        key = (competition_key, team_slug)
        if key in seen:
            continue
        seen.add(key)
        squad = latest_team_sidecar(conn, "club_squads", competition_key, team_slug)
        lineup_snapshot = latest_team_sidecar(conn, "team_lineup_snapshots", competition_key, team_slug)
        premium = latest_team_sidecar(conn, "site_team_premium_payloads", competition_key, team_slug)
        payload = {
            "schema": "team_page_payload_v1",
            "competition_key": competition_key,
            "team_slug": team_slug,
            "team": parse_payload(team),
            "squad": parse_payload(squad),
            "lineup_snapshot": parse_payload(lineup_snapshot),
            "premium": parse_payload(premium),
            "source_tables": [
                "team_intelligence",
                "club_squads",
                "team_lineup_snapshots",
                "site_team_premium_payloads",
            ],
        }
        rel_path = Path("payloads") / "teams" / competition_key / f"{team_slug}.json"
        object_key = f"{r2_prefix}/{rel_path.as_posix()}"
        digest, byte_count = write_json(payload_root / "teams" / competition_key / f"{team_slug}.json", payload)
        objects.append(
            {
                "object_key": object_key,
                "relative_path": rel_path.as_posix(),
                "domain": "team",
                "entity_key": f"{competition_key}/{team_slug}",
                "sha256": digest,
                "bytes": byte_count,
                "changed": previous.get(object_key) != digest,
            }
        )
    return objects


def compile_result_objects(
    output_dir: Path,
    r2_prefix: str,
    previous: dict[str, str],
    results_archive: Path,
    weekly_results: Path,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for source_path, name in ((results_archive, "results_archive"), (weekly_results, "weekly_results")):
        payload = read_json(source_path, {})
        rel_path = Path("payloads") / "results" / f"{name}.json"
        object_key = f"{r2_prefix}/{rel_path.as_posix()}"
        digest, byte_count = write_json(output_dir / rel_path, payload)
        objects.append(
            {
                "object_key": object_key,
                "relative_path": rel_path.as_posix(),
                "domain": "results",
                "entity_key": name,
                "source_path": str(source_path),
                "sha256": digest,
                "bytes": byte_count,
                "changed": previous.get(object_key) != digest,
            }
        )
    return objects


def compile_d1_delta(conn: sqlite3.Connection, previous: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows_manifest: list[dict[str, Any]] = []
    statements: list[str] = []
    current_keys: set[str] = set()
    pk_by_table: dict[str, list[str]] = {}
    for table in D1_TABLES:
        columns = table_columns(conn, table)
        pk_columns = table_pk_columns(conn, table)
        pk_by_table[table] = pk_columns
        for row in conn.execute(f"SELECT {', '.join(columns)} FROM {table}"):
            identity = row_identity(row, pk_columns)
            row_key = f"{table}:{identity}"
            digest = sha256_text(canonical_json(compact_table_row(row, columns)))
            changed = previous.get(row_key) != digest
            current_keys.add(row_key)
            rows_manifest.append(
                {
                    "row_key": row_key,
                    "table": table,
                    "identity": identity,
                    "sha256": digest,
                    "changed": changed,
                }
            )
            if changed:
                statements.append(insert_statement(table, columns, row))
    for row_key in sorted(set(previous) - current_keys):
        table, identity = row_key.split(":", 1)
        if table in D1_TABLES:
            statements.append(delete_statement(table, pk_by_table.get(table) or ["id"], identity))
    return rows_manifest, statements


def write_d1_delta(path: Path, statements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(statements)
    path.write_text((body + "\n") if body else "-- No D1 index row changes.\n", encoding="utf-8")


def write_upload_plan(output_dir: Path, objects: list[dict[str, Any]]) -> None:
    changed = [item for item in objects if item.get("changed")]
    plan = {
        "changed_objects": len(changed),
        "total_changed_bytes": sum(int(item.get("bytes") or 0) for item in changed),
        "objects": changed,
        "note": "Upload these relative paths to their object_key targets, then apply d1_changed_index.sql.",
    }
    write_json(output_dir / "upload_plan.json", plan)


def compile_publish(
    db_path: Path,
    output_dir: Path,
    r2_prefix: str,
    previous_manifest_path: Path | None,
    results_archive: Path,
    weekly_results: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    previous = previous_manifest(previous_manifest_path or manifest_path)
    previous_objects = previous_hash_map(previous, "objects")
    previous_d1_rows = previous_hash_map(previous, "d1_rows")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with connect(db_path) as conn:
        objects: list[dict[str, Any]] = []
        objects.extend(compile_fixture_objects(conn, output_dir, r2_prefix, previous_objects))
        objects.extend(compile_team_objects(conn, output_dir, r2_prefix, previous_objects))
        objects.extend(compile_result_objects(output_dir, r2_prefix, previous_objects, results_archive, weekly_results))
        d1_rows, d1_statements = compile_d1_delta(conn, previous_d1_rows)

    write_d1_delta(output_dir / "d1_changed_index.sql", d1_statements)
    write_upload_plan(output_dir, objects)

    manifest = {
        "schema": "site_publish_manifest_v1",
        "created_unix": int(started),
        "source_db": str(db_path),
        "r2_prefix": r2_prefix,
        "objects": objects,
        "d1_rows": d1_rows,
        "summary": {
            "objects_total": len(objects),
            "objects_changed": sum(1 for item in objects if item.get("changed")),
            "objects_total_bytes": sum(int(item.get("bytes") or 0) for item in objects),
            "objects_changed_bytes": sum(int(item.get("bytes") or 0) for item in objects if item.get("changed")),
            "d1_rows_total": len(d1_rows),
            "d1_rows_changed": sum(1 for item in d1_rows if item.get("changed")),
            "d1_delta_sql": str(output_dir / "d1_changed_index.sql"),
            "source_detail_tables_excluded_from_publish": list(SOURCE_DETAIL_TABLES),
        },
    }
    write_json(manifest_path, manifest)
    changed_manifest = {
        **manifest,
        "objects": [item for item in objects if item.get("changed")],
        "d1_rows": [item for item in d1_rows if item.get("changed")],
    }
    write_json(output_dir / "changed_manifest.json", changed_manifest)
    return manifest["summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile compact incremental site publish artifacts from local SQLite.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--previous-manifest", default="")
    parser.add_argument("--r2-prefix", default=DEFAULT_R2_PREFIX)
    parser.add_argument("--results-archive", default=str(DEFAULT_RESULTS_ARCHIVE))
    parser.add_argument("--weekly-results", default=str(DEFAULT_WEEKLY_RESULTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous = Path(args.previous_manifest) if args.previous_manifest else None
    summary = compile_publish(
        Path(args.db),
        Path(args.output_dir),
        args.r2_prefix.strip("/"),
        previous,
        Path(args.results_archive),
        Path(args.weekly_results),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
