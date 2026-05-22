#!/usr/bin/env python3
"""Audit API-Football historical World Cup backfill completeness.

Research-only QA for 2018/2022 API-Football World Cup player-power sources.
This catches smoke-test partials where the fixture master has 64 matches but
bundle/player raw files only contain the first N fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SEASONS = "2018,2022"
DEFAULT_RAW = Path("data_sources/api_football/raw")
DEFAULT_NORMALIZED = Path("data_sources/api_football/normalized")
DEFAULT_FEATURES = Path("data_sources/api_football/features")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/historical_api_backfill_readiness")


def count_fixture_ids(path: Path) -> int:
    ids: set[int] = set()
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in payload.get("response") or []:
                fixture = item.get("fixture") or {}
                if fixture.get("id") is not None:
                    ids.add(int(fixture["id"]))
    return len(ids)


def count_bundle_fixtures(path: Path) -> int:
    ids: set[int] = set()
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in payload.get("response") or []:
                fixture = item.get("fixture") or {}
                if fixture.get("id") is not None:
                    ids.add(int(fixture["id"]))
    return len(ids)


def count_fixture_player_payload(path: Path) -> tuple[int, int]:
    fixtures: set[int] = set()
    players = 0
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            fixture_id = ((payload.get("parameters") or {}).get("fixture"))
            if fixture_id is not None:
                try:
                    fixtures.add(int(fixture_id))
                except ValueError:
                    pass
            for team in payload.get("response") or []:
                players += len(team.get("players") or [])
    return len(fixtures), players


def count_jsonl_response_rows(path: Path) -> tuple[int, int]:
    payloads = 0
    rows = 0
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payloads += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = payload.get("response")
            if isinstance(response, list):
                rows += len(response)
    return payloads, rows


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, low_memory=False)))
    except Exception:
        return 0


def readiness(row: dict[str, object], expected: int) -> str:
    if row["fixtures_raw_fixtures"] != expected:
        return "BLOCKED_FIXTURE_MASTER_INCOMPLETE"
    if row["bundle_raw_fixtures"] != expected:
        return "PARTIAL_FIXTURE_DETAIL_BUNDLE"
    if row["fixture_players_raw_fixtures"] != expected:
        return "PARTIAL_FIXTURE_PLAYER_BUNDLE"
    if row["match_player_stats_rows"] < expected * 35:
        return "PLAYER_ROWS_LOW_CHECK_NORMALIZATION"
    return "READY_FOR_LAGGED_PLAYER_POWER_BACKBUILD"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    seasons = [int(item.strip()) for item in str(args.seasons).split(",") if item.strip()]
    rows = []
    expected = 64
    for season in seasons:
        stem = f"fixtures__league_1__season_{season}"
        fixtures_raw = args.raw_dir / f"{stem}__fixtures.jsonl"
        bundle_raw = args.raw_dir / f"{stem}__fixtures_bundle.jsonl"
        injuries_raw = args.raw_dir / f"{stem}__injuries.jsonl"
        players_raw = args.raw_dir / f"{stem}__fixtures_players.jsonl"
        roster_raw = args.raw_dir / f"players__league_1__season_{season}__players.jsonl"
        roster_payloads, roster_rows = count_jsonl_response_rows(roster_raw)
        injury_payloads, injury_rows = count_jsonl_response_rows(injuries_raw)
        fixture_player_fixtures, fixture_player_rows = count_fixture_player_payload(players_raw)
        row = {
            "season": season,
            "expected_fixtures": expected,
            "fixtures_raw_fixtures": count_fixture_ids(fixtures_raw),
            "bundle_raw_fixtures": count_bundle_fixtures(bundle_raw),
            "fixture_players_raw_fixtures": fixture_player_fixtures,
            "fixture_players_raw_player_rows": fixture_player_rows,
            "injury_payloads": injury_payloads,
            "injury_rows": injury_rows,
            "roster_payload_pages": roster_payloads,
            "roster_response_rows": roster_rows,
            "fixtures_master_rows": csv_rows(args.normalized_dir / f"fixtures_master__World_Cup__{season}.csv"),
            "match_team_stats_rows": csv_rows(args.normalized_dir / f"match_team_stats__World_Cup__{season}.csv"),
            "match_player_stats_rows": csv_rows(args.normalized_dir / f"match_player_stats__World_Cup__{season}.csv"),
            "lineups_rows": csv_rows(args.normalized_dir / f"lineups__World_Cup__{season}.csv"),
            "player_rolling_feature_rows": csv_rows(args.features_dir / f"api_player_rolling_features__World_Cup__{season}.csv"),
        }
        row["readiness_bucket"] = readiness(row, expected)
        rows.append(row)

    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "world_cup_api_historical_backfill_readiness.csv"
    out.to_csv(csv_path, index=False)
    md = [
        "# World Cup API Historical Backfill Readiness",
        "",
        markdown_table(out),
        "",
        "## Guardrail",
        "",
        "`PARTIAL_FIXTURE_DETAIL_BUNDLE` or `PARTIAL_FIXTURE_PLAYER_BUNDLE` is expected after a smoke test with `--limit-fixtures`. "
        "Do not use those derived feature files for model validation until the full 64-fixture raw bundles are present per season.",
        "",
        f"CSV: `{csv_path}`",
        "",
    ]
    (args.outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] wrote {args.outdir}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
