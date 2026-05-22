#!/usr/bin/env python3
"""Audit World Cup player-event source coverage.

Research-only. This confirms which player-event markets can be studied from
the local API-Football World Cup cache and which 2026 layers are still only
pre-tournament roster/context scaffolds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data_sources" / "api_football" / "normalized"
RAW = ROOT / "data_sources" / "api_football" / "raw"
WC_ROOT = ROOT / "data_sources" / "footystats_world_cup"
DEFAULT_OUTDIR = WC_ROOT / "player_event_source_audit"

MARKETS: dict[str, dict[str, Any]] = {
    "shots_0_5": {"label": "Player Shots 0.5+", "column": "shots_total", "threshold": 1},
    "shots_1_5": {"label": "Player Shots 1.5+", "column": "shots_total", "threshold": 2},
    "sot_0_5": {"label": "Player SOT 0.5+", "column": "shots_on_target", "threshold": 1},
    "keeper_saves_1_5": {"label": "Keeper Saves 1.5+", "column": "saves", "threshold": 2, "position": "G"},
    "keeper_saves_2_5": {"label": "Keeper Saves 2.5+", "column": "saves", "threshold": 3, "position": "G"},
    "tackles_1_5": {"label": "Player Tackles 1.5+", "column": "tackles", "threshold": 2},
    "fouls_1_5": {"label": "Player Fouls 1.5+", "column": "fouls_committed", "threshold": 2},
    "cards_0_5": {"label": "Player Cards 0.5+ Hazard", "column": "yellow_cards", "threshold": 1},
}


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except UnicodeDecodeError:
        return max(0, sum(1 for _ in path.open("r", encoding="latin-1")) - 1)


def season_paths(season: int) -> dict[str, Path]:
    return {
        "fixtures": NORMALIZED / f"fixtures_master__World_Cup__{season}.csv",
        "player_stats": NORMALIZED / f"match_player_stats__World_Cup__{season}.csv",
        "team_stats": NORMALIZED / f"match_team_stats__World_Cup__{season}.csv",
        "lineups": NORMALIZED / f"lineups__World_Cup__{season}.csv",
        "injuries": NORMALIZED / f"injuries__World_Cup__{season}.csv",
        "raw_fixtures": RAW / f"fixtures__league_1__season_{season}__fixtures.jsonl",
        "raw_fixture_bundle": RAW / f"fixtures__league_1__season_{season}__fixtures_bundle.jsonl",
        "raw_fixture_players": RAW / f"fixtures__league_1__season_{season}__fixtures_players.jsonl",
        "raw_injuries": RAW / f"fixtures__league_1__season_{season}__injuries.jsonl",
        "raw_players": RAW / f"players__league_1__season_{season}__players.jsonl",
    }


def audit_historical_season(season: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = season_paths(season)
    summary: dict[str, Any] = {"season": season}
    for name, path in paths.items():
        summary[f"{name}_exists"] = int(path.exists())
        summary[f"{name}_rows"] = csv_rows(path) if path.suffix == ".csv" else count_jsonl_rows(path)

    if not paths["player_stats"].exists():
        return [], summary

    stats = pd.read_csv(paths["player_stats"], low_memory=False)
    fixtures = pd.read_csv(paths["fixtures"], low_memory=False) if paths["fixtures"].exists() else pd.DataFrame()
    fixture_count = int(fixtures["fixture_id"].nunique()) if "fixture_id" in fixtures.columns else 0
    rows: list[dict[str, Any]] = []
    for market, spec in MARKETS.items():
        col = spec["column"]
        work = stats.copy()
        if "position" in spec:
            work = work[work["position"].astype(str).str.upper().eq(str(spec["position"]).upper())]
        has_col = col in work.columns
        graded = work[work["minutes"].fillna(0).astype(float).gt(0)].copy() if "minutes" in work.columns else work.copy()
        non_null = int(num(graded[col]).notna().sum()) if has_col else 0
        hit_rows = int(num(graded[col]).ge(float(spec["threshold"])).sum()) if has_col else 0
        rows.append(
            {
                "season": season,
                "market": market,
                "market_label": spec["label"],
                "source_column": col,
                "threshold": spec["threshold"],
                "fixtures": fixture_count,
                "player_rows": int(len(work)),
                "graded_player_rows": int(len(graded)),
                "non_null_actual_rows": non_null,
                "hit_rows": hit_rows,
                "hit_rate": hit_rows / non_null if non_null else pd.NA,
                "coverage_bucket": "READY_FOR_BACKTEST" if fixture_count == 64 and non_null else "MISSING_OR_PARTIAL",
            }
        )
    return rows, summary


def audit_2026() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = WC_ROOT / "api_bridge" / "world_cup_api_fixture_schedule.csv"
    roster = WC_ROOT / "player_intelligence_2026" / "world_cup_2026_api_player_roster_scheduled_teams_only.csv"
    additions = WC_ROOT / "additions_context_2026" / "world_cup_additions_player_source_rows.csv"
    summary = {
        "season": 2026,
        "schedule_exists": int(schedule.exists()),
        "schedule_rows": csv_rows(schedule),
        "roster_exists": int(roster.exists()),
        "roster_rows": csv_rows(roster),
        "additions_player_source_exists": int(additions.exists()),
        "additions_player_source_rows": csv_rows(additions),
        "raw_players_exists": int((RAW / "players__league_1__season_2026__players.jsonl").exists()),
        "raw_players_rows": count_jsonl_rows(RAW / "players__league_1__season_2026__players.jsonl"),
        "raw_fixtures_exists": int((RAW / "fixtures__league_1__season_2026__fixtures.jsonl").exists()),
        "raw_fixtures_rows": count_jsonl_rows(RAW / "fixtures__league_1__season_2026__fixtures.jsonl"),
    }
    fixture_rows = 0
    schedule_total_rows = 0
    if schedule.exists():
        sched = pd.read_csv(schedule, low_memory=False)
        schedule_total_rows = len(sched)
        if "season" in sched.columns:
            sched = sched[sched["season"].eq(2026)].copy()
        fixture_rows = int(sched["api_fixture_id"].nunique()) if "api_fixture_id" in sched.columns else len(sched)
    summary["schedule_total_rows"] = schedule_total_rows
    summary["schedule_rows"] = fixture_rows
    player_rows = csv_rows(roster)
    rows = []
    for market, spec in MARKETS.items():
        rows.append(
            {
                "season": 2026,
                "market": market,
                "market_label": spec["label"],
                "source_column": spec["column"],
                "threshold": spec["threshold"],
                "fixtures": fixture_rows,
                "player_rows": player_rows,
                "graded_player_rows": 0,
                "non_null_actual_rows": 0,
                "hit_rows": 0,
                "hit_rate": pd.NA,
                "coverage_bucket": "PRE_TOURNAMENT_ROSTER_CONTEXT_ONLY",
            }
        )
    return rows, summary


def write_summary(outdir: Path, coverage: pd.DataFrame, inventory: pd.DataFrame) -> None:
    lines = [
        "# World Cup Player-Event Source Audit",
        "",
        "- Research-only. No production deploy gates or priced props are changed.",
        "- Historical 2018/2022 coverage is judged from local normalized API-Football World Cup player stats.",
        "- 2026 is pre-tournament context only until official lineups, injuries, and match player stats arrive.",
        "",
        "## Market Coverage",
    ]
    for _, row in coverage.sort_values(["season", "market"]).iterrows():
        hit = "" if pd.isna(row["hit_rate"]) else f" hit_rate={float(row['hit_rate']):.3f}"
        lines.append(
            f"- {int(row['season'])} | {row['market_label']} | {row['coverage_bucket']} | fixtures={int(row['fixtures'])} | "
            f"graded_rows={int(row['graded_player_rows'])} | hits={int(row['hit_rows'])}{hit}"
        )
    lines += [
        "",
        "## Source Inventory",
    ]
    for _, row in inventory.sort_values("season").iterrows():
        season = int(row["season"])
        bits = []
        for col in inventory.columns:
            if col == "season" or not col.endswith("_rows"):
                continue
            bits.append(f"{col}={row[col]}")
        lines.append(f"- {season}: " + ", ".join(bits))
    lines += [
        "",
        "## Read",
        "- Shots, SOT, tackles, fouls, cards, and keeper saves are historically present for 2018 and 2022.",
        "- Historical injuries are not useful locally: API injury payloads exist but normalized injury rows are zero for both 2018 and 2022.",
        "- 2026 can support a ranked pre-tournament intelligence board, but official squad/lineup/injury truth remains pending.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local World Cup player-event source coverage.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for season in (2018, 2022):
        rows, summary = audit_historical_season(season)
        coverage_rows.extend(rows)
        inventory_rows.append(summary)
    rows, summary = audit_2026()
    coverage_rows.extend(rows)
    inventory_rows.append(summary)

    coverage = pd.DataFrame(coverage_rows)
    inventory = pd.DataFrame(inventory_rows)
    coverage.to_csv(outdir / "world_cup_player_event_market_coverage.csv", index=False)
    inventory.to_csv(outdir / "world_cup_player_event_source_inventory.csv", index=False)
    write_summary(outdir, coverage, inventory)
    print(f"[ok] wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
