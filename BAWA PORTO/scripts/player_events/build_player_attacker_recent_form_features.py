#!/usr/bin/env python3
"""
Build leak-safe rolling attacker recent-form features for player-event research.

The output is keyed by fixture/team/player and uses only matches before the
current fixture. It is research-only and intended for shots/SOT intelligence
audits, not production deploy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_attacker_recent_form_features"
DEFAULT_LEAGUES = ("England_Premier_League", "Spain_La_Liga")
DEFAULT_SEASONS = (2022, 2023, 2024)


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_actuals(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            fixtures_path = NORMALIZED_DIR / f"fixtures_master__{league}__{season}.csv"
            stats_path = NORMALIZED_DIR / f"match_player_stats__{league}__{season}.csv"
            fixtures = read_csv_if_exists(fixtures_path)
            stats = read_csv_if_exists(stats_path)
            if fixtures.empty or stats.empty:
                continue
            fixtures = fixtures[
                [
                    "fixture_id",
                    "fixture_key",
                    "match_date",
                    "kickoff_ts_utc",
                    "home_team_id",
                    "away_team_id",
                    "home_team_name",
                    "away_team_name",
                ]
            ].copy()
            merged = stats.merge(fixtures, on="fixture_id", how="left")
            merged["league_tag"] = league
            merged["season_tag"] = season
            merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce")
            merged["kickoff_ts_utc"] = pd.to_datetime(merged["kickoff_ts_utc"], errors="coerce", utc=True)
            merged["team_id"] = num(merged["team_id"]).astype("Int64")
            merged["player_id"] = num(merged["player_id"]).astype("Int64")
            merged["home_team_id"] = num(merged["home_team_id"]).astype("Int64")
            merged["away_team_id"] = num(merged["away_team_id"]).astype("Int64")
            merged["team_name"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["home_team_name"],
                merged["away_team_name"],
            )
            merged["player_team_side"] = np.where(merged["team_id"].eq(merged["home_team_id"]), "HOME", "AWAY")
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in [
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "goals",
        "assists",
        "passes_key",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = num(out[col]).fillna(0.0)
    return out.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)


def summarize_history(prev: list[dict[str, Any]], n: int, side: str | None = None) -> dict[str, float]:
    rows = prev[-n:]
    if side:
        rows = [r for r in reversed(prev) if r.get("player_team_side") == side][:n]
        rows = list(reversed(rows))
    apps = len(rows)
    minutes = sum(float(r.get("minutes", 0.0) or 0.0) for r in rows)
    starts = sum(float(r.get("started_flag", 0.0) or 0.0) for r in rows)
    shots = sum(float(r.get("shots_total", 0.0) or 0.0) for r in rows)
    sot = sum(float(r.get("shots_on_target", 0.0) or 0.0) for r in rows)
    goals = sum(float(r.get("goals", 0.0) or 0.0) for r in rows)
    assists = sum(float(r.get("assists", 0.0) or 0.0) for r in rows)
    key_passes = sum(float(r.get("passes_key", 0.0) or 0.0) for r in rows)
    prefix = f"attacker_recent_{'home_' if side == 'HOME' else 'away_' if side == 'AWAY' else ''}"
    return {
        f"{prefix}apps_l{n}": float(apps),
        f"{prefix}starts_l{n}": float(starts),
        f"{prefix}minutes_l{n}": float(minutes),
        f"{prefix}shots_l{n}": float(shots),
        f"{prefix}sot_l{n}": float(sot),
        f"{prefix}goals_l{n}": float(goals),
        f"{prefix}assists_l{n}": float(assists),
        f"{prefix}goal_contributions_l{n}": float(goals + assists),
        f"{prefix}key_passes_l{n}": float(key_passes),
        f"{prefix}shots_per90_l{n}": safe_div(shots * 90.0, minutes),
        f"{prefix}sot_per90_l{n}": safe_div(sot * 90.0, minutes),
        f"{prefix}goals_per90_l{n}": safe_div(goals * 90.0, minutes),
        f"{prefix}assists_per90_l{n}": safe_div(assists * 90.0, minutes),
        f"{prefix}goal_contributions_per90_l{n}": safe_div((goals + assists) * 90.0, minutes),
        f"{prefix}key_passes_per90_l{n}": safe_div(key_passes * 90.0, minutes),
        f"{prefix}start_share_l{n}": safe_div(starts, apps),
        f"{prefix}sot_share_l{n}": safe_div(sot, shots),
    }


def build_features(actuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    history: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for _, row in actuals.iterrows():
        if pd.isna(row.get("player_id")):
            continue
        player_key = (str(row.get("league_tag")), int(row.get("player_id")))
        prev = history.get(player_key, [])
        out: dict[str, Any] = {
            "fixture_id": int(row["fixture_id"]),
            "fixture_key": row.get("fixture_key"),
            "league_tag": row.get("league_tag"),
            "season_tag": int(row.get("season_tag")),
            "team_id": int(row["team_id"]),
            "team_name": row.get("team_name"),
            "player_id": int(row["player_id"]),
            "player_name": row.get("player_name"),
            "match_date": row.get("match_date"),
            "kickoff_ts_utc": row.get("kickoff_ts_utc"),
            "player_team_side": row.get("player_team_side"),
            "attacker_recent_history_matches": float(len(prev)),
            "attacker_recent_source_max_date": max([str(r.get("match_date", "")) for r in prev[-8:]], default=""),
            "attacker_recent_xg_available_flag": 0,
            "attacker_recent_xa_available_flag": 0,
            "attacker_recent_big_chances_available_flag": 0,
            "attacker_recent_touches_box_available_flag": 0,
        }
        for n in (5, 8):
            out.update(summarize_history(prev, n))
            out.update(summarize_history(prev, n, side="HOME"))
            out.update(summarize_history(prev, n, side="AWAY"))
        rows.append(out)
        history.setdefault(player_key, []).append(row.to_dict())
    return pd.DataFrame(rows)


def coverage_report(features: pd.DataFrame, leagues: tuple[str, ...], seasons: tuple[int, ...]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for (league, season), group in features.groupby(["league_tag", "season_tag"], dropna=False):
        rows.append(
            {
                "league_tag": league,
                "season_tag": season,
                "rows": int(len(group)),
                "players": int(group["player_id"].nunique()),
                "rows_with_any_history": int(group["attacker_recent_history_matches"].gt(0).sum()),
                "any_history_rate": float(group["attacker_recent_history_matches"].gt(0).mean()),
                "rows_with_l5_history": int(group["attacker_recent_apps_l5"].ge(5).sum()),
                "l5_history_rate": float(group["attacker_recent_apps_l5"].ge(5).mean()),
                "rows_with_l8_history": int(group["attacker_recent_apps_l8"].ge(8).sum()),
                "l8_history_rate": float(group["attacker_recent_apps_l8"].ge(8).mean()),
                "xg_xa_big_chances_touches_box_available": False,
            }
        )
    present = {(str(r["league_tag"]), int(r["season_tag"])) for r in rows}
    for league in leagues:
        for season in seasons:
            if (league, season) not in present:
                rows.append(
                    {
                        "league_tag": league,
                        "season_tag": season,
                        "rows": 0,
                        "players": 0,
                        "rows_with_any_history": 0,
                        "any_history_rate": 0.0,
                        "rows_with_l5_history": 0,
                        "l5_history_rate": 0.0,
                        "rows_with_l8_history": 0,
                        "l8_history_rate": 0.0,
                        "xg_xa_big_chances_touches_box_available": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(["league_tag", "season_tag"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 6)
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, features: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# Player Attacker Recent Form Features",
        "",
        "Research-only rolling feature table for player shots/SOT intelligence.",
        "",
        "## Safety",
        "- Uses only prior matches before each fixture.",
        "- Writes no production model artifacts.",
        "- Does not create priced player-prop odds or deploy routes.",
        "",
        "## Output",
        f"- rows: `{len(features)}`",
        f"- players: `{features['player_id'].nunique() if not features.empty else 0}`",
        "",
        "## Coverage",
        markdown_table(coverage),
        "",
        "## Available Recent Features",
        "- `shots_l5`, `shots_l8`, `sot_l5`, `sot_l8`",
        "- `goals_l5/l8`, `assists_l5/l8`, `goal_contributions_l5/l8`",
        "- `starts_l5/l8`, `apps_l5/l8`, `minutes_l5/l8`",
        "- home/away split versions for shots and SOT",
        "",
        "## Missing API Families",
        "- player-level `expected_goals` unavailable in current normalized files",
        "- player-level `expected_assists` unavailable in current normalized files",
        "- big chances unavailable in current normalized files",
        "- touches in box unavailable in current normalized files",
    ]
    (outdir / "PLAYER_ATTACKER_RECENT_FORM_FEATURES.md").write_text("\n".join(lines) + "\n")


def parse_csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    parser.add_argument("--seasons", default=",".join(map(str, DEFAULT_SEASONS)))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    leagues = parse_csv_tuple(args.leagues, str)
    seasons = parse_csv_tuple(args.seasons, int)
    args.outdir.mkdir(parents=True, exist_ok=True)

    actuals = load_actuals(leagues, seasons)
    features = build_features(actuals)
    coverage = coverage_report(features, leagues, seasons)

    features_path = args.outdir / "player_attacker_recent_form_features.csv"
    coverage_path = args.outdir / "player_attacker_recent_form_feature_coverage.csv"
    features.to_csv(features_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    write_report(args.outdir, features, coverage)

    print(f"WROTE {args.outdir}")
    print(f"rows={len(features)} players={features['player_id'].nunique() if not features.empty else 0}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
