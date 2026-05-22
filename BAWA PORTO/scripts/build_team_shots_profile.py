#!/usr/bin/env python3
"""Build research-only team shots/SOT pressure profiles.

This is an API-context feature board, not a deploy gate. It converts normalized
API-Football team match stats into leak-safe rolling team attack and concession
profiles that can support player shots/SOT, corners, keeper-save, and attacking
pressure watch labels.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "team_shots_profile"

TEAM_STATS_PREFIX = "match_team_stats__"
FIXTURES_PREFIX = "fixtures_master__"
ROLLING_WINDOWS = (5, 10)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_int_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_team_stats_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(rf"{TEAM_STATS_PREFIX}(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_csv_if_exists(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if usecols is None:
        return pd.read_csv(path, low_memory=False)
    header = pd.read_csv(path, nrows=0)
    available = [col for col in usecols if col in header.columns]
    return pd.read_csv(path, usecols=available, low_memory=False)


def safe_mean(records: list[dict[str, Any]], key: str, n: int) -> float:
    sample = records[-n:]
    if not sample:
        return np.nan
    values = [float(row.get(key, np.nan)) for row in sample if pd.notna(row.get(key, np.nan))]
    return float(np.mean(values)) if values else np.nan


def safe_rate(records: list[dict[str, Any]], predicate, n: int) -> float:
    sample = records[-n:]
    if not sample:
        return np.nan
    return float(sum(1 for row in sample if predicate(row)) / len(sample))


def rolling_summary(records: list[dict[str, Any]], side: str | None = None) -> dict[str, float]:
    work = records if side is None else [row for row in records if row.get("team_side") == side]
    out: dict[str, float] = {"history_matches": float(len(work))}
    metric_cols = [
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "shots_outside_box",
        "blocked_shots",
        "corners_for",
        "possession_pct",
        "passes_total",
        "passes_accurate",
        "goals_for",
        "shots_allowed",
        "sot_allowed",
        "box_shots_allowed",
        "corners_allowed",
        "goals_against",
        "fouls_for",
        "fouls_allowed",
        "cards_total",
    ]
    for n in ROLLING_WINDOWS:
        for col in metric_cols:
            out[f"{col}_l{n}"] = safe_mean(work, col, n)
        out[f"shot_accuracy_l{n}"] = (
            out[f"shots_on_goal_l{n}"] / out[f"shots_total_l{n}"]
            if out.get(f"shots_total_l{n}") and pd.notna(out.get(f"shots_total_l{n}"))
            else np.nan
        )
        out[f"high_shot_volume_rate_l{n}"] = safe_rate(work, lambda row: float(row.get("shots_total", 0) or 0) >= 14, n)
        out[f"high_sot_volume_rate_l{n}"] = safe_rate(work, lambda row: float(row.get("shots_on_goal", 0) or 0) >= 5, n)
        out[f"corner_5plus_rate_l{n}"] = safe_rate(work, lambda row: float(row.get("corners_for", 0) or 0) >= 5, n)
    return out


def load_team_fixture_rows(normalized_dir: Path, leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fixture_cols = [
        "fixture_id",
        "fixture_key",
        "league",
        "league_id",
        "season",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "kickoff_ts_utc",
        "status",
    ]
    stats_cols = [
        "fixture_id",
        "team_id",
        "team_name",
        "is_home",
        "goals_for",
        "goals_against",
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "shots_outside_box",
        "blocked_shots",
        "possession_pct",
        "passes_total",
        "passes_accurate",
        "corners_for",
        "fouls_for",
        "yellow_cards",
        "red_cards",
    ]
    for stats_path in sorted(normalized_dir.glob(f"{TEAM_STATS_PREFIX}*.csv")):
        parsed = parse_team_stats_tag(stats_path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        if leagues and league_tag not in leagues:
            continue
        if seasons and season_tag not in seasons:
            continue
        fixtures_path = normalized_dir / f"{FIXTURES_PREFIX}{league_tag}__{season_tag}.csv"
        fixtures = read_csv_if_exists(fixtures_path, usecols=fixture_cols)
        stats = read_csv_if_exists(stats_path, usecols=stats_cols)
        if fixtures.empty or stats.empty:
            continue
        merged = stats.merge(fixtures, on="fixture_id", how="left")
        merged["league_tag"] = league_tag
        merged["season_tag"] = season_tag
        merged["source_team_stats_file"] = stats_path.name
        frames.append(merged)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["kickoff_ts_utc"] = pd.to_datetime(out["kickoff_ts_utc"], errors="coerce", utc=True)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    for col in [
        "fixture_id",
        "team_id",
        "home_team_id",
        "away_team_id",
        "is_home",
        "goals_for",
        "goals_against",
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "shots_outside_box",
        "blocked_shots",
        "possession_pct",
        "passes_total",
        "passes_accurate",
        "corners_for",
        "fouls_for",
        "yellow_cards",
        "red_cards",
    ]:
        if col in out.columns:
            out[col] = num(out[col])
    out["team_side"] = np.where(out["is_home"].fillna(0).astype(int).eq(1), "HOME", "AWAY")
    out["opponent_team_id"] = np.where(out["team_side"].eq("HOME"), out["away_team_id"], out["home_team_id"])
    out["opponent_team_name"] = np.where(out["team_side"].eq("HOME"), out["away_team_name"], out["home_team_name"])
    out["cards_total"] = out["yellow_cards"].fillna(0) + (2 * out["red_cards"].fillna(0))

    opp_cols = [
        "fixture_id",
        "team_id",
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "corners_for",
        "fouls_for",
    ]
    opp = out[opp_cols].rename(
        columns={
            "team_id": "opponent_team_id",
            "shots_total": "shots_allowed",
            "shots_on_goal": "sot_allowed",
            "shots_inside_box": "box_shots_allowed",
            "corners_for": "corners_allowed",
            "fouls_for": "fouls_allowed",
        }
    )
    out = out.merge(opp, on=["fixture_id", "opponent_team_id"], how="left")
    out["team_name_norm"] = out["team_name"].map(norm_text)
    out = out.dropna(subset=["fixture_id", "team_id", "kickoff_ts_utc"]).copy()
    return out.sort_values(["kickoff_ts_utc", "fixture_id", "team_side"]).reset_index(drop=True)


def build_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    history: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    output_rows: list[dict[str, Any]] = []
    for fixture_id, fixture_rows in rows.groupby("fixture_id", sort=False):
        fixture_rows = fixture_rows.sort_values("team_side", ascending=False)
        current_records = []
        for _, team_row in fixture_rows.iterrows():
            key = (str(team_row["league_tag"]), int(team_row["team_id"]))
            records = history.get(key, [])
            opponent_key = (
                str(team_row["league_tag"]),
                int(team_row["opponent_team_id"]) if pd.notna(team_row["opponent_team_id"]) else -1,
            )
            opponent_records = history.get(opponent_key, [])
            overall = rolling_summary(records)
            opponent_overall = rolling_summary(opponent_records)
            home = rolling_summary(records, "HOME")
            away = rolling_summary(records, "AWAY")
            prefix_side = home if team_row["team_side"] == "HOME" else away
            rec = {
                "fixture_id": int(fixture_id),
                "fixture_key": team_row.get("fixture_key"),
                "league": team_row.get("league"),
                "league_tag": team_row.get("league_tag"),
                "season": team_row.get("season"),
                "season_tag": team_row.get("season_tag"),
                "match_date": team_row.get("match_date"),
                "kickoff_ts_utc": team_row.get("kickoff_ts_utc"),
                "team_id": int(team_row["team_id"]),
                "team_name": team_row.get("team_name"),
                "team_side": team_row.get("team_side"),
                "opponent_team_id": int(team_row["opponent_team_id"]) if pd.notna(team_row["opponent_team_id"]) else np.nan,
                "opponent_team_name": team_row.get("opponent_team_name"),
                "history_matches": overall["history_matches"],
                "opponent_history_matches": opponent_overall["history_matches"],
                "same_side_history_matches": prefix_side["history_matches"],
                "actual_shots_total": team_row.get("shots_total"),
                "actual_shots_on_goal": team_row.get("shots_on_goal"),
                "actual_corners_for": team_row.get("corners_for"),
                "actual_opponent_saves_proxy": team_row.get("shots_on_goal"),
            }
            for n in ROLLING_WINDOWS:
                rec.update(
                    {
                        f"team_shots_for_l{n}": overall[f"shots_total_l{n}"],
                        f"team_sot_for_l{n}": overall[f"shots_on_goal_l{n}"],
                        f"team_box_shots_for_l{n}": overall[f"shots_inside_box_l{n}"],
                        f"team_blocked_shots_l{n}": overall[f"blocked_shots_l{n}"],
                        f"team_corners_for_l{n}": overall[f"corners_for_l{n}"],
                        f"team_possession_l{n}": overall[f"possession_pct_l{n}"],
                        f"team_shot_accuracy_l{n}": overall[f"shot_accuracy_l{n}"],
                        f"team_high_shot_volume_rate_l{n}": overall[f"high_shot_volume_rate_l{n}"],
                        f"team_high_sot_volume_rate_l{n}": overall[f"high_sot_volume_rate_l{n}"],
                        f"team_corner_5plus_rate_l{n}": overall[f"corner_5plus_rate_l{n}"],
                        f"team_shots_allowed_l{n}": overall[f"shots_allowed_l{n}"],
                        f"team_sot_allowed_l{n}": overall[f"sot_allowed_l{n}"],
                        f"team_box_shots_allowed_l{n}": overall[f"box_shots_allowed_l{n}"],
                        f"team_corners_allowed_l{n}": overall[f"corners_allowed_l{n}"],
                        f"opponent_shots_allowed_l{n}": opponent_overall[f"shots_allowed_l{n}"],
                        f"opponent_sot_allowed_l{n}": opponent_overall[f"sot_allowed_l{n}"],
                        f"opponent_box_shots_allowed_l{n}": opponent_overall[f"box_shots_allowed_l{n}"],
                        f"opponent_corners_allowed_l{n}": opponent_overall[f"corners_allowed_l{n}"],
                        f"same_side_team_shots_for_l{n}": prefix_side[f"shots_total_l{n}"],
                        f"same_side_team_sot_for_l{n}": prefix_side[f"shots_on_goal_l{n}"],
                        f"same_side_team_corners_for_l{n}": prefix_side[f"corners_for_l{n}"],
                    }
                )
            output_rows.append(rec)
            current_records.append(team_row.to_dict())
        for row in current_records:
            key = (str(row["league_tag"]), int(row["team_id"]))
            history[key].append(row)
    profiles = pd.DataFrame(output_rows)
    profiles["profile_ready"] = profiles["history_matches"].ge(5) & profiles["opponent_history_matches"].ge(5)
    profiles["team_expected_shots"] = (
        0.45 * profiles["team_shots_for_l5"]
        + 0.25 * profiles["team_shots_for_l10"]
        + 0.20 * profiles["opponent_shots_allowed_l5"]
        + 0.10 * profiles["opponent_shots_allowed_l10"]
    )
    profiles["team_expected_sot"] = (
        0.45 * profiles["team_sot_for_l5"]
        + 0.25 * profiles["team_sot_for_l10"]
        + 0.20 * profiles["opponent_sot_allowed_l5"]
        + 0.10 * profiles["opponent_sot_allowed_l10"]
    )
    profiles["team_expected_box_shots"] = (
        0.50 * profiles["team_box_shots_for_l5"]
        + 0.25 * profiles["team_box_shots_for_l10"]
        + 0.25 * profiles["opponent_box_shots_allowed_l5"]
    )
    profiles["team_expected_corners"] = (
        0.50 * profiles["team_corners_for_l5"]
        + 0.25 * profiles["team_corners_for_l10"]
        + 0.25 * profiles["opponent_corners_allowed_l5"]
    )
    profiles["keeper_save_pressure_score"] = profiles["team_expected_sot"]
    profiles["attack_volume_score"] = (
        profiles["team_expected_shots"].rank(pct=True)
        + profiles["team_expected_sot"].rank(pct=True)
        + profiles["team_expected_box_shots"].rank(pct=True)
    ) / 3.0
    return profiles


def add_league_relative_labels(profiles: pd.DataFrame) -> pd.DataFrame:
    out = profiles.copy()
    for col in [
        "team_expected_shots",
        "team_expected_sot",
        "team_expected_box_shots",
        "team_expected_corners",
        "keeper_save_pressure_score",
    ]:
        out[f"{col}_league_pct"] = out.groupby("league_tag")[col].rank(pct=True)

    ready = out["profile_ready"].fillna(False)
    out["team_shots_profile_labels"] = ""
    labels = [
        (
            "TEAM_SHOTS_CORE",
            ready
            & out["team_expected_shots_league_pct"].ge(0.75)
            & out["team_shots_for_l5"].ge(out.groupby("league_tag")["team_shots_for_l5"].transform("quantile", 0.70)),
        ),
        (
            "TEAM_SOT_CORE",
            ready
            & out["team_expected_sot_league_pct"].ge(0.75)
            & out["team_sot_for_l5"].ge(out.groupby("league_tag")["team_sot_for_l5"].transform("quantile", 0.70)),
        ),
        (
            "CORNER_PRESSURE_WATCH",
            ready
            & out["team_expected_corners_league_pct"].ge(0.75)
            & out["team_corner_5plus_rate_l5"].ge(0.40),
        ),
        (
            "KEEPER_SAVE_WATCH",
            ready
            & out["keeper_save_pressure_score_league_pct"].ge(0.80)
            & out["team_high_sot_volume_rate_l5"].ge(0.40),
        ),
        (
            "ATTACK_VOLUME_WATCH",
            ready
            & out["attack_volume_score"].ge(0.75)
            & out["team_high_shot_volume_rate_l5"].ge(0.40),
        ),
    ]
    for label, mask in labels:
        out.loc[mask, "team_shots_profile_labels"] = np.where(
            out.loc[mask, "team_shots_profile_labels"].eq(""),
            label,
            out.loc[mask, "team_shots_profile_labels"] + "|" + label,
        )
    out["team_shots_profile_mode"] = np.where(
        out["team_shots_profile_labels"].ne(""),
        "RESEARCH_WATCH_LABELS",
        np.where(out["profile_ready"], "PROFILE_ONLY", "INSUFFICIENT_HISTORY"),
    )
    return out


def label_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labeled = profiles[profiles["team_shots_profile_labels"].fillna("").ne("")].copy()
    for _, row in labeled.iterrows():
        for label in str(row["team_shots_profile_labels"]).split("|"):
            if not label:
                continue
            rows.append(
                {
                    "league": row.get("league"),
                    "league_tag": row.get("league_tag"),
                    "label": label,
                    "team_name": row.get("team_name"),
                    "fixture_key": row.get("fixture_key"),
                    "team_expected_shots": row.get("team_expected_shots"),
                    "team_expected_sot": row.get("team_expected_sot"),
                    "team_expected_corners": row.get("team_expected_corners"),
                    "actual_shots_total": row.get("actual_shots_total"),
                    "actual_shots_on_goal": row.get("actual_shots_on_goal"),
                    "actual_corners_for": row.get("actual_corners_for"),
                }
            )
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    grouped = (
        exploded.groupby(["league", "league_tag", "label"], dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            fixtures=("fixture_key", "nunique"),
            teams=("team_name", "nunique"),
            mean_expected_shots=("team_expected_shots", "mean"),
            mean_expected_sot=("team_expected_sot", "mean"),
            mean_expected_corners=("team_expected_corners", "mean"),
            actual_shots_10plus_rate=("actual_shots_total", lambda s: float(num(s).ge(10).mean())),
            actual_shots_14plus_rate=("actual_shots_total", lambda s: float(num(s).ge(14).mean())),
            actual_sot_4plus_rate=("actual_shots_on_goal", lambda s: float(num(s).ge(4).mean())),
            actual_sot_5plus_rate=("actual_shots_on_goal", lambda s: float(num(s).ge(5).mean())),
            actual_corners_5plus_rate=("actual_corners_for", lambda s: float(num(s).ge(5).mean())),
        )
        .reset_index()
        .sort_values(["label", "rows"], ascending=[True, False])
    )
    return grouped


def fixture_view(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    home = profiles[profiles["team_side"].eq("HOME")].copy()
    away = profiles[profiles["team_side"].eq("AWAY")].copy()
    keep = [
        "fixture_id",
        "fixture_key",
        "league",
        "league_tag",
        "season",
        "match_date",
        "team_name",
        "opponent_team_name",
        "team_expected_shots",
        "team_expected_sot",
        "team_expected_corners",
        "keeper_save_pressure_score",
        "team_shots_profile_labels",
        "team_shots_profile_mode",
    ]
    home = home[keep].rename(columns={col: f"home_{col}" for col in keep if col not in {"fixture_id", "fixture_key", "league", "league_tag", "season", "match_date"}})
    away = away[keep].rename(columns={col: f"away_{col}" for col in keep if col not in {"fixture_id", "fixture_key", "league", "league_tag", "season", "match_date"}})
    merged = home.merge(away, on=["fixture_id", "fixture_key", "league", "league_tag", "season", "match_date"], how="outer")
    merged["match_expected_shots"] = merged["home_team_expected_shots"] + merged["away_team_expected_shots"]
    merged["match_expected_sot"] = merged["home_team_expected_sot"] + merged["away_team_expected_sot"]
    merged["match_expected_corners"] = merged["home_team_expected_corners"] + merged["away_team_expected_corners"]
    return merged


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 4)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, profiles: pd.DataFrame, summary: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    label_rows = int(profiles["team_shots_profile_labels"].fillna("").ne("").sum()) if not profiles.empty else 0
    lines = [
        "# Team Shots Profile",
        "",
        "Research-only API-context board for team shot volume, SOT pressure, corner pressure, and keeper-save pressure.",
        "",
        "## Safety",
        "- No production deploy tiers, slip routing, or protected goal-model gates are changed.",
        "- Labels are watch/intelligence signals only.",
        "- This board is designed to support player-event discovery, not priced odds.",
        "",
        "## Outputs",
        f"- team rows: `{len(profiles)}`",
        f"- fixtures: `{profiles['fixture_key'].nunique() if not profiles.empty and 'fixture_key' in profiles.columns else 0}`",
        f"- profile-ready rows: `{int(profiles['profile_ready'].sum()) if 'profile_ready' in profiles.columns else 0}`",
        f"- labeled watch rows: `{label_rows}`",
        f"- fixture-view rows: `{len(fixtures)}`",
        "",
        "## Watch Label Summary",
        markdown_table(summary),
        "",
        "## How To Use",
        "- `TEAM_SHOTS_CORE`: candidate support for player shots and attacking volume.",
        "- `TEAM_SOT_CORE`: candidate support for player SOT and keeper-save pressure.",
        "- `CORNER_PRESSURE_WATCH`: candidate support for team corners and wide/territory pressure.",
        "- `KEEPER_SAVE_WATCH`: candidate support for opposing keeper-save intelligence.",
        "- `ATTACK_VOLUME_WATCH`: broad attacking pressure context for player-event boards.",
    ]
    (outdir / "TEAM_SHOTS_PROFILE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=NORMALIZED_DIR)
    parser.add_argument("--leagues", help="Comma-separated league tags, e.g. England_Premier_League,Spain_La_Liga")
    parser.add_argument("--seasons", help="Comma-separated seasons, e.g. 2022,2023,2024")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    leagues = parse_csv_arg(args.leagues)
    seasons = parse_int_arg(args.seasons)

    rows = load_team_fixture_rows(args.normalized_dir, leagues, seasons)
    if rows.empty:
        raise SystemExit("No normalized fixture/team-stat rows found for requested filters.")

    profiles = add_league_relative_labels(build_profiles(rows))
    fixtures = fixture_view(profiles)
    summary = label_summary(profiles)

    profiles.to_csv(args.outdir / "TEAM_SHOTS_PROFILE_ROWS.csv", index=False)
    fixtures.to_csv(args.outdir / "TEAM_SHOTS_PROFILE_FIXTURE_VIEW.csv", index=False)
    summary.to_csv(args.outdir / "TEAM_SHOTS_PROFILE_LABEL_SUMMARY.csv", index=False)
    write_report(args.outdir, profiles, summary, fixtures)
    print(
        "WROTE TEAM_SHOTS_PROFILE "
        f"team_rows={len(profiles)} fixtures={profiles['fixture_key'].nunique()} "
        f"labels={profiles['team_shots_profile_labels'].fillna('').ne('').sum()} "
        f"outdir={args.outdir}"
    )


if __name__ == "__main__":
    main()
