#!/usr/bin/env python3
"""
Build leak-safe opponent attack allowance features for player-event research.

For each player fixture row, this computes what the upcoming opponent had
allowed to similar attacking roles before the fixture.

Research-only. No production deploy artifacts or priced prop probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
FEATURES_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_opponent_attack_allowance_features"
DEFAULT_LEAGUES = ("England_Premier_League", "Spain_La_Liga")
DEFAULT_SEASONS = (2022, 2023, 2024)

ATTACK_ROLE_MAP = {
    "Central striker": "central_striker",
    "Wide forward": "wide_forward",
    "Wide midfielder / winger": "wide_midfielder_winger",
}
ROLE_GROUPS = ("central_striker", "wide_forward", "wide_midfielder_winger", "attacker_any")


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_fixture_inputs(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = ["fixture_key", "match_date", "team_name", "player_name", "tactical_role", "position_group"]
    for league in leagues:
        for season in seasons:
            path = FEATURES_DIR / f"player_events_fixture_input__{league}__{season}.csv"
            df = read_csv_if_exists(path)
            if df.empty:
                continue
            available = [c for c in usecols if c in df.columns]
            df = df[available].copy()
            df["league_tag"] = league
            df["season_tag"] = season
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
            for col in ["team_id", "player_id", "home_team_id", "away_team_id"]:
                merged[col] = num(merged[col]).astype("Int64")
            merged["team_name"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["home_team_name"],
                merged["away_team_name"],
            )
            merged["opponent_team_id"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["away_team_id"],
                merged["home_team_id"],
            )
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["shots_total", "shots_on_target", "minutes"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = num(out[col]).fillna(0.0)
    return out.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)


def role_group(tactical_role: Any, position_group: Any) -> str:
    role = str(tactical_role or "")
    if role in ATTACK_ROLE_MAP:
        return ATTACK_ROLE_MAP[role]
    pos = str(position_group or "")
    if pos == "Forward":
        return "central_striker"
    return "other"


def attach_roles(actuals: pd.DataFrame, fixture_inputs: pd.DataFrame) -> pd.DataFrame:
    if fixture_inputs.empty:
        actuals["tactical_role"] = ""
        actuals["position_group"] = ""
        actuals["attack_role_group"] = "other"
        return actuals
    role_cols = ["fixture_key", "team_name", "player_name", "league_tag", "season_tag", "tactical_role", "position_group"]
    roles = fixture_inputs[[c for c in role_cols if c in fixture_inputs.columns]].drop_duplicates(
        ["fixture_key", "team_name", "player_name", "league_tag", "season_tag"]
    )
    joined = actuals.merge(
        roles,
        on=["fixture_key", "team_name", "player_name", "league_tag", "season_tag"],
        how="left",
    )
    joined["attack_role_group"] = [
        role_group(role, pos) for role, pos in zip(joined.get("tactical_role", ""), joined.get("position_group", ""))
    ]
    return joined


def aggregate_allowed_by_fixture(players: pd.DataFrame) -> pd.DataFrame:
    attack = players[players["attack_role_group"].isin(ROLE_GROUPS)].copy()
    if attack.empty:
        return pd.DataFrame()
    attack["shot_ge1"] = attack["shots_total"].ge(1).astype(float)
    attack["shot_ge2"] = attack["shots_total"].ge(2).astype(float)
    attack["sot_ge1"] = attack["shots_on_target"].ge(1).astype(float)
    attack["sot_ge2"] = attack["shots_on_target"].ge(2).astype(float)
    rows = []
    group_cols = [
        "league_tag",
        "season_tag",
        "fixture_id",
        "fixture_key",
        "match_date",
        "kickoff_ts_utc",
        "opponent_team_id",
        "attack_role_group",
    ]
    for key, group in attack.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, key))
        rows.append(
            {
                **rec,
                "players": float(len(group)),
                "shots": float(group["shots_total"].sum()),
                "sot": float(group["shots_on_target"].sum()),
                "shot_ge1_players": float(group["shot_ge1"].sum()),
                "shot_ge2_players": float(group["shot_ge2"].sum()),
                "sot_ge1_players": float(group["sot_ge1"].sum()),
                "sot_ge2_players": float(group["sot_ge2"].sum()),
            }
        )
    any_rows = []
    any_group_cols = [
        "league_tag",
        "season_tag",
        "fixture_id",
        "fixture_key",
        "match_date",
        "kickoff_ts_utc",
        "opponent_team_id",
    ]
    for key, group in attack.groupby(any_group_cols, dropna=False):
        rec = dict(zip(any_group_cols, key))
        any_rows.append(
            {
                **rec,
                "attack_role_group": "attacker_any",
                "players": float(len(group)),
                "shots": float(group["shots_total"].sum()),
                "sot": float(group["shots_on_target"].sum()),
                "shot_ge1_players": float(group["shot_ge1"].sum()),
                "shot_ge2_players": float(group["shot_ge2"].sum()),
                "sot_ge1_players": float(group["sot_ge1"].sum()),
                "sot_ge2_players": float(group["sot_ge2"].sum()),
            }
        )
    return pd.DataFrame(rows + any_rows).rename(columns={"opponent_team_id": "defending_team_id"})


def summarize_history(records: list[dict[str, Any]], n: int, prefix: str) -> dict[str, float]:
    sample = records[-n:]
    matches = len(sample)
    players = sum(float(r.get("players", 0.0) or 0.0) for r in sample)
    shots = sum(float(r.get("shots", 0.0) or 0.0) for r in sample)
    sot = sum(float(r.get("sot", 0.0) or 0.0) for r in sample)
    shot_ge1 = sum(float(r.get("shot_ge1_players", 0.0) or 0.0) for r in sample)
    shot_ge2 = sum(float(r.get("shot_ge2_players", 0.0) or 0.0) for r in sample)
    sot_ge1 = sum(float(r.get("sot_ge1_players", 0.0) or 0.0) for r in sample)
    sot_ge2 = sum(float(r.get("sot_ge2_players", 0.0) or 0.0) for r in sample)
    return {
        f"{prefix}_matches_l{n}": float(matches),
        f"{prefix}_players_l{n}": float(players),
        f"{prefix}_shots_l{n}": float(shots),
        f"{prefix}_sot_l{n}": float(sot),
        f"{prefix}_shots_per_match_l{n}": safe_div(shots, matches),
        f"{prefix}_sot_per_match_l{n}": safe_div(sot, matches),
        f"{prefix}_shots_per_player_l{n}": safe_div(shots, players),
        f"{prefix}_sot_per_player_l{n}": safe_div(sot, players),
        f"{prefix}_player_shot_ge1_rate_l{n}": safe_div(shot_ge1, players),
        f"{prefix}_player_shot_ge2_rate_l{n}": safe_div(shot_ge2, players),
        f"{prefix}_player_sot_ge1_rate_l{n}": safe_div(sot_ge1, players),
        f"{prefix}_player_sot_ge2_rate_l{n}": safe_div(sot_ge2, players),
    }


def build_features(players: pd.DataFrame, allowed: pd.DataFrame) -> pd.DataFrame:
    history: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    features = []
    fixtures = players[
        [
            "league_tag",
            "season_tag",
            "fixture_id",
            "fixture_key",
            "match_date",
            "kickoff_ts_utc",
            "team_id",
            "opponent_team_id",
            "player_id",
            "player_name",
            "team_name",
            "attack_role_group",
            "tactical_role",
        ]
    ].drop_duplicates(["fixture_id", "team_id", "player_id", "league_tag", "season_tag"])
    allowed_by_fixture = {fixture_id: group for fixture_id, group in allowed.groupby("fixture_id", dropna=False)}

    for fixture_id, fixture_players in fixtures.groupby("fixture_id", sort=False):
        for _, row in fixture_players.iterrows():
            league = str(row["league_tag"])
            opponent_team_id = int(row["opponent_team_id"])
            role = str(row["attack_role_group"])
            role_prefix = f"opp_attack_allowed_role"
            any_prefix = f"opp_attack_allowed_attacker_any"
            role_hist = history.get((league, opponent_team_id, role), [])
            any_hist = history.get((league, opponent_team_id, "attacker_any"), [])
            out = {
                "fixture_id": int(row["fixture_id"]),
                "fixture_key": row["fixture_key"],
                "league_tag": row["league_tag"],
                "season_tag": int(row["season_tag"]),
                "team_id": int(row["team_id"]),
                "opponent_team_id": opponent_team_id,
                "player_id": int(row["player_id"]),
                "player_name": row["player_name"],
                "team_name": row["team_name"],
                "attack_role_group": role,
                "tactical_role": row.get("tactical_role", ""),
                "opp_attack_allowed_role_source_matches": float(len(role_hist)),
                "opp_attack_allowed_attacker_any_source_matches": float(len(any_hist)),
            }
            for n in (5, 10):
                out.update(summarize_history(role_hist, n, role_prefix))
                out.update(summarize_history(any_hist, n, any_prefix))
            features.append(out)

        current_allowed = allowed_by_fixture.get(fixture_id)
        if current_allowed is not None:
            for _, rec in current_allowed.iterrows():
                key = (str(rec["league_tag"]), int(rec["defending_team_id"]), str(rec["attack_role_group"]))
                history.setdefault(key, []).append(rec.to_dict())
    return pd.DataFrame(features)


def coverage_report(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    return (
        features.groupby(["league_tag", "season_tag", "attack_role_group"], dropna=False)
        .agg(
            rows=("player_id", "size"),
            players=("player_id", "nunique"),
            rows_with_role_l5=("opp_attack_allowed_role_matches_l5", lambda s: int(pd.to_numeric(s, errors="coerce").ge(5).sum())),
            role_l5_rate=("opp_attack_allowed_role_matches_l5", lambda s: float(pd.to_numeric(s, errors="coerce").ge(5).mean())),
            rows_with_any_l5=("opp_attack_allowed_attacker_any_matches_l5", lambda s: int(pd.to_numeric(s, errors="coerce").ge(5).sum())),
            any_l5_rate=("opp_attack_allowed_attacker_any_matches_l5", lambda s: float(pd.to_numeric(s, errors="coerce").ge(5).mean())),
        )
        .reset_index()
        .sort_values(["league_tag", "season_tag", "attack_role_group"])
    )


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
                value = round(value, 6)
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, features: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# Player Event Opponent Attack Allowance Features",
        "",
        "Research-only rolling opponent allowance table for shots/SOT markets.",
        "",
        "## Safety",
        "- Uses only matches before the current fixture.",
        "- No production model artifacts or deploy routing changed.",
        "- No priced prop probabilities created.",
        "",
        "## Output",
        f"- rows: `{len(features)}`",
        f"- players: `{features['player_id'].nunique() if not features.empty else 0}`",
        "",
        "## Coverage",
        markdown_table(coverage),
        "",
        "## Feature Families",
        "- role-matched opponent allowance: shots/SOT allowed to the player's attack role",
        "- attacker-any opponent allowance: shots/SOT allowed to central strikers, wide forwards, and wide midfielders/wingers combined",
        "- windows: L5 and L10 previous opponent matches",
    ]
    (outdir / "PLAYER_EVENT_OPPONENT_ATTACK_ALLOWANCE_FEATURES.md").write_text("\n".join(lines) + "\n")


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

    inputs = load_fixture_inputs(leagues, seasons)
    actuals = load_actuals(leagues, seasons)
    players = attach_roles(actuals, inputs)
    allowed = aggregate_allowed_by_fixture(players)
    features = build_features(players, allowed)
    coverage = coverage_report(features)

    features.to_csv(args.outdir / "player_event_opponent_attack_allowance_features.csv", index=False)
    coverage.to_csv(args.outdir / "player_event_opponent_attack_allowance_coverage.csv", index=False)
    write_report(args.outdir, features, coverage)

    print(f"WROTE {args.outdir}")
    print(f"rows={len(features)} players={features['player_id'].nunique() if not features.empty else 0}")
    print(coverage.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
