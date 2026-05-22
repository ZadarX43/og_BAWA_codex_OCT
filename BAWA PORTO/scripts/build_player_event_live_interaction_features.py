#!/usr/bin/env python3
"""Project player-event interaction features onto fixture-input rows.

Research-only live bridge. Historical match_player_stats are used as the past,
and current player_events_fixture_input rows are the target rows. Outputs are
compatible with build_player_event_live_feature_join.py:

- player_attacker_recent_form_live_features.csv
- player_event_opponent_attack_allowance_live_features.csv

No priced player-prop odds, deploy routing, or production artifacts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name

NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
PLAYER_EVENTS_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_live_interaction_features"

ATTACK_ROLE_MAP = {
    "Central striker": "central_striker",
    "Wide forward": "wide_forward",
    "Wide midfielder / winger": "wide_midfielder_winger",
}
ROLE_GROUPS = {"central_striker", "wide_forward", "wide_midfielder_winger"}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_team(value: Any, league_tag: Any = None) -> str:
    return norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"player_events_fixture_input__(.+)__(\d{4})\.csv$", path.name)
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


def role_group(tactical_role: Any, position_group: Any) -> str:
    role = str(tactical_role or "")
    if role in ATTACK_ROLE_MAP:
        return ATTACK_ROLE_MAP[role]
    if str(position_group or "") == "Forward":
        return "central_striker"
    return "other"


def load_target_rows(input_dir: Path, leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "player_team_side",
        "position_group",
        "tactical_role",
    ]
    for path in sorted(input_dir.glob("player_events_fixture_input__*.csv")):
        tag = parse_tag(path)
        if tag is None:
            continue
        league_tag, season_tag = tag
        if leagues and league_tag not in leagues:
            continue
        if seasons and season_tag not in seasons:
            continue
        df = read_csv_if_exists(path, usecols=usecols)
        if df.empty:
            continue
        df["league_tag"] = league_tag
        df["season_tag"] = season_tag
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["team_name_norm"] = [norm_team(team, tag) for team, tag in zip(out["team_name"], out["league_tag"])]
    out["home_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["home_team_name"], out["league_tag"])]
    out["away_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["away_team_name"], out["league_tag"])]
    out["opponent_team_name"] = np.where(
        out["team_name_norm"].eq(out["home_team_norm"]),
        out["away_team_name"],
        out["home_team_name"],
    )
    out["opponent_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["opponent_team_name"], out["league_tag"])]
    out["attack_role_group"] = [
        role_group(role, pos) for role, pos in zip(out.get("tactical_role", ""), out.get("position_group", ""))
    ]
    return out.dropna(subset=["fixture_key", "match_date", "team_name", "player_name"]).copy()


def load_historical_actuals(leagues: set[str], seasons: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    stats_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "goals",
        "assists",
        "passes_key",
        "fouls_drawn",
        "fouls_committed",
    ]
    fixture_cols = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "kickoff_ts_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
    ]
    for league_tag in sorted(leagues):
        for season_tag in sorted(seasons):
            fixtures = read_csv_if_exists(
                NORMALIZED_DIR / f"fixtures_master__{league_tag}__{season_tag}.csv",
                usecols=fixture_cols,
            )
            stats = read_csv_if_exists(
                NORMALIZED_DIR / f"match_player_stats__{league_tag}__{season_tag}.csv",
                usecols=stats_cols,
            )
            if fixtures.empty or stats.empty:
                continue
            merged = stats.merge(fixtures, on="fixture_id", how="left")
            for col in ["team_id", "home_team_id", "away_team_id"]:
                merged[col] = num(merged[col]).astype("Int64")
            merged["team_name"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["home_team_name"],
                merged["away_team_name"],
            )
            merged["player_team_side"] = np.where(merged["team_id"].eq(merged["home_team_id"]), "HOME", "AWAY")
            merged["opponent_team_name"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["away_team_name"],
                merged["home_team_name"],
            )
            merged["league_tag"] = league_tag
            merged["season_tag"] = season_tag
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["kickoff_ts_utc"] = pd.to_datetime(out["kickoff_ts_utc"], errors="coerce", utc=True)
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["team_name_norm"] = [norm_team(team, tag) for team, tag in zip(out["team_name"], out["league_tag"])]
    out["opponent_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["opponent_team_name"], out["league_tag"])]
    for col in [
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "goals",
        "assists",
        "passes_key",
        "fouls_drawn",
        "fouls_committed",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = num(out[col]).fillna(0.0)
    return out.sort_values(["match_date", "fixture_id", "team_name_norm", "player_name_norm"]).reset_index(drop=True)


def load_historical_roles(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    role_cols = [
        "fixture_key",
        "team_name",
        "player_name",
        "league_tag",
        "season_tag",
        "tactical_role",
        "position_group",
    ]
    roles = targets[[c for c in role_cols if c in targets.columns]].drop_duplicates(
        ["fixture_key", "team_name", "player_name", "league_tag", "season_tag"]
    )
    joined = history.merge(roles, on=["fixture_key", "team_name", "player_name", "league_tag", "season_tag"], how="left")
    if "tactical_role" not in joined.columns:
        joined["tactical_role"] = ""
    if "position_group" not in joined.columns:
        joined["position_group"] = ""
    target_roles = targets[
        [col for col in ["league_tag", "player_name_norm", "tactical_role", "position_group"] if col in targets.columns]
    ].dropna(subset=["player_name_norm"]).drop_duplicates(["league_tag", "player_name_norm"], keep="last")
    if not target_roles.empty:
        target_roles = target_roles.rename(
            columns={
                "tactical_role": "_target_tactical_role",
                "position_group": "_target_position_group",
            }
        )
        joined = joined.merge(target_roles, on=["league_tag", "player_name_norm"], how="left")
        joined["tactical_role"] = joined["tactical_role"].where(
            joined["tactical_role"].fillna("").astype(str).ne(""),
            joined.get("_target_tactical_role", ""),
        )
        joined["position_group"] = joined["position_group"].where(
            joined["position_group"].fillna("").astype(str).ne(""),
            joined.get("_target_position_group", ""),
        )
        joined = joined.drop(columns=["_target_tactical_role", "_target_position_group"], errors="ignore")
    joined["attack_role_group"] = [
        role_group(role, pos) for role, pos in zip(joined.get("tactical_role", ""), joined.get("position_group", ""))
    ]
    return joined


def summarize_player_history(prev: pd.DataFrame, n: int, side: str | None = None) -> dict[str, float]:
    rows = prev.sort_values("match_date")
    if side:
        rows = rows[rows["player_team_side"].eq(side)]
    rows = rows.tail(n)
    apps = len(rows)
    minutes = float(rows["minutes"].sum()) if apps else 0.0
    starts = float(rows["started_flag"].sum()) if apps else 0.0
    shots = float(rows["shots_total"].sum()) if apps else 0.0
    sot = float(rows["shots_on_target"].sum()) if apps else 0.0
    goals = float(rows["goals"].sum()) if apps else 0.0
    assists = float(rows["assists"].sum()) if apps else 0.0
    key_passes = float(rows["passes_key"].sum()) if apps else 0.0
    fouls_won = float(rows["fouls_drawn"].sum()) if apps else 0.0
    fouls_committed = float(rows["fouls_committed"].sum()) if apps else 0.0
    prefix = f"attacker_recent_{'home_' if side == 'HOME' else 'away_' if side == 'AWAY' else ''}"
    return {
        f"{prefix}apps_l{n}": float(apps),
        f"{prefix}starts_l{n}": starts,
        f"{prefix}minutes_l{n}": minutes,
        f"{prefix}shots_l{n}": shots,
        f"{prefix}sot_l{n}": sot,
        f"{prefix}goals_l{n}": goals,
        f"{prefix}assists_l{n}": assists,
        f"{prefix}goal_contributions_l{n}": goals + assists,
        f"{prefix}key_passes_l{n}": key_passes,
        f"{prefix}fouls_won_l{n}": fouls_won,
        f"{prefix}fouls_committed_l{n}": fouls_committed,
        f"{prefix}shots_per90_l{n}": safe_div(shots * 90.0, minutes),
        f"{prefix}sot_per90_l{n}": safe_div(sot * 90.0, minutes),
        f"{prefix}goals_per90_l{n}": safe_div(goals * 90.0, minutes),
        f"{prefix}assists_per90_l{n}": safe_div(assists * 90.0, minutes),
        f"{prefix}goal_contributions_per90_l{n}": safe_div((goals + assists) * 90.0, minutes),
        f"{prefix}key_passes_per90_l{n}": safe_div(key_passes * 90.0, minutes),
        f"{prefix}fouls_won_per90_l{n}": safe_div(fouls_won * 90.0, minutes),
        f"{prefix}fouls_committed_per90_l{n}": safe_div(fouls_committed * 90.0, minutes),
        f"{prefix}start_share_l{n}": safe_div(starts, apps),
        f"{prefix}sot_share_l{n}": safe_div(sot, shots),
    }


def build_recent_features(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {key: group for key, group in history.groupby(["league_tag", "player_name_norm"], dropna=False)}
    for _, target in targets.iterrows():
        prev = grouped.get((target["league_tag"], target["player_name_norm"]), pd.DataFrame(columns=history.columns))
        if not prev.empty:
            prev = prev[prev["match_date"].lt(target["match_date"])].copy()
        out: dict[str, Any] = {
            "fixture_key": target["fixture_key"],
            "league_tag": target["league_tag"],
            "season_tag": int(target["season_tag"]),
            "team_name": target["team_name"],
            "player_name": target["player_name"],
            "match_date": target["match_date"],
            "player_team_side": target.get("player_team_side", ""),
            "attacker_recent_history_matches": float(len(prev)),
            "attacker_recent_source_max_date": str(prev["match_date"].max().date()) if not prev.empty and pd.notna(prev["match_date"].max()) else "",
            "attacker_recent_xg_available_flag": 0,
            "attacker_recent_xa_available_flag": 0,
            "attacker_recent_big_chances_available_flag": 0,
            "attacker_recent_touches_box_available_flag": 0,
        }
        for n in (5, 8):
            out.update(summarize_player_history(prev, n))
            out.update(summarize_player_history(prev, n, "HOME"))
            out.update(summarize_player_history(prev, n, "AWAY"))
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate_allowed(history: pd.DataFrame) -> pd.DataFrame:
    attack = history[history["attack_role_group"].isin(ROLE_GROUPS)].copy()
    if attack.empty:
        return pd.DataFrame()
    attack["shot_ge1"] = attack["shots_total"].ge(1).astype(float)
    attack["shot_ge2"] = attack["shots_total"].ge(2).astype(float)
    attack["sot_ge1"] = attack["shots_on_target"].ge(1).astype(float)
    attack["sot_ge2"] = attack["shots_on_target"].ge(2).astype(float)
    attack["fouled_ge1"] = attack["fouls_drawn"].ge(1).astype(float)
    attack["fouled_ge2"] = attack["fouls_drawn"].ge(2).astype(float)
    rows: list[dict[str, Any]] = []
    group_cols = ["league_tag", "match_date", "fixture_key", "opponent_team_norm", "attack_role_group"]
    for key, group in attack.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, key))
        rows.append(
            {
                **rec,
                "players": float(len(group)),
                "shots": float(group["shots_total"].sum()),
                "sot": float(group["shots_on_target"].sum()),
                "fouls_drawn": float(group["fouls_drawn"].sum()),
                "fouls_committed": float(group["fouls_committed"].sum()),
                "shot_ge1_players": float(group["shot_ge1"].sum()),
                "shot_ge2_players": float(group["shot_ge2"].sum()),
                "sot_ge1_players": float(group["sot_ge1"].sum()),
                "sot_ge2_players": float(group["sot_ge2"].sum()),
                "fouled_ge1_players": float(group["fouled_ge1"].sum()),
                "fouled_ge2_players": float(group["fouled_ge2"].sum()),
            }
        )
    any_rows: list[dict[str, Any]] = []
    any_group_cols = ["league_tag", "match_date", "fixture_key", "opponent_team_norm"]
    for key, group in attack.groupby(any_group_cols, dropna=False):
        rec = dict(zip(any_group_cols, key))
        any_rows.append(
            {
                **rec,
                "attack_role_group": "attacker_any",
                "players": float(len(group)),
                "shots": float(group["shots_total"].sum()),
                "sot": float(group["shots_on_target"].sum()),
                "fouls_drawn": float(group["fouls_drawn"].sum()),
                "fouls_committed": float(group["fouls_committed"].sum()),
                "shot_ge1_players": float(group["shot_ge1"].sum()),
                "shot_ge2_players": float(group["shot_ge2"].sum()),
                "sot_ge1_players": float(group["sot_ge1"].sum()),
                "sot_ge2_players": float(group["sot_ge2"].sum()),
                "fouled_ge1_players": float(group["fouled_ge1"].sum()),
                "fouled_ge2_players": float(group["fouled_ge2"].sum()),
            }
        )
    return pd.DataFrame(rows + any_rows).sort_values(["league_tag", "opponent_team_norm", "attack_role_group", "match_date"])


def summarize_allowed(prev: pd.DataFrame, n: int, prefix: str) -> dict[str, float]:
    if prev.empty or "match_date" not in prev.columns:
        return {
            f"{prefix}_matches_l{n}": 0.0,
            f"{prefix}_players_l{n}": 0.0,
            f"{prefix}_shots_l{n}": 0.0,
            f"{prefix}_sot_l{n}": 0.0,
            f"{prefix}_fouls_drawn_l{n}": 0.0,
            f"{prefix}_fouls_committed_l{n}": 0.0,
            f"{prefix}_shots_per_match_l{n}": 0.0,
            f"{prefix}_sot_per_match_l{n}": 0.0,
            f"{prefix}_fouls_drawn_per_match_l{n}": 0.0,
            f"{prefix}_fouls_committed_per_match_l{n}": 0.0,
            f"{prefix}_shots_per_player_l{n}": 0.0,
            f"{prefix}_sot_per_player_l{n}": 0.0,
            f"{prefix}_fouls_drawn_per_player_l{n}": 0.0,
            f"{prefix}_fouls_committed_per_player_l{n}": 0.0,
            f"{prefix}_player_shot_ge1_rate_l{n}": 0.0,
            f"{prefix}_player_shot_ge2_rate_l{n}": 0.0,
            f"{prefix}_player_sot_ge1_rate_l{n}": 0.0,
            f"{prefix}_player_sot_ge2_rate_l{n}": 0.0,
            f"{prefix}_player_fouled_ge1_rate_l{n}": 0.0,
            f"{prefix}_player_fouled_ge2_rate_l{n}": 0.0,
        }
    rows = prev.sort_values("match_date").tail(n)
    matches = len(rows)
    players = float(rows["players"].sum()) if matches else 0.0
    shots = float(rows["shots"].sum()) if matches else 0.0
    sot = float(rows["sot"].sum()) if matches else 0.0
    fouls_drawn = float(rows["fouls_drawn"].sum()) if matches else 0.0
    fouls_committed = float(rows["fouls_committed"].sum()) if matches else 0.0
    shot_ge1 = float(rows["shot_ge1_players"].sum()) if matches else 0.0
    shot_ge2 = float(rows["shot_ge2_players"].sum()) if matches else 0.0
    sot_ge1 = float(rows["sot_ge1_players"].sum()) if matches else 0.0
    sot_ge2 = float(rows["sot_ge2_players"].sum()) if matches else 0.0
    fouled_ge1 = float(rows["fouled_ge1_players"].sum()) if matches else 0.0
    fouled_ge2 = float(rows["fouled_ge2_players"].sum()) if matches else 0.0
    return {
        f"{prefix}_matches_l{n}": float(matches),
        f"{prefix}_players_l{n}": players,
        f"{prefix}_shots_l{n}": shots,
        f"{prefix}_sot_l{n}": sot,
        f"{prefix}_fouls_drawn_l{n}": fouls_drawn,
        f"{prefix}_fouls_committed_l{n}": fouls_committed,
        f"{prefix}_shots_per_match_l{n}": safe_div(shots, matches),
        f"{prefix}_sot_per_match_l{n}": safe_div(sot, matches),
        f"{prefix}_fouls_drawn_per_match_l{n}": safe_div(fouls_drawn, matches),
        f"{prefix}_fouls_committed_per_match_l{n}": safe_div(fouls_committed, matches),
        f"{prefix}_shots_per_player_l{n}": safe_div(shots, players),
        f"{prefix}_sot_per_player_l{n}": safe_div(sot, players),
        f"{prefix}_fouls_drawn_per_player_l{n}": safe_div(fouls_drawn, players),
        f"{prefix}_fouls_committed_per_player_l{n}": safe_div(fouls_committed, players),
        f"{prefix}_player_shot_ge1_rate_l{n}": safe_div(shot_ge1, players),
        f"{prefix}_player_shot_ge2_rate_l{n}": safe_div(shot_ge2, players),
        f"{prefix}_player_sot_ge1_rate_l{n}": safe_div(sot_ge1, players),
        f"{prefix}_player_sot_ge2_rate_l{n}": safe_div(sot_ge2, players),
        f"{prefix}_player_fouled_ge1_rate_l{n}": safe_div(fouled_ge1, players),
        f"{prefix}_player_fouled_ge2_rate_l{n}": safe_div(fouled_ge2, players),
    }


def build_opponent_features(targets: pd.DataFrame, allowed: pd.DataFrame) -> pd.DataFrame:
    required = {"league_tag", "opponent_team_norm", "attack_role_group"}
    if allowed.empty or not required.issubset(allowed.columns):
        rows = []
        for _, target in targets.iterrows():
            role = str(target.get("attack_role_group", "other"))
            out: dict[str, Any] = {
                "fixture_key": target["fixture_key"],
                "league_tag": target["league_tag"],
                "season_tag": int(target["season_tag"]),
                "team_name": target["team_name"],
                "player_name": target["player_name"],
                "match_date": target["match_date"],
                "opponent_team_name": target["opponent_team_name"],
                "attack_role_group": role,
                "tactical_role": target.get("tactical_role", ""),
                "opp_attack_allowed_role_source_matches": 0.0,
                "opp_attack_allowed_attacker_any_source_matches": 0.0,
            }
            for n in (5, 10):
                out.update(summarize_allowed(pd.DataFrame(), n, "opp_attack_allowed_role"))
                out.update(summarize_allowed(pd.DataFrame(), n, "opp_attack_allowed_attacker_any"))
            rows.append(out)
        return pd.DataFrame(rows)
    grouped = {
        key: group for key, group in allowed.groupby(["league_tag", "opponent_team_norm", "attack_role_group"], dropna=False)
    }
    rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        role = str(target.get("attack_role_group", "other"))
        role_hist = grouped.get((target["league_tag"], target["opponent_team_norm"], role), pd.DataFrame())
        any_hist = grouped.get((target["league_tag"], target["opponent_team_norm"], "attacker_any"), pd.DataFrame())
        if not role_hist.empty:
            role_hist = role_hist[role_hist["match_date"].lt(target["match_date"])].copy()
        if not any_hist.empty:
            any_hist = any_hist[any_hist["match_date"].lt(target["match_date"])].copy()
        out: dict[str, Any] = {
            "fixture_key": target["fixture_key"],
            "league_tag": target["league_tag"],
            "season_tag": int(target["season_tag"]),
            "team_name": target["team_name"],
            "player_name": target["player_name"],
            "match_date": target["match_date"],
            "opponent_team_name": target["opponent_team_name"],
            "attack_role_group": role,
            "tactical_role": target.get("tactical_role", ""),
            "opp_attack_allowed_role_source_matches": float(len(role_hist)),
            "opp_attack_allowed_attacker_any_source_matches": float(len(any_hist)),
        }
        for n in (5, 10):
            out.update(summarize_allowed(role_hist, n, "opp_attack_allowed_role"))
            out.update(summarize_allowed(any_hist, n, "opp_attack_allowed_attacker_any"))
        rows.append(out)
    return pd.DataFrame(rows)


def coverage(features: pd.DataFrame, source: str) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    if source == "recent":
        ready = features["attacker_recent_history_matches"].gt(0)
    else:
        ready = features["opp_attack_allowed_attacker_any_source_matches"].gt(0)
    out = features.assign(_ready=ready).groupby(["league_tag", "season_tag"], dropna=False).agg(
        rows=("fixture_key", "size"),
        fixtures=("fixture_key", "nunique"),
        players=("player_name", "nunique"),
        ready_rows=("_ready", "sum"),
        ready_rate=("_ready", "mean"),
    )
    return out.reset_index().sort_values(["league_tag", "season_tag"])


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
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, targets: pd.DataFrame, recent_cov: pd.DataFrame, opponent_cov: pd.DataFrame) -> None:
    lines = [
        "# Player Event Live Interaction Features",
        "",
        "Research-only live projection features for exact player-event interaction labels.",
        "",
        "## Safety",
        "- Uses historical match_player_stats as prior context.",
        "- Emits current fixture-input feature rows only.",
        "- No priced player-prop odds and no deploy routing.",
        "",
        "## Output",
        f"- target rows: `{len(targets)}`",
        f"- target fixtures: `{targets['fixture_key'].nunique() if not targets.empty else 0}`",
        "",
        "## Recent Form Coverage",
        markdown_table(recent_cov),
        "",
        "## Opponent Allowance Coverage",
        markdown_table(opponent_cov),
    ]
    (outdir / "PLAYER_EVENT_LIVE_INTERACTION_FEATURES.md").write_text("\n".join(lines) + "\n")


def parse_csv_set(value: str, cast=str) -> set:
    return {cast(part.strip()) for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PLAYER_EVENTS_DIR)
    parser.add_argument("--leagues", default="", help="Optional comma-separated league tags.")
    parser.add_argument("--target-seasons", default="", help="Optional comma-separated target seasons.")
    parser.add_argument("--history-seasons", default="2022,2023,2024")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    leagues = parse_csv_set(args.leagues, str) if args.leagues else None
    target_seasons = parse_csv_set(args.target_seasons, int) if args.target_seasons else None
    history_seasons = parse_csv_set(args.history_seasons, int)

    args.outdir.mkdir(parents=True, exist_ok=True)
    targets = load_target_rows(args.input_dir, leagues, target_seasons)
    if targets.empty:
        raise SystemExit("No player-event fixture input rows found.")
    history_leagues = set(targets["league_tag"].dropna().astype(str))
    history = load_historical_actuals(history_leagues, history_seasons)
    if history.empty:
        raise SystemExit("No historical player stats found for target leagues.")
    history = load_historical_roles(targets, history)

    recent = build_recent_features(targets, history)
    allowed = aggregate_allowed(history)
    opponent = build_opponent_features(targets, allowed)
    recent_cov = coverage(recent, "recent")
    opponent_cov = coverage(opponent, "opponent")

    recent.to_csv(args.outdir / "player_attacker_recent_form_live_features.csv", index=False)
    opponent.to_csv(args.outdir / "player_event_opponent_attack_allowance_live_features.csv", index=False)
    recent_cov.to_csv(args.outdir / "player_attacker_recent_form_live_coverage.csv", index=False)
    opponent_cov.to_csv(args.outdir / "player_event_opponent_attack_allowance_live_coverage.csv", index=False)
    write_report(args.outdir, targets, recent_cov, opponent_cov)

    print(f"WROTE {args.outdir}")
    print(f"target_rows={len(targets)} target_fixtures={targets['fixture_key'].nunique()}")
    print(f"recent_ready_rows={int(recent['attacker_recent_history_matches'].gt(0).sum())}")
    print(f"opponent_ready_rows={int(opponent['opp_attack_allowed_attacker_any_source_matches'].gt(0).sum())}")


if __name__ == "__main__":
    main()
