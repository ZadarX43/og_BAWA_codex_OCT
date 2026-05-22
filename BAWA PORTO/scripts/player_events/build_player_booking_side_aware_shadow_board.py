#!/usr/bin/env python3
"""Build side-aware player/team booking shadow rows.

Research-only sidecar. This converts the weekend lesson from live grading into
formal, gradable shadow rows:

- dominant/favourite goal shape -> underdog/chasing side card pressure
- balanced high-contact shape -> both-side card watch
- player contact/role/ref/context -> Player Card 0.5+ watch

No priced odds, no deploy routing, no slip generation, and no production
rulebook mutation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name  # noqa: E402
from scripts.build_live_shadow_research_dashboard import markdown_table  # noqa: E402


DEFAULT_ALLMARKETS = ROOT / "predictions_output" / "2026-05-09" / "BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11.csv"
DEFAULT_PLAYER_EVENT_BOARD = (
    ROOT
    / "reports"
    / "2026-05-08"
    / "player_event_exact_interaction_shadow_refresh_2026_05_09"
    / "player_event_live_feature_join"
    / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-10" / "player_booking_side_aware_shadow_board_2026_05_09"

PLAYER_STAGE = "PLAYER_CARD_0_5_SIDE_AWARE_WATCH"
TEAM_STAGE = "TEAM_CARDS_1_5_SIDE_AWARE_WATCH"

OUTPUT_COLUMNS = [
    "shadow_family",
    "shadow_stage",
    "fixture_key",
    "match_date",
    "league",
    "home_team_name",
    "away_team_name",
    "expression",
    "source_market",
    "source_selection",
    "source_deploy_tier",
    "source_tier",
    "combo_product",
    "combo_tier",
    "bookie_od",
    "model_prob",
    "value_edge",
    "value_edge_tier",
    "backtest_hit_rate",
    "backtest_graded",
    "league_stability_bucket",
    "team_stability_bucket",
    "watch_priority",
    "watch_flag",
    "guardrail",
    "reason",
    "player_name",
    "team_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "predicted_hit_rate_pct",
    "confidence_label",
    "lineup_watch_flags",
    "lineup_confirmation_status",
    "expected_minutes",
    "formation_matchup_label",
    "formation_pressure_score",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "fixture_foul_density_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "fixture_attack_pressure_score",
    "referee_name",
    "ref_cards_per_match",
    "interaction_match_mode",
    "interaction_label",
    "fouled_context_cell",
    "fouled_context_cell_label",
    "booking_pressure_side",
    "booking_pressure_mode",
    "booking_context_label",
    "booking_hazard_score",
    "booking_role_cell",
    "booking_side_pressure_score",
    "booking_player_contact_score",
    "booking_fixture_contact_score",
    "booking_recent_fouls_committed_per90",
    "booking_goal_state_signal",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def scalar(row: pd.Series, col: str, default: Any = "") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_team(value: Any, league_tag: Any = "") -> str:
    league = str(league_tag or "").replace(" ", "_")
    return norm_text(normalize_team_name(value, league))


def percentile(series: pd.Series) -> pd.Series:
    values = num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def safe_max(row: pd.Series, cols: list[str]) -> float:
    vals = [pd.to_numeric(row.get(col, np.nan), errors="coerce") for col in cols if col in row.index]
    vals = [float(v) for v in vals if pd.notna(v)]
    return max(vals) if vals else np.nan


def boolish(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip().isin(["1", "true", "yes", "y"])


def role_cell(row: pd.Series) -> str:
    text = f"{row.get('tactical_role', '')} {row.get('position_group', '')}".lower()
    if any(term in text for term in ["holding", "defensive midfielder", "central midfielder", "box-to-box", "destroyer"]):
        return "MIDFIELD_CONTACT"
    if any(term in text for term in ["full-back", "fullback", "wing-back", "wingback", "wide defender"]):
        return "WIDE_DEFENDER_CONTACT"
    if any(term in text for term in ["centre-back", "center-back", "central defender", "cb"]):
        return "CENTRE_BACK_STRICT"
    if any(term in text for term in ["forward", "striker", "winger", "attacker", "pressing"]):
        return "ATTACKER_PRESSER"
    return "OTHER"


def role_score(cell: str) -> float:
    return {
        "MIDFIELD_CONTACT": 0.95,
        "WIDE_DEFENDER_CONTACT": 0.86,
        "CENTRE_BACK_STRICT": 0.74,
        "ATTACKER_PRESSER": 0.46,
        "OTHER": 0.32,
    }.get(cell, 0.32)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def parse_lineup_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"lineups__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def load_confirmed_starters(lineups_root: Path | None) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str, str, str]]]:
    if lineups_root is None:
        return set(), set()
    if not lineups_root.exists():
        raise SystemExit(f"Missing lineups root: {lineups_root}")
    fixture_key_starters: set[tuple[str, str, str]] = set()
    match_starters: set[tuple[str, str, str, str, str]] = set()
    for lineup_path in sorted(lineups_root.glob("lineups__*.csv")):
        parsed = parse_lineup_tag(lineup_path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        fixture_path = lineups_root / f"fixtures_master__{league_tag}__{season_tag}.csv"
        if not fixture_path.exists():
            continue
        lineups = pd.read_csv(lineup_path, low_memory=False)
        fixtures = pd.read_csv(fixture_path, low_memory=False)
        if lineups.empty or fixtures.empty:
            continue
        lineups = lineups.merge(fixtures, on="fixture_id", how="left")
        lineups = lineups[num(lineups.get("is_starting_xi", 0)).fillna(0).eq(1)].copy()
        for _, row in lineups.iterrows():
            fixture_key = str(row.get("fixture_key", "")).strip()
            player = norm_text(row.get("player_name", ""))
            team_id = pd.to_numeric(row.get("team_id", np.nan), errors="coerce")
            home_id = pd.to_numeric(row.get("home_team_id", np.nan), errors="coerce")
            away_id = pd.to_numeric(row.get("away_team_id", np.nan), errors="coerce")
            if pd.notna(team_id) and pd.notna(home_id) and int(team_id) == int(home_id):
                team = norm_team(row.get("home_team_name", ""), league_tag)
            elif pd.notna(team_id) and pd.notna(away_id) and int(team_id) == int(away_id):
                team = norm_team(row.get("away_team_name", ""), league_tag)
            else:
                team = norm_team(row.get("team_name", ""), league_tag)
            match_date = str(row.get("match_date", "")).strip()
            home = norm_team(row.get("home_team_name", ""), league_tag)
            away = norm_team(row.get("away_team_name", ""), league_tag)
            if fixture_key and team and player:
                fixture_key_starters.add((fixture_key, team, player))
            if match_date and home and away and team and player:
                match_starters.add((match_date, home, away, team, player))
    return fixture_key_starters, match_starters


def league_to_tag(value: Any) -> str:
    return str(value or "").strip().replace(" ", "_")


def build_fixture_context(allmarkets: pd.DataFrame) -> pd.DataFrame:
    work = allmarkets.copy()
    work["match_date"] = pd.to_datetime(work.get("match_date"), errors="coerce").dt.date.astype("string")
    numeric_cols = [
        "cs_mass_home_win",
        "cs_mass_away_win",
        "confidence_home",
        "confidence_away",
        "p_home_ge2",
        "p_away_ge2",
        "home_ge2_confidence",
        "away_ge2_confidence",
        "p_meta_ou25",
        "cs_mass_over25",
        "mass_4plus_goals",
        "p_meta_btts",
        "cs_mass_btts_yes",
        "p00_est",
        "ftr_margin",
        "power_diff",
    ]
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = num(work[col])

    records: list[dict[str, Any]] = []
    for fixture_key, group in work.groupby("fixture_key", dropna=False):
        row = group.iloc[0]
        home_win = safe_max(row, ["cs_mass_home_win", "confidence_home"])
        away_win = safe_max(row, ["cs_mass_away_win", "confidence_away"])
        home_ge2 = safe_max(row, ["p_home_ge2", "home_ge2_confidence"])
        away_ge2 = safe_max(row, ["p_away_ge2", "away_ge2_confidence"])
        total_goal_signal = safe_max(row, ["p_meta_ou25", "cs_mass_over25", "mass_4plus_goals"])
        btts_signal = safe_max(row, ["p_meta_btts", "cs_mass_btts_yes"])
        power_diff = float(row.get("power_diff", np.nan)) if pd.notna(row.get("power_diff", np.nan)) else np.nan

        win_gap = (home_win if pd.notna(home_win) else 0.0) - (away_win if pd.notna(away_win) else 0.0)
        ge2_gap = (home_ge2 if pd.notna(home_ge2) else 0.0) - (away_ge2 if pd.notna(away_ge2) else 0.0)
        blended_gap = 0.55 * win_gap + 0.45 * ge2_gap
        if abs(blended_gap) < 0.035 and pd.notna(power_diff):
            blended_gap = float(np.sign(power_diff)) * min(abs(power_diff) / 500.0, 0.20)

        if blended_gap >= 0.075:
            dominance_side = "HOME"
            pressure_side = "AWAY"
            pressure_mode = "SIDE_AWARE_UNDERDOG_CARD_PRESSURE"
        elif blended_gap <= -0.075:
            dominance_side = "AWAY"
            pressure_side = "HOME"
            pressure_mode = "SIDE_AWARE_UNDERDOG_CARD_PRESSURE"
        else:
            dominance_side = "BALANCED"
            pressure_side = "BOTH"
            pressure_mode = "BALANCED_SCORELINE_CARD_PRESSURE"

        records.append(
            {
                "fixture_key": fixture_key,
                "match_date": row.get("match_date"),
                "league": row.get("league"),
                "home_team_name": row.get("home_team_name"),
                "away_team_name": row.get("away_team_name"),
                "home_win_signal": home_win,
                "away_win_signal": away_win,
                "home_ge2_signal": home_ge2,
                "away_ge2_signal": away_ge2,
                "total_goal_signal": total_goal_signal,
                "btts_signal": btts_signal,
                "ftr_margin": row.get("ftr_margin", np.nan),
                "p00_est": row.get("p00_est", np.nan),
                "dominance_side": dominance_side,
                "booking_pressure_side": pressure_side,
                "booking_pressure_mode": pressure_mode,
                "booking_goal_state_signal": abs(blended_gap),
            }
        )
    return pd.DataFrame(records)


def prepare_player_features(features: pd.DataFrame, fixture_context: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    if "fixture_key" not in work.columns:
        return pd.DataFrame()
    work = work[work["fixture_key"].astype(str).isin(set(fixture_context["fixture_key"].astype(str)))].copy()
    if work.empty:
        return work
    work["match_date"] = pd.to_datetime(work.get("match_date"), errors="coerce").dt.date.astype("string")
    for col in [
        "expected_minutes",
        "ref_cards_per_match",
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_wide_duel_score",
        "formation_pressure_score",
        "match_stakes_score",
        "support_score",
        "attacker_recent_fouls_committed_per90_l5",
        "attacker_recent_fouls_committed_per90_l8",
        "opp_attack_allowed_role_fouls_committed_per_player_l5",
        "opp_attack_allowed_role_fouls_committed_per_player_l10",
        "opp_attack_allowed_attacker_any_fouls_committed_per_player_l5",
        "opp_attack_allowed_attacker_any_fouls_committed_per_player_l10",
    ]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = num(work[col])

    if "expected_start_flag" not in work.columns:
        work["expected_start_flag"] = 1
    work["player_team_side"] = work.get(
        "player_team_side_x",
        work.get("player_team_side_y", work.get("player_team_side", "")),
    )
    work["player_name_norm"] = work.get("player_name", pd.Series("", index=work.index)).map(norm_text)
    work["team_name_norm"] = work.get("team_name", pd.Series("", index=work.index)).map(norm_text)
    work["role_cell"] = work.apply(role_cell, axis=1)
    work["_role_score"] = work["role_cell"].map(role_score).fillna(0.32)
    work["_recent_fouls_per90"] = work[
        ["attacker_recent_fouls_committed_per90_l5", "attacker_recent_fouls_committed_per90_l8"]
    ].max(axis=1)
    work["_opp_contact_allowed"] = work[
        [
            "opp_attack_allowed_role_fouls_committed_per_player_l5",
            "opp_attack_allowed_role_fouls_committed_per_player_l10",
            "opp_attack_allowed_attacker_any_fouls_committed_per_player_l5",
            "opp_attack_allowed_attacker_any_fouls_committed_per_player_l10",
        ]
    ].max(axis=1)
    work["_sort_hit_rate"] = num(work.get("predicted_hit_rate_pct", np.nan)).fillna(0)
    work = work.sort_values(
        ["fixture_key", "player_name_norm", "team_name_norm", "_sort_hit_rate", "support_score"],
        ascending=[True, True, True, False, False],
    ).drop_duplicates(["fixture_key", "player_name_norm", "team_name_norm"], keep="first")
    return work.merge(fixture_context, on="fixture_key", how="left", suffixes=("", "_fixture"))


def score_players(
    players: pd.DataFrame,
    confirmed_starters: tuple[set[tuple[str, str, str]], set[tuple[str, str, str, str, str]]] | None = None,
) -> pd.DataFrame:
    if players.empty:
        return players
    out = players.copy()
    fixture_key_starters, match_starters = confirmed_starters or (set(), set())
    if fixture_key_starters or match_starters:
        fixture_keys = [
            (
                str(row.get("fixture_key", "")).strip(),
                norm_team(row.get("team_name", ""), league_to_tag(row.get("league", row.get("competition", "")))),
                str(row.get("player_name_norm", "")).strip(),
            )
            for _, row in out.iterrows()
        ]
        match_keys = [
            (
                str(row.get("match_date", "")).strip(),
                norm_team(row.get("home_team_name", ""), league_to_tag(row.get("league", row.get("competition", "")))),
                norm_team(row.get("away_team_name", ""), league_to_tag(row.get("league", row.get("competition", "")))),
                norm_team(row.get("team_name", ""), league_to_tag(row.get("league", row.get("competition", "")))),
                str(row.get("player_name_norm", "")).strip(),
            )
            for _, row in out.iterrows()
        ]
        out["lineup_confirmation_status"] = [
            "CONFIRMED_STARTER"
            if fixture_key in fixture_key_starters or match_key in match_starters
            else "API_LINEUP_AVAILABLE_NOT_STARTING"
            for fixture_key, match_key in zip(fixture_keys, match_keys)
        ]
    else:
        out["lineup_confirmation_status"] = "LINEUP_NOT_ATTACHED_EXPECTED_START_ONLY"
    for col in ["_recent_fouls_per90", "_opp_contact_allowed", "ref_cards_per_match", "fixture_foul_density_score", "fixture_tackle_density_score", "fixture_wide_duel_score", "formation_pressure_score", "match_stakes_score"]:
        out[f"{col}_pct"] = percentile(out[col])

    pressure_side = out["booking_pressure_side"].fillna("").astype(str)
    player_side = out["player_team_side"].fillna("").astype(str).str.upper()
    out["booking_side_pressure_score"] = np.select(
        [
            pressure_side.eq("BOTH"),
            pressure_side.eq(player_side),
            pressure_side.ne("") & ~pressure_side.eq(player_side),
        ],
        [0.70, 1.00, 0.28],
        default=0.45,
    )
    out["booking_player_contact_score"] = (
        0.50 * out["_recent_fouls_per90_pct"]
        + 0.30 * out["_role_score"]
        + 0.20 * out["_opp_contact_allowed_pct"]
    ).clip(0, 1)
    out["booking_fixture_contact_score"] = (
        0.28 * out["ref_cards_per_match_pct"]
        + 0.24 * out["fixture_foul_density_score_pct"]
        + 0.18 * out["fixture_tackle_density_score_pct"]
        + 0.14 * out["fixture_wide_duel_score_pct"]
        + 0.10 * out["formation_pressure_score_pct"]
        + 0.06 * out["match_stakes_score_pct"]
    ).clip(0, 1)
    out["booking_hazard_score"] = (
        0.32 * out["booking_side_pressure_score"]
        + 0.34 * out["booking_player_contact_score"]
        + 0.24 * out["booking_fixture_contact_score"]
        + 0.10 * num(out["booking_goal_state_signal"]).clip(0, 0.25).fillna(0) / 0.25
    ).clip(0, 1)
    out["booking_context_label"] = np.select(
        [
            out["booking_hazard_score"].ge(0.74) & out["booking_side_pressure_score"].ge(0.70),
            out["booking_hazard_score"].ge(0.66) & out["booking_side_pressure_score"].ge(0.70),
            out["booking_hazard_score"].ge(0.58),
        ],
        ["BOOKING_SIDE_AWARE_CORE", "BOOKING_SIDE_AWARE_CONFIRM", "BOOKING_SIDE_AWARE_WATCH"],
        default="BOOKING_LOW",
    )
    return out


def build_player_rows(players: pd.DataFrame, max_rows_per_fixture: int, require_confirmed_starter: bool) -> pd.DataFrame:
    lineup_ok = (
        players.get("lineup_confirmation_status", pd.Series("", index=players.index)).astype(str).eq("CONFIRMED_STARTER")
        if require_confirmed_starter
        else pd.Series(True, index=players.index)
    )
    eligible = players[
        num(players.get("expected_minutes", 0)).ge(55)
        & boolish(players.get("expected_start_flag", pd.Series(1, index=players.index)))
        & lineup_ok
        & (
            players.get("booking_pressure_side", pd.Series("", index=players.index)).astype(str).eq("BOTH")
            | players.get("booking_pressure_side", pd.Series("", index=players.index)).astype(str).eq(
                players.get("player_team_side", pd.Series("", index=players.index)).astype(str).str.upper()
            )
        )
        & players["booking_context_label"].isin(
            ["BOOKING_SIDE_AWARE_CORE", "BOOKING_SIDE_AWARE_CONFIRM", "BOOKING_SIDE_AWARE_WATCH"]
        )
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    eligible = eligible.sort_values(
        ["fixture_key", "booking_hazard_score", "booking_side_pressure_score", "booking_player_contact_score"],
        ascending=[True, False, False, False],
    )
    eligible = eligible.groupby("fixture_key", group_keys=False).head(max_rows_per_fixture).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        label = str(row.get("booking_context_label", "BOOKING_SIDE_AWARE_WATCH"))
        priority = (
            "PRIORITY_CORE"
            if label == "BOOKING_SIDE_AWARE_CORE"
            else "PRIORITY_CONFIRM"
            if label == "BOOKING_SIDE_AWARE_CONFIRM"
            else "WATCH_ONLY_NOT_PROMOTION"
        )
        records.append(
            {
                "shadow_family": "PLAYER_BOOKING_SIDE_AWARE",
                "shadow_stage": PLAYER_STAGE,
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league", scalar(row, "competition")),
                "home_team_name": scalar(row, "home_team_name", scalar(row, "home_team_name_fixture")),
                "away_team_name": scalar(row, "away_team_name", scalar(row, "away_team_name_fixture")),
                "expression": "Player Card 0.5+",
                "source_market": "PLAYER_CARDS",
                "source_selection": scalar(row, "player_name"),
                "source_deploy_tier": "PLAYER_EVENT_BETA",
                "source_tier": label,
                "combo_product": "",
                "combo_tier": label,
                "bookie_od": np.nan,
                "model_prob": np.nan,
                "value_edge": np.nan,
                "value_edge_tier": "",
                "backtest_hit_rate": np.nan,
                "backtest_graded": np.nan,
                "league_stability_bucket": "",
                "team_stability_bucket": "",
                "watch_priority": priority,
                "watch_flag": True,
                "guardrail": "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION|SIDE_AWARE_BOOKING_CONTEXT",
                "reason": (
                    f"{label}|pressure_side={row.get('booking_pressure_side')}|"
                    f"mode={row.get('booking_pressure_mode')}|role={row.get('role_cell')}|"
                    f"hazard={row.get('booking_hazard_score'):.3f}"
                ),
                "player_name": scalar(row, "player_name"),
                "team_name": scalar(row, "team_name"),
                "player_team_side": scalar(row, "player_team_side"),
                "position_group": scalar(row, "position_group"),
                "tactical_role": scalar(row, "tactical_role"),
                "predicted_hit_rate_pct": np.nan,
                "confidence_label": label,
                "lineup_watch_flags": scalar(row, "lineup_watch_flags"),
                "lineup_confirmation_status": scalar(row, "lineup_confirmation_status"),
                "expected_minutes": scalar(row, "expected_minutes", np.nan),
                "formation_matchup_label": scalar(row, "formation_matchup_label"),
                "formation_pressure_score": scalar(row, "formation_pressure_score", np.nan),
                "fixture_style_label": scalar(row, "fixture_style_label"),
                "fixture_attacking_style_label": scalar(row, "fixture_attacking_style_label"),
                "fixture_foul_density_score": scalar(row, "fixture_foul_density_score", np.nan),
                "fixture_wide_duel_score": scalar(row, "fixture_wide_duel_score", np.nan),
                "fixture_territorial_stress_score": scalar(row, "fixture_territorial_stress_score", np.nan),
                "fixture_attack_pressure_score": scalar(row, "fixture_attack_pressure_score", np.nan),
                "referee_name": scalar(row, "referee_name"),
                "ref_cards_per_match": scalar(row, "ref_cards_per_match", np.nan),
                "interaction_match_mode": "SIDE_AWARE_BOOKING_SHADOW",
                "interaction_label": label,
                "fouled_context_cell": "",
                "fouled_context_cell_label": "",
                "booking_pressure_side": scalar(row, "booking_pressure_side"),
                "booking_pressure_mode": scalar(row, "booking_pressure_mode"),
                "booking_context_label": label,
                "booking_hazard_score": scalar(row, "booking_hazard_score", np.nan),
                "booking_role_cell": scalar(row, "role_cell"),
                "booking_side_pressure_score": scalar(row, "booking_side_pressure_score", np.nan),
                "booking_player_contact_score": scalar(row, "booking_player_contact_score", np.nan),
                "booking_fixture_contact_score": scalar(row, "booking_fixture_contact_score", np.nan),
                "booking_recent_fouls_committed_per90": scalar(row, "_recent_fouls_per90", np.nan),
                "booking_goal_state_signal": scalar(row, "booking_goal_state_signal", np.nan),
            }
        )
    return pd.DataFrame(records).reindex(columns=OUTPUT_COLUMNS)


def build_team_rows(fixture_context: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in fixture_context.iterrows():
        sides = ["HOME", "AWAY"] if row.get("booking_pressure_side") == "BOTH" else [str(row.get("booking_pressure_side"))]
        for side in sides:
            if side not in {"HOME", "AWAY"}:
                continue
            team_name = row.get("home_team_name") if side == "HOME" else row.get("away_team_name")
            label = (
                "TEAM_CARDS_SIDE_AWARE_CONFIRM"
                if row.get("booking_pressure_mode") == "SIDE_AWARE_UNDERDOG_CARD_PRESSURE"
                else "TEAM_CARDS_BALANCED_WATCH"
            )
            priority = "PRIORITY_CONFIRM" if label.endswith("CONFIRM") else "WATCH_ONLY_NOT_PROMOTION"
            records.append(
                {
                    "shadow_family": "PLAYER_BOOKING_SIDE_AWARE",
                    "shadow_stage": TEAM_STAGE,
                    "fixture_key": row.get("fixture_key"),
                    "match_date": row.get("match_date"),
                    "league": row.get("league"),
                    "home_team_name": row.get("home_team_name"),
                    "away_team_name": row.get("away_team_name"),
                    "expression": "Team Cards 1.5+",
                    "source_market": "TEAM_CARDS",
                    "source_selection": team_name,
                    "source_deploy_tier": "PLAYER_EVENT_BETA",
                    "source_tier": label,
                    "combo_product": "",
                    "combo_tier": label,
                    "bookie_od": np.nan,
                    "model_prob": np.nan,
                    "value_edge": np.nan,
                    "value_edge_tier": "",
                    "backtest_hit_rate": np.nan,
                    "backtest_graded": np.nan,
                    "league_stability_bucket": "",
                    "team_stability_bucket": "",
                    "watch_priority": priority,
                    "watch_flag": True,
                    "guardrail": "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION|SIDE_AWARE_TEAM_BOOKING_CONTEXT",
                    "reason": f"{label}|pressure_side={row.get('booking_pressure_side')}|mode={row.get('booking_pressure_mode')}",
                    "player_name": team_name,
                    "team_name": team_name,
                    "player_team_side": side,
                    "position_group": "TEAM",
                    "tactical_role": "TEAM_CARD_PRESSURE",
                    "confidence_label": label,
                    "interaction_match_mode": "SIDE_AWARE_TEAM_BOOKING_SHADOW",
                    "interaction_label": label,
                    "booking_pressure_side": row.get("booking_pressure_side"),
                    "booking_pressure_mode": row.get("booking_pressure_mode"),
                    "booking_context_label": label,
                    "booking_goal_state_signal": row.get("booking_goal_state_signal"),
                }
            )
    return pd.DataFrame(records).reindex(columns=OUTPUT_COLUMNS)


def write_report(outdir: Path, rows: pd.DataFrame, fixture_context: pd.DataFrame, source_allmarkets: Path, source_features: Path) -> None:
    counts = (
        rows.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows")
        if not rows.empty
        else pd.DataFrame(columns=["shadow_stage", "watch_priority", "rows"])
    )
    top_cols = [
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "expression",
        "player_name",
        "team_name",
        "watch_priority",
        "booking_pressure_side",
        "booking_pressure_mode",
        "booking_context_label",
        "booking_hazard_score",
        "booking_role_cell",
    ]
    lines = [
        "# Player Booking Side-Aware Shadow Board",
        "",
        "Research-only booking sidecar for officially gradable player/team card rows.",
        "",
        "## Sources",
        f"- all-markets goal context: `{source_allmarkets}`",
        f"- player-event feature board: `{source_features}`",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Fixture Card Pressure Context",
        markdown_table(
            fixture_context[
                [
                    "match_date",
                    "league",
                    "home_team_name",
                    "away_team_name",
                    "dominance_side",
                    "booking_pressure_side",
                    "booking_pressure_mode",
                    "booking_goal_state_signal",
                ]
            ].head(80)
        ),
        "",
        "## Top Rows",
        markdown_table(rows[[col for col in top_cols if col in rows.columns]].head(100)),
        "",
        "## Guardrails",
        "- No priced player-card odds.",
        "- No deploy promotion.",
        "- Side-aware rows are official-grading instrumentation only.",
        "- Dominance/chasing-side logic is a hypothesis until repeated live outcome accumulation proves it.",
    ]
    (outdir / "PLAYER_BOOKING_SIDE_AWARE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allmarkets-csv", type=Path, default=DEFAULT_ALLMARKETS)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_PLAYER_EVENT_BOARD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--target-date", default="")
    parser.add_argument("--target-leagues", default="")
    parser.add_argument("--max-player-rows-per-fixture", type=int, default=5)
    parser.add_argument("--lineups-root", type=Path, default=None)
    parser.add_argument(
        "--require-confirmed-starter",
        action="store_true",
        help="Only emit player-card rows for players found in attached API starting XIs.",
    )
    args = parser.parse_args()

    allmarkets = read_csv(args.allmarkets_csv)
    if args.target_date:
        allmarkets["_match_date_filter"] = pd.to_datetime(allmarkets.get("match_date"), errors="coerce").dt.date.astype(str)
        allmarkets = allmarkets[allmarkets["_match_date_filter"].eq(args.target_date)].copy()
    if args.target_leagues:
        leagues = {item.strip() for item in args.target_leagues.split(",") if item.strip()}
        allmarkets = allmarkets[allmarkets.get("league", pd.Series("", index=allmarkets.index)).isin(leagues)].copy()
    if allmarkets.empty:
        raise SystemExit("No all-markets rows after filters.")

    features = read_csv(args.player_event_board)
    confirmed_starters = load_confirmed_starters(args.lineups_root)
    if args.require_confirmed_starter and not confirmed_starters:
        raise SystemExit("--require-confirmed-starter was set but no confirmed starter keys were loaded.")
    fixture_context = build_fixture_context(allmarkets)
    players = prepare_player_features(features, fixture_context)
    scored_players = score_players(players, confirmed_starters=confirmed_starters)
    player_rows = build_player_rows(
        scored_players,
        args.max_player_rows_per_fixture,
        require_confirmed_starter=args.require_confirmed_starter,
    )
    team_rows = build_team_rows(fixture_context)
    row_parts = [part for part in [player_rows, team_rows] if not part.empty]
    rows = (
        pd.concat(row_parts, ignore_index=True, sort=False).reindex(columns=OUTPUT_COLUMNS)
        if row_parts
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.outdir / "PLAYER_BOOKING_SIDE_AWARE_SHADOW_BOARD.csv", index=False)
    fixture_context.to_csv(args.outdir / "PLAYER_BOOKING_SIDE_AWARE_FIXTURE_CONTEXT.csv", index=False)
    if not scored_players.empty:
        scored_players.to_csv(args.outdir / "PLAYER_BOOKING_SIDE_AWARE_PLAYER_POOL_SCORED.csv", index=False)
    write_report(args.outdir, rows, fixture_context, args.allmarkets_csv, args.player_event_board)
    print(f"WROTE {args.outdir / 'PLAYER_BOOKING_SIDE_AWARE_SHADOW_BOARD.csv'}")
    print(f"rows={len(rows)} player_rows={len(player_rows)} team_rows={len(team_rows)}")


if __name__ == "__main__":
    main()
