#!/usr/bin/env python3
"""
Build a beta player-event intelligence board with predicted hit-rate bands.

This is not a priced prop/value board. It converts local API-Football/player-event
feature frames into transparent threshold likelihood estimates for app/web
intelligence surfaces:

- Player shots 0.5+ / 1.5+ / 2.5+ / 3.5+
- Player shots on target 0.5+ / 1.5+
- Player tackles 1.5+ / 2.5+ / 3.5+
- Player fouls 0.5+ / 1.5+ / 2.5+
- Player fouled 0.5+ / 1.5+ / 2.5+
- Player cards 0.5+

Outputs are beta/manual-review only and must not be wired into deploy routing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
PLAYER_EVENTS_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_hitrate_band_board"


MARKET_SPECS = {
    "PLAYER_SHOTS": {
        "stat_col": "shots_per90",
        "thresholds": [0.5, 1.5, 2.5, 3.5],
        "threshold_label": "Shots",
    },
    "PLAYER_SOT": {
        "stat_col": "shots_on_target_per90",
        "thresholds": [0.5, 1.5],
        "threshold_label": "Shots on target",
    },
    "PLAYER_TACKLES": {
        "stat_col": "tackles_per90",
        "thresholds": [1.5, 2.5, 3.5],
        "threshold_label": "Tackles",
    },
    "PLAYER_FOULS": {
        "stat_col": "fouls_per90",
        "thresholds": [0.5, 1.5, 2.5],
        "threshold_label": "Fouls committed",
    },
    "PLAYER_FOULED": {
        "stat_col": "fouls_won_per90",
        "thresholds": [0.5, 1.5, 2.5],
        "threshold_label": "Fouled",
    },
    "PLAYER_CARDS": {
        "stat_col": "yellow_cards_per90",
        "thresholds": [0.5],
        "threshold_label": "Cards",
    },
}

BAND_CAPS = {
    ("PLAYER_SHOTS", 0.5): 0.88,
    ("PLAYER_SHOTS", 1.5): 0.76,
    ("PLAYER_SHOTS", 2.5): 0.58,
    ("PLAYER_SHOTS", 3.5): 0.42,
    ("PLAYER_SOT", 0.5): 0.72,
    ("PLAYER_SOT", 1.5): 0.48,
    ("PLAYER_TACKLES", 1.5): 0.82,
    ("PLAYER_TACKLES", 2.5): 0.68,
    ("PLAYER_TACKLES", 3.5): 0.52,
    ("PLAYER_FOULS", 0.5): 0.80,
    ("PLAYER_FOULS", 1.5): 0.64,
    ("PLAYER_FOULS", 2.5): 0.46,
    ("PLAYER_FOULED", 0.5): 0.82,
    ("PLAYER_FOULED", 1.5): 0.66,
    ("PLAYER_FOULED", 2.5): 0.48,
    ("PLAYER_CARDS", 0.5): 0.38,
}

DASHBOARD_MIN_PROBS = {
    ("PLAYER_SHOTS", 0.5): 0.72,
    ("PLAYER_SHOTS", 1.5): 0.60,
    ("PLAYER_SHOTS", 2.5): 0.44,
    ("PLAYER_SHOTS", 3.5): 0.32,
    ("PLAYER_SOT", 0.5): 0.58,
    ("PLAYER_SOT", 1.5): 0.36,
    ("PLAYER_TACKLES", 1.5): 0.66,
    ("PLAYER_TACKLES", 2.5): 0.54,
    ("PLAYER_TACKLES", 3.5): 0.40,
    ("PLAYER_FOULS", 0.5): 0.64,
    ("PLAYER_FOULS", 1.5): 0.52,
    ("PLAYER_FOULS", 2.5): 0.36,
    ("PLAYER_FOULED", 0.5): 0.66,
    ("PLAYER_FOULED", 1.5): 0.52,
    ("PLAYER_FOULED", 2.5): 0.36,
    ("PLAYER_CARDS", 0.5): 0.26,
}


BASE_COLUMNS = [
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
    "expected_start_flag",
    "expected_minutes",
    "referee_name",
    "ref_cards_per_match",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "formation_matchup_label",
    "formation_pressure_score",
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "fixture_attack_pressure_score",
    "fixture_corner_pressure_score",
    "fixture_territorial_stress_score",
    "og_goal_environment_score",
    "og_battle_on_score",
    "player_quality_score_l5",
    "player_form_tier",
    "minutes_last_3_matches",
    "days_rest",
    "recent_injury_return_flag",
    "suspension_risk_flag",
    "match_stakes_score",
    "rivalry_flag",
]


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clamp(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return low
    return max(low, min(high, float(value)))


def score01(row: pd.Series, col: str) -> float:
    value = row.get(col, 0.0)
    if pd.isna(value):
        return 0.0
    value = float(value)
    if value > 1.5:
        value = value / 100.0
    return clamp(value, 0.0, 1.0)


def ref_score(row: pd.Series) -> float:
    cards = row.get("ref_cards_per_match", np.nan)
    if pd.isna(cards):
        return 0.0
    return clamp(float(cards) / 5.5, 0.0, 1.0)


def poisson_survival(lam: float, threshold: float) -> float:
    """P(X >= ceil(threshold)) for Poisson(lambda)."""
    lam = max(0.0, float(lam))
    k = int(math.floor(threshold) + 1)
    cdf = 0.0
    for i in range(k):
        cdf += math.exp(-lam) * (lam**i) / math.factorial(i)
    return clamp(1.0 - cdf, 0.0, 0.995)


def context_multiplier(row: pd.Series, market: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    mult = 1.0

    if market in {"PLAYER_SHOTS", "PLAYER_SOT"}:
        attack = score01(row, "fixture_attack_pressure_score")
        corners = score01(row, "fixture_corner_pressure_score")
        territory = score01(row, "fixture_territorial_stress_score")
        goal_env = score01(row, "og_goal_environment_score")
        mult += 0.16 * attack + 0.08 * corners + 0.08 * territory + 0.10 * goal_env
        if attack >= 0.65:
            reasons.append("ATTACK_PRESSURE")
        if goal_env >= 0.60:
            reasons.append("GOAL_ENV_SUPPORT")
        if "ATTACK" in clean_label(row.get("fixture_attacking_style_label")).upper():
            mult += 0.05
            reasons.append("ATTACK_STYLE")
        mult = clamp(mult, 0.75, 1.35)

    elif market == "PLAYER_TACKLES":
        tackle = score01(row, "fixture_tackle_density_score")
        grind = score01(row, "fixture_midfield_grind_score")
        wide = score01(row, "fixture_wide_duel_score")
        battle = score01(row, "og_battle_on_score")
        mult += 0.16 * tackle + 0.12 * grind + 0.10 * wide + 0.08 * battle
        if tackle >= 0.65:
            reasons.append("TACKLE_DENSITY")
        if grind >= 0.65:
            reasons.append("MIDFIELD_GRIND")
        if wide >= 0.65:
            reasons.append("WIDE_DUEL")
        mult = clamp(mult, 0.75, 1.35)

    elif market == "PLAYER_FOULS":
        foul = score01(row, "fixture_foul_density_score")
        grind = score01(row, "fixture_midfield_grind_score")
        referee = ref_score(row)
        stakes = score01(row, "match_stakes_score")
        mult += 0.16 * foul + 0.10 * grind + 0.08 * referee + 0.06 * stakes
        if foul >= 0.65:
            reasons.append("FOUL_DENSITY")
        if referee >= 0.65:
            reasons.append("STRICT_REF")
        if stakes >= 0.65:
            reasons.append("STAKES")
        mult = clamp(mult, 0.75, 1.35)

    elif market == "PLAYER_FOULED":
        foul = score01(row, "fixture_foul_density_score")
        wide = score01(row, "fixture_wide_duel_score")
        territory = score01(row, "fixture_territorial_stress_score")
        attack = score01(row, "fixture_attack_pressure_score")
        referee = ref_score(row)
        mult += 0.14 * foul + 0.12 * wide + 0.10 * territory + 0.08 * attack + 0.05 * referee
        if foul >= 0.65:
            reasons.append("FOUL_DENSITY")
        if wide >= 0.65:
            reasons.append("WIDE_DUEL")
        if territory >= 0.65:
            reasons.append("TERRITORY_STRESS")
        if attack >= 0.65:
            reasons.append("ATTACK_PRESSURE")
        role = clean_label(row.get("tactical_role")).upper()
        if "WIDE" in role or "FORWARD" in role:
            mult += 0.05
            reasons.append("FOUL_DRAWING_ROLE")
        mult = clamp(mult, 0.75, 1.35)

    elif market == "PLAYER_CARDS":
        referee = ref_score(row)
        foul = score01(row, "fixture_foul_density_score")
        battle = score01(row, "og_battle_on_score")
        suspension = score01(row, "suspension_risk_flag")
        mult += 0.20 * referee + 0.12 * foul + 0.10 * battle + 0.08 * suspension
        if referee >= 0.65:
            reasons.append("STRICT_REF")
        if foul >= 0.65:
            reasons.append("FOUL_DENSITY")
        if suspension >= 0.5:
            reasons.append("SUSPENSION_RISK")
        mult = clamp(mult, 0.70, 1.45)

    return mult, reasons


def confidence_label(market: str, threshold: float, prob: float, base_lambda: float, expected_minutes: float, support_score: int) -> str:
    if expected_minutes < 45 or base_lambda <= 0:
        return "LOW_MINUTES_OR_RATE"
    if market == "PLAYER_CARDS":
        if prob >= 0.34 and support_score >= 3:
            return "STRONG_WATCH"
        if prob >= 0.28:
            return "WATCH"
        return "INFO_ONLY"
    band_min = DASHBOARD_MIN_PROBS.get((market, threshold), 0.55)
    if prob >= band_min and support_score >= 2 and threshold >= 1.5:
        return "ALT_WATCH"
    if prob >= 0.72 and support_score >= 3:
        return "SHADOW_CORE"
    if prob >= 0.62 and support_score >= 2:
        return "STRONG_WATCH"
    if prob >= 0.52:
        return "WATCH"
    return "INFO_ONLY"


def support_score(row: pd.Series, market: str, reasons: list[str]) -> int:
    score = 0
    if row.get("expected_start_flag", 0) == 1:
        score += 1
    if float(row.get("expected_minutes", 0) or 0) >= 60:
        score += 1
    if clean_label(row.get("player_form_tier")).upper() in {"ELITE", "STRONG"}:
        score += 1
    score += min(2, len(reasons))
    if market in {"PLAYER_SHOTS", "PLAYER_SOT"} and score01(row, "fixture_attack_pressure_score") >= 0.65:
        score += 1
    if market in {"PLAYER_TACKLES", "PLAYER_FOULS"} and score01(row, "fixture_midfield_grind_score") >= 0.65:
        score += 1
    if market == "PLAYER_FOULED" and (
        score01(row, "fixture_wide_duel_score") >= 0.65
        or score01(row, "fixture_territorial_stress_score") >= 0.65
    ):
        score += 1
    if market == "PLAYER_CARDS" and ref_score(row) >= 0.65:
        score += 1
    return int(score)


def lineup_flags(row: pd.Series) -> list[str]:
    flags = []
    if float(row.get("expected_minutes", 0) or 0) < 60:
        flags.append("MINUTES_WATCH")
    if float(row.get("days_rest", 99) or 99) <= 3:
        flags.append("REST_WATCH")
    if float(row.get("minutes_last_3_matches", 0) or 0) >= 240:
        flags.append("LOAD_WATCH")
    if float(row.get("recent_injury_return_flag", 0) or 0) >= 1:
        flags.append("RECENT_INJURY_RETURN")
    return flags


def read_feature_frames(input_dir: Path, leagues: set[str] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("player_events_fixture_input__*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if leagues and "league" in frame.columns:
            frame = frame[frame["league"].astype(str).isin(leagues)]
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    for col in BASE_COLUMNS + [spec["stat_col"] for spec in MARKET_SPECS.values()]:
        if col not in data.columns:
            data[col] = np.nan
    data["expected_minutes"] = num(data["expected_minutes"]).fillna(0.0)
    data["expected_start_flag"] = num(data["expected_start_flag"]).fillna(0).astype(int)
    return data


def build_board(features: pd.DataFrame, min_minutes: float, min_prob: float) -> pd.DataFrame:
    rows = []
    candidates = features[
        (features["expected_minutes"] >= min_minutes)
        & (features["player_name"].notna())
        & (features["team_name"].notna())
    ].copy()

    for _, row in candidates.iterrows():
        expected_minutes = float(row.get("expected_minutes", 0) or 0)
        for market, spec in MARKET_SPECS.items():
            stat_col = spec["stat_col"]
            per90 = row.get(stat_col, np.nan)
            if pd.isna(per90) or float(per90) <= 0:
                continue
            base_lambda = max(0.0, float(per90) * expected_minutes / 90.0)
            mult, reasons = context_multiplier(row, market)
            adj_lambda = base_lambda * mult
            support = support_score(row, market, reasons)
            flags = lineup_flags(row)
            for threshold in spec["thresholds"]:
                raw_prob = poisson_survival(adj_lambda, threshold)
                prob = min(raw_prob, BAND_CAPS.get((market, threshold), 0.90))
                if prob < min_prob:
                    continue
                label = confidence_label(market, threshold, prob, base_lambda, expected_minutes, support)
                threshold_name = f"{spec['threshold_label']} {threshold:.1f}+"
                rows.append(
                    {
                        "product_mode": "INTELLIGENCE_ONLY_NOT_PRICED_ODDS",
                        "market_family": market,
                        "threshold": threshold,
                        "threshold_name": threshold_name,
                        "raw_poisson_hit_rate": round(raw_prob, 4),
                        "predicted_hit_rate": round(prob, 4),
                        "predicted_hit_rate_pct": round(prob * 100.0, 1),
                        "confidence_label": label,
                        "support_score": support,
                        "base_lambda": round(base_lambda, 4),
                        "context_multiplier": round(mult, 4),
                        "adjusted_lambda": round(adj_lambda, 4),
                        "source_stat_per90": round(float(per90), 4),
                        "context_reason_codes": "|".join(reasons) if reasons else "BASE_RATE_ONLY",
                        "lineup_watch_flags": "|".join(flags) if flags else "NO_LINEUP_WATCH_FLAG",
                        **{col: row.get(col, np.nan) for col in BASE_COLUMNS},
                    }
                )

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    sort_cols = ["confidence_label", "predicted_hit_rate", "support_score", "match_date"]
    board["_confidence_rank"] = board["confidence_label"].map(
        {"SHADOW_CORE": 4, "STRONG_WATCH": 3, "WATCH": 2, "INFO_ONLY": 1, "LOW_MINUTES_OR_RATE": 0}
    ).fillna(0)
    board = board.sort_values(
        ["_confidence_rank", "predicted_hit_rate", "support_score", "match_date"],
        ascending=[False, False, False, True],
    ).drop(columns=["_confidence_rank"])
    return board


def build_team_profiles(features: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    rows = []
    candidates = features[
        (features["expected_minutes"] >= min_minutes)
        & (features["team_name"].notna())
        & (features["fixture_key"].notna())
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    def event_lambda(row: pd.Series, market: str) -> float:
        stat_col = MARKET_SPECS[market]["stat_col"]
        per90 = row.get(stat_col, np.nan)
        if pd.isna(per90) or float(per90) <= 0:
            return 0.0
        base = float(per90) * float(row.get("expected_minutes", 0) or 0) / 90.0
        mult, _ = context_multiplier(row, market)
        return base * mult

    for (fixture_key, team_name), group in candidates.groupby(["fixture_key", "team_name"], dropna=False):
        first = group.iloc[0]
        foul_lambda = float(group.apply(lambda r: event_lambda(r, "PLAYER_FOULS"), axis=1).sum())
        tackle_lambda = float(group.apply(lambda r: event_lambda(r, "PLAYER_TACKLES"), axis=1).sum())
        card_lambda = float(group.apply(lambda r: event_lambda(r, "PLAYER_CARDS"), axis=1).sum())
        flags = []
        if foul_lambda >= 12:
            flags.append("TEAM_FOUL_HEAVY")
        if tackle_lambda >= 18:
            flags.append("TEAM_TACKLE_HEAVY")
        if card_lambda >= 2.0:
            flags.append("TEAM_CARD_HEAT")
        if ref_score(first) >= 0.65:
            flags.append("STRICT_REF")
        if score01(first, "fixture_midfield_grind_score") >= 0.65:
            flags.append("MIDFIELD_GRIND")
        if score01(first, "fixture_wide_duel_score") >= 0.65:
            flags.append("WIDE_DUEL")
        profile = "CONTACT_HOTSPOT" if len(flags) >= 3 else "CONTACT_WATCH" if flags else "BASELINE_CONTEXT"
        rows.append(
            {
                "product_mode": "INTELLIGENCE_ONLY_NOT_PRICED_ODDS",
                "fixture_key": fixture_key,
                "match_date": first.get("match_date", np.nan),
                "competition": first.get("competition", np.nan),
                "league": first.get("league", np.nan),
                "home_team_name": first.get("home_team_name", np.nan),
                "away_team_name": first.get("away_team_name", np.nan),
                "team_name": team_name,
                "player_rows_used": int(len(group)),
                "team_expected_fouls": round(foul_lambda, 2),
                "team_expected_tackles": round(tackle_lambda, 2),
                "team_expected_cards": round(card_lambda, 2),
                "team_contact_profile": profile,
                "team_context_flags": "|".join(flags) if flags else "NO_MAJOR_CONTACT_FLAG",
                "referee_name": first.get("referee_name", np.nan),
                "ref_cards_per_match": first.get("ref_cards_per_match", np.nan),
                "fixture_style_label": first.get("fixture_style_label", np.nan),
                "formation_matchup_label": first.get("formation_matchup_label", np.nan),
                "fixture_foul_density_score": first.get("fixture_foul_density_score", np.nan),
                "fixture_tackle_density_score": first.get("fixture_tackle_density_score", np.nan),
                "fixture_midfield_grind_score": first.get("fixture_midfield_grind_score", np.nan),
                "fixture_wide_duel_score": first.get("fixture_wide_duel_score", np.nan),
            }
        )
    profiles = pd.DataFrame(rows)
    if profiles.empty:
        return profiles
    profiles["_profile_rank"] = profiles["team_contact_profile"].map(
        {"CONTACT_HOTSPOT": 3, "CONTACT_WATCH": 2, "BASELINE_CONTEXT": 1}
    ).fillna(0)
    return profiles.sort_values(
        ["_profile_rank", "team_expected_cards", "team_expected_fouls", "team_expected_tackles"],
        ascending=[False, False, False, False],
    ).drop(columns=["_profile_rank"])


def write_markdown(board: pd.DataFrame, out_path: Path, source_rows: int) -> None:
    lines = [
        "# Player Event Hit-Rate Band Board",
        "",
        "Beta intelligence board for player-event likelihood bands. These are not priced odds and not deploy picks.",
        "",
        "## Safety",
        "- Intelligence-only output.",
        "- No bookmaker price/value claim.",
        "- No production deploy routing.",
        "- Requires lineup and market availability review before user-facing betting language.",
        "",
        "## Summary",
        f"- Source player rows: `{source_rows}`",
        f"- Band rows emitted: `{len(board)}`",
    ]
    if not board.empty:
        summary = (
            board.groupby(["market_family", "confidence_label"], dropna=False)
            .agg(rows=("fixture_key", "count"), fixtures=("fixture_key", "nunique"), players=("player_name", "nunique"), avg_hit_rate=("predicted_hit_rate", "mean"))
            .reset_index()
            .sort_values(["market_family", "rows"], ascending=[True, False])
        )
        summary["avg_hit_rate"] = (summary["avg_hit_rate"] * 100.0).round(1)
        lines.extend(["", "## Market x Confidence", markdown_table(summary)])

        top = board.head(40)[
            [
                "match_date",
                "league",
                "home_team_name",
                "away_team_name",
                "player_name",
                "team_name",
                "market_family",
                "threshold_name",
                "predicted_hit_rate_pct",
                "confidence_label",
                "context_reason_codes",
                "lineup_watch_flags",
            ]
        ]
        lines.extend(["", "## Top Rows", markdown_table(top)])
    out_path.write_text("\n".join(lines) + "\n")


def dashboard_slice(board: pd.DataFrame, per_band_limit: int) -> pd.DataFrame:
    if board.empty:
        return board.copy()
    mins = pd.Series(
        [
            DASHBOARD_MIN_PROBS.get((row.market_family, float(row.threshold)), 0.55)
            for row in board[["market_family", "threshold"]].itertuples(index=False)
        ],
        index=board.index,
    )
    priority = board[
        board["confidence_label"].isin(["SHADOW_CORE", "STRONG_WATCH", "WATCH", "ALT_WATCH"])
        & (board["predicted_hit_rate"] >= mins)
        & ~board["lineup_watch_flags"].astype(str).str.contains("MINUTES_WATCH", na=False)
    ].copy()
    if priority.empty:
        return priority
    priority["_confidence_rank"] = priority["confidence_label"].map({"SHADOW_CORE": 4, "STRONG_WATCH": 3, "WATCH": 2, "ALT_WATCH": 1}).fillna(0)
    priority = priority.sort_values(
        ["market_family", "threshold", "_confidence_rank", "predicted_hit_rate", "support_score", "match_date"],
        ascending=[True, True, False, False, False, True],
    )
    priority = (
        priority.groupby(["market_family", "threshold"], group_keys=False)
        .head(per_band_limit)
        .drop(columns=["_confidence_rank"])
        .sort_values(["predicted_hit_rate", "support_score", "match_date"], ascending=[False, False, True])
    )
    return priority


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PLAYER_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--leagues", default="", help="Optional comma-separated league filter.")
    parser.add_argument("--min-minutes", type=float, default=45.0)
    parser.add_argument("--min-prob", type=float, default=0.25)
    parser.add_argument("--dashboard-per-band-limit", type=int, default=500)
    args = parser.parse_args()

    leagues = {x.strip() for x in args.leagues.split(",") if x.strip()} or None
    args.outdir.mkdir(parents=True, exist_ok=True)

    features = read_feature_frames(args.input_dir, leagues)
    board = build_board(features, min_minutes=args.min_minutes, min_prob=args.min_prob)

    board_path = args.outdir / "PLAYER_EVENT_HITRATE_BAND_BOARD.csv"
    dashboard_path = args.outdir / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv"
    team_profile_path = args.outdir / "TEAM_CONTACT_EVENT_PROFILE.csv"
    summary_path = args.outdir / "PLAYER_EVENT_HITRATE_BAND_SUMMARY.csv"
    md_path = args.outdir / "PLAYER_EVENT_HITRATE_BAND_BOARD.md"

    board.to_csv(board_path, index=False)
    dashboard = dashboard_slice(board, args.dashboard_per_band_limit)
    dashboard.to_csv(dashboard_path, index=False)
    team_profiles = build_team_profiles(features, args.min_minutes)
    team_profiles.to_csv(team_profile_path, index=False)
    if board.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            board.groupby(["league", "market_family", "threshold_name", "confidence_label"], dropna=False)
            .agg(rows=("fixture_key", "count"), fixtures=("fixture_key", "nunique"), players=("player_name", "nunique"), avg_hit_rate=("predicted_hit_rate", "mean"))
            .reset_index()
            .sort_values(["league", "market_family", "threshold_name", "confidence_label"])
        )
        summary["avg_hit_rate"] = (summary["avg_hit_rate"] * 100.0).round(1)
    summary.to_csv(summary_path, index=False)
    write_markdown(board, md_path, len(features))

    print(f"WROTE {args.outdir}")
    print(f"source_rows={len(features)} band_rows={len(board)} dashboard_rows={len(dashboard)} team_profile_rows={len(team_profiles)}")


if __name__ == "__main__":
    main()
