#!/usr/bin/env python3
"""Build product-safe copy labels for player-event shadow rows.

Research-only app/web helper. Converts shadow watch rows into intelligence
language such as CORE WATCH, FORM + MATCHUP, ROLE EDGE, and LINEUP WATCH. It
does not create priced odds, deploy picks, slips, or production routing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAYER_EVENT_BOARD = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_exact_interaction_shadow_refresh_current_projection_exact_fouled_strict_policy"
    / "player_event_interaction_live_shadow_board_exact"
    / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "player_events" / "dashboard_copy_pack"

STAGE_COPY = {
    "PLAYER_SHOTS_1_5_INTERACTION_WATCH": {
        "market_display": "Player Shots 1.5+",
        "short_headline": "Shot volume watch",
        "market_note": "Recent attacker volume and opponent allowance agree.",
    },
    "PLAYER_SHOTS_2_5_INTERACTION_WATCH": {
        "market_display": "Player Shots 2.5+",
        "short_headline": "High shot-volume watch",
        "market_note": "Use only for high-usage attackers with strong role and matchup support.",
    },
    "PLAYER_SOT_0_5_INTERACTION_WATCH": {
        "market_display": "Player SOT 0.5+",
        "short_headline": "On-target threat watch",
        "market_note": "Recent attacker form, role, and opponent shot quality allowance support one shot on target.",
    },
    "PLAYER_FOULED_0_5_INTERACTION_WATCH": {
        "market_display": "Player Fouled 0.5+",
        "short_headline": "Fouled-player core watch",
        "market_note": "Role, recent fouls drawn, opponent concession, and contact context align.",
    },
    "PLAYER_FOULED_1_5_INTERACTION_WATCH": {
        "market_display": "Player Fouled 1.5+",
        "short_headline": "Fouled-player secondary watch",
        "market_note": "Higher threshold; keep as confirmation/watch unless live evidence strengthens.",
    },
    "PLAYER_TACKLES_1_5_LIVE_SHADOW": {
        "market_display": "Player Tackles 1.5+",
        "short_headline": "Tackle volume beta watch",
        "market_note": "Tackles closeout supports high-context top cells; keep shadow-only until repeated live outcomes accumulate.",
    },
    "PLAYER_TACKLES_2_5_LIVE_SHADOW": {
        "market_display": "Player Tackles 2.5+",
        "short_headline": "High tackle-volume watch",
        "market_note": "Higher threshold tackles watch; use only with strong minutes, role, and battle-context support.",
    },
    "KEEPER_SAVES_1_5_LIVE_SHADOW": {
        "market_display": "Keeper Saves 1.5+",
        "short_headline": "Keeper workload watch",
        "market_note": "Opponent SOT pressure and keeper-save context support elevated save volume.",
    },
    "KEEPER_SAVES_2_5_LIVE_SHADOW": {
        "market_display": "Keeper Saves 2.5+",
        "short_headline": "High keeper workload watch",
        "market_note": "Higher threshold keeper-save watch; use only when shot-on-target pressure is strong.",
    },
    "KEEPER_SAVES_3_5_LIVE_SHADOW": {
        "market_display": "Keeper Saves 3.5+",
        "short_headline": "Monster keeper workload watch",
        "market_note": "Very high threshold; keep as secondary watch until repeated live evidence accumulates.",
    },
    "TEAM_CORNERS_4_5_LIVE_SHADOW": {
        "market_display": "Team Corners 4.5+",
        "short_headline": "Corner pressure watch",
        "market_note": "Team corner pressure, attack territory, and opponent concession context align.",
    },
    "TEAM_CORNERS_5_5_LIVE_SHADOW": {
        "market_display": "Team Corners 5.5+",
        "short_headline": "High corner pressure watch",
        "market_note": "Higher threshold corner watch; keep evidence-led and team-context driven.",
    },
    "KEY_PASSES_0_5_LIVE_SHADOW": {
        "market_display": "Player Key Passes 0.5+",
        "short_headline": "Creator involvement watch",
        "market_note": "Recent creator profile and current match attack context support one key pass.",
    },
    "KEY_PASSES_1_5_LIVE_SHADOW": {
        "market_display": "Player Key Passes 1.5+",
        "short_headline": "High creator involvement watch",
        "market_note": "Higher threshold creator watch; strongest for repeat creators with secure minutes.",
    },
    "ASSIST_0_5_LIVE_WATCH": {
        "market_display": "Player Assist Watch 0.5+",
        "short_headline": "Assist-threat watch",
        "market_note": "Assist is teammate-finish dependent, so keep this as creator threat rather than a confident prop.",
    },
}

ROLE_EDGE_TERMS = ("STRIKER", "CENTRE_FORWARD", "CENTER_FORWARD", "WINGER", "WIDE_FORWARD", "ATTACKING_MID")
CORE_PRIORITIES = {"PRIORITY_CORE"}
CONFIRM_PRIORITIES = {"PRIORITY_CONFIRM"}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    if number <= 1.0:
        number *= 100.0
    return f"{number:.1f}%"


def product_label(row: pd.Series) -> str:
    priority = clean_text(row.get("watch_priority"))
    confidence = clean_text(row.get("confidence_label"))
    match_mode = clean_text(row.get("interaction_match_mode"))
    expected_minutes = row.get("expected_minutes", np.nan)
    if priority in CORE_PRIORITIES or confidence == "SHADOW_CORE":
        return "CORE WATCH"
    if match_mode == "EXACT_INTERACTION":
        return "FORM + MATCHUP"
    if priority in CONFIRM_PRIORITIES:
        return "CONFIRM WATCH"
    try:
        if float(expected_minutes) < 60:
            return "LINEUP WATCH"
    except (TypeError, ValueError):
        pass
    return "WATCH"


def role_label(row: pd.Series) -> str:
    role = clean_text(row.get("tactical_role")).upper().replace(" ", "_")
    position = clean_text(row.get("position_group")).upper().replace(" ", "_")
    joined = f"{role} {position}"
    if any(term in joined for term in ROLE_EDGE_TERMS):
        return "ROLE EDGE"
    return ""


def lineup_label(row: pd.Series) -> str:
    flags = clean_text(row.get("lineup_watch_flags"))
    try:
        minutes = float(row.get("expected_minutes"))
    except (TypeError, ValueError):
        minutes = np.nan
    labels = []
    if flags and flags != "NO_LINEUP_WATCH_FLAG":
        labels.append("LINEUP WATCH")
    if pd.notna(minutes) and minutes < 60:
        labels.append("MINUTES WATCH")
    return " | ".join(dict.fromkeys(labels))


def context_label(row: pd.Series) -> str:
    labels = []
    if clean_text(row.get("interaction_match_mode")) == "EXACT_INTERACTION":
        labels.append("FORM + MATCHUP")
    if clean_text(row.get("formation_matchup_label")):
        labels.append("TACTICAL CONTEXT")
    if clean_text(row.get("referee_name")):
        labels.append("REF CONTEXT")
    role = role_label(row)
    if role:
        labels.append(role)
    lineup = lineup_label(row)
    if lineup:
        labels.append(lineup)
    return " | ".join(dict.fromkeys(labels))


def confidence_sentence(row: pd.Series) -> str:
    stage = clean_text(row.get("shadow_stage"))
    spec = STAGE_COPY.get(stage, {})
    player = clean_text(row.get("player_name")) or clean_text(row.get("team_name")) or "Player"
    market = spec.get("market_display", clean_text(row.get("expression")) or stage)
    hit_rate = pct(row.get("predicted_hit_rate") if pd.notna(row.get("predicted_hit_rate", np.nan)) else row.get("backtest_hit_rate"))
    product = product_label(row)
    base = f"{player} is a {product} for {market}."
    if hit_rate:
        base += f" Historical band signal: {hit_rate}."
    note = spec.get("market_note", "")
    if note:
        base += f" {note}"
    return base


def build_copy_pack(board: pd.DataFrame) -> pd.DataFrame:
    rows = board.copy()
    if rows.empty:
        return rows
    rows = rows[rows.get("shadow_stage", pd.Series("", index=rows.index)).astype(str).isin(STAGE_COPY)].copy()
    if rows.empty:
        return rows
    rows["market_display"] = rows["shadow_stage"].map(lambda stage: STAGE_COPY.get(stage, {}).get("market_display", stage))
    rows["short_headline"] = rows["shadow_stage"].map(lambda stage: STAGE_COPY.get(stage, {}).get("short_headline", "Player-event watch"))
    rows["product_confidence_label"] = rows.apply(product_label, axis=1)
    rows["product_context_label"] = rows.apply(context_label, axis=1)
    rows["product_role_label"] = rows.apply(role_label, axis=1)
    rows["product_lineup_label"] = rows.apply(lineup_label, axis=1)
    rows["product_confidence_sentence"] = rows.apply(confidence_sentence, axis=1)
    rows["product_guardrail_copy"] = (
        "Research shadow only. This is a hit-rate intelligence label, not a priced prop odd or automated bet."
    )
    predicted = rows["predicted_hit_rate"] if "predicted_hit_rate" in rows.columns else pd.Series(np.nan, index=rows.index)
    backtest = rows["backtest_hit_rate"] if "backtest_hit_rate" in rows.columns else pd.Series(np.nan, index=rows.index)
    rows["display_probability_band"] = [pct(pred) if pct(pred) else pct(bt) for pred, bt in zip(predicted, backtest)]
    keep = [
        "match_date",
        "league",
        "fixture_key",
        "home_team_name",
        "away_team_name",
        "player_name",
        "team_name",
        "player_team_side",
        "position_group",
        "tactical_role",
        "market_display",
        "short_headline",
        "product_confidence_label",
        "product_context_label",
        "display_probability_band",
        "watch_priority",
        "confidence_label",
        "interaction_match_mode",
        "fouled_context_cell_label",
        "expected_minutes",
        "formation_matchup_label",
        "fixture_foul_density_score",
        "fixture_wide_duel_score",
        "referee_name",
        "product_confidence_sentence",
        "product_guardrail_copy",
    ]
    keep = [col for col in keep if col in rows.columns]
    return rows[keep].sort_values(["match_date", "league", "fixture_key", "product_confidence_label", "player_name"])


def summarize(copy_pack: pd.DataFrame) -> pd.DataFrame:
    if copy_pack.empty:
        return pd.DataFrame(columns=["market_display", "product_confidence_label", "rows"])
    return (
        copy_pack.groupby(["market_display", "product_confidence_label"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["market_display", "rows"], ascending=[True, False])
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in work.columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, copy_pack: pd.DataFrame, summary: pd.DataFrame, input_path: Path) -> None:
    lines = [
        "# Player Event Dashboard Copy Pack",
        "",
        "Product-safe copy and labels for player-event shadow rows.",
        "",
        "## Safety",
        "- Intelligence/watch language only.",
        "- No priced player-prop odds.",
        "- No deploy routing, slips, or production rulebook changes.",
        "",
        "## Input",
        f"- `{input_path}`",
        "",
        "## Overall",
        f"- copy rows: `{len(copy_pack)}`",
        "",
        "## Label Counts",
        markdown_table(summary),
        "",
        "## Sample Rows",
        markdown_table(
            copy_pack[
                [
                    col
                    for col in [
                        "match_date",
                        "league",
                        "player_name",
                        "team_name",
                        "market_display",
                        "product_confidence_label",
                        "product_context_label",
                        "display_probability_band",
                    ]
                    if col in copy_pack.columns
                ]
            ],
            max_rows=30,
        ),
    ]
    (outdir / "PLAYER_EVENT_DASHBOARD_COPY_PACK.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_PLAYER_EVENT_BOARD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.player_event_board.exists():
        raise SystemExit(f"Missing player-event board: {args.player_event_board}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    board = pd.read_csv(args.player_event_board, low_memory=False)
    copy_pack = build_copy_pack(board)
    summary = summarize(copy_pack)
    copy_pack.to_csv(args.outdir / "PLAYER_EVENT_DASHBOARD_COPY_PACK.csv", index=False)
    summary.to_csv(args.outdir / "PLAYER_EVENT_DASHBOARD_COPY_PACK_SUMMARY.csv", index=False)
    write_report(args.outdir, copy_pack, summary, args.player_event_board)

    print(f"WROTE {args.outdir}")
    print(f"copy_rows={len(copy_pack)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
