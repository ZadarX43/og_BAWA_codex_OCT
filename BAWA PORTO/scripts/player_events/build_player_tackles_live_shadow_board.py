#!/usr/bin/env python3
"""Build strict tackles live-shadow watch rows.

Research-only. Converts current player-event hit-rate band rows into tackles
watch labels using conservative role/minutes/context filters. It does not
create priced odds, deploy picks, slips, or production routing changes.
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
    / "player_event_hitrate_band_board"
    / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_tackles_live_shadow_board"

ROLE_CELL_CONFIG = {
    "HOLDING_MIDFIELDER": {
        "terms": ("holding midfielder", "central midfielder"),
        "ge15": {
            "p": 0.815,
            "minutes": 78,
            "support": 6,
            "tackle_density": 0.72,
            "midfield_grind": 0.78,
            "wide_duel": 0.0,
            "limit": 35,
            "priority": "PRIORITY_CONFIRM",
        },
        "ge25": {
            "p": 0.665,
            "minutes": 80,
            "support": 6,
            "tackle_density": 0.82,
            "midfield_grind": 0.82,
            "wide_duel": 0.0,
            "limit": 20,
            "priority": "WATCH_ONLY_NOT_PROMOTION",
        },
    },
    "WIDE_DEFENDER_WING_BACK": {
        "terms": ("wide defender", "wing-back"),
        "ge15": {
            "p": 0.815,
            "minutes": 78,
            "support": 6,
            "tackle_density": 0.72,
            "midfield_grind": 0.0,
            "wide_duel": 0.91,
            "limit": 25,
            "priority": "PRIORITY_CONFIRM",
        },
        "ge25": {
            "p": 0.66,
            "minutes": 80,
            "support": 6,
            "tackle_density": 0.80,
            "midfield_grind": 0.0,
            "wide_duel": 0.93,
            "limit": 15,
            "priority": "WATCH_ONLY_NOT_PROMOTION",
        },
    },
    "CENTRE_BACK_STRICT": {
        "terms": ("centre-back", "center-back", "enforcer"),
        "ge15": {
            "p": 0.82,
            "minutes": 85,
            "support": 6,
            "tackle_density": 0.92,
            "midfield_grind": 0.88,
            "wide_duel": 0.96,
            "limit": 12,
            "priority": "WATCH_ONLY_NOT_PROMOTION",
        },
        "ge25": {
            "p": 0.68,
            "minutes": 88,
            "support": 6,
            "tackle_density": 0.96,
            "midfield_grind": 0.90,
            "wide_duel": 0.97,
            "limit": 6,
            "priority": "WATCH_ONLY_NOT_PROMOTION",
        },
    },
}

SHADOW_COLUMNS = [
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
    "watch_priority",
    "watch_flag",
    "guardrail",
    "reason",
    "player_name",
    "team_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "predicted_hit_rate",
    "predicted_hit_rate_pct",
    "confidence_label",
    "support_score",
    "context_reason_codes",
    "lineup_watch_flags",
    "expected_minutes",
    "formation_matchup_label",
    "formation_pressure_score",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_wide_duel_score",
    "fixture_midfield_grind_score",
    "fixture_territorial_stress_score",
    "fixture_attack_pressure_score",
    "referee_name",
    "ref_cards_per_match",
    "interaction_match_mode",
    "interaction_label",
    "tackles_role_cell",
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


def role_text(df: pd.DataFrame) -> pd.Series:
    role = df.get("tactical_role", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    position = df.get("position_group", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    return role + " " + position


def role_cell_mask(text: pd.Series, role_cell: str) -> pd.Series:
    terms = ROLE_CELL_CONFIG[role_cell]["terms"]
    return pd.Series(False, index=text.index) | np.logical_or.reduce([text.str.contains(term, regex=False) for term in terms])


def context_mask(rows: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=rows.index)
    mask &= rows["predicted_hit_rate"].ge(config["p"])
    mask &= rows["support_score"].ge(config["support"])
    mask &= rows["expected_minutes"].ge(config["minutes"])
    mask &= rows["fixture_tackle_density_score"].ge(config["tackle_density"])
    if config["midfield_grind"] > 0:
        mask &= rows["fixture_midfield_grind_score"].ge(config["midfield_grind"])
    if config["wide_duel"] > 0:
        mask &= rows["fixture_wide_duel_score"].ge(config["wide_duel"])
    return mask


def select_tackles(board: pd.DataFrame, per_market_limit: int) -> pd.DataFrame:
    rows = board[board.get("market_family", pd.Series("", index=board.index)).astype(str).eq("PLAYER_TACKLES")].copy()
    if rows.empty:
        return rows
    rows["predicted_hit_rate"] = num(rows.get("predicted_hit_rate", np.nan))
    rows["expected_minutes"] = num(rows.get("expected_minutes", np.nan))
    rows["support_score"] = num(rows.get("support_score", np.nan))
    rows["fixture_tackle_density_score"] = num(rows.get("fixture_tackle_density_score", np.nan))
    rows["fixture_midfield_grind_score"] = num(rows.get("fixture_midfield_grind_score", np.nan))
    rows["fixture_wide_duel_score"] = num(rows.get("fixture_wide_duel_score", np.nan))
    rows["_role_text"] = role_text(rows)

    selected: list[pd.DataFrame] = []
    for role_cell, role_config in ROLE_CELL_CONFIG.items():
        role_mask = role_cell_mask(rows["_role_text"], role_cell)
        for threshold_key, threshold_name, shadow_stage, expression in [
            ("ge15", "Tackles 1.5+", "PLAYER_TACKLES_1_5_LIVE_SHADOW", "Player Tackles 1.5+"),
            ("ge25", "Tackles 2.5+", "PLAYER_TACKLES_2_5_LIVE_SHADOW", "Player Tackles 2.5+"),
        ]:
            config = role_config[threshold_key]
            mask = rows["threshold_name"].astype(str).eq(threshold_name) & role_mask & context_mask(rows, config)
            cell = rows[mask].copy()
            if cell.empty:
                continue
            cell["shadow_stage"] = shadow_stage
            cell["expression"] = expression
            cell["watch_priority"] = config["priority"]
            cell["tackles_role_cell"] = role_cell
            cell["interaction_label"] = f"TACKLES_{role_cell}_{threshold_key.upper()}_ROLE_CELL"
            cell = cell.sort_values(
                ["predicted_hit_rate", "support_score", "expected_minutes", "fixture_tackle_density_score"],
                ascending=[False, False, False, False],
            ).head(int(config["limit"]))
            selected.append(cell)

    out = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(
        ["shadow_stage", "tackles_role_cell", "predicted_hit_rate", "support_score", "expected_minutes"],
        ascending=[True, True, False, False, False],
    )
    out = out.groupby("shadow_stage", group_keys=False).head(per_market_limit).copy()
    out["shadow_family"] = "PLAYER_EVENT_TACKLES"
    out["source_market"] = "PLAYER_TACKLES"
    out["source_selection"] = out["player_name"]
    out["source_deploy_tier"] = "PLAYER_EVENT_BETA"
    out["source_tier"] = out.get("confidence_label", "")
    out["combo_product"] = out["expression"]
    out["combo_tier"] = out.get("confidence_label", "")
    out["bookie_od"] = np.nan
    out["model_prob"] = out["predicted_hit_rate"]
    out["value_edge"] = np.nan
    out["value_edge_tier"] = ""
    out["backtest_hit_rate"] = np.where(out["shadow_stage"].eq("PLAYER_TACKLES_1_5_LIVE_SHADOW"), 0.7141, 0.0)
    out["backtest_graded"] = np.where(out["shadow_stage"].eq("PLAYER_TACKLES_1_5_LIVE_SHADOW"), 822, np.nan)
    out["watch_flag"] = True
    out["guardrail"] = "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION"
    out["reason"] = np.where(
        out["shadow_stage"].eq("PLAYER_TACKLES_1_5_LIVE_SHADOW"),
        "TACKLES_ROLE_CELL_TIGHTENED_CLOSEOUT_SHADOW_ONLY",
        "TACKLES_2_5_ROLE_CELL_TIGHTENED_WATCH_ONLY",
    )
    out["interaction_match_mode"] = "EXACT_LIVE_HITRATE_CONTEXT"
    for col in SHADOW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SHADOW_COLUMNS].reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_PLAYER_EVENT_BOARD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--per-market-limit", type=int, default=80)
    args = parser.parse_args()

    if not args.player_event_board.exists():
        raise SystemExit(f"Missing player-event board: {args.player_event_board}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(args.player_event_board, low_memory=False)
    shadow = select_tackles(board, args.per_market_limit)
    counts = shadow.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows") if not shadow.empty else pd.DataFrame()
    shadow.to_csv(args.outdir / "PLAYER_TACKLES_LIVE_SHADOW_BOARD.csv", index=False)
    counts.to_csv(args.outdir / "PLAYER_TACKLES_LIVE_SHADOW_COUNTS.csv", index=False)
    lines = [
        "# Player Tackles Live Shadow Board",
        "",
        "Research-only tackles watch rows from current player-event hit-rate bands.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Sample",
        markdown_table(
            shadow[
                [
                    col
                    for col in [
                        "match_date",
                        "league",
                        "player_name",
                        "team_name",
                        "expression",
                        "predicted_hit_rate",
                        "expected_minutes",
                        "fixture_tackle_density_score",
                        "tackles_role_cell",
                        "watch_priority",
                    ]
                    if col in shadow.columns
                ]
            ],
            max_rows=40,
        ),
    ]
    (args.outdir / "PLAYER_TACKLES_LIVE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {args.outdir}")
    print(f"shadow_rows={len(shadow)}")
    if not counts.empty:
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
