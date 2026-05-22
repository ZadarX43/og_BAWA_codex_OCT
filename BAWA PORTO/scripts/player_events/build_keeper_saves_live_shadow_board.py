#!/usr/bin/env python3
"""Build current-board Keeper Saves Intelligence live-shadow rows.

Research-only. Converts attacking-team SOT pressure into named opposing
keeper saves watch labels. It does not create priced odds, deploy picks, slips,
or production routing changes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEAM_SHOTS_CONTEXT = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "live_shadow_research_dashboard_with_corners"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__TEAM_SHOTS_PROFILE_CONTEXT.csv"
)
DEFAULT_CURRENT_INPUTS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_current_board_fixture_inputs_clean"
    / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_ALL.csv"
)
DEFAULT_TEAM_PROFILE_ROWS = ROOT / "reports" / "2026-05-06" / "team_shots_profile" / "TEAM_SHOTS_PROFILE_ROWS.csv"
DEFAULT_AUDIT_CELLS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "keeper_saves_intelligence_audit_foundation"
    / "keeper_saves_intelligence_threshold_cells.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "keeper_saves_live_shadow_board"

LEAGUE_TAG_ALIASES = {
    "belgium pro": "Belgium_Pro",
    "brazil serie a": "Brazil_Serie_A",
    "england championship": "England_Championship",
    "england efl league 1": "England_EFL_League_1",
    "england premier league": "England_Premier_League",
    "france ligue 1": "France_Ligue_1",
    "germany bundesliga": "Germany_Bundesliga",
    "italy serie a": "Italy_Serie_A",
    "netherlands eredivisie": "Netherlands_Eredivisie",
    "norway eliteserien": "Norway_Eliteserien",
    "portugal liga": "Portugal_Liga",
    "scotland premiership": "Scotland_Premiership",
    "spain la liga": "Spain_La_Liga",
    "usa mls": "USA_MLS",
}

STAGE_CONFIG = {
    "KEEPER_SAVES_1_5_LIVE_SHADOW": {
        "expression": "Keeper Saves 1.5+",
        "market": "Keeper Saves 1.5+",
        "min_score": 0.72,
        "min_expected_sot": 4.65,
        "priority": "PRIORITY_CORE",
        "limit": 90,
    },
    "KEEPER_SAVES_2_5_LIVE_SHADOW": {
        "expression": "Keeper Saves 2.5+",
        "market": "Keeper Saves 2.5+",
        "min_score": 0.78,
        "min_expected_sot": 5.25,
        "priority": "PRIORITY_CONFIRM",
        "limit": 75,
    },
    "KEEPER_SAVES_3_5_LIVE_SHADOW": {
        "expression": "Keeper Saves 3.5+",
        "market": "Keeper Saves 3.5+",
        "min_score": 0.84,
        "min_expected_sot": 5.85,
        "priority": "WATCH_ONLY_NOT_PROMOTION",
        "limit": 60,
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
    "predicted_hit_rate",
    "predicted_hit_rate_pct",
    "confidence_label",
    "context_reason_codes",
    "lineup_watch_flags",
    "expected_minutes",
    "interaction_match_mode",
    "interaction_label",
    "keeper_save_pressure_score",
    "keeper_expected_sot_faced",
    "keeper_expected_sot_faced_pct",
    "attacking_team_name",
    "attacking_team_side",
    "match_expected_sot",
    "home_keeper_save_pressure_score",
    "away_keeper_save_pressure_score",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def league_tag_for(league: Any) -> str:
    return LEAGUE_TAG_ALIASES.get(norm_text(league), "")


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


def percentile_from_distribution(value: float, dist: pd.Series) -> float:
    if pd.isna(value) or dist.empty:
        return np.nan
    return float((dist <= float(value)).mean())


def distributions(profile_rows: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series]:
    rows = profile_rows.copy()
    rows["keeper_save_pressure_score"] = num(rows.get("keeper_save_pressure_score", np.nan))
    rows = rows[rows["keeper_save_pressure_score"].notna()].copy()
    by_league = {
        str(league): group["keeper_save_pressure_score"].dropna().sort_values().reset_index(drop=True)
        for league, group in rows.groupby("league_tag", dropna=False)
    }
    return by_league, rows["keeper_save_pressure_score"].dropna().sort_values().reset_index(drop=True)


def load_backtest_lookup(path: Path) -> dict[str, dict[str, float]]:
    cells = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    lookup: dict[str, dict[str, float]] = {}
    if cells.empty:
        return lookup
    for market in ["Keeper Saves 1.5+", "Keeper Saves 2.5+", "Keeper Saves 3.5+"]:
        subset = cells[
            cells.get("market_display", pd.Series("", index=cells.index)).astype(str).eq(market)
            & cells.get("cell_label", pd.Series("", index=cells.index)).astype(str).eq("SCORE_GE_0.85")
        ].copy()
        if subset.empty:
            subset = cells[cells.get("market_display", pd.Series("", index=cells.index)).astype(str).eq(market)].copy()
        if subset.empty:
            continue
        row = subset.sort_values(["recommended_beta_label", "lift_vs_baseline", "graded_rows"], ascending=[True, False, False]).iloc[0]
        lookup[market] = {"hit_rate": float(row.get("hit_rate", np.nan)), "graded": float(row.get("graded_rows", np.nan))}
    return lookup


def load_keeper_candidates(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    gk = rows[rows.get("position_group", pd.Series("", index=rows.index)).astype(str).str.contains("Goalkeeper", case=False, na=False)].copy()
    if gk.empty:
        return gk
    gk["expected_minutes"] = num(gk.get("expected_minutes", np.nan))
    gk["_team_key"] = gk["team_name"].map(norm_text)
    gk = gk.sort_values(
        ["fixture_key", "_team_key", "expected_start_flag", "expected_minutes"],
        ascending=[True, True, False, False],
    )
    return gk.drop_duplicates(["fixture_key", "_team_key"], keep="first")


def build_pressure_rows(context: pd.DataFrame, keepers: pd.DataFrame, profile_rows: pd.DataFrame) -> pd.DataFrame:
    by_league, global_dist = distributions(profile_rows)
    keeper_lookup = {
        (str(row.fixture_key), norm_text(row.team_name)): row
        for row in keepers.itertuples(index=False)
    }
    out: list[dict[str, Any]] = []
    for _, row in context.iterrows():
        league_tag = league_tag_for(row.get("league", ""))
        if not league_tag:
            # Context files may not carry league; infer from keeper candidate rows.
            fixture_keepers = keepers[keepers["fixture_key"].astype(str).eq(str(row.get("fixture_key")))]
            league_tag = league_tag_for(fixture_keepers["league"].iloc[0]) if not fixture_keepers.empty else ""
        dist = by_league.get(league_tag, global_dist)
        for attacking_side, keeper_side in [("home", "away"), ("away", "home")]:
            attacking_team = row.get(f"{attacking_side}_team_name")
            keeper_team = row.get(f"{keeper_side}_team_name")
            if pd.isna(attacking_team) or pd.isna(keeper_team):
                continue
            expected_sot = row.get(f"{attacking_side}_keeper_save_pressure_score", row.get(f"{attacking_side}_team_expected_sot", np.nan))
            pct = percentile_from_distribution(expected_sot, dist)
            score = float(np.clip(0.88 * np.nan_to_num(pct, nan=0.0) + 0.12 * np.nan_to_num(row.get("match_expected_sot", np.nan), nan=0.0) / 11.0, 0.0, 1.0))
            keeper = keeper_lookup.get((str(row.get("fixture_key")), norm_text(keeper_team)))
            out.append(
                {
                    "fixture_key": row.get("fixture_key"),
                    "match_date": row.get("match_date"),
                    "league": row.get("league"),
                    "home_team_name": row.get("home_team_name"),
                    "away_team_name": row.get("away_team_name"),
                    "player_name": getattr(keeper, "player_name", f"{keeper_team} goalkeeper"),
                    "team_name": keeper_team,
                    "player_team_side": keeper_side.upper(),
                    "expected_minutes": getattr(keeper, "expected_minutes", np.nan),
                    "lineup_watch_flags": "KEEPER_NAME_PROJECTED" if keeper is not None else "KEEPER_NAME_MISSING_TEAM_LEVEL",
                    "attacking_team_name": attacking_team,
                    "attacking_team_side": attacking_side.upper(),
                    "keeper_expected_sot_faced": expected_sot,
                    "keeper_expected_sot_faced_pct": pct,
                    "keeper_save_pressure_score": score,
                    "match_expected_sot": row.get("match_expected_sot"),
                    "home_keeper_save_pressure_score": row.get("home_keeper_save_pressure_score"),
                    "away_keeper_save_pressure_score": row.get("away_keeper_save_pressure_score"),
                    "interaction_match_mode": row.get("team_shots_profile_context_mode", "TEAM_SHOTS_PROFILE_CONTEXT"),
                }
            )
    return pd.DataFrame(out)


def select_shadow(rows: pd.DataFrame, backtest_lookup: dict[str, dict[str, float]], per_stage_limit: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for stage, config in STAGE_CONFIG.items():
        market = config["market"]
        mask = rows["keeper_save_pressure_score"].ge(config["min_score"]) & rows["keeper_expected_sot_faced"].ge(config["min_expected_sot"])
        cell = rows[mask].copy()
        if cell.empty:
            continue
        cell = cell.sort_values(
            ["keeper_save_pressure_score", "keeper_expected_sot_faced", "match_expected_sot"],
            ascending=[False, False, False],
        ).head(min(int(config["limit"]), per_stage_limit))
        cell["shadow_family"] = "KEEPER_SAVES_INTELLIGENCE"
        cell["shadow_stage"] = stage
        cell["expression"] = config["expression"]
        cell["source_market"] = "KEEPER_SAVES_INTELLIGENCE"
        cell["source_selection"] = cell["player_name"]
        cell["source_deploy_tier"] = "PLAYER_EVENT_BETA"
        cell["source_tier"] = "KEEPER_SAVES_CORE_WATCH" if config["priority"] == "PRIORITY_CORE" else "KEEPER_SAVES_CONFIRM_WATCH"
        cell["combo_product"] = cell["expression"]
        cell["combo_tier"] = cell["source_tier"]
        cell["bookie_od"] = np.nan
        cell["model_prob"] = cell["keeper_save_pressure_score"]
        cell["predicted_hit_rate"] = cell["keeper_save_pressure_score"]
        cell["predicted_hit_rate_pct"] = cell["keeper_save_pressure_score"] * 100.0
        cell["value_edge"] = np.nan
        cell["value_edge_tier"] = ""
        cell["backtest_hit_rate"] = backtest_lookup.get(market, {}).get("hit_rate", np.nan)
        cell["backtest_graded"] = backtest_lookup.get(market, {}).get("graded", np.nan)
        cell["watch_priority"] = config["priority"]
        cell["watch_flag"] = True
        cell["guardrail"] = "KEEPER_SAVES_INTELLIGENCE_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION"
        cell["reason"] = "ATTACKING_SOT_PRESSURE_TO_OPPOSING_KEEPER_SAVES_SHADOW"
        cell["confidence_label"] = cell["source_tier"]
        cell["context_reason_codes"] = np.where(
            cell["keeper_expected_sot_faced_pct"].ge(0.90),
            "HIGH_SOT_FACED_PERCENTILE|KEEPER_SAVE_PRESSURE",
            "KEEPER_SAVE_PRESSURE|TEAM_SOT_CONTEXT",
        )
        cell["interaction_label"] = "OPPOSING_KEEPER_SAVE_PRESSURE"
        selected.append(cell)
    out = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()
    for col in SHADOW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SHADOW_COLUMNS] if not out.empty else pd.DataFrame(columns=SHADOW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-shots-context", type=Path, default=DEFAULT_TEAM_SHOTS_CONTEXT)
    parser.add_argument("--current-inputs", type=Path, default=DEFAULT_CURRENT_INPUTS)
    parser.add_argument("--team-profile-rows", type=Path, default=DEFAULT_TEAM_PROFILE_ROWS)
    parser.add_argument("--audit-cells", type=Path, default=DEFAULT_AUDIT_CELLS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--per-stage-limit", type=int, default=90)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(args.team_shots_context, low_memory=False)
    current_inputs = pd.read_csv(args.current_inputs, low_memory=False)
    keepers = load_keeper_candidates(args.current_inputs)
    # The context exported by the dashboard may not carry names/league, so add it from current inputs.
    meta_cols = ["fixture_key", "match_date", "league", "home_team_name", "away_team_name"]
    meta = current_inputs[meta_cols].drop_duplicates("fixture_key", keep="first")
    context = context.merge(meta, on="fixture_key", how="left", suffixes=("", "_meta"))
    for col in ["match_date", "league", "home_team_name", "away_team_name"]:
        meta_col = f"{col}_meta"
        if meta_col in context.columns:
            context[col] = context[col].combine_first(context[meta_col]) if col in context.columns else context[meta_col]
    profile_rows = pd.read_csv(args.team_profile_rows, low_memory=False)
    backtest_lookup = load_backtest_lookup(args.audit_cells)
    pressure_rows = build_pressure_rows(context, keepers, profile_rows)
    shadow = select_shadow(pressure_rows, backtest_lookup, args.per_stage_limit)

    pressure_rows.to_csv(args.outdir / "KEEPER_SAVES_LIVE_PRESSURE_ROWS.csv", index=False)
    shadow.to_csv(args.outdir / "KEEPER_SAVES_LIVE_SHADOW_BOARD.csv", index=False)
    counts = shadow.groupby(["shadow_family", "shadow_stage"], dropna=False).size().reset_index(name="rows") if not shadow.empty else pd.DataFrame(columns=["shadow_family", "shadow_stage", "rows"])
    counts.to_csv(args.outdir / "KEEPER_SAVES_LIVE_SHADOW_COUNTS.csv", index=False)

    report_cols = [
        "shadow_stage",
        "fixture_key",
        "league",
        "player_name",
        "team_name",
        "attacking_team_name",
        "expression",
        "keeper_save_pressure_score",
        "keeper_expected_sot_faced",
        "keeper_expected_sot_faced_pct",
        "match_expected_sot",
        "backtest_hit_rate",
        "backtest_graded",
        "watch_priority",
    ]
    lines = [
        "# Keeper Saves Live Shadow Board",
        "",
        "Research-only current-board keeper saves pressure rows. No deploy tiers, priced odds, or slips are changed.",
        "",
        "## Summary",
        f"- context_rows: {len(context)}",
        f"- keeper_candidates: {len(keepers)}",
        f"- pressure_rows: {len(pressure_rows)}",
        f"- shadow_rows: {len(shadow)}",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Top Rows",
        markdown_table(shadow[report_cols] if not shadow.empty else shadow, max_rows=45),
        "",
        "## Guardrail",
        "- Keeper Saves Intelligence is dashboard context only.",
        "- These rows are named watch labels where current keeper projection exists; otherwise they remain team-level goalkeeper labels.",
        "- No priced prop odds, deploy picks, or automated slip legs are created.",
    ]
    (args.outdir / "KEEPER_SAVES_LIVE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE: {args.outdir}")
    print(f"pressure_rows={len(pressure_rows)} shadow_rows={len(shadow)}")
    if not counts.empty:
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
