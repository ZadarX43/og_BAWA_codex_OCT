#!/usr/bin/env python3
"""Build current-board corners intelligence live-shadow rows.

Research-only. Converts team-shots profile context into team-corners watch
labels for the unified live shadow dashboard. It does not create priced odds,
deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOARD_DIR = ROOT / "predictions_output" / "2026-05-02"
DEFAULT_TEAM_SHOTS_CONTEXT = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "live_shadow_research_dashboard_with_tightened_tackles"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__TEAM_SHOTS_PROFILE_CONTEXT.csv"
)
DEFAULT_TEAM_PROFILE_ROWS = ROOT / "reports" / "2026-05-06" / "team_shots_profile" / "TEAM_SHOTS_PROFILE_ROWS.csv"
DEFAULT_AUDIT_CELLS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "corners_intelligence_audit_foundation"
    / "corners_intelligence_threshold_cells.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "corners_intelligence_live_shadow_board"

LEAGUE_TAG_ALIASES = {
    "belgium pro": "Belgium_Pro",
    "brazil serie a": "Brazil_Serie_A",
    "champions league": "Champions_League",
    "england championship": "England_Championship",
    "england efl league 1": "England_EFL_League_1",
    "england fa cup": "England_FA_Cup",
    "england premier league": "England_Premier_League",
    "europa conference": "Europa_Conference",
    "europa league": "Europa_League",
    "france ligue 1": "France_Ligue_1",
    "germany bundesliga": "Germany_Bundesliga",
    "italy serie a": "Italy_Serie_A",
    "japan j1": "Japan_J1",
    "netherlands eredivisie": "Netherlands_Eredivisie",
    "norway eliteserien": "Norway_Eliteserien",
    "portugal liga": "Portugal_Liga",
    "scotland premiership": "Scotland_Premiership",
    "spain la liga": "Spain_La_Liga",
    "usa mls": "USA_MLS",
}

STAGE_CONFIG = {
    "TEAM_CORNERS_4_5_LIVE_SHADOW": {
        "expression": "Team Corners 4.5+",
        "actual_market": "Team Corners 4.5+",
        "min_score": 0.72,
        "min_expected": 4.75,
        "priority": "PRIORITY_CORE",
        "limit": 80,
    },
    "TEAM_CORNERS_5_5_LIVE_SHADOW": {
        "expression": "Team Corners 5.5+",
        "actual_market": "Team Corners 5.5+",
        "min_score": 0.78,
        "min_expected": 5.35,
        "priority": "PRIORITY_CONFIRM",
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
    "interaction_match_mode",
    "interaction_label",
    "team_corner_pressure_score",
    "team_expected_corners",
    "team_expected_corners_pct",
    "match_expected_corners",
    "home_team_expected_corners",
    "away_team_expected_corners",
    "home_team_shots_profile_labels",
    "away_team_shots_profile_labels",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def discover_fixture_range(board_dir: Path, requested: str = "") -> str:
    if requested:
        return requested
    ranges: dict[str, float] = {}
    for path in board_dir.glob("*__DEPLOY_TIER_*__*.csv"):
        match = re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2})", path.name)
        if not match:
            continue
        ranges[match.group(1)] = max(ranges.get(match.group(1), 0.0), path.stat().st_mtime)
    if not ranges:
        raise SystemExit(f"No deploy tier files with fixture range found in {board_dir}")
    return max(ranges.items(), key=lambda item: item[1])[0]


def load_fixture_meta(board_dir: Path, fixture_range: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(board_dir.glob(f"*{fixture_range}*__DEPLOY_TIER_*__*.csv")):
        header = pd.read_csv(path, nrows=0)
        keep = [
            col
            for col in [
                "fixture_key",
                "match_date",
                "league",
                "home_team_name",
                "away_team_name",
                "deploy_tier",
                "tier",
            ]
            if col in header.columns
        ]
        if "fixture_key" not in keep:
            continue
        frames.append(pd.read_csv(path, usecols=keep, low_memory=False))
    if not frames:
        raise SystemExit(f"No fixture rows loaded for range {fixture_range}")
    meta = pd.concat(frames, ignore_index=True, sort=False)
    meta = meta.dropna(subset=["fixture_key"]).copy()
    meta["fixture_key"] = meta["fixture_key"].astype(str)
    if "deploy_tier" in meta.columns:
        tier_summary = meta.groupby("fixture_key")["deploy_tier"].apply(lambda s: "|".join(sorted(set(s.dropna().astype(str))))).reset_index(name="source_tiers_present")
    else:
        tier_summary = pd.DataFrame({"fixture_key": meta["fixture_key"].drop_duplicates(), "source_tiers_present": ""})
    first = meta.sort_values(["fixture_key"]).drop_duplicates("fixture_key", keep="first")
    return first.drop(columns=["deploy_tier", "tier"], errors="ignore").merge(tier_summary, on="fixture_key", how="left")


def load_backtest_lookup(path: Path) -> dict[str, dict[str, float]]:
    cells = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    lookup: dict[str, dict[str, float]] = {}
    if cells.empty:
        return lookup
    for market in ["Team Corners 4.5+", "Team Corners 5.5+"]:
        subset = cells[
            cells.get("market_display", pd.Series("", index=cells.index)).astype(str).eq(market)
            & cells.get("cell_label", pd.Series("", index=cells.index)).astype(str).eq("TEAM_SCORE_GE_0.80")
        ].copy()
        if subset.empty:
            subset = cells[cells.get("market_display", pd.Series("", index=cells.index)).astype(str).eq(market)].copy()
        if subset.empty:
            continue
        row = subset.sort_values(["recommended_beta_label", "lift_vs_baseline", "graded_rows"], ascending=[True, False, False]).iloc[0]
        lookup[market] = {
            "hit_rate": float(row.get("hit_rate", np.nan)),
            "graded": float(row.get("graded_rows", np.nan)),
        }
    return lookup


def league_tag_for(league: Any) -> str:
    return LEAGUE_TAG_ALIASES.get(norm_text(league), "")


def profile_distribution(profile_rows: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series]:
    rows = profile_rows.copy()
    rows["team_expected_corners"] = num(rows.get("team_expected_corners", np.nan))
    rows = rows[rows["team_expected_corners"].notna()].copy()
    by_league = {
        str(league_tag): group["team_expected_corners"].dropna().sort_values().reset_index(drop=True)
        for league_tag, group in rows.groupby("league_tag", dropna=False)
    }
    global_series = rows["team_expected_corners"].dropna().sort_values().reset_index(drop=True)
    return by_league, global_series


def percentile_from_distribution(value: float, dist: pd.Series) -> float:
    if pd.isna(value) or dist.empty:
        return np.nan
    return float((dist <= float(value)).mean())


def build_side_rows(context: pd.DataFrame, meta: pd.DataFrame, profile_rows: pd.DataFrame) -> pd.DataFrame:
    context = context.copy()
    context["fixture_key"] = context["fixture_key"].astype(str)
    rows = context.merge(meta, on="fixture_key", how="left")
    by_league, global_dist = profile_distribution(profile_rows)
    out_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        league_tag = league_tag_for(row.get("league"))
        dist = by_league.get(league_tag, global_dist)
        for side in ["home", "away"]:
            expected = row.get(f"{side}_team_expected_corners", np.nan)
            pct = percentile_from_distribution(expected, dist)
            label_text = str(row.get(f"{side}_team_shots_profile_labels", "") or "")
            label_boost = 0.08 if "CORNER_PRESSURE_WATCH" in label_text else 0.0
            score = float(np.nan_to_num(pct, nan=0.0) * 0.84 + np.nan_to_num(row.get("match_expected_corners", np.nan), nan=0.0) / 14.0 * 0.08 + label_boost)
            score = float(np.clip(score, 0.0, 1.0))
            team_name = row.get(f"{side}_team_name") if f"{side}_team_name" in row.index else row.get(f"{side}_team")
            if not team_name or pd.isna(team_name):
                team_name = row.get("home_team_name") if side == "home" else row.get("away_team_name")
            out_rows.append(
                {
                    "fixture_key": row.get("fixture_key"),
                    "match_date": row.get("match_date"),
                    "league": row.get("league"),
                    "league_tag": league_tag,
                    "home_team_name": row.get("home_team_name"),
                    "away_team_name": row.get("away_team_name"),
                    "team_name": team_name,
                    "player_team_side": side.upper(),
                    "team_expected_corners": expected,
                    "team_expected_corners_pct": pct,
                    "team_corner_pressure_score": score,
                    "match_expected_corners": row.get("match_expected_corners"),
                    "home_team_expected_corners": row.get("home_team_expected_corners"),
                    "away_team_expected_corners": row.get("away_team_expected_corners"),
                    "home_team_shots_profile_labels": row.get("home_team_shots_profile_labels"),
                    "away_team_shots_profile_labels": row.get("away_team_shots_profile_labels"),
                    "source_tiers_present": row.get("source_tiers_present", ""),
                    "team_shots_profile_context_mode": row.get("team_shots_profile_context_mode", ""),
                }
            )
    return pd.DataFrame(out_rows)


def select_shadow(side_rows: pd.DataFrame, backtest_lookup: dict[str, dict[str, float]], per_stage_limit: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for stage, config in STAGE_CONFIG.items():
        market = config["actual_market"]
        mask = side_rows["team_corner_pressure_score"].ge(config["min_score"]) & side_rows["team_expected_corners"].ge(config["min_expected"])
        cell = side_rows[mask].copy()
        if cell.empty:
            continue
        cell = cell.sort_values(
            ["team_corner_pressure_score", "team_expected_corners", "match_expected_corners"],
            ascending=[False, False, False],
        ).head(min(int(config["limit"]), per_stage_limit))
        cell["shadow_family"] = "CORNERS_INTELLIGENCE"
        cell["shadow_stage"] = stage
        cell["expression"] = config["expression"]
        cell["source_market"] = "CORNERS_INTELLIGENCE"
        cell["source_selection"] = cell["team_name"]
        cell["source_deploy_tier"] = "PLAYER_EVENT_BETA"
        cell["source_tier"] = "CORNERS_CORE_WATCH" if config["priority"] == "PRIORITY_CORE" else "CORNERS_CONFIRM_WATCH"
        cell["combo_product"] = cell["expression"]
        cell["combo_tier"] = cell["source_tier"]
        cell["bookie_od"] = np.nan
        cell["model_prob"] = cell["team_corner_pressure_score"]
        cell["predicted_hit_rate"] = cell["team_corner_pressure_score"]
        cell["predicted_hit_rate_pct"] = cell["team_corner_pressure_score"] * 100.0
        cell["value_edge"] = np.nan
        cell["value_edge_tier"] = ""
        cell["backtest_hit_rate"] = backtest_lookup.get(market, {}).get("hit_rate", np.nan)
        cell["backtest_graded"] = backtest_lookup.get(market, {}).get("graded", np.nan)
        cell["watch_priority"] = config["priority"]
        cell["watch_flag"] = True
        cell["guardrail"] = "CORNERS_INTELLIGENCE_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION"
        cell["reason"] = "TEAM_CORNERS_PRESSURE_LIVE_SHADOW_ONLY"
        cell["confidence_label"] = cell["source_tier"]
        cell["context_reason_codes"] = np.where(
            cell["team_expected_corners_pct"].ge(0.90),
            "HIGH_EXPECTED_CORNERS_PERCENTILE|TEAM_PROFILE_CONTEXT",
            "TEAM_PROFILE_CONTEXT|CORNER_PRESSURE",
        )
        cell["lineup_watch_flags"] = ""
        cell["interaction_match_mode"] = cell["team_shots_profile_context_mode"]
        cell["interaction_label"] = np.where(
            cell["player_team_side"].eq("HOME"),
            "HOME_TEAM_CORNER_PRESSURE",
            "AWAY_TEAM_CORNER_PRESSURE",
        )
        selected.append(cell)
    out = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()
    for col in SHADOW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SHADOW_COLUMNS] if not out.empty else pd.DataFrame(columns=SHADOW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", type=Path, default=DEFAULT_BOARD_DIR)
    parser.add_argument("--fixture-range", default="")
    parser.add_argument("--team-shots-context", type=Path, default=DEFAULT_TEAM_SHOTS_CONTEXT)
    parser.add_argument("--team-profile-rows", type=Path, default=DEFAULT_TEAM_PROFILE_ROWS)
    parser.add_argument("--audit-cells", type=Path, default=DEFAULT_AUDIT_CELLS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--per-stage-limit", type=int, default=80)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fixture_range = discover_fixture_range(args.board_dir, args.fixture_range)
    meta = load_fixture_meta(args.board_dir, fixture_range)
    context = pd.read_csv(args.team_shots_context, low_memory=False)
    profile_rows = pd.read_csv(args.team_profile_rows, low_memory=False)
    backtest_lookup = load_backtest_lookup(args.audit_cells)
    side_rows = build_side_rows(context, meta, profile_rows)
    shadow = select_shadow(side_rows, backtest_lookup, args.per_stage_limit)

    side_rows.to_csv(args.outdir / "CORNERS_INTELLIGENCE_LIVE_SIDE_ROWS.csv", index=False)
    shadow.to_csv(args.outdir / "CORNERS_INTELLIGENCE_LIVE_SHADOW_BOARD.csv", index=False)
    counts = shadow.groupby(["shadow_family", "shadow_stage"], dropna=False).size().reset_index(name="rows") if not shadow.empty else pd.DataFrame(columns=["shadow_family", "shadow_stage", "rows"])
    counts.to_csv(args.outdir / "CORNERS_INTELLIGENCE_LIVE_SHADOW_COUNTS.csv", index=False)

    report_cols = [
        "shadow_stage",
        "fixture_key",
        "league",
        "team_name",
        "expression",
        "team_corner_pressure_score",
        "team_expected_corners",
        "team_expected_corners_pct",
        "match_expected_corners",
        "backtest_hit_rate",
        "backtest_graded",
        "watch_priority",
    ]
    lines = [
        "# Corners Intelligence Live Shadow Board",
        "",
        "Research-only current-board corner pressure rows. No deploy tiers, priced odds, or slips are changed.",
        "",
        "## Summary",
        f"- fixture_range: {fixture_range}",
        f"- source_fixtures: {meta['fixture_key'].nunique()}",
        f"- context_rows: {len(context)}",
        f"- side_rows: {len(side_rows)}",
        f"- shadow_rows: {len(shadow)}",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Top Rows",
        markdown_table(shadow[report_cols] if not shadow.empty else shadow, max_rows=40),
        "",
        "## Guardrail",
        "- Corners Intelligence is dashboard context only.",
        "- These rows are not priced odds, deploy picks, or automated slip legs.",
        "- Team 4.5+/5.5+ corner signals should be live-shadow tracked before any product prominence change.",
    ]
    (args.outdir / "CORNERS_INTELLIGENCE_LIVE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE: {args.outdir}")
    print(f"fixture_range={fixture_range} side_rows={len(side_rows)} shadow_rows={len(shadow)}")
    if not counts.empty:
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
