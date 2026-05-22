#!/usr/bin/env python3
"""
Research-only interaction audit for player-event attack markets.

Tests whether recent attacker form becomes more reliable when the opponent has
also allowed shots/SOT to the player's attacking role.

No production artifacts, deploy routing, or priced player-prop probabilities
are written. Outputs are beta/intelligence proof tables only.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MARKET_AUDIT_DIR = ROOT / "reports" / "2026-05-06" / "player_events_market_goal_spine_model_audit"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_recent_form_opponent_allowance_interaction_audit"

QUANTILES = (0.60, 0.70, 0.80, 0.90)
MIN_MONTH_ROWS = 15


@dataclass(frozen=True)
class MarketSpec:
    market: str
    display_market: str
    target_label: str
    input_path: Path
    hit_col: str
    actual_col: str
    beta_core_min_hit: float
    research_ready_min_hit: float
    watch_min_hit: float
    recent_features: tuple[str, ...]
    opponent_features: tuple[str, ...]


MARKETS = {
    "shots_ge2": MarketSpec(
        market="shots_ge2",
        display_market="Player Shots 1.5+",
        target_label="shots_total >= 2",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge2_joined_scored.csv",
        hit_col="actual_hit_ge2",
        actual_col="shots_total",
        beta_core_min_hit=0.48,
        research_ready_min_hit=0.43,
        watch_min_hit=0.38,
        recent_features=(
            "pred_rate_plus_attacker_recent_form",
            "attacker_recent_shots_per90_l8",
            "attacker_recent_shots_l8",
            "attacker_recent_sot_per90_l8",
            "attacker_recent_home_shots_per90_l8",
            "attacker_recent_away_shots_per90_l8",
        ),
        opponent_features=(
            "pred_rate_plus_opponent_attack_allowance",
            "opp_attack_allowed_role_shots_per_player_l10",
            "opp_attack_allowed_role_shots_per_match_l10",
            "opp_attack_allowed_role_player_shot_ge2_rate_l10",
            "opp_attack_allowed_role_player_shot_ge1_rate_l10",
            "opp_attack_allowed_role_sot_per_player_l10",
            "opp_attack_allowed_attacker_any_shots_per_player_l10",
        ),
    ),
    "shots_ge3": MarketSpec(
        market="shots_ge3",
        display_market="Player Shots 2.5+",
        target_label="shots_total >= 3",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge3_joined_scored.csv",
        hit_col="actual_hit_ge3",
        actual_col="shots_total",
        beta_core_min_hit=0.24,
        research_ready_min_hit=0.20,
        watch_min_hit=0.16,
        recent_features=(
            "pred_rate_plus_attacker_recent_form",
            "attacker_recent_shots_per90_l8",
            "attacker_recent_shots_l8",
            "attacker_recent_sot_per90_l8",
            "attacker_recent_home_shots_per90_l8",
            "attacker_recent_away_shots_per90_l8",
        ),
        opponent_features=(
            "pred_rate_plus_opponent_attack_allowance",
            "opp_attack_allowed_role_shots_per_player_l10",
            "opp_attack_allowed_role_shots_per_match_l10",
            "opp_attack_allowed_role_player_shot_ge2_rate_l10",
            "opp_attack_allowed_role_sot_per_player_l10",
            "opp_attack_allowed_role_sot_per_match_l10",
            "opp_attack_allowed_attacker_any_shots_per_player_l10",
        ),
    ),
    "shots_on_target": MarketSpec(
        market="shots_on_target",
        display_market="Player SOT 0.5+",
        target_label="shots_on_target >= 1",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_on_target_joined_scored.csv",
        hit_col="actual_hit_ge1",
        actual_col="shots_on_target",
        beta_core_min_hit=0.52,
        research_ready_min_hit=0.48,
        watch_min_hit=0.43,
        recent_features=(
            "pred_rate_plus_attacker_recent_form",
            "attacker_recent_sot_per90_l8",
            "attacker_recent_sot_l8",
            "attacker_recent_shots_per90_l8",
            "attacker_recent_home_sot_per90_l8",
            "attacker_recent_away_sot_per90_l8",
        ),
        opponent_features=(
            "pred_rate_plus_opponent_attack_allowance",
            "opp_attack_allowed_role_sot_per_player_l10",
            "opp_attack_allowed_role_sot_per_match_l10",
            "opp_attack_allowed_role_player_sot_ge1_rate_l10",
            "opp_attack_allowed_role_shots_per_player_l10",
            "opp_attack_allowed_role_player_shot_ge1_rate_l10",
            "opp_attack_allowed_attacker_any_sot_per_player_l10",
        ),
    ),
}


BASE_COLS = [
    "fixture_id",
    "fixture_key",
    "match_date",
    "eval_month",
    "competition",
    "league",
    "league_tag",
    "season_tag",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "position_group",
    "tactical_role",
]


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_market(spec: MarketSpec) -> pd.DataFrame:
    if not spec.input_path.exists():
        return pd.DataFrame()
    header = pd.read_csv(spec.input_path, nrows=0)
    requested = (
        BASE_COLS
        + [spec.hit_col, spec.actual_col]
        + list(spec.recent_features)
        + list(spec.opponent_features)
    )
    usecols = [c for c in requested if c in header.columns]
    df = pd.read_csv(spec.input_path, usecols=usecols, low_memory=False)
    df["market"] = spec.market
    df["display_market"] = spec.display_market
    df["target_label"] = spec.target_label
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "eval_month" not in df.columns:
        df["eval_month"] = df["match_date"].dt.to_period("M").astype(str)
    if "season_tag" not in df.columns:
        df["season_tag"] = df["match_date"].dt.year
    if "league_tag" not in df.columns:
        df["league_tag"] = np.nan
    if df["league_tag"].isna().all():
        fallback = df["competition"] if "competition" in df.columns else df.get("league", np.nan)
        df["league_tag"] = fallback.astype(str).str.replace(" ", "_", regex=False)
        df.loc[df["league_tag"].isin(["nan", "None", ""]), "league_tag"] = np.nan
    df[spec.hit_col] = num(df[spec.hit_col])
    return df.dropna(subset=["match_date", spec.hit_col]).copy()


def threshold_value(series: pd.Series, quantile: float) -> float:
    values = num(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(quantile))


def stable_month_share(subset: pd.DataFrame, hit_col: str, compare_hit: float) -> tuple[int, int, float]:
    monthly = (
        subset.groupby("eval_month", dropna=False)
        .agg(rows=(hit_col, "size"), hit_rate=(hit_col, "mean"))
        .reset_index()
    )
    monthly = monthly[monthly["rows"].ge(MIN_MONTH_ROWS)].copy()
    if monthly.empty:
        return 0, 0, np.nan
    stable = int(monthly["hit_rate"].ge(compare_hit).sum())
    return int(len(monthly)), stable, float(stable / len(monthly))


def label_candidate(
    *,
    spec: MarketSpec,
    rows: int,
    hit_rate: float,
    lift_vs_baseline: float,
    lift_vs_recent_only: float,
    lift_vs_opponent_only: float,
    months_ge_min: int,
    stable_vs_recent: float,
) -> str:
    if rows < 80 or months_ge_min < 3 or math.isnan(hit_rate):
        return "DO_NOT_USE"
    if (
        rows >= 500
        and months_ge_min >= 10
        and hit_rate >= spec.beta_core_min_hit
        and lift_vs_baseline >= 0.08
        and lift_vs_recent_only >= 0.02
        and lift_vs_opponent_only >= 0.02
        and stable_vs_recent >= 0.60
    ):
        return "INTERACTION_CORE"
    if (
        rows >= 250
        and months_ge_min >= 8
        and hit_rate >= spec.research_ready_min_hit
        and lift_vs_baseline >= 0.06
        and lift_vs_opponent_only >= 0.02
        and lift_vs_recent_only >= 0.00
    ):
        return "INTERACTION_READY"
    if (
        rows >= 250
        and months_ge_min >= 8
        and hit_rate >= spec.research_ready_min_hit
        and lift_vs_baseline >= 0.06
        and lift_vs_opponent_only >= 0.02
        and lift_vs_recent_only >= -0.02
    ):
        return "CONFIRMER_ONLY"
    if rows >= 120 and months_ge_min >= 5 and hit_rate >= spec.watch_min_hit and lift_vs_baseline >= 0.03:
        return "WATCH"
    return "DO_NOT_USE"


def summarize_interaction(
    df: pd.DataFrame,
    spec: MarketSpec,
    recent_feature: str,
    opponent_feature: str,
    recent_quantile: float,
    opponent_quantile: float,
) -> dict[str, Any] | None:
    recent_t = threshold_value(df[recent_feature], recent_quantile)
    opp_t = threshold_value(df[opponent_feature], opponent_quantile)
    if math.isnan(recent_t) or math.isnan(opp_t):
        return None

    valid = df[[spec.hit_col, "eval_month", recent_feature, opponent_feature]].dropna().copy()
    if len(valid) < 250 or valid[spec.hit_col].nunique() < 2:
        return None

    baseline_hit = float(valid[spec.hit_col].mean())
    recent_mask = num(valid[recent_feature]).ge(recent_t)
    opponent_mask = num(valid[opponent_feature]).ge(opp_t)
    combo = valid[recent_mask & opponent_mask].copy()
    recent_only = valid[recent_mask].copy()
    opponent_only = valid[opponent_mask].copy()
    if combo.empty:
        return None

    combo_hit = float(combo[spec.hit_col].mean())
    recent_hit = float(recent_only[spec.hit_col].mean()) if not recent_only.empty else np.nan
    opponent_hit = float(opponent_only[spec.hit_col].mean()) if not opponent_only.empty else np.nan
    months_ge_min, stable_baseline, stable_vs_baseline = stable_month_share(combo, spec.hit_col, baseline_hit)
    _, stable_recent, stable_vs_recent = stable_month_share(combo, spec.hit_col, recent_hit)
    _, stable_opp, stable_vs_opp = stable_month_share(combo, spec.hit_col, opponent_hit)

    lift_vs_baseline = combo_hit - baseline_hit
    lift_vs_recent = combo_hit - recent_hit
    lift_vs_opp = combo_hit - opponent_hit
    return {
        "market": spec.market,
        "display_market": spec.display_market,
        "target_label": spec.target_label,
        "recent_feature": recent_feature,
        "opponent_feature": opponent_feature,
        "recent_quantile": recent_quantile,
        "opponent_quantile": opponent_quantile,
        "recent_threshold": recent_t,
        "opponent_threshold": opp_t,
        "baseline_rows": int(len(valid)),
        "baseline_hit": baseline_hit,
        "recent_only_rows": int(len(recent_only)),
        "recent_only_hit": recent_hit,
        "opponent_only_rows": int(len(opponent_only)),
        "opponent_only_hit": opponent_hit,
        "interaction_rows": int(len(combo)),
        "interaction_hit": combo_hit,
        "lift_vs_baseline": lift_vs_baseline,
        "lift_vs_recent_only": lift_vs_recent,
        "lift_vs_opponent_only": lift_vs_opp,
        "months_ge_min": months_ge_min,
        "stable_months_vs_baseline": stable_baseline,
        "stable_month_share_vs_baseline": stable_vs_baseline,
        "stable_months_vs_recent_only": stable_recent,
        "stable_month_share_vs_recent_only": stable_vs_recent,
        "stable_months_vs_opponent_only": stable_opp,
        "stable_month_share_vs_opponent_only": stable_vs_opp,
        "interaction_label": label_candidate(
            spec=spec,
            rows=int(len(combo)),
            hit_rate=combo_hit,
            lift_vs_baseline=lift_vs_baseline,
            lift_vs_recent_only=lift_vs_recent,
            lift_vs_opponent_only=lift_vs_opp,
            months_ge_min=months_ge_min,
            stable_vs_recent=stable_vs_recent,
        ),
    }


def interaction_row_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    return (
        num(df[row["recent_feature"]]).ge(float(row["recent_threshold"]))
        & num(df[row["opponent_feature"]]).ge(float(row["opponent_threshold"]))
    )


def group_breakdown(df_by_market: dict[str, pd.DataFrame], grid: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keep_labels = {"INTERACTION_CORE", "INTERACTION_READY", "CONFIRMER_ONLY", "WATCH"}
    candidates = grid[grid["interaction_label"].isin(keep_labels)].copy()
    for _, candidate in candidates.iterrows():
        df = df_by_market.get(str(candidate["market"]))
        if df is None or df.empty:
            continue
        mask = interaction_row_mask(df, candidate)
        subset = df[mask].copy()
        if subset.empty:
            continue
        for key, group in subset.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            if len(group) < 20:
                continue
            record = candidate[
                [
                    "market",
                    "display_market",
                    "recent_feature",
                    "opponent_feature",
                    "recent_quantile",
                    "opponent_quantile",
                    "interaction_label",
                    "baseline_hit",
                    "recent_only_hit",
                    "opponent_only_hit",
                ]
            ].to_dict()
            record.update(dict(zip(group_cols, key)))
            hit_rate = float(group[candidate["target_hit_col"]].mean()) if "target_hit_col" in candidate else np.nan
            rows.append(
                {
                    **record,
                    "rows": int(len(group)),
                    "hit_rate": hit_rate,
                    "lift_vs_baseline": hit_rate - float(candidate["baseline_hit"]),
                    "lift_vs_recent_only": hit_rate - float(candidate["recent_only_hit"]),
                    "lift_vs_opponent_only": hit_rate - float(candidate["opponent_only_hit"]),
                }
            )
    return pd.DataFrame(rows)


def add_target_hit_col(grid: pd.DataFrame, market_specs: dict[str, MarketSpec]) -> pd.DataFrame:
    out = grid.copy()
    out["target_hit_col"] = out["market"].map({market: spec.hit_col for market, spec in market_specs.items()})
    return out


def build_market_grid(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_recent = [c for c in spec.recent_features if c in df.columns]
    available_opp = [c for c in spec.opponent_features if c in df.columns]
    for recent_feature in available_recent:
        for opponent_feature in available_opp:
            for recent_q in QUANTILES:
                for opponent_q in QUANTILES:
                    row = summarize_interaction(
                        df,
                        spec,
                        recent_feature,
                        opponent_feature,
                        recent_q,
                        opponent_q,
                    )
                    if row is not None:
                        rows.append(row)
    return pd.DataFrame(rows)


def top_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    label_rank = {
        "INTERACTION_CORE": 0,
        "INTERACTION_READY": 1,
        "CONFIRMER_ONLY": 2,
        "WATCH": 3,
        "DO_NOT_USE": 4,
    }
    work = grid.copy()
    work["label_rank"] = work["interaction_label"].map(label_rank).fillna(9)
    return (
        work.sort_values(
            [
                "label_rank",
                "interaction_hit",
                "lift_vs_recent_only",
                "interaction_rows",
                "stable_month_share_vs_recent_only",
            ],
            ascending=[True, False, False, False, False],
        )
        .drop(columns=["label_rank"])
        .groupby("market", as_index=False, group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
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


def write_report(outdir: Path, grid: pd.DataFrame, top: pd.DataFrame, league: pd.DataFrame, role: pd.DataFrame) -> None:
    summary_cols = [
        "market",
        "recent_feature",
        "opponent_feature",
        "recent_quantile",
        "opponent_quantile",
        "interaction_rows",
        "interaction_hit",
        "lift_vs_baseline",
        "lift_vs_recent_only",
        "lift_vs_opponent_only",
        "stable_month_share_vs_recent_only",
        "interaction_label",
    ]
    label_counts = (
        grid.groupby(["market", "interaction_label"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["market", "interaction_label"])
    )
    lines = [
        "# Player Event Recent Form x Opponent Allowance Interaction Audit",
        "",
        "Research-only beta/intelligence audit for shots/SOT markets.",
        "",
        "## Safety",
        "- No production artifacts written.",
        "- No deploy routing, tiers, slips, or priced player-prop probabilities changed.",
        "- Candidate labels are research labels for manual review and live-shadow QA only.",
        "",
        "## Tested Markets",
        "- Player Shots 1.5+",
        "- Player Shots 2.5+",
        "- Player SOT 0.5+",
        "",
        "## Label Counts",
        markdown_table(label_counts, max_rows=40),
        "",
        "## Best Interaction Candidates",
        markdown_table(top[[c for c in summary_cols if c in top.columns]], max_rows=30),
        "",
        "## Best League Breakdowns",
        markdown_table(
            league.sort_values(["hit_rate", "rows"], ascending=[False, False]).head(30)
            if not league.empty
            else league
        ),
        "",
        "## Best Role Breakdowns",
        markdown_table(
            role.sort_values(["hit_rate", "rows"], ascending=[False, False]).head(30)
            if not role.empty
            else role
        ),
        "",
        "## Interpretation Rules",
        "- `INTERACTION_CORE` means the intersection beat both recent-form-only and opponent-only by useful margins.",
        "- `CONFIRMER_ONLY` means opponent allowance supports filtering but did not clearly beat recent-form-only.",
        "- Prefer recent-form-only when interaction lift is negative versus recent form.",
        "- Treat all outputs as beta intelligence until live-shadow repeats confirm the pattern.",
    ]
    (outdir / "PLAYER_EVENT_RECENT_FORM_OPPONENT_ALLOWANCE_INTERACTION_AUDIT.md").write_text(
        "\n".join(lines) + "\n"
    )


def parse_csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="shots_ge2,shots_ge3,shots_on_target")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    markets = parse_csv_tuple(args.markets, str)
    args.outdir.mkdir(parents=True, exist_ok=True)

    grids: list[pd.DataFrame] = []
    dfs: dict[str, pd.DataFrame] = {}
    selected_specs: dict[str, MarketSpec] = {}
    for market in markets:
        if market not in MARKETS:
            raise SystemExit(f"Unsupported market: {market}. Supported: {', '.join(MARKETS)}")
        spec = MARKETS[market]
        df = read_market(spec)
        if df.empty:
            continue
        dfs[market] = df
        selected_specs[market] = spec
        grids.append(build_market_grid(df, spec))

    grid = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    grid = add_target_hit_col(grid, selected_specs) if not grid.empty else grid
    top = top_candidates(grid)
    league = group_breakdown(dfs, grid, ["league_tag", "season_tag"]) if not grid.empty else pd.DataFrame()
    role = group_breakdown(dfs, grid, ["tactical_role"]) if not grid.empty else pd.DataFrame()

    grid.to_csv(args.outdir / "player_event_recent_form_opponent_allowance_interaction_grid.csv", index=False)
    top.to_csv(args.outdir / "player_event_recent_form_opponent_allowance_top_candidates.csv", index=False)
    league.to_csv(args.outdir / "player_event_recent_form_opponent_allowance_league_breakdown.csv", index=False)
    role.to_csv(args.outdir / "player_event_recent_form_opponent_allowance_role_breakdown.csv", index=False)
    write_report(args.outdir, grid, top, league, role)

    print(f"WROTE {args.outdir}")
    if not top.empty:
        cols = [
            "market",
            "recent_feature",
            "opponent_feature",
            "recent_quantile",
            "opponent_quantile",
            "interaction_rows",
            "interaction_hit",
            "lift_vs_recent_only",
            "interaction_label",
        ]
        print(top[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
