#!/usr/bin/env python3
"""
Research-only threshold stability audit for player-event intelligence bands.

Consumes scored player-event audit outputs and tests whether probability bands
are stable by market, league, season, month, player, and team.

No deploy routing, production artifacts, or priced player-prop probabilities
are written. This is beta intelligence for manual review and live-shadow QA.
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
TACKLES_AUDIT_DIR = ROOT / "reports" / "2026-05-06" / "player_events_goal_spine_model_audit"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit"

THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TOP_FRACTIONS = (0.05, 0.10)
MIN_MONTH_ROWS = 20


@dataclass(frozen=True)
class MarketSpec:
    market: str
    display_market: str
    target_label: str
    input_path: Path
    hit_col: str
    actual_col: str
    preferred_variants: tuple[str, ...]
    beta_core_min_hit: float
    research_ready_min_hit: float
    watch_min_hit: float


MARKETS = {
    "shots": MarketSpec(
        market="shots",
        display_market="Player Shots 0.5+",
        target_label="shots_total >= 1",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_joined_scored.csv",
        hit_col="actual_hit_ge1",
        actual_col="shots_total",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.72,
        research_ready_min_hit=0.68,
        watch_min_hit=0.63,
    ),
    "shots_ge2": MarketSpec(
        market="shots_ge2",
        display_market="Player Shots 1.5+",
        target_label="shots_total >= 2",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge2_joined_scored.csv",
        hit_col="actual_hit_ge2",
        actual_col="shots_total",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.48,
        research_ready_min_hit=0.43,
        watch_min_hit=0.38,
    ),
    "shots_ge3": MarketSpec(
        market="shots_ge3",
        display_market="Player Shots 2.5+",
        target_label="shots_total >= 3",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge3_joined_scored.csv",
        hit_col="actual_hit_ge3",
        actual_col="shots_total",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.24,
        research_ready_min_hit=0.20,
        watch_min_hit=0.16,
    ),
    "shots_on_target": MarketSpec(
        market="shots_on_target",
        display_market="Player SOT 0.5+",
        target_label="shots_on_target >= 1",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_shots_on_target_joined_scored.csv",
        hit_col="actual_hit_ge1",
        actual_col="shots_on_target",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.52,
        research_ready_min_hit=0.48,
        watch_min_hit=0.43,
    ),
    "sot_ge2_attackers": MarketSpec(
        market="sot_ge2_attackers",
        display_market="Striker/Winger SOT 1.5+",
        target_label="attacking roles shots_on_target >= 2",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_sot_ge2_attackers_joined_scored.csv",
        hit_col="actual_hit_ge2",
        actual_col="shots_on_target",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.30,
        research_ready_min_hit=0.26,
        watch_min_hit=0.23,
    ),
    "sot_ge3_attackers": MarketSpec(
        market="sot_ge3_attackers",
        display_market="Striker/Winger SOT 2.5+",
        target_label="attacking roles shots_on_target >= 3",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_sot_ge3_attackers_joined_scored.csv",
        hit_col="actual_hit_ge3",
        actual_col="shots_on_target",
        preferred_variants=(
            "pred_rate_plus_attacker_recent_form",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_opponent_attack_allowance",
            "pred_rate_plus_all_context",
            "pred_rate_isotonic_oof",
        ),
        beta_core_min_hit=0.10,
        research_ready_min_hit=0.085,
        watch_min_hit=0.07,
    ),
    "fouls_committed": MarketSpec(
        market="fouls_committed",
        display_market="Player Fouls 1.5+",
        target_label="fouls_committed >= 2",
        input_path=MARKET_AUDIT_DIR / "player_events_goal_spine_fouls_committed_joined_scored.csv",
        hit_col="actual_hit_ge2",
        actual_col="fouls_committed",
        preferred_variants=(
            "pred_rate_isotonic_oof",
            "pred_rate_plus_player_layer",
            "pred_rate_plus_all_context",
        ),
        beta_core_min_hit=0.38,
        research_ready_min_hit=0.34,
        watch_min_hit=0.30,
    ),
    "tackles": MarketSpec(
        market="tackles",
        display_market="Player Tackles 1.5+",
        target_label="tackles >= 2",
        input_path=TACKLES_AUDIT_DIR / "player_events_goal_spine_tackles_joined_scored.csv",
        hit_col="actual_hit_ge2",
        actual_col="actual_tackles",
        preferred_variants=(
            "pred_nb_isotonic_oof",
            "pred_nb_plus_all_context",
            "pred_raw_nb",
        ),
        beta_core_min_hit=0.62,
        research_ready_min_hit=0.58,
        watch_min_hit=0.54,
    ),
}


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_market(spec: MarketSpec) -> pd.DataFrame:
    if not spec.input_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(spec.input_path, low_memory=False)
    df["market"] = spec.market
    df["display_market"] = spec.display_market
    df["target_label"] = spec.target_label
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "eval_month" not in df.columns:
        df["eval_month"] = df["match_date"].dt.to_period("M").astype(str)
    if "league_tag" not in df.columns and "league" in df.columns:
        df["league_tag"] = df["league"].astype(str).str.replace(" ", "_", regex=False)
    if "season_tag" not in df.columns:
        df["season_tag"] = df["match_date"].dt.year
    df[spec.hit_col] = num(df[spec.hit_col])
    df[spec.actual_col] = num(df[spec.actual_col]) if spec.actual_col in df.columns else np.nan
    return df.dropna(subset=["match_date", spec.hit_col]).copy()


def available_variants(df: pd.DataFrame, spec: MarketSpec) -> list[str]:
    preferred = [v for v in spec.preferred_variants if v in df.columns]
    extras = [
        c
        for c in df.columns
        if c.startswith("pred_") and c not in preferred and c not in {"pred_raw_rate_poisson", "pred_raw_nb"}
    ]
    return preferred + extras[:4]


def beta_label(
    *,
    rows: int,
    hit_rate: float,
    lift_pp: float,
    months_ge_min: int,
    stable_month_share: float,
    spec: MarketSpec,
) -> str:
    if rows < 80 or months_ge_min < 3 or math.isnan(hit_rate):
        return "DO_NOT_USE"
    if (
        rows >= 500
        and months_ge_min >= 10
        and hit_rate >= spec.beta_core_min_hit
        and lift_pp >= 0.06
        and stable_month_share >= 0.70
    ):
        return "BETA_CORE"
    if (
        rows >= 250
        and months_ge_min >= 8
        and hit_rate >= spec.research_ready_min_hit
        and lift_pp >= 0.04
        and stable_month_share >= 0.60
    ):
        return "RESEARCH_READY"
    if rows >= 120 and months_ge_min >= 5 and hit_rate >= spec.watch_min_hit and lift_pp >= 0.02:
        return "WATCH"
    return "DO_NOT_USE"


def summarize_subset(
    subset: pd.DataFrame,
    spec: MarketSpec,
    pred_col: str,
    baseline_hit: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    if subset.empty:
        return {}
    month = (
        subset.groupby("eval_month", dropna=False)
        .agg(month_rows=(spec.hit_col, "size"), month_hit_rate=(spec.hit_col, "mean"))
        .reset_index()
    )
    stable_floor = max(baseline_hit + 0.02, spec.watch_min_hit)
    month_eligible = month[month["month_rows"].ge(MIN_MONTH_ROWS)].copy()
    stable_share = (
        float(month_eligible["month_hit_rate"].ge(stable_floor).mean()) if not month_eligible.empty else np.nan
    )
    hit_rate = float(subset[spec.hit_col].mean())
    lift_pp = hit_rate - baseline_hit
    rows = int(len(subset))
    return {
        "market": spec.market,
        "display_market": spec.display_market,
        "target_label": spec.target_label,
        "prediction_variant": pred_col,
        "threshold": threshold,
        "rows": rows,
        "fixtures": int(subset["fixture_key"].nunique()) if "fixture_key" in subset.columns else np.nan,
        "players": int(subset["player_name"].nunique()) if "player_name" in subset.columns else np.nan,
        "teams": int(subset["team_name"].nunique()) if "team_name" in subset.columns else np.nan,
        "leagues": int(subset["league_tag"].nunique()) if "league_tag" in subset.columns else np.nan,
        "seasons": int(subset["season_tag"].nunique()) if "season_tag" in subset.columns else np.nan,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline_hit,
        "lift_pp": lift_pp,
        "mean_pred": float(num(subset[pred_col]).mean()),
        "median_pred": float(num(subset[pred_col]).median()),
        "months": int(month["eval_month"].nunique()),
        "months_ge_min_rows": int(len(month_eligible)),
        "monthly_hit_mean": float(month_eligible["month_hit_rate"].mean()) if not month_eligible.empty else np.nan,
        "monthly_hit_std": float(month_eligible["month_hit_rate"].std(ddof=0)) if len(month_eligible) > 1 else 0.0,
        "monthly_hit_min": float(month_eligible["month_hit_rate"].min()) if not month_eligible.empty else np.nan,
        "stable_month_share": stable_share,
        "recommended_beta_label": beta_label(
            rows=rows,
            hit_rate=hit_rate,
            lift_pp=lift_pp,
            months_ge_min=int(len(month_eligible)),
            stable_month_share=stable_share if not math.isnan(stable_share) else 0.0,
            spec=spec,
        ),
    }


def threshold_table(df: pd.DataFrame, spec: MarketSpec, pred_col: str) -> pd.DataFrame:
    work = df.dropna(subset=[pred_col, spec.hit_col]).copy()
    if work.empty:
        return pd.DataFrame()
    baseline_hit = float(work[spec.hit_col].mean())
    rows = []
    for threshold in THRESHOLDS:
        subset = work[num(work[pred_col]).ge(threshold)].copy()
        row = summarize_subset(subset, spec, pred_col, baseline_hit, threshold=threshold)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def league_season_threshold_table(df: pd.DataFrame, spec: MarketSpec, pred_col: str) -> pd.DataFrame:
    work = df.dropna(subset=[pred_col, spec.hit_col]).copy()
    if work.empty:
        return pd.DataFrame()
    baseline_hit = float(work[spec.hit_col].mean())
    rows = []
    group_cols = ["league_tag", "season_tag"]
    for (league, season), group in work.groupby(group_cols, dropna=False):
        for threshold in THRESHOLDS:
            subset = group[num(group[pred_col]).ge(threshold)].copy()
            if subset.empty:
                continue
            hit_rate = float(subset[spec.hit_col].mean())
            rows.append(
                {
                    "market": spec.market,
                    "prediction_variant": pred_col,
                    "league_tag": league,
                    "season_tag": season,
                    "threshold": threshold,
                    "rows": int(len(subset)),
                    "fixtures": int(subset["fixture_key"].nunique()) if "fixture_key" in subset.columns else np.nan,
                    "players": int(subset["player_name"].nunique()) if "player_name" in subset.columns else np.nan,
                    "teams": int(subset["team_name"].nunique()) if "team_name" in subset.columns else np.nan,
                    "hit_rate": hit_rate,
                    "baseline_hit_rate": baseline_hit,
                    "lift_pp": hit_rate - baseline_hit,
                    "mean_pred": float(num(subset[pred_col]).mean()),
                }
            )
    return pd.DataFrame(rows)


def top_slice_table(df: pd.DataFrame, spec: MarketSpec, pred_col: str) -> pd.DataFrame:
    work = df.dropna(subset=[pred_col, spec.hit_col]).copy()
    if work.empty:
        return pd.DataFrame()
    baseline_hit = float(work[spec.hit_col].mean())
    rows = []
    for frac in TOP_FRACTIONS:
        n = max(1, math.ceil(len(work) * frac))
        subset = work.nlargest(n, pred_col).copy()
        row = summarize_subset(subset, spec, pred_col, baseline_hit, threshold=None)
        if row:
            row["slice"] = f"top_{int(frac * 100)}pct"
            row["threshold"] = float(num(subset[pred_col]).min())
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_top_slice_table(df: pd.DataFrame, spec: MarketSpec, pred_col: str) -> pd.DataFrame:
    work = df.dropna(subset=[pred_col, spec.hit_col, "eval_month"]).copy()
    if work.empty:
        return pd.DataFrame()
    rows = []
    for month, month_df in work.groupby("eval_month", dropna=False):
        for frac in TOP_FRACTIONS:
            n = max(1, math.ceil(len(month_df) * frac))
            subset = month_df.nlargest(n, pred_col).copy()
            rows.append(
                {
                    "market": spec.market,
                    "prediction_variant": pred_col,
                    "eval_month": month,
                    "slice": f"top_{int(frac * 100)}pct",
                    "rows": int(len(subset)),
                    "hit_rate": float(subset[spec.hit_col].mean()),
                    "mean_pred": float(num(subset[pred_col]).mean()),
                    "threshold_min": float(num(subset[pred_col]).min()),
                }
            )
    return pd.DataFrame(rows)


def repeatability_table(df: pd.DataFrame, spec: MarketSpec, pred_col: str, threshold: float, entity: str) -> pd.DataFrame:
    work = df.dropna(subset=[pred_col, spec.hit_col]).copy()
    work = work[num(work[pred_col]).ge(threshold)].copy()
    if work.empty or entity not in work.columns:
        return pd.DataFrame()
    group_cols = [entity]
    if entity == "player_name" and "team_name" in work.columns:
        group_cols = ["team_name", "player_name"]
    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            market=(spec.hit_col, lambda _: spec.market),
            prediction_variant=(spec.hit_col, lambda _: pred_col),
            threshold=(spec.hit_col, lambda _: threshold),
            rows=(spec.hit_col, "size"),
            hit_rate=(spec.hit_col, "mean"),
            mean_pred=(pred_col, "mean"),
            leagues=("league_tag", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
            seasons=("season_tag", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )
    return summary[summary["rows"].ge(4)].sort_values(["hit_rate", "rows"], ascending=[False, False])


def best_threshold_for_repeatability(thresholds: pd.DataFrame, spec: MarketSpec, pred_col: str) -> float:
    if thresholds.empty:
        return 0.60
    work = thresholds[
        thresholds["prediction_variant"].eq(pred_col)
        & thresholds["recommended_beta_label"].isin(["BETA_CORE", "RESEARCH_READY", "WATCH"])
    ].copy()
    if work.empty:
        work = thresholds[thresholds["prediction_variant"].eq(pred_col)].copy()
    if work.empty:
        return 0.60
    work = work.sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False])
    label_rank = {"BETA_CORE": 0, "RESEARCH_READY": 1, "WATCH": 2, "DO_NOT_USE": 3}
    work["_rank"] = work["recommended_beta_label"].map(label_rank).fillna(9)
    work = work.sort_values(["_rank", "hit_rate", "rows"], ascending=[True, False, False])
    return float(work.iloc[0]["threshold"])


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows) if max_rows is not None else df
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


def top_by_market(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if df.empty or "market" not in df.columns:
        return df
    sort_cols = [c for c in ["market", "hit_rate", "rows"] if c in df.columns]
    work = df.sort_values(sort_cols, ascending=[True, False, False][: len(sort_cols)]).copy()
    return work.groupby("market", group_keys=False, dropna=False).head(n).reset_index(drop=True)


def write_report(
    outdir: Path,
    overall: pd.DataFrame,
    top_slices: pd.DataFrame,
    league_season: pd.DataFrame,
    player_repeat: pd.DataFrame,
    team_repeat: pd.DataFrame,
    candidate_cells: pd.DataFrame,
) -> None:
    headline_cols = [
        "market",
        "prediction_variant",
        "threshold",
        "rows",
        "hit_rate",
        "baseline_hit_rate",
        "lift_pp",
        "months_ge_min_rows",
        "stable_month_share",
        "recommended_beta_label",
    ]
    best = overall.sort_values(
        ["recommended_beta_label", "hit_rate", "rows"],
        ascending=[True, False, False],
    ).copy()
    label_rank = {"BETA_CORE": 0, "RESEARCH_READY": 1, "WATCH": 2, "DO_NOT_USE": 3}
    best["_rank"] = best["recommended_beta_label"].map(label_rank).fillna(9)
    best = best.sort_values(["_rank", "hit_rate", "rows"], ascending=[True, False, False])

    lines = [
        "# Player Event Threshold Stability Audit",
        "",
        "Research-only beta/intelligence audit. No production deploy, no priced prop odds, no slip routing.",
        "",
        "## What This Tests",
        "- Probability threshold bands such as `0.55+`, `0.60+`, `0.65+`, `0.70+`.",
        "- Market x league x season stability.",
        "- Monthly top-decile and top-5% stability.",
        "- Player/team repeatability for shortlist candidates.",
        "",
        "## Best Candidate Bands",
        markdown_table(best[headline_cols], max_rows=25),
        "",
        "## Top Slice Summary",
        markdown_table(
            top_slices[
                [
                    "market",
                    "slice",
                    "prediction_variant",
                    "threshold",
                    "rows",
                    "hit_rate",
                    "lift_pp",
                    "stable_month_share",
                    "recommended_beta_label",
                ]
            ].sort_values(["recommended_beta_label", "hit_rate"], ascending=[True, False]),
            max_rows=20,
        ),
        "",
        "## Live-Shadow Candidate Cells",
        markdown_table(
            candidate_cells[
                [
                    "market",
                    "display_market",
                    "prediction_variant",
                    "threshold",
                    "rows",
                    "hit_rate",
                    "lift_pp",
                    "stable_month_share",
                    "recommended_beta_label",
                ]
            ].sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False]),
            max_rows=30,
        ),
        "",
        "## League/Season Watch Cells",
        markdown_table(
            league_season[league_season["rows"].ge(100)]
            .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])[
                [
                    "market",
                    "prediction_variant",
                    "league_tag",
                    "season_tag",
                    "threshold",
                    "rows",
                    "hit_rate",
                    "lift_pp",
                ]
            ],
            max_rows=35,
        ),
        "",
        "## Repeatability Samples",
        "Player repeatability:",
        markdown_table(top_by_market(player_repeat, n=8), max_rows=40),
        "",
        "Team repeatability:",
        markdown_table(top_by_market(team_repeat, n=8), max_rows=40),
        "",
        "## Read",
        "- `BETA_CORE` means the band is stable enough for live-shadow priority, not production promotion.",
        "- `RESEARCH_READY` means keep auditing and consider inclusion in the forward-facing intelligence board.",
        "- `WATCH` means interesting but not boring enough.",
        "- `DO_NOT_USE` means no dashboard confidence label yet.",
        "",
        "## Next",
        "- Use `BETA_CORE` and best `RESEARCH_READY` cells for the live-shadow Player Intelligence Board.",
        "- Keep league/team repeatability visible so one hot aggregate does not hide weak local shape.",
        "- Re-run after more live boards before any user-facing confidence copy is hardened.",
    ]
    (outdir / "PLAYER_EVENT_THRESHOLD_STABILITY_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--markets",
        default="shots,shots_ge2,shots_ge3,shots_on_target,sot_ge2_attackers,sot_ge3_attackers,fouls_committed,tackles",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    requested_markets = [part.strip() for part in args.markets.split(",") if part.strip()]

    overall_frames: list[pd.DataFrame] = []
    league_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    monthly_top_frames: list[pd.DataFrame] = []
    player_repeat_frames: list[pd.DataFrame] = []
    team_repeat_frames: list[pd.DataFrame] = []

    for market in requested_markets:
        if market not in MARKETS:
            raise SystemExit(f"Unsupported market: {market}. Supported: {', '.join(MARKETS)}")
        spec = MARKETS[market]
        df = read_market(spec)
        if df.empty:
            print(f"SKIP {market}: missing or empty {spec.input_path}")
            continue
        variants = available_variants(df, spec)
        if not variants:
            print(f"SKIP {market}: no prediction variants found")
            continue

        market_thresholds: list[pd.DataFrame] = []
        for pred_col in variants:
            thresholds = threshold_table(df, spec, pred_col)
            if thresholds.empty:
                continue
            market_thresholds.append(thresholds)
            league_frames.append(league_season_threshold_table(df, spec, pred_col))
            top_frames.append(top_slice_table(df, spec, pred_col))
            monthly_top_frames.append(monthly_top_slice_table(df, spec, pred_col))

        if market_thresholds:
            market_overall = pd.concat(market_thresholds, ignore_index=True)
            overall_frames.append(market_overall)
            primary_pred = variants[0]
            repeat_threshold = best_threshold_for_repeatability(market_overall, spec, primary_pred)
            player_repeat_frames.append(repeatability_table(df, spec, primary_pred, repeat_threshold, "player_name"))
            team_repeat_frames.append(repeatability_table(df, spec, primary_pred, repeat_threshold, "team_name"))

    overall = pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame()
    league_season = pd.concat(league_frames, ignore_index=True) if league_frames else pd.DataFrame()
    top_slices = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    monthly_top = pd.concat(monthly_top_frames, ignore_index=True) if monthly_top_frames else pd.DataFrame()
    player_repeat = pd.concat(player_repeat_frames, ignore_index=True) if player_repeat_frames else pd.DataFrame()
    team_repeat = pd.concat(team_repeat_frames, ignore_index=True) if team_repeat_frames else pd.DataFrame()

    overall.to_csv(args.outdir / "player_event_threshold_bands.csv", index=False)
    league_season.to_csv(args.outdir / "player_event_league_season_threshold_bands.csv", index=False)
    top_slices.to_csv(args.outdir / "player_event_top_slice_stability.csv", index=False)
    monthly_top.to_csv(args.outdir / "player_event_monthly_top_slice_stability.csv", index=False)
    player_repeat.to_csv(args.outdir / "player_event_player_repeatability.csv", index=False)
    team_repeat.to_csv(args.outdir / "player_event_team_repeatability.csv", index=False)
    candidate_cells = overall[
        overall["recommended_beta_label"].isin(["BETA_CORE", "RESEARCH_READY"])
        & overall["rows"].ge(250)
        & overall["stable_month_share"].ge(0.70)
    ].copy()
    candidate_cells.to_csv(args.outdir / "player_event_live_shadow_candidate_cells.csv", index=False)
    write_report(args.outdir, overall, top_slices, league_season, player_repeat, team_repeat, candidate_cells)

    print(f"WROTE {args.outdir}")
    if not overall.empty:
        print(
            overall[
                [
                    "market",
                    "prediction_variant",
                    "threshold",
                    "rows",
                    "hit_rate",
                    "lift_pp",
                    "stable_month_share",
                    "recommended_beta_label",
                ]
            ]
            .sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False])
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
