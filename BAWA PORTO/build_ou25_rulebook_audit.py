#!/usr/bin/env python3
"""One-click OU25 audit scaffold (walk-forward, 3y).

Outputs a BTTS-style OU25 audit pack:
  - overall / by-pick / by-tier / by-league tables
  - elite+standard-only views
  - min-50 graded league slice
  - gate suppressor summary (from OU25 gate audits)
  - whitelist/blacklist suggestions
  - feature deltas (wins vs losses) for core separators

Default input:
  predictions_output/walk_forward/w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv
Default output:
  predictions_output/walk_forward/_MASTER/OU25_3Y_RULEBOOK_AUDIT
"""

from __future__ import annotations

import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("predictions_output/walk_forward")
DEFAULT_OUT = DEFAULT_BASE / "_MASTER" / "OU25_3Y_RULEBOOK_AUDIT"
DEFAULT_MD = DEFAULT_OUT / "OU25_3Y_RULEBOOK_AUDIT.md"

# Heuristic thresholds for suggestions (tweak as needed)
WHITELIST_MIN_GRADED = 80
WHITELIST_MIN_HIT = 0.70
WHITELIST_MIN_ROI = 0.20

BLACKLIST_MIN_GRADED = 80
BLACKLIST_MAX_HIT = 0.55
BLACKLIST_MAX_ROI = -0.02

FEATURES_CORE = [
    "exp_goals_sum",
    "p00_est",
    "cs_home",
    "cs_away",
    "cs_max",
    "cs_sum",
    "model_p_for_bookie",
    "bookie_od",
    "bookie_implied",
    "btts_top3_mass",
    "fts_sum",
    "xg_sum_pre_match",
    "bookie_lambda_total_fit",
    "home_ge2_confidence",
    "away_ge2_confidence",
    "scored_rate_5_away",
    "xg_for_avg_5_away",
    "clean_sheet_rate_5_away",
]

# Policy groups (kept in sync with deploy_rulebook.py)
OU25_OVER25_TIER1_LEAGUES = {
    "Netherlands Eredivisie",
    "Belgium Pro",
    "Champions League",
    "Europa League",
    "England FA Cup",
}

OU25_OVER25_TIER2_LEAGUES = {
    "Norway Eliteserien",
    "Spain La Liga",
    "Japan J1",
    "USA MLS",
    "Europa Conference",
}

OU25_OVER25_BASELINE_ONLY_LEAGUES = {
    "England Championship",
    "Portugal Liga",
    "Brazil Serie A",
}

OU25_OVER25_BLACKLIST_LEAGUES = {
    "Austria Bundesliga",
    "Australia A-League",
    "Czech First League",
    "Denmark Superliga",
    "South Korea K League",
    "Swiss Super League",
}

SWEEP_THRESHOLDS = {
    "exp_goals_sum": [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6],
    "p00_est": [0.08, 0.06, 0.05, 0.04, 0.035],
    "cs_max": [0.40, 0.35, 0.30, 0.25, 0.22],
    "cs_sum": [0.80, 0.70, 0.60, 0.50, 0.45, 0.40],
    "btts_top3_mass": [0.12, 0.15, 0.18, 0.20, 0.22],
    "fts_sum": [0.45, 0.50, 0.55, 0.60],
    "xg_sum_pre_match": [2.2, 2.4, 2.6, 2.8, 3.0],
    "bookie_lambda_total_fit": [2.6, 2.8, 3.0, 3.2],
}

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_pick(df: pd.DataFrame) -> pd.Series:
    pick = df.get("bookie_pick", df.get("selection", pd.Series("", index=df.index)))
    return pick.astype("string").fillna("").str.upper().str.strip()


def _profit_series(hit: pd.Series, bookie_od: pd.Series) -> pd.Series:
    h = _safe_num(hit)
    od = _safe_num(bookie_od)
    return np.where(h == 1, od - 1.0, np.where(h == 0, -1.0, np.nan))


def _summarize(df: pd.DataFrame, hit_col: str) -> dict[str, object]:
    hit = _safe_num(df.get(hit_col, np.nan))
    graded = hit.notna().sum()
    wins = (hit == 1).sum()
    losses = (hit == 0).sum()
    profit = _profit_series(hit, df.get("bookie_od", np.nan))
    profit_sum = float(np.nansum(profit))
    roi = float(profit_sum / graded) if graded else float("nan")
    return {
        "rows": int(len(df)),
        "graded": int(graded),
        "wins": float(wins),
        "losses": float(losses),
        "hit_rate": float(wins / graded) if graded else float("nan"),
        "roi": roi,
        "profit": profit_sum,
        "avg_bookie_od": float(_safe_num(df.get("bookie_od", np.nan)).mean()),
        "avg_model_p": float(_safe_num(df.get("model_p_for_bookie", np.nan)).mean()),
    }


def _group_summary(df: pd.DataFrame, hit_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update(_summarize(g, hit_col))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "graded" in out.columns:
        out = out.sort_values(["graded", "hit_rate", "roi"], ascending=[False, False, False], na_position="last")
    return out.reset_index(drop=True)


def _feature_deltas(df: pd.DataFrame, hit_col: str, features: list[str]) -> pd.DataFrame:
    hit = _safe_num(df.get(hit_col, np.nan))
    win = hit == 1
    loss = hit == 0
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        s = _safe_num(df[f])
        if s.notna().sum() == 0:
            continue
        rows.append({
            "feature": f,
            "win_mean": float(s[win].mean()),
            "loss_mean": float(s[loss].mean()),
            "delta": float(s[win].mean() - s[loss].mean()),
            "count_win": int(win.sum()),
            "count_loss": int(loss.sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _add_balanced_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    graded = pd.to_numeric(out.get("graded", np.nan), errors="coerce").fillna(0.0)
    hit_rate = pd.to_numeric(out.get("hit_rate", np.nan), errors="coerce").fillna(0.0)
    out["balanced_score"] = hit_rate * np.log1p(graded)
    return out


def _league_threshold_sweeps(df: pd.DataFrame, hit_col: str, pick: str) -> pd.DataFrame:
    sub = df.loc[df["bookie_pick"].eq(pick)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for league, g in sub.groupby("league", dropna=False):
        if _safe_num(g.get(hit_col, np.nan)).notna().sum() < 50:
            continue
        for feat, thresh_list in SWEEP_THRESHOLDS.items():
            if feat not in g.columns:
                continue
            s = _safe_num(g[feat])
            for t in thresh_list:
                if feat in {"p00_est", "cs_max", "cs_sum"}:
                    m = s <= float(t)
                    op = "<="
                else:
                    m = s >= float(t)
                    op = ">="
                rule = f"{feat}{op}{t:.3f}".rstrip("0").rstrip(".")
                gg = g.loc[m].copy()
                if gg.empty:
                    continue
                rows.append({
                    "league": league,
                    "pick": pick,
                    "feature": feat,
                    "op": op,
                    "threshold": float(t),
                    "rule": rule,
                    **_summarize(gg, hit_col),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _add_balanced_score(out)
        out = out.sort_values(
            ["league", "balanced_score", "roi", "hit_rate", "graded"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    return out


def _topn_per_league_feature(sweeps: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if sweeps.empty:
        return sweeps
    ranked = sweeps.sort_values(
        ["league", "feature", "balanced_score", "roi", "hit_rate", "graded"],
        ascending=[True, True, False, False, False, False],
    )
    out = ranked.groupby(["league", "feature"], dropna=False).head(top_n).reset_index(drop=True)
    return out


def _plot_sweep_curves(sweeps: pd.DataFrame, outdir: Path, pick: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if sweeps.empty:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (league, feature), g in sweeps.groupby(["league", "feature"], dropna=False):
        if g.empty:
            continue
        g = g.copy()
        g["threshold"] = pd.to_numeric(g["threshold"], errors="coerce")
        g = g.sort_values("threshold")
        if g["threshold"].notna().sum() < 2:
            continue
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(g["threshold"], g["hit_rate"], marker="o", label="hit_rate")
        ax1.set_xlabel(f"{feature} threshold ({g['op'].iloc[0]})")
        ax1.set_ylabel("hit_rate")
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.plot(g["threshold"], g["roi"], marker="x", color="tab:red", label="roi")
        ax2.set_ylabel("roi")
        title = f"OU25 {pick} | {league} | {feature}"
        plt.title(title)
        fig.tight_layout()
        fname = f"OU25_SWEEP_{pick}_{league}_{feature}.png".replace(" ", "_")
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)


def _extract_window_id(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("")
    win = s.str.extract(r"(w\d+_\d{4}_\d{2}_\d{2}_\d{4}_\d{2}_\d{2})", expand=False)
    return win.fillna(s)


def _best_feature_for_group(g: pd.DataFrame, hit_col: str) -> str:
    rows = []
    for feat, thresh_list in SWEEP_THRESHOLDS.items():
        if feat not in g.columns:
            continue
        s = _safe_num(g[feat])
        for t in thresh_list:
            if feat in {"p00_est", "cs_max", "cs_sum"}:
                m = s <= float(t)
            else:
                m = s >= float(t)
            gg = g.loc[m]
            if gg.empty:
                continue
            row = _summarize(gg, hit_col)
            row["feature"] = feat
            rows.append(row)
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    df = _add_balanced_score(df)
    df = df.sort_values(["balanced_score", "roi", "hit_rate", "graded"], ascending=[False, False, False, False])
    return str(df.iloc[0]["feature"])


def _rule_stability(df: pd.DataFrame, hit_col: str, pick: str) -> pd.DataFrame:
    if "__src" not in df.columns:
        return pd.DataFrame()
    sub = df.loc[df["bookie_pick"].eq(pick)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["__window"] = _extract_window_id(sub["__src"])

    rows = []
    for (window_id, league), g in sub.groupby(["__window", "league"], dropna=False):
        if _safe_num(g.get(hit_col, np.nan)).notna().sum() < 20:
            continue
        feat = _best_feature_for_group(g, hit_col)
        if not feat:
            continue
        rows.append({"league": league, "window_id": window_id, "feature": feat})

    if not rows:
        return pd.DataFrame()
    dfw = pd.DataFrame(rows)
    counts = (
        dfw.groupby(["league", "feature"], dropna=False)
        .size()
        .reset_index(name="wins")
    )
    total = dfw.groupby(["league"], dropna=False).size().reset_index(name="windows")
    out = counts.merge(total, on="league", how="left")
    out["share"] = out["wins"] / out["windows"].replace(0, np.nan)
    out = out.sort_values(["league", "share", "wins"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def _rule_stability_by_threshold(df: pd.DataFrame, hit_col: str, pick: str) -> pd.DataFrame:
    if "__src" not in df.columns:
        return pd.DataFrame()
    sub = df.loc[df["bookie_pick"].eq(pick)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["__window"] = _extract_window_id(sub["__src"])

    rows = []
    for (window_id, league), g in sub.groupby(["__window", "league"], dropna=False):
        if _safe_num(g.get(hit_col, np.nan)).notna().sum() < 20:
            continue
        candidates = []
        for feat, thresh_list in SWEEP_THRESHOLDS.items():
            if feat not in g.columns:
                continue
            s = _safe_num(g[feat])
            for t in thresh_list:
                if feat in {"p00_est", "cs_max", "cs_sum"}:
                    m = s <= float(t)
                    op = "<="
                else:
                    m = s >= float(t)
                    op = ">="
                gg = g.loc[m]
                if gg.empty:
                    continue
                row = _summarize(gg, hit_col)
                row["feature"] = feat
                row["threshold"] = float(t)
                row["op"] = op
                candidates.append(row)
        if not candidates:
            continue
        cand = pd.DataFrame(candidates)
        cand = _add_balanced_score(cand)
        cand = cand.sort_values(["balanced_score", "roi", "hit_rate", "graded"], ascending=[False, False, False, False])
        top = cand.iloc[0]
        rows.append({
            "league": league,
            "window_id": window_id,
            "feature": str(top["feature"]),
            "op": str(top["op"]),
            "threshold": float(top["threshold"]),
        })

    if not rows:
        return pd.DataFrame()
    dfw = pd.DataFrame(rows)
    dfw["rule"] = dfw["feature"] + dfw["op"] + dfw["threshold"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip("."))
    counts = (
        dfw.groupby(["league", "rule"], dropna=False)
        .size()
        .reset_index(name="wins")
    )
    total = dfw.groupby(["league"], dropna=False).size().reset_index(name="windows")
    out = counts.merge(total, on="league", how="left")
    out["share"] = out["wins"] / out["windows"].replace(0, np.nan)
    out = out.sort_values(["league", "share", "wins"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def _write_md_report(outdir: Path) -> None:
    def _read(name: str) -> pd.DataFrame:
        p = outdir / name
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    overall = _read("OU25_3Y_RULEBOOK_AUDIT__OVERALL.csv")
    by_pick = _read("OU25_3Y_RULEBOOK_AUDIT__BY_PICK.csv")
    by_tier = _read("OU25_3Y_RULEBOOK_AUDIT__BY_TIER.csv")
    by_league = _read("OU25_3Y_RULEBOOK_AUDIT__BY_LEAGUE.csv")
    min50 = _read("OU25_3Y_RULEBOOK_AUDIT__MIN50_BY_LEAGUE.csv")
    es_overall = _read("OU25_3Y_ELITE_STANDARD_ONLY__OVERALL.csv")
    es_league = _read("OU25_3Y_ELITE_STANDARD_ONLY__BY_LEAGUE.csv")
    sugg = _read("OU25_WHITELIST_BLACKLIST_SUGGESTIONS.csv")
    gates = _read("OU25_GATE_SUPPRESSORS__SUMMARY.csv")
    best_over = _read("OU25_LEAGUE_SWEEPS_BEST__OVER25.csv")
    best_under = _read("OU25_LEAGUE_SWEEPS_BEST__UNDER25.csv")
    top_over = _read("OU25_LEAGUE_SWEEPS_TOPN__OVER25.csv")
    top_under = _read("OU25_LEAGUE_SWEEPS_TOPN__UNDER25.csv")
    best_overall_over = _read("OU25_LEAGUE_SWEEPS_BEST_OVERALL__OVER25.csv")
    best_overall_under = _read("OU25_LEAGUE_SWEEPS_BEST_OVERALL__UNDER25.csv")
    recs = _read("OU25_LEAGUE_GATE_RECOMMENDATIONS.csv")
    stability_over = _read("OU25_RULE_STABILITY__OVER25.csv")
    stability_under = _read("OU25_RULE_STABILITY__UNDER25.csv")
    stability_over_rule = _read("OU25_RULE_STABILITY_RULE__OVER25.csv")
    stability_under_rule = _read("OU25_RULE_STABILITY_RULE__UNDER25.csv")
    best_overall_both = _read("OU25_LEAGUE_SWEEPS_BEST_OVERALL__BOTH.csv")
    policy_summary = _read("OU25_OVER25_POLICY_PASS_SUMMARY.csv")
    policy_by_league = _read("OU25_OVER25_POLICY_PASS_SUMMARY__BY_LEAGUE.csv")
    policy_by_tier = _read("OU25_OVER25_POLICY_PASS_SUMMARY__BY_TIER.csv")
    policy_by_group = _read("OU25_OVER25_POLICY_PASS_SUMMARY__BY_POLICY_GROUP.csv")
    policy_perf_group = _read("OU25_OVER25_POLICY_PERF__BY_POLICY_GROUP.csv")
    policy_perf_league = _read("OU25_OVER25_POLICY_PERF__BY_POLICY_GROUP_LEAGUE.csv")
    policy_perf_tag = _read("OU25_OVER25_POLICY_PERF__BY_PASS_TAG.csv")
    policy_perf_tag_league = _read("OU25_OVER25_POLICY_PERF__BY_PASS_TAG_LEAGUE.csv")
    policy_live_only = _read("OU25_OVER25_LIVE_ONLY_PERF.csv")
    live_top_roi = _read("OU25_OVER25_LIVE_TOP_ROI_BY_LEAGUE.csv")
    live_tier_heat = _read("OU25_OVER25_LIVE_TIER_HEATMAP_BY_LEAGUE.csv")
    policy_summary_export = _read("OU25_OVER25_POLICY_CONFIG_EXPORT.csv")
    live_leagues_final = _read("OU25_OVER25_LIVE_LEAGUES_FINAL.csv")
    btts_overlap = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP.csv")
    btts_overlap_by_league = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__BY_LEAGUE.csv")
    btts_overlap_live_only = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__LIVE_ONLY.csv")
    btts_overlap_live_only_league = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__LIVE_ONLY_BY_LEAGUE.csv")
    btts_overlap_top_lift = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__TOP_LIFT_BY_LEAGUE.csv")
    btts_overlap_allow = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__ALLOWLIST.csv")
    btts_overlap_allow_thresh = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__ALLOWLIST_THRESHOLDS.csv")
    btts_overlap_by_tier = _read("OU25_OVER25_BTTS_YES_LIVE_OVERLAP__BY_OU25_TIER.csv")
    btts_rescue_audit = _read("OU25_OVER25_BTTS_RESCUE_AUDIT.csv")
    btts_rescue_audit_league = _read("OU25_OVER25_BTTS_RESCUE_AUDIT__BY_LEAGUE.csv")
    btts_rescue_combined = _read("OU25_OVER25_BTTS_RESCUE_COMBINED_AUDIT.csv")
    btts_rescue_combined_league = _read("OU25_OVER25_BTTS_RESCUE_COMBINED_AUDIT__BY_LEAGUE.csv")
    btts_rescue_watch_audit = _read("OU25_OVER25_BTTS_RESCUE_WATCHLIST_CONDITION_AUDIT.csv")
    btts_rescue_watch_best = _read("OU25_OVER25_BTTS_RESCUE_WATCHLIST_BEST_RULES.csv")
    policy_ranked_roi = _read("OU25_OVER25_POLICY_GROUP_RANKS__TOP_ROI.csv")
    policy_ranked_hit = _read("OU25_OVER25_POLICY_GROUP_RANKS__TOP_HIT.csv")
    policy_ranked_profit = _read("OU25_OVER25_POLICY_GROUP_RANKS__TOP_PROFIT.csv")
    join_strategies = _read("OU25_DEPLOY_TO_SCORED_JOIN_STRATEGIES.csv")
    merge_audit = _read("OU25_DEPLOY_TO_SCORED_MERGE_AUDIT.csv")

    def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "_(no data)_"

        head = df.head(max_rows).copy()

        try:
            return head.to_markdown(index=False)
        except Exception:
            cols = [str(c) for c in head.columns]

            def _fmt(v: object) -> str:
                if pd.isna(v):
                    return ""
                if isinstance(v, (float, np.floating)):
                    return f"{float(v):.6f}".rstrip("0").rstrip(".")
                return str(v)

            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            body = [
                "| " + " | ".join(_fmt(v) for v in row) + " |"
                for row in head.itertuples(index=False, name=None)
            ]
            return "\n".join([header, sep, *body])

    lines = []
    lines.append("# OU25 3Y Rulebook Audit")
    lines.append("")
    lines.append("## Overall")
    lines.append(_md_table(overall, 10))
    lines.append("")
    lines.append("## By Pick")
    lines.append(_md_table(by_pick, 10))
    lines.append("")
    lines.append("## By Tier")
    lines.append(_md_table(by_tier, 10))
    lines.append("")
    lines.append("## Top Leagues (All)")
    if not by_league.empty:
        by_league = by_league.sort_values(["graded", "roi"], ascending=[False, False])
    lines.append(_md_table(by_league, 25))
    lines.append("")
    lines.append("## Min-50 Leagues (All)")
    lines.append(_md_table(min50, 50))
    lines.append("")
    lines.append("## Elite+Standard Only (Overall)")
    lines.append(_md_table(es_overall, 10))
    lines.append("")
    lines.append("## Elite+Standard Only (Leagues)")
    lines.append(_md_table(es_league, 25))
    lines.append("")
    lines.append("## Whitelist / Blacklist Suggestions (OVER25)")
    lines.append(_md_table(sugg, 200))
    lines.append("")
    lines.append("## Gate Suppressors (All Windows)")
    lines.append(_md_table(gates, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Pass Summary")
    lines.append("Note: policy tag counts are sourced from deploy outputs (`02_deploy`). Tagged performance uses scored outputs (`03_scored`) after a deploy→scored join; merge quality is reported in `OU25_DEPLOY_TO_SCORED_MERGE_AUDIT.csv`.")
    if not merge_audit.empty:
        sel = merge_audit.loc[merge_audit["metric"].eq("selected_strategy"), "value"]
        rate = merge_audit.loc[merge_audit["metric"].eq("merge_rate_pct"), "value"]
        sel_str = sel.iloc[0] if not sel.empty else "UNKNOWN"
        rate_str = rate.iloc[0] if not rate.empty else "UNKNOWN"
        lines.append(f"Selected join strategy: `{sel_str}` | Merge rate: `{rate_str}%`")
        lines.append("Join strategy audit: `OU25_DEPLOY_TO_SCORED_JOIN_STRATEGIES.csv`")
        lines.append("Mismatch sample: `OU25_DEPLOY_TO_SCORED_UNMATCHED_SAMPLE.csv`")
    lines.append(_md_table(policy_summary, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Pass Summary by League")
    lines.append(_md_table(policy_by_league, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Pass Summary by Tier")
    lines.append(_md_table(policy_by_tier, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Pass Summary by Policy Group")
    lines.append(_md_table(policy_by_group, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Performance by Policy Group")
    lines.append(_md_table(policy_perf_group, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Performance by Policy Group + League")
    lines.append(_md_table(policy_perf_league, 100))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Performance by Pass Tag (Tagged Scored Rows)")
    lines.append(_md_table(policy_perf_tag, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Performance by Pass Tag + League (Tagged Scored Rows)")
    lines.append(_md_table(policy_perf_tag_league, 100))
    lines.append("")
    lines.append("## OU25 OVER25 Live vs Observe vs Blacklist (Tagged Scored Rows)")
    lines.append(_md_table(policy_live_only, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Live-Only Top ROI by League (Tagged Scored Rows)")
    lines.append(_md_table(live_top_roi, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Tier1 vs Tier2 vs Fallback Heatmap (Tagged Scored Rows)")
    lines.append(_md_table(live_tier_heat, 100))
    lines.append("")
    lines.append("## OU25 OVER25 Policy Config Export")
    lines.append(_md_table(policy_summary_export, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Live Leagues (Final, Appeared in Tagged Data)")
    lines.append(_md_table(live_leagues_final, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Conditional on BTTS-YES Live (Overlap Audit)")
    lines.append(_md_table(btts_overlap, 20))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS-YES Rescue Lane Summary")
    summary_rows = []
    if not btts_overlap.empty:
        row = btts_overlap.iloc[0].to_dict()
        row["scope"] = "GLOBAL_OVERLAP"
        summary_rows.append(row)
    if not btts_overlap_live_only.empty:
        row = btts_overlap_live_only.iloc[0].to_dict()
        row["scope"] = "LIVE_ONLY_OVERLAP"
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ["scope"] + [c for c in summary_df.columns if c != "scope"]
        summary_df = summary_df[cols]
        lines.append(_md_table(summary_df, 10))
    else:
        lines.append("_(no data)_")
    lines.append("")
    if not btts_overlap_allow.empty:
        allow_count = int(len(btts_overlap_allow))
        thresh = {}
        if not btts_overlap_allow_thresh.empty:
            for k in ["min_overlap_rows", "min_roi_lift", "min_overlap_roi"]:
                if k in btts_overlap_allow_thresh.columns:
                    thresh[k] = btts_overlap_allow_thresh.iloc[0][k]
        meta_rows = [{"metric": "allowlist_count", "value": allow_count}]
        for k, v in thresh.items():
            meta_rows.append({"metric": k, "value": v})
        lines.append(_md_table(pd.DataFrame(meta_rows), 10))
        lines.append("")
    lines.append("## OU25 OVER25 Conditional on BTTS-YES Live (By League)")
    lines.append(_md_table(btts_overlap_by_league, 50))
    lines.append("")
    lines.append("## OU25 OVER25 Conditional on BTTS-YES Live (Live-Only)")
    lines.append(_md_table(btts_overlap_live_only, 20))
    lines.append("")
    lines.append("## OU25 OVER25 Conditional on BTTS-YES Live (Live-Only by League)")
    lines.append(_md_table(btts_overlap_live_only_league, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS-YES Overlap — Top ROI Lift by League")
    lines.append(_md_table(btts_overlap_top_lift, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS-YES Overlap — Recommended Allowlist")
    lines.append(_md_table(btts_overlap_allow, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS-YES Overlap — Allowlist Thresholds")
    lines.append(_md_table(btts_overlap_allow_thresh, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — Summary")
    lines.append(_md_table(btts_rescue_audit, 20))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — By League")
    lines.append(_md_table(btts_rescue_audit_league, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — Combined Live Product")
    lines.append(_md_table(btts_rescue_combined, 20))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — Combined Live Product by League")
    lines.append(_md_table(btts_rescue_combined_league, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — Watchlist Condition Audit")
    lines.append(_md_table(btts_rescue_watch_audit, 50))
    lines.append("")
    lines.append("## OU25 OVER25 BTTS Rescue Lane — Watchlist Best Rule per League")
    lines.append(_md_table(btts_rescue_watch_best, 20))
    lines.append("")

    # Compact production summary
    lines.append("## OU25 Production Summary (BTTS Rescue)")
    prod_rows = []
    if not btts_rescue_combined.empty:
        try:
            base = btts_rescue_combined.loc[btts_rescue_combined["cohort"].eq("existing_live_only")].head(1)
            comb = btts_rescue_combined.loc[btts_rescue_combined["cohort"].eq("combined_live_plus_rescued")].head(1)
            resc = btts_rescue_combined.loc[btts_rescue_combined["cohort"].eq("rescued_only")].head(1)
            if not base.empty:
                prod_rows.append({"metric": "baseline_live_rows", "value": base.iloc[0].get("rows", "")})
                prod_rows.append({"metric": "baseline_live_hit_rate", "value": base.iloc[0].get("hit_rate", "")})
                prod_rows.append({"metric": "baseline_live_roi", "value": base.iloc[0].get("roi", "")})
            if not resc.empty:
                prod_rows.append({"metric": "rescued_rows", "value": resc.iloc[0].get("rows", "")})
                prod_rows.append({"metric": "rescued_hit_rate", "value": resc.iloc[0].get("hit_rate", "")})
                prod_rows.append({"metric": "rescued_roi", "value": resc.iloc[0].get("roi", "")})
            if not comb.empty:
                prod_rows.append({"metric": "combined_rows", "value": comb.iloc[0].get("rows", "")})
                prod_rows.append({"metric": "combined_hit_rate", "value": comb.iloc[0].get("hit_rate", "")})
                prod_rows.append({"metric": "combined_roi", "value": comb.iloc[0].get("roi", "")})
                prod_rows.append({"metric": "combined_delta_hit_vs_live", "value": comb.iloc[0].get("delta_hit_rate_vs_existing_live", "")})
                prod_rows.append({"metric": "combined_delta_roi_vs_live", "value": comb.iloc[0].get("delta_roi_vs_existing_live", "")})
        except Exception:
            pass
    lines.append(_md_table(pd.DataFrame(prod_rows), 20) if prod_rows else "_(no data)_")
    lines.append("")

    # Write final rulebook markdown export (after core sections are available)
    try:
        final_lines = []
        final_lines.append("# OU25 OVER25 BTTS Rescue Rulebook — Final")
        final_lines.append("")
        final_lines.append("## Core Rescue Allowlist")
        final_lines.append(_md_table(btts_overlap_allow, 50))
        final_lines.append("")
        final_lines.append("## Watchlist Rescue Rules (Best per League)")
        final_lines.append(_md_table(btts_rescue_watch_best, 20))
        final_lines.append("")
        final_lines.append("## Production Summary")
        final_lines.append(_md_table(pd.DataFrame(prod_rows), 20) if prod_rows else "_(no data)_")
        final_lines.append("")
        final_lines.append("Notes:")
        final_lines.append("- Rescue lane promotes OU25 OVER25 from OBSERVE to STANDARD only.")
        final_lines.append("- BLACKLIST, BASELINE_ONLY, and REVIEW remain protected.")
        final_lines.append("- Denylist tagging persists for overlap rows failing watchlist rules.")
        (outdir / "OU25_OVER25_BTTS_RESCUE_RULEBOOK_FINAL.md").write_text(
            "\n".join(final_lines),
            encoding="utf-8",
        )
    except Exception:
        pass
    lines.append("## OU25 BTTS Rescue Rulebook — Final")
    final_md_path = outdir / "OU25_OVER25_BTTS_RESCUE_RULEBOOK_FINAL.md"
    try:
        final_md_text = final_md_path.read_text(encoding="utf-8")
        lines.append(final_md_text)
    except Exception:
        lines.append("_(no final rulebook export found)_")
    lines.append("")
    lines.append("## OU25 OVER25 BTTS-YES Overlap — Split by OU25 Tier")
    lines.append(_md_table(btts_overlap_by_tier, 20))
    lines.append("")
    lines.append("## Policy Group Rankings — Top ROI")
    lines.append(_md_table(policy_ranked_roi, 100))
    lines.append("")
    lines.append("## Policy Group Rankings — Top Hit Rate")
    lines.append(_md_table(policy_ranked_hit, 100))
    lines.append("")
    lines.append("## Policy Group Rankings — Top Profit")
    lines.append(_md_table(policy_ranked_profit, 100))
    lines.append("")
    lines.append("## Best Sweep Rules by League (OVER25)")
    lines.append(_md_table(best_over, 50))
    lines.append("")
    lines.append("## Best Sweep Rules by League (UNDER25)")
    lines.append(_md_table(best_under, 50))
    lines.append("")
    lines.append("## Best Overall Rule per League (OVER25)")
    lines.append(_md_table(best_overall_over, 50))
    lines.append("")
    lines.append("## Best Overall Rule per League (UNDER25)")
    lines.append(_md_table(best_overall_under, 50))
    lines.append("")
    lines.append("## League Gate Recommendations (OVER + UNDER)")
    lines.append(_md_table(recs, 100))
    lines.append("")
    lines.append("## Best Overall Rule per League (Both Picks)")
    lines.append(_md_table(best_overall_both, 50))
    lines.append("")
    lines.append("## Top Sweep Rules by League (OVER25)")
    lines.append(_md_table(top_over, 100))
    lines.append("")
    lines.append("## Top Sweep Rules by League (UNDER25)")
    lines.append(_md_table(top_under, 100))
    lines.append("")
    lines.append("## Rule Stability (OVER25)")
    lines.append(_md_table(stability_over, 100))
    lines.append("")
    lines.append("## Rule Stability (UNDER25)")
    lines.append(_md_table(stability_under, 100))
    lines.append("")
    lines.append("## Rule Stability by Threshold (OVER25)")
    lines.append(_md_table(stability_over_rule, 100))
    lines.append("")
    lines.append("## Rule Stability by Threshold (UNDER25)")
    lines.append(_md_table(stability_under_rule, 100))
    lines.append("")

    (outdir / "OU25_3Y_RULEBOOK_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _collect_scored_paths(base: Path) -> list[Path]:
    return sorted(base.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def _load_scored(base: Path) -> pd.DataFrame:
    paths = _collect_scored_paths(base)
    if not paths:
        raise SystemExit(f"No scored DEPLOY_COMBINED files found under {base}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = str(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("Failed to load any scored files.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _collect_deploy_paths(base: Path) -> list[Path]:
    return sorted(base.glob("w*/02_deploy/BOOKIE_*__FTR_accuracy.csv"))


def _load_deploy(base: Path) -> pd.DataFrame:
    paths = _collect_deploy_paths(base)
    if not paths:
        raise SystemExit(f"No deploy files found under {base}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = str(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("Failed to load any deploy files.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _ensure_cs_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cs_home" not in out.columns:
        out["cs_home"] = np.nan
    if "cs_away" not in out.columns:
        out["cs_away"] = np.nan
    ch = _safe_num(out["cs_home"])
    ca = _safe_num(out["cs_away"])
    if "cs_max" not in out.columns:
        out["cs_max"] = pd.concat([ch, ca], axis=1).max(axis=1)
    if "cs_sum" not in out.columns:
        out["cs_sum"] = ch + ca
    return out


def _norm_str(x: pd.Series) -> pd.Series:
    return x.astype("string").fillna("").str.strip()


def _norm_market(x: pd.Series) -> pd.Series:
    return _norm_str(x).str.lower()


def _norm_pick(x: pd.Series) -> pd.Series:
    return _norm_str(x).str.upper()


def _norm_ou25_pick(df: pd.DataFrame) -> pd.Series:
    bp = _norm_pick(df.get("bookie_pick", pd.Series("", index=df.index)))
    sel = _norm_pick(df.get("selection", pd.Series("", index=df.index)))
    pick = bp.where(bp.ne(""), sel)
    pick = pick.replace({"OVER": "OVER25", "UNDER": "UNDER25"})
    return pick


def _suggest_whitelist_blacklist(league_df: pd.DataFrame) -> pd.DataFrame:
    if league_df.empty:
        return league_df
    out = league_df.copy()
    out["suggestion"] = ""
    out.loc[
        (out["graded"] >= WHITELIST_MIN_GRADED)
        & (out["hit_rate"] >= WHITELIST_MIN_HIT)
        & (out["roi"] >= WHITELIST_MIN_ROI),
        "suggestion",
    ] = "WHITELIST"
    out.loc[
        (out["graded"] >= BLACKLIST_MIN_GRADED)
        & ((out["hit_rate"] <= BLACKLIST_MAX_HIT) | (out["roi"] <= BLACKLIST_MAX_ROI)),
        "suggestion",
    ] = "BLACKLIST"
    out.loc[out["suggestion"].eq(""), "suggestion"] = "BASELINE_ONLY"
    return out


def _gate_suppressors(base: Path, outdir: Path) -> None:
    all_path = base / "_MASTER" / "OU25_GATE_AUDIT__ALL_WINDOWS.csv"
    summary_path = base / "_MASTER" / "OU25_GATE_AUDIT__SUMMARY.csv"
    if not all_path.exists():
        return
    all_df = pd.read_csv(all_path)
    gate_cols = [
        "ou25_over_signal_fail",
        "ou25_over_model_floor_fail",
        "ou25_over_struct_combo_fail",
        "ou25_over_top3_fail",
        "ou25_over_one_sided_veto",
        "ou25_under_model_floor_fail",
        "ou25_under_struct_fail",
        "ou25_under_low_goal_fail",
    ]
    rows = []
    for c in gate_cols:
        if c in all_df.columns:
            rows.append({"gate": c, "count": int(pd.to_numeric(all_df[c], errors="coerce").fillna(0).sum())})
    if rows:
        out = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        out.to_csv(outdir / "OU25_GATE_SUPPRESSORS__SUMMARY.csv", index=False)
    if summary_path.exists():
        try:
            pd.read_csv(summary_path).to_csv(outdir / "OU25_GATE_AUDIT__SUMMARY.csv", index=False)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OU25 3Y rulebook audit scaffold")
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--elite-standard-only", action="store_true", help="Restrict to ELITE+STANDARD only")
    ap.add_argument("--top-n", type=int, default=3, help="Top-N sweep rules per league+feature (default: 3)")
    ap.add_argument("--plots", action="store_true", help="Write sweep curve PNGs (requires matplotlib)")
    ap.add_argument("--btts-allow-min-rows", type=int, default=50, help="BTTS overlap allowlist min graded rows (default: 50)")
    ap.add_argument("--btts-allow-min-roi-lift", type=float, default=0.15, help="BTTS overlap allowlist min ROI lift (default: 0.15)")
    ap.add_argument("--btts-allow-min-overlap-roi", type=float, default=0.05, help="BTTS overlap allowlist min overlap ROI (default: 0.05)")
    ap.add_argument("--btts-allowlist-json", type=str, default="", help="Optional custom path for BTTS rescue allowlist JSON")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_all = _load_scored(base)
    df = df_all.loc[df_all.get("market", "").astype(str).str.lower().str.strip().eq("ou25")].copy()
    if df.empty:
        raise SystemExit("No OU25 rows found in scored deploy files.")

    df_all["bookie_pick"] = _normalize_pick(df_all)
    df["bookie_pick"] = _normalize_pick(df)
    df_all["market_norm"] = _norm_market(df_all.get("market", pd.Series("", index=df_all.index)))
    df_all["league_norm"] = _norm_str(df_all.get("league", pd.Series("", index=df_all.index)))
    df_all["fixture_key_norm"] = _norm_str(df_all.get("fixture_key", pd.Series("", index=df_all.index)))
    df_all["fixture_key_ascii_norm"] = _norm_str(df_all.get("fixture_key_ascii", pd.Series("", index=df_all.index)))
    df_all["pick_norm"] = _norm_pick(df_all.get("bookie_pick", pd.Series("", index=df_all.index)))
    df_all["pick_ou25_norm"] = _norm_ou25_pick(df_all)
    df["market_norm"] = _norm_market(df.get("market", pd.Series("", index=df.index)))
    df["league_norm"] = _norm_str(df.get("league", pd.Series("", index=df.index)))
    df["fixture_key_norm"] = _norm_str(df.get("fixture_key", pd.Series("", index=df.index)))
    df["fixture_key_ascii_norm"] = _norm_str(df.get("fixture_key_ascii", pd.Series("", index=df.index)))
    df["pick_norm"] = _norm_pick(df.get("bookie_pick", pd.Series("", index=df.index)))
    df["pick_ou25_norm"] = _norm_ou25_pick(df)
    hit_col = "ou25_hit" if "ou25_hit" in df.columns else ("hit" if "hit" in df.columns else "")
    if not hit_col:
        raise SystemExit("No OU25 hit column found (expected 'ou25_hit' or 'hit').")

    tier_col = "deploy_tier" if "deploy_tier" in df.columns else ("tier" if "tier" in df.columns else "")
    if not tier_col:
        df["deploy_tier"] = ""
        tier_col = "deploy_tier"

    df = _ensure_cs_cols(df)

    deploy_df = None
    deploy_tags = None
    deploy_over_dedup = None
    try:
        deploy_df = _load_deploy(base)
        deploy_df = deploy_df.loc[
            deploy_df.get("market", "").astype(str).str.lower().str.strip().eq("ou25")
        ].copy()
        if not deploy_df.empty:
            deploy_df["bookie_pick"] = _normalize_pick(deploy_df)
            tag_col = "context_reason_codes" if "context_reason_codes" in deploy_df.columns else (
                "reason_codes" if "reason_codes" in deploy_df.columns else ""
            )
            deploy_df["market_norm"] = _norm_market(deploy_df.get("market", pd.Series("", index=deploy_df.index)))
            deploy_df["league_norm"] = _norm_str(deploy_df.get("league", pd.Series("", index=deploy_df.index)))
            deploy_df["fixture_key_norm"] = _norm_str(deploy_df.get("fixture_key", pd.Series("", index=deploy_df.index)))
            deploy_df["fixture_key_ascii_norm"] = _norm_str(deploy_df.get("fixture_key_ascii", pd.Series("", index=deploy_df.index)))
            deploy_df["pick_norm"] = _norm_pick(deploy_df.get("bookie_pick", pd.Series("", index=deploy_df.index)))
            deploy_df["pick_ou25_norm"] = _norm_ou25_pick(deploy_df)
            deploy_over = deploy_df.loc[deploy_df["pick_ou25_norm"].eq("OVER25")].copy()
            if tag_col and tag_col in deploy_over.columns:
                deploy_over["context_reason_codes"] = deploy_over[tag_col]
            deploy_over_dedup = deploy_over.drop_duplicates(
                subset=["league", "fixture_key", "market", "bookie_pick"]
            ).copy()
            if tag_col:
                deploy_tags = deploy_df[
                    ["league", "fixture_key", "market", "bookie_pick", tag_col]
                ].drop_duplicates(
                    subset=["league", "fixture_key", "market", "bookie_pick"]
                ).copy()
                deploy_tags = deploy_tags.rename(columns={tag_col: "context_reason_codes"})
    except SystemExit:
        deploy_df = None
        deploy_tags = None
        deploy_over_dedup = None

    # Overall
    overall = pd.DataFrame([{"scope": "OU25_ALL", **_summarize(df, hit_col)}])
    overall.to_csv(outdir / "OU25_3Y_RULEBOOK_AUDIT__OVERALL.csv", index=False)

    by_pick = _group_summary(df, hit_col, ["bookie_pick"])
    by_pick.to_csv(outdir / "OU25_3Y_RULEBOOK_AUDIT__BY_PICK.csv", index=False)

    by_tier = _group_summary(df, hit_col, [tier_col])
    by_tier.to_csv(outdir / "OU25_3Y_RULEBOOK_AUDIT__BY_TIER.csv", index=False)

    by_league = _group_summary(df, hit_col, ["league"])
    by_league.to_csv(outdir / "OU25_3Y_RULEBOOK_AUDIT__BY_LEAGUE.csv", index=False)

    min50 = by_league.loc[by_league["graded"] >= 50].copy()
    min50.to_csv(outdir / "OU25_3Y_RULEBOOK_AUDIT__MIN50_BY_LEAGUE.csv", index=False)

    # Elite + Standard only view
    es = df.loc[df[tier_col].isin(["ELITE", "STANDARD"])].copy()
    if not es.empty:
        es_overall = pd.DataFrame([{"scope": "OU25_ELITE_STANDARD_ONLY", **_summarize(es, hit_col)}])
        es_overall.to_csv(outdir / "OU25_3Y_ELITE_STANDARD_ONLY__OVERALL.csv", index=False)
        _group_summary(es, hit_col, ["bookie_pick"]).to_csv(outdir / "OU25_3Y_ELITE_STANDARD_ONLY__BY_PICK.csv", index=False)
        es_league = _group_summary(es, hit_col, ["league"])
        es_league.to_csv(outdir / "OU25_3Y_ELITE_STANDARD_ONLY__BY_LEAGUE.csv", index=False)
        es_league.loc[es_league["graded"] >= 50].to_csv(outdir / "OU25_3Y_ELITE_STANDARD_ONLY__MIN50_BY_LEAGUE.csv", index=False)

    # Feature deltas (OVER25 and UNDER25)
    for pick in ["OVER25", "UNDER25"]:
        sub = df.loc[df["bookie_pick"].eq(pick)].copy()
        if sub.empty:
            continue
        _feature = _feature_deltas(sub, hit_col, FEATURES_CORE)
        if not _feature.empty:
            _feature.to_csv(outdir / f"OU25_FEATURE_DELTAS__{pick}.csv", index=False)

        sub_es = es.loc[es["bookie_pick"].eq(pick)].copy() if not es.empty else pd.DataFrame()
        if not sub_es.empty:
            _feature_es = _feature_deltas(sub_es, hit_col, FEATURES_CORE)
            if not _feature_es.empty:
                _feature_es.to_csv(outdir / f"OU25_FEATURE_DELTAS__{pick}__ELITE_STANDARD.csv", index=False)

        sweeps = _league_threshold_sweeps(df, hit_col, pick)
        if not sweeps.empty:
            sweeps.to_csv(outdir / f"OU25_LEAGUE_SWEEPS__{pick}.csv", index=False)

            best = _topn_per_league_feature(sweeps, 1)
            best.to_csv(outdir / f"OU25_LEAGUE_SWEEPS_BEST__{pick}.csv", index=False)

            topn = _topn_per_league_feature(sweeps, int(args.top_n))
            topn.to_csv(outdir / f"OU25_LEAGUE_SWEEPS_TOPN__{pick}.csv", index=False)

            best_overall = (
                sweeps.sort_values(
                    ["league", "balanced_score", "roi", "hit_rate", "graded"],
                    ascending=[True, False, False, False, False],
                )
                .groupby(["league"], dropna=False)
                .head(1)
                .reset_index(drop=True)
            )
            best_overall.to_csv(outdir / f"OU25_LEAGUE_SWEEPS_BEST_OVERALL__{pick}.csv", index=False)

            if bool(args.plots):
                _plot_sweep_curves(sweeps, outdir, pick)

    # League gate recommendations (OVER + UNDER)
    best_overall_over = outdir / "OU25_LEAGUE_SWEEPS_BEST_OVERALL__OVER25.csv"
    best_overall_under = outdir / "OU25_LEAGUE_SWEEPS_BEST_OVERALL__UNDER25.csv"
    if best_overall_over.exists() or best_overall_under.exists():
        over_df = pd.read_csv(best_overall_over) if best_overall_over.exists() else pd.DataFrame(columns=["league"])
        under_df = pd.read_csv(best_overall_under) if best_overall_under.exists() else pd.DataFrame(columns=["league"])

        if "league" not in over_df.columns:
            over_df = pd.DataFrame(columns=["league"])
        if "league" not in under_df.columns:
            under_df = pd.DataFrame(columns=["league"])

        over_df = over_df.rename(columns={c: f"over_{c}" for c in over_df.columns if c != "league"})
        under_df = under_df.rename(columns={c: f"under_{c}" for c in under_df.columns if c != "league"})

        recs = over_df.merge(under_df, on="league", how="outer")
        if not recs.empty:
            recs.to_csv(outdir / "OU25_LEAGUE_GATE_RECOMMENDATIONS.csv", index=False)

    # Best overall rule per league across BOTH picks
    if best_overall_over.exists() and best_overall_under.exists():
        over_df = pd.read_csv(best_overall_over)
        under_df = pd.read_csv(best_overall_under)

        frames = []
        if (not over_df.empty) and ("league" in over_df.columns):
            over_df = over_df.copy()
            over_df["pick"] = "OVER25"
            frames.append(over_df)
        if (not under_df.empty) and ("league" in under_df.columns):
            under_df = under_df.copy()
            under_df["pick"] = "UNDER25"
            frames.append(under_df)

        if frames:
            both = pd.concat(frames, axis=0, ignore_index=True)
            both = both.sort_values(
                ["league", "balanced_score", "roi", "hit_rate", "graded"],
                ascending=[True, False, False, False, False],
            ).groupby(["league"], dropna=False).head(1).reset_index(drop=True)
            both.to_csv(outdir / "OU25_LEAGUE_SWEEPS_BEST_OVERALL__BOTH.csv", index=False)

    # Rule stability by pick (feature wins by window)
    stability_over = _rule_stability(df, hit_col, "OVER25")
    if not stability_over.empty:
        stability_over.to_csv(outdir / "OU25_RULE_STABILITY__OVER25.csv", index=False)
    stability_under = _rule_stability(df, hit_col, "UNDER25")
    if not stability_under.empty:
        stability_under.to_csv(outdir / "OU25_RULE_STABILITY__UNDER25.csv", index=False)

    stability_over_rule = _rule_stability_by_threshold(df, hit_col, "OVER25")
    if not stability_over_rule.empty:
        stability_over_rule.to_csv(outdir / "OU25_RULE_STABILITY_RULE__OVER25.csv", index=False)
    stability_under_rule = _rule_stability_by_threshold(df, hit_col, "UNDER25")
    if not stability_under_rule.empty:
        stability_under_rule.to_csv(outdir / "OU25_RULE_STABILITY_RULE__UNDER25.csv", index=False)

    # Suggestions (based on OVER25 min-graded, elite+standard if available)
    sug_src = es if not es.empty else df
    sug_pick = "OVER25"
    sug = _group_summary(sug_src.loc[sug_src["bookie_pick"].eq(sug_pick)].copy(), hit_col, ["league"])
    if not sug.empty:
        sug = _suggest_whitelist_blacklist(sug)
        sug.to_csv(outdir / "OU25_WHITELIST_BLACKLIST_SUGGESTIONS.csv", index=False)

    # Gate suppressors
    _gate_suppressors(base, outdir)

    # OU25 OVER25 conditional on BTTS YES live overlap
    overlap_keys_btts_live = None
    global_overlap_summary = None
    try:
        ou25_over = df_all.loc[
            df_all["market_norm"].eq("ou25") & df_all["pick_ou25_norm"].eq("OVER25")
        ].copy()
        btts_yes = df_all.loc[
            df_all["market_norm"].eq("btts") & df_all["pick_norm"].eq("YES")
        ].copy()
        if not btts_yes.empty:
            if "deploy_pass" in btts_yes.columns:
                btts_live = btts_yes.loc[_safe_num(btts_yes["deploy_pass"]) == 1].copy()
            else:
                btts_live = btts_yes.loc[btts_yes.get("deploy_tier", "").astype(str).isin(["ELITE", "STANDARD"])].copy()
        else:
            btts_live = btts_yes

        if not ou25_over.empty and not btts_live.empty:
            key_cols = ["league_norm", "fixture_key_norm"]
            ou25_keys = ou25_over[key_cols].drop_duplicates()
            btts_keys = btts_live[key_cols].drop_duplicates()
            overlap_keys = ou25_keys.merge(btts_keys, on=key_cols, how="inner")
            overlap = ou25_over.merge(overlap_keys, on=key_cols, how="inner")
            overlap_keys_btts_live = overlap_keys.copy()

            base_sum = _summarize(ou25_over, hit_col)
            overlap_sum = _summarize(overlap, hit_col)
            total_over = float(base_sum.get("graded", 0))
            total_overlap = float(overlap_sum.get("graded", 0))
            coverage = (total_overlap / total_over) if total_over > 0 else float("nan")

            rows = [{
                "baseline_rows": int(total_over),
                "overlap_rows": int(total_overlap),
                "overlap_coverage_pct": round(coverage * 100.0, 2) if coverage == coverage else float("nan"),
                "baseline_hit_rate": float(base_sum.get("hit_rate", float("nan"))),
                "overlap_hit_rate": float(overlap_sum.get("hit_rate", float("nan"))),
                "hit_rate_lift": float(overlap_sum.get("hit_rate", float("nan")) - base_sum.get("hit_rate", float("nan"))),
                "baseline_roi": float(base_sum.get("roi", float("nan"))),
                "overlap_roi": float(overlap_sum.get("roi", float("nan"))),
                "roi_lift": float(overlap_sum.get("roi", float("nan")) - base_sum.get("roi", float("nan"))),
            }]
            global_overlap_summary = rows[0].copy()
            pd.DataFrame(rows).to_csv(outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP.csv", index=False)

            # By-league version
            rows_league = []
            for lg, idx in ou25_over.get("league", pd.Series("", index=ou25_over.index)).astype("string").fillna("").groupby(
                ou25_over.get("league", pd.Series("", index=ou25_over.index)).astype("string").fillna("")
            ).groups.items():
                sub_base = ou25_over.loc[list(idx)].copy()
                sub_keys = sub_base[key_cols].drop_duplicates()
                sub_overlap_keys = sub_keys.merge(btts_keys, on=key_cols, how="inner")
                sub_overlap = sub_base.merge(sub_overlap_keys, on=key_cols, how="inner")
                base_sum_l = _summarize(sub_base, hit_col)
                overlap_sum_l = _summarize(sub_overlap, hit_col)
                total_over_l = float(base_sum_l.get("graded", 0))
                total_overlap_l = float(overlap_sum_l.get("graded", 0))
                coverage_l = (total_overlap_l / total_over_l) if total_over_l > 0 else float("nan")
                rows_league.append({
                    "league": lg,
                    "baseline_rows": int(total_over_l),
                    "overlap_rows": int(total_overlap_l),
                    "overlap_coverage_pct": round(coverage_l * 100.0, 2) if coverage_l == coverage_l else float("nan"),
                    "baseline_hit_rate": float(base_sum_l.get("hit_rate", float("nan"))),
                    "overlap_hit_rate": float(overlap_sum_l.get("hit_rate", float("nan"))),
                    "hit_rate_lift": float(overlap_sum_l.get("hit_rate", float("nan")) - base_sum_l.get("hit_rate", float("nan"))),
                    "baseline_roi": float(base_sum_l.get("roi", float("nan"))),
                    "overlap_roi": float(overlap_sum_l.get("roi", float("nan"))),
                    "roi_lift": float(overlap_sum_l.get("roi", float("nan")) - base_sum_l.get("roi", float("nan"))),
                })
            pd.DataFrame(rows_league).to_csv(outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__BY_LEAGUE.csv", index=False)

            # Top ROI lift by league
            by_league_df = pd.DataFrame(rows_league)
            if not by_league_df.empty:
                top_lift = by_league_df.sort_values(
                    ["roi_lift", "overlap_roi", "overlap_rows"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)
                top_lift.to_csv(outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__TOP_LIFT_BY_LEAGUE.csv", index=False)

                # Recommended overlap allowlist (tunable thresholds)
                allow = top_lift.loc[
                    (top_lift["overlap_rows"] >= int(args.btts_allow_min_rows))
                    & (top_lift["roi_lift"] >= float(args.btts_allow_min_roi_lift))
                    & (top_lift["overlap_roi"] >= float(args.btts_allow_min_overlap_roi))
                ].copy()
                allow_leagues = sorted(allow["league"].dropna().astype("string").tolist())
                pd.DataFrame({"league": allow_leagues}).to_csv(
                    outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__ALLOWLIST.csv",
                    index=False,
                )
                allow_out = allow.copy()
                allow_out["min_overlap_rows"] = int(args.btts_allow_min_rows)
                allow_out["min_roi_lift"] = float(args.btts_allow_min_roi_lift)
                allow_out["min_overlap_roi"] = float(args.btts_allow_min_overlap_roi)
                allow_out.to_csv(
                    outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__ALLOWLIST_THRESHOLDS.csv",
                    index=False,
                )

                # Paste-ready rulebook fragment for rescue allowlist
                allowlist_lines = ["OU25_OVER25_BTTS_RESCUE_ALLOWLIST = {"] + [
                    f'    "{lg}",' for lg in allow_leagues
                ] + ["}"]
                (outdir / "OU25_OVER25_BTTS_RESCUE_RULEBOOK_FRAGMENT.txt").write_text(
                    "\n".join(allowlist_lines),
                    encoding="utf-8",
                )

                # JSON allowlist export
                allow_json = {
                    "allowlist": allow_leagues,
                    "thresholds": {
                        "min_overlap_rows": int(args.btts_allow_min_rows),
                        "min_roi_lift": float(args.btts_allow_min_roi_lift),
                        "min_overlap_roi": float(args.btts_allow_min_overlap_roi),
                    },
                }
                (outdir / "OU25_OVER25_BTTS_RESCUE_ALLOWLIST.json").write_text(
                    json.dumps(allow_json, indent=2),
                    encoding="utf-8",
                )
                if str(args.btts_allowlist_json).strip():
                    custom_path = Path(str(args.btts_allowlist_json)).expanduser()
                    custom_path.parent.mkdir(parents=True, exist_ok=True)
                    custom_path.write_text(json.dumps(allow_json, indent=2), encoding="utf-8")

                # Deploy config patch (unified diff) for deploy_rulebook.py
                try:
                    deploy_path = Path("deploy_rulebook.py")
                    if deploy_path.exists():
                        src_text = deploy_path.read_text(encoding="utf-8")
                        lines = src_text.splitlines(keepends=False)
                        start_idx = None
                        end_idx = None
                        for i, line in enumerate(lines):
                            if line.strip().startswith("OU25_OVER25_BTTS_RESCUE_ALLOWLIST = {"):
                                start_idx = i
                                for j in range(i + 1, len(lines)):
                                    if lines[j].strip() == "}":
                                        end_idx = j
                                        break
                                break
                        if start_idx is not None and end_idx is not None:
                            gen_stamp = datetime.utcnow().strftime("%Y-%m-%d")
                            comment = f"# AUTO-GENERATED BTTS rescue allowlist (UTC {gen_stamp})"
                            replace_start = start_idx
                            if start_idx > 0 and lines[start_idx - 1].strip().startswith("# AUTO-GENERATED BTTS rescue allowlist"):
                                replace_start = start_idx - 1
                            new_block = [comment, "OU25_OVER25_BTTS_RESCUE_ALLOWLIST = {"] + [
                                f'    "{lg}",' for lg in allow_leagues
                            ] + ["}"]
                            new_lines = lines[:replace_start] + new_block + lines[end_idx + 1 :]
                            diff = difflib.unified_diff(
                                lines,
                                new_lines,
                                fromfile="deploy_rulebook.py",
                                tofile="deploy_rulebook.py",
                                lineterm="",
                            )
                            diff_text = "\n".join(diff).strip()
                            if diff_text:
                                (outdir / "OU25_OVER25_BTTS_RESCUE_ALLOWLIST.patch").write_text(
                                    diff_text + "\n",
                                    encoding="utf-8",
                                )
                except Exception:
                    pass

            # Split overlap by OU25 tier (Tier1/Tier2/Fallback)
            tier_tags = [
                "OU25_OVER25_TIER1_PASS",
                "OU25_OVER25_TIER2_PASS",
                "OU25_OVER25_FALLBACK_PASS",
            ]
            tier_rows = []
            for tag in tier_tags:
                tag_mask = rc_tagged.str.contains(tag, regex=False)
                base_tag = scored_tagged.loc[tag_mask].copy()
                if base_tag.empty:
                    continue
                base_sum_t = _summarize(base_tag, hit_col)
                base_rows_t = float(base_sum_t.get("graded", 0))
                overlap_tag = base_tag.merge(overlap_keys, on=key_cols, how="inner")
                overlap_sum_t = _summarize(overlap_tag, hit_col)
                overlap_rows_t = float(overlap_sum_t.get("graded", 0))
                coverage_t = (overlap_rows_t / base_rows_t) if base_rows_t > 0 else float("nan")
                tier_rows.append({
                    "ou25_tier_tag": tag,
                    "baseline_rows": int(base_rows_t),
                    "overlap_rows": int(overlap_rows_t),
                    "overlap_coverage_pct": round(coverage_t * 100.0, 2) if coverage_t == coverage_t else float("nan"),
                    "baseline_hit_rate": float(base_sum_t.get("hit_rate", float("nan"))),
                    "overlap_hit_rate": float(overlap_sum_t.get("hit_rate", float("nan"))),
                    "hit_rate_lift": float(overlap_sum_t.get("hit_rate", float("nan")) - base_sum_t.get("hit_rate", float("nan"))),
                    "baseline_roi": float(base_sum_t.get("roi", float("nan"))),
                    "overlap_roi": float(overlap_sum_t.get("roi", float("nan"))),
                    "roi_lift": float(overlap_sum_t.get("roi", float("nan")) - base_sum_t.get("roi", float("nan"))),
                })
            if tier_rows:
                pd.DataFrame(tier_rows).to_csv(
                    outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__BY_OU25_TIER.csv",
                    index=False,
                )

    except Exception:
        pass

    # OU25 OVER25 policy pass summary (reason tags)
    policy_src = None
    reason_col = None
    if deploy_df is not None and not deploy_df.empty:
        if "context_reason_codes" in deploy_df.columns:
            policy_src = deploy_df
            reason_col = "context_reason_codes"
        elif "reason_codes" in deploy_df.columns:
            policy_src = deploy_df
            reason_col = "reason_codes"
    if policy_src is None:
        if "context_reason_codes" in df.columns:
            policy_src = df
            reason_col = "context_reason_codes"
        elif "reason_codes" in df.columns:
            policy_src = df
            reason_col = "reason_codes"

    if policy_src is not None and reason_col:
        over_policy = policy_src.loc[policy_src["bookie_pick"].eq("OVER25")].copy()
        rc = over_policy[reason_col].astype("string").fillna("")
        def _count(token: str) -> int:
            return int(rc.str.contains(token, regex=False).sum())
        total = int(len(over_policy))
        rows = [
            {"bucket": "OU25_OVER25_TIER1_PASS", "rows": _count("OU25_OVER25_TIER1_PASS")},
            {"bucket": "OU25_OVER25_TIER2_PASS", "rows": _count("OU25_OVER25_TIER2_PASS")},
            {"bucket": "OU25_OVER25_FALLBACK_PASS", "rows": _count("OU25_OVER25_FALLBACK_PASS")},
            {"bucket": "OU25_OVER25_BASELINE_OBSERVE", "rows": _count("OU25_OVER25_BASELINE_OBSERVE")},
            {"bucket": "OU25_OVER25_BLACKLIST", "rows": _count("OU25_OVER25_BLACKLIST")},
            {"bucket": "OU25_OVER25_STRUCT_FAIL", "rows": _count("OU25_OVER25_STRUCT_FAIL")},
            {"bucket": "OU25_OVER25_TOTAL_ROWS", "rows": total},
        ]
        pd.DataFrame(rows).to_csv(outdir / "OU25_OVER25_POLICY_PASS_SUMMARY.csv", index=False)

        # Split by policy lists (Tier1/Tier2/Baseline/Blacklist)
        def _policy_group(league_name: str) -> str:
            if league_name in OU25_OVER25_TIER1_LEAGUES:
                return "TIER1"
            if league_name in OU25_OVER25_TIER2_LEAGUES:
                return "TIER2"
            if league_name in OU25_OVER25_BASELINE_ONLY_LEAGUES:
                return "BASELINE_ONLY"
            if league_name in OU25_OVER25_BLACKLIST_LEAGUES:
                return "BLACKLIST"
            return "OTHER"

        league_series = over_policy.get("league", pd.Series("", index=over_policy.index)).astype("string").fillna("").replace("", "UNKNOWN")
        policy_group = league_series.map(_policy_group)

        policy_rows = []
        for grp, idx in policy_group.groupby(policy_group).groups.items():
            sub = over_policy.loc[list(idx)].copy()
            sub_rc = sub[reason_col].astype("string").fillna("")
            policy_rows.append({
                "policy_group": grp,
                "tier1_pass": int(sub_rc.str.contains("OU25_OVER25_TIER1_PASS", regex=False).sum()),
                "tier2_pass": int(sub_rc.str.contains("OU25_OVER25_TIER2_PASS", regex=False).sum()),
                "fallback_pass": int(sub_rc.str.contains("OU25_OVER25_FALLBACK_PASS", regex=False).sum()),
                "baseline_observe": int(sub_rc.str.contains("OU25_OVER25_BASELINE_OBSERVE", regex=False).sum()),
                "blacklist": int(sub_rc.str.contains("OU25_OVER25_BLACKLIST", regex=False).sum()),
                "struct_fail": int(sub_rc.str.contains("OU25_OVER25_STRUCT_FAIL", regex=False).sum()),
                "rows": int(len(sub)),
            })
        policy_df = pd.DataFrame(policy_rows)
        if not policy_df.empty:
            policy_df = policy_df.sort_values("rows", ascending=False).reset_index(drop=True)
            policy_df.to_csv(outdir / "OU25_OVER25_POLICY_PASS_SUMMARY__BY_POLICY_GROUP.csv", index=False)

        # Policy group performance summary (from scored rows)
        over_scored = df.loc[df["bookie_pick"].eq("OVER25")].copy()
        scored_league = over_scored.get("league", pd.Series("", index=over_scored.index)).astype("string").fillna("").replace("", "UNKNOWN")
        policy_group_scored = scored_league.map(_policy_group)
        perf_rows = []
        for grp, idx in policy_group_scored.groupby(policy_group_scored).groups.items():
            sub = over_scored.loc[list(idx)].copy()
            perf = _summarize(sub, hit_col)
            perf_rows.append({
                "policy_group": grp,
                **perf,
            })
        perf_df = pd.DataFrame(perf_rows)
        if not perf_df.empty:
            perf_df = perf_df.sort_values("graded", ascending=False).reset_index(drop=True)
            perf_df.to_csv(outdir / "OU25_OVER25_POLICY_PERF__BY_POLICY_GROUP.csv", index=False)

        # Policy group performance by league (wide table)
        perf_league_rows = []
        for lg, idx in scored_league.groupby(scored_league).groups.items():
            sub = over_scored.loc[list(idx)].copy()
            grp = _policy_group(lg)
            perf = _summarize(sub, hit_col)
            perf_league_rows.append({
                "policy_group": grp,
                "league": lg,
                **perf,
            })
        perf_league = pd.DataFrame(perf_league_rows)
        if not perf_league.empty:
            perf_league = perf_league.sort_values(["policy_group", "graded"], ascending=[True, False]).reset_index(drop=True)
            perf_league.to_csv(outdir / "OU25_OVER25_POLICY_PERF__BY_POLICY_GROUP_LEAGUE.csv", index=False)

            def _rank(df_in: pd.DataFrame, sort_col: str, name: str) -> None:
                if df_in.empty or sort_col not in df_in.columns:
                    return
                ranked = df_in.sort_values(["policy_group", sort_col, "graded"], ascending=[True, False, False]).copy()
                ranked = ranked.groupby("policy_group", dropna=False).head(10).reset_index(drop=True)
                ranked.to_csv(outdir / name, index=False)

            _rank(perf_league, "roi", "OU25_OVER25_POLICY_GROUP_RANKS__TOP_ROI.csv")
            _rank(perf_league, "hit_rate", "OU25_OVER25_POLICY_GROUP_RANKS__TOP_HIT.csv")
            _rank(perf_league, "profit", "OU25_OVER25_POLICY_GROUP_RANKS__TOP_PROFIT.csv")

        # Policy performance by pass tag (merge deploy tags into scored rows)
        pass_tags = [
            "OU25_OVER25_TIER1_PASS",
            "OU25_OVER25_TIER2_PASS",
            "OU25_OVER25_FALLBACK_PASS",
            "OU25_OVER25_BASELINE_OBSERVE",
            "OU25_OVER25_BLACKLIST",
        ]
        min_graded_by_tag = {
            "OU25_OVER25_TIER1_PASS": 30,
            "OU25_OVER25_TIER2_PASS": 30,
            "OU25_OVER25_FALLBACK_PASS": 10,
            "OU25_OVER25_BASELINE_OBSERVE": 5,
            "OU25_OVER25_BLACKLIST": 5,
        }
        tag_perf_rows = []
        tag_perf_league_rows = []
        live_only_rows = []
        if deploy_tags is not None and deploy_over_dedup is not None:
            over_scored = over_scored.copy()
            deploy_over_dedup = deploy_over_dedup.copy()

            # Key diagnostics (OU25 OVER25 only)
            def _diag_counts(df_in: pd.DataFrame, prefix: str) -> list[dict]:
                out_rows = []
                out_rows.append({"metric": f"{prefix}_rows", "value": int(len(df_in))})
                out_rows.append({"metric": f"{prefix}_distinct_league", "value": int(df_in.get("league_norm", pd.Series(dtype='string')).nunique())})
                out_rows.append({"metric": f"{prefix}_distinct_fixture_key", "value": int(df_in.get("fixture_key_norm", pd.Series(dtype='string')).nunique())})
                out_rows.append({"metric": f"{prefix}_distinct_fixture_key_ascii", "value": int(df_in.get("fixture_key_ascii_norm", pd.Series(dtype='string')).nunique())})
                out_rows.append({"metric": f"{prefix}_distinct_market", "value": int(df_in.get("market_norm", pd.Series(dtype='string')).nunique())})
                out_rows.append({"metric": f"{prefix}_distinct_bookie_pick", "value": int(df_in.get("pick_norm", pd.Series(dtype='string')).nunique())})
                out_rows.append({"metric": f"{prefix}_distinct_selection", "value": int(_norm_pick(df_in.get('selection', pd.Series('', index=df_in.index))).nunique())})
                return out_rows

            diag_rows = []
            diag_rows.extend(_diag_counts(deploy_over_dedup, "deploy"))
            diag_rows.extend(_diag_counts(over_scored, "scored"))

            deploy_fixture_set = set(deploy_over_dedup.get("fixture_key_norm", pd.Series("", index=deploy_over_dedup.index)))
            scored_fixture_set = set(over_scored.get("fixture_key_norm", pd.Series("", index=over_scored.index)))
            deploy_fixture_ascii_set = set(deploy_over_dedup.get("fixture_key_ascii_norm", pd.Series("", index=deploy_over_dedup.index)))
            scored_fixture_ascii_set = set(over_scored.get("fixture_key_ascii_norm", pd.Series("", index=over_scored.index)))
            diag_rows.append({"metric": "scored_fixture_key_not_in_deploy", "value": int(len(scored_fixture_set - deploy_fixture_set))})
            diag_rows.append({"metric": "deploy_fixture_key_not_in_scored", "value": int(len(deploy_fixture_set - scored_fixture_set))})
            diag_rows.append({"metric": "scored_fixture_key_ascii_not_in_deploy", "value": int(len(scored_fixture_ascii_set - deploy_fixture_ascii_set))})
            diag_rows.append({"metric": "deploy_fixture_key_ascii_not_in_scored", "value": int(len(deploy_fixture_ascii_set - scored_fixture_ascii_set))})
            diag_rows.append({"metric": "scored_pick_vs_selection_mismatch", "value": int((over_scored.get("pick_norm", pd.Series("", index=over_scored.index)) != _norm_pick(over_scored.get("selection", pd.Series("", index=over_scored.index)))).sum())})
            diag_rows.append({"metric": "deploy_pick_vs_selection_mismatch", "value": int((deploy_over_dedup.get("pick_norm", pd.Series("", index=deploy_over_dedup.index)) != _norm_pick(deploy_over_dedup.get("selection", pd.Series("", index=deploy_over_dedup.index)))).sum())})

            # Join strategies
            strategies = [
                ("strategy_a", ["league_norm", "fixture_key_norm", "market_norm", "pick_norm"]),
                ("strategy_b", ["league_norm", "fixture_key_ascii_norm", "market_norm", "pick_norm"]),
                ("strategy_c", ["league_norm", "fixture_key_norm", "market_norm", "pick_ou25_norm"]),
                ("strategy_d", ["league_norm", "fixture_key_ascii_norm", "market_norm", "pick_ou25_norm"]),
                ("strategy_e", ["fixture_key_norm", "market_norm", "pick_ou25_norm"]),
                ("strategy_f", ["fixture_key_ascii_norm", "market_norm", "pick_ou25_norm"]),
            ]

            join_rows = []
            best_strategy = None
            best_rate = -1.0
            best_matched = 0
            for name, keys in strategies:
                deploy_key = deploy_over_dedup.drop_duplicates(subset=keys).copy()
                scored_key = over_scored.copy()
                merged = scored_key.merge(
                    deploy_key[keys + ["context_reason_codes"]],
                    on=keys,
                    how="left",
                    suffixes=("", "_deploy"),
                )
                matched = int(merged["context_reason_codes"].astype("string").fillna("").ne("").sum())
                total_scored = int(len(scored_key))
                unmatched = int(total_scored - matched)
                rate = (matched / total_scored) if total_scored > 0 else 0.0
                join_rows.append({
                    "strategy": name,
                    "deploy_rows": int(len(deploy_key)),
                    "scored_rows": total_scored,
                    "matched_rows": matched,
                    "unmatched_rows": unmatched,
                    "merge_rate_pct": round(rate * 100.0, 2),
                })
                if (rate > best_rate) or (rate == best_rate and best_strategy is None):
                    best_rate = rate
                    best_strategy = (name, keys)
                    best_matched = matched

            pd.DataFrame(join_rows).to_csv(outdir / "OU25_DEPLOY_TO_SCORED_JOIN_STRATEGIES.csv", index=False)

            # Apply best strategy
            sel_name, sel_keys = best_strategy if best_strategy else ("strategy_a", strategies[0][1])
            deploy_key = deploy_over_dedup.drop_duplicates(subset=sel_keys).copy()
            scored_tagged = over_scored.merge(
                deploy_key[sel_keys + ["context_reason_codes"]],
                on=sel_keys,
                how="left",
                suffixes=("", "_deploy"),
            )
            if "context_reason_codes_deploy" in scored_tagged.columns:
                rc_tagged = scored_tagged.get("context_reason_codes_deploy", pd.Series("", index=scored_tagged.index)).astype("string").fillna("")
            else:
                rc_tagged = scored_tagged.get("context_reason_codes", pd.Series("", index=scored_tagged.index)).astype("string").fillna("")

            for tag in pass_tags:
                mask = rc_tagged.str.contains(tag, regex=False)
                perf = _summarize(scored_tagged.loc[mask].copy(), hit_col)
                tag_perf_rows.append({"pass_tag": tag, **perf})

            scored_league_series = scored_tagged.get("league", pd.Series("", index=scored_tagged.index)).astype("string").fillna("").replace("", "UNKNOWN")
            # Per-league performance for each pass tag (use deploy-tagged column)
            for tag in pass_tags:
                m_tag = rc_tagged.str.contains(tag, regex=False)
                sub_tag = scored_tagged.loc[m_tag].copy()
                if sub_tag.empty:
                    continue
                league_series_tag = sub_tag.get("league", pd.Series("", index=sub_tag.index)).astype("string").fillna("").replace("", "UNKNOWN")
                for lg, idx in league_series_tag.groupby(league_series_tag).groups.items():
                    sub = sub_tag.loc[list(idx)].copy()
                    perf = _summarize(sub, hit_col)
                    if perf.get("graded", 0) >= min_graded_by_tag.get(tag, 1):
                        tag_perf_league_rows.append({"league": lg, "pass_tag": tag, **perf})
            # Sort per tag by ROI/hit/graded
            if tag_perf_league_rows:
                tpl = pd.DataFrame(tag_perf_league_rows)
                tpl = tpl.sort_values(
                    ["pass_tag", "roi", "hit_rate", "graded"],
                    ascending=[True, False, False, False],
                ).reset_index(drop=True)
                tag_perf_league_rows = tpl.to_dict("records")

            # Final live league list (Tier1+Tier2 that appeared)
            if tag_perf_league_rows:
                tpl = pd.DataFrame(tag_perf_league_rows)
                live_leagues = (
                    tpl.loc[tpl["pass_tag"].isin(["OU25_OVER25_TIER1_PASS", "OU25_OVER25_TIER2_PASS"]), "league"]
                    .dropna()
                    .astype("string")
                    .unique()
                )
                live_out = pd.DataFrame({"league": sorted([str(x) for x in live_leagues])})
                live_out.to_csv(outdir / "OU25_OVER25_LIVE_LEAGUES_FINAL.csv", index=False)

            live_mask = (
                rc_tagged.str.contains("OU25_OVER25_TIER1_PASS", regex=False)
                | rc_tagged.str.contains("OU25_OVER25_TIER2_PASS", regex=False)
                | rc_tagged.str.contains("OU25_OVER25_FALLBACK_PASS", regex=False)
            )
            live_perf = _summarize(scored_tagged.loc[live_mask].copy(), hit_col)
            live_only_rows.append({"cohort": "LIVE", **live_perf})

            obs_mask = rc_tagged.str.contains("OU25_OVER25_BASELINE_OBSERVE", regex=False)
            obs_perf = _summarize(scored_tagged.loc[obs_mask].copy(), hit_col)
            live_only_rows.append({"cohort": "OBSERVE_ONLY", **obs_perf})

            blk_mask = rc_tagged.str.contains("OU25_OVER25_BLACKLIST", regex=False)
            blk_perf = _summarize(scored_tagged.loc[blk_mask].copy(), hit_col)
            live_only_rows.append({"cohort": "BLACKLIST", **blk_perf})

            # Rescue lane audit (BTTS live rescue tag)
            rescue_mask = rc_tagged.str.contains("OU25_OVER25_BTTS_LIVE_RESCUE", regex=False)
            rescue_rows = scored_tagged.loc[rescue_mask].copy()
            rescue_sum = _summarize(rescue_rows, hit_col)
            live_base_rows = scored_tagged.loc[live_mask].copy()
            live_base_sum = _summarize(live_base_rows, hit_col)

            rescue_summary = [{
                "rescued_rows_total": int(rescue_sum.get("graded", 0)),
                "rescued_rows_hit_rate": float(rescue_sum.get("hit_rate", float("nan"))),
                "rescued_rows_roi": float(rescue_sum.get("roi", float("nan"))),
                "baseline_live_rows": int(live_base_sum.get("graded", 0)),
                "baseline_live_hit_rate": float(live_base_sum.get("hit_rate", float("nan"))),
                "baseline_live_roi": float(live_base_sum.get("roi", float("nan"))),
                "rescued_rows_vs_baseline_live_delta_hit_rate": float(
                    rescue_sum.get("hit_rate", float("nan")) - live_base_sum.get("hit_rate", float("nan"))
                ),
                "rescued_rows_vs_baseline_live_delta_roi": float(
                    rescue_sum.get("roi", float("nan")) - live_base_sum.get("roi", float("nan"))
                ),
            }]
            pd.DataFrame(rescue_summary).to_csv(outdir / "OU25_OVER25_BTTS_RESCUE_AUDIT.csv", index=False)

            if not rescue_rows.empty:
                rescue_by_league = []
                league_series_rescue = rescue_rows.get("league", pd.Series("", index=rescue_rows.index)).astype("string").fillna("").replace("", "UNKNOWN")
                for lg, idx in league_series_rescue.groupby(league_series_rescue).groups.items():
                    sub = rescue_rows.loc[list(idx)].copy()
                    sub_sum = _summarize(sub, hit_col)
                    base_sub = live_base_rows.loc[
                        live_base_rows.get("league", pd.Series("", index=live_base_rows.index)).astype("string").fillna("").eq(lg)
                    ].copy()
                    base_sub_sum = _summarize(base_sub, hit_col)
                    rescue_by_league.append({
                        "league": lg,
                        "rescued_rows_total": int(sub_sum.get("graded", 0)),
                        "rescued_rows_hit_rate": float(sub_sum.get("hit_rate", float("nan"))),
                        "rescued_rows_roi": float(sub_sum.get("roi", float("nan"))),
                        "baseline_live_rows": int(base_sub_sum.get("graded", 0)),
                        "baseline_live_hit_rate": float(base_sub_sum.get("hit_rate", float("nan"))),
                        "baseline_live_roi": float(base_sub_sum.get("roi", float("nan"))),
                        "rescued_rows_vs_baseline_live_delta_hit_rate": float(
                            sub_sum.get("hit_rate", float("nan")) - base_sub_sum.get("hit_rate", float("nan"))
                        ),
                        "rescued_rows_vs_baseline_live_delta_roi": float(
                            sub_sum.get("roi", float("nan")) - base_sub_sum.get("roi", float("nan"))
                        ),
                    })
                if rescue_by_league:
                    rescue_df = pd.DataFrame(rescue_by_league).sort_values(
                        ["rescued_rows_vs_baseline_live_delta_roi", "rescued_rows_total"],
                        ascending=[False, False],
                    ).reset_index(drop=True)
                    rescue_df.to_csv(outdir / "OU25_OVER25_BTTS_RESCUE_AUDIT__BY_LEAGUE.csv", index=False)

            # Combined live product (existing live + rescued)
            existing_live = scored_tagged.loc[live_mask].copy()
            rescued_only = scored_tagged.loc[rescue_mask].copy()
            combined_live = scored_tagged.loc[live_mask | rescue_mask].copy()

            existing_live_sum = _summarize(existing_live, hit_col)
            rescued_only_sum = _summarize(rescued_only, hit_col)
            combined_sum = _summarize(combined_live, hit_col)

            combined_rows = [
                {"cohort": "existing_live_only", **existing_live_sum},
                {"cohort": "rescued_only", **rescued_only_sum},
                {"cohort": "combined_live_plus_rescued", **combined_sum},
            ]
            combined_df = pd.DataFrame(combined_rows)
            if not combined_df.empty:
                baseline_row = combined_df.loc[combined_df["cohort"].eq("existing_live_only")].head(1)
                if not baseline_row.empty:
                    b_hit = float(baseline_row.iloc[0].get("hit_rate", float("nan")))
                    b_roi = float(baseline_row.iloc[0].get("roi", float("nan")))
                    combined_df["delta_hit_rate_vs_existing_live"] = combined_df["hit_rate"] - b_hit
                    combined_df["delta_roi_vs_existing_live"] = combined_df["roi"] - b_roi
                combined_df.to_csv(outdir / "OU25_OVER25_BTTS_RESCUE_COMBINED_AUDIT.csv", index=False)

                # Production final summary (flattened)
                prod_rows = []
                for cohort in ["existing_live_only", "rescued_only", "combined_live_plus_rescued"]:
                    row = combined_df.loc[combined_df["cohort"].eq(cohort)].head(1)
                    if row.empty:
                        continue
                    r = row.iloc[0].to_dict()
                    prefix = cohort
                    prod_rows.append({f"{prefix}_rows": r.get("rows"), f"{prefix}_graded": r.get("graded"),
                                      f"{prefix}_hit_rate": r.get("hit_rate"), f"{prefix}_roi": r.get("roi"),
                                      f"{prefix}_profit": r.get("profit")})
                if prod_rows:
                    # merge into single row
                    merged = {}
                    for d in prod_rows:
                        merged.update(d)
                    merged["delta_hit_rate_vs_existing_live"] = combined_df.loc[
                        combined_df["cohort"].eq("combined_live_plus_rescued"),
                        "delta_hit_rate_vs_existing_live",
                    ].iloc[0] if not combined_df.loc[
                        combined_df["cohort"].eq("combined_live_plus_rescued")
                    ].empty else float("nan")
                    merged["delta_roi_vs_existing_live"] = combined_df.loc[
                        combined_df["cohort"].eq("combined_live_plus_rescued"),
                        "delta_roi_vs_existing_live",
                    ].iloc[0] if not combined_df.loc[
                        combined_df["cohort"].eq("combined_live_plus_rescued")
                    ].empty else float("nan")
                    pd.DataFrame([merged]).to_csv(outdir / "OU25_OVER25_PRODUCTION_FINAL_SUMMARY.csv", index=False)

            # Combined live product by league
            by_league_rows = []
            league_series = scored_tagged.get("league", pd.Series("", index=scored_tagged.index)).astype("string").fillna("").replace("", "UNKNOWN")
            for lg, idx in league_series.groupby(league_series).groups.items():
                sub = scored_tagged.loc[list(idx)].copy()
                sub_live = sub.loc[live_mask.loc[sub.index]].copy()
                sub_resc = sub.loc[rescue_mask.loc[sub.index]].copy()
                sub_comb = sub.loc[live_mask.loc[sub.index] | rescue_mask.loc[sub.index]].copy()

                sub_live_sum = _summarize(sub_live, hit_col)
                sub_resc_sum = _summarize(sub_resc, hit_col)
                sub_comb_sum = _summarize(sub_comb, hit_col)

                rows = [
                    {"league": lg, "cohort": "existing_live_only", **sub_live_sum},
                    {"league": lg, "cohort": "rescued_only", **sub_resc_sum},
                    {"league": lg, "cohort": "combined_live_plus_rescued", **sub_comb_sum},
                ]
                for r in rows:
                    r["delta_hit_rate_vs_existing_live"] = r.get("hit_rate", float("nan")) - sub_live_sum.get("hit_rate", float("nan"))
                    r["delta_roi_vs_existing_live"] = r.get("roi", float("nan")) - sub_live_sum.get("roi", float("nan"))
                by_league_rows.extend(rows)

            if by_league_rows:
                pd.DataFrame(by_league_rows).to_csv(
                    outdir / "OU25_OVER25_BTTS_RESCUE_COMBINED_AUDIT__BY_LEAGUE.csv",
                    index=False,
                )

            # Watchlist rescue condition audit (for leagues where rescue underperforms live)
            watchlist_leagues = {
                "Netherlands Eredivisie",
                "Norway Eliteserien",
                "Belgium Pro",
            }
            cond_specs = [
                ("model_p_for_bookie>=0.80", ("model_p_for_bookie", ">=", 0.80)),
                ("exp_goals_sum>=3.2", ("exp_goals_sum", ">=", 3.2)),
                ("bookie_lambda_total_fit>=3.0", ("bookie_lambda_total_fit", ">=", 3.0)),
                ("p00_est<=0.06", ("p00_est", "<=", 0.06)),
            ]
            watch_rows = []
            for lg in sorted(watchlist_leagues):
                base_live = scored_tagged.loc[live_mask & scored_tagged.get("league", pd.Series("", index=scored_tagged.index)).astype("string").fillna("").eq(lg)].copy()
                base_live_sum = _summarize(base_live, hit_col)
                resc_base = scored_tagged.loc[rescue_mask & scored_tagged.get("league", pd.Series("", index=scored_tagged.index)).astype("string").fillna("").eq(lg)].copy()
                for cond_name, (col, op, val) in cond_specs:
                    if col not in resc_base.columns or resc_base.empty:
                        continue
                    s = pd.to_numeric(resc_base.get(col, np.nan), errors="coerce")
                    if op == ">=":
                        m = s >= float(val)
                    else:
                        m = s <= float(val)
                    sub = resc_base.loc[m].copy()
                    sub_sum = _summarize(sub, hit_col)
                    watch_rows.append({
                        "league": lg,
                        "condition": cond_name,
                        "rows": int(sub_sum.get("graded", 0)),
                        "hit_rate": float(sub_sum.get("hit_rate", float("nan"))),
                        "roi": float(sub_sum.get("roi", float("nan"))),
                        "delta_hit_rate_vs_live": float(sub_sum.get("hit_rate", float("nan")) - base_live_sum.get("hit_rate", float("nan"))),
                        "delta_roi_vs_live": float(sub_sum.get("roi", float("nan")) - base_live_sum.get("roi", float("nan"))),
                    })
            if watch_rows:
                watch_df = pd.DataFrame(watch_rows).sort_values(
                    ["league", "delta_roi_vs_live", "rows"],
                    ascending=[True, False, False],
                ).reset_index(drop=True)
                watch_df.to_csv(outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_CONDITION_AUDIT.csv", index=False)

                # Auto-select best rule per watchlist league
                best_rows = []
                for lg, g in watch_df.groupby("league", dropna=False):
                    g = g.sort_values(["delta_roi_vs_live", "rows"], ascending=[False, False])
                    best_rows.append(g.iloc[0].to_dict())
                best_df = pd.DataFrame(best_rows)
                if not best_df.empty:
                    best_df.to_csv(outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_BEST_RULES.csv", index=False)

                    # Write rulebook fragment + patch (league-specific rules)
                    rules_lines = ["OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES = {"]
                    for _, r in best_df.iterrows():
                        lg = str(r.get("league", ""))
                        cond = str(r.get("condition", ""))
                        if cond.startswith("model_p_for_bookie"):
                            rule = {"type": "model_p_for_bookie", "op": ">=", "value": 0.80}
                        elif cond.startswith("exp_goals_sum"):
                            rule = {"type": "exp_goals_sum", "op": ">=", "value": 3.2}
                        elif cond.startswith("bookie_lambda_total_fit"):
                            rule = {"type": "bookie_lambda_total_fit", "op": ">=", "value": 3.0}
                        else:
                            rule = {"type": "p00_est", "op": "<=", "value": 0.06}
                        rules_lines.append(f'    "{lg}": {rule},')
                    rules_lines.append("}")
                    (outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES_FRAGMENT.txt").write_text(
                        "\n".join(rules_lines),
                        encoding="utf-8",
                    )

                    # Final watchlist rules exports (json + patch + md)
                    rules_json = {}
                    for _, r in best_df.iterrows():
                        lg = str(r.get("league", ""))
                        cond = str(r.get("condition", ""))
                        if cond.startswith("model_p_for_bookie"):
                            rule = {"type": "model_p_for_bookie", "op": ">=", "value": 0.80}
                        elif cond.startswith("exp_goals_sum"):
                            rule = {"type": "exp_goals_sum", "op": ">=", "value": 3.2}
                        elif cond.startswith("bookie_lambda_total_fit"):
                            rule = {"type": "bookie_lambda_total_fit", "op": ">=", "value": 3.0}
                        else:
                            rule = {"type": "p00_est", "op": "<=", "value": 0.06}
                        rules_json[lg] = rule
                    (outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES_FINAL.json").write_text(
                        json.dumps(rules_json, indent=2),
                        encoding="utf-8",
                    )

                    try:
                        deploy_path = Path("deploy_rulebook.py")
                        if deploy_path.exists():
                            src_text = deploy_path.read_text(encoding="utf-8")
                            lines = src_text.splitlines(keepends=False)
                            start_idx = None
                            end_idx = None
                            for i, line in enumerate(lines):
                                if line.strip().startswith("OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES = {"):
                                    start_idx = i
                                    for j in range(i + 1, len(lines)):
                                        if lines[j].strip() == "}":
                                            end_idx = j
                                            break
                                    break
                            if start_idx is not None and end_idx is not None:
                                gen_stamp = datetime.utcnow().strftime("%Y-%m-%d")
                                comment = f"# AUTO-GENERATED BTTS rescue watchlist rules (UTC {gen_stamp})"
                                replace_start = start_idx
                                if start_idx > 0 and lines[start_idx - 1].strip().startswith("# AUTO-GENERATED BTTS rescue watchlist rules"):
                                    replace_start = start_idx - 1
                                new_block = [comment] + rules_lines
                                new_lines = lines[:replace_start] + new_block + lines[end_idx + 1 :]
                                diff = difflib.unified_diff(
                                    lines,
                                    new_lines,
                                    fromfile="deploy_rulebook.py",
                                    tofile="deploy_rulebook.py",
                                    lineterm="",
                                )
                                diff_text = "\n".join(diff).strip()
                                if diff_text:
                                    (outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES.patch").write_text(
                                        diff_text + "\n",
                                        encoding="utf-8",
                                    )
                                    (outdir / "OU25_OVER25_BTTS_RESCUE_WATCHLIST_RULES_FINAL.patch").write_text(
                                        diff_text + "\n",
                                        encoding="utf-8",
                                    )
                    except Exception:
                        pass

            # BTTS-YES live overlap: LIVE-only baseline (tagged live cohorts only)
            if overlap_keys_btts_live is not None and not overlap_keys_btts_live.empty:
                live_tag_mask = (
                    rc_tagged.str.contains("OU25_OVER25_TIER1_PASS", regex=False)
                    | rc_tagged.str.contains("OU25_OVER25_TIER2_PASS", regex=False)
                    | rc_tagged.str.contains("OU25_OVER25_FALLBACK_PASS", regex=False)
                )
                ou25_live_only = scored_tagged.loc[live_tag_mask].copy()
                live_sum = _summarize(ou25_live_only, hit_col)
                total_live = float(live_sum.get("graded", 0))
                live_overlap = ou25_live_only.merge(overlap_keys_btts_live, on=["league_norm", "fixture_key_norm"], how="inner")
                live_overlap_sum = _summarize(live_overlap, hit_col)
                total_live_overlap = float(live_overlap_sum.get("graded", 0))
                coverage_live = (total_live_overlap / total_live) if total_live > 0 else float("nan")
                rows_live = [{
                    "baseline_rows": int(total_live),
                    "overlap_rows": int(total_live_overlap),
                    "overlap_coverage_pct": round(coverage_live * 100.0, 2) if coverage_live == coverage_live else float("nan"),
                    "baseline_hit_rate": float(live_sum.get("hit_rate", float("nan"))),
                    "overlap_hit_rate": float(live_overlap_sum.get("hit_rate", float("nan"))),
                    "hit_rate_lift": float(live_overlap_sum.get("hit_rate", float("nan")) - live_sum.get("hit_rate", float("nan"))),
                    "baseline_roi": float(live_sum.get("roi", float("nan"))),
                    "overlap_roi": float(live_overlap_sum.get("roi", float("nan"))),
                    "roi_lift": float(live_overlap_sum.get("roi", float("nan")) - live_sum.get("roi", float("nan"))),
                }]
                pd.DataFrame(rows_live).to_csv(outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__LIVE_ONLY.csv", index=False)

                # Live-only by league
                rows_live_league = []
                if not ou25_live_only.empty:
                    for lg, idx in ou25_live_only.get("league", pd.Series("", index=ou25_live_only.index)).astype("string").fillna("").groupby(
                        ou25_live_only.get("league", pd.Series("", index=ou25_live_only.index)).astype("string").fillna("")
                    ).groups.items():
                        sub_base = ou25_live_only.loc[list(idx)].copy()
                        sub_overlap = sub_base.merge(overlap_keys_btts_live, on=["league_norm", "fixture_key_norm"], how="inner")
                        base_sum_l = _summarize(sub_base, hit_col)
                        overlap_sum_l = _summarize(sub_overlap, hit_col)
                        total_over_l = float(base_sum_l.get("graded", 0))
                        total_overlap_l = float(overlap_sum_l.get("graded", 0))
                        coverage_l = (total_overlap_l / total_over_l) if total_over_l > 0 else float("nan")
                        rows_live_league.append({
                            "league": lg,
                            "baseline_rows": int(total_over_l),
                            "overlap_rows": int(total_overlap_l),
                            "overlap_coverage_pct": round(coverage_l * 100.0, 2) if coverage_l == coverage_l else float("nan"),
                            "baseline_hit_rate": float(base_sum_l.get("hit_rate", float("nan"))),
                            "overlap_hit_rate": float(overlap_sum_l.get("hit_rate", float("nan"))),
                            "hit_rate_lift": float(overlap_sum_l.get("hit_rate", float("nan")) - base_sum_l.get("hit_rate", float("nan"))),
                            "baseline_roi": float(base_sum_l.get("roi", float("nan"))),
                            "overlap_roi": float(overlap_sum_l.get("roi", float("nan"))),
                            "roi_lift": float(overlap_sum_l.get("roi", float("nan")) - base_sum_l.get("roi", float("nan"))),
                        })
                if rows_live_league:
                    pd.DataFrame(rows_live_league).to_csv(
                        outdir / "OU25_OVER25_BTTS_YES_LIVE_OVERLAP__LIVE_ONLY_BY_LEAGUE.csv",
                        index=False,
                    )

                if global_overlap_summary is not None:
                    try:
                        if (
                            int(global_overlap_summary.get("baseline_rows", -1)) == int(rows_live[0]["baseline_rows"])
                            and int(global_overlap_summary.get("overlap_rows", -1)) == int(rows_live[0]["overlap_rows"])
                        ):
                            print("[ou25_audit] WARNING: LIVE_ONLY overlap equals global overlap; live filter likely failed")
                    except Exception:
                        pass

            # Merge audit (selected strategy)
            deploy_tagged = deploy_over_dedup.loc[
                deploy_over_dedup.get("context_reason_codes", pd.Series("", index=deploy_over_dedup.index)).astype("string").fillna("").str.contains("OU25_OVER25_", regex=False)
            ].copy()
            total_deploy_over = int(len(deploy_over_dedup))
            total_deploy_tagged = int(len(deploy_tagged))
            total_scored = int(len(over_scored))
            matched_scored = int(rc_tagged.ne("").sum())
            unmatched_scored = int(total_scored - matched_scored)
            merge_rate = (matched_scored / total_scored) if total_scored > 0 else 0.0
            tagged_match_rate = (matched_scored / total_deploy_tagged) if total_deploy_tagged > 0 else 0.0
            tag_coverage = (total_deploy_tagged / total_scored) if total_scored > 0 else 0.0
            audit_rows = [
                {"metric": "total_deploy_ou25_over25_rows", "value": total_deploy_over},
                {"metric": "total_deploy_ou25_over25_tagged_rows", "value": total_deploy_tagged},
                {"metric": "total_scored_ou25_over25_rows", "value": total_scored},
                {"metric": "matched_scored_rows_after_join", "value": matched_scored},
                {"metric": "unmatched_scored_rows", "value": unmatched_scored},
                {"metric": "merge_rate_pct", "value": round(merge_rate * 100.0, 2)},
                {"metric": "tagged_match_rate_pct", "value": round(tagged_match_rate * 100.0, 2)},
                {"metric": "tag_coverage_pct", "value": round(tag_coverage * 100.0, 2)},
                {"metric": "selected_strategy", "value": sel_name},
            ]
            audit_rows.extend(diag_rows)
            pd.DataFrame(audit_rows).to_csv(outdir / "OU25_DEPLOY_TO_SCORED_MERGE_AUDIT.csv", index=False)

            unmatched = scored_tagged.loc[rc_tagged.eq("")].copy()
            if not unmatched.empty:
                sample_cols = [c for c in ("league", "fixture_key", "fixture_key_ascii", "market", "bookie_pick", "selection", "match_date", "__src") if c in unmatched.columns]
                unmatched.loc[:, sample_cols].head(100).to_csv(
                    outdir / "OU25_DEPLOY_TO_SCORED_UNMATCHED_SAMPLE.csv",
                    index=False,
                )
            if tagged_match_rate < 0.95:
                print(f"[ou25_audit] WARNING: deploy→scored tagged match rate {tagged_match_rate*100.0:.2f}% (<95%) using {sel_name}")
            else:
                print(f"[ou25_audit] deploy→scored tagged match rate {tagged_match_rate*100.0:.2f}% using {sel_name}")

        if tag_perf_rows:
            pd.DataFrame(tag_perf_rows).to_csv(outdir / "OU25_OVER25_POLICY_PERF__BY_PASS_TAG.csv", index=False)
        if tag_perf_league_rows:
            pd.DataFrame(tag_perf_league_rows).to_csv(outdir / "OU25_OVER25_POLICY_PERF__BY_PASS_TAG_LEAGUE.csv", index=False)
        if live_only_rows:
            pd.DataFrame(live_only_rows).to_csv(outdir / "OU25_OVER25_LIVE_ONLY_PERF.csv", index=False)

        # Live-only top ROI by league (tagged scored rows)
        if tag_perf_league_rows:
            live_tags = {
                "OU25_OVER25_TIER1_PASS",
                "OU25_OVER25_TIER2_PASS",
                "OU25_OVER25_FALLBACK_PASS",
            }
            tpl = pd.DataFrame(tag_perf_league_rows)
            live_tpl = tpl.loc[tpl["pass_tag"].isin(list(live_tags))].copy()
            if not live_tpl.empty:
                live_tpl = live_tpl.sort_values(
                    ["roi", "hit_rate", "graded"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)
                live_tpl.to_csv(outdir / "OU25_OVER25_LIVE_TOP_ROI_BY_LEAGUE.csv", index=False)

        # Tier1 vs Tier2 vs Fallback heatmap (wide, by league)
        if tag_perf_league_rows:
            tpl = pd.DataFrame(tag_perf_league_rows)
            live_tpl = tpl.loc[
                tpl["pass_tag"].isin([
                    "OU25_OVER25_TIER1_PASS",
                    "OU25_OVER25_TIER2_PASS",
                    "OU25_OVER25_FALLBACK_PASS",
                ])
            ].copy()
            if not live_tpl.empty:
                def _wide(col: str) -> pd.DataFrame:
                    return live_tpl.pivot_table(index="league", columns="pass_tag", values=col, aggfunc="first")
                wide_hit = _wide("hit_rate").add_prefix("hit_")
                wide_roi = _wide("roi").add_prefix("roi_")
                wide_profit = _wide("profit").add_prefix("profit_")
                wide_graded = _wide("graded").add_prefix("graded_")
                heat = pd.concat([wide_hit, wide_roi, wide_profit, wide_graded], axis=1).reset_index()
                roi_cols = [c for c in heat.columns if c.startswith("roi_")]
                if roi_cols:
                    heat = heat.sort_values([roi_cols[0]], ascending=[False]).reset_index(drop=True)
                heat.to_csv(outdir / "OU25_OVER25_LIVE_TIER_HEATMAP_BY_LEAGUE.csv", index=False)

        # Policy config export (paste-ready)
        policy_rows = [
            {"group": "OU25_OVER25_TIER1_LEAGUES", "values": sorted(list(OU25_OVER25_TIER1_LEAGUES))},
            {"group": "OU25_OVER25_TIER2_LEAGUES", "values": sorted(list(OU25_OVER25_TIER2_LEAGUES))},
            {"group": "OU25_OVER25_BASELINE_ONLY_LEAGUES", "values": sorted(list(OU25_OVER25_BASELINE_ONLY_LEAGUES))},
            {"group": "OU25_OVER25_BLACKLIST_LEAGUES", "values": sorted(list(OU25_OVER25_BLACKLIST_LEAGUES))},
        ]
        pd.DataFrame(policy_rows).to_csv(outdir / "OU25_OVER25_POLICY_CONFIG_EXPORT.csv", index=False)

        # By league
        league = over_policy.get("league", pd.Series("", index=over_policy.index)).astype("string").fillna("").replace("", "UNKNOWN")
        by_league_rows = []
        for lg, idx in league.groupby(league).groups.items():
            sub = over_policy.loc[list(idx)].copy()
            sub_rc = sub[reason_col].astype("string").fillna("")
            by_league_rows.append({
                "league": lg,
                "tier1_pass": int(sub_rc.str.contains("OU25_OVER25_TIER1_PASS", regex=False).sum()),
                "tier2_pass": int(sub_rc.str.contains("OU25_OVER25_TIER2_PASS", regex=False).sum()),
                "fallback_pass": int(sub_rc.str.contains("OU25_OVER25_FALLBACK_PASS", regex=False).sum()),
                "baseline_observe": int(sub_rc.str.contains("OU25_OVER25_BASELINE_OBSERVE", regex=False).sum()),
                "blacklist": int(sub_rc.str.contains("OU25_OVER25_BLACKLIST", regex=False).sum()),
                "struct_fail": int(sub_rc.str.contains("OU25_OVER25_STRUCT_FAIL", regex=False).sum()),
                "rows": int(len(sub)),
            })
        by_league = pd.DataFrame(by_league_rows)
        if not by_league.empty:
            by_league = by_league.sort_values("rows", ascending=False).reset_index(drop=True)
            by_league.to_csv(outdir / "OU25_OVER25_POLICY_PASS_SUMMARY__BY_LEAGUE.csv", index=False)

        # By tier
        tier_col = "deploy_tier" if "deploy_tier" in over_policy.columns else ("tier" if "tier" in over_policy.columns else "")
        if tier_col:
            tier = over_policy.get(tier_col, pd.Series("", index=over_policy.index)).astype("string").fillna("").replace("", "UNKNOWN")
            by_tier_rows = []
            for tr, idx in tier.groupby(tier).groups.items():
                sub = over_policy.loc[list(idx)].copy()
                sub_rc = sub[reason_col].astype("string").fillna("")
                by_tier_rows.append({
                    "tier": tr,
                    "tier1_pass": int(sub_rc.str.contains("OU25_OVER25_TIER1_PASS", regex=False).sum()),
                    "tier2_pass": int(sub_rc.str.contains("OU25_OVER25_TIER2_PASS", regex=False).sum()),
                    "fallback_pass": int(sub_rc.str.contains("OU25_OVER25_FALLBACK_PASS", regex=False).sum()),
                    "baseline_observe": int(sub_rc.str.contains("OU25_OVER25_BASELINE_OBSERVE", regex=False).sum()),
                    "blacklist": int(sub_rc.str.contains("OU25_OVER25_BLACKLIST", regex=False).sum()),
                    "struct_fail": int(sub_rc.str.contains("OU25_OVER25_STRUCT_FAIL", regex=False).sum()),
                    "rows": int(len(sub)),
                })
            by_tier = pd.DataFrame(by_tier_rows)
            if not by_tier.empty:
                by_tier = by_tier.sort_values("rows", ascending=False).reset_index(drop=True)
                by_tier.to_csv(outdir / "OU25_OVER25_POLICY_PASS_SUMMARY__BY_TIER.csv", index=False)

    # Markdown report
    _write_md_report(outdir)

    print(f"WROTE: {outdir}")


if __name__ == "__main__":
    main()
