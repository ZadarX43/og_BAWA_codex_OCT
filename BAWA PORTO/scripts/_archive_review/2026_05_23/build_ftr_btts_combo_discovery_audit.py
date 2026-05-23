#!/usr/bin/env python3
"""Discover FTR + BTTS same-fixture combo lanes.

Research-only audit for:
  - HOME_AND_BTTS_YES
  - HOME_AND_BTTS_NO
  - AWAY_AND_BTTS_YES
  - AWAY_AND_BTTS_NO

Synthetic odds are estimated by multiplying the available FTR and BTTS row odds.
This is a discovery audit, not deploy behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/ftr_btts_combo_discovery_audit")


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_market_norm"] = out.get("market_norm", out.get("market", "")).astype("string").fillna("").str.lower().str.strip()
    out["_selection_norm"] = out.get("selection", out.get("bookie_pick", "")).astype("string").fillna("").str.upper().str.strip()
    return out


def build_combo_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize(df)
    ftr = df[df["_market_norm"].eq("ftr") & df["_selection_norm"].isin(["HOME", "AWAY"])].copy()
    btts = df[df["_market_norm"].eq("btts") & df["_selection_norm"].isin(["YES", "NO"])].copy()
    if ftr.empty or btts.empty:
        return pd.DataFrame()

    keep_ftr = [
        "league", "match_date", "home_team_name", "away_team_name", "fixture_key", "window_id",
        "_selection_norm", "deploy_tier", "tier", "bookie_od", "model_p_for_bookie",
        "p_meta_ftr", "ftr_margin", "pick_side_mass_top3", "pick_side_margin_top3",
        "value_edge", "value_edge_tier", "context_reason_codes", "ftr_hit",
        "home_team_goal_count", "away_team_goal_count",
    ]
    keep_btts = [
        "league", "fixture_key", "_selection_norm", "deploy_tier", "tier", "bookie_od",
        "model_p_for_bookie", "p_meta_btts", "cs_mass_btts_yes", "cs_mass_btts_no",
        "p00_est", "p_home_fts", "p_away_fts", "value_edge", "value_edge_tier",
        "btts_yes_hit", "btts_no_hit",
    ]
    ftr = ftr[[c for c in keep_ftr if c in ftr.columns]].copy()
    btts = btts[[c for c in keep_btts if c in btts.columns]].copy()

    ftr = ftr.rename(columns={
        "_selection_norm": "ftr_side",
        "deploy_tier": "ftr_deploy_tier",
        "tier": "ftr_tier",
        "bookie_od": "ftr_od",
        "model_p_for_bookie": "ftr_model_p",
        "value_edge": "ftr_value_edge",
        "value_edge_tier": "ftr_value_edge_tier",
    })
    btts = btts.rename(columns={
        "_selection_norm": "btts_side",
        "deploy_tier": "btts_deploy_tier",
        "tier": "btts_tier",
        "bookie_od": "btts_od",
        "model_p_for_bookie": "btts_model_p",
        "value_edge": "btts_value_edge",
        "value_edge_tier": "btts_value_edge_tier",
    })

    merged = ftr.merge(btts, on=["league", "fixture_key"], how="inner", suffixes=("", "_bttsrow"))
    if merged.empty:
        return merged

    merged["combo_product"] = merged["ftr_side"] + "_AND_BTTS_" + merged["btts_side"]
    btts_hit = np.where(
        merged["btts_side"].eq("YES"),
        num(merged.get("btts_yes_hit", np.nan)),
        num(merged.get("btts_no_hit", np.nan)),
    )
    merged["combo_correct"] = np.where(
        num(merged.get("ftr_hit", np.nan)).eq(1) & pd.Series(btts_hit, index=merged.index).eq(1),
        1.0,
        np.where(num(merged.get("ftr_hit", np.nan)).notna() & pd.Series(btts_hit, index=merged.index).notna(), 0.0, np.nan),
    )
    merged["synthetic_combo_od"] = num(merged.get("ftr_od", np.nan)) * num(merged.get("btts_od", np.nan))
    merged["synthetic_combo_model_p"] = num(merged.get("ftr_model_p", np.nan)) * num(merged.get("btts_model_p", np.nan))
    merged["synthetic_combo_implied"] = 1.0 / merged["synthetic_combo_od"].replace(0, np.nan)
    merged["synthetic_combo_value_edge"] = merged["synthetic_combo_model_p"] - merged["synthetic_combo_implied"]
    merged["both_value_premium"] = (
        merged.get("ftr_value_edge_tier", "").astype("string").fillna("").str.upper().eq("PREMIUM")
        & merged.get("btts_value_edge_tier", "").astype("string").fillna("").str.upper().eq("PREMIUM")
    )
    return merged


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["combo_correct"])
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        odds = num(group.get("synthetic_combo_od", np.nan))
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_synth_od": float(odds.mean()) if odds.notna().any() else np.nan,
                "avg_synth_model_p": float(num(group.get("synthetic_combo_model_p", np.nan)).mean()),
                "avg_synth_value_edge": float(num(group.get("synthetic_combo_value_edge", np.nan)).mean()),
                "profit": float(np.nansum(profit)) if graded else np.nan,
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sweep(combo: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows = []
    features = [
        "synthetic_combo_model_p",
        "synthetic_combo_value_edge",
        "p_meta_ftr",
        "p_meta_btts",
        "pick_side_margin_top3",
        "pick_side_mass_top3",
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "p00_est",
    ]
    qs = [0.50, 0.60, 0.70, 0.80, 0.90]
    for (league, product), group in combo.groupby(["league", "combo_product"], dropna=False):
        for feat in features:
            if feat not in group.columns:
                continue
            vals = num(group[feat]).dropna()
            if vals.empty:
                continue
            for q in qs:
                threshold = float(vals.quantile(q))
                sub = group[num(group[feat]).ge(threshold)]
                if int(num(sub["combo_correct"]).notna().sum()) < min_rows:
                    continue
                row = {"league": league, "combo_product": product, "feature": feat, "op": ">=", "threshold": threshold, "quantile": q}
                row.update(scorecard(sub, []).iloc[0].to_dict())
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False])


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = ["| " + " | ".join(text.columns) + " |", "| " + " | ".join(["---"] * len(text.columns)) + " |"]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-level", default=str(DEFAULT_ROW_LEVEL))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min-rows", type=int, default=40)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.row_level)
    combo = build_combo_rows(df)
    combo.to_csv(outdir / "ftr_btts_combo_rows_scored.csv", index=False)

    by_product = scorecard(combo, ["combo_product"])
    by_product.to_csv(outdir / "ftr_btts_combo_by_product.csv", index=False)
    by_league = scorecard(combo, ["league", "combo_product"])
    by_league.to_csv(outdir / "ftr_btts_combo_by_league_product.csv", index=False)
    by_tiers = scorecard(combo, ["league", "combo_product", "ftr_deploy_tier", "btts_deploy_tier"])
    by_tiers.to_csv(outdir / "ftr_btts_combo_by_league_product_tiers.csv", index=False)
    sweep = threshold_sweep(combo, args.min_rows)
    sweep.to_csv(outdir / "ftr_btts_combo_threshold_sweep.csv", index=False)

    best = by_league[(by_league["graded"] >= args.min_rows)].sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False]).head(30)
    sweep_best = sweep.head(30) if not sweep.empty else sweep

    summary = [
        "# FTR + BTTS Combo Discovery Audit",
        "",
        "Research-only same-fixture composite audit. Synthetic odds are FTR odds multiplied by BTTS odds.",
        "",
        "## Product Summary",
        md_table(by_product.sort_values(["hit_rate", "graded"], ascending=[False, False])),
        "",
        "## Best League/Product Cells",
        md_table(best),
        "",
        "## Best Threshold Candidates",
        md_table(sweep_best[["league", "combo_product", "feature", "threshold", "graded", "wins", "hit_rate", "avg_synth_od", "roi"]]) if not sweep_best.empty else "_No threshold candidates._",
        "",
        "## Read",
        "",
        "- Treat this as discovery only, not deploy routing.",
        "- Strong cells need window stability and live shadow proof before any promotion conversation.",
        "- Popular FTR+BTTS triples should remain separate from FTR rescue.",
    ]
    (outdir / "ftr_btts_combo_discovery_audit_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
