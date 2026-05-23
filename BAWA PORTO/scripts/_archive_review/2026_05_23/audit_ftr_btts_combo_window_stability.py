#!/usr/bin/env python3
"""Audit window stability for FTR + BTTS combo discovery lanes.

Research-only. This does not create deploy routing. It tests whether same-fixture
FTR + BTTS expressions are stable enough to graduate from discovery into shadow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "reports/2026-05-06/ftr_btts_combo_discovery_audit/ftr_btts_combo_rows_scored.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/ftr_btts_combo_window_stability")

SPECIAL_COMPS = {
    "England FA Cup",
    "Champions League",
    "Europa League",
    "Europa Conference",
}


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def profit_series(hit: pd.Series, odds: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(hit.eq(1), odds - 1.0, np.where(hit.eq(0), -1.0, np.nan)),
        index=hit.index,
    )


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        hit = num(group["combo_correct"])
        odds = num(group.get("synthetic_combo_od", pd.Series(np.nan, index=group.index)))
        profit = profit_series(hit, odds)
        graded = int(hit.notna().sum())
        wins = float(hit.eq(1).sum())

        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int(hit.eq(0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_synth_od": float(odds.mean()) if odds.notna().any() else np.nan,
                "avg_synth_model_p": float(num(group.get("synthetic_combo_model_p", np.nan)).mean()),
                "avg_synth_value_edge": float(num(group.get("synthetic_combo_value_edge", np.nan)).mean()),
                "profit": float(profit.sum(skipna=True)) if graded else np.nan,
                "roi": float(profit.sum(skipna=True) / graded) if graded else np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def stability(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    window = scorecard(df, group_cols + ["window_id"])
    rows = []
    for keys, group in window.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        graded = int(group["graded"].sum())
        wins = float(group["wins"].sum())
        profit = float(group["profit"].sum(skipna=True))
        weighted_avg_od = (
            float((group["avg_synth_od"] * group["graded"]).sum() / graded)
            if graded and "avg_synth_od" in group.columns
            else np.nan
        )
        weighted_avg_model_p = (
            float((group["avg_synth_model_p"] * group["graded"]).sum() / graded)
            if graded and "avg_synth_model_p" in group.columns
            else np.nan
        )
        weighted_avg_value_edge = (
            float((group["avg_synth_value_edge"] * group["graded"]).sum() / graded)
            if graded and "avg_synth_value_edge" in group.columns
            else np.nan
        )
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "active_windows": int(group["window_id"].nunique()),
                "rows": int(group["rows"].sum()),
                "graded": graded,
                "wins": wins,
                "losses": int(group["losses"].sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit": profit,
                "roi": profit / graded if graded else np.nan,
                "avg_synth_od": weighted_avg_od,
                "avg_synth_model_p": weighted_avg_model_p,
                "avg_synth_value_edge": weighted_avg_value_edge,
                "median_rows_per_window": float(group["rows"].median()),
                "p25_rows_per_window": float(group["rows"].quantile(0.25)),
                "windows_below_50": int(group["hit_rate"].lt(0.50).sum()),
                "windows_below_45": int(group["hit_rate"].lt(0.45).sum()),
                "windows_negative_roi": int(group["roi"].lt(0.0).sum()),
                "mean_window_hit_rate": float(group["hit_rate"].mean()),
                "median_window_hit_rate": float(group["hit_rate"].median()),
                "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
                "p10_window_hit_rate": float(group["hit_rate"].quantile(0.10)),
                "min_window_hit_rate": float(group["hit_rate"].min()),
                "median_window_roi": float(group["roi"].median()),
                "p25_window_roi": float(group["roi"].quantile(0.25)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def discovery_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Keep products aligned with the current combo research posture."""
    out = df.copy()
    product = out["combo_product"].astype("string")
    league = out["league"].astype("string")
    is_no = product.isin(["HOME_AND_BTTS_NO", "AWAY_AND_BTTS_NO"])
    is_special_yes = product.isin(["HOME_AND_BTTS_YES", "AWAY_AND_BTTS_YES"]) & league.isin(SPECIAL_COMPS)
    out = out.loc[is_no | is_special_yes].copy()
    out["combo_research_family"] = np.where(
        out["combo_product"].astype("string").str.endswith("_NO"),
        "FTR_PLUS_BTTS_NO",
        "SPECIAL_COMP_FTR_PLUS_BTTS_YES",
    )
    return out


def threshold_candidate_rows(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """Build strict candidate subsets from simple high-signal thresholds."""
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
    quantiles = [0.70, 0.80, 0.90]
    rows = []

    for (league, product), group in df.groupby(["league", "combo_product"], dropna=False):
        for feature in features:
            if feature not in group.columns:
                continue
            vals = num(group[feature]).dropna()
            if vals.empty:
                continue
            for quantile in quantiles:
                threshold = float(vals.quantile(quantile))
                sub = group.loc[num(group[feature]).ge(threshold)].copy()
                graded = int(num(sub["combo_correct"]).notna().sum())
                if graded < min_rows:
                    continue
                card = scorecard(sub, []).iloc[0].to_dict()
                rows.append(
                    {
                        "league": league,
                        "combo_product": product,
                        "feature": feature,
                        "op": ">=",
                        "threshold": threshold,
                        "quantile": quantile,
                        **card,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["hit_rate", "graded", "roi"],
        ascending=[False, False, False],
    )


def apply_candidate(df: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    mask = (
        df["league"].astype("string").eq(str(candidate["league"]))
        & df["combo_product"].astype("string").eq(str(candidate["combo_product"]))
        & num(df[candidate["feature"]]).ge(float(candidate["threshold"]))
    )
    out = df.loc[mask].copy()
    out["candidate_id"] = (
        out["league"].astype("string")
        + "|"
        + out["combo_product"].astype("string")
        + "|"
        + str(candidate["feature"])
        + ">="
        + f"{float(candidate['threshold']):.6f}"
    )
    out["candidate_feature"] = str(candidate["feature"])
    out["candidate_threshold"] = float(candidate["threshold"])
    return out


def candidate_stability(df: pd.DataFrame, candidates: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_parts = []
    for _, candidate in candidates.head(top_n).iterrows():
        selected_parts.append(apply_candidate(df, candidate))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return selected, pd.DataFrame()

    stable = stability(
        selected,
        ["candidate_id", "league", "combo_product", "candidate_feature", "candidate_threshold"],
    )
    return selected, stable.sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False])


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")

    lines = [
        "| " + " | ".join(text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--top-candidates", type=int, default=30)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"league", "combo_product", "combo_correct", "window_id", "synthetic_combo_od"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    scoped = discovery_scope(df)
    scoped.to_csv(outdir / "ftr_btts_combo_window_scope_rows.csv", index=False)

    product_stability = stability(scoped, ["combo_product"])
    product_stability.to_csv(outdir / "ftr_btts_combo_product_stability.csv", index=False)

    family_stability = stability(scoped, ["combo_research_family"])
    family_stability.to_csv(outdir / "ftr_btts_combo_family_stability.csv", index=False)

    league_product_stability = stability(scoped, ["league", "combo_product"])
    league_product_stability.to_csv(outdir / "ftr_btts_combo_league_product_stability.csv", index=False)

    candidates = threshold_candidate_rows(scoped, args.min_rows)
    candidates.to_csv(outdir / "ftr_btts_combo_window_threshold_candidates.csv", index=False)

    selected, selected_stability = candidate_stability(scoped, candidates, args.top_candidates)
    selected.to_csv(outdir / "ftr_btts_combo_window_threshold_candidate_rows.csv", index=False)
    selected_stability.to_csv(outdir / "ftr_btts_combo_window_threshold_candidate_stability.csv", index=False)

    broad = league_product_stability[
        league_product_stability["graded"].ge(args.min_rows)
    ].sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False]).head(25)

    summary = [
        "# FTR + BTTS Combo Window Stability",
        "",
        "Research-only audit for same-fixture FTR + BTTS expressions.",
        "This keeps combo products separate from FTR rescue and does not change deployment routing.",
        "",
        "## Research Families",
        md_table(
            family_stability[
                [
                    "combo_research_family",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_synth_od",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_negative_roi",
                    "median_rows_per_window",
                ]
            ].sort_values(["hit_rate", "graded"], ascending=[False, False])
        ),
        "",
        "## Product Stability",
        md_table(
            product_stability[
                [
                    "combo_product",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_synth_od",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_negative_roi",
                    "median_rows_per_window",
                ]
            ].sort_values(["hit_rate", "graded"], ascending=[False, False])
        ),
        "",
        "## Best Broad League/Product Cells",
        md_table(
            broad[
                [
                    "league",
                    "combo_product",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_synth_od",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_negative_roi",
                    "median_rows_per_window",
                ]
            ]
        ),
        "",
        "## Best Threshold Candidate Stability",
        md_table(
            selected_stability[
                [
                    "league",
                    "combo_product",
                    "candidate_feature",
                    "candidate_threshold",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_negative_roi",
                    "median_rows_per_window",
                ]
            ].head(20)
            if not selected_stability.empty
            else selected_stability
        ),
        "",
        "## Read",
        "",
        "- `HOME/AWAY + BTTS NO` is the main broad research family.",
        "- `HOME/AWAY + BTTS YES` should stay special-competition-only until independently proven.",
        "- Candidate rows need live shadow proof before any promotion conversation.",
    ]
    (outdir / "ftr_btts_combo_window_stability_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
