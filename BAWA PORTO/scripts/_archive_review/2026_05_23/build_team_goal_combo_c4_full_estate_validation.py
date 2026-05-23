#!/usr/bin/env python3
"""Validate Team-Goal Combo C4 posture on the full Phase 8H estate.

Research-only. This validates team-goal/dominance combos as their own lane,
not as hidden FTR rescue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/team_goal_combo_c4_full_estate_validation")

PROMOTE_NOW = {
    ("Spain La Liga", "HOME_WIN_AND_HOME_GE2"),
    ("Germany Bundesliga", "HOME_WIN_AND_HOME_GE2"),
}
WATCH = {
    ("USA MLS", "HOME_WIN_AND_HOME_GE2"),
    ("USA MLS", "AWAY_WIN_AND_AWAY_GE2"),
}
EXCLUDE = {
    ("Belgium Pro", "HOME_WIN_AND_HOME_GE2"),
    ("Belgium Pro", "AWAY_WIN_AND_AWAY_GE2"),
}


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def combo_rows(df: pd.DataFrame) -> pd.DataFrame:
    ftr = df[df.get("market_norm", df.get("market", "")).astype("string").str.lower().eq("ftr")].copy()
    if ftr.empty:
        return ftr

    allowed = num(ftr.get("ftr_combo_live_allowed", 0)).eq(1)
    product = ftr.get("ftr_combo_live_product", pd.Series("", index=ftr.index)).astype("string").fillna("")
    ftr = ftr.loc[allowed & product.ne("")].copy()
    if ftr.empty:
        return ftr

    product = ftr.get("ftr_combo_live_product", pd.Series("", index=ftr.index)).astype("string").fillna("")
    home = product.eq("HOME_WIN_AND_HOME_GE2")
    away = product.eq("AWAY_WIN_AND_AWAY_GE2")
    ftr["combo_product"] = product
    ftr["combo_tier"] = ftr.get("ftr_combo_live_tier", pd.Series("", index=ftr.index)).astype("string").fillna("")
    ftr["combo_correct"] = np.nan
    ftr.loc[home, "combo_correct"] = num(ftr.loc[home, "hw_and_hge2_hit"])
    ftr.loc[away, "combo_correct"] = num(ftr.loc[away, "aw_and_age2_hit"])
    ftr["combo_prob"] = np.where(home, num(ftr.get("hw_hge2_combo_prob", np.nan)), num(ftr.get("aw_age2_combo_prob", np.nan)))
    ftr["combo_lambda"] = np.where(home, num(ftr.get("hw_hge2_combo_lambda", np.nan)), num(ftr.get("aw_age2_combo_lambda", np.nan)))
    ftr["combo_ge2_gap"] = np.where(home, num(ftr.get("hw_hge2_ge2_gap", np.nan)), num(ftr.get("aw_age2_ge2_gap", np.nan)))
    return ftr[ftr["combo_correct"].notna()].copy()


def scorecard(df: pd.DataFrame, group_cols: list[str], hit_col: str = "combo_correct") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group[hit_col])
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        odds = num(group.get("bookie_od", pd.Series(np.nan, index=group.index)))
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_bookie_od": float(odds.mean()) if odds.notna().any() else np.nan,
                "avg_combo_prob": float(num(group.get("combo_prob", np.nan)).mean()),
                "profit": float(np.nansum(profit)) if graded else np.nan,
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def policy_table(combo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (league, product), group in combo.groupby(["league", "combo_product"], dropna=False):
        key = (str(league), str(product))
        if key in PROMOTE_NOW:
            posture = "PROMOTE_SHADOW_PROOF"
            reason = "C3 proof lane and full-estate combo lane clear strong quality."
        elif key in WATCH:
            posture = "WATCH_ONLY"
            reason = "Promising but sample/league volatility requires independent repeat proof."
        elif key in EXCLUDE:
            posture = "EXCLUDE"
            reason = "Belgium combo shape failed C1/C2 benchmark quality."
        else:
            posture = "OBSERVE"
            reason = "No explicit C3 proof posture yet."
        row = {"league": league, "combo_product": product, "combo_posture": posture, "reason": reason}
        row.update(scorecard(group, []).iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["combo_posture", "league", "combo_product"])


def window_stability(combo: pd.DataFrame) -> pd.DataFrame:
    if combo.empty or "window_id" not in combo.columns:
        return pd.DataFrame()
    win = scorecard(combo, ["league", "combo_product", "combo_tier", "window_id"])
    rows = []
    for keys, group in win.groupby(["league", "combo_product", "combo_tier"], dropna=False):
        league, product, tier = keys
        graded = int(group["graded"].sum())
        wins = float(group["wins"].sum())
        row = {
            "league": league,
            "combo_product": product,
            "combo_tier": tier,
            "active_windows": int(group["window_id"].nunique()),
            "graded": graded,
            "wins": wins,
            "hit_rate": wins / graded if graded else np.nan,
            "median_window_hit_rate": float(group["hit_rate"].median()),
            "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
            "windows_below_90": int((group["hit_rate"] < 0.90).sum()),
            "median_rows_per_window": float(group["rows"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["league", "combo_product", "combo_tier"])


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
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.row_level)
    combo = combo_rows(df)
    combo.to_csv(outdir / "team_goal_combo_c4_full_estate_rows.csv", index=False)

    by_league_tier = scorecard(combo, ["league", "combo_product", "combo_tier"])
    by_league_tier.to_csv(outdir / "team_goal_combo_c4_by_league_tier.csv", index=False)
    by_policy = policy_table(combo)
    by_policy.to_csv(outdir / "team_goal_combo_c4_policy_posture.csv", index=False)
    stability = window_stability(combo)
    stability.to_csv(outdir / "team_goal_combo_c4_window_stability.csv", index=False)

    proof = by_policy[by_policy["combo_posture"].eq("PROMOTE_SHADOW_PROOF")].copy()
    watch = by_policy[by_policy["combo_posture"].isin(["WATCH_ONLY", "EXCLUDE"])].copy()

    summary = [
        "# Team-Goal Combo C4 Full-Estate Validation",
        "",
        "Research-only validation. Team-goal combos remain a separate lane and are not FTR rescue.",
        "",
        "## Proof Posture",
        md_table(proof[["league", "combo_product", "graded", "wins", "hit_rate", "roi", "combo_posture"]]),
        "",
        "## Watch / Exclude",
        md_table(watch[["league", "combo_product", "graded", "wins", "hit_rate", "roi", "combo_posture", "reason"]]),
        "",
        "## Top League/Tier Cells",
        md_table(by_league_tier.sort_values(["hit_rate", "graded"], ascending=[False, False]).head(30)),
        "",
        "## Read",
        "",
        "- Spain and Germany HOME_WIN_AND_HOME_GE2 remain the proof posture candidates.",
        "- USA MLS remains watch-only despite attractive C1/C2 samples.",
        "- Belgium combos remain excluded from restoration.",
        "- No live sidecar should be built from this lane until window stability and live-board shadow QA are reviewed.",
    ]
    (outdir / "team_goal_combo_c4_full_estate_validation_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
