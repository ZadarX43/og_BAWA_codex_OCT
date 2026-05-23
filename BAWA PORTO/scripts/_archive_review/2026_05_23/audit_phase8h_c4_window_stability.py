#!/usr/bin/env python3
"""Audit C4 recovery-ring stability by walk-forward window.

Research-only. Applies the C4 recommended ring policy to the full-estate
row-level replay export, then reports market / league / ring stability over
139 walk-forward windows. This does not modify live deployment logic.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_POLICY = Path(
    "reports/2026-05-06/phase8h_c4_recovery_rings/phase8h_c4_recommended_ring_policy_by_league.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_c4_window_stability")

PHASE8E_THRESHOLDS = {
    ("btts", "England FA Cup"): ("p_meta_btts", 0.80),
    ("btts", "Europa Conference"): ("p_meta_btts", 0.88),
    ("btts", "Champions League"): ("p_meta_btts", 0.80),
    ("btts", "Brazil Serie A"): ("p_meta_btts", 0.85),
    ("btts", "Italy Serie A"): ("p_meta_btts", 0.85),
    ("btts", "Spain La Liga"): ("p_meta_btts", 0.80),
    ("btts", "France Ligue 1"): ("p_meta_btts", 0.88),
    ("btts", "Scotland Premiership"): ("p_meta_btts", 0.80),
    ("btts", "Belgium Pro"): ("p_meta_btts", 0.88),
    ("btts", "Norway Eliteserien"): ("p_meta_btts", 0.80),
    ("btts", "Netherlands Eredivisie"): ("p_meta_btts", 0.88),
    ("btts", "Japan J1"): ("p_meta_btts", 0.80),
    ("btts", "USA MLS"): ("p_meta_btts", 0.85),
    ("ou25", "England FA Cup"): ("p_meta_ou25", 0.80),
    ("ou25", "Europa Conference"): ("p_meta_ou25", 0.80),
    ("ou25", "Germany Bundesliga"): ("p_meta_ou25", 0.88),
    ("ou25", "Europa League"): ("p_meta_ou25", 0.85),
    ("ou25", "Brazil Serie A"): ("p_meta_ou25", 0.80),
    ("ou25", "Champions League"): ("p_meta_ou25", 0.80),
    ("ou25", "Portugal Liga"): ("p_meta_ou25", 0.90),
    ("ou25", "Netherlands Eredivisie"): ("p_meta_ou25", 0.88),
    ("ou25", "Scotland Premiership"): ("p_meta_ou25", 0.90),
    ("ou25", "Spain La Liga"): ("p_meta_ou25", 0.80),
    ("ou25", "Japan J1"): ("p_meta_ou25", 0.80),
    ("ou25", "Norway Eliteserien"): ("p_meta_ou25", 0.85),
    ("ou25", "USA MLS"): ("p_meta_ou25", 0.80),
}


def num(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_policy_source(source: str, market: str, league: str) -> tuple[str, str, float] | None:
    if source == "phase8e_threshold":
        threshold = PHASE8E_THRESHOLDS.get((market, league))
        if not threshold:
            return None
        feature, value = threshold
        return feature, ">=", value
    match = re.match(r"^([A-Za-z0-9_]+)\s*(>=|<=)\s*([-+]?[0-9]*\.?[0-9]+)$", str(source).strip())
    if not match:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def apply_condition(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    values = num(df[feature])
    if op == "<=":
        return values.le(threshold).fillna(False)
    return values.ge(threshold).fillna(False)


def apply_policy(rows: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, rule in policy.iterrows():
        if rule["recovery_ring"] == "OBSERVE":
            continue
        market = str(rule["market"])
        league = str(rule["league"])
        parsed = parse_policy_source(str(rule["policy_source"]), market, league)
        if parsed is None:
            continue
        feature, op, threshold = parsed
        selection = "YES" if market == "btts" else "OVER25"
        mask = (
            rows["market_norm"].astype("string").eq(market)
            & rows["league"].astype("string").eq(league)
            & rows.get("selection", rows.get("bookie_pick", "")).astype("string").str.upper().eq(selection)
            & apply_condition(rows, feature, op, threshold)
        )
        part = rows.loc[mask].copy()
        if part.empty:
            continue
        part["recovery_ring"] = rule["recovery_ring"]
        part["policy_source"] = rule["policy_source"]
        part["policy_feature"] = feature
        part["policy_op"] = op
        part["policy_threshold"] = threshold
        selected.append(part)

    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["dedupe_key"] = (
        out["league"].astype("string")
        + "||"
        + out["fixture_key"].astype("string")
        + "||"
        + out["market_norm"].astype("string")
        + "||"
        + out.get("selection", out.get("bookie_pick", "")).astype("string")
    )
    return out.drop_duplicates("dedupe_key", keep="first").copy()


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["correct"])
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def stability_summary(selected: pd.DataFrame) -> pd.DataFrame:
    by_window = scorecard(selected, ["market_norm", "window_id"])
    if by_window.empty:
        return by_window
    rows = []
    for market, group in by_window.groupby("market_norm", dropna=False):
        active = group[group["graded"] > 0].copy()
        rows.append(
            {
                "market": market,
                "active_windows": int(len(active)),
                "total_graded": int(active["graded"].sum()),
                "total_wins": float(active["wins"].sum()),
                "overall_hit_rate": active["wins"].sum() / active["graded"].sum()
                if active["graded"].sum()
                else np.nan,
                "median_window_hit_rate": active["hit_rate"].median(),
                "p25_window_hit_rate": active["hit_rate"].quantile(0.25),
                "p10_window_hit_rate": active["hit_rate"].quantile(0.10),
                "windows_below_90": int((active["hit_rate"] < 0.90).sum()),
                "windows_below_85": int((active["hit_rate"] < 0.85).sum()),
                "median_rows_per_window": active["graded"].median(),
            }
        )
    return pd.DataFrame(rows)


def league_stability(selected: pd.DataFrame) -> pd.DataFrame:
    by_window = scorecard(selected, ["market_norm", "league", "recovery_ring", "window_id"])
    if by_window.empty:
        return by_window
    rows = []
    for keys, group in by_window.groupby(["market_norm", "league", "recovery_ring"], dropna=False):
        market, league, ring = keys
        active = group[group["graded"] > 0].copy()
        rows.append(
            {
                "market": market,
                "league": league,
                "recovery_ring": ring,
                "active_windows": int(len(active)),
                "total_graded": int(active["graded"].sum()),
                "total_wins": float(active["wins"].sum()),
                "overall_hit_rate": active["wins"].sum() / active["graded"].sum()
                if active["graded"].sum()
                else np.nan,
                "median_window_hit_rate": active["hit_rate"].median(),
                "p25_window_hit_rate": active["hit_rate"].quantile(0.25),
                "windows_below_90": int((active["hit_rate"] < 0.90).sum()),
                "median_rows_per_window": active["graded"].median(),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        else:
            show[col] = show[col].astype("string").fillna("")
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def write_summary(outdir: Path, market: pd.DataFrame, league: pd.DataFrame) -> None:
    watch = league[
        (league["total_graded"] >= 20)
        & ((league["overall_hit_rate"] < 0.91) | (league["windows_below_90"] >= 3))
    ].sort_values(["market", "overall_hit_rate", "windows_below_90"], ascending=[True, True, False])
    lines = [
        "# Phase 8H C4b Window Stability",
        "",
        "Research-only stability audit for the C4 recommended ring policy.",
        "",
        "## Market Stability",
        markdown_table(market),
        "",
        "## League Watch Cells",
        markdown_table(watch),
        "",
        "## Interpretation",
        "",
        "- Use this before translating any C4 ring into live deployment logic.",
        "- Aggregate hit rate is not enough; league/window stability decides restore order.",
        "- Watch cells are not automatic rejects, but they need ring-by-ring review before live promotion.",
        "",
    ]
    (outdir / "phase8h_c4b_window_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.row_level, low_memory=False)
    policy = pd.read_csv(args.policy)
    selected = apply_policy(rows, policy)

    selected.to_csv(args.outdir / "phase8h_c4b_policy_selected_rows.csv", index=False)
    window_card = scorecard(selected, ["market_norm", "recovery_ring", "window_id"])
    window_card.to_csv(args.outdir / "phase8h_c4b_window_scorecard.csv", index=False)
    market_stab = stability_summary(selected)
    market_stab.to_csv(args.outdir / "phase8h_c4b_market_stability.csv", index=False)
    league_stab = league_stability(selected)
    league_stab.to_csv(args.outdir / "phase8h_c4b_league_stability.csv", index=False)
    write_summary(args.outdir, market_stab, league_stab)
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

