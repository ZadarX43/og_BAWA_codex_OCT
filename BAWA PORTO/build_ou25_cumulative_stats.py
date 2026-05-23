#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

SRC = Path("ou25_branch_comparison.csv")
OUT = Path("ou25_branch_cumulative_stats.csv")

NUMERIC_COLS = [
    "rows",
    "hit",
    "roi",
    "avg_odds",
    "top_q",
    "markets",
    "leagues",
    "ou25_band1_low",
    "ou25_band1_high",
    "ou25_band2_low",
    "ou25_band2_high",
]


def weighted_mean(g: pd.DataFrame, col: str, weight: str = "rows") -> float:
    if col not in g.columns or weight not in g.columns:
        return float("nan")
    x = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna()
    if not m.any() or float(w[m].sum()) == 0.0:
        return float("nan")
    return float((x[m] * w[m]).sum() / w[m].sum())


def safe_mean(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    s = pd.to_numeric(g[col], errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.mean())


def safe_max(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    s = pd.to_numeric(g[col], errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.max())


def safe_min(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    s = pd.to_numeric(g[col], errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.min())


def safe_std(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    s = pd.to_numeric(g[col], errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.std(ddof=0))


def best_row_value(g: pd.DataFrame, metric_col: str, value_col: str, which: str) -> object:
    if metric_col not in g.columns or value_col not in g.columns:
        return ""
    metric = pd.to_numeric(g[metric_col], errors="coerce")
    valid = metric.notna()
    if not valid.any():
        return ""
    idx = metric.idxmax() if which == "max" else metric.idxmin()
    return g.loc[idx, value_col]


if not SRC.exists():
    raise SystemExit(f"missing source csv: {SRC}")

df = pd.read_csv(SRC, low_memory=False)
if df.empty:
    raise SystemExit(f"source csv is empty: {SRC}")

for c in NUMERIC_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

rows: list[dict] = []
for branch, g in df.groupby("branch", dropna=False):
    g = g.copy().reset_index(drop=True)

    total_rows = int(pd.to_numeric(g["rows"], errors="coerce").fillna(0).sum()) if "rows" in g.columns else 0
    variants_present = int(len(g))

    rows.append(
        {
            "branch": branch,
            "sweep_type": g["sweep_type"].dropna().iloc[0] if "sweep_type" in g.columns and g["sweep_type"].notna().any() else "",
            "pick_mode": g["pick_mode"].dropna().iloc[0] if "pick_mode" in g.columns and g["pick_mode"].notna().any() else "",
            "variants_present": variants_present,
            "datasets_present": int(g["dataset"].nunique(dropna=True)) if "dataset" in g.columns else 0,
            "total_rows": total_rows,
            "weighted_hit": weighted_mean(g, "hit"),
            "weighted_roi": weighted_mean(g, "roi"),
            "weighted_avg_odds": weighted_mean(g, "avg_odds"),
            "max_rows": safe_max(g, "rows"),
            "min_rows": safe_min(g, "rows"),
            "best_dataset_by_hit": best_row_value(g, "hit", "dataset", "max") if "dataset" in g.columns else "",
            "best_hit": safe_max(g, "hit"),
            "worst_dataset_by_hit": best_row_value(g, "hit", "dataset", "min") if "dataset" in g.columns else "",
            "worst_hit": safe_min(g, "hit"),
            "best_dataset_by_roi": best_row_value(g, "roi", "dataset", "max") if "dataset" in g.columns else "",
            "best_roi": safe_max(g, "roi"),
            "worst_dataset_by_roi": best_row_value(g, "roi", "dataset", "min") if "dataset" in g.columns else "",
            "worst_roi": safe_min(g, "roi"),
            "hit_std": safe_std(g, "hit"),
            "roi_std": safe_std(g, "roi"),
            "avg_top_q": safe_mean(g, "top_q"),
            "avg_band1_low": safe_mean(g, "ou25_band1_low"),
            "avg_band1_high": safe_mean(g, "ou25_band1_high"),
            "avg_band2_low": safe_mean(g, "ou25_band2_low"),
            "avg_band2_high": safe_mean(g, "ou25_band2_high"),
            "single_variant_only": bool(variants_present == 1),
        }
    )

out_df = pd.DataFrame(rows)
if out_df.empty:
    raise SystemExit("no grouped rows were produced from ou25_branch_comparison.csv")

out_df = out_df.sort_values(
    ["weighted_roi", "weighted_hit", "total_rows"],
    ascending=[False, False, False],
).reset_index(drop=True)

out_df.to_csv(OUT, index=False)
print(out_df.to_string(index=False))
print(f"\nWROTE: {OUT}")