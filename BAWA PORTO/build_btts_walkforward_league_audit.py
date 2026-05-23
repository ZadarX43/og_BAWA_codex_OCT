

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/walk_forward")
OUT_CSV = Path("btts_walkforward_league_audit.csv")
OUT_MD = Path("btts_walkforward_league_audit.md")
TARGET_MARKET = "btts"
TARGET_LABEL = "btts_valueev"
MIN_MONTHS_FOR_AUDIT = 1


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _safe_num(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _weighted_mean(g: pd.DataFrame, col: str, weight: str = "rows") -> float:
    x = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna()
    if not m.any() or float(w[m].sum()) == 0.0:
        return float("nan")
    return float((x[m] * w[m]).sum() / w[m].sum())


def _month_sort_key(month_tag: str) -> tuple[int, int]:
    parts = str(month_tag).split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (9999, 9999)


def _max_drawdown_from_monthly_roi(roi_series: pd.Series) -> float:
    s = pd.to_numeric(roi_series, errors="coerce").fillna(0.0)
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min())


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _build_month_league_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    d["bookie_od"] = pd.to_numeric(d["bookie_od"], errors="coerce")
    d = d[d["correct"].notna()].copy()

    if d.empty:
        return pd.DataFrame(columns=["league", "rows", "hit", "roi", "avg_odds"])

    rows: list[dict[str, Any]] = []
    for league, g in d.groupby("league", dropna=False):
        scored = g[g["correct"].notna()].copy()
        roi_df = scored[scored["bookie_od"].notna()].copy()
        rows.append(
            {
                "league": str(league),
                "rows": int(len(scored)),
                "hit": float(scored["correct"].mean()) if len(scored) else float("nan"),
                "roi": float((roi_df["correct"] * roi_df["bookie_od"] - 1.0).mean()) if len(roi_df) else float("nan"),
                "avg_odds": float(roi_df["bookie_od"].mean()) if len(roi_df) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["roi", "rows"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"walk-forward root not found: {ROOT}")

    month_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: _month_sort_key(p.name))
    if not month_dirs:
        raise SystemExit(f"no month directories found under: {ROOT}")

    month_rows: list[dict[str, Any]] = []

    for month_dir in month_dirs:
        month_tag = month_dir.name
        src = month_dir / "BTTS_VALUEEV_GATED_ROWS.csv"
        if not src.exists():
            continue

        df = _read_csv(src)
        if df.empty:
            continue

        if "market" in df.columns:
            df = df[df["market"].astype(str).str.lower().str.strip().eq(TARGET_MARKET)].copy()
        if df.empty:
            continue

        if "league" not in df.columns or "correct" not in df.columns or "bookie_od" not in df.columns:
            continue

        summary = _build_month_league_summary(df)
        if summary.empty:
            continue

        for _, r in summary.iterrows():
            month_rows.append(
                {
                    "branch": TARGET_LABEL,
                    "month": month_tag,
                    "league": str(r["league"]),
                    "rows": _safe_num(r["rows"]),
                    "hit": _safe_num(r["hit"]),
                    "roi": _safe_num(r["roi"]),
                    "avg_odds": _safe_num(r["avg_odds"]),
                    "source_csv": str(src),
                }
            )

    month_df = pd.DataFrame(month_rows)
    if month_df.empty:
        raise SystemExit(f"no BTTS league-level rows found under: {ROOT}")

    month_df["rows"] = pd.to_numeric(month_df["rows"], errors="coerce")
    month_df["hit"] = pd.to_numeric(month_df["hit"], errors="coerce")
    month_df["roi"] = pd.to_numeric(month_df["roi"], errors="coerce")
    month_df["avg_odds"] = pd.to_numeric(month_df["avg_odds"], errors="coerce")
    month_df["month"] = month_df["month"].astype(str)
    month_df["league"] = month_df["league"].astype(str)

    league_rows: list[dict[str, Any]] = []

    for (branch, league), g in month_df.groupby(["branch", "league"], dropna=False):
        g = g.sort_values("month").reset_index(drop=True)
        months_present = int(g["month"].nunique())
        if months_present < MIN_MONTHS_FOR_AUDIT:
            continue

        profitable_months = int((pd.to_numeric(g["roi"], errors="coerce") > 0).sum())
        losing_months = int((pd.to_numeric(g["roi"], errors="coerce") <= 0).sum())

        roi_series = pd.to_numeric(g["roi"], errors="coerce")
        hit_series = pd.to_numeric(g["hit"], errors="coerce")
        row_series = pd.to_numeric(g["rows"], errors="coerce")

        best_roi_idx = roi_series.idxmax() if roi_series.notna().any() else None
        worst_roi_idx = roi_series.idxmin() if roi_series.notna().any() else None
        best_hit_idx = hit_series.idxmax() if hit_series.notna().any() else None
        worst_hit_idx = hit_series.idxmin() if hit_series.notna().any() else None

        league_rows.append(
            {
                "branch": str(branch),
                "league": str(league),
                "months_present": months_present,
                "total_rows": int(row_series.fillna(0).sum()),
                "weighted_hit": _weighted_mean(g, "hit"),
                "weighted_roi": _weighted_mean(g, "roi"),
                "weighted_avg_odds": _weighted_mean(g, "avg_odds"),
                "avg_rows_per_month": float(row_series.mean()) if row_series.notna().any() else float("nan"),
                "min_rows_month": float(row_series.min()) if row_series.notna().any() else float("nan"),
                "max_rows_month": float(row_series.max()) if row_series.notna().any() else float("nan"),
                "profitable_months": profitable_months,
                "losing_months": losing_months,
                "best_month_by_roi": g.loc[best_roi_idx, "month"] if best_roi_idx is not None else "",
                "best_roi": float(g.loc[best_roi_idx, "roi"]) if best_roi_idx is not None else float("nan"),
                "worst_month_by_roi": g.loc[worst_roi_idx, "month"] if worst_roi_idx is not None else "",
                "worst_roi": float(g.loc[worst_roi_idx, "roi"]) if worst_roi_idx is not None else float("nan"),
                "best_month_by_hit": g.loc[best_hit_idx, "month"] if best_hit_idx is not None else "",
                "best_hit": float(g.loc[best_hit_idx, "hit"]) if best_hit_idx is not None else float("nan"),
                "worst_month_by_hit": g.loc[worst_hit_idx, "month"] if worst_hit_idx is not None else "",
                "worst_hit": float(g.loc[worst_hit_idx, "hit"]) if worst_hit_idx is not None else float("nan"),
                "roi_std": float(roi_series.std(ddof=0)) if roi_series.notna().any() else float("nan"),
                "hit_std": float(hit_series.std(ddof=0)) if hit_series.notna().any() else float("nan"),
                "max_drawdown": _max_drawdown_from_monthly_roi(roi_series),
            }
        )

    league_df = pd.DataFrame(league_rows)
    if league_df.empty:
        raise SystemExit("no BTTS league audit rows built")

    league_df = league_df.sort_values(
        ["weighted_roi", "weighted_hit", "total_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    league_df.to_csv(OUT_CSV, index=False)

    top_cols = [
        "branch",
        "league",
        "months_present",
        "total_rows",
        "weighted_hit",
        "weighted_roi",
        "weighted_avg_odds",
        "profitable_months",
        "losing_months",
        "max_drawdown",
    ]
    top_table = league_df[top_cols].copy()

    detail_table = month_df[["month", "branch", "league", "rows", "hit", "roi", "avg_odds"]].copy()
    detail_table = detail_table.sort_values(["league", "month"]).reset_index(drop=True)

    md_lines: list[str] = []
    md_lines.append("# BTTS Walk-Forward League Audit")
    md_lines.append("")
    md_lines.append(f"- Walk-forward root: `{ROOT}`")
    md_lines.append(f"- Branch label: `{TARGET_LABEL}`")
    md_lines.append(f"- Leagues audited: `{league_df['league'].nunique()}`")
    md_lines.append(f"- Months covered: `{month_df['month'].nunique()}`")
    md_lines.append("")
    md_lines.append("## League leaderboard")
    md_lines.append("")
    md_lines.append(_to_markdown(top_table))
    md_lines.append("")
    md_lines.append("## Month-by-month league detail")
    md_lines.append("")
    md_lines.append(_to_markdown(detail_table))

    OUT_MD.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(league_df.to_string(index=False))
    print(f"\nWROTE: {OUT_CSV}")
    print(f"WROTE: {OUT_MD}")


if __name__ == "__main__":
    main()