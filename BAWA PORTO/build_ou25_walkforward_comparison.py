#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/ou25_walkforward")
OUT_CSV = Path("ou25_walkforward_comparison.csv")
OUT_MD = Path("ou25_walkforward_comparison.md")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _month_sort_key(month_tag: str) -> tuple:
    # expects YYYY-MM if possible; otherwise lexical fallback
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


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"walk-forward root not found: {ROOT}")

    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], reverse=True)
    if not run_dirs:
        raise SystemExit(f"no run directories found under: {ROOT}")

    # Use the most recent run directory by name
    run_root = run_dirs[0]

    rows: list[dict[str, Any]] = []

    for month_dir in sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: _month_sort_key(p.name)):
        month_tag = month_dir.name

        for branch_dir in sorted([p for p in month_dir.iterdir() if p.is_dir()]):
            summary_files = sorted(branch_dir.glob("*__SUMMARY.json"))
            if not summary_files:
                continue

            summary = _read_json(summary_files[0])
            cfg = summary.get("config", {})
            meta = summary.get("summary", {})

            rows.append(
                {
                    "run_tag": run_root.name,
                    "month": month_tag,
                    "branch": branch_dir.name,
                    "tag": summary.get("tag", ""),
                    "rows": _safe_num(meta.get("rows")),
                    "hit": _safe_num(meta.get("hit")),
                    "roi": _safe_num(meta.get("roi")),
                    "avg_odds": _safe_num(meta.get("avg_od")),
                    "markets": _safe_num(meta.get("markets")),
                    "leagues": _safe_num(meta.get("leagues")),
                    "top_q": _safe_num(cfg.get("top_q")),
                    "ou25_band1_low": _safe_num((cfg.get("ou25_band1") or [None, None])[0]),
                    "ou25_band1_high": _safe_num((cfg.get("ou25_band1") or [None, None])[1]),
                    "ou25_band2_low": _safe_num((cfg.get("ou25_band2") or [None, None])[0]),
                    "ou25_band2_high": _safe_num((cfg.get("ou25_band2") or [None, None])[1]),
                    "include_ou25": cfg.get("include_ou25"),
                    "summary_json": str(summary_files[0]),
                    "filtered_csv": str(summary.get("artifacts", {}).get("filtered_csv", "")),
                    "market_summary_csv": str(summary.get("artifacts", {}).get("market_summary_csv", "")),
                    "league_summary_csv": str(summary.get("artifacts", {}).get("league_summary_csv", "")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no walk-forward summary rows found under: {run_root}")

    df["rows"] = pd.to_numeric(df["rows"], errors="coerce")
    df["hit"] = pd.to_numeric(df["hit"], errors="coerce")
    df["roi"] = pd.to_numeric(df["roi"], errors="coerce")
    df["avg_odds"] = pd.to_numeric(df["avg_odds"], errors="coerce")
    df["month"] = df["month"].astype(str)

    # Per-month detail output
    month_detail = df.sort_values(["branch", "month"]).reset_index(drop=True)

    # Branch summary
    summary_rows: list[dict[str, Any]] = []
    for branch, g in month_detail.groupby("branch", dropna=False):
        g = g.sort_values("month").reset_index(drop=True)

        profitable_months = int((pd.to_numeric(g["roi"], errors="coerce") > 0).sum())
        losing_months = int((pd.to_numeric(g["roi"], errors="coerce") <= 0).sum())
        months_present = int(g["month"].nunique())

        roi_series = pd.to_numeric(g["roi"], errors="coerce")
        hit_series = pd.to_numeric(g["hit"], errors="coerce")
        row_series = pd.to_numeric(g["rows"], errors="coerce")

        best_roi_idx = roi_series.idxmax() if roi_series.notna().any() else None
        worst_roi_idx = roi_series.idxmin() if roi_series.notna().any() else None
        best_hit_idx = hit_series.idxmax() if hit_series.notna().any() else None
        worst_hit_idx = hit_series.idxmin() if hit_series.notna().any() else None

        summary_rows.append(
            {
                "branch": branch,
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
                "top_q": float(pd.to_numeric(g["top_q"], errors="coerce").mean()) if "top_q" in g.columns else float("nan"),
                "ou25_band1_low": float(pd.to_numeric(g["ou25_band1_low"], errors="coerce").mean()),
                "ou25_band1_high": float(pd.to_numeric(g["ou25_band1_high"], errors="coerce").mean()),
                "ou25_band2_low": float(pd.to_numeric(g["ou25_band2_low"], errors="coerce").mean()),
                "ou25_band2_high": float(pd.to_numeric(g["ou25_band2_high"], errors="coerce").mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["weighted_roi", "weighted_hit", "total_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Write CSV as combined workbook-style flat file: first summary, then month detail
    summary_csv = OUT_CSV
    summary_df.to_csv(summary_csv, index=False)

    # Markdown
    top_cols = [
        "branch",
        "months_present",
        "total_rows",
        "weighted_hit",
        "weighted_roi",
        "weighted_avg_odds",
        "profitable_months",
        "losing_months",
        "worst_roi",
        "max_drawdown",
    ]
    top_table = summary_df[top_cols].copy()

    md_lines: list[str] = []
    md_lines.append("# OU25 Walk-Forward Comparison")
    md_lines.append("")
    md_lines.append(f"- Run root: `{run_root}`")
    md_lines.append(f"- Branches: `{summary_df['branch'].nunique()}`")
    md_lines.append(f"- Months covered: `{month_detail['month'].nunique()}`")
    md_lines.append("")
    md_lines.append("## Branch leaderboard")
    md_lines.append("")
    md_lines.append(_to_markdown(top_table))
    md_lines.append("")
    md_lines.append("## Month-by-month detail")
    md_lines.append("")
    md_lines.append(_to_markdown(month_detail[[
        "month", "branch", "rows", "hit", "roi", "avg_odds"
    ]].sort_values(["branch", "month"]).reset_index(drop=True)))

    OUT_MD.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWROTE: {OUT_CSV}")
    print(f"WROTE: {OUT_MD}")


if __name__ == "__main__":
    main()