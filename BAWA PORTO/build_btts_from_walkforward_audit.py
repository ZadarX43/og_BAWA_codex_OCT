

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/walk_forward")
OUT_CSV = Path("btts_walkforward_branch_audit.csv")
OUT_MD = Path("btts_walkforward_branch_audit.md")


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


def _find_btts_summary(month_dir: Path) -> Path | None:
    p = month_dir / "investor_table_btts_valueEV.csv"
    if p.exists():
        return p
    return None


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"walk-forward root not found: {ROOT}")

    rows: list[dict[str, Any]] = []

    month_dirs = sorted(
        [p for p in ROOT.iterdir() if p.is_dir()],
        key=lambda p: _month_sort_key(p.name),
    )

    for month_dir in month_dirs:
        month_tag = month_dir.name
        src = _find_btts_summary(month_dir)
        if src is None:
            continue

        df = pd.read_csv(src)
        if df.empty:
            continue

        required = {"league", "n_btts", "hit_btts", "roi_btts", "avg_od_btts"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{src} missing required columns: {sorted(missing)}")

        d = df.copy()
        d["league"] = d["league"].astype(str)
        d["rows"] = pd.to_numeric(d["n_btts"], errors="coerce")
        d["hit"] = pd.to_numeric(d["hit_btts"], errors="coerce")
        d["roi"] = pd.to_numeric(d["roi_btts"], errors="coerce")
        d["avg_odds"] = pd.to_numeric(d["avg_od_btts"], errors="coerce")
        d["month"] = month_tag
        d["branch"] = "btts_valueev"
        d["source_csv"] = str(src)

        for _, r in d.iterrows():
            rows.append(
                {
                    "month": month_tag,
                    "branch": "btts_valueev",
                    "league": str(r["league"]),
                    "rows": _safe_num(r["rows"]),
                    "hit": _safe_num(r["hit"]),
                    "roi": _safe_num(r["roi"]),
                    "avg_odds": _safe_num(r["avg_odds"]),
                    "source_csv": str(src),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no BTTS walk-forward rows found under: {ROOT}")

    df["rows"] = pd.to_numeric(df["rows"], errors="coerce")
    df["hit"] = pd.to_numeric(df["hit"], errors="coerce")
    df["roi"] = pd.to_numeric(df["roi"], errors="coerce")
    df["avg_odds"] = pd.to_numeric(df["avg_odds"], errors="coerce")
    df["month"] = df["month"].astype(str)
    df["league"] = df["league"].astype(str)

    month_detail = df.sort_values(["branch", "league", "month"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for branch, g in month_detail.groupby("branch", dropna=False):
        g = g.sort_values("month").reset_index(drop=True)

        profitable_months = int((pd.to_numeric(g.groupby("month")["roi"].mean(), errors="coerce") > 0).sum())
        losing_months = int((pd.to_numeric(g.groupby("month")["roi"].mean(), errors="coerce") <= 0).sum())
        months_present = int(g["month"].nunique())

        roi_series = pd.to_numeric(g.groupby("month")["roi"].mean(), errors="coerce")
        hit_series = pd.to_numeric(g.groupby("month")["hit"].mean(), errors="coerce")
        row_series = pd.to_numeric(g.groupby("month")["rows"].sum(), errors="coerce")
        month_index = list(g.groupby("month", dropna=False).size().index)

        best_roi_idx = roi_series.idxmax() if roi_series.notna().any() else None
        worst_roi_idx = roi_series.idxmin() if roi_series.notna().any() else None
        best_hit_idx = hit_series.idxmax() if hit_series.notna().any() else None
        worst_hit_idx = hit_series.idxmin() if hit_series.notna().any() else None

        summary_rows.append(
            {
                "branch": branch,
                "months_present": months_present,
                "total_rows": int(pd.to_numeric(g["rows"], errors="coerce").fillna(0).sum()),
                "weighted_hit": _weighted_mean(g, "hit"),
                "weighted_roi": _weighted_mean(g, "roi"),
                "weighted_avg_odds": _weighted_mean(g, "avg_odds"),
                "avg_rows_per_month": float(row_series.mean()) if row_series.notna().any() else float("nan"),
                "min_rows_month": float(row_series.min()) if row_series.notna().any() else float("nan"),
                "max_rows_month": float(row_series.max()) if row_series.notna().any() else float("nan"),
                "profitable_months": profitable_months,
                "losing_months": losing_months,
                "best_month_by_roi": str(best_roi_idx) if best_roi_idx is not None else "",
                "best_roi": float(roi_series.loc[best_roi_idx]) if best_roi_idx is not None else float("nan"),
                "worst_month_by_roi": str(worst_roi_idx) if worst_roi_idx is not None else "",
                "worst_roi": float(roi_series.loc[worst_roi_idx]) if worst_roi_idx is not None else float("nan"),
                "best_month_by_hit": str(best_hit_idx) if best_hit_idx is not None else "",
                "best_hit": float(hit_series.loc[best_hit_idx]) if best_hit_idx is not None else float("nan"),
                "worst_month_by_hit": str(worst_hit_idx) if worst_hit_idx is not None else "",
                "worst_hit": float(hit_series.loc[worst_hit_idx]) if worst_hit_idx is not None else float("nan"),
                "roi_std": float(roi_series.std(ddof=0)) if roi_series.notna().any() else float("nan"),
                "hit_std": float(hit_series.std(ddof=0)) if hit_series.notna().any() else float("nan"),
                "max_drawdown": _max_drawdown_from_monthly_roi(roi_series),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["weighted_roi", "weighted_hit", "total_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_CSV, index=False)

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
    md_lines.append("# BTTS Walk-Forward Branch Audit")
    md_lines.append("")
    md_lines.append(f"- Walk-forward root: `{ROOT}`")
    md_lines.append(f"- Branches: `{summary_df['branch'].nunique()}`")
    md_lines.append(f"- Months covered: `{month_detail['month'].nunique()}`")
    md_lines.append("")
    md_lines.append("## Branch leaderboard")
    md_lines.append("")
    md_lines.append(_to_markdown(top_table))
    md_lines.append("")
    md_lines.append("## Month-by-month league detail")
    md_lines.append("")
    md_lines.append(
        _to_markdown(
            month_detail[["month", "league", "branch", "rows", "hit", "roi", "avg_odds"]]
            .sort_values(["league", "month"])
            .reset_index(drop=True)
        )
    )

    OUT_MD.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWROTE: {OUT_CSV}")
    print(f"WROTE: {OUT_MD}")


if __name__ == "__main__":
    main()