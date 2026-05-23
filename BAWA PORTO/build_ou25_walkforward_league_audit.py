#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/ou25_walkforward")
OUT_CSV = Path("ou25_walkforward_league_audit.csv")
OUT_MD = Path("ou25_walkforward_league_audit.md")


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


def _pick_latest_run_root(root: Path) -> Path:
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    if not run_dirs:
        raise SystemExit(f"no run directories found under: {root}")
    return run_dirs[0]


def _find_branch_outputs(branch_dir: Path) -> tuple[Path | None, Path | None]:
    league_summary_files = sorted(branch_dir.glob("*__LEAGUE_SUMMARY.csv"))
    summary_json_files = sorted(branch_dir.glob("*__SUMMARY.json"))
    league_summary_csv = league_summary_files[0] if league_summary_files else None
    summary_json = summary_json_files[0] if summary_json_files else None
    return league_summary_csv, summary_json


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"walk-forward root not found: {ROOT}")

    run_root = _pick_latest_run_root(ROOT)
    rows: list[dict[str, Any]] = []

    month_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    month_dirs = sorted(month_dirs, key=lambda p: _month_sort_key(p.name))

    for month_dir in month_dirs:
        month_tag = month_dir.name
        branch_dirs = sorted([p for p in month_dir.iterdir() if p.is_dir()])

        for branch_dir in branch_dirs:
            league_summary_csv, summary_json_path = _find_branch_outputs(branch_dir)
            if league_summary_csv is None:
                continue

            cfg: dict[str, Any] = {}
            tag = ""
            filtered_csv = ""
            if summary_json_path is not None:
                payload = _read_json(summary_json_path)
                cfg = payload.get("config", {}) or {}
                tag = str(payload.get("tag", "") or "")
                filtered_csv = str((payload.get("artifacts", {}) or {}).get("filtered_csv", "") or "")

            league_df = pd.read_csv(league_summary_csv, low_memory=False)
            if league_df.empty:
                continue

            required_cols = {"league", "rows", "hit", "roi", "avg_od"}
            missing = [c for c in required_cols if c not in league_df.columns]
            if missing:
                raise SystemExit(f"league summary missing required columns {missing}: {league_summary_csv}")

            league_df = league_df.copy()
            league_df["league"] = league_df["league"].astype(str)
            league_df["rows"] = pd.to_numeric(league_df["rows"], errors="coerce")
            league_df["hit"] = pd.to_numeric(league_df["hit"], errors="coerce")
            league_df["roi"] = pd.to_numeric(league_df["roi"], errors="coerce")
            league_df["avg_od"] = pd.to_numeric(league_df["avg_od"], errors="coerce")

            for _, r in league_df.iterrows():
                rows.append(
                    {
                        "run_tag": run_root.name,
                        "month": month_tag,
                        "branch": branch_dir.name,
                        "tag": tag,
                        "league": r.get("league"),
                        "rows": _safe_num(r.get("rows")),
                        "hit": _safe_num(r.get("hit")),
                        "roi": _safe_num(r.get("roi")),
                        "avg_odds": _safe_num(r.get("avg_od")),
                        "n_bookie": _safe_num(r.get("n_bookie")),
                        "n_model_fair": _safe_num(r.get("n_model_fair")),
                        "n_unknown": _safe_num(r.get("n_unknown")),
                        "top_q": _safe_num(cfg.get("top_q")),
                        "ou25_band1_low": _safe_num((cfg.get("ou25_band1") or [None, None])[0]),
                        "ou25_band1_high": _safe_num((cfg.get("ou25_band1") or [None, None])[1]),
                        "ou25_band2_low": _safe_num((cfg.get("ou25_band2") or [None, None])[0]),
                        "ou25_band2_high": _safe_num((cfg.get("ou25_band2") or [None, None])[1]),
                        "include_ou25": cfg.get("include_ou25"),
                        "league_summary_csv": str(league_summary_csv),
                        "summary_json": str(summary_json_path) if summary_json_path is not None else "",
                        "filtered_csv": filtered_csv,
                    }
                )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit(f"no league-level walk-forward rows found under: {run_root}")

    detail_df["rows"] = pd.to_numeric(detail_df["rows"], errors="coerce")
    detail_df["hit"] = pd.to_numeric(detail_df["hit"], errors="coerce")
    detail_df["roi"] = pd.to_numeric(detail_df["roi"], errors="coerce")
    detail_df["avg_odds"] = pd.to_numeric(detail_df["avg_odds"], errors="coerce")
    detail_df["month"] = detail_df["month"].astype(str)
    detail_df["league"] = detail_df["league"].astype(str)
    detail_df["branch"] = detail_df["branch"].astype(str)

    summary_rows: list[dict[str, Any]] = []
    grouped = detail_df.groupby(["branch", "league"], dropna=False)

    for (branch, league), g in grouped:
        g = g.sort_values("month").reset_index(drop=True)
        roi_series = pd.to_numeric(g["roi"], errors="coerce")
        hit_series = pd.to_numeric(g["hit"], errors="coerce")
        row_series = pd.to_numeric(g["rows"], errors="coerce")

        profitable_months = int((roi_series > 0).sum())
        losing_months = int((roi_series <= 0).sum())
        months_present = int(g["month"].nunique())

        best_roi_idx = roi_series.idxmax() if roi_series.notna().any() else None
        worst_roi_idx = roi_series.idxmin() if roi_series.notna().any() else None
        best_hit_idx = hit_series.idxmax() if hit_series.notna().any() else None
        worst_hit_idx = hit_series.idxmin() if hit_series.notna().any() else None

        summary_rows.append(
            {
                "branch": branch,
                "league": league,
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
                "top_q": float(pd.to_numeric(g["top_q"], errors="coerce").mean()),
                "ou25_band1_low": float(pd.to_numeric(g["ou25_band1_low"], errors="coerce").mean()),
                "ou25_band1_high": float(pd.to_numeric(g["ou25_band1_high"], errors="coerce").mean()),
                "ou25_band2_low": float(pd.to_numeric(g["ou25_band2_low"], errors="coerce").mean()),
                "ou25_band2_high": float(pd.to_numeric(g["ou25_band2_high"], errors="coerce").mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["weighted_roi", "weighted_hit", "total_rows", "months_present"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_CSV, index=False)

    top_table = summary_df[
        [
            "branch",
            "league",
            "months_present",
            "total_rows",
            "weighted_hit",
            "weighted_roi",
            "profitable_months",
            "losing_months",
            "worst_roi",
            "max_drawdown",
        ]
    ].copy()

    stable_table = summary_df[
        (summary_df["months_present"] >= 3) & (summary_df["profitable_months"] == summary_df["months_present"])
    ].copy()
    stable_table = stable_table.head(20)

    fragile_table = summary_df.copy()
    fragile_table = fragile_table.sort_values(["worst_roi", "weighted_roi", "months_present"], ascending=[True, False, False]).head(20)

    detail_view = detail_df[
        ["month", "branch", "league", "rows", "hit", "roi", "avg_odds"]
    ].sort_values(["branch", "league", "month"]).reset_index(drop=True)

    md_lines: list[str] = []
    md_lines.append("# OU25 Walk-Forward League Audit")
    md_lines.append("")
    md_lines.append(f"- Run root: `{run_root}`")
    md_lines.append(f"- Branches: `{summary_df['branch'].nunique()}`")
    md_lines.append(f"- Leagues: `{summary_df['league'].nunique()}`")
    md_lines.append(f"- Months covered: `{detail_df['month'].nunique()}`")
    md_lines.append("")
    md_lines.append("## League leaderboard")
    md_lines.append("")
    md_lines.append(_to_markdown(top_table.head(50)))
    md_lines.append("")
    md_lines.append("## Stable leagues")
    md_lines.append("")
    md_lines.append(_to_markdown(stable_table[[
        "branch", "league", "months_present", "total_rows", "weighted_hit", "weighted_roi", "worst_roi", "max_drawdown"
    ]]))
    md_lines.append("")
    md_lines.append("## Fragile leagues")
    md_lines.append("")
    md_lines.append(_to_markdown(fragile_table[[
        "branch", "league", "months_present", "total_rows", "weighted_hit", "weighted_roi", "worst_roi", "max_drawdown"
    ]]))
    md_lines.append("")
    md_lines.append("## Month-by-month league detail")
    md_lines.append("")
    md_lines.append(_to_markdown(detail_view))

    OUT_MD.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWROTE: {OUT_CSV}")
    print(f"WROTE: {OUT_MD}")


if __name__ == "__main__":
    main()