
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/ou25_walkforward")
OUT_LEAGUE_CSV = Path("over25_from_ou25_league_audit.csv")
OUT_LEAGUE_MD = Path("over25_from_ou25_league_audit.md")
OUT_BRANCH_CSV = Path("over25_from_ou25_branch_audit.csv")
OUT_BRANCH_MD = Path("over25_from_ou25_branch_audit.md")
OUT_POLICY_JSON = Path("over25_from_ou25_policy.json")

TARGET_BRANCHES = {
    "ou25_combined_baseline",
    "ou25_combined_topq_080",
    "ou25_mode_over_only",
    "ou25_band2_178_195",
    "ou25_band1_124_176",
}


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


def _find_latest_run_root(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"walk-forward root not found: {root}")
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    if not run_dirs:
        raise SystemExit(f"no run directories found under: {root}")
    return run_dirs[0]


def _pick_filtered_csv_from_summary(summary_path: Path) -> Path | None:
    summary = _read_json(summary_path)
    art = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    filtered_csv = art.get("filtered_csv")
    if filtered_csv:
        p = Path(str(filtered_csv))
        if p.exists():
            return p
    candidates = [
        p for p in summary_path.parent.glob("*.csv")
        if "__MARKET_SUMMARY" not in p.name
        and "__LEAGUE_SUMMARY" not in p.name
        and "__TIER_" not in p.name
    ]
    return sorted(candidates)[0] if candidates else None


def _summarise_group(g: pd.DataFrame, group_name: str) -> dict[str, Any]:
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

    return {
        group_name: g.iloc[0][group_name],
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
        "ou25_band1_low": float(pd.to_numeric(g["ou25_band1_low"], errors="coerce").mean()) if "ou25_band1_low" in g.columns else float("nan"),
        "ou25_band1_high": float(pd.to_numeric(g["ou25_band1_high"], errors="coerce").mean()) if "ou25_band1_high" in g.columns else float("nan"),
        "ou25_band2_low": float(pd.to_numeric(g["ou25_band2_low"], errors="coerce").mean()) if "ou25_band2_low" in g.columns else float("nan"),
        "ou25_band2_high": float(pd.to_numeric(g["ou25_band2_high"], errors="coerce").mean()) if "ou25_band2_high" in g.columns else float("nan"),
    }


def main() -> None:
    run_root = _find_latest_run_root(ROOT)

    month_rows: list[dict[str, Any]] = []
    league_rows: list[dict[str, Any]] = []

    for month_dir in sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: _month_sort_key(p.name)):
        month_tag = month_dir.name

        for branch_dir in sorted([p for p in month_dir.iterdir() if p.is_dir()]):
            branch = branch_dir.name
            if branch not in TARGET_BRANCHES:
                continue

            summary_files = sorted(branch_dir.glob("*__SUMMARY.json"))
            if not summary_files:
                continue

            summary_path = summary_files[0]
            summary = _read_json(summary_path)
            cfg = summary.get("config", {}) if isinstance(summary, dict) else {}

            filtered_csv = _pick_filtered_csv_from_summary(summary_path)
            if filtered_csv is None or not filtered_csv.exists():
                continue

            df = pd.read_csv(filtered_csv, low_memory=False)
            if df.empty:
                continue
            if "market" not in df.columns or "bookie_pick" not in df.columns:
                continue

            d = df.copy()
            d["market"] = d["market"].astype(str).str.lower().str.strip()
            d["bookie_pick"] = d["bookie_pick"].astype(str).str.upper().str.strip()
            d = d[(d["market"] == "ou25") & (d["bookie_pick"] == "OVER")].copy()
            if d.empty:
                continue

            d["correct"] = pd.to_numeric(d.get("correct"), errors="coerce")
            d["bookie_od"] = pd.to_numeric(d.get("bookie_od"), errors="coerce")
            scored = d[d["correct"].notna()].copy()
            roi_df = scored[scored["bookie_od"].notna()].copy()
            if scored.empty:
                continue

            month_rows.append(
                {
                    "run_tag": run_root.name,
                    "month": month_tag,
                    "branch": branch,
                    "rows": int(len(scored)),
                    "hit": float(scored["correct"].mean()),
                    "roi": float((roi_df["correct"] * roi_df["bookie_od"] - 1.0).mean()) if not roi_df.empty else float("nan"),
                    "avg_odds": float(roi_df["bookie_od"].mean()) if not roi_df.empty else float("nan"),
                    "top_q": _safe_num(cfg.get("top_q")),
                    "ou25_band1_low": _safe_num((cfg.get("ou25_band1") or [None, None])[0]),
                    "ou25_band1_high": _safe_num((cfg.get("ou25_band1") or [None, None])[1]),
                    "ou25_band2_low": _safe_num((cfg.get("ou25_band2") or [None, None])[0]),
                    "ou25_band2_high": _safe_num((cfg.get("ou25_band2") or [None, None])[1]),
                    "filtered_csv": str(filtered_csv),
                }
            )

            if "league" not in scored.columns:
                continue

            for league, g in scored.groupby("league", dropna=False):
                gr = roi_df[roi_df["league"] == league]
                league_rows.append(
                    {
                        "run_tag": run_root.name,
                        "month": month_tag,
                        "branch": branch,
                        "league": str(league),
                        "rows": int(len(g)),
                        "hit": float(g["correct"].mean()) if len(g) else float("nan"),
                        "roi": float((gr["correct"] * gr["bookie_od"] - 1.0).mean()) if len(gr) else float("nan"),
                        "avg_odds": float(gr["bookie_od"].mean()) if len(gr) else float("nan"),
                        "top_q": _safe_num(cfg.get("top_q")),
                        "ou25_band1_low": _safe_num((cfg.get("ou25_band1") or [None, None])[0]),
                        "ou25_band1_high": _safe_num((cfg.get("ou25_band1") or [None, None])[1]),
                        "ou25_band2_low": _safe_num((cfg.get("ou25_band2") or [None, None])[0]),
                        "ou25_band2_high": _safe_num((cfg.get("ou25_band2") or [None, None])[1]),
                    }
                )

    month_df = pd.DataFrame(month_rows)
    if month_df.empty:
        raise SystemExit(f"no OVER-only OU25 rows found under: {run_root}")

    branch_summary_rows: list[dict[str, Any]] = []
    for branch, g in month_df.groupby("branch", dropna=False):
        branch_summary_rows.append(_summarise_group(g, "branch"))
    branch_summary = pd.DataFrame(branch_summary_rows).sort_values(
        ["weighted_roi", "weighted_hit", "total_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    league_df = pd.DataFrame(league_rows)
    if league_df.empty:
        raise SystemExit("OVER-only audit built branch rows but no league rows were found")

    league_summary_rows: list[dict[str, Any]] = []
    for (branch, league), g in league_df.groupby(["branch", "league"], dropna=False):
        row = _summarise_group(g, "branch")
        row["league"] = league
        league_summary_rows.append(row)
    league_summary = pd.DataFrame(league_summary_rows)
    league_summary = league_summary[[
        "branch", "league", "months_present", "total_rows", "weighted_hit", "weighted_roi",
        "weighted_avg_odds", "avg_rows_per_month", "min_rows_month", "max_rows_month",
        "profitable_months", "losing_months", "best_month_by_roi", "best_roi",
        "worst_month_by_roi", "worst_roi", "best_month_by_hit", "best_hit",
        "worst_month_by_hit", "worst_hit", "roi_std", "hit_std", "max_drawdown",
        "top_q", "ou25_band1_low", "ou25_band1_high", "ou25_band2_low", "ou25_band2_high",
    ]].sort_values(["weighted_roi", "weighted_hit", "total_rows"], ascending=[False, False, False]).reset_index(drop=True)

    branch_summary.to_csv(OUT_BRANCH_CSV, index=False)
    league_summary.to_csv(OUT_LEAGUE_CSV, index=False)

    branch_md_lines: list[str] = []
    branch_md_lines.append("# Over 2.5 From Existing OU25 Walk-Forward Audit")
    branch_md_lines.append("")
    branch_md_lines.append(f"- Run root: `{run_root}`")
    branch_md_lines.append("- Filter applied: `market == ou25` and `bookie_pick == OVER`")
    branch_md_lines.append("- Purpose: determine whether filtering existing OU25 outputs is enough for Over 2.5 deployment.")
    branch_md_lines.append("")
    branch_md_lines.append("## Branch leaderboard")
    branch_md_lines.append("")
    branch_md_lines.append(_to_markdown(branch_summary[[
        "branch", "months_present", "total_rows", "weighted_hit", "weighted_roi",
        "weighted_avg_odds", "profitable_months", "losing_months", "worst_roi", "max_drawdown"
    ]]))
    branch_md_lines.append("")
    branch_md_lines.append("## Month-by-month detail")
    branch_md_lines.append("")
    branch_md_lines.append(_to_markdown(
        month_df[["month", "branch", "rows", "hit", "roi", "avg_odds"]]
        .sort_values(["branch", "month"])
        .reset_index(drop=True)
    ))
    OUT_BRANCH_MD.write_text("\n".join(branch_md_lines).strip() + "\n", encoding="utf-8")

    league_md_lines: list[str] = []
    league_md_lines.append("# Over 2.5 League Audit From Existing OU25 Walk-Forward Outputs")
    league_md_lines.append("")
    league_md_lines.append(f"- Run root: `{run_root}`")
    league_md_lines.append("- Filter applied: `market == ou25` and `bookie_pick == OVER`")
    league_md_lines.append("- Table sorted by weighted ROI, then hit rate, then total rows.")
    league_md_lines.append("")
    league_md_lines.append(_to_markdown(league_summary))
    OUT_LEAGUE_MD.write_text("\n".join(league_md_lines).strip() + "\n", encoding="utf-8")

    deploy_floor_rows = 12
    deploy_floor_months = 4
    deploy_floor_roi = 0.05

    deployable = league_summary[
        (pd.to_numeric(league_summary["total_rows"], errors="coerce") >= deploy_floor_rows)
        & (pd.to_numeric(league_summary["months_present"], errors="coerce") >= deploy_floor_months)
        & (pd.to_numeric(league_summary["weighted_roi"], errors="coerce") >= deploy_floor_roi)
    ].copy()

    policy = {
        "source": "existing_ou25_walkforward_filtered_to_over_only",
        "run_root": str(run_root),
        "filter": {
            "market": "ou25",
            "bookie_pick": "OVER",
        },
        "deploy_thresholds": {
            "min_total_rows": deploy_floor_rows,
            "min_months_present": deploy_floor_months,
            "min_weighted_roi": deploy_floor_roi,
        },
        "branch_leaderboard": branch_summary[[
            "branch", "months_present", "total_rows", "weighted_hit", "weighted_roi", "weighted_avg_odds"
        ]].to_dict(orient="records"),
        "deployable_league_rows": deployable[[
            "branch", "league", "months_present", "total_rows", "weighted_hit", "weighted_roi", "weighted_avg_odds",
            "worst_roi", "max_drawdown"
        ]].to_dict(orient="records"),
        "top_branch": str(branch_summary.iloc[0]["branch"]) if not branch_summary.empty else None,
        "top_branch_weighted_roi": float(branch_summary.iloc[0]["weighted_roi"]) if not branch_summary.empty else None,
        "recommendation": (
            "If branch-level and league-level OVER-only performance is strong enough, filtering the existing OU25 product may be sufficient. "
            "If the filtered OVER-only audit is too thin or unstable in target leagues, build a dedicated Over 2.5 model next."
        ),
        "artifacts": {
            "branch_csv": str(OUT_BRANCH_CSV),
            "branch_md": str(OUT_BRANCH_MD),
            "league_csv": str(OUT_LEAGUE_CSV),
            "league_md": str(OUT_LEAGUE_MD),
        },
    }
    OUT_POLICY_JSON.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print(branch_summary.to_string(index=False))
    print(f"\nWROTE: {OUT_BRANCH_CSV}")
    print(f"WROTE: {OUT_BRANCH_MD}")
    print(f"WROTE: {OUT_LEAGUE_CSV}")
    print(f"WROTE: {OUT_LEAGUE_MD}")
    print(f"WROTE: {OUT_POLICY_JSON}")


if __name__ == "__main__":
    main()