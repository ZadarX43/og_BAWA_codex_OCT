#!/usr/bin/env python3
"""UNDER25 + BTTS NO audit scaffold (walk-forward).

Outputs:
  - UNDER25_BTTS_NO_OVERLAP*.csv (global/by league/live-only)
  - BTTS_NO_UNDER25_OVERLAP*.csv (global/by league/live-only)
  - *_ALLOWLIST_THRESHOLDS.csv
  - *_RESCUE_COMBINED_AUDIT.csv (existing live / rescued / combined)
  - *_RESCUE_COMBINED_AUDIT__BY_LEAGUE.csv
  - UNDER25_BTTS_NO_PRODUCTION_SUMMARY.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("predictions_output/walk_forward")
DEFAULT_OUT = DEFAULT_BASE / "_MASTER" / "UNDER25_BTTS_NO_AUDIT"


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_pick(s: pd.Series) -> pd.Series:
    txt = s.astype("string").fillna("").str.upper().str.strip()
    txt = txt.replace({
        "UNDER": "UNDER25",
        "UNDER 2.5": "UNDER25",
        "UNDER2.5": "UNDER25",
        "U25": "UNDER25",
        "U2.5": "UNDER25",
        "UNDER2,5": "UNDER25",
        "ECOVER_REBUILT_.PY": "UNDER25",
    })
    txt = txt.replace({
        "OVER": "OVER25",
        "OVER 2.5": "OVER25",
        "OVER2.5": "OVER25",
        "O25": "OVER25",
        "O2.5": "OVER25",
    })
    txt = txt.replace({
        "YES": "YES",
        "Y": "YES",
        "NO": "NO",
        "N": "NO",
        "BTTS NO": "NO",
        "BOTH TEAMS TO SCORE NO": "NO",
    })
    return txt


def _profit_series(hit: pd.Series, bookie_od: pd.Series) -> pd.Series:
    h = _safe_num(hit)
    od = _safe_num(bookie_od)
    return np.where(h == 1, od - 1.0, np.where(h == 0, -1.0, np.nan))


def _summarize(df: pd.DataFrame, hit_col: str) -> dict[str, object]:
    hit = _safe_num(df.get(hit_col, np.nan))
    graded = hit.notna().sum()
    wins = (hit == 1).sum()
    losses = (hit == 0).sum()
    profit = _profit_series(hit, df.get("bookie_od", np.nan))
    profit_sum = float(np.nansum(profit))
    roi = float(profit_sum / graded) if graded else float("nan")
    return {
        "rows": int(len(df)),
        "graded": int(graded),
        "wins": float(wins),
        "losses": float(losses),
        "hit_rate": float(wins / graded) if graded else float("nan"),
        "roi": roi,
        "profit": profit_sum,
        "avg_bookie_od": float(_safe_num(df.get("bookie_od", np.nan)).mean()),
        "avg_model_p": float(_safe_num(df.get("model_p_for_bookie", np.nan)).mean()),
    }


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    safe = df.loc[:, cols].copy()
    safe = safe.fillna("")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in safe.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _overlap_tables(
    baseline: pd.DataFrame,
    overlap_keys: set[tuple[str, str]],
    hit_col: str,
    label: str,
    outdir: Path,
    live_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if baseline.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    league = baseline.get("league", pd.Series("", index=baseline.index)).astype("string").fillna("")
    fx = baseline.get("fixture_key", pd.Series("", index=baseline.index)).astype("string").fillna("")
    keys = list(zip(league, fx))
    overlap_mask = pd.Series([k in overlap_keys for k in keys], index=baseline.index)

    base_sum = _summarize(baseline, hit_col)
    overlap = baseline.loc[overlap_mask].copy()
    overlap_sum = _summarize(overlap, hit_col)

    global_row = {
        "baseline_rows": base_sum["rows"],
        "overlap_rows": overlap_sum["rows"],
        "overlap_coverage_pct": round((overlap_sum["rows"] / base_sum["rows"]) * 100, 2) if base_sum["rows"] else 0.0,
        "baseline_hit_rate": base_sum["hit_rate"],
        "overlap_hit_rate": overlap_sum["hit_rate"],
        "hit_rate_lift": overlap_sum["hit_rate"] - base_sum["hit_rate"],
        "baseline_roi": base_sum["roi"],
        "overlap_roi": overlap_sum["roi"],
        "roi_lift": overlap_sum["roi"] - base_sum["roi"],
    }
    global_df = pd.DataFrame([global_row])

    rows = []
    for lg, g in baseline.groupby("league", dropna=False):
        lg = "" if pd.isna(lg) else lg
        g_keys = list(zip(
            g.get("league", pd.Series("", index=g.index)).astype("string").fillna(""),
            g.get("fixture_key", pd.Series("", index=g.index)).astype("string").fillna("")
        ))
        g_overlap_mask = pd.Series([k in overlap_keys for k in g_keys], index=g.index)
        g_overlap = g.loc[g_overlap_mask].copy()
        g_base = _summarize(g, hit_col)
        g_over = _summarize(g_overlap, hit_col)
        rows.append({
            "league": lg,
            "baseline_rows": g_base["rows"],
            "overlap_rows": g_over["rows"],
            "overlap_coverage_pct": round((g_over["rows"] / g_base["rows"]) * 100, 2) if g_base["rows"] else 0.0,
            "baseline_hit_rate": g_base["hit_rate"],
            "overlap_hit_rate": g_over["hit_rate"],
            "hit_rate_lift": g_over["hit_rate"] - g_base["hit_rate"],
            "baseline_roi": g_base["roi"],
            "overlap_roi": g_over["roi"],
            "roi_lift": g_over["roi"] - g_base["roi"],
        })
    by_league = pd.DataFrame(rows).sort_values("roi_lift", ascending=False, na_position="last").reset_index(drop=True)

    suffix = "__LIVE_ONLY" if live_only else ""
    global_df.to_csv(outdir / f"{label}{suffix}.csv", index=False)
    by_league.to_csv(outdir / f"{label}{suffix}__BY_LEAGUE.csv", index=False)

    return global_df, by_league, overlap


def _allowlist_from_league(
    by_league: pd.DataFrame,
    min_overlap_rows: int,
    min_roi_lift: float,
    min_overlap_roi: float,
    out_path: Path,
) -> pd.DataFrame:
    if by_league.empty:
        out = pd.DataFrame()
        out.to_csv(out_path, index=False)
        return out
    out = by_league.loc[
        (by_league["overlap_rows"] >= min_overlap_rows)
        & (by_league["roi_lift"] >= min_roi_lift)
        & (by_league["overlap_roi"] >= min_overlap_roi)
    ].copy()
    out["min_overlap_rows"] = min_overlap_rows
    out["min_roi_lift"] = min_roi_lift
    out["min_overlap_roi"] = min_overlap_roi
    out.to_csv(out_path, index=False)
    return out


def _rescue_combined(
    baseline_live: pd.DataFrame,
    baseline_all: pd.DataFrame,
    rescue_mask: pd.Series,
    hit_col: str,
    label: str,
    outdir: Path,
) -> pd.DataFrame:
    rescued = baseline_all.loc[rescue_mask].copy()
    combined = pd.concat([baseline_live, rescued], axis=0, ignore_index=True) if not rescued.empty else baseline_live

    live_sum = _summarize(baseline_live, hit_col)
    rescue_sum = _summarize(rescued, hit_col)
    combined_sum = _summarize(combined, hit_col)

    rows = [
        {"cohort": "existing_live_only", **live_sum},
        {"cohort": "rescued_only", **rescue_sum},
        {"cohort": "combined_live_plus_rescued", **combined_sum},
    ]
    out = pd.DataFrame(rows)
    out["delta_hit_rate_vs_existing_live"] = out["hit_rate"] - live_sum["hit_rate"]
    out["delta_roi_vs_existing_live"] = out["roi"] - live_sum["roi"]
    out.to_csv(outdir / f"{label}_RESCUE_COMBINED_AUDIT.csv", index=False)

    # by league
    rows_lg = []
    for lg, g in baseline_all.groupby("league", dropna=False):
        lg = "" if pd.isna(lg) else lg
        g_live = g.loc[baseline_live.index.intersection(g.index)].copy()
        g_rescue = g.loc[rescue_mask.loc[g.index]].copy()
        g_combined = pd.concat([g_live, g_rescue], axis=0, ignore_index=True) if not g_rescue.empty else g_live

        g_live_sum = _summarize(g_live, hit_col)
        g_rescue_sum = _summarize(g_rescue, hit_col)
        g_combined_sum = _summarize(g_combined, hit_col)
        rows_lg.append({
            "league": lg,
            "cohort": "existing_live_only",
            **g_live_sum,
            "delta_hit_rate_vs_existing_live": 0.0,
            "delta_roi_vs_existing_live": 0.0,
        })
        rows_lg.append({
            "league": lg,
            "cohort": "rescued_only",
            **g_rescue_sum,
            "delta_hit_rate_vs_existing_live": g_rescue_sum["hit_rate"] - g_live_sum["hit_rate"],
            "delta_roi_vs_existing_live": g_rescue_sum["roi"] - g_live_sum["roi"],
        })
        rows_lg.append({
            "league": lg,
            "cohort": "combined_live_plus_rescued",
            **g_combined_sum,
            "delta_hit_rate_vs_existing_live": g_combined_sum["hit_rate"] - g_live_sum["hit_rate"],
            "delta_roi_vs_existing_live": g_combined_sum["roi"] - g_live_sum["roi"],
        })
    pd.DataFrame(rows_lg).to_csv(outdir / f"{label}_RESCUE_COMBINED_AUDIT__BY_LEAGUE.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="UNDER25 + BTTS NO audit scaffold")
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--min-overlap-rows", type=int, default=50)
    ap.add_argument("--min-roi-lift", type=float, default=0.10)
    ap.add_argument("--min-overlap-roi", type=float, default=0.05)
    ap.add_argument("--include-overlap", action="store_true", help="Also write overlap/rescue outputs.")
    ap.add_argument("--top-roi-n", type=int, default=10)
    ap.add_argument("--top-roi-min-graded", type=int, default=50)
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use candidate-scored universe. UNDER25 research requires RAW candidates.
    candidate_after = sorted(base.glob("w*/03_scored/DEPLOY_CANDIDATES_AFTER_GATES_SCORED_*.csv"))
    candidate_raw = sorted(base.glob("w*/03_scored/DEPLOY_CANDIDATES_RAW_SCORED_*.csv"))

    if not candidate_raw:
        raise SystemExit(
            "No RAW candidate-scored files found. Refusing to fall back to deploy-scored outputs.\n"
            "Expected:\n"
            "  - w*/03_scored/DEPLOY_CANDIDATES_RAW_SCORED_*.csv\n"
            "Please re-run walkforward with --score-candidates."
        )

    if not candidate_after:
        print("[under25_btts_no_audit] WARNING: no AFTER_GATES candidate-scored files found; live-only views may be limited.")

    print(f"[under25_btts_no_audit] using source: DEPLOY_CANDIDATES_RAW_SCORED (n={len(candidate_raw)})")
    df = pd.concat([pd.read_csv(p) for p in candidate_raw], axis=0, ignore_index=True)

    # Helper to normalize picks + score hits
    def _prep(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        mk = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
        pick_raw_local = out.get("bookie_pick", out.get("selection", pd.Series("", index=out.index)))
        pick_local = _normalize_pick(pick_raw_local)
        out["__pick_norm"] = pick_local
        if "actual_over25" in out.columns:
            ou = pd.to_numeric(out["actual_over25"], errors="coerce")
            out["under25_hit"] = np.where(
                mk.eq("ou25") & pick_local.eq("UNDER25") & ou.notna(),
                (ou == 0).astype(float),
                np.nan,
            )
        else:
            out["under25_hit"] = np.nan
        if "actual_btts_yes" in out.columns:
            ab = _safe_num(out["actual_btts_yes"])
            out["btts_no_hit"] = np.where(ab.notna(), (ab == 0).astype(int), np.nan)
        else:
            out["btts_no_hit"] = np.nan
        return out

    df = _prep(df)

    # Use AFTER_GATES for live overlap keys when available
    df_after = pd.DataFrame()
    if candidate_after:
        df_after = pd.concat([pd.read_csv(p) for p in candidate_after], axis=0, ignore_index=True)
        df_after = _prep(df_after)

    # Live mask
    if not df_after.empty:
        if "deploy_tier" in df_after.columns:
            live_mask_after = df_after["deploy_tier"].astype("string").fillna("").str.upper().isin(["ELITE", "STANDARD"])
        else:
            live_mask_after = _safe_num(df_after.get("deploy_pass", pd.Series(0, index=df_after.index))) == 1
    else:
        if "deploy_tier" in df.columns:
            live_mask_after = df["deploy_tier"].astype("string").fillna("").str.upper().isin(["ELITE", "STANDARD"])
        else:
            live_mask_after = _safe_num(df.get("deploy_pass", pd.Series(0, index=df.index))) == 1

    # Live mask for raw (kept for baseline live_only rows)
    if "deploy_tier" in df.columns:
        live_mask = df["deploy_tier"].astype("string").fillna("").str.upper().isin(["ELITE", "STANDARD"])
    else:
        live_mask = _safe_num(df.get("deploy_pass", pd.Series(0, index=df.index))) == 1

    market = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
    pick = df.get("__pick_norm", _normalize_pick(df.get("bookie_pick", df.get("selection", pd.Series("", index=df.index)))))

    # UNDER25 baseline (OU25 market)
    under_mask = market.eq("ou25") & pick.eq("UNDER25")
    under = df.loc[under_mask].copy()
    under_live = under.loc[live_mask.loc[under.index]].copy()

    # BTTS NO baseline
    btts_mask = market.eq("btts") & pick.eq("NO")
    btts_no = df.loc[btts_mask].copy()
    btts_no_live = btts_no.loc[live_mask.loc[btts_no.index]].copy()

    # Baseline audits (separate lanes)
    under_base = _summarize(under, "under25_hit")
    under_live_sum = _summarize(under_live, "under25_hit")
    pd.DataFrame([
        {"cohort": "baseline_all", **under_base},
        {"cohort": "live_only", **under_live_sum},
    ]).to_csv(outdir / "UNDER25_BASELINE_AUDIT.csv", index=False)

    btts_base = _summarize(btts_no, "btts_no_hit")
    btts_live_sum = _summarize(btts_no_live, "btts_no_hit")
    pd.DataFrame([
        {"cohort": "baseline_all", **btts_base},
        {"cohort": "live_only", **btts_live_sum},
    ]).to_csv(outdir / "BTTS_NO_BASELINE_AUDIT.csv", index=False)

    # Compact global summaries
    pd.DataFrame([{"market": "UNDER25", **under_base}]).to_csv(outdir / "UNDER25_BASELINE_AUDIT__SUMMARY.csv", index=False)
    pd.DataFrame([{"market": "BTTS_NO", **btts_base}]).to_csv(outdir / "BTTS_NO_BASELINE_AUDIT__SUMMARY.csv", index=False)

    # Baseline audits by league
    def _baseline_by_league(frame: pd.DataFrame, hit_col: str, label: str) -> None:
        if frame.empty:
            pd.DataFrame().to_csv(outdir / label, index=False)
            return
        rows = []
        for lg, sub in frame.groupby(frame.get("league", pd.Series("", index=frame.index)).astype("string").fillna(""), dropna=False):
            row = {"league": lg}
            row.update(_summarize(sub, hit_col))
            rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / label, index=False)

    _baseline_by_league(under, "under25_hit", "UNDER25_BASELINE_AUDIT__BY_LEAGUE.csv")
    _baseline_by_league(btts_no, "btts_no_hit", "BTTS_NO_BASELINE_AUDIT__BY_LEAGUE.csv")

    # Top ROI league shortlists
    def _top_roi(df: pd.DataFrame, hit_col: str, out_name: str) -> pd.DataFrame:
        if df.empty:
            pd.DataFrame().to_csv(outdir / out_name, index=False)
            return pd.DataFrame()
        rows = []
        for lg, sub in df.groupby(df.get("league", pd.Series("", index=df.index)).astype("string").fillna(""), dropna=False):
            row = {"league": lg}
            row.update(_summarize(sub, hit_col))
            rows.append(row)
        out = pd.DataFrame(rows)
        if out.empty:
            out.to_csv(outdir / out_name, index=False)
            return out
        filtered = out.loc[pd.to_numeric(out.get("graded", 0), errors="coerce").fillna(0) >= args.top_roi_min_graded].copy()
        if filtered.empty:
            filtered = out.copy()
        filtered = filtered.sort_values(["roi", "hit_rate"], ascending=[False, False]).head(args.top_roi_n)
        filtered.to_csv(outdir / out_name, index=False)
        return filtered

    under_top = _top_roi(under, "under25_hit", "UNDER25_TOP_ROI_LEAGUES.csv")
    btts_top = _top_roi(btts_no, "btts_no_hit", "BTTS_NO_TOP_ROI_LEAGUES.csv")

    if _safe_num(under.get("under25_hit", pd.Series(np.nan, index=under.index))).notna().sum() == 0:
        print("[under25_btts_no_audit] WARNING: UNDER25 rows have no hit labels in scored files; overlap/rescue for UNDER25 will be empty.")

    # Build live overlap key sets
    def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
        if frame.empty:
            return set()
        lg = frame.get("league", pd.Series("", index=frame.index)).astype("string").fillna("")
        fx = frame.get("fixture_key", pd.Series("", index=frame.index)).astype("string").fillna("")
        return set(zip(lg, fx))

    # Live keys should come from AFTER_GATES universe when available.
    if not df_after.empty:
        mk_after = df_after.get("market", pd.Series("", index=df_after.index)).astype("string").fillna("").str.lower().str.strip()
        pick_after = df_after.get("__pick_norm", _normalize_pick(df_after.get("bookie_pick", df_after.get("selection", pd.Series("", index=df_after.index)))))
        under_after = df_after.loc[mk_after.eq("ou25") & pick_after.eq("UNDER25")].copy()
        btts_after = df_after.loc[mk_after.eq("btts") & pick_after.eq("NO")].copy()
        under_live_after = under_after.loc[live_mask_after.loc[under_after.index]].copy()
        btts_live_after = btts_after.loc[live_mask_after.loc[btts_after.index]].copy()
        btts_no_live_keys = _key_set(btts_live_after)
        under_live_keys = _key_set(under_live_after)
    else:
        btts_no_live_keys = _key_set(btts_no_live)
        under_live_keys = _key_set(under_live)

    if args.include_overlap:
        # Support overlap: UNDER25 conditional on BTTS NO live
        under_global, under_by_league, _ = _overlap_tables(
            under, btts_no_live_keys, "under25_hit", "UNDER25_CONDITIONAL_ON_BTTS_NO", outdir, live_only=False
        )
        under_live_global, under_live_by_league, _ = _overlap_tables(
            under_live, btts_no_live_keys, "under25_hit", "UNDER25_CONDITIONAL_ON_BTTS_NO", outdir, live_only=True
        )
        if not under_global.empty:
            under_global.to_csv(outdir / "UNDER25_CONDITIONAL_ON_BTTS_NO.csv", index=False)

        # Live-only summary (single row)
        if not under_live_global.empty:
            under_live_global.to_csv(outdir / "UNDER25_CONDITIONAL_ON_BTTS_NO__LIVE_ONLY.csv", index=False)

        # Support overlap: BTTS NO conditional on UNDER25 live
        btts_global, btts_by_league, _ = _overlap_tables(
            btts_no, under_live_keys, "btts_no_hit", "BTTS_NO_CONDITIONAL_ON_UNDER25", outdir, live_only=False
        )
        btts_live_global, btts_live_by_league, _ = _overlap_tables(
            btts_no_live, under_live_keys, "btts_no_hit", "BTTS_NO_CONDITIONAL_ON_UNDER25", outdir, live_only=True
        )
        if not btts_global.empty:
            btts_global.to_csv(outdir / "BTTS_NO_CONDITIONAL_ON_UNDER25.csv", index=False)

        # Allowlist suggestions (global overlap by league)
        under_allow = _allowlist_from_league(
            under_by_league,
            args.min_overlap_rows,
            args.min_roi_lift,
            args.min_overlap_roi,
            outdir / "UNDER25_CONDITIONAL_ON_BTTS_NO__ALLOWLIST_THRESHOLDS.csv",
        )
        _allowlist_from_league(
            btts_by_league,
            args.min_overlap_rows,
            args.min_roi_lift,
            args.min_overlap_roi,
            outdir / "BTTS_NO_CONDITIONAL_ON_UNDER25__ALLOWLIST_THRESHOLDS.csv",
        )

        # Top-lift leagues for UNDER25 conditional on BTTS-NO
        if not under_by_league.empty:
            top_lift = under_by_league.sort_values(["roi_lift", "hit_rate_lift"], ascending=[False, False]).head(args.top_roi_n)
            top_lift.to_csv(outdir / "UNDER25_CONDITIONAL_ON_BTTS_NO__TOP_LIFT_LEAGUES.csv", index=False)

        # Support summary: baseline = all UNDER25 candidates, supported_only = overlap with BTTS-NO live
        allow_leagues = set(under_allow["league"].astype("string").fillna("").tolist()) if not under_allow.empty else set()
        under_keys = list(zip(
            under.get("league", pd.Series("", index=under.index)).astype("string").fillna(""),
            under.get("fixture_key", pd.Series("", index=under.index)).astype("string").fillna(""),
        ))
        under_overlap_live = pd.Series([k in btts_no_live_keys for k in under_keys], index=under.index)
        if allow_leagues:
            under_overlap_live &= under.get("league", pd.Series("", index=under.index)).astype("string").fillna("").isin(list(allow_leagues))
        supported = under.loc[under_overlap_live].copy()
        base_sum = _summarize(under, "under25_hit")
        supp_sum = _summarize(supported, "under25_hit")
        support_rows = [
            {"cohort": "baseline_all", **base_sum, "delta_hit_rate_vs_baseline": 0.0, "delta_roi_vs_baseline": 0.0},
            {
                "cohort": "supported_only",
                **supp_sum,
                "delta_hit_rate_vs_baseline": supp_sum["hit_rate"] - base_sum["hit_rate"],
                "delta_roi_vs_baseline": supp_sum["roi"] - base_sum["roi"],
            },
        ]
        support_df = pd.DataFrame(support_rows)
        support_df.to_csv(outdir / "UNDER25_BTTS_NO_SUPPORT_SUMMARY.csv", index=False)
        support_df.to_csv(outdir / "UNDER25_BTTS_NO_SUPPORT_AUDIT.csv", index=False)

        # Clean allowlist output
        if not under_allow.empty:
            under_allow.to_csv(outdir / "UNDER25_BTTS_NO_SUPPORT_ALLOWLIST.csv", index=False)
            try:
                (outdir / "UNDER25_BTTS_NO_SUPPORT_ALLOWLIST.json").write_text(
                    "\n".join(under_allow["league"].astype("string").fillna("").tolist()),
                    encoding="utf-8",
                )
            except Exception:
                pass

    # Sign-off one-pager
    sign_lines = []
    sign_lines.append("# UNDER25 + BTTS NO Baseline Sign‑Off")
    sign_lines.append("")
    sign_lines.append(f"Source: DEPLOY_CANDIDATES_RAW_SCORED (n={len(candidate_raw)})")
    sign_lines.append("")
    sign_lines.append("## UNDER25 Global")
    sign_lines.append(_markdown_table(pd.DataFrame([under_base]), ["rows", "graded", "wins", "losses", "hit_rate", "roi", "profit", "avg_bookie_od", "avg_model_p"]))
    sign_lines.append("")
    sign_lines.append("## UNDER25 Top ROI Leagues")
    sign_lines.append(_markdown_table(under_top, ["league", "graded", "hit_rate", "roi", "profit", "avg_bookie_od", "avg_model_p"]))
    sign_lines.append("")
    sign_lines.append("## BTTS NO Global")
    sign_lines.append(_markdown_table(pd.DataFrame([btts_base]), ["rows", "graded", "wins", "losses", "hit_rate", "roi", "profit", "avg_bookie_od", "avg_model_p"]))
    sign_lines.append("")
    sign_lines.append("## BTTS NO Top ROI Leagues")
    sign_lines.append(_markdown_table(btts_top, ["league", "graded", "hit_rate", "roi", "profit", "avg_bookie_od", "avg_model_p"]))
    (outdir / "UNDER25_BTTS_NO_SIGNOFF.md").write_text("\n".join(sign_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
