#!/usr/bin/env python3
"""FTR combo rescue audit using combo product filters as rescue signals.

Inputs:
  predictions_output/walk_forward/_MASTER/FTR_COMBO_MASTER_AUDITS/FTR_COMBO__COMBINED.csv
  predictions_output/walk_forward/_MASTER/FTR_COMBO_MASTER_AUDITS/FTR_COMBO__SHORTLIST_FILTER_SWEEPS.csv

Outputs:
  - FTR_COMBO_RESCUE_AUDIT.csv
  - FTR_COMBO_RESCUE_AUDIT__BY_LEAGUE.csv
  - FTR_COMBO_RESCUE_AUDIT__BEST_RULES.csv
  - FTR_COMBO_RESCUE_SIGNOFF.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("predictions_output/walk_forward/_MASTER/FTR_COMBO_MASTER_AUDITS")
DEFAULT_OUT = DEFAULT_BASE


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _summarize(df: pd.DataFrame, hit_col: str) -> dict[str, object]:
    hit = _safe_num(df.get(hit_col, np.nan))
    graded = hit.notna().sum()
    wins = (hit == 1).sum()
    losses = (hit == 0).sum()
    profit = np.where(hit == 1, _safe_num(df.get("bookie_od", np.nan)) - 1.0, np.where(hit == 0, -1.0, np.nan))
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


def _is_vetoed(ftr: pd.DataFrame, mode: str) -> pd.Series:
    veto = pd.Series(False, index=ftr.index)
    for col in ["deterministic_veto_reason", "deploy_veto_reason"]:
        if col in ftr.columns:
            veto = veto | ftr[col].astype("string").fillna("").ne("")
    if mode == "loose":
        if "signal_mismatch_flag" in ftr.columns:
            veto = veto | (_safe_num(ftr["signal_mismatch_flag"]) == 1)
        return veto
    for col in [
        "hard_not_glue_flag",
        "not_glue_flag",
        "signal_mismatch_flag",
        "draw_risk_flag",
        "chaos_risk_flag",
        "ftr_drawtrap_flag",
        "cs_pathological_00_flag",
        "demote_to_observe",
    ]:
        if col in ftr.columns:
            veto = veto | (_safe_num(ftr[col]) == 1)
    return veto


def _build_filter_mask(df: pd.DataFrame, expr: str) -> pd.Series:
    """Parse simple filter expressions like 'lambda>=2.50 & combo_p>=0.65 & ge2_gap<0.25'."""
    if not expr:
        return pd.Series(True, index=df.index)
    expr = expr.replace(" ", "")
    parts = expr.split("&")
    mask = pd.Series(True, index=df.index)

    col_map = {
        "lambda": "combo_lambda",
        "combo_p": "combo_prob",
        "ge2_gap": "ge2_gap",
        "power_diff": "power_diff",
    }

    for part in parts:
        part = part.replace("-band", "")
        op = None
        for candidate in [">=", "<=", ">", "<"]:
            if candidate in part:
                op = candidate
                left, right = part.split(candidate, 1)
                break
        if op is None:
            continue
        col = col_map.get(left, left)
        if col not in df.columns:
            continue
        val = float(right)
        s = _safe_num(df[col])
        if op == ">=":
            m = s >= val
        elif op == "<=":
            m = s <= val
        elif op == ">":
            m = s > val
        else:
            m = s < val
        mask = mask & m.fillna(False)
    return mask


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_"
    headers = list(df.columns)
    rows = [headers]
    for _, r in df.iterrows():
        rows.append([("" if pd.isna(v) else str(v)) for v in r.tolist()])
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    out = [fmt(headers), "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"]
    out += [fmt(row) for row in rows[1:]]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FTR combo rescue audit")
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--veto-mode", type=str, default="strict", choices=["strict", "loose"])
    ap.add_argument("--allow-min-rows", type=int, default=50)
    ap.add_argument("--allow-min-roi-lift", type=float, default=0.01)
    ap.add_argument("--core-min-rows", type=int, default=150)
    ap.add_argument("--core-min-roi-lift", type=float, default=0.02)
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined_path = base / "FTR_COMBO__COMBINED.csv"
    shortlist_path = base / "FTR_COMBO__SHORTLIST_FILTER_SWEEPS.csv"
    if not combined_path.exists():
        raise SystemExit(f"Missing {combined_path}")
    if not shortlist_path.exists():
        raise SystemExit(f"Missing {shortlist_path}")

    df = pd.read_csv(combined_path)
    shortlist = pd.read_csv(shortlist_path)

    # Baseline live mask
    if "deploy_tier" in df.columns:
        live_mask = df.get("deploy_tier", "").astype(str).isin(["ELITE", "STANDARD"])
    else:
        live_mask = _safe_num(df.get("deploy_pass", pd.Series(0, index=df.index))) == 1

    veto = _is_vetoed(df, args.veto_mode)
    non_live = ~live_mask

    live_base = df.loc[live_mask].copy()
    live_sum = _summarize(live_base, "ftr_hit")

    rows = []
    rows_league = []

    def _rule_mask(_df: pd.DataFrame, _combo_market: str, _filter_family: str, _filter_value: str) -> pd.Series:
        if "combo_side" not in _df.columns:
            return pd.Series(False, index=_df.index)
        base_mask = _df["combo_side"].astype("string").fillna("").eq(_combo_market)

        if _filter_family in {"COMBO_P_FLOOR", "LAMBDA_FLOOR"}:
            expr = f"{'combo_p' if _filter_family=='COMBO_P_FLOOR' else 'lambda'}{_filter_value}"
        else:
            expr = _filter_value
        return base_mask & _build_filter_mask(_df, expr)

    for _, r in shortlist.iterrows():
        combo_market = str(r.get("combo_market", "")).strip()
        filter_family = str(r.get("filter_family", "")).strip()
        filter_value = str(r.get("filter_value", "")).strip()

        rescue_mask = non_live & (~veto) & _rule_mask(df, combo_market, filter_family, filter_value)

        rescue = df.loc[rescue_mask].copy()
        combined = pd.concat([live_base, rescue], axis=0, ignore_index=True) if not rescue.empty else live_base

        rescue_sum = _summarize(rescue, "ftr_hit")
        combined_sum = _summarize(combined, "ftr_hit")

        row = {
            "combo_market": combo_market,
            "filter_family": filter_family,
            "filter_value": filter_value,
            "rescue_rows": rescue_sum.get("rows", 0),
            "rescue_hit_rate": rescue_sum.get("hit_rate", np.nan),
            "rescue_roi": rescue_sum.get("roi", np.nan),
            "combined_rows": combined_sum.get("rows", 0),
            "combined_hit_rate": combined_sum.get("hit_rate", np.nan),
            "combined_roi": combined_sum.get("roi", np.nan),
            "delta_hit_rate_vs_existing_live": combined_sum.get("hit_rate", np.nan) - live_sum.get("hit_rate", np.nan),
            "delta_roi_vs_existing_live": combined_sum.get("roi", np.nan) - live_sum.get("roi", np.nan),
        }
        rows.append(row)

        # by league
        league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("")
        for lg, idx in league.groupby(league).groups.items():
            sub = df.loc[list(idx)].copy()
            sub_live = sub.loc[live_mask.loc[sub.index]].copy()
            sub_live_sum = _summarize(sub_live, "ftr_hit")
            sub_rescue = sub.loc[rescue_mask.loc[sub.index]].copy()
            sub_combined = pd.concat([sub_live, sub_rescue], axis=0, ignore_index=True) if not sub_rescue.empty else sub_live
            sub_combined_sum = _summarize(sub_combined, "ftr_hit")
            rows_league.append({
                "league": lg,
                "combo_market": combo_market,
                "filter_family": filter_family,
                "filter_value": filter_value,
                "rescue_rows": _summarize(sub_rescue, "ftr_hit").get("rows", 0),
                "rescue_hit_rate": _summarize(sub_rescue, "ftr_hit").get("hit_rate", np.nan),
                "rescue_roi": _summarize(sub_rescue, "ftr_hit").get("roi", np.nan),
                "combined_rows": sub_combined_sum.get("rows", 0),
                "combined_hit_rate": sub_combined_sum.get("hit_rate", np.nan),
                "combined_roi": sub_combined_sum.get("roi", np.nan),
                "delta_hit_rate_vs_existing_live": sub_combined_sum.get("hit_rate", np.nan) - sub_live_sum.get("hit_rate", np.nan),
                "delta_roi_vs_existing_live": sub_combined_sum.get("roi", np.nan) - sub_live_sum.get("roi", np.nan),
            })

    best = pd.DataFrame()
    best = pd.DataFrame()
    allow = pd.DataFrame()
    core = pd.DataFrame()
    watch = pd.DataFrame()
    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(outdir / "FTR_COMBO_RESCUE_AUDIT.csv", index=False)

        best = out_df.sort_values(["delta_roi_vs_existing_live", "rescue_rows"], ascending=[False, False]).head(50)
        best.to_csv(outdir / "FTR_COMBO_RESCUE_AUDIT__BEST_RULES.csv", index=False)

        allow = out_df.loc[
            (out_df["rescue_rows"] >= int(args.allow_min_rows))
            & (out_df["delta_roi_vs_existing_live"] >= float(args.allow_min_roi_lift))
        ].copy()
        allow.to_csv(outdir / "FTR_COMBO_RESCUE_ALLOWLIST.csv", index=False)

        core = allow.loc[
            (allow["rescue_rows"] >= int(args.core_min_rows))
            & (allow["delta_roi_vs_existing_live"] >= float(args.core_min_roi_lift))
        ].copy()
        core.to_csv(outdir / "FTR_COMBO_RESCUE_CORE_ALLOWLIST.csv", index=False)

        watch = allow.loc[~allow.index.isin(core.index)].copy()
        watch.to_csv(outdir / "FTR_COMBO_RESCUE_WATCHLIST.csv", index=False)

    if rows_league:
        pd.DataFrame(rows_league).to_csv(outdir / "FTR_COMBO_RESCUE_AUDIT__BY_LEAGUE.csv", index=False)

    # Combined live + rescued summary using allowlist union
    if rows and not allow.empty:
        allow_rules = allow[["combo_market", "filter_family", "filter_value"]].drop_duplicates()
        allow_mask = pd.Series(False, index=df.index)
        for _, rr in allow_rules.iterrows():
            allow_mask = allow_mask | _rule_mask(
                df,
                str(rr.get("combo_market", "")).strip(),
                str(rr.get("filter_family", "")).strip(),
                str(rr.get("filter_value", "")).strip(),
            )
        allow_rescue_mask = non_live & (~veto) & allow_mask
        rescue_union = df.loc[allow_rescue_mask].copy()
        combined_union = pd.concat([live_base, rescue_union], axis=0, ignore_index=True) if not rescue_union.empty else live_base

        live_sum = _summarize(live_base, "ftr_hit")
        rescue_sum = _summarize(rescue_union, "ftr_hit")
        combined_sum = _summarize(combined_union, "ftr_hit")

        summary = {
            "existing_live_rows": live_sum.get("rows", 0),
            "existing_live_graded": live_sum.get("graded", 0),
            "existing_live_hit_rate": live_sum.get("hit_rate", np.nan),
            "existing_live_roi": live_sum.get("roi", np.nan),
            "existing_live_profit": live_sum.get("profit", np.nan),
            "rescued_rows": rescue_sum.get("rows", 0),
            "rescued_graded": rescue_sum.get("graded", 0),
            "rescued_hit_rate": rescue_sum.get("hit_rate", np.nan),
            "rescued_roi": rescue_sum.get("roi", np.nan),
            "rescued_profit": rescue_sum.get("profit", np.nan),
            "combined_rows": combined_sum.get("rows", 0),
            "combined_graded": combined_sum.get("graded", 0),
            "combined_hit_rate": combined_sum.get("hit_rate", np.nan),
            "combined_roi": combined_sum.get("roi", np.nan),
            "combined_profit": combined_sum.get("profit", np.nan),
            "delta_hit_rate_vs_existing_live": combined_sum.get("hit_rate", np.nan) - live_sum.get("hit_rate", np.nan),
            "delta_roi_vs_existing_live": combined_sum.get("roi", np.nan) - live_sum.get("roi", np.nan),
        }
        pd.DataFrame([summary]).to_csv(outdir / "FTR_COMBO_RESCUE_PRODUCTION_SUMMARY.csv", index=False)

        # Per-league rescue counts for allowlist union
        league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("")
        rows_counts = []
        for lg, idx in league.groupby(league).groups.items():
            sub = df.loc[list(idx)].copy()
            sub_live = sub.loc[live_mask.loc[sub.index]].copy()
            sub_live_sum = _summarize(sub_live, "ftr_hit")
            sub_rescue = sub.loc[allow_rescue_mask.loc[sub.index]].copy()
            sub_rescue_sum = _summarize(sub_rescue, "ftr_hit")
            rows_counts.append({
                "league": lg,
                "baseline_live_rows": sub_live_sum.get("rows", 0),
                "rescued_rows": sub_rescue_sum.get("rows", 0),
                "rescued_hit_rate": sub_rescue_sum.get("hit_rate", np.nan),
                "rescued_roi": sub_rescue_sum.get("roi", np.nan),
                "baseline_live_hit_rate": sub_live_sum.get("hit_rate", np.nan),
                "baseline_live_roi": sub_live_sum.get("roi", np.nan),
                "delta_hit_rate_vs_live": sub_rescue_sum.get("hit_rate", np.nan) - sub_live_sum.get("hit_rate", np.nan),
                "delta_roi_vs_live": sub_rescue_sum.get("roi", np.nan) - sub_live_sum.get("roi", np.nan),
            })
        pd.DataFrame(rows_counts).to_csv(outdir / "FTR_COMBO_RESCUE_COUNTS__BY_LEAGUE.csv", index=False)

        # Allowlist JSON + patch fragment
        allow_rules_json = allow_rules.to_dict(orient="records")
        core_rules_json = core[["combo_market", "filter_family", "filter_value"]].drop_duplicates().to_dict(orient="records") if not core.empty else []
        watch_rules_json = watch[["combo_market", "filter_family", "filter_value"]].drop_duplicates().to_dict(orient="records") if not watch.empty else []
        (outdir / "FTR_COMBO_RESCUE_ALLOWLIST.json").write_text(
            json.dumps(allow_rules_json, indent=2),
            encoding="utf-8",
        )
        (outdir / "FTR_COMBO_RESCUE_CORE_ALLOWLIST.json").write_text(
            json.dumps(core_rules_json, indent=2),
            encoding="utf-8",
        )
        (outdir / "FTR_COMBO_RESCUE_WATCHLIST.json").write_text(
            json.dumps(watch_rules_json, indent=2),
            encoding="utf-8",
        )
        patch_lines = []
        patch_lines.append("# --- FTR combo rescue allowlists (auto-generated) ---")
        patch_lines.append("FTR_COMBO_RESCUE_CORE_RULES = [")
        for r in core_rules_json:
            patch_lines.append(f"    {r},")
        patch_lines.append("]")
        patch_lines.append("")
        patch_lines.append("FTR_COMBO_RESCUE_WATCHLIST_RULES = [")
        for r in watch_rules_json:
            patch_lines.append(f"    {r},")
        patch_lines.append("]")
        (outdir / "FTR_COMBO_RESCUE_ALLOWLIST_FRAGMENT.txt").write_text("\n".join(patch_lines), encoding="utf-8")

    # Signoff markdown
    sign = []
    sign.append("# FTR Combo Rescue — Audit Signoff")
    sign.append("")
    sign.append(f"Veto mode: {args.veto_mode}")
    sign.append("")
    if rows:
        sign.append("## Top Rules (Global Lift)")
        sign.append(_md_table(best))
    else:
        sign.append("## Top Rules (Global Lift)")
        sign.append("_(no data)_")
    (outdir / "FTR_COMBO_RESCUE_SIGNOFF.md").write_text("\n".join(sign), encoding="utf-8")


if __name__ == "__main__":
    main()
