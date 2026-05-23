#!/usr/bin/env python3
"""FTR rescue audit using BTTS-YES / OU25 OVER25 as conditional support.

Goal:
  Test whether BTTS/OU25 signals improve FTR quality for borderline near-miss rows.
Outputs:
  - FTR_BTTS_OU25_LIFT_AUDIT.csv
  - FTR_BTTS_OU25_LIFT_AUDIT__BY_LEAGUE.csv
  - FTR_BTTS_OU25_LIFT_AUDIT__BY_SIDE.csv
  - FTR_BTTS_OU25_LIFT_AUDIT__BEST_RULES.csv
  - FTR_BTTS_OU25_LIFT_SIGNOFF.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("predictions_output/walk_forward")
DEFAULT_OUT = DEFAULT_BASE / "_MASTER" / "FTR_BTTS_OU25_RESCUE_AUDIT"


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_(no data)_"

    head = df.head(max_rows).copy()

    try:
        return head.to_markdown(index=False)
    except Exception:
        cols = [str(c) for c in head.columns]

        def _fmt(v: object) -> str:
            if pd.isna(v):
                return ""
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.6f}".rstrip("0").rstrip(".")
            return str(v)

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = [
            "| " + " | ".join(_fmt(v) for v in row) + " |"
            for row in head.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *body])


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


def _collect_scored_paths(base: Path) -> list[Path]:
    return sorted(base.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def _load_scored(base: Path) -> pd.DataFrame:
    paths = _collect_scored_paths(base)
    if not paths:
        raise SystemExit(f"No scored DEPLOY_COMBINED files found under {base}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = str(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("Failed to load any scored files.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _collect_deploy_paths(base: Path) -> list[Path]:
    return sorted(base.glob("w*/02_deploy/DEPLOY_COMBINED_*.csv"))


def _load_deploy(base: Path) -> pd.DataFrame:
    paths = _collect_deploy_paths(base)
    if not paths:
        raise SystemExit(f"No deploy DEPLOY_COMBINED files found under {base}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = str(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("Failed to load any deploy files.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _norm_str(x: pd.Series) -> pd.Series:
    return x.astype("string").fillna("").str.strip()


def _norm_market(x: pd.Series) -> pd.Series:
    return _norm_str(x).str.lower()


def _norm_pick(x: pd.Series) -> pd.Series:
    return _norm_str(x).str.upper()


def _norm_ou25_pick(df: pd.DataFrame) -> pd.Series:
    bp = _norm_pick(df.get("bookie_pick", pd.Series("", index=df.index)))
    sel = _norm_pick(df.get("selection", pd.Series("", index=df.index)))
    pick = bp.where(bp.ne(""), sel)
    pick = pick.replace({"OVER": "OVER25", "UNDER": "UNDER25"})
    return pick


def _live_mask(df_in: pd.DataFrame) -> pd.Series:
    if "deploy_pass" in df_in.columns:
        return _safe_num(df_in["deploy_pass"]) == 1
    return df_in.get("deploy_tier", "").astype(str).isin(["ELITE", "STANDARD"])


def _is_vetoed(ftr: pd.DataFrame) -> pd.Series:
    veto = pd.Series(False, index=ftr.index)
    for col in ["deterministic_veto_reason", "deploy_veto_reason"]:
        if col in ftr.columns:
            veto = veto | ftr[col].astype("string").fillna("").ne("")
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


def _is_vetoed_loose(ftr: pd.DataFrame) -> pd.Series:
    """Loose veto: only hard veto reasons + explicit mismatch."""
    veto = pd.Series(False, index=ftr.index)
    for col in ["deterministic_veto_reason", "deploy_veto_reason"]:
        if col in ftr.columns:
            veto = veto | ftr[col].astype("string").fillna("").ne("")
    for col in ["signal_mismatch_flag"]:
        if col in ftr.columns:
            veto = veto | (_safe_num(ftr[col]) == 1)
    return veto


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FTR rescue audit using BTTS/OU25")
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--bands", type=str, default="0.02,0.03,0.05", help="Comma-separated near-miss bands")
    ap.add_argument("--quantile", type=float, default=0.05, help="Quantile for live threshold estimation (default: 0.05)")
    ap.add_argument("--veto-mode", type=str, default="strict", choices=["strict", "loose"], help="Veto strictness (default: strict)")
    ap.add_argument("--live-source", type=str, default="auto", choices=["auto", "deploy_tier", "deploy_pass"], help="Live mask source (default: auto)")
    ap.add_argument("--allow-min-rows", type=int, default=50, help="Auto-allowlist min rows (default: 50)")
    ap.add_argument("--allow-min-roi-lift", type=float, default=0.05, help="Auto-allowlist min ROI lift vs live (default: 0.05)")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    deploy_all = _load_deploy(base)
    deploy_all["market_norm"] = _norm_market(deploy_all.get("market", pd.Series("", index=deploy_all.index)))
    deploy_all["league_norm"] = _norm_str(deploy_all.get("league", pd.Series("", index=deploy_all.index)))
    deploy_all["fixture_key_norm"] = _norm_str(deploy_all.get("fixture_key", pd.Series("", index=deploy_all.index)))
    deploy_all["pick_norm"] = _norm_pick(deploy_all.get("bookie_pick", pd.Series("", index=deploy_all.index)))
    deploy_all["pick_ou25_norm"] = _norm_ou25_pick(deploy_all)

    scored_all = _load_scored(base)
    scored_all["market_norm"] = _norm_market(scored_all.get("market", pd.Series("", index=scored_all.index)))
    scored_all["league_norm"] = _norm_str(scored_all.get("league", pd.Series("", index=scored_all.index)))
    scored_all["fixture_key_norm"] = _norm_str(scored_all.get("fixture_key", pd.Series("", index=scored_all.index)))
    scored_all["pick_norm"] = _norm_pick(scored_all.get("bookie_pick", pd.Series("", index=scored_all.index)))
    scored_all["pick_ou25_norm"] = _norm_ou25_pick(scored_all)

    ftr_deploy = deploy_all.loc[deploy_all["market_norm"].eq("ftr")].copy()
    if ftr_deploy.empty:
        raise SystemExit("No FTR rows found in deploy files.")

    scored_ftr = scored_all.loc[scored_all["market_norm"].eq("ftr")].copy()
    key_cols = ["league_norm", "fixture_key_norm", "market_norm", "pick_norm"]
    scored_ftr = scored_ftr.drop_duplicates(subset=key_cols)
    ftr = ftr_deploy.merge(
        scored_ftr[key_cols + ["ftr_hit"]],
        on=key_cols,
        how="left",
        suffixes=("", "_scored"),
    )

    if args.live_source == "deploy_tier":
        ftr_live = ftr.get("deploy_tier", "").astype(str).isin(["ELITE", "STANDARD"])
    elif args.live_source == "deploy_pass":
        ftr_live = _safe_num(ftr.get("deploy_pass", pd.Series(0, index=ftr.index))) == 1
    else:
        # auto: prefer deploy_tier if present, else deploy_pass
        if "deploy_tier" in ftr.columns:
            ftr_live = ftr.get("deploy_tier", "").astype(str).isin(["ELITE", "STANDARD"])
        else:
            ftr_live = _safe_num(ftr.get("deploy_pass", pd.Series(0, index=ftr.index))) == 1
    veto = _is_vetoed(ftr) if args.veto_mode == "strict" else _is_vetoed_loose(ftr)

    conf = pd.to_numeric(ftr.get("model_p_for_bookie", ftr.get("confidence", np.nan)), errors="coerce")
    margin = pd.to_numeric(ftr.get("ftr_margin", np.nan), errors="coerce")
    league = ftr.get("league", pd.Series("", index=ftr.index)).astype("string").fillna("").str.strip()
    pick = ftr.get("bookie_pick", ftr.get("selection", pd.Series("", index=ftr.index))).astype("string").fillna("").str.upper().str.strip()

    live_df = ftr.loc[ftr_live].copy()
    if not live_df.empty:
        live_conf = pd.to_numeric(live_df.get("model_p_for_bookie", live_df.get("confidence", np.nan)), errors="coerce")
        live_margin = pd.to_numeric(live_df.get("ftr_margin", np.nan), errors="coerce")
        live_league = live_df.get("league", pd.Series("", index=live_df.index)).astype("string").fillna("").str.strip()
        conf_min = live_df.assign(_conf=live_conf).groupby(live_league)["_conf"].quantile(float(args.quantile)).to_dict()
        marg_min = live_df.assign(_marg=live_margin).groupby(live_league)["_marg"].quantile(float(args.quantile)).to_dict()
    else:
        conf_min = {}
        marg_min = {}

    conf_floor = league.map(lambda lg: conf_min.get(str(lg), np.nan))
    marg_floor = league.map(lambda lg: marg_min.get(str(lg), np.nan))

    # BTTS/OU25 live keys
    btts_live = deploy_all.loc[deploy_all["market_norm"].eq("btts") & deploy_all["pick_norm"].eq("YES")].copy()
    btts_live = btts_live.loc[_live_mask(btts_live)].copy()
    ou25_live = deploy_all.loc[deploy_all["market_norm"].eq("ou25") & deploy_all["pick_ou25_norm"].eq("OVER25")].copy()
    ou25_live = ou25_live.loc[_live_mask(ou25_live)].copy()

    btts_keys = set(zip(btts_live.get("league_norm", pd.Series("", index=btts_live.index)),
                        btts_live.get("fixture_key_norm", pd.Series("", index=btts_live.index))))
    ou25_keys = set(zip(ou25_live.get("league_norm", pd.Series("", index=ou25_live.index)),
                        ou25_live.get("fixture_key_norm", pd.Series("", index=ou25_live.index))))

    ftr_keys = list(zip(ftr.get("league_norm", pd.Series("", index=ftr.index)),
                        ftr.get("fixture_key_norm", pd.Series("", index=ftr.index))))
    has_btts = pd.Series([k in btts_keys for k in ftr_keys], index=ftr.index)
    has_ou25 = pd.Series([k in ou25_keys for k in ftr_keys], index=ftr.index)

    bands = [float(x.strip()) for x in str(args.bands).split(",") if x.strip()]
    rows = []
    rows_league = []
    rows_side = []

    live_base = ftr.loc[ftr_live].copy()
    live_sum = _summarize(live_base, "ftr_hit")

    for band in bands:
        near_miss = (
            (~ftr_live)
            & (~veto)
            & conf.notna()
            & margin.notna()
            & conf_floor.notna()
            & marg_floor.notna()
            & (conf >= (conf_floor - band))
            & (conf < conf_floor)
            & (margin >= (marg_floor - band))
            & (margin < marg_floor)
        )

        cohorts = {
            "existing_live_only": ftr_live,
            "borderline_near_miss_only": near_miss,
            "borderline_plus_btts": near_miss & has_btts,
            "borderline_plus_ou25": near_miss & has_ou25,
            "borderline_plus_btts_ou25": near_miss & has_btts & has_ou25,
            "combined_live_plus_rescued": ftr_live | (near_miss & has_btts & has_ou25),
        }

        for name, mask in cohorts.items():
            sub = ftr.loc[mask].copy()
            perf = _summarize(sub, "ftr_hit")
            row = {"band": band, "cohort": name, **perf}
            row["delta_hit_rate_vs_existing_live"] = perf.get("hit_rate", np.nan) - live_sum.get("hit_rate", np.nan)
            row["delta_roi_vs_existing_live"] = perf.get("roi", np.nan) - live_sum.get("roi", np.nan)
            rows.append(row)

        # by league
        for lg, idx in league.groupby(league).groups.items():
            base_live_lg = ftr.loc[ftr_live & league.eq(lg)].copy()
            base_sum_lg = _summarize(base_live_lg, "ftr_hit")
            for name, mask in cohorts.items():
                sub = ftr.loc[mask & league.eq(lg)].copy()
                perf = _summarize(sub, "ftr_hit")
                row = {"band": band, "league": lg, "cohort": name, **perf}
                row["delta_hit_rate_vs_existing_live"] = perf.get("hit_rate", np.nan) - base_sum_lg.get("hit_rate", np.nan)
                row["delta_roi_vs_existing_live"] = perf.get("roi", np.nan) - base_sum_lg.get("roi", np.nan)
                rows_league.append(row)

        # by side
        for side, idx in pick.groupby(pick).groups.items():
            base_live_side = ftr.loc[ftr_live & pick.eq(side)].copy()
            base_sum_side = _summarize(base_live_side, "ftr_hit")
            for name, mask in cohorts.items():
                sub = ftr.loc[mask & pick.eq(side)].copy()
                perf = _summarize(sub, "ftr_hit")
                row = {"band": band, "side": side, "cohort": name, **perf}
                row["delta_hit_rate_vs_existing_live"] = perf.get("hit_rate", np.nan) - base_sum_side.get("hit_rate", np.nan)
                row["delta_roi_vs_existing_live"] = perf.get("roi", np.nan) - base_sum_side.get("roi", np.nan)
                rows_side.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(outdir / "FTR_BTTS_OU25_LIFT_AUDIT.csv", index=False)
    if rows_league:
        pd.DataFrame(rows_league).to_csv(outdir / "FTR_BTTS_OU25_LIFT_AUDIT__BY_LEAGUE.csv", index=False)
    if rows_side:
        pd.DataFrame(rows_side).to_csv(outdir / "FTR_BTTS_OU25_LIFT_AUDIT__BY_SIDE.csv", index=False)

    # Diagnostics
    try:
        diag = {
            "ftr_rows": int(len(ftr)),
            "ftr_live_rows": int(ftr_live.sum()),
            "ftr_non_live_rows": int((~ftr_live).sum()),
            "ftr_conf_nan": int(conf.isna().sum()),
            "ftr_margin_nan": int(margin.isna().sum()),
            "ftr_vetoed_rows": int(veto.sum()),
            "live_source": args.live_source,
            "quantile": float(args.quantile),
        }
        pd.DataFrame([diag]).to_csv(outdir / "FTR_BTTS_OU25_LIFT_DIAGNOSTICS.csv", index=False)
        if int((~ftr_live).sum()) == 0:
            print("[ftr_audit] WARNING: no non-live FTR rows; borderline cohorts will be empty.")
    except Exception:
        pass

    # Best rules (by league/band with highest ROI lift among borderline+signals)
    if rows_league:
        df_best = pd.DataFrame(rows_league)
        df_best = df_best.loc[df_best["cohort"].str.contains("borderline_plus", na=False)]
        if not df_best.empty:
            best_rows = []
            for (lg, band), g in df_best.groupby(["league", "band"], dropna=False):
                g = g.sort_values(["delta_roi_vs_existing_live", "rows"], ascending=[False, False])
                best_rows.append(g.iloc[0].to_dict())
            pd.DataFrame(best_rows).to_csv(outdir / "FTR_BTTS_OU25_LIFT_AUDIT__BEST_RULES.csv", index=False)
            best_df = pd.DataFrame(best_rows)
            allow_df = best_df.loc[
                (best_df["rows"] >= int(args.allow_min_rows))
                & (best_df["delta_roi_vs_existing_live"] >= float(args.allow_min_roi_lift))
            ].copy()
            allow_df.to_csv(outdir / "FTR_BTTS_OU25_RESCUE_ALLOWLIST.csv", index=False)

            # Core vs watchlist split
            core_df = allow_df.copy()
            watch_df = best_df.loc[~best_df["league"].isin(core_df["league"])].copy()
            core_df.to_csv(outdir / "FTR_BTTS_OU25_RESCUE_CORE_ALLOWLIST.csv", index=False)
            watch_df.to_csv(outdir / "FTR_BTTS_OU25_RESCUE_WATCHLIST.csv", index=False)

            # Patch fragment for per-league rules (best band + cohort)
            frag_lines = ["FTR_BTTS_OU25_RESCUE_ALLOWLIST = {"]
            for _, r in allow_df.iterrows():
                lg = str(r.get("league", ""))
                band = r.get("band", "")
                cohort = r.get("cohort", "")
                frag_lines.append(f'    "{lg}": {{"band": {band}, "cohort": "{cohort}"}},')
            frag_lines.append("}")
            (outdir / "FTR_BTTS_OU25_RESCUE_ALLOWLIST_FRAGMENT.txt").write_text(
                "\n".join(frag_lines),
                encoding="utf-8",
            )

    # Production summary (global)
    try:
        df_global = pd.DataFrame(rows)
        if not df_global.empty:
            base = df_global.loc[df_global["cohort"].eq("existing_live_only")].sort_values("band").head(1)
            comb = df_global.loc[df_global["cohort"].eq("combined_live_plus_rescued")].sort_values("band").head(1)
            resc = df_global.loc[df_global["cohort"].eq("borderline_plus_btts_ou25")].sort_values("band").head(1)
            summary = {}
            if not base.empty:
                b = base.iloc[0]
                summary.update({
                    "existing_live_rows": b.get("rows"),
                    "existing_live_graded": b.get("graded"),
                    "existing_live_hit_rate": b.get("hit_rate"),
                    "existing_live_roi": b.get("roi"),
                    "existing_live_profit": b.get("profit"),
                })
            if not resc.empty:
                r = resc.iloc[0]
                summary.update({
                    "rescued_rows": r.get("rows"),
                    "rescued_graded": r.get("graded"),
                    "rescued_hit_rate": r.get("hit_rate"),
                    "rescued_roi": r.get("roi"),
                    "rescued_profit": r.get("profit"),
                })
            if not comb.empty:
                c = comb.iloc[0]
                summary.update({
                    "combined_rows": c.get("rows"),
                    "combined_graded": c.get("graded"),
                    "combined_hit_rate": c.get("hit_rate"),
                    "combined_roi": c.get("roi"),
                    "combined_profit": c.get("profit"),
                    "delta_hit_rate_vs_existing_live": c.get("delta_hit_rate_vs_existing_live"),
                    "delta_roi_vs_existing_live": c.get("delta_roi_vs_existing_live"),
                })
            if summary:
                pd.DataFrame([summary]).to_csv(outdir / "FTR_BTTS_OU25_RESCUE_PRODUCTION_SUMMARY.csv", index=False)
    except Exception:
        pass

    # Signoff markdown
    sign_lines = []
    sign_lines.append("# FTR BTTS/OU25 Rescue — Lift Audit Signoff")
    sign_lines.append("")
    sign_lines.append(f"Live baseline compared against borderline near-miss cohorts with conditional BTTS/OU25 signals. Veto mode: {args.veto_mode}.")
    sign_lines.append("")
    if rows:
        df_global = pd.DataFrame(rows)
        best = df_global.loc[df_global["cohort"].eq("borderline_plus_btts_ou25")].sort_values(
            ["delta_roi_vs_existing_live", "rows"], ascending=[False, False]
        ).head(3)
        sign_lines.append("## Top Global Lift (BTTS+OU25)")
        sign_lines.append(_md_table(best, 10))
        sign_lines.append("")
    allow_path = outdir / "FTR_BTTS_OU25_RESCUE_ALLOWLIST.csv"
    if allow_path.exists():
        sign_lines.append("## Auto-Allowlist (Best Rules Filtered)")
        try:
            allow_df = pd.read_csv(allow_path)
            sign_lines.append(_md_table(allow_df, 20))
        except Exception:
            sign_lines.append("_(no data)_")
        sign_lines.append("")
    (outdir / "FTR_BTTS_OU25_LIFT_SIGNOFF.md").write_text("\n".join(sign_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
