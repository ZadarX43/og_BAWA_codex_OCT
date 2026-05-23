#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _roi_from_level_stakes(df: pd.DataFrame) -> tuple[int, int, int, float, float]:
    if df.empty:
        return 0, 0, 0, np.nan, 0.0

    hit = _safe_num(df.get("btts_yes_hit", pd.Series(np.nan, index=df.index))).fillna(0).astype(int)
    odds = _safe_num(df.get("bookie_od", pd.Series(np.nan, index=df.index)))

    graded_mask = hit.isin([0, 1]) & odds.notna() & (odds > 1.0)
    g = df.loc[graded_mask].copy()
    if g.empty:
        return len(df), 0, 0, np.nan, 0.0

    ghit = _safe_num(g["btts_yes_hit"]).fillna(0).astype(int)
    godds = _safe_num(g["bookie_od"])

    wins = int((ghit == 1).sum())
    losses = int((ghit == 0).sum())
    graded = int(len(g))

    profit = np.where(ghit.eq(1), godds - 1.0, -1.0).sum()
    hit_rate = wins / graded if graded > 0 else np.nan
    roi = profit / graded if graded > 0 else np.nan

    return graded, wins, losses, hit_rate, float(profit)


def _summarise(df: pd.DataFrame, scenario_name: str) -> dict:
    rows = int(len(df))
    graded, wins, losses, hit_rate, profit = _roi_from_level_stakes(df)

    return {
        "scenario": scenario_name,
        "rows": rows,
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit_rate,
        "roi_level_stake": (profit / graded) if graded > 0 else np.nan,
        "level_stake_profit": profit,
        "avg_bookie_od": float(_safe_num(df.get("bookie_od", pd.Series(np.nan, index=df.index))).mean()) if rows > 0 else np.nan,
        "avg_model_p_for_bookie": float(_safe_num(df.get("model_p_for_bookie", pd.Series(np.nan, index=df.index))).mean()) if rows > 0 else np.nan,
    }


def _by_group(df: pd.DataFrame, scenario_name: str, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[
            "scenario", group_col, "rows", "graded", "wins", "losses",
            "hit_rate", "roi_level_stake", "level_stake_profit",
            "avg_bookie_od", "avg_model_p_for_bookie"
        ])

    out = []
    for key, g in df.groupby(group_col, dropna=False):
        row = _summarise(g, scenario_name)
        row[group_col] = key
        out.append(row)

    cols = [
        "scenario", group_col, "rows", "graded", "wins", "losses",
        "hit_rate", "roi_level_stake", "level_stake_profit",
        "avg_bookie_od", "avg_model_p_for_bookie"
    ]
    return pd.DataFrame(out)[cols].sort_values(["rows", "wins"], ascending=[False, False]).reset_index(drop=True)


def _norm_signal_bucket(df: pd.DataFrame) -> pd.Series:
    sig = _safe_str(df.get("signal_btts_runtime_eval", df.get("signal_btts_runtime", pd.Series("", index=df.index)))).str.upper()

    out = pd.Series("OTHER", index=df.index, dtype="string")
    out.loc[sig.eq("VERY_STRONG_YES")] = "VERY_STRONG_YES"
    out.loc[sig.eq("STRONG_YES")] = "STRONG_YES"
    out.loc[sig.eq("WEAK_YES")] = "WEAK_YES"
    out.loc[sig.eq("NEUTRAL")] = "NEUTRAL"
    return out


def _norm_first_exclusion(df: pd.DataFrame) -> pd.Series:
    return _safe_str(df.get("first_exclusion", pd.Series("", index=df.index))).str.upper()


def _league_family_map() -> Dict[str, str]:
    return {
        "USA MLS": "WEAK_RESCUE",
        "France Ligue 1": "WEAK_RESCUE",
        "Spain La Liga": "WEAK_RESCUE",
        "Germany Bundesliga 2": "WEAK_RESCUE",
        "England FA Cup": "WEAK_RESCUE",

        "Netherlands Eredivisie": "FTS_OVERRIDE",
        "Europa Conference": "FTS_OVERRIDE",
        "Japan J1": "FTS_OVERRIDE",
        "Belgium Pro": "FTS_OVERRIDE",
        "England FA Cup": "FTS_OVERRIDE",
        "Norway Eliteserien": "FTS_OVERRIDE",

        "Brazil Serie A": "BRAZIL_SPECIAL",

        "England Championship": "NEUTRAL_HEAVY",
        "England Premier League": "NEUTRAL_HEAVY",
        "Germany Bundesliga": "NEUTRAL_HEAVY",
        "Swiss Super League": "NEUTRAL_HEAVY",
        "Czech First League": "NEUTRAL_HEAVY",
    }


def _build_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["signal_bucket_explicit"] = _norm_signal_bucket(out)
    out["first_exclusion"] = _norm_first_exclusion(out)
    out["league"] = _safe_str(out.get("league", pd.Series("", index=out.index)))

    # Reconstruct "current live" from audit logic:
    # final_live_pass = 1 is source of truth if present.
    final_live = _safe_num(out.get("final_live_pass", pd.Series(0, index=out.index))).fillna(0).astype(int).eq(1)
    out["is_current_live"] = final_live.astype(int)

    # League families
    fam_map = _league_family_map()
    out["league_family"] = out["league"].map(fam_map).fillna("OTHER")

    # Rescue candidates
    out["is_weak"] = out["signal_bucket_explicit"].eq("WEAK_YES")
    out["is_neutral"] = out["signal_bucket_explicit"].eq("NEUTRAL")
    out["is_brazil"] = out["league"].eq("Brazil Serie A")

    # FTS override shape
    exp_goals_sum = _safe_num(out.get("exp_goals_sum", pd.Series(np.nan, index=out.index)))
    btts_top3_mass = _safe_num(out.get("btts_top3_mass", pd.Series(np.nan, index=out.index)))
    lam_fit = _safe_num(out.get("bookie_lambda_total_fit", pd.Series(np.nan, index=out.index)))
    p00_est = _safe_num(out.get("p00_est", pd.Series(np.nan, index=out.index)))

    # Conservative FTS override
    out["fts_override_ok"] = (
        exp_goals_sum.ge(3.20).fillna(False)
        & btts_top3_mass.ge(0.17).fillna(False)
        & lam_fit.ge(3.00).fillna(False)
        & p00_est.le(0.055).fillna(False)
    )

    # Selective neutral rescue
    out["neutral_rescue_ok"] = (
        out["is_neutral"]
        & out["league_family"].eq("NEUTRAL_HEAVY")
        & exp_goals_sum.ge(3.25).fillna(False)
        & btts_top3_mass.ge(0.18).fillna(False)
        & lam_fit.ge(3.05).fillna(False)
        & p00_est.le(0.050).fillna(False)
    )

    return out


def _scenario_current_live(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["is_current_live"].eq(1)].copy()


def _scenario_live_plus_weak_by_league(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["is_current_live"].eq(1)
        | (
            df["first_exclusion"].eq("LABEL_GATE_WEAK_YES")
            & df["league_family"].eq("WEAK_RESCUE")
        )
    )
    return df.loc[keep].copy()


def _scenario_live_plus_fts_override(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["is_current_live"].eq(1)
        | (
            df["first_exclusion"].eq("FTS_VETO")
            & df["league_family"].eq("FTS_OVERRIDE")
            & df["fts_override_ok"].eq(True)
        )
    )
    return df.loc[keep].copy()


def _scenario_live_plus_weak_plus_fts(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["is_current_live"].eq(1)
        | (
            df["first_exclusion"].eq("LABEL_GATE_WEAK_YES")
            & df["league_family"].eq("WEAK_RESCUE")
        )
        | (
            df["first_exclusion"].eq("FTS_VETO")
            & df["league_family"].eq("FTS_OVERRIDE")
            & df["fts_override_ok"].eq(True)
        )
    )
    return df.loc[keep].copy()


def _scenario_weak_by_league_family_only(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["is_weak"].eq(True)
        & df["league_family"].eq("WEAK_RESCUE")
    )
    return df.loc[keep].copy()


def _scenario_live_plus_selective_neutral(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        df["is_current_live"].eq(1)
        | df["neutral_rescue_ok"].eq(True)
    )
    return df.loc[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="Path to BTTS_YES_WINNERS__ALL_WITH_EXCLUSIONS.csv",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for scenario audit",
    )
    args = ap.parse_args()

    src = Path(args.src)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)
    df = _build_flags(df)

    scenarios = {
        "SCENARIO_CURRENT_LIVE_REPAIRED": _scenario_current_live(df),
        "SCENARIO_LIVE_PLUS_WEAK_BY_LEAGUE": _scenario_live_plus_weak_by_league(df),
        "SCENARIO_LIVE_PLUS_FTS_OVERRIDE": _scenario_live_plus_fts_override(df),
        "SCENARIO_LIVE_PLUS_WEAK_PLUS_FTS_OVERRIDE": _scenario_live_plus_weak_plus_fts(df),
        "SCENARIO_WEAK_BY_LEAGUE_FAMILY_ONLY": _scenario_weak_by_league_family_only(df),
        "SCENARIO_LIVE_PLUS_SELECTIVE_NEUTRAL": _scenario_live_plus_selective_neutral(df),
    }

    summary_rows: List[dict] = []
    by_league_parts: List[pd.DataFrame] = []
    by_signal_parts: List[pd.DataFrame] = []
    by_excl_parts: List[pd.DataFrame] = []
    compare_rows: List[dict] = []

    baseline_name = "SCENARIO_CURRENT_LIVE_REPAIRED"
    baseline = scenarios[baseline_name]
    baseline_summary = _summarise(baseline, baseline_name)
    baseline_rows = baseline_summary["rows"]
    baseline_hit = baseline_summary["hit_rate"]
    baseline_roi = baseline_summary["roi_level_stake"]
    baseline_profit = baseline_summary["level_stake_profit"]

    for name, sdf in scenarios.items():
        sdf = sdf.copy()

        # Save raw scenario rows
        sdf.to_csv(outdir / f"{name}__ROWS.csv", index=False)

        # Summary
        srow = _summarise(sdf, name)
        summary_rows.append(srow)

        # By league / signal / first exclusion
        by_league_parts.append(_by_group(sdf, name, "league"))
        by_signal_parts.append(_by_group(sdf, name, "signal_bucket_explicit"))
        by_excl_parts.append(_by_group(sdf, name, "first_exclusion"))

        compare_rows.append({
            "scenario": name,
            "rows": srow["rows"],
            "graded": srow["graded"],
            "wins": srow["wins"],
            "losses": srow["losses"],
            "hit_rate": srow["hit_rate"],
            "roi_level_stake": srow["roi_level_stake"],
            "level_stake_profit": srow["level_stake_profit"],
            "delta_rows_vs_baseline": srow["rows"] - baseline_rows if pd.notna(baseline_rows) else np.nan,
            "delta_hit_rate_vs_baseline": srow["hit_rate"] - baseline_hit if pd.notna(srow["hit_rate"]) and pd.notna(baseline_hit) else np.nan,
            "delta_roi_vs_baseline": srow["roi_level_stake"] - baseline_roi if pd.notna(srow["roi_level_stake"]) and pd.notna(baseline_roi) else np.nan,
            "delta_profit_vs_baseline": srow["level_stake_profit"] - baseline_profit if pd.notna(srow["level_stake_profit"]) and pd.notna(baseline_profit) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("roi_level_stake", ascending=False, na_position="last")
    compare_df = pd.DataFrame(compare_rows).sort_values("delta_profit_vs_baseline", ascending=False, na_position="last")
    by_league_df = pd.concat(by_league_parts, ignore_index=True) if by_league_parts else pd.DataFrame()
    by_signal_df = pd.concat(by_signal_parts, ignore_index=True) if by_signal_parts else pd.DataFrame()
    by_excl_df = pd.concat(by_excl_parts, ignore_index=True) if by_excl_parts else pd.DataFrame()

    summary_df.to_csv(outdir / "SCENARIO_SUMMARY.csv", index=False)
    compare_df.to_csv(outdir / "SCENARIO_COMPARE_VS_BASELINE.csv", index=False)
    by_league_df.to_csv(outdir / "SCENARIO_BY_LEAGUE.csv", index=False)
    by_signal_df.to_csv(outdir / "SCENARIO_BY_SIGNAL_BUCKET.csv", index=False)
    by_excl_df.to_csv(outdir / "SCENARIO_BY_FIRST_EXCLUSION.csv", index=False)

    print(f"OUTDIR: {outdir}")
    print("\nSCENARIO SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nCOMPARE VS BASELINE")
    print(compare_df.to_string(index=False))

    print("\nTOP LEAGUE VIEW")
    if not by_league_df.empty:
        top_league = by_league_df.sort_values(
            ["scenario", "roi_level_stake", "rows"],
            ascending=[True, False, False],
            na_position="last",
        ).groupby("scenario", as_index=False).head(12)
        print(top_league.to_string(index=False))

    print("\nTOP SIGNAL VIEW")
    if not by_signal_df.empty:
        print(by_signal_df.sort_values(["scenario", "rows"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()