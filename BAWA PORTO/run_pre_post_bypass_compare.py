#!/usr/bin/env python3
"""Run baseline vs bypass back-to-back and emit diff CSV for hit-rate/ROI.

Assumes bookie_allmarkets.py + deploy_rulebook.py outputs in predictions_output.
Baseline run uses current env; bypass run sets IGNORE_LOCKED_THRESHOLDS=1,
IGNORE_LOCKED_FTR_MIX=1, NO_DRAW_GATE=1 unless overridden.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List
import re

import pandas as pd
import shutil


BYPASS_ENVS = {
    "IGNORE_LOCKED_THRESHOLDS": "1",
    "IGNORE_LOCKED_FTR_MIX": "1",
    "NO_DRAW_GATE": "1",
}


def _run(cmd: List[str], env: dict, label: str) -> None:
    print(f"\n[run:{label}] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def _find_latest(path: Path, pattern: str) -> Path:
    hits = list(path.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files found for pattern: {pattern} under {path}")
    return max(hits, key=lambda p: p.stat().st_mtime)

def _tag_bookie(path: Path, label: str) -> Path:
    tagged = path.with_name(f"{path.stem}__{label.upper()}{path.suffix}")
    if tagged.exists():
        return tagged
    shutil.copy2(path, tagged)
    return tagged


def _league_tag(name: str) -> str:
    s = str(name).strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def _resolve_merged_path(merged_root: Path, league: str) -> Path | None:
    tag = _league_tag(league)
    direct = merged_root / f"{tag}__merged.csv"
    if direct.exists():
        return direct
    # Try a loose match on normalized stems
    norm_target = tag.lower()
    for p in merged_root.glob("*__merged.csv"):
        stem = p.name.replace("__merged.csv", "")
        stem_norm = _league_tag(stem).lower()
        if stem_norm == norm_target:
            return p
    return None


def _load_results_for_league(merged_root: Path, league: str) -> pd.DataFrame:
    path = _resolve_merged_path(merged_root, league)
    if path is None or not path.exists():
        return pd.DataFrame()
    usecols = [
        "fixture_key",
        "match_date",
        "date_GMT",
        "status",
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
        "total_goal_count",
    ]
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in usecols)
    return df


def _attach_results(df: pd.DataFrame, merged_root: Path, label: str = "run") -> pd.DataFrame:
    if df.empty or "league" not in df.columns:
        return df
    merged_root = Path(merged_root)
    if not merged_root.exists():
        print(f"⚠️  Results merged dir not found: {merged_root} (skipping results join)")
        return df

    out_frames = []
    join_rows = []
    for league, grp in df.groupby(df["league"].astype("string").str.strip(), dropna=False):
        g = grp.copy()
        res = _load_results_for_league(merged_root, league)
        if res.empty:
            join_rows.append({
                "league": str(league),
                "rows": int(len(g)),
                "joined": 0,
                "join_rate": 0.0 if len(g) else float("nan"),
                "method": "none",
            })
            out_frames.append(g)
            continue
        # Prefer fixture_key join
        if "fixture_key" in g.columns and "fixture_key" in res.columns:
            g = g.merge(
                res,
                on="fixture_key",
                how="left",
                suffixes=("", "__res"),
            )
            method = "fixture_key"
        else:
            # Fallback join on date + teams (best-effort)
            for col in ["match_date", "home_team_name", "away_team_name"]:
                if col not in g.columns and col in res.columns:
                    g[col] = pd.NA
            g = g.merge(
                res,
                on=["match_date", "home_team_name", "away_team_name"],
                how="left",
                suffixes=("", "__res"),
            )
            method = "match_date+teams"
        # If fixture_key join missed a lot, try a secondary fallback join
        if "status" in g.columns:
            miss_mask = g["status"].isna()
            if miss_mask.any():
                g_missing = g.loc[miss_mask, :].copy()
                for col in ["match_date", "home_team_name", "away_team_name"]:
                    if col not in g_missing.columns and col in res.columns:
                        g_missing[col] = pd.NA
                g_missing = g_missing.drop(columns=[c for c in res.columns if c in g_missing.columns and c not in ["match_date","home_team_name","away_team_name"]], errors="ignore")
                g_missing = g_missing.merge(
                    res,
                    on=["match_date", "home_team_name", "away_team_name"],
                    how="left",
                    suffixes=("", "__res2"),
                )
                # Fill missing result fields from fallback
                for col in ["status", "home_team_goal_count", "away_team_goal_count", "total_goal_count"]:
                    if col in g_missing.columns:
                        g.loc[miss_mask, col] = g_missing[col].values
                method = method + "+fallback"
        # Compute actuals (only for completed fixtures)
        status = g.get("status").astype("string").str.lower().str.strip() if "status" in g.columns else pd.Series(pd.NA, index=g.index, dtype="string")
        complete_mask = status.eq("complete")
        h = pd.to_numeric(g.get("home_team_goal_count"), errors="coerce").where(complete_mask)
        a = pd.to_numeric(g.get("away_team_goal_count"), errors="coerce").where(complete_mask)
        tg = pd.to_numeric(g.get("total_goal_count"), errors="coerce")
        tg = tg.where(complete_mask).fillna(h + a)

        actual_ftr = pd.Series(pd.NA, index=g.index, dtype="string")
        actual_ftr.loc[h > a] = "HOME"
        actual_ftr.loc[h < a] = "AWAY"
        actual_ftr.loc[h == a] = "DRAW"

        actual_btts = pd.Series(pd.NA, index=g.index, dtype="string")
        actual_btts.loc[(h > 0) & (a > 0)] = "YES"
        actual_btts.loc[(h == 0) | (a == 0)] = "NO"

        actual_ou25 = pd.Series(pd.NA, index=g.index, dtype="string")
        actual_ou25.loc[tg > 2.5] = "OVER25"
        actual_ou25.loc[tg <= 2.5] = "UNDER25"

        pick = g.get("selection", g.get("bookie_pick", pd.NA)).astype("string").str.upper().str.strip()

        g["ftr_hit"] = (pick.eq(actual_ftr)).astype("float").where(actual_ftr.notna())
        g["btts_yes_hit"] = (pick.eq(actual_btts)).astype("float").where(actual_btts.notna())
        g["ou25_hit"] = (pick.eq(actual_ou25)).astype("float").where(actual_ou25.notna())

        joined_any = int(status.notna().sum()) if len(g) else 0
        joined_complete = int(status.eq("complete").sum()) if len(g) else 0
        joined = int(g["ftr_hit"].notna().sum() if "ftr_hit" in g.columns else g["home_team_goal_count"].notna().sum())
        join_rows.append({
            "league": str(league),
            "rows": int(len(g)),
            "joined": joined,
            "join_rate": (joined / len(g)) if len(g) else float("nan"),
            "matched_any": joined_any,
            "matched_complete": joined_complete,
            "matched_any_rate": (joined_any / len(g)) if len(g) else float("nan"),
            "matched_complete_rate": (joined_complete / len(g)) if len(g) else float("nan"),
            "method": method,
        })
        out_frames.append(g)

    if join_rows:
        join_df = pd.DataFrame(join_rows).sort_values(["join_rate", "league"], ascending=[True, True])
        print(f"\nResult join coverage ({label}):")
        print(join_df.to_string(index=False))
    return pd.concat(out_frames, axis=0, ignore_index=True) if out_frames else df


def _load_deploy(path: Path, merged_root: Path | None = None, label: str = "run") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if merged_root is not None:
        df = _attach_results(df, Path(merged_root), label=label)
    return df


def _tier_series(df: pd.DataFrame) -> pd.Series:
    if "source_tier_file" in df.columns:
        return df["source_tier_file"].astype("string").str.upper().str.strip()
    if "pool_tier" in df.columns:
        return df["pool_tier"].astype("string").str.upper().str.strip()
    return pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Expect standard deploy columns
    for col in ["market"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"
    market = df["market"].astype("string").str.lower().str.strip()
    tier = _tier_series(df)

    # Map hit columns by market
    hit_map = {
        "ftr": "ftr_hit",
        "ou25": "ou25_hit",
        "btts": "btts_yes_hit",
        "tg15": "tg15_hit",
        "tg25": "tg25_hit",
    }

    rows = []
    for mk, hit_col in hit_map.items():
        sub = df.loc[market.eq(mk)].copy()
        if sub.empty:
            continue
        if hit_col in sub.columns:
            sub[hit_col] = pd.to_numeric(sub[hit_col], errors="coerce")
            graded = sub.loc[sub[hit_col].notna()].copy()
            wins = float(graded[hit_col].fillna(0).sum()) if not graded.empty else 0.0
            losses = int(len(graded) - wins) if not graded.empty else 0
            hit_rate = wins / len(graded) if len(graded) else float("nan")
            avg_od = float(pd.to_numeric(sub.get("bookie_od", pd.NA), errors="coerce").mean()) if len(sub) else float("nan")
            # Level-stake ROI if odds available
            if "bookie_od" in sub.columns:
                odds = pd.to_numeric(sub.get("bookie_od"), errors="coerce")
                profit = float((graded[hit_col] * (odds - 1.0)).fillna(0).sum() - losses)
                stake = float(len(graded))
                roi = profit / stake if stake else float("nan")
            else:
                roi = float("nan")
        else:
            graded = sub.iloc[0:0].copy()
            wins = 0.0
            losses = 0
            hit_rate = float("nan")
            avg_od = float(pd.to_numeric(sub.get("bookie_od", pd.NA), errors="coerce").mean()) if len(sub) else float("nan")
            roi = float("nan")
        rows.append({
            "market": mk,
            "tier": "ALL",
            "rows": int(len(sub)),
            "graded": int(len(graded)),
            "wins": wins,
            "losses": losses,
            "hit_rate": hit_rate,
            "avg_bookie_od": avg_od,
            "roi_level_stake": roi,
        })

        # Also by tier if present
        for tier_name, grp in sub.groupby(tier, dropna=False):
            g2 = grp.copy()
            if hit_col in g2.columns:
                g2[hit_col] = pd.to_numeric(g2[hit_col], errors="coerce")
                graded2 = g2.loc[g2[hit_col].notna()].copy()
                wins2 = float(graded2[hit_col].fillna(0).sum()) if not graded2.empty else 0.0
                losses2 = int(len(graded2) - wins2) if not graded2.empty else 0
                hit_rate2 = wins2 / len(graded2) if len(graded2) else float("nan")
                avg_od2 = float(pd.to_numeric(g2.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g2) else float("nan")
                if "bookie_od" in g2.columns:
                    odds2 = pd.to_numeric(g2.get("bookie_od"), errors="coerce")
                    profit2 = float((graded2[hit_col] * (odds2 - 1.0)).fillna(0).sum() - losses2)
                    stake2 = float(len(graded2))
                    roi2 = profit2 / stake2 if stake2 else float("nan")
                else:
                    roi2 = float("nan")
            else:
                graded2 = g2.iloc[0:0].copy()
                wins2 = 0.0
                losses2 = 0
                hit_rate2 = float("nan")
                avg_od2 = float(pd.to_numeric(g2.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g2) else float("nan")
                roi2 = float("nan")
            rows.append({
                "market": mk,
                "tier": str(tier_name),
                "rows": int(len(g2)),
                "graded": int(len(graded2)),
                "wins": wins2,
                "losses": losses2,
                "hit_rate": hit_rate2,
                "avg_bookie_od": avg_od2,
                "roi_level_stake": roi2,
            })

    return pd.DataFrame(rows)

def _summarize_by_league(df: pd.DataFrame) -> pd.DataFrame:
    if "league" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["league"] = df["league"].astype("string").fillna("").str.strip()
    market = df["market"].astype("string").str.lower().str.strip()
    tier = _tier_series(df)

    hit_map = {
        "ftr": "ftr_hit",
        "ou25": "ou25_hit",
        "btts": "btts_yes_hit",
        "tg15": "tg15_hit",
        "tg25": "tg25_hit",
    }
    rows = []
    for mk, hit_col in hit_map.items():
        sub = df.loc[market.eq(mk)].copy()
        if sub.empty:
            continue
        has_hit = hit_col in sub.columns
        if has_hit:
            sub[hit_col] = pd.to_numeric(sub[hit_col], errors="coerce")
        for league_name, grp in sub.groupby("league", dropna=False):
            g = grp.copy()
            if has_hit:
                graded = g.loc[g[hit_col].notna()].copy()
                wins = float(graded[hit_col].fillna(0).sum()) if not graded.empty else 0.0
                losses = int(len(graded) - wins) if not graded.empty else 0
                hit_rate = wins / len(graded) if len(graded) else float("nan")
                avg_od = float(pd.to_numeric(g.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g) else float("nan")
                if "bookie_od" in g.columns:
                    odds = pd.to_numeric(g.get("bookie_od"), errors="coerce")
                    profit = float((graded[hit_col] * (odds - 1.0)).fillna(0).sum() - losses)
                    stake = float(len(graded))
                    roi = profit / stake if stake else float("nan")
                else:
                    roi = float("nan")
            else:
                graded = g.iloc[0:0].copy()
                wins = 0.0
                losses = 0
                hit_rate = float("nan")
                avg_od = float(pd.to_numeric(g.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g) else float("nan")
                roi = float("nan")
            rows.append({
                "league": str(league_name),
                "market": mk,
                "tier": "ALL",
                "rows": int(len(g)),
                "graded": int(len(graded)),
                "wins": wins,
                "losses": losses,
                "hit_rate": hit_rate,
                "avg_bookie_od": avg_od,
                "roi_level_stake": roi,
            })

            for tier_name, grp_t in g.groupby(tier, dropna=False):
                gt = grp_t.copy()
                if has_hit:
                    graded_t = gt.loc[gt[hit_col].notna()].copy()
                    wins_t = float(graded_t[hit_col].fillna(0).sum()) if not graded_t.empty else 0.0
                    losses_t = int(len(graded_t) - wins_t) if not graded_t.empty else 0
                    hit_rate_t = wins_t / len(graded_t) if len(graded_t) else float("nan")
                    avg_od_t = float(pd.to_numeric(gt.get("bookie_od", pd.NA), errors="coerce").mean()) if len(gt) else float("nan")
                    if "bookie_od" in gt.columns:
                        odds_t = pd.to_numeric(gt.get("bookie_od"), errors="coerce")
                        profit_t = float((graded_t[hit_col] * (odds_t - 1.0)).fillna(0).sum() - losses_t)
                        stake_t = float(len(graded_t))
                        roi_t = profit_t / stake_t if stake_t else float("nan")
                    else:
                        roi_t = float("nan")
                else:
                    graded_t = gt.iloc[0:0].copy()
                    wins_t = 0.0
                    losses_t = 0
                    hit_rate_t = float("nan")
                    avg_od_t = float(pd.to_numeric(gt.get("bookie_od", pd.NA), errors="coerce").mean()) if len(gt) else float("nan")
                    roi_t = float("nan")
                rows.append({
                    "league": str(league_name),
                    "market": mk,
                    "tier": str(tier_name),
                    "rows": int(len(gt)),
                    "graded": int(len(graded_t)),
                    "wins": wins_t,
                    "losses": losses_t,
                    "hit_rate": hit_rate_t,
                    "avg_bookie_od": avg_od_t,
                    "roi_level_stake": roi_t,
                })
    return pd.DataFrame(rows)

def _filter_tiers(df: pd.DataFrame) -> pd.DataFrame:
    if "source_tier_file" not in df.columns and "pool_tier" not in df.columns:
        return df.iloc[0:0].copy()
    tiers = _tier_series(df)
    keep = tiers.isin(["ELITE", "STANDARD", "OBSERVE"])
    return df.loc[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline vs bypass and compare deploy performance")
    ap.add_argument("--date-from", required=True)
    ap.add_argument("--date-to", required=True)
    ap.add_argument("--leagues", required=True)
    ap.add_argument("--markets", default="ftr,btts,ou25")
    ap.add_argument("--implied-min", default="0.20")
    ap.add_argument("--ou25-implied-min", default="0.20")
    ap.add_argument("--btts-implied-min", default="0.20")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out-dir", default="predictions_output/pre_post_compare")
    ap.add_argument("--keep-draw-gate", action="store_true")
    ap.add_argument("--deploy-preset", default="", help="Optional deploy_rulebook preset (e.g. V1)")
    ap.add_argument("--deploy-deterministic", default="", help="Pass --deterministic to deploy_rulebook.py (e.g. off)")
    ap.add_argument("--extra-env", action="append", default=[], help="Extra env override KEY=VAL (repeatable)")
    ap.add_argument("--results-merged-dir", default="Matches/__merged__", help="Directory containing <LEAGUE>__merged.csv with results")
    ap.add_argument("--baseline-ftr-margin-scale", default="", help="Set FTR_MARGIN_SCALE for baseline (e.g. 1.5)")
    ap.add_argument("--bypass-ftr-margin-scale", default="", help="Set FTR_MARGIN_SCALE for bypass (e.g. 0.0)")
    ap.add_argument("--baseline-ftr-margin-cap", default="", help="Set FTR_MARGIN_MAX_CAP for baseline (e.g. 0.50)")
    ap.add_argument("--bypass-ftr-margin-cap", default="", help="Set FTR_MARGIN_MAX_CAP for bypass (e.g. 0.50)")
    ap.add_argument("--baseline-ftr-margin-override", default="", help="Set FTR_MARGIN_MIN_OVERRIDE for baseline (e.g. 0.35)")
    ap.add_argument("--bypass-ftr-margin-override", default="", help="Set FTR_MARGIN_MIN_OVERRIDE for bypass (e.g. 0.00)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_paths: list[Path] = []

    base_env = os.environ.copy()
    bypass_env = base_env.copy()
    bypass_env.update(BYPASS_ENVS)
    if args.keep_draw_gate:
        bypass_env["NO_DRAW_GATE"] = "0"
    if args.baseline_ftr_margin_scale:
        base_env["FTR_MARGIN_SCALE"] = str(args.baseline_ftr_margin_scale)
    if args.bypass_ftr_margin_scale:
        bypass_env["FTR_MARGIN_SCALE"] = str(args.bypass_ftr_margin_scale)
    if args.baseline_ftr_margin_cap:
        base_env["FTR_MARGIN_MAX_CAP"] = str(args.baseline_ftr_margin_cap)
    if args.bypass_ftr_margin_cap:
        bypass_env["FTR_MARGIN_MAX_CAP"] = str(args.bypass_ftr_margin_cap)
    if args.baseline_ftr_margin_override:
        base_env["FTR_MARGIN_MIN_OVERRIDE"] = str(args.baseline_ftr_margin_override)
    if args.bypass_ftr_margin_override:
        bypass_env["FTR_MARGIN_MIN_OVERRIDE"] = str(args.bypass_ftr_margin_override)
    for item in args.extra_env:
        if "=" in item:
            k, v = item.split("=", 1)
            bypass_env[k.strip()] = v

    cmd = [
        sys.executable,
        "bookie_allmarkets.py",
        "--date-from", args.date_from,
        "--date-to", args.date_to,
        "--leagues", args.leagues,
        "--markets", args.markets,
        "--implied-min", str(args.implied_min),
        "--ou25-implied-min", str(args.ou25_implied_min),
        "--btts-implied-min", str(args.btts_implied_min),
    ]
    if args.debug:
        cmd.append("--debug")

    # 1) Baseline
    _run(cmd, base_env, "baseline")
    base_bookie = _find_latest(Path("predictions_output"), f"BOOKIE_*_{args.date_from}_to_{args.date_to}*.csv")
    base_bookie_tagged = _tag_bookie(base_bookie, "baseline")
    summary_paths.append(base_bookie_tagged)

    # 2) Bypass
    _run(cmd, bypass_env, "bypass")
    bypass_bookie = _find_latest(Path("predictions_output"), f"BOOKIE_*_{args.date_from}_to_{args.date_to}*.csv")
    bypass_bookie_tagged = _tag_bookie(bypass_bookie, "bypass")
    summary_paths.append(bypass_bookie_tagged)

    # Deploy both
    def _deploy(src: Path, label: str) -> Path:
        deploy_dir = out_dir / label
        deploy_dir.mkdir(parents=True, exist_ok=True)
        dcmd = [sys.executable, "deploy_rulebook.py", "--src", str(src), "--outdir", str(deploy_dir)]
        if args.deploy_preset:
            dcmd.extend(["--preset", str(args.deploy_preset)])
        if args.deploy_deterministic:
            dcmd.extend(["--deterministic", str(args.deploy_deterministic)])
        _run(dcmd, base_env, f"deploy_{label}")
        # Prefer combined deploy output if present, else fall back to candidates
        for pattern in ("*DEPLOY_COMBINED*.csv", "DEPLOY_CANDIDATES_AFTER_GATES.csv", "DEPLOY_CANDIDATES_RAW.csv"):
            try:
                return _find_latest(deploy_dir, pattern)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"No deploy output found under {deploy_dir}")

    base_deploy = _deploy(base_bookie, "baseline")
    bypass_deploy = _deploy(bypass_bookie, "bypass")
    summary_paths.extend([base_deploy, bypass_deploy])

    # Summaries (overall market/tier)
    base_raw = _load_deploy(base_deploy, merged_root=args.results_merged_dir, label="baseline")
    bypass_raw = _load_deploy(bypass_deploy, merged_root=args.results_merged_dir, label="bypass")

    base_df = _summarize(base_raw)
    bypass_df = _summarize(bypass_raw)

    base_df["run"] = "baseline"
    bypass_df["run"] = "bypass"

    merged = base_df.merge(
        bypass_df,
        on=["market", "tier"],
        how="outer",
        suffixes=("_base", "_bypass"),
    )

    # Diff columns
    for col in ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "roi_level_stake"]:
        merged[f"{col}_diff"] = merged[f"{col}_bypass"] - merged[f"{col}_base"]

    out_csv = out_dir / f"PRE_POST_DIFF_{args.date_from}_to_{args.date_to}.csv"
    merged.to_csv(out_csv, index=False)
    summary_paths.append(out_csv)
    print(f"\nWrote diff CSV: {out_csv}")

    # Per-league diffs
    base_lg = _summarize_by_league(base_raw)
    bypass_lg = _summarize_by_league(bypass_raw)
    if not base_lg.empty or not bypass_lg.empty:
        merged_lg = base_lg.merge(
            bypass_lg,
            on=["league", "market", "tier"],
            how="outer",
            suffixes=("_base", "_bypass"),
        )
        for col in ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "roi_level_stake"]:
            merged_lg[f"{col}_diff"] = merged_lg[f"{col}_bypass"] - merged_lg[f"{col}_base"]
        out_lg = out_dir / f"PRE_POST_DIFF_BY_LEAGUE_{args.date_from}_to_{args.date_to}.csv"
        merged_lg.to_csv(out_lg, index=False)
        summary_paths.append(out_lg)
        print(f"Wrote per-league diff CSV: {out_lg}")

        # Per-market leaderboards (league-level, tier=ALL)
        try:
            lg_all = merged_lg.loc[merged_lg["tier"].astype("string").str.upper().eq("ALL")].copy()
            for mk in lg_all["market"].astype("string").str.lower().unique():
                sub = lg_all.loc[lg_all["market"].astype("string").str.lower().eq(mk)].copy()
                if sub.empty:
                    continue
                sub["roi_level_stake_diff"] = pd.to_numeric(sub.get("roi_level_stake_diff"), errors="coerce")
                sub = sub.sort_values(
                    ["roi_level_stake_diff", "hit_rate_diff", "league"],
                    ascending=[False, False, True],
                    na_position="last",
                ).reset_index(drop=True)
                out_mk = out_dir / f"PRE_POST_LEADERBOARD_{str(mk).upper()}_{args.date_from}_to_{args.date_to}.csv"
                sub.to_csv(out_mk, index=False)
                summary_paths.append(out_mk)
                print(f"Wrote per-market leaderboard CSV: {out_mk}")
        except Exception:
            pass

    # Tier-only summary (ELITE/STANDARD/OBSERVE)
    base_tier = _summarize(_filter_tiers(base_raw))
    bypass_tier = _summarize(_filter_tiers(bypass_raw))
    if not base_tier.empty or not bypass_tier.empty:
        merged_tier = base_tier.merge(
            bypass_tier,
            on=["market", "tier"],
            how="outer",
            suffixes=("_base", "_bypass"),
        )
        for col in ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "roi_level_stake"]:
            merged_tier[f"{col}_diff"] = merged_tier[f"{col}_bypass"] - merged_tier[f"{col}_base"]
        out_tier = out_dir / f"PRE_POST_DIFF_TIERS_{args.date_from}_to_{args.date_to}.csv"
        merged_tier.to_csv(out_tier, index=False)
        summary_paths.append(out_tier)
        print(f"Wrote tier-only diff CSV: {out_tier}")

    # Side-by-side per-league accuracy/ROI (ALL markets combined)
    def _agg_league_all(df: pd.DataFrame) -> pd.DataFrame:
        if "league" not in df.columns:
            return pd.DataFrame()
        out = []
        for lg, grp in df.groupby(df["league"].astype("string").str.strip(), dropna=False):
            hit_cols = [c for c in ["ftr_hit", "ou25_hit", "btts_yes_hit"] if c in grp.columns]
            if not hit_cols:
                continue
            hit_col = hit_cols[0]
            g = grp.copy()
            g[hit_col] = pd.to_numeric(g[hit_col], errors="coerce")
            graded = g.loc[g[hit_col].notna()].copy()
            wins = float(graded[hit_col].fillna(0).sum()) if not graded.empty else 0.0
            losses = int(len(graded) - wins) if not graded.empty else 0
            hit_rate = wins / len(graded) if len(graded) else float("nan")
            avg_od = float(pd.to_numeric(g.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g) else float("nan")
            if "bookie_od" in g.columns:
                odds = pd.to_numeric(g.get("bookie_od"), errors="coerce")
                profit = float((graded[hit_col] * (odds - 1.0)).fillna(0).sum() - losses)
                stake = float(len(graded))
                roi = profit / stake if stake else float("nan")
            else:
                roi = float("nan")
            out.append({
                "league": str(lg),
                "rows": int(len(g)),
                "graded": int(len(graded)),
                "wins": wins,
                "losses": losses,
                "hit_rate": hit_rate,
                "avg_bookie_od": avg_od,
                "roi_level_stake": roi,
            })
        return pd.DataFrame(out)

    base_all = _agg_league_all(base_raw)
    bypass_all = _agg_league_all(bypass_raw)
    if not base_all.empty or not bypass_all.empty:
        merged_all = base_all.merge(bypass_all, on=["league"], how="outer", suffixes=("_base", "_bypass"))
        for col in ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "roi_level_stake"]:
            merged_all[f"{col}_diff"] = merged_all[f"{col}_bypass"] - merged_all[f"{col}_base"]
        out_all = out_dir / f"PRE_POST_LEAGUE_SIDE_BY_SIDE_{args.date_from}_to_{args.date_to}.csv"
        merged_all.to_csv(out_all, index=False)
        summary_paths.append(out_all)
        print(f"Wrote per-league side-by-side CSV: {out_all}")

    # League-only summary across all markets (combined ROI/hit-rate)
    def _agg_league_all_markets(df: pd.DataFrame) -> pd.DataFrame:
        if "league" not in df.columns or "market" not in df.columns:
            return pd.DataFrame()
        hit_map = {
            "ftr": "ftr_hit",
            "ou25": "ou25_hit",
            "btts": "btts_yes_hit",
        }
        out = []
        df = df.copy()
        df["league"] = df["league"].astype("string").fillna("").str.strip()
        df["market"] = df["market"].astype("string").str.lower().str.strip()

        for lg, grp in df.groupby("league", dropna=False):
            g = grp.copy()
            # Build a unified hit column per row based on market
            hit = pd.Series(float("nan"), index=g.index, dtype="float64")
            for mk, col in hit_map.items():
                if col in g.columns:
                    mask = g["market"].eq(mk)
                    hit.loc[mask] = pd.to_numeric(g.loc[mask, col], errors="coerce")
            graded = g.loc[hit.notna()].copy()
            wins = float(hit.loc[hit.notna()].fillna(0).sum()) if not graded.empty else 0.0
            losses = int(len(graded) - wins) if not graded.empty else 0
            hit_rate = wins / len(graded) if len(graded) else float("nan")
            avg_od = float(pd.to_numeric(g.get("bookie_od", pd.NA), errors="coerce").mean()) if len(g) else float("nan")
            if "bookie_od" in g.columns:
                odds = pd.to_numeric(g.get("bookie_od"), errors="coerce")
                profit = float((hit.fillna(0) * (odds - 1.0)).fillna(0).sum() - losses)
                stake = float(len(graded))
                roi = profit / stake if stake else float("nan")
            else:
                roi = float("nan")
            out.append({
                "league": str(lg),
                "rows": int(len(g)),
                "graded": int(len(graded)),
                "wins": wins,
                "losses": losses,
                "hit_rate": hit_rate,
                "avg_bookie_od": avg_od,
                "roi_level_stake": roi,
            })
        return pd.DataFrame(out)

    base_combined = _agg_league_all_markets(base_raw)
    bypass_combined = _agg_league_all_markets(bypass_raw)
    if not base_combined.empty or not bypass_combined.empty:
        merged_combined = base_combined.merge(
            bypass_combined,
            on=["league"],
            how="outer",
            suffixes=("_base", "_bypass"),
        )
        for col in ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "roi_level_stake"]:
            merged_combined[f"{col}_diff"] = merged_combined[f"{col}_bypass"] - merged_combined[f"{col}_base"]
        out_combined = out_dir / f"PRE_POST_LEAGUE_COMBINED_ALL_MARKETS_{args.date_from}_to_{args.date_to}.csv"
        merged_combined.to_csv(out_combined, index=False)
        summary_paths.append(out_combined)
        print(f"Wrote league-only combined CSV: {out_combined}")

        # Ranked leaderboard (ROI diff desc)
        try:
            leaderboard = merged_combined.copy()
            leaderboard["roi_level_stake_diff"] = pd.to_numeric(
                leaderboard.get("roi_level_stake_diff"), errors="coerce"
            )
            leaderboard = leaderboard.sort_values(
                ["roi_level_stake_diff", "hit_rate_diff", "league"],
                ascending=[False, False, True],
                na_position="last",
            ).reset_index(drop=True)
            out_lb = out_dir / f"PRE_POST_LEAGUE_LEADERBOARD_{args.date_from}_to_{args.date_to}.csv"
            leaderboard.to_csv(out_lb, index=False)
            summary_paths.append(out_lb)
            print(f"Wrote ranked leaderboard CSV: {out_lb}")

            # Top + Bottom 10 snippet for quick sharing
            top_n = leaderboard.head(10)
            bottom_n = leaderboard.tail(10)
            snippet = pd.concat([top_n, bottom_n], ignore_index=True)
            out_snip = out_dir / f"PRE_POST_LEAGUE_LEADERBOARD_TOP_BOTTOM10_{args.date_from}_to_{args.date_to}.csv"
            snippet.to_csv(out_snip, index=False)
            summary_paths.append(out_snip)
            print(f"Wrote top/bottom 10 snippet CSV: {out_snip}")

            # Top takeaways summary (ROI diff if available, else hit_rate diff)
            print("\nTop takeaways:")
            metric = "roi_level_stake_diff"
            if leaderboard[metric].notna().any():
                top_row = leaderboard.loc[leaderboard[metric].notna()].head(1).iloc[0]
                bot_row = leaderboard.loc[leaderboard[metric].notna()].tail(1).iloc[0]
                print(f"  Biggest ROI lift: {top_row['league']} ({metric}={top_row[metric]:.4f})")
                print(f"  Biggest ROI drop: {bot_row['league']} ({metric}={bot_row[metric]:.4f})")
            elif "hit_rate_diff" in leaderboard.columns and leaderboard["hit_rate_diff"].notna().any():
                top_row = leaderboard.loc[leaderboard["hit_rate_diff"].notna()].head(1).iloc[0]
                bot_row = leaderboard.loc[leaderboard["hit_rate_diff"].notna()].tail(1).iloc[0]
                print(f"  Biggest hit-rate lift: {top_row['league']} (hit_rate_diff={top_row['hit_rate_diff']:.4f})")
                print(f"  Biggest hit-rate drop: {bot_row['league']} (hit_rate_diff={bot_row['hit_rate_diff']:.4f})")
            else:
                print("  ROI/hit-rate diffs unavailable (no hit columns in deploy outputs).")
        except Exception:
            pass

    if summary_paths:
        print("\nSummary outputs (copy/paste friendly):")
        print("```")
        for p in summary_paths:
            print(str(p))
        print("```")
    else:
        print("\nSummary outputs: none generated (no deploy CSVs found).")


if __name__ == "__main__":
    main()
