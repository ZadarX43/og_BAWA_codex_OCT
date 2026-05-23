
#!/usr/bin/env python3
"""run_frozen_specialist_walkforward.py

Run frozen specialist-overlay walk-forward month by month.

Purpose
-------
For each requested month this runner will:
  1) locate scored specialist backtest files (or other specialist CSV inputs)
  2) apply frozen specialist profiles via apply_frozen_specialist_profiles.py
  3) archive per-month outputs into a dedicated walk-forward tree
  4) build compact month-level summaries so you can inspect:
       - family hit by month
       - hard vs soft month summaries
       - total pass rows / families surviving each month

This is the specialist-overlay equivalent of the frozen priced-market runner.
It is intentionally hit-rate / profile oriented rather than ROI oriented.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SRC_GLOB = "predictions_output/specialists_backtests/specialists_test/*/*__BACKTEST_SCORED.csv"
DEFAULT_OUTROOT = "predictions_output/specialists_backtests/walkforward_frozen"
DEFAULT_APPLY_SCRIPT = "apply_frozen_specialist_profiles.py"


def _month_seq(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise SystemExit(f"end-month {end_month} is before start-month {start_month}")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def _safe_num(s: pd.Series | object) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_str(s: pd.Series | object) -> pd.Series:
    return pd.Series(s).astype("string").fillna("").str.strip()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = _safe_num(values)
    w = _safe_num(weights)
    m = v.notna() & w.notna() & (w > 0)
    if not bool(m.any()):
        return float("nan")
    return float((v[m] * w[m]).sum() / w[m].sum())


def _find_month_inputs(root: Path, month: str, src_glob: str) -> list[Path]:
    month_dir = root / month
    if month_dir.exists():
        direct = sorted(month_dir.glob("*__BACKTEST_SCORED.csv"))
        if direct:
            return direct

    candidates = sorted(Path(".").glob(src_glob))
    keep: list[Path] = []
    token = f"/{month}/"
    token2 = f"\\{month}\\"
    for p in candidates:
        ps = str(p)
        if token in ps or token2 in ps:
            keep.append(p)
    return keep


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        if str(src) == str(dst):
            return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_family_month_view(family_month_csv: Path, month: str) -> pd.DataFrame:
    if not family_month_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(family_month_csv, low_memory=False)
    if df.empty:
        return df

    for c in ["n", "hit", "avg_model_p_for_bookie", "selected_attack_score_mean", "opponent_attack_score_mean", "selected_win_prob_mean", "weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "month" not in df.columns:
        df["month"] = month
    else:
        df["month"] = _norm_str(df["month"])
        df = df[df["month"] == month].copy()

    if df.empty:
        return df

    if "overlay_family" not in df.columns:
        return pd.DataFrame()

    out = df.sort_values(["weight_class", "overlay_family", "league", "side"]).reset_index(drop=True)
    return out


def _build_weight_class_month_view(pass_rows_csv: Path, month: str) -> pd.DataFrame:
    if not pass_rows_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(pass_rows_csv, low_memory=False)
    if df.empty:
        return df

    df["month"] = _norm_str(df.get("month", pd.Series("", index=df.index)))
    if "month" in df.columns:
        df = df[df["month"] == month].copy()
    if df.empty:
        return df

    for c in ["correct_num", "model_p_for_bookie_num", "selected_win_prob", "weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows: list[dict[str, object]] = []
    for cls, g in df.groupby("weight_class", dropna=False):
        n = int(len(g))
        hit = float(g["correct_num"].mean()) if "correct_num" in g.columns and g["correct_num"].notna().any() else float("nan")
        avg_model_p = float(g["model_p_for_bookie_num"].mean()) if "model_p_for_bookie_num" in g.columns and g["model_p_for_bookie_num"].notna().any() else float("nan")
        avg_sel_win = float(g["selected_win_prob"].mean()) if "selected_win_prob" in g.columns and g["selected_win_prob"].notna().any() else float("nan")
        fam_n = int(_norm_str(g.get("overlay_family", pd.Series("", index=g.index))).replace("", pd.NA).dropna().nunique())
        lg_n = int(_norm_str(g.get("league", pd.Series("", index=g.index))).replace("", pd.NA).dropna().nunique())
        rows.append(
            {
                "month": month,
                "weight_class": str(cls),
                "rows": n,
                "families": fam_n,
                "leagues": lg_n,
                "hit": hit,
                "avg_model_p_for_bookie": avg_model_p,
                "avg_selected_win_prob": avg_sel_win,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["weight_class"]).reset_index(drop=True)


def _build_month_rollup(pass_rows_csv: Path, month: str) -> pd.DataFrame:
    if not pass_rows_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(pass_rows_csv, low_memory=False)
    if df.empty:
        return df

    df["month"] = _norm_str(df.get("month", pd.Series("", index=df.index)))
    if "month" in df.columns:
        df = df[df["month"] == month].copy()
    if df.empty:
        return df

    for c in ["correct_num", "model_p_for_bookie_num", "selected_win_prob"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = [
        {
            "month": month,
            "rows": int(len(df)),
            "families": int(_norm_str(df.get("overlay_family", pd.Series("", index=df.index))).replace("", pd.NA).dropna().nunique()),
            "leagues": int(_norm_str(df.get("league", pd.Series("", index=df.index))).replace("", pd.NA).dropna().nunique()),
            "hard_rows": int((_norm_str(df.get("weight_class", pd.Series("", index=df.index))).str.lower() == "hard").sum()),
            "soft_rows": int((_norm_str(df.get("weight_class", pd.Series("", index=df.index))).str.lower() == "soft").sum()),
            "hit": float(df["correct_num"].mean()) if "correct_num" in df.columns and df["correct_num"].notna().any() else float("nan"),
            "avg_model_p_for_bookie": float(df["model_p_for_bookie_num"].mean()) if "model_p_for_bookie_num" in df.columns and df["model_p_for_bookie_num"].notna().any() else float("nan"),
            "avg_selected_win_prob": float(df["selected_win_prob"].mean()) if "selected_win_prob" in df.columns and df["selected_win_prob"].notna().any() else float("nan"),
        }
    ]
    return pd.DataFrame(rows)


def _run_apply_script(
    python_bin: str,
    apply_script: Path,
    input_files: list[Path],
    outdir: Path,
    families: str | None,
    quiet: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [python_bin, str(apply_script)]
    for p in input_files:
        cmd.extend(["--src", str(p)])
    cmd.extend(["--outdir", str(outdir)])
    if families:
        cmd.extend(["--families", families])

    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=quiet,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run frozen specialist walk-forward month by month.")
    ap.add_argument("--start-month", required=True, help="YYYY-MM")
    ap.add_argument("--end-month", required=True, help="YYYY-MM")
    ap.add_argument("--src-root", default="predictions_output/specialists_backtests/specialists_test", help="Root folder containing month subfolders")
    ap.add_argument("--src-glob", default=DEFAULT_SRC_GLOB, help="Fallback glob for specialist scored inputs")
    ap.add_argument("--apply-script", default=DEFAULT_APPLY_SCRIPT, help="Path to apply_frozen_specialist_profiles.py")
    ap.add_argument("--outroot", default=DEFAULT_OUTROOT, help="Walk-forward output root")
    ap.add_argument("--run-tag", default="specialists_frozen", help="Subfolder name under outroot")
    ap.add_argument("--families", default=None, help="Optional comma-separated overlay_family whitelist")
    ap.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use for child runs")
    ap.add_argument("--overwrite", action="store_true", help="Re-run months even if outputs already exist")
    ap.add_argument("--quiet-child", action="store_true", help="Capture child script stdout/stderr instead of streaming")
    args = ap.parse_args()

    months = _month_seq(args.start_month, args.end_month)
    src_root = Path(args.src_root)
    apply_script = Path(args.apply_script)
    if not apply_script.exists():
        raise SystemExit(f"apply script not found: {apply_script}")

    outroot = Path(args.outroot) / args.run_tag
    outroot.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    month_rollups: list[pd.DataFrame] = []
    family_month_rollups: list[pd.DataFrame] = []
    class_month_rollups: list[pd.DataFrame] = []

    for month in months:
        month_inputs = _find_month_inputs(src_root, month, args.src_glob)
        month_dir = outroot / month
        month_dir.mkdir(parents=True, exist_ok=True)

        pass_rows_csv = month_dir / "frozen_specialist_pass_rows.csv"
        family_month_csv = month_dir / "frozen_specialist_family_month_summary.csv"
        family_league_csv = month_dir / "frozen_specialist_family_league_summary.csv"
        class_summary_csv = month_dir / "frozen_specialist_weight_class_summary.csv"
        child_log = month_dir / "apply_frozen_specialist_profiles.log"

        rerun_needed = args.overwrite or not pass_rows_csv.exists() or not family_month_csv.exists()

        print(f"\n=== {month} ===")
        print(f"INPUT_FILES_FOUND: {len(month_inputs)}")

        if not month_inputs:
            manifest_rows.append(
                {
                    "month": month,
                    "input_files": 0,
                    "pass_rows": 0,
                    "families": 0,
                    "hard_rows": 0,
                    "soft_rows": 0,
                    "hit": np.nan,
                    "avg_model_p_for_bookie": np.nan,
                    "status": "no_inputs",
                }
            )
            print("STATUS: no input files found for month")
            continue

        if rerun_needed:
            result = _run_apply_script(
                python_bin=args.python_bin,
                apply_script=apply_script,
                input_files=month_inputs,
                outdir=month_dir,
                families=args.families,
                quiet=args.quiet_child,
            )
            if args.quiet_child:
                child_log.write_text((result.stdout or "") + "\n" + (result.stderr or ""))
                print(f"CHILD_LOG: {child_log}")
        else:
            print("STATUS: skipped child run (existing outputs present)")

        fam_month_df = _build_family_month_view(family_month_csv, month)
        class_month_df = _build_weight_class_month_view(pass_rows_csv, month)
        month_rollup_df = _build_month_rollup(pass_rows_csv, month)

        if not fam_month_df.empty:
            fam_month_view_path = month_dir / "walkforward_family_hit_by_month.csv"
            fam_month_df.to_csv(fam_month_view_path, index=False)
            family_month_rollups.append(fam_month_df)
            print("FAMILY HIT BY MONTH:")
            print(fam_month_df[[c for c in ["overlay_family", "weight_class", "league", "market", "side", "n", "hit", "avg_model_p_for_bookie"] if c in fam_month_df.columns]].to_string(index=False))

        if not class_month_df.empty:
            class_month_view_path = month_dir / "walkforward_hard_soft_month_summary.csv"
            class_month_df.to_csv(class_month_view_path, index=False)
            class_month_rollups.append(class_month_df)
            print("HARD VS SOFT MONTH SUMMARY:")
            print(class_month_df.to_string(index=False))

        if not month_rollup_df.empty:
            month_rollup_path = month_dir / "walkforward_month_rollup.csv"
            month_rollup_df.to_csv(month_rollup_path, index=False)
            month_rollups.append(month_rollup_df)
            row = month_rollup_df.iloc[0].to_dict()
            row["input_files"] = len(month_inputs)
            row["status"] = "ok"
            manifest_rows.append(row)
        else:
            manifest_rows.append(
                {
                    "month": month,
                    "input_files": len(month_inputs),
                    "pass_rows": 0,
                    "families": 0,
                    "hard_rows": 0,
                    "soft_rows": 0,
                    "hit": np.nan,
                    "avg_model_p_for_bookie": np.nan,
                    "status": "no_pass_rows",
                }
            )
            print("STATUS: no pass rows for month")

        _copy_if_exists(pass_rows_csv, month_dir / pass_rows_csv.name)
        _copy_if_exists(family_month_csv, month_dir / family_month_csv.name)
        _copy_if_exists(family_league_csv, month_dir / family_league_csv.name)
        _copy_if_exists(class_summary_csv, month_dir / class_summary_csv.name)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = outroot / "walkforward_month_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    if family_month_rollups:
        all_family_month = pd.concat(family_month_rollups, ignore_index=True)
    else:
        all_family_month = pd.DataFrame()
    family_month_all_path = outroot / "walkforward_all_family_hit_by_month.csv"
    all_family_month.to_csv(family_month_all_path, index=False)

    if class_month_rollups:
        all_class_month = pd.concat(class_month_rollups, ignore_index=True)
    else:
        all_class_month = pd.DataFrame()
    class_month_all_path = outroot / "walkforward_all_hard_soft_month_summary.csv"
    all_class_month.to_csv(class_month_all_path, index=False)

    if month_rollups:
        all_month_rollups = pd.concat(month_rollups, ignore_index=True)
    else:
        all_month_rollups = pd.DataFrame()
    month_rollup_all_path = outroot / "walkforward_all_month_rollup.csv"
    all_month_rollups.to_csv(month_rollup_all_path, index=False)

    print("\n=== WALKFORWARD COMPLETE ===")
    print(f"MONTHS_REQUESTED: {len(months)}")
    print(f"WROTE: {manifest_path}")
    print(f"WROTE: {family_month_all_path}")
    print(f"WROTE: {class_month_all_path}")
    print(f"WROTE: {month_rollup_all_path}")

    if not manifest_df.empty:
        print("\n=== MONTH MANIFEST ===")
        print(manifest_df.to_string(index=False))

    if not all_family_month.empty:
        print("\n=== FAMILY HIT BY MONTH (ALL) ===")
        cols = [c for c in ["month", "overlay_family", "weight_class", "league", "market", "side", "n", "hit", "avg_model_p_for_bookie"] if c in all_family_month.columns]
        print(all_family_month[cols].sort_values(["month", "weight_class", "overlay_family"]).to_string(index=False))

    if not all_class_month.empty:
        print("\n=== HARD VS SOFT MONTH SUMMARIES (ALL) ===")
        print(all_class_month.sort_values(["month", "weight_class"]).to_string(index=False))


if __name__ == "__main__":
    main()