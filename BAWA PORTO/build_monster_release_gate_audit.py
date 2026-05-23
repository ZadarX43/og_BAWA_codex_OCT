#!/usr/bin/env python3
"""Classify windows into monster release buckets from walk-forward audit outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_PAT = re.compile(r"w\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})")
TARGET_SIZES = [10, 12, 14]
RELEASE_ORDER = ["NO_MONSTER", "MONSTER_10_ONLY", "MONSTER_12_READY", "MONSTER_14_READY"]
RELEASE_DECISION_ORDER = ["NO_MONSTER", "RELEASE_10", "RELEASE_12", "RELEASE_14"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build monster release gate audit from walk-forward slip outputs")
    p.add_argument("--audit-dir", required=True, help="Slip audit dir containing SLIP_WALKFORWARD__WINDOW_SUMMARY.csv")
    p.add_argument("--outdir", required=True, help="Directory for release gate audit outputs")
    return p.parse_args()


def parse_window_date(window_id: str) -> pd.Timestamp:
    m = WINDOW_PAT.match(str(window_id))
    if not m:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))


def season_phase(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "UNKNOWN"
    month = int(ts.month)
    if month in {8, 9}:
        return "EARLY_SEASON"
    if month in {10, 11, 12}:
        return "AUTUMN_DENSE"
    if month in {1, 2}:
        return "WINTER_RESET"
    if month in {3, 4, 5}:
        return "RUN_IN"
    return "SUMMER_TRANSITION"


def select_release(row10: pd.Series, row12: pd.Series, row14: pd.Series) -> tuple[str, int, str]:
    if int(row14.get("buildable_flag", 0) or 0) == 1:
        return "MONSTER_14_READY", 14, "14_buildable"
    if int(row12.get("buildable_flag", 0) or 0) == 1:
        return "MONSTER_12_READY", 12, "12_buildable_14_not_ready"
    if int(row10.get("buildable_flag", 0) or 0) == 1:
        return "MONSTER_10_ONLY", 10, "10_buildable_12_not_ready"
    return "NO_MONSTER", 0, str(row10.get("failure_reason", "no_monster_candidate")).strip() or "no_monster_candidate"


def classify_confidence(release_class: str, available_rows: float, monster_candidate_rows: float, slip8_survived: float) -> tuple[str, str]:
    available_rows = float(available_rows or 0.0)
    monster_candidate_rows = float(monster_candidate_rows or 0.0)
    slip8_survived = float(slip8_survived or 0.0)
    if release_class == "MONSTER_14_READY":
        return "HIGH", "Purity gate cleared 14; schedule depth and tail feasibility support a full monster release."
    if release_class == "MONSTER_12_READY":
        if available_rows >= 45 and monster_candidate_rows >= 16 and slip8_survived >= 1:
            return "HIGH", "Purity gate cleared 12 with strong board depth and a clean core."
        return "MEDIUM", "Purity gate cleared 12; release 12 rather than forcing a 14."
    if release_class == "MONSTER_10_ONLY":
        if available_rows >= 40 and monster_candidate_rows >= 13:
            return "MEDIUM", "Board can support a 10, but tail depth is not strong enough for 12+."
        return "LOW", "Only a 10 is justified; keep the monster modest and lean on the core product."
    return "NO_RELEASE", "No monster release: stay with the reliable 6/7/8/9 core product this window."


def release_decision_from_class(release_class: str) -> tuple[str, int]:
    if release_class == "MONSTER_14_READY":
        return "RELEASE_14", 14
    if release_class == "MONSTER_12_READY":
        return "RELEASE_12", 12
    if release_class == "MONSTER_10_ONLY":
        return "RELEASE_10", 10
    return "NO_MONSTER", 0


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    window_summary_path = audit_dir / "SLIP_WALKFORWARD__WINDOW_SUMMARY.csv"
    if not window_summary_path.exists():
        raise SystemExit(f"Missing {window_summary_path}")

    df = pd.read_csv(window_summary_path, low_memory=False)
    if df.empty:
        raise SystemExit("Window summary is empty.")

    if "monster_mode" not in df.columns:
        df["monster_mode"] = "default"

    detail_rows: list[dict] = []
    for (monster_mode, window_id), grp in df.groupby(["monster_mode", "window_id"], dropna=False):
        grp = grp.copy()
        by_size = {int(size): sub.iloc[0] for size, sub in grp.groupby("slip_size", dropna=False)}
        missing = [size for size in [8, 10, 12, 14] if size not in by_size]
        if missing:
            continue

        row8 = by_size[8]
        row10 = by_size[10]
        row12 = by_size[12]
        row14 = by_size[14]

        release_class, release_slip_size, release_reason = select_release(row10, row12, row14)
        release_row = {10: row10, 12: row12, 14: row14}.get(release_slip_size)

        ts = parse_window_date(str(window_id))
        detail_rows.append({
            "monster_mode": monster_mode,
            "window_id": window_id,
            "window_date_from": ts.date().isoformat() if not pd.isna(ts) else "",
            "season_phase": season_phase(ts),
            "window_month": int(ts.month) if not pd.isna(ts) else pd.NA,
            "release_class": release_class,
            "release_slip_size": release_slip_size,
            "release_reason": release_reason,
            "available_rows": int(row10.get("available_rows", 0) or 0),
            "monster_candidate_rows": int(row10.get("monster_candidate_rows", 0) or 0),
            "monster_safe_rows": int(row10.get("monster_safe_rows", 0) or 0),
            "slip_8_survived_all": int(row8.get("survived_all", 0) or 0),
            "slip_8_legs_landed": int(row8.get("legs_landed", 0) or 0),
            "slip_10_buildable": int(row10.get("buildable_flag", 0) or 0),
            "slip_10_failure_reason": str(row10.get("failure_reason", "")),
            "slip_10_survived_all": int(row10.get("survived_all", 0) or 0),
            "slip_10_legs_landed": int(row10.get("legs_landed", 0) or 0),
            "slip_10_weakest_failed_safe_flag": pd.to_numeric(row10.get("weakest_failed_safe_flag"), errors="coerce"),
            "slip_12_buildable": int(row12.get("buildable_flag", 0) or 0),
            "slip_12_failure_reason": str(row12.get("failure_reason", "")),
            "slip_12_survived_all": int(row12.get("survived_all", 0) or 0),
            "slip_12_legs_landed": int(row12.get("legs_landed", 0) or 0),
            "slip_12_weakest_failed_safe_flag": pd.to_numeric(row12.get("weakest_failed_safe_flag"), errors="coerce"),
            "slip_12_deep_tail_strict_rows_used": pd.to_numeric(row12.get("deep_tail_strict_rows_used"), errors="coerce"),
            "slip_12_deep_tail_soft_rows_used": pd.to_numeric(row12.get("deep_tail_soft_rows_used"), errors="coerce"),
            "slip_12_deep_tail_fallback_rows_used": pd.to_numeric(row12.get("deep_tail_fallback_rows_used"), errors="coerce"),
            "slip_12_deep_tail_relaxed_cap_rows_used": pd.to_numeric(row12.get("deep_tail_relaxed_cap_rows_used"), errors="coerce"),
            "slip_14_buildable": int(row14.get("buildable_flag", 0) or 0),
            "slip_14_failure_reason": str(row14.get("failure_reason", "")),
            "slip_14_survived_all": int(row14.get("survived_all", 0) or 0),
            "slip_14_legs_landed": int(row14.get("legs_landed", 0) or 0),
            "slip_14_weakest_failed_safe_flag": pd.to_numeric(row14.get("weakest_failed_safe_flag"), errors="coerce"),
            "slip_14_deep_tail_strict_rows_used": pd.to_numeric(row14.get("deep_tail_strict_rows_used"), errors="coerce"),
            "slip_14_deep_tail_soft_rows_used": pd.to_numeric(row14.get("deep_tail_soft_rows_used"), errors="coerce"),
            "slip_14_deep_tail_fallback_rows_used": pd.to_numeric(row14.get("deep_tail_fallback_rows_used"), errors="coerce"),
            "slip_14_deep_tail_relaxed_cap_rows_used": pd.to_numeric(row14.get("deep_tail_relaxed_cap_rows_used"), errors="coerce"),
            "release_buildable_flag": int(release_row.get("buildable_flag", 0) or 0) if release_row is not None else 0,
            "release_survived_all": int(release_row.get("survived_all", 0) or 0) if release_row is not None else pd.NA,
            "release_legs_landed": int(release_row.get("legs_landed", 0) or 0) if release_row is not None else pd.NA,
            "release_legs_failed": int(release_row.get("legs_failed", 0) or 0) if release_row is not None else pd.NA,
            "release_complete_when_built": pd.to_numeric(release_row.get("survived_all"), errors="coerce") if release_row is not None else pd.NA,
            "release_weakest_failed_safe_flag": pd.to_numeric(release_row.get("weakest_failed_safe_flag"), errors="coerce") if release_row is not None else pd.NA,
        })

    if not detail_rows:
        raise SystemExit("No release gate rows were produced.")

    detail = pd.DataFrame(detail_rows)
    detail["release_class"] = pd.Categorical(detail["release_class"], RELEASE_ORDER, ordered=True)
    detail = detail.sort_values(["monster_mode", "release_class", "window_id"]).reset_index(drop=True)

    detail_path = outdir / "MONSTER_RELEASE_GATE__WINDOW_DETAIL.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["monster_mode", "release_class"], dropna=False)
        .agg(
            windows=("window_id", "size"),
            mean_available_rows=("available_rows", "mean"),
            mean_monster_candidate_rows=("monster_candidate_rows", "mean"),
            mean_8_complete_rate=("slip_8_survived_all", "mean"),
            mean_8_legs_landed=("slip_8_legs_landed", "mean"),
            mean_release_complete_rate=("release_survived_all", "mean"),
            mean_release_legs_landed=("release_legs_landed", "mean"),
            mean_release_legs_failed=("release_legs_failed", "mean"),
            mean_release_weakest_failed_safe_rate=("release_weakest_failed_safe_flag", "mean"),
            mean_slip_12_buildable=("slip_12_buildable", "mean"),
            mean_slip_14_buildable=("slip_14_buildable", "mean"),
        )
        .reset_index()
    )
    mode_totals = (
        detail.groupby("monster_mode", dropna=False)
        .agg(total_windows=("window_id", "size"))
        .reset_index()
    )
    summary = summary.merge(mode_totals, on="monster_mode", how="left")
    summary["release_rate"] = summary["windows"] / summary["total_windows"].replace({0: np.nan})
    summary = summary.sort_values(["monster_mode", "release_class"])
    summary_path = outdir / "MONSTER_RELEASE_GATE__SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    timing = (
        detail.groupby(["monster_mode", "release_class", "season_phase", "window_month"], dropna=False)
        .agg(
            windows=("window_id", "size"),
            mean_release_complete_rate=("release_survived_all", "mean"),
            mean_release_legs_landed=("release_legs_landed", "mean"),
            mean_8_complete_rate=("slip_8_survived_all", "mean"),
        )
        .reset_index()
        .sort_values(["monster_mode", "windows", "release_class"], ascending=[True, False, True])
    )
    timing_path = outdir / "MONSTER_RELEASE_GATE__TIMING.csv"
    timing.to_csv(timing_path, index=False)

    comparison = (
        detail.groupby(["monster_mode", "release_class", "release_slip_size"], dropna=False)
        .agg(
            windows=("window_id", "size"),
            release_complete_rate=("release_survived_all", "mean"),
            release_legs_landed=("release_legs_landed", "mean"),
            release_legs_failed=("release_legs_failed", "mean"),
            slip_8_complete_rate=("slip_8_survived_all", "mean"),
        )
        .reset_index()
        .sort_values(["monster_mode", "release_class", "release_slip_size"])
    )
    comparison_path = outdir / "MONSTER_RELEASE_GATE__COMPARISON.csv"
    comparison.to_csv(comparison_path, index=False)

    decision_rows: list[dict] = []
    purity_detail = detail[detail["monster_mode"].eq("purity")].copy()
    for _, row in purity_detail.iterrows():
        release_decision, release_size = release_decision_from_class(str(row["release_class"]))
        confidence, release_note = classify_confidence(
            str(row["release_class"]),
            pd.to_numeric(row.get("available_rows"), errors="coerce"),
            pd.to_numeric(row.get("monster_candidate_rows"), errors="coerce"),
            pd.to_numeric(row.get("slip_8_survived_all"), errors="coerce"),
        )
        decision_rows.append({
            "window_id": row["window_id"],
            "window_date_from": row["window_date_from"],
            "season_phase": row["season_phase"],
            "window_month": row["window_month"],
            "monster_mode": "purity",
            "release_class": row["release_class"],
            "release_decision": release_decision,
            "release_slip_size": release_size,
            "release_confidence": confidence,
            "release_note": release_note,
            "available_rows": row["available_rows"],
            "monster_candidate_rows": row["monster_candidate_rows"],
            "monster_safe_rows": row["monster_safe_rows"],
            "slip_8_survived_all": row["slip_8_survived_all"],
            "slip_8_legs_landed": row["slip_8_legs_landed"],
            "slip_10_buildable": row["slip_10_buildable"],
            "slip_12_buildable": row["slip_12_buildable"],
            "slip_14_buildable": row["slip_14_buildable"],
            "slip_10_failure_reason": row["slip_10_failure_reason"],
            "slip_12_failure_reason": row["slip_12_failure_reason"],
            "slip_14_failure_reason": row["slip_14_failure_reason"],
        })

    decisions = pd.DataFrame(decision_rows)
    decisions["release_decision"] = pd.Categorical(decisions["release_decision"], RELEASE_DECISION_ORDER, ordered=True)
    decisions = decisions.sort_values(["release_decision", "window_id"]).reset_index(drop=True)
    decisions_path = outdir / "MONSTER_RELEASE_GATE__DECISIONS.csv"
    decisions.to_csv(decisions_path, index=False)

    live_weekly_export = decisions[
        [
            "window_id",
            "window_date_from",
            "season_phase",
            "release_decision",
            "release_slip_size",
            "release_confidence",
            "release_note",
        ]
    ].copy()
    live_weekly_export = live_weekly_export.rename(
        columns={
            "release_slip_size": "release_size",
            "release_confidence": "confidence",
            "release_note": "note",
        }
    )
    live_weekly_export = live_weekly_export.sort_values(["window_date_from", "window_id"]).reset_index(drop=True)
    live_weekly_export_path = outdir / "MONSTER_RELEASE_GATE__LIVE_WEEKLY_EXPORT.csv"
    live_weekly_export.to_csv(live_weekly_export_path, index=False)

    print("WROTE:")
    print(detail_path)
    print(summary_path)
    print(timing_path)
    print(comparison_path)
    print(decisions_path)
    print(live_weekly_export_path)
    print("\nMONSTER RELEASE SUMMARY\n")
    print(summary.to_string(index=False))
    print("\nLIVE WEEKLY EXPORT SAMPLE\n")
    print(live_weekly_export.head(30).to_string(index=False))
    print("\nRELEASE DECISIONS\n")
    print(decisions.head(30).to_string(index=False))
    print("\nWINDOW DETAIL SAMPLE\n")
    print(detail.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
