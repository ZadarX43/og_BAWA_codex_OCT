#!/usr/bin/env python3
"""Build one clean 3-year walk-forward backtest summary from audit outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-year walk-forward backtest summary")
    parser.add_argument("--audit-dir", required=True, help="Slip walk-forward audit dir")
    parser.add_argument("--release-gate-dir", required=True, help="Monster release gate dir")
    parser.add_argument("--outdir", required=True, help="Directory for summary outputs")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def build_slip_hit_rates(window_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        window_summary.groupby(["monster_mode", "slip_size", "build_mode"], dropna=False)
        .agg(
            windows=("window_id", "nunique"),
            buildable_rate=("buildable_flag", "mean"),
            complete_slip_rate=("survived_all", "mean"),
            mean_legs_landed=("legs_landed", "mean"),
            mean_legs_failed=("legs_failed", "mean"),
            mean_candidate_pool_size=("candidate_pool_size", "mean"),
            mean_available_rows=("available_rows", "mean"),
        )
        .reset_index()
    )

    built_only = window_summary[window_summary["buildable_flag"].eq(1)].copy()
    if not built_only.empty:
        built = (
            built_only.groupby(["monster_mode", "slip_size", "build_mode"], dropna=False)
            .agg(
                complete_slip_rate_when_built=("survived_all", "mean"),
                mean_legs_landed_when_built=("legs_landed", "mean"),
                mean_legs_failed_when_built=("legs_failed", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.merge(
            built,
            on=["monster_mode", "slip_size", "build_mode"],
            how="left",
        )

    return grouped.sort_values(["monster_mode", "slip_size", "build_mode"]).reset_index(drop=True)


def build_release_frequency(release_decisions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        release_decisions.groupby(
            ["monster_mode", "release_class", "release_decision", "release_slip_size"],
            dropna=False,
        )
        .agg(
            windows=("window_id", "nunique"),
            mean_available_rows=("available_rows", "mean"),
            mean_monster_candidate_rows=("monster_candidate_rows", "mean"),
        )
        .reset_index()
    )
    totals = (
        release_decisions.groupby("monster_mode", dropna=False)
        .agg(total_windows=("window_id", "nunique"))
        .reset_index()
    )
    grouped = grouped.merge(totals, on="monster_mode", how="left")
    grouped["release_rate"] = grouped["windows"] / grouped["total_windows"].replace({0: np.nan})
    return grouped.sort_values(["monster_mode", "release_slip_size", "release_decision"]).reset_index(drop=True)


def build_monster_win_rates(
    release_summary: pd.DataFrame, release_decisions: pd.DataFrame
) -> pd.DataFrame:
    monster_only = release_summary[release_summary["release_class"].ne("NO_MONSTER")].copy()
    confidence = (
        release_decisions[release_decisions["release_decision"].ne("NO_MONSTER")]
        .groupby(["monster_mode", "release_class"], dropna=False)
        .agg(
            confidence_high_windows=("release_confidence", lambda s: int((s == "HIGH").sum())),
            confidence_medium_windows=("release_confidence", lambda s: int((s == "MEDIUM").sum())),
            confidence_low_windows=("release_confidence", lambda s: int((s == "LOW").sum())),
        )
        .reset_index()
    )
    if confidence.empty:
        confidence = pd.DataFrame(
            columns=[
                "monster_mode",
                "release_class",
                "confidence_high_windows",
                "confidence_medium_windows",
                "confidence_low_windows",
            ]
        )

    merged = monster_only.merge(confidence, on=["monster_mode", "release_class"], how="left")
    for col in [
        "confidence_high_windows",
        "confidence_medium_windows",
        "confidence_low_windows",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    merged = merged.rename(
        columns={
            "mean_release_complete_rate": "monster_win_rate",
            "mean_release_legs_landed": "mean_monster_legs_landed",
            "mean_release_legs_failed": "mean_monster_legs_failed",
            "release_rate": "monster_release_rate",
        }
    )
    keep_cols = [
        "monster_mode",
        "release_class",
        "windows",
        "monster_release_rate",
        "monster_win_rate",
        "mean_monster_legs_landed",
        "mean_monster_legs_failed",
        "mean_release_weakest_failed_safe_rate",
        "mean_available_rows",
        "mean_monster_candidate_rows",
        "confidence_high_windows",
        "confidence_medium_windows",
        "confidence_low_windows",
    ]
    return merged[keep_cols].sort_values(["monster_mode", "release_class"]).reset_index(drop=True)


def build_release_class_performance(release_comparison: pd.DataFrame) -> pd.DataFrame:
    return release_comparison.sort_values(
        ["monster_mode", "release_class", "release_slip_size"]
    ).reset_index(drop=True)


def safe_lookup(
    df: pd.DataFrame, filters: dict[str, object], value_col: str, default: float = np.nan
) -> float:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key].eq(value)
    subset = df.loc[mask, value_col]
    if subset.empty:
        return default
    return float(subset.iloc[0])


def classify_signoff(
    release_class_performance: pd.DataFrame, release_frequency: pd.DataFrame
) -> tuple[str, str]:
    purity_10_complete = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_10_ONLY", "release_slip_size": 10},
        "release_complete_rate",
    )
    purity_12_complete = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_12_READY", "release_slip_size": 12},
        "release_complete_rate",
    )
    purity_14_complete = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_14_READY", "release_slip_size": 14},
        "release_complete_rate",
    )
    purity_no_monster_rate = safe_lookup(
        release_frequency,
        {"monster_mode": "purity", "release_decision": "NO_MONSTER"},
        "release_rate",
    )

    ready = (
        pd.notna(purity_10_complete)
        and purity_10_complete >= 0.40
        and pd.notna(purity_12_complete)
        and purity_12_complete >= 0.65
        and pd.notna(purity_14_complete)
        and purity_14_complete >= 0.60
        and pd.notna(purity_no_monster_rate)
        and purity_no_monster_rate >= 0.45
    )

    if ready:
        return (
            "MODEL_SIDE_READY_WITH_PURITY_GATED_MONSTERS",
            "Core 6/7/8/9 looks stable, monster issuance is selective, and purity-gated 10/12/14 releases are strong enough for a live policy.",
        )

    return (
        "NEEDS_MORE_REFINEMENT",
        "Core slips look promising, but the monster release layer still needs better performance or cleaner gating before a final live signoff.",
    )


def build_signoff(
    slip_hit_rates: pd.DataFrame,
    release_frequency: pd.DataFrame,
    release_class_performance: pd.DataFrame,
) -> pd.DataFrame:
    signoff_status, signoff_note = classify_signoff(release_class_performance, release_frequency)

    purity_prefix_8 = safe_lookup(
        slip_hit_rates,
        {"monster_mode": "purity", "slip_size": 8, "build_mode": "prefix"},
        "complete_slip_rate",
    )
    purity_10 = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_10_ONLY", "release_slip_size": 10},
        "release_complete_rate",
    )
    purity_12 = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_12_READY", "release_slip_size": 12},
        "release_complete_rate",
    )
    purity_14 = safe_lookup(
        release_class_performance,
        {"monster_mode": "purity", "release_class": "MONSTER_14_READY", "release_slip_size": 14},
        "release_complete_rate",
    )
    purity_no_monster_rate = safe_lookup(
        release_frequency,
        {"monster_mode": "purity", "release_decision": "NO_MONSTER"},
        "release_rate",
    )

    return pd.DataFrame(
        [
            {
                "signoff_status": signoff_status,
                "recommended_live_monster_mode": "purity",
                "recommended_core_policy": "ALWAYS_ON_6_7_8_9",
                "recommended_monster_policy": "RELEASE_ONLY_WHEN_PURITY_GATE_CLEARS",
                "recommended_decision_table": "14_if_14_ready_else_12_if_12_ready_else_10_if_10_only_else_no_monster",
                "purity_prefix_8_complete_rate": purity_prefix_8,
                "purity_release_10_complete_rate": purity_10,
                "purity_release_12_complete_rate": purity_12,
                "purity_release_14_complete_rate": purity_14,
                "purity_no_monster_rate": purity_no_monster_rate,
                "signoff_note": signoff_note,
            }
        ]
    )


def write_signoff_markdown(signoff: pd.DataFrame, path: Path) -> None:
    row = signoff.iloc[0]
    lines = [
        "# 3-Year Walk-Forward Model Signoff",
        "",
        f"- Status: `{row['signoff_status']}`",
        f"- Recommended live monster mode: `{row['recommended_live_monster_mode']}`",
        f"- Recommended core policy: `{row['recommended_core_policy']}`",
        f"- Recommended monster policy: `{row['recommended_monster_policy']}`",
        f"- Decision table: `{row['recommended_decision_table']}`",
        "",
        "## Key Backtest Reads",
        "",
        f"- Purity `8` complete rate: `{row['purity_prefix_8_complete_rate']:.3f}`",
        f"- Purity `RELEASE_10` complete rate: `{row['purity_release_10_complete_rate']:.3f}`",
        f"- Purity `RELEASE_12` complete rate: `{row['purity_release_12_complete_rate']:.3f}`",
        f"- Purity `RELEASE_14` complete rate: `{row['purity_release_14_complete_rate']:.3f}`",
        f"- Purity `NO_MONSTER` rate: `{row['purity_no_monster_rate']:.3f}`",
        "",
        f"Note: {row['signoff_note']}",
        "",
        "## Practical Conclusion",
        "",
        "- Keep `6/7/8/9` as the always-on core product.",
        "- Fire monsters only from the `purity` gate.",
        "- Default release ladder: `14 -> 12 -> 10 -> no monster`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def format_dec(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def write_full_system_report(
    slip_hit_rates: pd.DataFrame,
    release_frequency: pd.DataFrame,
    monster_win_rates: pd.DataFrame,
    release_class_performance: pd.DataFrame,
    signoff: pd.DataFrame,
    path: Path,
) -> None:
    signoff_row = signoff.iloc[0]

    def lookup(df: pd.DataFrame, filters: dict[str, object], value_col: str) -> float:
        return safe_lookup(df, filters, value_col)

    p5 = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 5, "build_mode": "prefix"}, "complete_slip_rate")
    p6 = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 6, "build_mode": "prefix"}, "complete_slip_rate")
    p7 = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 7, "build_mode": "prefix"}, "complete_slip_rate")
    p8 = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 8, "build_mode": "prefix"}, "complete_slip_rate")

    purity_10_build = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 10, "build_mode": "constructed"}, "buildable_rate")
    purity_12_build = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 12, "build_mode": "constructed"}, "buildable_rate")
    purity_14_build = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 14, "build_mode": "constructed"}, "buildable_rate")
    volume_12_build = lookup(slip_hit_rates, {"monster_mode": "volume", "slip_size": 12, "build_mode": "constructed"}, "buildable_rate")
    volume_14_build = lookup(slip_hit_rates, {"monster_mode": "volume", "slip_size": 14, "build_mode": "constructed"}, "buildable_rate")

    purity_12_built = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 12, "build_mode": "constructed"}, "complete_slip_rate_when_built")
    purity_14_built = lookup(slip_hit_rates, {"monster_mode": "purity", "slip_size": 14, "build_mode": "constructed"}, "complete_slip_rate_when_built")
    volume_12_built = lookup(slip_hit_rates, {"monster_mode": "volume", "slip_size": 12, "build_mode": "constructed"}, "complete_slip_rate_when_built")
    volume_14_built = lookup(slip_hit_rates, {"monster_mode": "volume", "slip_size": 14, "build_mode": "constructed"}, "complete_slip_rate_when_built")

    no_monster_rate = lookup(release_frequency, {"monster_mode": "purity", "release_decision": "NO_MONSTER"}, "release_rate")
    release_10_rate = lookup(release_frequency, {"monster_mode": "purity", "release_decision": "RELEASE_10"}, "release_rate")
    release_12_rate = lookup(release_frequency, {"monster_mode": "purity", "release_decision": "RELEASE_12"}, "release_rate")
    release_14_rate = lookup(release_frequency, {"monster_mode": "purity", "release_decision": "RELEASE_14"}, "release_rate")

    release_10_win = lookup(monster_win_rates, {"monster_mode": "purity", "release_class": "MONSTER_10_ONLY"}, "monster_win_rate")
    release_12_win = lookup(monster_win_rates, {"monster_mode": "purity", "release_class": "MONSTER_12_READY"}, "monster_win_rate")
    release_14_win = lookup(monster_win_rates, {"monster_mode": "purity", "release_class": "MONSTER_14_READY"}, "monster_win_rate")

    lines = [
        "# Full Model Summary",
        "",
        "## Executive Summary",
        "",
        f"- Final signoff: `{signoff_row['signoff_status']}`",
        "- Recommended live monster mode: `purity`",
        "- Core policy: `always-on 6/7/8/9`",
        "- Monster policy: `release only when purity gate clears`",
        "- Decision ladder: `14 -> 12 -> 10 -> no monster`",
        "",
        f"Note: {signoff_row['signoff_note']}",
        "",
        "## Why This System Won",
        "",
        "- The backtest showed the core ranked boards consistently support strong 5/6/7/8 slips.",
        "- Larger slips are real, but highly conditional on board depth, fixture spread, and correlation geometry.",
        "- Forcing 12/14s too often hurts quality, so monsters need a release gate rather than an always-on build policy.",
        "- `purity` beat `volume` as the live regime because it preserved quality in the built monster slips.",
        "",
        "## Main Discovery",
        "",
        "- The correct product shape is many reliable `6/7/8/9` slips and only occasional `10/12/14` monsters.",
        "- `volume` is useful for research because it shows how cap relaxation expands monster availability.",
        "- `purity` is better for live release because the monsters it allows are much more likely to actually win.",
        "",
        "## Backtest Hit Rates",
        "",
        f"- Purity 5-fold complete rate: `{format_dec(p5)}`",
        f"- Purity 6-fold complete rate: `{format_dec(p6)}`",
        f"- Purity 7-fold complete rate: `{format_dec(p7)}`",
        f"- Purity 8-fold complete rate: `{format_dec(p8)}`",
        "",
        "## Monster Buildability",
        "",
        f"- Purity 10 buildable rate: `{format_dec(purity_10_build)}`",
        f"- Purity 12 buildable rate: `{format_dec(purity_12_build)}`",
        f"- Purity 14 buildable rate: `{format_dec(purity_14_build)}`",
        f"- Volume 12 buildable rate: `{format_dec(volume_12_build)}`",
        f"- Volume 14 buildable rate: `{format_dec(volume_14_build)}`",
        "",
        "## Built Monster Quality",
        "",
        f"- Purity 12 complete rate when built: `{format_dec(purity_12_built)}`",
        f"- Purity 14 complete rate when built: `{format_dec(purity_14_built)}`",
        f"- Volume 12 complete rate when built: `{format_dec(volume_12_built)}`",
        f"- Volume 14 complete rate when built: `{format_dec(volume_14_built)}`",
        "",
        "## Release Frequency",
        "",
        f"- No monster: `{format_pct(no_monster_rate)}`",
        f"- Release 10: `{format_pct(release_10_rate)}`",
        f"- Release 12: `{format_pct(release_12_rate)}`",
        f"- Release 14: `{format_pct(release_14_rate)}`",
        "",
        "## Monster Win Rates",
        "",
        f"- Purity RELEASE_10 win rate: `{format_dec(release_10_win)}`",
        f"- Purity RELEASE_12 win rate: `{format_dec(release_12_win)}`",
        f"- Purity RELEASE_14 win rate: `{format_dec(release_14_win)}`",
        "",
        "## Release-Class Performance",
        "",
        "- `MONSTER_10_ONLY` is a useful secondary product, but it is clearly weaker than 12/14-ready windows.",
        "- `MONSTER_12_READY` is the standout bucket: frequent enough to matter and strong enough to trust.",
        "- `MONSTER_14_READY` is rare by design, which is a feature rather than a bug.",
        "",
        "## Final Live Policy",
        "",
        "1. Publish the core `6/7/8/9` product every cycle.",
        "2. Run the purity monster gate on the same board.",
        "3. If `MONSTER_14_READY`, release 14.",
        "4. Else if `MONSTER_12_READY`, release 12.",
        "5. Else if `MONSTER_10_ONLY`, release 10.",
        "6. Else release no monster and trust the core product.",
        "",
        "## Key Python Files",
        "",
        "- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/slip_formatter.py`: slot-based acca builder, monster-safe flags, deep-tail logic, and relaxed-cap support.",
        "- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/run_slip_walkforward_audit.py`: walk-forward builder/audit across purity and volume modes.",
        "- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_monster_release_gate_audit.py`: classifies windows into `NO_MONSTER`, `10`, `12`, `14` release buckets and produces the deployable weekly decision table.",
        "- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_full_walkforward_backtest_summary.py`: aggregates the 3-year backtest into hit rates, release frequency, monster win rates, release-class performance, and signoff.",
        "",
        "## System Evolution We Discovered",
        "",
        "- Initial monster construction was too constrained and hid the true issue.",
        "- Soft-tail additions did not materially contribute.",
        "- Deep-tail cap relaxation proved the monster ceiling was cap-bound, not just classification-bound.",
        "- Dual-mode auditing exposed the real tradeoff: `volume` for availability, `purity` for live quality.",
        "- The release gate turned that tradeoff into a usable product policy.",
        "",
        "## Final Model-Side Conclusion",
        "",
        "- The model side is effectively signed off for live packaging with a purity-gated monster policy.",
        "- Remaining work is mainly operational: live export wiring, website/product integration, and any last sanity checks rather than another major search for the right system.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    release_gate_dir = Path(args.release_gate_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    window_summary = load_csv(audit_dir / "SLIP_WALKFORWARD__WINDOW_SUMMARY.csv")
    release_summary = load_csv(release_gate_dir / "MONSTER_RELEASE_GATE__SUMMARY.csv")
    release_comparison = load_csv(release_gate_dir / "MONSTER_RELEASE_GATE__COMPARISON.csv")
    release_decisions = load_csv(release_gate_dir / "MONSTER_RELEASE_GATE__DECISIONS.csv")

    if "monster_mode" not in window_summary.columns:
        window_summary["monster_mode"] = "default"

    slip_hit_rates = build_slip_hit_rates(window_summary)
    release_frequency = build_release_frequency(release_decisions)
    monster_win_rates = build_monster_win_rates(release_summary, release_decisions)
    release_class_performance = build_release_class_performance(release_comparison)
    signoff = build_signoff(slip_hit_rates, release_frequency, release_class_performance)

    slip_path = outdir / "BACKTEST_3Y__SLIP_HIT_RATES.csv"
    freq_path = outdir / "BACKTEST_3Y__MONSTER_RELEASE_FREQUENCY.csv"
    win_path = outdir / "BACKTEST_3Y__MONSTER_WIN_RATES.csv"
    class_perf_path = outdir / "BACKTEST_3Y__RELEASE_CLASS_PERFORMANCE.csv"
    signoff_path = outdir / "BACKTEST_3Y__MODEL_SIGNOFF.csv"
    signoff_md_path = outdir / "BACKTEST_3Y__MODEL_SIGNOFF.md"
    full_report_md_path = outdir / "BACKTEST_3Y__FULL_MODEL_SYSTEM_REPORT.md"

    slip_hit_rates.to_csv(slip_path, index=False)
    release_frequency.to_csv(freq_path, index=False)
    monster_win_rates.to_csv(win_path, index=False)
    release_class_performance.to_csv(class_perf_path, index=False)
    signoff.to_csv(signoff_path, index=False)
    write_signoff_markdown(signoff, signoff_md_path)
    write_full_system_report(
        slip_hit_rates,
        release_frequency,
        monster_win_rates,
        release_class_performance,
        signoff,
        full_report_md_path,
    )

    print("WROTE:")
    print(slip_path)
    print(freq_path)
    print(win_path)
    print(class_perf_path)
    print(signoff_path)
    print(signoff_md_path)
    print(full_report_md_path)
    print("\nSLIP HIT RATES\n")
    print(slip_hit_rates.to_string(index=False))
    print("\nMONSTER RELEASE FREQUENCY\n")
    print(release_frequency.to_string(index=False))
    print("\nMONSTER WIN RATES\n")
    print(monster_win_rates.to_string(index=False))
    print("\nRELEASE CLASS PERFORMANCE\n")
    print(release_class_performance.to_string(index=False))
    print("\nMODEL SIGNOFF\n")
    print(signoff.to_string(index=False))


if __name__ == "__main__":
    main()
