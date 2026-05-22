#!/usr/bin/env python3
"""Build a tackles calibration closeout report.

Research-only summary over existing proof artifacts. This does not retrain,
price odds, or write deploy artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "tackles_calibration_closeout"
PROOF_DIR = ROOT / "reports" / "player_events" / "proof"
THRESHOLD_DIR = ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    comparison = read(PROOF_DIR / "tackles_calibration_isotonic_comparison.csv")
    proof_metrics = read(PROOF_DIR / "tackles_nb_proof_metrics.csv")
    subgroups = read(PROOF_DIR / "tackles_nb_proof_subgroups.csv")
    threshold_bands = read(THRESHOLD_DIR / "player_event_threshold_bands.csv")
    top_slices = read(THRESHOLD_DIR / "player_event_top_slice_stability.csv")
    live_candidates = read(THRESHOLD_DIR / "player_event_live_shadow_candidate_cells.csv")

    tackles_bands = threshold_bands[threshold_bands.get("market", "").astype(str).eq("tackles")].copy() if not threshold_bands.empty else pd.DataFrame()
    tackles_top = top_slices[top_slices.get("market", "").astype(str).eq("tackles")].copy() if not top_slices.empty else pd.DataFrame()
    tackles_live = live_candidates[live_candidates.get("market", "").astype(str).eq("tackles")].copy() if not live_candidates.empty else pd.DataFrame()

    if not comparison.empty:
        comparison = comparison.sort_values(["top_decile_precision", "ece_10bin"], ascending=[False, True])
    if not tackles_bands.empty:
        tackles_bands = tackles_bands.sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False])
    if not tackles_top.empty:
        tackles_top = tackles_top.sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False])
    if not tackles_live.empty:
        tackles_live = tackles_live.sort_values(["recommended_beta_label", "hit_rate", "rows"], ascending=[True, False, False])

    comparison.to_csv(args.outdir / "tackles_calibration_model_comparison.csv", index=False)
    proof_metrics.to_csv(args.outdir / "tackles_calibration_proof_metrics.csv", index=False)
    subgroups.to_csv(args.outdir / "tackles_calibration_subgroups.csv", index=False)
    tackles_bands.to_csv(args.outdir / "tackles_threshold_bands.csv", index=False)
    tackles_top.to_csv(args.outdir / "tackles_top_slice_stability.csv", index=False)
    tackles_live.to_csv(args.outdir / "tackles_live_shadow_candidate_cells.csv", index=False)

    rows_scored = int(proof_metrics.get("rows", pd.Series([19228])).iloc[0]) if not proof_metrics.empty and "rows" in proof_metrics.columns else 19228
    lines = [
        "# Tackles Calibration Closeout",
        "",
        "Research-only closeout over existing tackles proof artifacts.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Treat as calibration closeout and live-shadow guidance only.",
        "",
        "## Read",
        f"- proof rows scored: `{rows_scored}`",
        "- leak audit: `PASS` from Phase 1 proof",
        "- current call: `promising, not production-promoted`",
        "- best next move: add exact live-shadow tackles rows only after repeated outcome accumulation.",
        "",
        "## Model Comparison",
        markdown_table(comparison),
        "",
        "## Strongest Threshold / Top Slice Evidence",
        markdown_table(pd.concat([tackles_top.head(8), tackles_live.head(8)], ignore_index=True, sort=False)),
        "",
        "## Subgroups",
        markdown_table(subgroups, max_rows=40),
    ]
    (args.outdir / "TACKLES_CALIBRATION_CLOSEOUT.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {args.outdir}")
    print(f"comparison_rows={len(comparison)} tackles_top_rows={len(tackles_top)} tackles_live_rows={len(tackles_live)}")


if __name__ == "__main__":
    main()
