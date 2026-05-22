from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reason(row: pd.Series) -> str:
    expected_hit = float(pd.to_numeric(row.get("expected_hit_rate_3y"), errors="coerce") or 0.0)
    source = str(row.get("lookback_source", ""))
    pressure = float(pd.to_numeric(row.get("formation_pressure_score"), errors="coerce") or 0.0)
    player_q = float(pd.to_numeric(row.get("player_quality_score_l5"), errors="coerce") or 0.0)
    score_delta = float(pd.to_numeric(row.get("score_delta_vs_3y"), errors="coerce") or 0.0)
    observed_hit = int(pd.to_numeric(row.get("observed_success_flag"), errors="coerce") or 0)

    if observed_hit == 1 and expected_hit == 0.0:
        return "UNDERRATED_BROADER_POOL_HIT"
    if expected_hit <= 0.20 and source in {"ROLE_MARKET", "FAMILY_ROLE", "EXACT"}:
        return "LOW_EXPECTED_HIT_DESPITE_STRONG_SOURCE"
    if source == "GLOBAL":
        return "GLOBAL_POOL_TOO_BLUNT"
    if pressure < 0.20:
        return "LOW_ATTACK_PRESSURE"
    if player_q < 70:
        return "PLAYER_QUALITY_NOT_STRONG_ENOUGH"
    if score_delta > 8 and observed_hit == 0:
        return "SCORE_HIGH_BUT_NO_SHOT_HIT"
    return "POOL_NEEDS_SEPARATE_FILTER"


def build(runner_csv: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    df = pd.read_csv(runner_csv, low_memory=False)
    shots = df[df["market"].astype(str).eq("shots")].copy()
    if shots.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        shots.to_csv(output_csv, index=False)
        output_md.write_text("# Shots Broader Pool Audit\n\nNo shots rows matched.\n")
        return shots

    numeric_cols = [
        "expected_hit_rate_3y",
        "score_delta_vs_3y",
        "selection_gate_flag",
        "observed_success_flag",
        "formation_pressure_score",
        "player_quality_score_l5",
        "lookback_rows",
    ]
    for col in numeric_cols:
        shots[col] = pd.to_numeric(shots[col], errors="coerce")

    shots["broader_pool_reason"] = shots.apply(_reason, axis=1)

    summary = (
        shots.groupby("broader_pool_reason", dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            observed_hits=("observed_success_flag", "sum"),
            avg_expected_hit=("expected_hit_rate_3y", "mean"),
            avg_pressure=("formation_pressure_score", "mean"),
            avg_player_quality=("player_quality_score_l5", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "observed_hits"], ascending=[False, False])
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    lines = [
        "# Shots Broader Pool Audit",
        "",
        "- Treats `shots` as a broader attacking pool lane, separate from premium `shots_on_target`.",
        "- Focus: explain why the current runner is not selecting any `shots` rows and what kind of pool logic might still be useful.",
        "",
        "## Summary",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['broader_pool_reason']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | observed_hits={int(row['observed_hits'])} | "
            f"avg_expected_hit={float(row['avg_expected_hit']):.3f} | avg_pressure={float(row['avg_pressure']):.3f} | avg_player_quality={float(row['avg_player_quality']):.1f}"
        )

    hits = shots[pd.to_numeric(shots["observed_success_flag"], errors="coerce").fillna(0) == 1].copy()
    lines.extend(["", "## Actual Shot Hits In The Broader Pool"])
    if hits.empty:
        lines.append("- No actual shot hits found.")
    else:
        for _, row in hits.sort_values(["score_delta_vs_3y"], ascending=[False]).iterrows():
            lines.append(
                f"- {row['fixture_key']} | {row['player_name']} | role={row['tactical_role']} | source={row['lookback_source']} | "
                f"expected_hit={float(row['expected_hit_rate_3y']):.3f} | pressure={float(row['formation_pressure_score']):.3f} | "
                f"player_q={float(row['player_quality_score_l5']):.1f} | reason={row['broader_pool_reason']}"
            )

    lines.extend(
        [
            "",
            "## Read",
            "- `shots` should likely stay broader and more exploratory than `shots_on_target`.",
            "- The goal is not to force `shots` into the same premium precision band, but to identify a larger, sensible attacking pool that still behaves better than random.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broader-pool audit for the shots market.")
    parser.add_argument(
        "--runner-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-04__tuned_cycle_6_recent_relaxed_contact" / "player_events_3y_backtest_runner.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "shots_broader_pool_audit.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "shots_broader_pool_audit.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.runner_csv), Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
