from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(runner_csv: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    df = pd.read_csv(runner_csv, low_memory=False)
    cohort = df[
        df["market"].astype(str).eq("shots_on_target")
        & df["review_family"].astype(str).eq("4231v433")
        & df["prematch_risk_focus"].astype(str).eq("no core structural flag")
    ].copy()

    if cohort.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(output_csv, index=False)
        output_md.write_text("# SOT 4231v433 Fallback Hit Threshold Audit\n\nNo cohort rows matched.\n")
        return cohort

    numeric_cols = [
        "expected_hit_rate_3y",
        "score_delta_vs_3y",
        "selection_gate_flag",
        "observed_success_flag",
        "lookback_rows",
    ]
    for col in numeric_cols:
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce")

    source_summary = (
        cohort.groupby("lookback_source", dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            avg_expected_hit=("expected_hit_rate_3y", "mean"),
            selected=("selection_gate_flag", "sum"),
            observed_hits=("observed_success_flag", "sum"),
            blocked_actual_hits=("observed_success_flag", lambda s: 0),
        )
        .reset_index()
    )

    blocked = cohort[cohort["selection_gate_flag"] == 0].copy()
    if not blocked.empty:
        blocked_hits = (
            blocked.groupby("lookback_source", dropna=False)["observed_success_flag"]
            .sum()
            .reset_index(name="blocked_actual_hits")
        )
        source_summary = source_summary.drop(columns=["blocked_actual_hits"]).merge(
            blocked_hits,
            on="lookback_source",
            how="left",
        )
        source_summary["blocked_actual_hits"] = pd.to_numeric(
            source_summary["blocked_actual_hits"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        source_summary["blocked_actual_hits"] = 0

    source_summary.to_csv(output_csv, index=False)

    lines = [
        "# SOT 4231v433 Fallback Hit Threshold Audit",
        "",
        "- Cohort: `shots_on_target | 4231v433 | no core structural flag`.",
        "- Purpose: compare fallback sources and judge whether `HIT_THRESHOLD = 0.55` is too strict by source.",
        "",
        "## Source Comparison",
    ]
    for _, row in source_summary.sort_values(["rows", "avg_expected_hit"], ascending=[False, False]).iterrows():
        lines.append(
            f"- {row['lookback_source']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | "
            f"avg_expected_hit={float(row['avg_expected_hit']):.3f} | selected={int(row['selected'])} | "
            f"observed_hits={int(row['observed_hits'])} | blocked_actual_hits={int(row['blocked_actual_hits'])}"
        )

    lines.extend(
        [
            "",
            "## Read",
            "- If blocked actual hits cluster in fallback sources with average expected hit just below `0.55`, that is evidence the global hit threshold may be too strict for those fallback buckets.",
            "- If blocked actual hits do not appear, lowering the threshold there is more likely to add noise than value.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return source_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether HIT_THRESHOLD=0.55 is too strict by fallback source for the SOT 4231v433 cohort.")
    parser.add_argument(
        "--runner-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-03__231829" / "player_events_3y_backtest_runner.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "sot_4231v433_fallback_hit_threshold_audit.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "sot_4231v433_fallback_hit_threshold_audit.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.runner_csv), Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
