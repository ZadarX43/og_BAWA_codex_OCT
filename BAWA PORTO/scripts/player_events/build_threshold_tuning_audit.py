from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_audit(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Threshold Tuning Audit\n\nNo rows matched.\n")
        return df

    summary = (
        df.groupby(["audit_label", "market", "review_family", "prematch_risk_focus"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            avg_expected_hit=("expected_hit_rate_3y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_score_delta=("score_delta_vs_3y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_runner_score=("score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values(["rows", "avg_score_delta"], ascending=[False, True])
    )
    summary["tuning_signal"] = summary.apply(
        lambda row: (
            "LOWER_SCORE_GATE"
            if row["audit_label"] == "MISSED_CORRECT_SELECTION" and row["avg_score_delta"] < -4.0
            else "RAISE_SCORE_GATE"
            if row["audit_label"] == "NEAR_MISS" and row["avg_score_delta"] > 2.0
            else "HOLD_AND_REVIEW"
        ),
        axis=1,
    )
    summary.to_csv(output_csv, index=False)

    lines = [
        "# Threshold Tuning Audit",
        "",
        "- `LOWER_SCORE_GATE`: rows are landing below the current 3Y score expectation, so the gate may be too strict.",
        "- `RAISE_SCORE_GATE`: rows are clearing the gate above expectation and still missing, so the gate may be too loose.",
        "- `HOLD_AND_REVIEW`: the miss pattern is not directional enough yet.",
        "",
    ]
    for signal, sub in summary.groupby("tuning_signal", sort=False):
        lines.append(f"## {signal}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['audit_label']} | {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_expected_hit={row['avg_expected_hit']:.3f} | avg_score_delta={row['avg_score_delta']:.2f}"
            )
        lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a threshold tuning audit from the player-market miss audit.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_audit(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
