from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_audit(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Player Market Near-Miss Audit\n\nNo rows matched.\n")
        return df

    miss_rows = df[(pd.to_numeric(df["near_miss_flag"], errors="coerce").fillna(0).ge(1)) | (pd.to_numeric(df["missed_correct_flag"], errors="coerce").fillna(0).ge(1))].copy()
    if miss_rows.empty:
        miss_rows.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Player Market Near-Miss Audit\n\nNo near misses or missed correct selections matched.\n")
        return miss_rows

    miss_rows["audit_label"] = miss_rows.apply(
        lambda row: "NEAR_MISS" if int(row.get("near_miss_flag", 0)) == 1 else "MISSED_CORRECT_SELECTION",
        axis=1,
    )
    miss_rows.to_csv(output_csv, index=False)

    summary = (
        miss_rows.groupby(["audit_label", "market", "review_family", "prematch_risk_focus"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            avg_expected_hit=("expected_hit_rate_3y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_score_delta=("score_delta_vs_3y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values(["rows", "avg_expected_hit"], ascending=[False, False])
    )

    lines = [
        "# Player Market Near-Miss Audit",
        "",
        "- `NEAR_MISS`: the walkforward gate would have supported the row, but the observed outcome proxy missed.",
        "- `MISSED_CORRECT_SELECTION`: the walkforward gate would not have supported the row, but the observed outcome proxy hit.",
        f"- rows={len(miss_rows)} | fixtures={miss_rows['fixture_key'].nunique()}",
        "",
    ]
    for label, sub in summary.groupby("audit_label", sort=False):
        lines.append(f"## {label}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_expected_hit={row['avg_expected_hit']:.3f} | avg_score_delta={row['avg_score_delta']:.2f}"
            )
        lines.append("")

    lines.append("## Example Rows")
    for _, row in miss_rows.head(12).iterrows():
        lines.append(
            f"- {row['audit_label']} | {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | {row['market']} | family={row['review_family']} | role={row['tactical_role']} | expected_hit={float(row['expected_hit_rate_3y']):.3f} | observed={int(row['observed_success_flag'])} | score_delta={float(row['score_delta_vs_3y']):.2f}"
        )
    lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return miss_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a near-miss / missed correct selections audit from the rolling player-market walkforward runner.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_audit(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
