from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = REPO_ROOT / "reports" / "player_events" / "quality_audits"


def build(input_csv: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        output_md.write_text("# Player Events Research Threshold Overrides\n\nNo tuning rows matched.\n")
        return df

    # First-pass research-only cycle:
    # 1. prefer APPLY_SOFT_PATCH / TRIAL_PATCH
    # 2. avoid contradictory duplicate cohorts
    # 3. prefer LOWER actions first for this cycle
    ranked = df[df["patch_confidence"].isin(["APPLY_SOFT_PATCH", "TRIAL_PATCH"])].copy()
    ranked = ranked.sort_values(
        ["recommended_trial_order", "patch_confidence", "tuning_signal"],
        ascending=[True, True, True],
    )

    selected_rows = []
    seen_keys: set[tuple[str, str, str]] = set()
    for _, row in ranked.iterrows():
        cohort_key = (
            str(row["market"]),
            str(row["review_family"]),
            str(row["prematch_risk_focus"]),
        )
        if cohort_key in seen_keys:
            continue
        if str(row["proposed_gate_action"]).upper() != "LOWER":
            continue
        seen_keys.add(cohort_key)
        selected_rows.append(row)

    out = pd.DataFrame(selected_rows).copy()
    if out.empty:
        out = pd.DataFrame(columns=list(df.columns) + ["applied_score_cut_shift", "override_reason"])
    else:
        out["applied_score_cut_shift"] = pd.to_numeric(out["proposed_score_cut_shift"], errors="coerce").fillna(0.0)
        out["override_reason"] = "FIRST_RESEARCH_ONLY_TUNING_CYCLE"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Research Threshold Overrides",
        "",
        "- First-pass research-only override set derived from the ranked tuning action board.",
        "- Conflict handling: if the same cohort has both a `LOWER` and `RAISE` idea, this first cycle keeps the stronger lower/soft-patch case and defers the contradictory raise for later comparison.",
        "",
        "## Applied Now",
    ]
    if out.empty:
        lines.append("- No overrides selected.")
    else:
        for _, row in out.iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | "
                f"patch={row['patch_confidence']} | action={row['proposed_gate_action']} {float(row['applied_score_cut_shift']):+.1f} | "
                f"rank={int(row['recommended_trial_order'])}"
            )
    lines.extend(
        [
            "",
            "## Deferred",
            "- `shots_on_target | 4231v433 | no core structural flag | RAISE +6.0` stays deferred for now because it conflicts with the stronger soft-lower case on the same cohort.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-pass research-only threshold override set from the player-events 3Y tuning action board.")
    parser.add_argument(
        "--input-csv",
        default=str(QUALITY_ROOT / "player_events_3y_tuning_action_board.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(QUALITY_ROOT / "player_events_research_threshold_overrides.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(QUALITY_ROOT / "player_events_research_threshold_overrides.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.input_csv), Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
