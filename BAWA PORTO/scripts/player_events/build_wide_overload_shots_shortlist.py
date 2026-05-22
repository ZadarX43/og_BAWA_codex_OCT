from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_player_shots_on_target_report import _build_soti
from render_player_shots_report import _build_psi


WIDE_ROLES = {
    "Wide forward",
    "Wide midfielder / winger",
    "Wide defender / wing-back",
}


def build_shortlist(
    input_csv: str,
    output_csv: str,
    output_md: str,
    shot_min_score: float = 84.0,
    sot_min_score: float = 80.0,
    max_names_per_fixture: int = 3,
    min_quality_edge: float = 3.0,
    matchup_labels: list[str] | None = None,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv)
    shots = _build_psi(base.copy()).rename(columns={"player_shot_index": "shot_score", "confidence_label": "shot_confidence"})
    sot = _build_soti(base.copy()).rename(columns={"player_sot_index": "sot_score", "confidence_label": "sot_confidence"})
    merged = shots.merge(
        sot[
            [
                "fixture_key",
                "player_name",
                "sot_score",
                "sot_confidence",
            ]
        ],
        on=["fixture_key", "player_name"],
        how="left",
    )
    merged["tactical_role"] = merged["tactical_role"].astype("string")
    merged = merged[merged["tactical_role"].isin(WIDE_ROLES)].copy()
    merged = merged[merged["formation_wide_overload_flag"].eq(1)].copy()
    if matchup_labels:
        allowed = {x.strip() for x in matchup_labels if x.strip()}
        merged = merged[merged["formation_matchup_label"].astype("string").isin(allowed)].copy()
    merged = merged[merged["starting_xi_quality_edge"].ge(min_quality_edge)].copy()
    merged = merged[merged["shot_score"].ge(shot_min_score)].copy()
    merged = merged[
        merged["fixture_attacking_style_label"].astype("string").str.upper().isin(["ATTACK_WAVE", "CORNER_SIEGE", "BALANCED_ATTACK"])
    ].copy()
    merged["wide_concentration_combo_score"] = (
        0.55 * pd.to_numeric(merged["shot_score"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(merged["sot_score"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(merged["fixture_attack_pressure_score"], errors="coerce").fillna(0.0)
        + 8.0 * pd.to_numeric(merged["fixture_corner_pressure_score"], errors="coerce").fillna(0.0)
        + 6.0 * pd.to_numeric(merged["starting_xi_quality_edge"], errors="coerce").clip(lower=0.0, upper=20.0).fillna(0.0) / 20.0
    )
    merged["sot_support_flag"] = merged["sot_score"].ge(sot_min_score).astype(int)
    merged = (
        merged.sort_values(["fixture_key", "wide_concentration_combo_score"], ascending=[True, False])
        .groupby("fixture_key", as_index=False, group_keys=False)
        .head(max_names_per_fixture)
        .reset_index(drop=True)
    )
    merged["shortlist_reason_bucket"] = (
        merged["team_formation"].astype("string")
        + "__VS__"
        + merged["opponent_formation"].astype("string")
        + "__"
        + merged["tactical_role"].astype("string").str.replace(" ", "_", regex=False).str.upper()
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    lines = [
        "# Wide-Overload Shots Concentration Shortlist",
        "",
        "## Rules",
        f"- matchup labels: {', '.join(matchup_labels) if matchup_labels else 'all'}",
        f"- minimum shots score: {shot_min_score}",
        f"- minimum SOT support score: {sot_min_score}",
        f"- minimum quality edge: {min_quality_edge}",
        f"- max names per fixture: {max_names_per_fixture}",
        "",
        "## Board Summary",
        f"- rows: {len(merged)}",
        f"- fixtures covered: {merged['fixture_key'].nunique() if not merged.empty else 0}",
        f"- average combo score: {round(float(merged['wide_concentration_combo_score'].mean()), 2) if not merged.empty else 0.0}",
        "",
        "## Fixture Samples",
    ]
    for fixture_key, group in merged.groupby("fixture_key", sort=False):
        fx = group.iloc[0]
        lines.append(f"### {fixture_key}")
        lines.append(
            f"- attack: {fx['fixture_attacking_style_label']} | contact: {fx['fixture_style_label']} | formation: {fx['formation_matchup_label']} | edge={fx['starting_xi_quality_edge']:.1f}"
        )
        for row in group.itertuples(index=False):
            sot_tag = "yes" if int(getattr(row, "sot_support_flag", 0)) == 1 else "no"
            lines.append(
                f"- {row.player_name} ({row.team_name}) | {row.tactical_role} | shots={row.shot_score:.1f} | sot={row.sot_score:.1f} | sot_support={sot_tag} | combo={row.wide_concentration_combo_score:.1f}"
            )
        lines.append("")
        if len(lines) > 140:
            break
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a favorite wide-overload shots concentration shortlist.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--shot-min-score", type=float, default=84.0)
    parser.add_argument("--sot-min-score", type=float, default=80.0)
    parser.add_argument("--max-names-per-fixture", type=int, default=3)
    parser.add_argument("--min-quality-edge", type=float, default=3.0)
    parser.add_argument("--matchup-labels", default="4-3-3 vs 4-2-3-1,4-2-3-1 vs 4-3-3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_shortlist(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        shot_min_score=args.shot_min_score,
        sot_min_score=args.sot_min_score,
        max_names_per_fixture=args.max_names_per_fixture,
        min_quality_edge=args.min_quality_edge,
        matchup_labels=[x.strip() for x in args.matchup_labels.split(",") if x.strip()],
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")


if __name__ == "__main__":
    main()
