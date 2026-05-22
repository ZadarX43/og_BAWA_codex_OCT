from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_combined_attacking_board import _build_fixture_gate
from render_player_shots_report import _build_psi


def export_shots_4plus_candidates(
    input_csv: str,
    output_csv: str,
    output_md: str,
    shot_min_score: float = 76.0,
    min_fixture_quality_score: float = 0.70,
    allowed_attacking_style_labels: list[str] | None = None,
    min_shot_name_count: int = 4,
    min_top4_avg_score: float = 96.0,
    min_battle_on_score: float = 0.58,
    require_high_goal_env: bool = True,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv)
    scored = _build_psi(base)
    gate = _build_fixture_gate(base, min_fixture_quality_score=min_fixture_quality_score)
    gate_cols = [
        "fixture_key", "fixture_attack_quality_score", "strong_attack_support_flag", "strong_goal_env_support_flag",
        "fixture_attack_quality_pass_flag", "fixture_attacking_style_label", "fixture_style_label", "fixture_attack_reason_codes",
        "og_goal_environment_label", "og_battle_on_score",
    ]
    overlap = [c for c in gate_cols if c != "fixture_key" and c in scored.columns]
    if overlap:
        scored = scored.drop(columns=overlap)
    scored = scored.merge(gate[gate_cols], on="fixture_key", how="left")
    scored = scored[
        scored["player_shot_index"].ge(shot_min_score)
        & scored["fixture_attack_quality_pass_flag"].eq(1)
        & scored["strong_attack_support_flag"].eq(1)
        & scored["strong_goal_env_support_flag"].eq(1)
        & scored["og_battle_on_score"].ge(min_battle_on_score)
    ].copy()
    if require_high_goal_env:
        scored = scored[scored["og_goal_environment_label"].astype("string").str.upper().eq("HIGH")].copy()
    if allowed_attacking_style_labels:
        allowed = {x.strip().upper() for x in allowed_attacking_style_labels if x.strip()}
        scored = scored[scored["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()

    rows = []
    for fixture_key, group in scored.groupby("fixture_key", sort=False):
        ranked = group.sort_values("player_shot_index", ascending=False).reset_index(drop=True)
        if len(ranked) < min_shot_name_count:
            continue
        top4_avg_score = float(ranked["player_shot_index"].head(4).mean())
        if top4_avg_score < min_top4_avg_score:
            continue
        rows.append(
            {
                "fixture_key": fixture_key,
                "league": ranked["league"].iloc[0],
                "home_team_name": ranked["home_team_name"].iloc[0],
                "away_team_name": ranked["away_team_name"].iloc[0],
                "fixture_attacking_style_label": ranked["fixture_attacking_style_label"].iloc[0],
                "fixture_attack_quality_score": float(ranked["fixture_attack_quality_score"].iloc[0]),
                "og_goal_environment_label": ranked["og_goal_environment_label"].iloc[0],
                "og_battle_on_score": float(ranked["og_battle_on_score"].iloc[0]),
                "shot_name_count": len(ranked),
                "top4_avg_score": round(top4_avg_score, 4),
                "top4_names": " | ".join(ranked["player_name"].astype(str).head(4).tolist()),
                "top4_scores": " | ".join([f"{x:.1f}" for x in ranked["player_shot_index"].head(4).tolist()]),
                "reason_codes": ranked["fixture_attack_reason_codes"].iloc[0],
            }
        )
    out = pd.DataFrame(rows).sort_values(["shot_name_count", "top4_avg_score", "fixture_attack_quality_score", "fixture_key"], ascending=[False, False, False, True]) if rows else pd.DataFrame()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = ["# Shots 4+ Candidate Export", ""]
    if out.empty:
        lines.append("- no qualifying 4+ shot-name fixtures")
    else:
        for row in out.itertuples(index=False):
            lines.append(
                f"- {row.fixture_key} | {row.home_team_name} vs {row.away_team_name} | attack={row.fixture_attacking_style_label} | quality={row.fixture_attack_quality_score:.3f} | top4_avg={row.top4_avg_score:.1f} | names={row.shot_name_count} | {row.top4_names} | scores={row.top4_scores}"
            )
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export elite fixtures with 4+ viable shot names.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--shot-min-score", type=float, default=76.0)
    parser.add_argument("--min-fixture-quality-score", type=float, default=0.70)
    parser.add_argument("--allowed-attacking-style-labels", default="")
    parser.add_argument("--min-shot-name-count", type=int, default=4)
    parser.add_argument("--min-top4-avg-score", type=float, default=96.0)
    parser.add_argument("--min-battle-on-score", type=float, default=0.58)
    parser.add_argument("--require-high-goal-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    styles = [x.strip() for x in args.allowed_attacking_style_labels.split(",") if x.strip()]
    out = export_shots_4plus_candidates(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        shot_min_score=args.shot_min_score,
        min_fixture_quality_score=args.min_fixture_quality_score,
        allowed_attacking_style_labels=styles or None,
        min_shot_name_count=args.min_shot_name_count,
        min_top4_avg_score=args.min_top4_avg_score,
        min_battle_on_score=args.min_battle_on_score,
        require_high_goal_env=args.require_high_goal_env,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"fixtures: {len(out)}")


if __name__ == "__main__":
    main()
