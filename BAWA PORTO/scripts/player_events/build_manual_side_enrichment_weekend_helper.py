from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_FORMATIONS = {"4-2-3-1 vs 4-4-2", "4-4-2 vs 4-2-3-1", "4-2-3-1 vs 4-3-3", "4-3-3 vs 4-2-3-1"}
TARGET_ROLES = {"Wide defender / wing-back", "Wide midfielder / winger", "Wide forward"}


def build_helper(input_csv: str, output_csv: str, output_md: str, top_n: int = 40) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[df["formation_matchup_label"].isin(TARGET_FORMATIONS)].copy()
    df = df[df["tactical_role"].isin(TARGET_ROLES)].copy()
    if df.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Manual Side-Enrichment Weekend Helper\n\nNo rows matched.\n")
        return out

    if "fixture_quality_score" not in df.columns:
        df["fixture_quality_score"] = (
            0.30 * pd.to_numeric(df.get("og_goal_environment_score", 0.0), errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(df.get("fixture_foul_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.20 * pd.to_numeric(df.get("fixture_attack_pressure_score", 0.0), errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(df.get("fixture_tackle_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(df.get("formation_pressure_score", 0.0), errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=1.0)
    else:
        df["fixture_quality_score"] = pd.to_numeric(df["fixture_quality_score"], errors="coerce").fillna(0.0)
    df["side_enrichment_priority"] = (
        0.28 * pd.to_numeric(df.get("formation_wide_overload_flag", 0.0), errors="coerce").fillna(0.0)
        + 0.22 * pd.to_numeric(df.get("weak_flank_overload_flag", 0.0), errors="coerce").fillna(0.0)
        + 0.18 * pd.to_numeric(df.get("fixture_wide_duel_score", 0.0), errors="coerce").fillna(0.0)
        + 0.14 * pd.to_numeric(df.get("formation_pressure_score", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(df.get("fixture_quality_score", 0.0), errors="coerce").fillna(0.0)
        + 0.08 * ((-pd.to_numeric(df.get("starting_xi_quality_edge", 0.0), errors="coerce").fillna(0.0)).clip(lower=0.0, upper=20.0) / 20.0)
    )
    df = df.sort_values(["side_enrichment_priority", "fixture_quality_score"], ascending=[False, False])
    keep = [
        "fixture_key","match_date","competition","home_team_name","away_team_name","team_name","player_name",
        "player_team_side","position_group","tactical_role","formation_matchup_label","wide_overload_target_side",
        "formation_left_wide_overload_score","formation_right_wide_overload_score","formation_pressure_score",
        "fixture_wide_duel_score","fixture_style_label","starting_xi_quality_edge","player_quality_score_l5",
        "manual_pitch_side","manual_overload_target_side","manual_side_override_flag","side_enrichment_priority",
    ]
    out = df[keep].head(top_n).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Manual Side-Enrichment Weekend Helper",
        "",
        "Use this as the short confirmation list before manually pinning LEFT/RIGHT for elite fixtures.",
        "",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | {row['tactical_role']} | formation {row['formation_matchup_label']} | target={row['wide_overload_target_side']} | left_score={row['formation_left_wide_overload_score']:.3f} | right_score={row['formation_right_wide_overload_score']:.3f} | priority={row['side_enrichment_priority']:.3f}"
        )
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weekend helper for manual side enrichment.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = build_helper(args.input, args.output_csv, args.output_md, top_n=args.top_n)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)}")
