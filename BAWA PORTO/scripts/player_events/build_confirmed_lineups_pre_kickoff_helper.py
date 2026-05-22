from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_MARKETS = {"fouls_committed", "tackles", "yellow_cards"}
TARGET_WIDE_ROLES = {"Wide defender / wing-back", "Wide midfielder / winger", "Wide forward"}


def _ensure_fixture_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fixture_quality_score" not in out.columns:
        out["fixture_quality_score"] = (
            0.30 * pd.to_numeric(out.get("og_goal_environment_score", 0.0), errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(out.get("fixture_foul_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.20 * pd.to_numeric(out.get("fixture_attack_pressure_score", 0.0), errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(out.get("fixture_tackle_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out.get("formation_pressure_score", 0.0), errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=1.0)
    return out


def build_helper(master_weekend_csv: str, fixture_input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    weekend = pd.read_csv(master_weekend_csv, low_memory=False)
    fixture_input = pd.read_csv(fixture_input_csv, low_memory=False)
    if weekend.empty or fixture_input.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Confirmed-Lineups Pre-Kickoff Helper\n\nNo rows matched.\n")
        return out

    elite_fixtures = sorted(weekend["fixture_key"].dropna().astype(str).unique())
    candidates = fixture_input[fixture_input["fixture_key"].astype(str).isin(elite_fixtures)].copy()
    candidates = _ensure_fixture_quality(candidates)
    candidates = candidates[
        candidates["tactical_role"].isin(TARGET_WIDE_ROLES)
        & pd.to_numeric(candidates["underdog_fullback_wide_overload_flag"], errors="coerce").fillna(0).ge(1)
        & candidates["manual_pitch_side"].astype(str).fillna("UNSET").isin(["UNSET", "NONE", "NAN"])
    ].copy()
    if candidates.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Confirmed-Lineups Pre-Kickoff Helper\n\nNo elite fixtures currently need manual side confirmation.\n")
        return out

    weekend_contact = weekend[weekend["market"].isin(TARGET_MARKETS)].copy()
    fixture_priority = (
        weekend_contact.groupby("fixture_key", as_index=False)
        .agg(
            priority_rows=("market", "count"),
            best_priority_bucket=("priority_rank", "min"),
            best_contact_score=("market_score", "max"),
            fixture_confidence_note=("fixture_confidence_note", "first"),
            source_family_tag=("source_family_tag", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    out = candidates.merge(fixture_priority, on="fixture_key", how="left")
    out["manual_side_needed_flag"] = 1
    out["lineup_confirmation_priority"] = (
        0.30 * pd.to_numeric(out["formation_pressure_score"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["fixture_quality_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["fixture_wide_duel_score"], errors="coerce").fillna(0.0)
        + 0.15 * ((-pd.to_numeric(out["starting_xi_quality_edge"], errors="coerce").fillna(0.0)).clip(lower=0.0, upper=20.0) / 20.0)
        + 0.15 * pd.to_numeric(out["player_quality_score_l5"], errors="coerce").fillna(0.0) / 100.0
    )
    keep = [
        "fixture_key",
        "match_date",
        "competition",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "tactical_role",
        "formation_matchup_label",
        "wide_overload_target_side",
        "formation_left_wide_overload_score",
        "formation_right_wide_overload_score",
        "fixture_style_label",
        "fixture_quality_score",
        "formation_pressure_score",
        "starting_xi_quality_edge",
        "player_quality_score_l5",
        "source_family_tag",
        "best_contact_score",
        "fixture_confidence_note",
        "lineup_confirmation_priority",
        "manual_side_needed_flag",
    ]
    out = out[keep].sort_values(
        ["lineup_confirmation_priority", "fixture_quality_score", "best_contact_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Confirmed-Lineups Pre-Kickoff Helper",
        "",
        "Use this after official lineups land. These are the elite-fixture wide-role players that still need LEFT/RIGHT confirmation.",
        "",
    ]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | {first['formation_matchup_label']} | families={first['source_family_tag']} | note={first['fixture_confidence_note']}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['tactical_role']} | target_side={row['wide_overload_target_side']} | left_score={row['formation_left_wide_overload_score']:.3f} | right_score={row['formation_right_wide_overload_score']:.3f} | priority={row['lineup_confirmation_priority']:.3f}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a confirmed-lineups helper for elite fixtures that still need manual side confirmation.")
    parser.add_argument("--master-weekend-csv", required=True)
    parser.add_argument("--fixture-input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = build_helper(args.master_weekend_csv, args.fixture_input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")
