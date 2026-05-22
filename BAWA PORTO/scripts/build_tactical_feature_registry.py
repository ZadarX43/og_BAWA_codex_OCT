#!/usr/bin/env python3
"""Build the tactical feature registry.

Research-only registry that maps tactical ideas from the knowledge vault to
measurable sources, leakage risk, target markets, and implementation status.
It does not change production prediction, routing, deploy tiers, or slips.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-08" / "tactical_feature_registry"


REGISTRY_ROWS: list[dict[str, Any]] = [
    {
        "feature_id": "GOAL_RATE_FIELD_TILT",
        "family": "GOAL_RATE_ADJUSTMENT",
        "tactical_concept": "territorial control / field tilt",
        "measurable_source": "team shots profile, possession/pass territory proxies, future event-provider field tilt",
        "source_columns": "home_team_expected_shots|away_team_expected_shots|home_team_possession_l5|away_team_possession_l5|home_team_passes_l5|away_team_passes_l5",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "FTR|BTTS|OU25|TEAM_GOALS_15|GOAL_RANGE|WINNING_MARGIN",
        "target_shadow_stages": "GOAL_RANGE_SHADOW|WINNING_MARGIN_2_PLUS_SHADOW|WINNING_MARGIN_3_PLUS_WATCH|TOTAL_SHOTS_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "FIXTURE_MARKET_INTELLIGENCE_BOARD",
        "validation_status": "NEEDS_WALKFORWARD",
    },
    {
        "feature_id": "GOAL_RATE_CENTRAL_ACCESS",
        "family": "GOAL_RATE_ADJUSTMENT",
        "tactical_concept": "zone-14 access, cutbacks, through balls, box occupation",
        "measurable_source": "manual/team tactical vault tags, future event-provider zone/cutback data",
        "source_columns": "manual_central_access_flag|zone14_receptions|cutbacks|box_occupation_count",
        "pre_match_availability": "MANUAL_REGISTRY_FIRST",
        "leakage_risk": "MEDIUM",
        "target_markets": "FTR|OU25|TEAM_GOALS_15|TEAM_GOALS_25|WIN_GE2|GOAL_RANGE",
        "target_shadow_stages": "TEAM_GOAL_MARKET|TEAM_GOAL_COMBO|WINNING_MARGIN_2_PLUS_SHADOW",
        "implementation_status": "PLANNED",
        "first_consumer": "TACTICAL_GOAL_RATE_ADJUSTMENT_AUDIT",
        "validation_status": "NOT_STARTED",
    },
    {
        "feature_id": "REST_DEFENCE_TRANSITION_EXPOSURE",
        "family": "REST_DEFENCE_TRANSITION",
        "tactical_concept": "weak rest defence creates opponent transition SOT",
        "measurable_source": "direct attacks conceded, high turnovers conceded, opponent outlet quality, SOT pressure",
        "source_columns": "fixture_territorial_stress_score|fixture_attack_pressure_score|away_team_expected_sot|home_team_expected_sot|keeper_expected_sot_faced",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW_TO_MEDIUM",
        "target_markets": "BTTS|OU25|MATCH_OVER_35|TOTAL_SOT|KEEPER_SAVES|PLAYER_SHOTS|PLAYER_SOT",
        "target_shadow_stages": "TOTAL_SOT_SHADOW|KEEPER_SAVES_1_5_LIVE_SHADOW|KEEPER_SAVES_2_5_LIVE_SHADOW|PLAYER_SHOTS_1_5_INTERACTION_WATCH|PLAYER_SOT_0_5_INTERACTION_WATCH",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "KEEPER_SAVES_INTELLIGENCE|FIXTURE_MARKET_INTELLIGENCE_BOARD",
        "validation_status": "LIVE_ACCUMULATING",
    },
    {
        "feature_id": "WIDE_ISOLATION_DRIBBLER",
        "family": "WIDE_ISOLATION",
        "tactical_concept": "wide 1v1 isolation for chaos dribblers / inverted wingers",
        "measurable_source": "wide duel score, wide overload flags, fouls drawn profile, winger role",
        "source_columns": "fixture_wide_duel_score|formation_wide_overload_flag|fouls_won_per90|tactical_role|position_group",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_FOULED|PLAYER_CARDS|PLAYER_TACKLES|TEAM_CORNERS|KEY_PASSES",
        "target_shadow_stages": "PLAYER_FOULED_0_5_INTERACTION_WATCH|PLAYER_FOULED_1_5_INTERACTION_WATCH|TEAM_CORNERS_4_5_LIVE_SHADOW|TEAM_MOST_CORNERS_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "PLAYER_EVENT_INTERACTION|CORNERS_INTELLIGENCE",
        "validation_status": "LIVE_ACCUMULATING",
    },
    {
        "feature_id": "WIDE_ISOLATION_INSIDE_FORWARD_SHOT",
        "family": "WIDE_ISOLATION",
        "tactical_concept": "inside forward isolation creates shots/SOT",
        "measurable_source": "role archetype, shots per90, opponent allowance, attack pressure",
        "source_columns": "tactical_role|shots_per90|shots_on_target_per90|fixture_attack_pressure_score|opp_attack_allowed_*",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_SHOTS|PLAYER_SOT|GOAL_INVOLVEMENT",
        "target_shadow_stages": "PLAYER_SHOTS_1_5_INTERACTION_WATCH|PLAYER_SHOTS_2_5_INTERACTION_WATCH|PLAYER_SOT_0_5_INTERACTION_WATCH",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "PLAYER_EVENT_INTERACTION",
        "validation_status": "LIVE_ACCUMULATING",
    },
    {
        "feature_id": "ROLE_ALLOWANCE_STRIKER_ARCHETYPE",
        "family": "ROLE_ALLOWANCE",
        "tactical_concept": "striker archetype changes shot/SOT/foul opportunity",
        "measurable_source": "role archetype registry, player event rates, opponent role allowance",
        "source_columns": "tactical_role|position_group|shots_per90|fouls_won_per90|opp_allowed_striker_*",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_SHOTS|PLAYER_SOT|PLAYER_FOULED|GOAL_INVOLVEMENT",
        "target_shadow_stages": "PLAYER_SHOTS_1_5_INTERACTION_WATCH|PLAYER_SHOTS_2_5_INTERACTION_WATCH|PLAYER_SOT_0_5_INTERACTION_WATCH|PLAYER_FOULED_0_5_INTERACTION_WATCH",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "PLAYER_EVENT_INTERACTION",
        "validation_status": "NEEDS_3Y_V2_AUDIT",
    },
    {
        "feature_id": "ROLE_ALLOWANCE_MIDFIELD_DESTROYER",
        "family": "ROLE_ALLOWANCE",
        "tactical_concept": "destroyer / holding midfielder tackle and foul volume",
        "measurable_source": "tactical role, tackles per90, foul density, opponent possession projection",
        "source_columns": "tactical_role|tackles_per90|fixture_tackle_density_score|opponent_possession_projection",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_TACKLES|PLAYER_FOULS_COMMITTED|PLAYER_CARDS",
        "target_shadow_stages": "PLAYER_TACKLES_1_5_LIVE_SHADOW|PLAYER_TACKLES_2_5_LIVE_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "PLAYER_EVENT_TACKLES",
        "validation_status": "LIVE_ACCUMULATING",
    },
    {
        "feature_id": "ROLE_ALLOWANCE_FULLBACK_OVERLAP",
        "family": "ROLE_ALLOWANCE",
        "tactical_concept": "overlapping/wingback full-back drives fouls, tackles, key passes, corners",
        "measurable_source": "full-back tactical role, wide overload, key passes, crosses/corners proxies",
        "source_columns": "tactical_role|formation_wide_overload_flag|key_passes_per90|team_corners_for_l5",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_KEY_PASSES|PLAYER_FOULED|PLAYER_TACKLES|TEAM_CORNERS",
        "target_shadow_stages": "KEY_PASSES_0_5_LIVE_SHADOW|PLAYER_FOULED_0_5_INTERACTION_WATCH|PLAYER_TACKLES_1_5_LIVE_SHADOW|TEAM_MOST_CORNERS_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "KEY_PASS_ASSIST_INTELLIGENCE|PLAYER_EVENT_TACKLES",
        "validation_status": "NEEDS_3Y_V2_AUDIT",
    },
    {
        "feature_id": "LINEUP_FRAGILITY_PRESS_BREAKER",
        "family": "LINEUP_FRAGILITY",
        "tactical_concept": "missing press breaker damages build-up floor",
        "measurable_source": "lineup role registry, confirmed lineups, pass/carry profile",
        "source_columns": "confirmed_starter_flag|press_breaker_role_flag|lineup_watch_flags",
        "pre_match_availability": "CONFIRMED_LINEUP_REQUIRED",
        "leakage_risk": "LOW_IF_PRE_KICKOFF",
        "target_markets": "FTR|BTTS|OU25|TEAM_GOALS|PLAYER_EVENTS",
        "target_shadow_stages": "ALL",
        "implementation_status": "PLANNED",
        "first_consumer": "LINEUP_FRAGILITY_SIDECAR",
        "validation_status": "NOT_STARTED",
    },
    {
        "feature_id": "LINEUP_FRAGILITY_REST_DEFENCE_MID",
        "family": "LINEUP_FRAGILITY",
        "tactical_concept": "missing destroyer/rest-defence midfielder raises transition risk",
        "measurable_source": "lineup role registry, confirmed lineups, defensive transition profile",
        "source_columns": "confirmed_starter_flag|rest_defence_mid_role_flag|lineup_watch_flags",
        "pre_match_availability": "CONFIRMED_LINEUP_REQUIRED",
        "leakage_risk": "LOW_IF_PRE_KICKOFF",
        "target_markets": "BTTS|OU25|TOTAL_SOT|KEEPER_SAVES|PLAYER_SHOTS",
        "target_shadow_stages": "TOTAL_SOT_SHADOW|KEEPER_SAVES_1_5_LIVE_SHADOW|PLAYER_SHOTS_1_5_INTERACTION_WATCH",
        "implementation_status": "PLANNED",
        "first_consumer": "LINEUP_FRAGILITY_SIDECAR",
        "validation_status": "NOT_STARTED",
    },
    {
        "feature_id": "SET_PIECE_CORNER_PRESSURE",
        "family": "SET_PIECE_TERRITORY",
        "tactical_concept": "wide pressure and blocked crosses create corner volume",
        "measurable_source": "team corner pressure, attack territory, wide overload",
        "source_columns": "fixture_corner_pressure_score|team_corners_for_l5|team_corners_against_l5|fixture_wide_duel_score",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "TEAM_CORNERS|TEAM_MOST_CORNERS|TOTAL_SHOTS",
        "target_shadow_stages": "TEAM_CORNERS_4_5_LIVE_SHADOW|TEAM_CORNERS_5_5_LIVE_SHADOW|TEAM_MOST_CORNERS_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "CORNERS_INTELLIGENCE|FIXTURE_MARKET_INTELLIGENCE_BOARD",
        "validation_status": "AWAITING_LIVE_RESULTS",
    },
    {
        "feature_id": "KEEPER_WORKLOAD_SOT_PRESSURE",
        "family": "KEEPER_SAVE_PRESSURE",
        "tactical_concept": "opponent SOT pressure creates save-volume environment",
        "measurable_source": "team SOT profile, keeper projected starter, goal spine pressure",
        "source_columns": "keeper_expected_sot_faced|home_team_expected_sot|away_team_expected_sot|og_goal_environment_score",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "KEEPER_SAVES|TOTAL_SOT|TEAM_MOST_SOT",
        "target_shadow_stages": "KEEPER_SAVES_1_5_LIVE_SHADOW|KEEPER_SAVES_2_5_LIVE_SHADOW|KEEPER_SAVES_3_5_LIVE_SHADOW|TOTAL_SOT_SHADOW|TEAM_MOST_SOT_SHADOW",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "KEEPER_SAVES_INTELLIGENCE|FIXTURE_MARKET_INTELLIGENCE_BOARD",
        "validation_status": "LIVE_ACCUMULATING",
    },
    {
        "feature_id": "CARD_FOUL_ECOSYSTEM",
        "family": "REFEREE_AND_FOUL_ECOSYSTEM",
        "tactical_concept": "ref strictness plus team foul profile affects cards/team-most-cards",
        "measurable_source": "referee profile, foul density, team yellows/fouls",
        "source_columns": "ref_cards_per_match|fixture_foul_density_score|team_avg_yellows|team_avg_fouls",
        "pre_match_availability": "PROFILE_PROXY_NOW",
        "leakage_risk": "LOW",
        "target_markets": "PLAYER_CARDS|TEAM_MOST_CARDS|PLAYER_FOULS_COMMITTED",
        "target_shadow_stages": "TEAM_MOST_CARDS_WATCH|PLAYER_CARDS_0_5|PLAYER_FOULS_COMMITTED",
        "implementation_status": "SIDE_CAR_PARTIAL",
        "first_consumer": "FIXTURE_MARKET_INTELLIGENCE_BOARD|CARDS_HAZARD_AUDIT",
        "validation_status": "RESEARCH_ONLY",
    },
]


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, registry: pd.DataFrame) -> None:
    family_counts = registry.groupby(["family", "implementation_status"], dropna=False).size().reset_index(name="rows")
    leakage_counts = registry.groupby(["leakage_risk"], dropna=False).size().reset_index(name="rows")
    lines = [
        "# Tactical Feature Registry",
        "",
        "Research-only registry mapping tactical vault concepts to measurable features and market hooks.",
        "",
        "## Safety",
        "- No production prediction changes.",
        "- No deploy routing or tier changes.",
        "- No tactical feature should become production behavior without walkforward proof.",
        "",
        "## Family Counts",
        markdown_table(family_counts),
        "",
        "## Leakage Counts",
        markdown_table(leakage_counts),
        "",
        "## Registry",
        markdown_table(
            registry[
                [
                    "feature_id",
                    "family",
                    "tactical_concept",
                    "pre_match_availability",
                    "leakage_risk",
                    "target_markets",
                    "implementation_status",
                    "validation_status",
                ]
            ],
            max_rows=120,
        ),
    ]
    (outdir / "TACTICAL_FEATURE_REGISTRY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    registry = pd.DataFrame(REGISTRY_ROWS)
    registry.to_csv(args.outdir / "TACTICAL_FEATURE_REGISTRY.csv", index=False)
    write_report(args.outdir, registry)
    print(f"WROTE {args.outdir}")
    print(f"rows={len(registry)}")
    print(registry.groupby(["family", "implementation_status"]).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
