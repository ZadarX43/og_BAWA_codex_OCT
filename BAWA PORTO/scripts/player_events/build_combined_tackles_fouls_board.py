from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_combined_player_events_board import _build_fixture_gate
from render_fouls_committed_report import _build_fci
from render_player_tackles_report import _build_tki


def _prep_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "fouls_committed":
        scored = _build_fci(df.copy())
        scored = scored.rename(
            columns={
                "foul_commitment_index": "market_score",
                "confidence_label": "market_confidence",
            }
        )
        scored["market"] = "fouls_committed"
        return scored
    if market == "tackles":
        scored = _build_tki(df.copy())
        scored = scored.rename(
            columns={
                "player_tackle_index": "market_score",
                "confidence_label": "market_confidence",
            }
        )
        scored["market"] = "tackles"
        return scored
    raise ValueError(f"Unsupported market: {market}")


def build_combined_board(
    input_csv: str,
    output_csv: str,
    output_md: str,
    foul_min_score: float = 64.0,
    tackle_min_score: float = 66.0,
    max_fouls_per_fixture: int = 1,
    max_tackles_per_fixture: int = 1,
    leagues_filter: list[str] | None = None,
    min_fixture_quality_score: float = 0.72,
    goal_overlay_column: str = "og_goal_environment_score",
    require_any_support: bool = True,
    require_goal_env_support: bool = True,
    require_attack_support: bool = False,
    dual_trigger_only: bool = True,
    allowed_style_labels: list[str] | None = None,
    require_underdog_fullback_wide_overload: bool = False,
    require_underdog_dm_midfield_grind: bool = False,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv)
    if leagues_filter:
        lf = {x.strip().lower() for x in leagues_filter if x.strip()}
        base = base[base["league"].astype("string").str.lower().isin(lf)].copy()

    fixture_gate = _build_fixture_gate(
        base,
        goal_overlay_column=goal_overlay_column,
        min_fixture_quality_score=min_fixture_quality_score,
    )
    gate_cols = [
        "fixture_key",
        "fixture_ref_score",
        "fixture_team_pressure_score",
        "fixture_stakes_score",
        "fixture_possession_stress_score",
        "fixture_lineup_quality_score",
        "goal_environment_overlay_score",
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "fixture_wide_duel_score",
        "fixture_style_label",
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "fixture_territorial_stress_score",
        "fixture_attacking_style_label",
        "strong_ref_support_flag",
        "strong_team_pressure_support_flag",
        "strong_goal_environment_support_flag",
        "strong_contact_style_support_flag",
        "strong_attack_support_flag",
        "strong_lineup_quality_support_flag",
        "fixture_quality_score",
        "fixture_quality_pass_flag",
        "fixture_quality_reason_codes",
    ]

    fouls = _prep_market(base, "fouls_committed")
    tackles = _prep_market(base, "tackles")
    for df in (fouls, tackles):
        overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in df.columns]
        if overlap_cols:
            df.drop(columns=overlap_cols, inplace=True)

    fouls = fouls.merge(fixture_gate[gate_cols], on="fixture_key", how="left")
    tackles = tackles.merge(fixture_gate[gate_cols], on="fixture_key", how="left")

    fouls = fouls[fouls["market_score"].ge(foul_min_score)].copy()
    tackles = tackles[tackles["market_score"].ge(tackle_min_score)].copy()
    fouls = fouls[fouls["fixture_quality_pass_flag"].eq(1)].copy()
    tackles = tackles[tackles["fixture_quality_pass_flag"].eq(1)].copy()

    if require_any_support:
        def _mask(df: pd.DataFrame) -> pd.Series:
            return (
                df["strong_ref_support_flag"].eq(1)
                | df["strong_team_pressure_support_flag"].eq(1)
                | df["strong_goal_environment_support_flag"].eq(1)
                | df["strong_contact_style_support_flag"].eq(1)
                | df["strong_attack_support_flag"].eq(1)
                | df["strong_lineup_quality_support_flag"].eq(1)
            )

        fouls = fouls[_mask(fouls)].copy()
        tackles = tackles[_mask(tackles)].copy()
    if require_goal_env_support:
        fouls = fouls[fouls["strong_goal_environment_support_flag"].eq(1)].copy()
        tackles = tackles[tackles["strong_goal_environment_support_flag"].eq(1)].copy()
    if require_attack_support:
        fouls = fouls[fouls["strong_attack_support_flag"].eq(1)].copy()
        tackles = tackles[tackles["strong_attack_support_flag"].eq(1)].copy()
    if allowed_style_labels:
        allowed = {x.strip().upper() for x in allowed_style_labels if x.strip()}
        fouls = fouls[fouls["fixture_style_label"].astype("string").str.upper().isin(allowed)].copy()
        tackles = tackles[tackles["fixture_style_label"].astype("string").str.upper().isin(allowed)].copy()
    if require_underdog_fullback_wide_overload:
        fouls = fouls[fouls["underdog_fullback_wide_overload_flag"].eq(1)].copy()
        tackles = tackles[tackles["underdog_fullback_wide_overload_flag"].eq(1)].copy()
    if require_underdog_dm_midfield_grind:
        fouls = fouls[fouls["underdog_dm_midfield_grind_flag"].eq(1)].copy()
        tackles = tackles[tackles["underdog_dm_midfield_grind_flag"].eq(1)].copy()

    fouls = (
        fouls.sort_values(["fixture_key", "market_score"], ascending=[True, False])
        .groupby("fixture_key", as_index=False, group_keys=False)
        .head(max_fouls_per_fixture)
    )
    tackles = (
        tackles.sort_values(["fixture_key", "market_score"], ascending=[True, False])
        .groupby("fixture_key", as_index=False, group_keys=False)
        .head(max_tackles_per_fixture)
    )

    keep_cols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "player_team_side",
        "position_group",
        "tactical_role",
        "manual_pitch_side",
        "manual_flank_role",
        "manual_overload_target_side",
        "manual_side_override_flag",
        "fouls_per90",
        "tackles_per90",
        "interceptions_per90",
        "team_avg_fouls",
        "team_avg_yellows",
        "opponent_possession_projection",
        "fixture_ref_score",
        "fixture_team_pressure_score",
        "fixture_stakes_score",
        "fixture_possession_stress_score",
        "fixture_lineup_quality_score",
        "goal_environment_overlay_score",
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "fixture_wide_duel_score",
        "fixture_style_label",
        "team_formation",
        "opponent_formation",
        "formation_matchup_label",
        "formation_wide_overload_flag",
        "formation_midfield_grind_flag",
        "underdog_fullback_wide_overload_flag",
        "underdog_dm_midfield_grind_flag",
        "formation_pressure_score",
        "strong_ref_support_flag",
        "strong_team_pressure_support_flag",
        "strong_goal_environment_support_flag",
        "strong_contact_style_support_flag",
        "strong_lineup_quality_support_flag",
        "fixture_quality_score",
        "fixture_quality_pass_flag",
        "fixture_quality_reason_codes",
        "og_battle_on_score",
        "og_goal_environment_label",
        "player_quality_score_l5",
        "player_form_tier",
        "player_quality_rank_in_position",
        "starting_xi_team_quality_score",
        "starting_xi_defensive_quality_score",
        "starting_xi_quality_edge",
        "market",
        "market_score",
        "market_confidence",
        "analyst_notes",
    ]

    out = pd.concat([fouls[keep_cols], tackles[keep_cols]], ignore_index=True)
    if not out.empty:
        trigger_counts = (
            out.groupby(["fixture_key", "player_name"], as_index=False)
            .agg(
                trigger_count=("market", "nunique"),
                trigger_markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
            )
        )
        out = out.merge(trigger_counts, on=["fixture_key", "player_name"], how="left")
        out["same_player_dual_trigger_flag"] = out["trigger_count"].ge(2).astype(int)
    else:
        out["trigger_count"] = pd.Series(dtype=int)
        out["trigger_markets"] = pd.Series(dtype="string")
        out["same_player_dual_trigger_flag"] = pd.Series(dtype=int)

    if dual_trigger_only:
        out = out[out["same_player_dual_trigger_flag"].eq(1)].copy()

    if not out.empty:
        out["combo_reason_bucket"] = (
            out["fixture_style_label"].astype("string").str.upper()
            + "__"
            + out["formation_matchup_label"].astype("string").str.replace(" vs ", "__VS__", regex=False).str.upper()
            + "__"
            + out["trigger_markets"].astype("string").str.replace("|", "_", regex=False).str.upper()
        )
    else:
        out["combo_reason_bucket"] = pd.Series(dtype="string")

    out = out.sort_values(["match_date", "fixture_key", "market", "market_score"], ascending=[False, True, True, False]).reset_index(drop=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    lines = [
        "# Combined Tackles + Fouls Board",
        "",
        "## Rules",
        f"- max fouls names per fixture: {max_fouls_per_fixture}",
        f"- max tackles names per fixture: {max_tackles_per_fixture}",
        f"- fouls minimum score: {foul_min_score}",
        f"- tackles minimum score: {tackle_min_score}",
        f"- minimum fixture quality score: {min_fixture_quality_score}",
        f"- dual-trigger only: {dual_trigger_only}",
        f"- allowed style labels: {', '.join(allowed_style_labels) if allowed_style_labels else 'all'}",
        f"- require underdog full-back + wide overload: {require_underdog_fullback_wide_overload}",
        f"- require underdog DM + midfield grind: {require_underdog_dm_midfield_grind}",
        "",
        "## Board Summary",
        f"- rows: {len(out)}",
        f"- fixtures covered: {out['fixture_key'].nunique() if not out.empty else 0}",
        f"- fouls rows: {int((out['market'] == 'fouls_committed').sum()) if not out.empty else 0}",
        f"- tackles rows: {int((out['market'] == 'tackles').sum()) if not out.empty else 0}",
        f"- average fixture quality score: {round(float(out['fixture_quality_score'].mean()), 3) if not out.empty else 0.0}",
        "",
        "## Fixture Samples",
    ]
    for fixture_key, group in out.groupby("fixture_key", sort=False):
        fx = group.iloc[0]
        lines.append(f"### {fixture_key}")
        lines.append(
            f"- fixture quality: {fx['fixture_quality_score']:.3f} | contact: {fx['fixture_style_label']} | formation: {fx['formation_matchup_label']} | battle_on: {fx['og_battle_on_score']:.3f} | reasons: {fx['fixture_quality_reason_codes']}"
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"- `{row.market}`: {row.player_name} ({row.team_name}) | {row.tactical_role} | score={row.market_score:.1f} | quality={row.player_quality_score_l5:.1f} | edge={row.starting_xi_quality_edge:.1f} | form_pressure={row.formation_pressure_score:.2f}"
            )
        lines.append("")
        if len(lines) > 120:
            break

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined same-fixture tackles + fouls beta board.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--foul-min-score", type=float, default=64.0)
    parser.add_argument("--tackle-min-score", type=float, default=66.0)
    parser.add_argument("--max-fouls-per-fixture", type=int, default=1)
    parser.add_argument("--max-tackles-per-fixture", type=int, default=1)
    parser.add_argument("--leagues", default="")
    parser.add_argument("--min-fixture-quality-score", type=float, default=0.72)
    parser.add_argument("--goal-overlay-column", default="og_goal_environment_score")
    parser.add_argument("--require-any-support", action="store_true")
    parser.add_argument("--require-goal-env-support", action="store_true")
    parser.add_argument("--require-attack-support", action="store_true")
    parser.add_argument("--dual-trigger-only", action="store_true")
    parser.add_argument("--allowed-style-labels", default="")
    parser.add_argument("--require-underdog-fullback-wide-overload", action="store_true")
    parser.add_argument("--require-underdog-dm-midfield-grind", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leagues_filter = [x.strip() for x in args.leagues.split(",") if x.strip()]
    allowed_style_labels = [x.strip() for x in args.allowed_style_labels.split(",") if x.strip()]
    out = build_combined_board(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        foul_min_score=args.foul_min_score,
        tackle_min_score=args.tackle_min_score,
        max_fouls_per_fixture=args.max_fouls_per_fixture,
        max_tackles_per_fixture=args.max_tackles_per_fixture,
        leagues_filter=leagues_filter or None,
        min_fixture_quality_score=args.min_fixture_quality_score,
        goal_overlay_column=args.goal_overlay_column,
        require_any_support=args.require_any_support,
        require_goal_env_support=args.require_goal_env_support,
        require_attack_support=args.require_attack_support,
        dual_trigger_only=args.dual_trigger_only,
        allowed_style_labels=allowed_style_labels or None,
        require_underdog_fullback_wide_overload=args.require_underdog_fullback_wide_overload,
        require_underdog_dm_midfield_grind=args.require_underdog_dm_midfield_grind,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
    print(f"fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")


if __name__ == "__main__":
    main()
