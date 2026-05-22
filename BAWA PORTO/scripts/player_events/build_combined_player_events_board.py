from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_fouls_committed_report import _build_fci
from render_yellow_card_report import _build_bpi


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _norm_inverse_ratio(series: pd.Series, lo: float = 1.0, hi: float = 10.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    out = out.clip(lower=lo, upper=hi)
    return ((hi - out) / (hi - lo)).fillna(0.0)


def _prep_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "yellow_card":
        scored = _build_bpi(df.copy())
        scored = scored.rename(
            columns={
                "booking_probability_index": "market_score",
                "confidence_label": "market_confidence",
            }
        )
        scored["market"] = "yellow_card"
    elif market == "fouls_committed":
        scored = _build_fci(df.copy())
        scored = scored.rename(
            columns={
                "foul_commitment_index": "market_score",
                "confidence_label": "market_confidence",
            }
        )
        scored["market"] = "fouls_committed"
    else:
        raise ValueError(f"Unsupported market: {market}")
    return scored


def _build_fixture_gate(
    base: pd.DataFrame,
    goal_overlay_column: str = "",
    min_fixture_quality_score: float = 0.0,
) -> pd.DataFrame:
    team_level = (
        base[
            [
                "fixture_key",
                "player_team_side",
                "team_avg_fouls",
                "team_avg_yellows",
                "opponent_possession_projection",
                "match_stakes_score",
                "ref_cards_per_match",
                "ref_foul_to_card_ratio",
                "ref_dissent_strictness",
                "ref_timewasting_strictness",
                "starting_xi_team_quality_score",
                "starting_xi_attack_quality_score",
                "starting_xi_defensive_quality_score",
                "starting_xi_quality_edge",
                "player_quality_score_l5",
            ]
        ]
        .drop_duplicates(subset=["fixture_key", "player_team_side"])
        .copy()
    )
    fixture = (
        team_level.groupby("fixture_key", as_index=False)
        .agg(
            team_avg_fouls_sum=("team_avg_fouls", "sum"),
            team_avg_yellows_sum=("team_avg_yellows", "sum"),
            opp_possession_max=("opponent_possession_projection", "max"),
            match_stakes_score=("match_stakes_score", "max"),
            ref_cards_per_match=("ref_cards_per_match", "max"),
            ref_foul_to_card_ratio=("ref_foul_to_card_ratio", "max"),
            ref_dissent_strictness=("ref_dissent_strictness", "max"),
            ref_timewasting_strictness=("ref_timewasting_strictness", "max"),
            starting_xi_team_quality_avg=("starting_xi_team_quality_score", "mean"),
            starting_xi_attack_quality_avg=("starting_xi_attack_quality_score", "mean"),
            starting_xi_defensive_quality_avg=("starting_xi_defensive_quality_score", "mean"),
            starting_xi_quality_edge_max=("starting_xi_quality_edge", "max"),
            starting_xi_quality_edge_min=("starting_xi_quality_edge", "min"),
            player_quality_score_avg=("player_quality_score_l5", "mean"),
        )
    )
    fixture["fixture_ref_score"] = (
        0.35 * _norm_cap(fixture["ref_cards_per_match"], 6.0)
        + 0.20 * _norm_inverse_ratio(fixture["ref_foul_to_card_ratio"], lo=4.0, hi=10.0)
        + 0.25 * _norm_cap(fixture["ref_dissent_strictness"], 1.0)
        + 0.20 * _norm_cap(fixture["ref_timewasting_strictness"], 1.0)
    )
    fixture["fixture_team_pressure_score"] = (
        0.55 * _norm_cap(fixture["team_avg_fouls_sum"], 30.0)
        + 0.45 * _norm_cap(fixture["team_avg_yellows_sum"], 5.0)
    )
    fixture["fixture_stakes_score"] = _norm_cap(fixture["match_stakes_score"], 5.0)
    fixture["fixture_possession_stress_score"] = _norm_cap(fixture["opp_possession_max"], 70.0)
    fixture["fixture_lineup_quality_score"] = (
        0.30 * _norm_cap(fixture["starting_xi_team_quality_avg"], 90.0)
        + 0.20 * _norm_cap(fixture["starting_xi_attack_quality_avg"], 90.0)
        + 0.20 * _norm_cap(fixture["starting_xi_defensive_quality_avg"], 90.0)
        + 0.15 * _norm_cap(fixture["player_quality_score_avg"], 100.0)
        + 0.15 * _norm_cap((fixture["starting_xi_quality_edge_max"] - fixture["starting_xi_quality_edge_min"]).abs(), 20.0)
    )
    fixture["fixture_foul_density_score"] = 0.0
    fixture["fixture_tackle_density_score"] = 0.0
    fixture["fixture_midfield_grind_score"] = 0.0
    fixture["fixture_wide_duel_score"] = 0.0
    fixture["fixture_style_label"] = "UNSET"
    style_cols = [
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "fixture_wide_duel_score",
        "fixture_style_label",
    ]
    style_overlay = (
        base[["fixture_key"] + [c for c in style_cols if c in base.columns]]
        .drop_duplicates(subset=["fixture_key"])
        .copy()
    )
    if len(style_overlay.columns) > 1:
        fixture = fixture.merge(style_overlay, on="fixture_key", how="left", suffixes=("", "_new"))
        for c in style_cols[:-1]:
            new_col = f"{c}_new"
            if new_col in fixture.columns:
                fixture[c] = pd.to_numeric(fixture[new_col], errors="coerce").fillna(fixture[c])
                fixture = fixture.drop(columns=[new_col])
        if "fixture_style_label_new" in fixture.columns:
            fixture["fixture_style_label"] = (
                fixture["fixture_style_label_new"].astype("string").fillna(fixture["fixture_style_label"])
            )
            fixture = fixture.drop(columns=["fixture_style_label_new"])
    fixture["fixture_attack_pressure_score"] = 0.0
    fixture["fixture_corner_pressure_score"] = 0.0
    fixture["fixture_territorial_stress_score"] = 0.0
    fixture["fixture_attacking_style_label"] = "UNSET"
    attack_cols = [
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "fixture_territorial_stress_score",
        "fixture_attacking_style_label",
    ]
    attack_overlay = (
        base[["fixture_key"] + [c for c in attack_cols if c in base.columns]]
        .drop_duplicates(subset=["fixture_key"])
        .copy()
    )
    if len(attack_overlay.columns) > 1:
        fixture = fixture.merge(attack_overlay, on="fixture_key", how="left", suffixes=("", "_new"))
        for c in attack_cols[:-1]:
            new_col = f"{c}_new"
            if new_col in fixture.columns:
                fixture[c] = pd.to_numeric(fixture[new_col], errors="coerce").fillna(fixture[c])
                fixture = fixture.drop(columns=[new_col])
        if "fixture_attacking_style_label_new" in fixture.columns:
            fixture["fixture_attacking_style_label"] = (
                fixture["fixture_attacking_style_label_new"].astype("string").fillna(fixture["fixture_attacking_style_label"])
            )
            fixture = fixture.drop(columns=["fixture_attacking_style_label_new"])
    fixture["goal_environment_overlay_score"] = 0.0
    fixture["strong_goal_environment_support_flag"] = 0
    if goal_overlay_column and goal_overlay_column in base.columns:
        goal_overlay = (
            base[["fixture_key", goal_overlay_column]]
            .drop_duplicates(subset=["fixture_key"])
            .rename(columns={goal_overlay_column: "goal_environment_overlay_score"})
        )
        fixture = fixture.merge(goal_overlay, on="fixture_key", how="left", suffixes=("", "_new"))
        if "goal_environment_overlay_score_new" in fixture.columns:
            fixture["goal_environment_overlay_score"] = pd.to_numeric(
                fixture["goal_environment_overlay_score_new"], errors="coerce"
            ).fillna(fixture["goal_environment_overlay_score"])
            fixture = fixture.drop(columns=["goal_environment_overlay_score_new"])
        fixture["strong_goal_environment_support_flag"] = fixture["goal_environment_overlay_score"].ge(0.60).astype(int)

    fixture["fixture_quality_score"] = (
        0.25 * fixture["fixture_ref_score"]
        + 0.25 * fixture["fixture_team_pressure_score"]
        + 0.12 * fixture["fixture_stakes_score"]
        + 0.08 * fixture["fixture_possession_stress_score"]
        + 0.15 * _norm_cap(fixture["goal_environment_overlay_score"], 1.0)
        + 0.08 * _norm_cap(fixture["fixture_foul_density_score"], 1.0)
        + 0.04 * _norm_cap(fixture["fixture_tackle_density_score"], 1.0)
        + 0.03 * _norm_cap(fixture["fixture_midfield_grind_score"], 1.0)
        + 0.08 * _norm_cap(fixture["fixture_attack_pressure_score"], 1.0)
        + 0.05 * _norm_cap(fixture["fixture_corner_pressure_score"], 1.0)
        + 0.05 * _norm_cap(fixture["fixture_territorial_stress_score"], 1.0)
        + 0.10 * fixture["fixture_lineup_quality_score"]
    )
    fixture["strong_ref_support_flag"] = fixture["fixture_ref_score"].ge(0.60).astype(int)
    fixture["strong_team_pressure_support_flag"] = fixture["fixture_team_pressure_score"].ge(0.58).astype(int)
    fixture["strong_contact_style_support_flag"] = (
        fixture["fixture_foul_density_score"].ge(0.62)
        | fixture["fixture_tackle_density_score"].ge(0.62)
        | fixture["fixture_midfield_grind_score"].ge(0.64)
        | fixture["fixture_wide_duel_score"].ge(0.66)
        | fixture["fixture_style_label"].isin(["AGGRESSIVE_BOTH", "MIDFIELD_GRIND", "WIDE_DUEL_GAME"])
    ).astype(int)
    fixture["strong_attack_support_flag"] = (
        fixture["fixture_attack_pressure_score"].ge(0.66)
        | fixture["fixture_corner_pressure_score"].ge(0.66)
        | fixture["fixture_territorial_stress_score"].ge(0.66)
        | fixture["fixture_attacking_style_label"].isin(["ATTACK_WAVE", "CORNER_SIEGE", "TERRITORY_TILT"])
    ).astype(int)
    fixture["strong_lineup_quality_support_flag"] = fixture["fixture_lineup_quality_score"].ge(0.66).astype(int)
    fixture["fixture_quality_pass_flag"] = fixture["fixture_quality_score"].ge(min_fixture_quality_score).astype(int)

    reasons = []
    for row in fixture.itertuples(index=False):
        toks = []
        if row.fixture_ref_score >= 0.60:
            toks.append("STRICT_REF_SUPPORT")
        if row.fixture_team_pressure_score >= 0.58:
            toks.append("TEAM_PRESSURE_SUPPORT")
        if row.fixture_stakes_score >= 0.60:
            toks.append("MATCH_STAKES_SUPPORT")
        if row.fixture_possession_stress_score >= 0.70:
            toks.append("POSSESSION_STRESS_SUPPORT")
        if float(getattr(row, "fixture_foul_density_score", 0.0)) >= 0.62:
            toks.append("FOUL_DENSITY_SUPPORT")
        if float(getattr(row, "fixture_tackle_density_score", 0.0)) >= 0.62:
            toks.append("TACKLE_DENSITY_SUPPORT")
        if str(getattr(row, "fixture_style_label", "") or "") in {"AGGRESSIVE_BOTH", "MIDFIELD_GRIND", "WIDE_DUEL_GAME", "ONE_SIDED_PRESSURE"}:
            toks.append(f"STYLE_{str(getattr(row, 'fixture_style_label', '')).upper()}")
        if float(getattr(row, "fixture_attack_pressure_score", 0.0)) >= 0.66:
            toks.append("ATTACK_PRESSURE_SUPPORT")
        if float(getattr(row, "fixture_corner_pressure_score", 0.0)) >= 0.66:
            toks.append("CORNER_PRESSURE_SUPPORT")
        if float(getattr(row, "fixture_territorial_stress_score", 0.0)) >= 0.66:
            toks.append("TERRITORY_STRESS_SUPPORT")
        if str(getattr(row, "fixture_attacking_style_label", "") or "") in {"ATTACK_WAVE", "CORNER_SIEGE", "TERRITORY_TILT"}:
            toks.append(f"ATTACK_STYLE_{str(getattr(row, 'fixture_attacking_style_label', '')).upper()}")
        if float(getattr(row, "fixture_lineup_quality_score", 0.0)) >= 0.66:
            toks.append("LINEUP_QUALITY_SUPPORT")
        if getattr(row, "goal_environment_overlay_score", 0.0) and float(getattr(row, "goal_environment_overlay_score", 0.0)) >= 0.60:
            toks.append("GOAL_ENV_SUPPORT")
        if not toks:
            toks.append("LOW_CONTEXT")
        reasons.append("|".join(toks))
    fixture["fixture_quality_reason_codes"] = reasons
    return fixture


def build_combined_board(
    input_csv: str,
    output_csv: str,
    output_md: str,
    yc_min_score: float = 44.0,
    foul_min_score: float = 56.0,
    max_yc_per_fixture: int = 2,
    max_fouls_per_fixture: int = 2,
    leagues_filter: list[str] | None = None,
    min_fixture_quality_score: float = 0.0,
    goal_overlay_column: str = "",
    require_any_support: bool = False,
    require_goal_env_support: bool = False,
    dual_trigger_only: bool = False,
    allowed_style_labels: list[str] | None = None,
    require_attack_support: bool = False,
    allowed_attacking_style_labels: list[str] | None = None,
    allowed_style_attack_combos: list[str] | None = None,
    require_underdog_fullback_wide_overload: bool = False,
    require_underdog_dm_midfield_grind: bool = False,
    min_negative_quality_edge: float = 0.0,
    require_strong_central_battle: bool = False,
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

    yc = _prep_market(base, "yellow_card")
    fouls = _prep_market(base, "fouls_committed")
    overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in yc.columns]
    if overlap_cols:
        yc = yc.drop(columns=overlap_cols)
    overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in fouls.columns]
    if overlap_cols:
        fouls = fouls.drop(columns=overlap_cols)
    yc = yc.merge(fixture_gate[gate_cols], on="fixture_key", how="left")
    fouls = fouls.merge(fixture_gate[gate_cols], on="fixture_key", how="left")

    yc = yc[yc["market_score"].ge(yc_min_score)].copy()
    fouls = fouls[fouls["market_score"].ge(foul_min_score)].copy()
    yc = yc[yc["fixture_quality_pass_flag"].eq(1)].copy()
    fouls = fouls[fouls["fixture_quality_pass_flag"].eq(1)].copy()
    if require_any_support:
        support_mask_yc = (
            yc["strong_ref_support_flag"].eq(1)
            | yc["strong_team_pressure_support_flag"].eq(1)
            | yc["strong_goal_environment_support_flag"].eq(1)
            | yc["strong_contact_style_support_flag"].eq(1)
            | yc["strong_attack_support_flag"].eq(1)
            | yc["strong_lineup_quality_support_flag"].eq(1)
        )
        support_mask_f = (
            fouls["strong_ref_support_flag"].eq(1)
            | fouls["strong_team_pressure_support_flag"].eq(1)
            | fouls["strong_goal_environment_support_flag"].eq(1)
            | fouls["strong_contact_style_support_flag"].eq(1)
            | fouls["strong_attack_support_flag"].eq(1)
            | fouls["strong_lineup_quality_support_flag"].eq(1)
        )
        yc = yc[support_mask_yc].copy()
        fouls = fouls[support_mask_f].copy()
    if require_goal_env_support:
        yc = yc[yc["strong_goal_environment_support_flag"].eq(1)].copy()
        fouls = fouls[fouls["strong_goal_environment_support_flag"].eq(1)].copy()
    if require_attack_support:
        yc = yc[yc["strong_attack_support_flag"].eq(1)].copy()
        fouls = fouls[fouls["strong_attack_support_flag"].eq(1)].copy()
    if allowed_style_labels:
        allowed = {x.strip().upper() for x in allowed_style_labels if x.strip()}
        yc = yc[yc["fixture_style_label"].astype("string").str.upper().isin(allowed)].copy()
        fouls = fouls[fouls["fixture_style_label"].astype("string").str.upper().isin(allowed)].copy()
    if allowed_attacking_style_labels:
        allowed = {x.strip().upper() for x in allowed_attacking_style_labels if x.strip()}
        yc = yc[yc["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()
        fouls = fouls[fouls["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()
    if allowed_style_attack_combos:
        allowed = {x.strip().upper() for x in allowed_style_attack_combos if x.strip()}
        yc_combo = (
            yc["fixture_style_label"].astype("string").str.upper()
            + " + "
            + yc["fixture_attacking_style_label"].astype("string").str.upper()
        )
        fouls_combo = (
            fouls["fixture_style_label"].astype("string").str.upper()
            + " + "
            + fouls["fixture_attacking_style_label"].astype("string").str.upper()
        )
        yc = yc[yc_combo.isin(allowed)].copy()
        fouls = fouls[fouls_combo.isin(allowed)].copy()
    if require_underdog_fullback_wide_overload:
        yc = yc[yc["underdog_fullback_wide_overload_flag"].eq(1)].copy()
        fouls = fouls[fouls["underdog_fullback_wide_overload_flag"].eq(1)].copy()
    if require_underdog_dm_midfield_grind:
        yc = yc[yc["underdog_dm_midfield_grind_flag"].eq(1)].copy()
        fouls = fouls[fouls["underdog_dm_midfield_grind_flag"].eq(1)].copy()
    if min_negative_quality_edge > 0:
        yc = yc[pd.to_numeric(yc["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-float(min_negative_quality_edge))].copy()
        fouls = fouls[pd.to_numeric(fouls["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-float(min_negative_quality_edge))].copy()
    if require_strong_central_battle:
        yc = yc[
            yc["central_battle_flag"].eq(1)
            & pd.to_numeric(yc["fixture_midfield_grind_score"], errors="coerce").fillna(0.0).ge(0.62)
        ].copy()
        fouls = fouls[
            fouls["central_battle_flag"].eq(1)
            & pd.to_numeric(fouls["fixture_midfield_grind_score"], errors="coerce").fillna(0.0).ge(0.62)
        ].copy()

    yc = yc.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_yc_per_fixture)
    fouls = fouls.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_fouls_per_fixture)
    yc["combo_reason_bucket"] = ""
    fouls["combo_reason_bucket"] = ""

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
        "yellow_cards_per90",
        "tackles_per90",
        "team_avg_fouls",
        "team_avg_yellows",
        "opponent_possession_projection",
        "referee_name",
        "ref_cards_per_match",
        "ref_foul_to_card_ratio",
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
        "og_battle_on_score",
        "og_goal_environment_label",
        "player_quality_score_l5",
        "player_form_tier",
        "player_quality_rank_in_position",
        "starting_xi_team_quality_score",
        "starting_xi_attack_quality_score",
        "starting_xi_defensive_quality_score",
        "starting_xi_quality_edge",
        "market",
        "market_score",
        "market_confidence",
        "combo_reason_bucket",
        "analyst_notes",
    ]
    out = pd.concat([yc[keep_cols], fouls[keep_cols]], ignore_index=True)
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
        out["elite_single_trigger_flag"] = (
            out["same_player_dual_trigger_flag"].eq(0)
            & (
                ((out["market"] == "yellow_card") & out["market_score"].ge(yc_min_score + 8.0))
                | ((out["market"] == "fouls_committed") & out["market_score"].ge(foul_min_score + 8.0))
            )
        ).astype(int)
    else:
        out["trigger_count"] = pd.Series(dtype=int)
        out["trigger_markets"] = pd.Series(dtype="string")
        out["same_player_dual_trigger_flag"] = pd.Series(dtype=int)
        out["elite_single_trigger_flag"] = pd.Series(dtype=int)
    if dual_trigger_only:
        out = out[out["same_player_dual_trigger_flag"].eq(1)].copy()
    if not out.empty:
        out["combo_reason_bucket"] = (
            out["fixture_style_label"].astype("string").str.upper()
            + "__"
            + out["fixture_attacking_style_label"].astype("string").str.upper()
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
        "# Combined Player Events Board",
        "",
        "## Rules",
        f"- max yellow-card names per fixture: {max_yc_per_fixture}",
        f"- max fouls names per fixture: {max_fouls_per_fixture}",
        f"- yellow-card minimum score: {yc_min_score}",
        f"- fouls minimum score: {foul_min_score}",
        f"- minimum fixture quality score: {min_fixture_quality_score}",
        f"- goal overlay column: {goal_overlay_column or 'not applied'}",
        f"- require strong ref/team-pressure/goal-env support: {require_any_support}",
        f"- require goal-environment support specifically: {require_goal_env_support}",
        f"- require attack support specifically: {require_attack_support}",
        f"- dual-trigger only: {dual_trigger_only}",
        f"- allowed style labels: {', '.join(allowed_style_labels) if allowed_style_labels else 'all'}",
        f"- allowed attacking style labels: {', '.join(allowed_attacking_style_labels) if allowed_attacking_style_labels else 'all'}",
        f"- allowed contact+attack combos: {', '.join(allowed_style_attack_combos) if allowed_style_attack_combos else 'all'}",
        f"- require underdog full-back + wide overload: {require_underdog_fullback_wide_overload}",
        f"- require underdog DM + midfield grind: {require_underdog_dm_midfield_grind}",
        "",
        "## Board Summary",
        f"- rows: {len(out)}",
        f"- fixtures covered: {out['fixture_key'].nunique() if not out.empty else 0}",
        f"- yellow-card rows: {int((out['market'] == 'yellow_card').sum()) if not out.empty else 0}",
        f"- fouls rows: {int((out['market'] == 'fouls_committed').sum()) if not out.empty else 0}",
        f"- average fixture quality score: {round(float(out['fixture_quality_score'].mean()), 3) if not out.empty else 0.0}",
        f"- dual-trigger rows: {int(out['same_player_dual_trigger_flag'].sum()) if not out.empty else 0}",
        f"- elite single-trigger rows: {int(out['elite_single_trigger_flag'].sum()) if not out.empty else 0}",
        "",
        "## Fixture Samples",
    ]
    sample_cols = ["fixture_key", "market", "player_name", "team_name", "tactical_role", "market_score", "market_confidence"]
    for fixture_key, group in out.groupby("fixture_key", sort=False):
        lines.append(f"### {fixture_key}")
        fx = group.iloc[0]
        lines.append(
            f"- fixture quality: {fx['fixture_quality_score']:.3f} | battle_on: {fx['og_battle_on_score']:.3f} | goal_env: {fx['og_goal_environment_label']} | contact: {fx['fixture_style_label']} | attack: {fx['fixture_attacking_style_label']} | formation: {fx['formation_matchup_label']} | reasons: {fx['fixture_quality_reason_codes']}"
        )
        for row in group[sample_cols].itertuples(index=False):
            row0 = group[group["player_name"].eq(row.player_name)].iloc[0]
            flag_bits = []
            if int(row0["same_player_dual_trigger_flag"]) == 1:
                flag_bits.append(f"dual:{row0['trigger_markets']}")
            if int(row0["elite_single_trigger_flag"]) == 1:
                flag_bits.append("elite-single")
            extra = f" | {'; '.join(flag_bits)}" if flag_bits else ""
            lines.append(
                f"- `{row.market}`: {row.player_name} ({row.team_name}) | {row.tactical_role} | score={row.market_score:.1f} | {row.market_confidence}{extra}"
            )
        lines.append("")
        if len(lines) > 120:
            break

    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined same-fixture yellow-card + fouls beta board.")
    parser.add_argument("--input", required=True, help="player_events_fixture_input csv")
    parser.add_argument("--output-csv", required=True, help="combined board csv path")
    parser.add_argument("--output-md", required=True, help="combined board markdown path")
    parser.add_argument("--yc-min-score", type=float, default=44.0)
    parser.add_argument("--foul-min-score", type=float, default=56.0)
    parser.add_argument("--max-yc-per-fixture", type=int, default=2)
    parser.add_argument("--max-fouls-per-fixture", type=int, default=2)
    parser.add_argument("--leagues", default="", help="Comma-separated league names to include")
    parser.add_argument("--min-fixture-quality-score", type=float, default=0.0)
    parser.add_argument("--goal-overlay-column", default="", help="Optional fixture-level goal environment score column already present in the input")
    parser.add_argument("--require-any-support", action="store_true", help="Require at least one of strong referee, team-pressure, or goal-environment support")
    parser.add_argument("--require-goal-env-support", action="store_true", help="Require strong goal-environment support specifically")
    parser.add_argument("--dual-trigger-only", action="store_true", help="Keep only players that qualify in both YC and fouls lanes")
    parser.add_argument("--allowed-style-labels", default="", help="Comma-separated fixture style labels to keep, e.g. AGGRESSIVE_BOTH,MIDFIELD_GRIND")
    parser.add_argument("--require-attack-support", action="store_true", help="Require strong attacking-style support specifically")
    parser.add_argument("--allowed-attacking-style-labels", default="", help="Comma-separated attacking style labels to keep, e.g. ATTACK_WAVE,CORNER_SIEGE")
    parser.add_argument("--allowed-style-attack-combos", default="", help="Comma-separated contact+attack combos to keep, e.g. AGGRESSIVE_BOTH + ATTACK_WAVE")
    parser.add_argument("--require-underdog-fullback-wide-overload", action="store_true")
    parser.add_argument("--require-underdog-dm-midfield-grind", action="store_true")
    parser.add_argument("--min-negative-quality-edge", type=float, default=0.0)
    parser.add_argument("--require-strong-central-battle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    allowed_style_labels = [x.strip() for x in args.allowed_style_labels.split(",") if x.strip()]
    allowed_attacking_style_labels = [x.strip() for x in args.allowed_attacking_style_labels.split(",") if x.strip()]
    allowed_style_attack_combos = [x.strip() for x in args.allowed_style_attack_combos.split(",") if x.strip()]
    df = build_combined_board(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        yc_min_score=args.yc_min_score,
        foul_min_score=args.foul_min_score,
        max_yc_per_fixture=args.max_yc_per_fixture,
        max_fouls_per_fixture=args.max_fouls_per_fixture,
        leagues_filter=leagues or None,
        min_fixture_quality_score=args.min_fixture_quality_score,
        goal_overlay_column=args.goal_overlay_column,
        require_any_support=args.require_any_support,
        require_goal_env_support=args.require_goal_env_support,
        dual_trigger_only=args.dual_trigger_only,
        allowed_style_labels=allowed_style_labels or None,
        require_attack_support=args.require_attack_support,
        allowed_attacking_style_labels=allowed_attacking_style_labels or None,
        allowed_style_attack_combos=allowed_style_attack_combos or None,
        require_underdog_fullback_wide_overload=args.require_underdog_fullback_wide_overload,
        require_underdog_dm_midfield_grind=args.require_underdog_dm_midfield_grind,
        min_negative_quality_edge=args.min_negative_quality_edge,
        require_strong_central_battle=args.require_strong_central_battle,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")


if __name__ == "__main__":
    main()
