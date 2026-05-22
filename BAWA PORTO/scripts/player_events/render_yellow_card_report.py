from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _scalar(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() == "nan":
        return fallback
    return text


def _clamp_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if not isinstance(s, pd.Series):
        value = pd.to_numeric(pd.Series([s]), errors="coerce").fillna(0.0).astype(float).iloc[0]
        return float(min(max(value, lo), hi))
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return out.clip(lower=lo, upper=hi)


def _norm_cap(s: pd.Series, cap: float) -> pd.Series:
    if not isinstance(s, pd.Series):
        value = pd.to_numeric(pd.Series([s]), errors="coerce").fillna(0.0).astype(float).iloc[0]
        value = min(max(float(value), 0.0), cap)
        return value / cap if cap else 0.0
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _norm_inverse_ratio(s: pd.Series, lo: float = 1.0, hi: float = 6.0) -> pd.Series:
    if not isinstance(s, pd.Series):
        value = pd.to_numeric(pd.Series([s]), errors="coerce").fillna(hi).astype(float).iloc[0]
        value = min(max(float(value), lo), hi)
        return (hi - value) / (hi - lo)
    out = pd.to_numeric(s, errors="coerce").astype(float)
    out = out.clip(lower=lo, upper=hi)
    return ((hi - out) / (hi - lo)).fillna(0.0)


def _string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _num(value: object, fallback: float = 0.0) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(fallback).iloc[0]
    return float(out)


def _league_role_bonus(df: pd.DataFrame) -> pd.Series:
    league = _string(df.get("league", pd.Series("", index=df.index))).str.lower()
    role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
    out = pd.Series(0.0, index=df.index, dtype=float)

    elite_scope = league.isin(["serie a", "la liga", "uefa europa league"])
    out += ((role.eq("holding midfielder")) & elite_scope).astype(float) * 0.08
    out += ((role.eq("wide defender / wing-back")) & elite_scope).astype(float) * 0.08
    out += ((role.eq("centre-back enforcer")) & elite_scope).astype(float) * 0.04
    out -= ((role.eq("central striker")) & elite_scope).astype(float) * 0.07
    out -= ((role.eq("goalkeeper")) & elite_scope).astype(float) * 0.10
    return out


def _fixture_style_bonus(df: pd.DataFrame) -> pd.Series:
    label = _string(df.get("fixture_style_label", pd.Series("", index=df.index))).str.upper()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += label.eq("AGGRESSIVE_BOTH").astype(float) * 0.08
    out += label.eq("MIDFIELD_GRIND").astype(float) * 0.07
    out += label.eq("WIDE_DUEL_GAME").astype(float) * 0.04
    out += label.eq("ONE_SIDED_PRESSURE").astype(float) * 0.02
    out += 0.06 * _norm_cap(df.get("fixture_foul_density_score", 0.0), 1.0)
    out += 0.04 * _norm_cap(df.get("fixture_tackle_density_score", 0.0), 1.0)
    out += 0.04 * _norm_cap(df.get("fixture_midfield_grind_score", 0.0), 1.0)
    out += 0.03 * _norm_cap(df.get("fixture_wide_duel_score", 0.0), 1.0)
    return out


def _quality_risk_bonus(df: pd.DataFrame) -> pd.Series:
    quality_score = _norm_cap(df.get("player_quality_score_l5", 0.0), 100.0)
    quality_pct = _norm_cap(df.get("player_quality_percentile_in_position", 0.0), 1.0)
    pass_accuracy = _norm_cap(df.get("pass_accuracy_pct_l5", 0.0), 100.0)
    quality_edge = pd.to_numeric(_optional_series(df, "starting_xi_quality_edge"), errors="coerce").fillna(0.0).astype(float)
    negative_edge = ((-quality_edge).clip(lower=0.0, upper=20.0) / 20.0).fillna(0.0)
    opponent_quality = _norm_cap(df.get("opponent_starting_xi_team_quality_score", 0.0), 90.0)
    return (
        0.35 * (1.0 - quality_score)
        + 0.20 * (1.0 - quality_pct)
        + 0.15 * (1.0 - pass_accuracy)
        + 0.20 * negative_edge
        + 0.10 * opponent_quality
    )


def _formation_mismatch_bonus(df: pd.DataFrame) -> pd.Series:
    lineup_card_delta = pd.to_numeric(_optional_series(df, "lineup_xi_card_risk_delta"), errors="coerce").fillna(0.0)
    formation_mismatch = pd.to_numeric(_optional_series(df, "formation_mismatch_flag"), errors="coerce").fillna(0.0)
    return (
        0.30 * _norm_cap(df.get("formation_pressure_score", 0.0), 1.0)
        + 0.22 * _norm_cap(df.get("formation_wide_overload_flag", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("formation_midfield_grind_flag", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("underdog_fullback_wide_overload_flag", 0.0), 1.0)
        + 0.12 * _norm_cap(df.get("underdog_dm_midfield_grind_flag", 0.0), 1.0)
        + 0.12 * _norm_cap((-lineup_card_delta).clip(lower=0.0, upper=2.0), 2.0)
        + 0.08 * _norm_cap(formation_mismatch, 1.0)
    )


def _optional_series(df: pd.DataFrame, name: str, fallback: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(fallback, index=df.index)


def _non_contact_booking_components(df: pd.DataFrame) -> pd.DataFrame:
    """Score card routes that do not require a foul/contact cascade."""
    out = pd.DataFrame(index=df.index)
    out["celebration"] = _norm_cap(_optional_series(df, "celebration_card_risk"), 1.0)
    out["dissent"] = (
        0.55 * _norm_cap(_optional_series(df, "dissent_card_risk"), 1.0)
        + 0.45 * _norm_cap(_optional_series(df, "ref_dissent_strictness"), 5.0)
    )
    out["delay_time_wasting"] = (
        0.45 * _norm_cap(_optional_series(df, "delay_restart_card_risk"), 1.0)
        + 0.35 * _norm_cap(_optional_series(df, "time_wasting_card_risk"), 1.0)
        + 0.20 * _norm_cap(_optional_series(df, "ref_timewasting_strictness"), 5.0)
    )
    out["keeper_dissent"] = _norm_cap(_optional_series(df, "keeper_dissent_card_risk"), 1.0)
    out["mass_confrontation"] = _norm_cap(_optional_series(df, "mass_confrontation_card_risk"), 1.0)
    out["late_added_time"] = (
        0.35 * _norm_cap(_optional_series(df, "added_time_card_risk"), 1.0)
        + 0.25 * _norm_cap(_optional_series(df, "late_game_pressure_risk"), 1.0)
        + 0.20 * _norm_cap(_optional_series(df, "late_card_risk_flag"), 1.0)
        + 0.20 * _norm_cap(_optional_series(df, "added_time_minutes"), 10.0)
    )
    out["frustration"] = _norm_cap(_optional_series(df, "trailing_frustration_risk"), 1.0)
    out["explicit_non_contact"] = _norm_cap(_optional_series(df, "non_contact_card_risk"), 1.0)
    return out.fillna(0.0).clip(lower=0.0, upper=1.0)


def _dominant_non_contact_bucket(components: pd.DataFrame, score: pd.Series) -> pd.Series:
    labels = components.idxmax(axis=1).astype("string")
    labels = labels.where(score.ge(0.12), "none")
    return labels.fillna("none")


def _build_bpi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fixture_context = (
        0.6 * _norm_cap(out.get("match_stakes_score", 0.0), 5.0)
        + 0.4 * _norm_cap(out.get("rivalry_flag", 0.0), 1.0)
    )
    player_history = (
        0.45 * _norm_cap(out.get("fouls_per90", 0.0), 4.0)
        + 0.35 * _norm_cap(out.get("yellow_cards_per90", 0.0), 0.5)
        + 0.20 * _norm_inverse_ratio(out.get("booking_efficiency", 6.0))
    )
    player_performance = (
        0.35 * _norm_cap(out.get("tackles_per90", 0.0), 4.0)
        + 0.20 * _norm_cap(out.get("interceptions_per90", 0.0), 3.0)
        + 0.25 * _norm_cap(out.get("dribbles_faced_per90", 0.0), 6.0)
        + 0.20 * _norm_cap(out.get("ground_duel_loss_rate", 0.0), 1.0)
    )
    referee_profile = (
        0.55 * _norm_cap(out.get("ref_cards_per_match", 0.0), 6.0)
        + 0.45 * _norm_inverse_ratio(out.get("ref_foul_to_card_ratio", 25.0), lo=4.0, hi=25.0)
    )
    tactical = (
        0.25 * _norm_cap(out.get("left_flank_dominance", 0.0), 1.0)
        + 0.25 * _norm_cap(out.get("right_flank_dominance", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("central_attack_dominance", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("counterattack_defender_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("central_battle_flag", 0.0), 1.0)
    )
    directional_pressure = (
        0.38 * _norm_cap(out.get("power_gap_directional_pressure_score", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("weaker_side_under_pressure_flag", 0.0), 1.0)
        + 0.17 * _norm_cap(out.get("weak_flank_overload_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("weak_midfield_overload_flag", 0.0), 1.0)
        + 0.10 * _norm_cap(out.get("weak_territory_protection_flag", 0.0), 1.0)
    )
    market = _norm_inverse_ratio(out.get("book_yellow_odds", 10.0), lo=1.5, hi=10.0)
    psychology = (
        0.45 * _norm_cap(out.get("temperament_flag", 0.0), 1.0)
        + 0.30 * _norm_cap(out.get("late_game_pressure_risk", 0.0), 1.0)
        + 0.25 * _norm_cap(out.get("trailing_frustration_risk", 0.0), 1.0)
    )
    environment = (
        0.55 * (1.0 - _norm_cap(out.get("days_rest", 7.0), 7.0))
        + 0.45 * _norm_cap(out.get("recent_injury_return_flag", 0.0), 1.0)
    )
    team_dynamics = (
        0.50 * _norm_cap(out.get("team_avg_fouls", 0.0), 18.0)
        + 0.50 * _norm_cap(out.get("team_avg_yellows", 0.0), 3.5)
    )

    out["bpi_fixture_context"] = fixture_context
    out["bpi_player_history"] = player_history
    out["bpi_player_performance"] = player_performance
    out["bpi_referee_profile"] = referee_profile
    out["bpi_tactical"] = tactical
    out["bpi_directional_pressure"] = directional_pressure
    out["bpi_market"] = market
    out["bpi_psychology"] = psychology
    out["bpi_environment"] = environment
    out["bpi_team_dynamics"] = team_dynamics
    out["bpi_role_bonus"] = _league_role_bonus(out)
    out["bpi_fixture_style_bonus"] = _fixture_style_bonus(out)
    out["bpi_quality_risk"] = _quality_risk_bonus(out)
    out["bpi_formation_mismatch"] = _formation_mismatch_bonus(out)
    non_contact_components = _non_contact_booking_components(out)
    out["bpi_non_contact_card_layer"] = (
        0.16 * non_contact_components["celebration"]
        + 0.16 * non_contact_components["dissent"]
        + 0.14 * non_contact_components["delay_time_wasting"]
        + 0.10 * non_contact_components["keeper_dissent"]
        + 0.12 * non_contact_components["mass_confrontation"]
        + 0.14 * non_contact_components["late_added_time"]
        + 0.10 * non_contact_components["frustration"]
        + 0.08 * non_contact_components["explicit_non_contact"]
    ).clip(lower=0.0, upper=1.0)
    out["non_contact_card_bucket"] = _dominant_non_contact_bucket(
        non_contact_components,
        out["bpi_non_contact_card_layer"],
    )
    out["bpi_contact_cascade_layer"] = (
        0.25 * player_history
        + 0.20 * player_performance
        + 0.15 * referee_profile
        + 0.14 * tactical
        + 0.14 * directional_pressure
        + 0.12 * psychology
    ).clip(lower=0.0, upper=1.0)
    out["booking_cascade_type"] = "CONTACT_CASCADE"
    out.loc[
        out["bpi_non_contact_card_layer"].ge(0.20) & out["bpi_contact_cascade_layer"].lt(0.45),
        "booking_cascade_type",
    ] = "NON_CONTACT_LAYER"
    out.loc[
        out["bpi_non_contact_card_layer"].ge(0.20) & out["bpi_contact_cascade_layer"].ge(0.45),
        "booking_cascade_type",
    ] = "HYBRID_CONTACT_NON_CONTACT"

    out["booking_probability_index"] = (
        0.15 * fixture_context
        + 0.15 * player_history
        + 0.15 * player_performance
        + 0.15 * referee_profile
        + 0.07 * tactical
        + 0.08 * directional_pressure
        + 0.10 * market
        + 0.10 * psychology
        + 0.05 * environment
        + 0.05 * team_dynamics
        + out["bpi_role_bonus"]
        + out["bpi_fixture_style_bonus"]
        + 0.08 * out["bpi_quality_risk"]
        + 0.08 * out["bpi_formation_mismatch"]
        + 0.06 * out["bpi_non_contact_card_layer"]
    ) * 100.0

    score = out["booking_probability_index"].fillna(0.0)
    out["confidence_label"] = pd.cut(
        score,
        bins=[-1, 42, 62, 200],
        labels=["Low", "Moderate", "High"],
    ).astype("string")

    return out


def _pick_fixture(df: pd.DataFrame, fixture_key: str) -> pd.DataFrame:
    out = df.copy()
    if fixture_key:
        matched = out[_string(out["fixture_key"]).eq(fixture_key)]
        if matched.empty:
            raise SystemExit(f"Fixture key not found: {fixture_key}")
        return matched

    keys = _string(out["fixture_key"]).drop_duplicates().tolist()
    if len(keys) != 1:
        raise SystemExit(
            "Input contains multiple fixtures. Supply --fixture-key to render a single report."
        )
    return out[_string(out["fixture_key"]).eq(keys[0])]


def _team_section(team_df: pd.DataFrame, team_name: str) -> str:
    top = team_df.sort_values("booking_probability_index", ascending=False).head(3).copy()
    lines = [
        f"### {team_name}",
        f"- average fouls per match: {team_df['team_avg_fouls'].dropna().iloc[0] if team_df['team_avg_fouls'].notna().any() else 'n/a'}",
        f"- average yellow cards per match: {team_df['team_avg_yellows'].dropna().iloc[0] if team_df['team_avg_yellows'].notna().any() else 'n/a'}",
        "",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        notes = getattr(row, "analyst_notes", "") or ""
        lines.extend(
            [
                f"#### {i}. {row.player_name}",
                f"- role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
                f"- fouls per 90: {getattr(row, 'fouls_per90', 'n/a')}",
                f"- yellow cards per 90: {getattr(row, 'yellow_cards_per90', 'n/a')}",
                f"- BPI: {getattr(row, 'booking_probability_index', 0.0):.1f}",
                f"- contact cascade: {getattr(row, 'bpi_contact_cascade_layer', 0.0):.3f}",
                f"- non-contact layer: {getattr(row, 'bpi_non_contact_card_layer', 0.0):.3f} ({getattr(row, 'non_contact_card_bucket', 'none')})",
                f"- confidence: {getattr(row, 'confidence_label', 'n/a')}",
                f"- risk notes: {notes if str(notes).strip() else 'historical discipline + matchup + referee profile'}",
                "",
            ]
        )
    return "\n".join(lines)


def _contact_summary(first: pd.Series) -> list[str]:
    return [
        "## Contact Style Snapshot",
        f"- fixture style: {_scalar(first.get('fixture_style_label', ''), 'UNSET')}",
        f"- attacking style: {_scalar(first.get('fixture_attacking_style_label', ''), 'UNSET')}",
        f"- foul density: {_num(first.get('fixture_foul_density_score', 0.0)):.3f}",
        f"- tackle density: {_num(first.get('fixture_tackle_density_score', 0.0)):.3f}",
        f"- midfield grind: {_num(first.get('fixture_midfield_grind_score', 0.0)):.3f}",
        f"- wide duel score: {_num(first.get('fixture_wide_duel_score', 0.0)):.3f}",
        f"- attack pressure: {_num(first.get('fixture_attack_pressure_score', 0.0)):.3f}",
        f"- corner pressure: {_num(first.get('fixture_corner_pressure_score', 0.0)):.3f}",
        f"- territorial stress: {_num(first.get('fixture_territorial_stress_score', 0.0)):.3f}",
        f"- home fouls l5/l10: {_num(first.get('home_team_fouls_l5', 0.0)):.2f} / {_num(first.get('home_team_fouls_l10', 0.0)):.2f}",
        f"- away fouls l5/l10: {_num(first.get('away_team_fouls_l5', 0.0)):.2f} / {_num(first.get('away_team_fouls_l10', 0.0)):.2f}",
        f"- home tackles l5/l10: {_num(first.get('home_team_tackles_l5', 0.0)):.2f} / {_num(first.get('home_team_tackles_l10', 0.0)):.2f}",
        f"- away tackles l5/l10: {_num(first.get('away_team_tackles_l5', 0.0)):.2f} / {_num(first.get('away_team_tackles_l10', 0.0)):.2f}",
        f"- home shots l5/l10: {_num(first.get('home_team_shots_l5', 0.0)):.2f} / {_num(first.get('home_team_shots_l10', 0.0)):.2f}",
        f"- away shots l5/l10: {_num(first.get('away_team_shots_l5', 0.0)):.2f} / {_num(first.get('away_team_shots_l10', 0.0)):.2f}",
        f"- home SOT l5/l10: {_num(first.get('home_team_shots_on_goal_l5', 0.0)):.2f} / {_num(first.get('home_team_shots_on_goal_l10', 0.0)):.2f}",
        f"- away SOT l5/l10: {_num(first.get('away_team_shots_on_goal_l5', 0.0)):.2f} / {_num(first.get('away_team_shots_on_goal_l10', 0.0)):.2f}",
        f"- home corners for/against l5: {_num(first.get('home_team_corners_for_l5', 0.0)):.2f} / {_num(first.get('home_team_corners_against_l5', 0.0)):.2f}",
        f"- away corners for/against l5: {_num(first.get('away_team_corners_for_l5', 0.0)):.2f} / {_num(first.get('away_team_corners_against_l5', 0.0)):.2f}",
        f"- h2h total fouls l5/l10: {_num(first.get('h2h_total_fouls_l5', 0.0)):.2f} / {_num(first.get('h2h_total_fouls_l10', 0.0)):.2f}",
        f"- h2h total tackles l5/l10: {_num(first.get('h2h_total_tackles_l5', 0.0)):.2f} / {_num(first.get('h2h_total_tackles_l10', 0.0)):.2f}",
        f"- h2h total shots l5: {_num(first.get('h2h_total_shots_l5', 0.0)):.2f}",
        f"- h2h total shots on goal l5: {_num(first.get('h2h_total_shots_on_goal_l5', 0.0)):.2f}",
        f"- h2h total corners l5: {_num(first.get('h2h_total_corners_l5', 0.0)):.2f}",
        "",
    ]


def render_report(df: pd.DataFrame) -> str:
    first = df.iloc[0]
    home = _scalar(first.get("home_team_name", ""))
    away = _scalar(first.get("away_team_name", ""))
    league = _scalar(first.get("league", ""))
    competition = _scalar(first.get("competition", league), league)
    fixture_key = _scalar(first.get("fixture_key", ""))
    referee = _scalar(first.get("referee_name", ""), "TBC")
    venue = _scalar(first.get("venue", ""), "TBC")
    weather = _scalar(first.get("weather_summary", ""), "TBC")
    match_date = _scalar(first.get("match_date", ""))

    home_df = df[_string(df["team_name"]).eq(home)].copy()
    away_df = df[_string(df["team_name"]).eq(away)].copy()
    if home_df.empty or away_df.empty:
        side = _string(df.get("player_team_side", pd.Series("", index=df.index))).str.upper()
        home_df = df[side.eq("HOME")].copy()
        away_df = df[side.eq("AWAY")].copy()

    ranked = df.sort_values("booking_probability_index", ascending=False).copy()
    high = ranked[ranked["confidence_label"].eq("High")].head(4)
    moderate = ranked[ranked["confidence_label"].eq("Moderate")].head(4)

    summary_lines = [
        f"# BAWA Yellow Card Prediction: {home} vs. {away}",
        "",
        "## Fixture Overview",
        f"- fixture key: `{fixture_key}`",
        f"- competition: {competition}",
        f"- league: {league}",
        f"- match date: {match_date}",
        f"- venue: {venue}",
        f"- referee: {referee}",
        f"- weather: {weather}",
        "",
        "## Referee Influence",
        f"- cards per match: {first.get('ref_cards_per_match', 'n/a')}",
        f"- foul-to-card ratio: {first.get('ref_foul_to_card_ratio', 'n/a')}",
        f"- dissent strictness: {first.get('ref_dissent_strictness', 'n/a')}",
        f"- time-wasting strictness: {first.get('ref_timewasting_strictness', 'n/a')}",
        "",
        *_contact_summary(first),
        "## Team Disciplinary Profiles",
        _team_section(home_df, home) if not home_df.empty else f"### {home}\n- no team rows found\n",
        _team_section(away_df, away) if not away_df.empty else f"### {away}\n- no team rows found\n",
        "## Ranked Yellow Card Candidates",
    ]

    for idx, row in enumerate(ranked.head(6).itertuples(index=False), start=1):
        summary_lines.extend(
            [
                f"{idx}. {row.player_name} ({row.team_name})",
                f"   - selection confidence: {row.confidence_label}",
                f"   - BPI: {row.booking_probability_index:.1f}",
                f"   - role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
                f"   - fouls/90: {getattr(row, 'fouls_per90', 'n/a')} | yellows/90: {getattr(row, 'yellow_cards_per90', 'n/a')}",
                f"   - cascade: {getattr(row, 'booking_cascade_type', 'CONTACT_CASCADE')} | contact={getattr(row, 'bpi_contact_cascade_layer', 0.0):.3f} | non-contact={getattr(row, 'bpi_non_contact_card_layer', 0.0):.3f} ({getattr(row, 'non_contact_card_bucket', 'none')})",
            ]
        )

    non_contact_watch = ranked[ranked["bpi_non_contact_card_layer"].ge(0.12)].head(6)
    summary_lines.extend(["", "## Non-Contact Card Layer"])
    if non_contact_watch.empty:
        summary_lines.append("- no non-contact card route elevated from the available inputs")
    else:
        for row in non_contact_watch.itertuples(index=False):
            summary_lines.append(
                f"- {row.player_name} ({row.team_name}) | bucket={row.non_contact_card_bucket} | non-contact={row.bpi_non_contact_card_layer:.3f} | cascade={row.booking_cascade_type}"
            )

    summary_lines.extend(
        [
            "",
            "## Confidence Buckets",
            "### High Confidence",
        ]
    )
    for row in high.itertuples(index=False):
        summary_lines.append(
            f"- {row.player_name} ({row.team_name}) — BPI {row.booking_probability_index:.1f}"
        )
    summary_lines.extend(["", "### Moderate Confidence"])
    for row in moderate.itertuples(index=False):
        summary_lines.append(
            f"- {row.player_name} ({row.team_name}) — BPI {row.booking_probability_index:.1f}"
        )

    summary_lines.extend(
        [
            "",
            "## Final Recommendation",
            f"- most likely booked for {home}: {home_df.sort_values('booking_probability_index', ascending=False).iloc[0]['player_name'] if not home_df.empty else 'n/a'}",
            f"- most likely booked for {away}: {away_df.sort_values('booking_probability_index', ascending=False).iloc[0]['player_name'] if not away_df.empty else 'n/a'}",
            "- use final lineups and late referee confirmation to adjust fringe candidates before kickoff",
            "",
        ]
    )
    return "\n".join(summary_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a semi-automated BAWA yellow card report from player-event fixture inputs.")
    parser.add_argument("--input", required=True, help="CSV input matching PLAYER_EVENTS_INPUT_SCHEMA.csv")
    parser.add_argument("--fixture-key", default="", help="Render a single fixture key from the input")
    parser.add_argument("--outdir", default="reports/player_events", help="Output directory for markdown report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    if "fixture_key" not in df.columns:
        raise SystemExit("Input file must include fixture_key column.")

    fixture_df = _pick_fixture(df, args.fixture_key)
    scored = _build_bpi(fixture_df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixture_key = _string(scored["fixture_key"]).iloc[0]
    safe_key = fixture_key.replace("/", "_").replace(" ", "_")
    out_path = outdir / f"{safe_key}__yellow_card_report.md"
    out_path.write_text(render_report(scored))

    print(f"WROTE: {out_path}")
    print(f"rows: {len(scored)}")
    print(
        "top_candidates:",
        scored.sort_values("booking_probability_index", ascending=False)[["player_name", "team_name", "booking_probability_index"]]
        .head(5)
        .to_dict(orient="records"),
    )


if __name__ == "__main__":
    main()
