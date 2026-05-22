from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _norm_cap(s: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _scalar(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() == "nan":
        return fallback
    return text


def _num(value: object, fallback: float = 0.0) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(fallback).iloc[0]
    return float(out)


def _league_role_bonus(df: pd.DataFrame) -> pd.Series:
    league = _string(df.get("league", pd.Series("", index=df.index))).str.lower()
    role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
    out = pd.Series(0.0, index=df.index, dtype=float)
    focus_scope = league.isin(["serie a", "la liga", "uefa europa league"])
    out += ((role.eq("holding midfielder")) & focus_scope).astype(float) * 0.10
    out += ((role.eq("wide defender / wing-back")) & focus_scope).astype(float) * 0.08
    out += ((role.eq("centre-back enforcer")) & focus_scope).astype(float) * 0.03
    out -= ((role.eq("central striker")) & focus_scope).astype(float) * 0.06
    out -= ((role.eq("goalkeeper")) & focus_scope).astype(float) * 0.20
    return out


def _fixture_style_bonus(df: pd.DataFrame) -> pd.Series:
    label = _string(df.get("fixture_style_label", pd.Series("", index=df.index))).str.upper()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += label.eq("AGGRESSIVE_BOTH").astype(float) * 0.12
    out += label.eq("MIDFIELD_GRIND").astype(float) * 0.10
    out += label.eq("WIDE_DUEL_GAME").astype(float) * 0.08
    out += label.eq("ONE_SIDED_PRESSURE").astype(float) * 0.05
    out += 0.08 * _norm_cap(df.get("fixture_foul_density_score", 0.0), 1.0)
    out += 0.05 * _norm_cap(df.get("fixture_tackle_density_score", 0.0), 1.0)
    out += 0.05 * _norm_cap(df.get("fixture_midfield_grind_score", 0.0), 1.0)
    out += 0.04 * _norm_cap(df.get("fixture_wide_duel_score", 0.0), 1.0)
    return out


def _quality_pressure_bonus(df: pd.DataFrame) -> pd.Series:
    quality_score = _norm_cap(df.get("player_quality_score_l5", 0.0), 100.0)
    quality_pct = _norm_cap(df.get("player_quality_percentile_in_position", 0.0), 1.0)
    quality_edge = pd.to_numeric(df.get("starting_xi_quality_edge", 0.0), errors="coerce").fillna(0.0).astype(float)
    negative_edge = ((-quality_edge).clip(lower=0.0, upper=20.0) / 20.0).fillna(0.0)
    team_def_quality = _norm_cap(df.get("starting_xi_defensive_quality_score", 0.0), 90.0)
    return (
        0.28 * (1.0 - quality_score)
        + 0.17 * (1.0 - quality_pct)
        + 0.30 * negative_edge
        + 0.15 * (1.0 - team_def_quality)
        + 0.10 * _norm_cap(df.get("weaker_side_under_pressure_flag", 0.0), 1.0)
    )


def _formation_mismatch_bonus(df: pd.DataFrame) -> pd.Series:
    return (
        0.26 * _norm_cap(df.get("formation_pressure_score", 0.0), 1.0)
        + 0.20 * _norm_cap(df.get("formation_wide_overload_flag", 0.0), 1.0)
        + 0.16 * _norm_cap(df.get("formation_midfield_grind_flag", 0.0), 1.0)
        + 0.20 * _norm_cap(df.get("underdog_fullback_wide_overload_flag", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("underdog_dm_midfield_grind_flag", 0.0), 1.0)
        + 0.12 * _norm_cap((-pd.to_numeric(df.get("lineup_xi_tackle_pressure_delta", 0.0), errors="coerce")).clip(lower=0.0, upper=4.0), 4.0)
        + 0.08 * _norm_cap((-pd.to_numeric(df.get("lineup_formation_defence_delta", 0.0), errors="coerce")).clip(lower=0.0, upper=3.0), 3.0)
    )


def _build_fci(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    history = (
        0.50 * _norm_cap(out.get("fouls_per90", 0.0), 4.0)
        + 0.20 * _norm_cap(out.get("tackles_per90", 0.0), 4.0)
        + 0.10 * _norm_cap(out.get("interceptions_per90", 0.0), 3.0)
        + 0.20 * _norm_cap(out.get("temperament_flag", 0.0), 1.0)
    )
    role_pressure = (
        0.30 * _norm_cap(out.get("counterattack_defender_flag", 0.0), 1.0)
        + 0.25 * _norm_cap(out.get("central_battle_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("lead_protection_foul_risk", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("late_game_pressure_risk", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("trailing_frustration_risk", 0.0), 1.0)
    )
    team_context = (
        0.35 * _norm_cap(out.get("team_avg_fouls", 0.0), 18.0)
        + 0.25 * _norm_cap(out.get("opponent_possession_projection", 0.0), 70.0)
        + 0.20 * _norm_cap(out.get("match_stakes_score", 0.0), 5.0)
        + 0.20 * _norm_cap(out.get("ref_cards_per_match", 0.0), 6.0)
    )
    directional_pressure = (
        0.40 * _norm_cap(out.get("power_gap_directional_pressure_score", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("weaker_side_under_pressure_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("weak_flank_overload_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("weak_midfield_overload_flag", 0.0), 1.0)
        + 0.10 * _norm_cap(out.get("weak_territory_protection_flag", 0.0), 1.0)
    )
    flank_central = (
        0.30 * _norm_cap(out.get("left_flank_dominance", 0.0), 1.0)
        + 0.30 * _norm_cap(out.get("right_flank_dominance", 0.0), 1.0)
        + 0.40 * _norm_cap(out.get("central_attack_dominance", 0.0), 1.0)
    )

    out["fci_history"] = history
    out["fci_role_pressure"] = role_pressure
    out["fci_team_context"] = team_context
    out["fci_directional_pressure"] = directional_pressure
    out["fci_flank_central"] = flank_central
    out["fci_role_bonus"] = _league_role_bonus(out)
    out["fci_fixture_style_bonus"] = _fixture_style_bonus(out)
    out["fci_quality_pressure"] = _quality_pressure_bonus(out)
    out["fci_formation_mismatch"] = _formation_mismatch_bonus(out)
    out["foul_commitment_index"] = (
        0.38 * history
        + 0.25 * role_pressure
        + 0.16 * team_context
        + 0.11 * directional_pressure
        + 0.10 * flank_central
        + out["fci_role_bonus"]
        + out["fci_fixture_style_bonus"]
        + 0.09 * out["fci_quality_pressure"]
        + 0.10 * out["fci_formation_mismatch"]
    ) * 100.0

    score = out["foul_commitment_index"].fillna(0.0)
    out["confidence_label"] = pd.cut(
        score,
        bins=[-1, 40, 58, 200],
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
        raise SystemExit("Input contains multiple fixtures. Supply --fixture-key to render a single report.")
    return out[_string(out["fixture_key"]).eq(keys[0])]


def _team_section(team_df: pd.DataFrame, team_name: str) -> str:
    top = team_df.sort_values("foul_commitment_index", ascending=False).head(3).copy()
    lines = [
        f"### {team_name}",
        f"- average fouls per match: {team_df['team_avg_fouls'].dropna().iloc[0] if team_df['team_avg_fouls'].notna().any() else 'n/a'}",
        "",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.extend(
            [
                f"#### {i}. {row.player_name}",
                f"- role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
                f"- fouls per 90: {getattr(row, 'fouls_per90', 'n/a')}",
                f"- tackles per 90: {getattr(row, 'tackles_per90', 'n/a')}",
                f"- FCI: {getattr(row, 'foul_commitment_index', 0.0):.1f}",
                f"- confidence: {getattr(row, 'confidence_label', 'n/a')}",
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
    match_date = _scalar(first.get("match_date", ""))

    home_df = df[_string(df["team_name"]).eq(home)].copy()
    away_df = df[_string(df["team_name"]).eq(away)].copy()
    if home_df.empty or away_df.empty:
        side = _string(df.get("player_team_side", pd.Series("", index=df.index))).str.upper()
        home_df = df[side.eq("HOME")].copy()
        away_df = df[side.eq("AWAY")].copy()

    ranked = df.sort_values("foul_commitment_index", ascending=False).copy()
    high = ranked[ranked["confidence_label"].eq("High")].head(5)
    moderate = ranked[ranked["confidence_label"].eq("Moderate")].head(5)

    lines = [
        f"# BAWA Fouls Committed Prediction: {home} vs. {away}",
        "",
        "## Fixture Overview",
        f"- fixture key: `{fixture_key}`",
        f"- competition: {competition}",
        f"- league: {league}",
        f"- match date: {match_date}",
        f"- venue: {venue}",
        f"- referee: {referee}",
        "",
        *_contact_summary(first),
        "## Team Foul Profiles",
        _team_section(home_df, home) if not home_df.empty else f"### {home}\n- no team rows found\n",
        _team_section(away_df, away) if not away_df.empty else f"### {away}\n- no team rows found\n",
        "## Ranked Fouls Candidates",
    ]
    for idx, row in enumerate(ranked.head(8).itertuples(index=False), start=1):
        lines.extend(
            [
                f"{idx}. {row.player_name} ({row.team_name})",
                f"   - confidence: {row.confidence_label}",
                f"   - FCI: {row.foul_commitment_index:.1f}",
                f"   - role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
                f"   - fouls/90: {getattr(row, 'fouls_per90', 'n/a')} | tackles/90: {getattr(row, 'tackles_per90', 'n/a')}",
            ]
        )
    lines.extend(["", "## Confidence Buckets", "### High Confidence"])
    for row in high.itertuples(index=False):
        lines.append(f"- {row.player_name} ({row.team_name}) — FCI {row.foul_commitment_index:.1f}")
    lines.extend(["", "### Moderate Confidence"])
    for row in moderate.itertuples(index=False):
        lines.append(f"- {row.player_name} ({row.team_name}) — FCI {row.foul_commitment_index:.1f}")
    lines.extend(
        [
            "",
            "## Final Recommendation",
            f"- strongest foul pressure for {home}: {home_df.sort_values('foul_commitment_index', ascending=False).iloc[0]['player_name'] if not home_df.empty else 'n/a'}",
            f"- strongest foul pressure for {away}: {away_df.sort_values('foul_commitment_index', ascending=False).iloc[0]['player_name'] if not away_df.empty else 'n/a'}",
            "- treat this as beta shortlist output until player foul lines and bookmaker coverage are wired in",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a semi-automated BAWA fouls committed report from player-event fixture inputs.")
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
    scored = _build_fci(fixture_df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixture_key = _string(scored["fixture_key"]).iloc[0]
    safe_key = fixture_key.replace("/", "_").replace(" ", "_")
    out_path = outdir / f"{safe_key}__fouls_committed_report.md"
    out_path.write_text(render_report(scored))

    print(f"WROTE: {out_path}")
    print(f"rows: {len(scored)}")
    print(
        "top_candidates:",
        scored.sort_values("foul_commitment_index", ascending=False)[["player_name", "team_name", "foul_commitment_index"]]
        .head(5)
        .to_dict(orient="records"),
    )


if __name__ == "__main__":
    main()
