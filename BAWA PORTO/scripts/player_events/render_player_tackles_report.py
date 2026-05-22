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


def _tackle_role_bonus(df: pd.DataFrame) -> pd.Series:
    role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += role.eq("wide defender / wing-back").astype(float) * 0.12
    out += role.eq("holding midfielder").astype(float) * 0.12
    out += role.eq("centre-back enforcer").astype(float) * 0.08
    out += role.eq("central midfielder").astype(float) * 0.05
    out -= role.eq("central striker").astype(float) * 0.12
    out -= role.eq("wide forward").astype(float) * 0.06
    out -= role.eq("goalkeeper").astype(float) * 0.35
    return out


def _contact_pressure_bonus(df: pd.DataFrame) -> pd.Series:
    label = _string(df.get("fixture_style_label", pd.Series("", index=df.index))).str.upper()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += label.eq("AGGRESSIVE_BOTH").astype(float) * 0.10
    out += label.eq("MIDFIELD_GRIND").astype(float) * 0.12
    out += label.eq("WIDE_DUEL_GAME").astype(float) * 0.08
    out += label.eq("ONE_SIDED_PRESSURE").astype(float) * 0.05
    out += 0.08 * _norm_cap(df.get("fixture_tackle_density_score", 0.0), 1.0)
    out += 0.06 * _norm_cap(df.get("fixture_foul_density_score", 0.0), 1.0)
    out += 0.06 * _norm_cap(df.get("fixture_midfield_grind_score", 0.0), 1.0)
    out += 0.04 * _norm_cap(df.get("fixture_wide_duel_score", 0.0), 1.0)
    return out


def _quality_pressure_bonus(df: pd.DataFrame) -> pd.Series:
    quality_score = _norm_cap(df.get("player_quality_score_l5", 0.0), 100.0)
    quality_pct = _norm_cap(df.get("player_quality_percentile_in_position", 0.0), 1.0)
    quality_edge = pd.to_numeric(df.get("starting_xi_quality_edge", 0.0), errors="coerce").fillna(0.0).astype(float)
    negative_edge = ((-quality_edge).clip(lower=0.0, upper=20.0) / 20.0).fillna(0.0)
    opponent_quality = _norm_cap(df.get("opponent_starting_xi_team_quality_score", 0.0), 90.0)
    return (
        0.26 * (1.0 - quality_score)
        + 0.14 * (1.0 - quality_pct)
        + 0.28 * negative_edge
        + 0.17 * opponent_quality
        + 0.15 * _norm_cap(df.get("power_gap_directional_pressure_score", 0.0), 1.0)
    )


def _formation_mismatch_bonus(df: pd.DataFrame) -> pd.Series:
    return (
        0.24 * _norm_cap(df.get("formation_pressure_score", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("formation_wide_overload_flag", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("formation_midfield_grind_flag", 0.0), 1.0)
        + 0.22 * _norm_cap(df.get("underdog_fullback_wide_overload_flag", 0.0), 1.0)
        + 0.18 * _norm_cap(df.get("underdog_dm_midfield_grind_flag", 0.0), 1.0)
        + 0.14 * _norm_cap((-pd.to_numeric(df.get("lineup_xi_tackle_pressure_delta", 0.0), errors="coerce")).clip(lower=0.0, upper=4.0), 4.0)
        + 0.08 * _norm_cap(pd.to_numeric(df.get("formation_mismatch_flag", 0.0), errors="coerce"), 1.0)
    )


def _build_tki(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    player_actions = (
        0.42 * _norm_cap(out.get("tackles_per90", 0.0), 4.5)
        + 0.20 * _norm_cap(out.get("interceptions_per90", 0.0), 3.0)
        + 0.16 * _norm_cap(out.get("ground_duel_loss_rate", 0.0), 1.0)
        + 0.12 * _norm_cap(out.get("dribbles_faced_per90", 0.0), 6.0)
        + 0.10 * _norm_cap(out.get("expected_minutes", 0.0), 90.0)
    )
    role_pressure = (
        0.28 * _norm_cap(out.get("central_battle_flag", 0.0), 1.0)
        + 0.22 * _norm_cap(out.get("counterattack_defender_flag", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("weak_flank_overload_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("weak_midfield_overload_flag", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("weak_territory_protection_flag", 0.0), 1.0)
    )
    team_context = (
        0.30 * _norm_cap(out.get("home_team_tackles_l5", 0.0) + out.get("away_team_tackles_l5", 0.0), 38.0)
        + 0.20 * _norm_cap(out.get("home_team_tackles_l10", 0.0) + out.get("away_team_tackles_l10", 0.0), 38.0)
        + 0.15 * _norm_cap(out.get("h2h_total_tackles_l5", 0.0), 40.0)
        + 0.15 * _norm_cap(out.get("opponent_possession_projection", 0.0), 70.0)
        + 0.20 * _norm_cap(out.get("team_avg_fouls", 0.0), 18.0)
    )
    out["tki_player_actions"] = player_actions
    out["tki_role_pressure"] = role_pressure
    out["tki_team_context"] = team_context
    out["tki_role_bonus"] = _tackle_role_bonus(out)
    out["tki_contact_pressure_bonus"] = _contact_pressure_bonus(out)
    out["tki_quality_pressure_bonus"] = _quality_pressure_bonus(out)
    out["tki_formation_mismatch_bonus"] = _formation_mismatch_bonus(out)
    out["player_tackle_index"] = (
        0.36 * player_actions
        + 0.22 * role_pressure
        + 0.18 * team_context
        + out["tki_role_bonus"]
        + out["tki_contact_pressure_bonus"]
        + 0.12 * out["tki_quality_pressure_bonus"]
        + 0.10 * out["tki_formation_mismatch_bonus"]
    ) * 100.0
    score = out["player_tackle_index"].fillna(0.0)
    out["confidence_label"] = pd.cut(score, bins=[-1, 40, 58, 200], labels=["Low", "Moderate", "High"]).astype("string")
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


def render_report(df: pd.DataFrame) -> str:
    first = df.iloc[0]
    home = _scalar(first.get("home_team_name", ""))
    away = _scalar(first.get("away_team_name", ""))
    league = _scalar(first.get("league", ""))
    fixture_key = _scalar(first.get("fixture_key", ""))
    ranked = df.sort_values("player_tackle_index", ascending=False).copy()
    lines = [
        f"# BAWA Player Tackles Prediction: {home} vs. {away}",
        "",
        "## Fixture Overview",
        f"- fixture key: `{fixture_key}`",
        f"- league: {league}",
        f"- contact style: {_scalar(first.get('fixture_style_label', ''), 'UNSET')}",
        f"- tackle density: {float(pd.to_numeric(pd.Series([first.get('fixture_tackle_density_score', 0.0)]), errors='coerce').fillna(0.0).iloc[0]):.3f}",
        f"- midfield grind: {float(pd.to_numeric(pd.Series([first.get('fixture_midfield_grind_score', 0.0)]), errors='coerce').fillna(0.0).iloc[0]):.3f}",
        "",
        "## Ranked Tackle Candidates",
    ]
    for idx, row in enumerate(ranked.head(8).itertuples(index=False), start=1):
        lines.extend(
            [
                f"{idx}. {row.player_name} ({row.team_name})",
                f"   - confidence: {row.confidence_label}",
                f"   - TKI: {row.player_tackle_index:.1f}",
                f"   - role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
                f"   - tackles/90: {getattr(row, 'tackles_per90', 'n/a')} | interceptions/90: {getattr(row, 'interceptions_per90', 'n/a')}",
            ]
        )
    lines.extend(["", "## Final Recommendation", "- beta shortlist only; not a priced tackles market model yet", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a beta player tackles report from player-event fixture inputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixture-key", default="")
    parser.add_argument("--outdir", default="reports/player_events/tackles_beta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    fixture_df = _pick_fixture(df, args.fixture_key)
    scored = _build_tki(fixture_df)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixture_key = _string(scored["fixture_key"]).iloc[0]
    safe_key = fixture_key.replace("/", "_").replace(" ", "_")
    out_path = outdir / f"{safe_key}__player_tackles_report.md"
    out_path.write_text(render_report(scored))
    print(f"WROTE: {out_path}")
    print(f"rows: {len(scored)}")
    print("top_candidates:", scored.sort_values("player_tackle_index", ascending=False)[["player_name", "team_name", "player_tackle_index"]].head(5).to_dict(orient="records"))


if __name__ == "__main__":
    main()
