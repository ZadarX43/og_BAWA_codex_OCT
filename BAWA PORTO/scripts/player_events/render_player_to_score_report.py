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


def _goal_role_bonus(df: pd.DataFrame) -> pd.Series:
    role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += role.eq("central striker").astype(float) * 0.18
    out += role.eq("wide forward").astype(float) * 0.10
    out += role.eq("wide midfielder / winger").astype(float) * 0.04
    out += role.eq("central midfielder").astype(float) * 0.02
    out -= role.eq("holding midfielder").astype(float) * 0.08
    out -= role.eq("centre-back enforcer").astype(float) * 0.22
    out -= role.eq("wide defender / wing-back").astype(float) * 0.14
    out -= role.eq("goalkeeper").astype(float) * 0.35
    return out


def _attack_style_bonus(df: pd.DataFrame) -> pd.Series:
    label = _string(df.get("fixture_attacking_style_label", pd.Series("", index=df.index))).str.upper()
    out = pd.Series(0.0, index=df.index, dtype=float)
    out += label.eq("ATTACK_WAVE").astype(float) * 0.12
    out += label.eq("CORNER_SIEGE").astype(float) * 0.05
    out += label.eq("TERRITORY_TILT").astype(float) * 0.06
    out += 0.10 * _norm_cap(df.get("fixture_attack_pressure_score", 0.0), 1.0)
    out += 0.05 * _norm_cap(df.get("fixture_corner_pressure_score", 0.0), 1.0)
    out += 0.05 * _norm_cap(df.get("fixture_territorial_stress_score", 0.0), 1.0)
    out += 0.08 * _norm_cap(df.get("og_goal_environment_score", 0.0), 1.0)
    return out


def _build_gsi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    finisher = (
        0.34 * _norm_cap(out.get("goals_per90", 0.0), 1.0)
        + 0.26 * _norm_cap(out.get("shots_on_target_per90", 0.0), 1.8)
        + 0.18 * _norm_cap(out.get("shots_per90", 0.0), 4.0)
        + 0.12 * _norm_cap(out.get("expected_minutes", 0.0), 90.0)
        + 0.10 * _norm_cap(out.get("fouls_won_per90", 0.0), 3.0)
    )
    team_goal_env = (
        0.24 * _norm_cap(out.get("home_team_shots_on_goal_l5", 0.0) + out.get("away_team_shots_on_goal_l5", 0.0), 10.0)
        + 0.18 * _norm_cap(out.get("home_team_shots_l5", 0.0) + out.get("away_team_shots_l5", 0.0), 30.0)
        + 0.18 * _norm_cap(out.get("h2h_total_shots_on_goal_l5", 0.0), 10.0)
        + 0.15 * _norm_cap(out.get("og_goal_environment_score", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("og_battle_on_score", 0.0), 1.0)
        + 0.10 * _norm_cap(out.get("fixture_attack_pressure_score", 0.0), 1.0)
    )
    role_shape = (
        0.25 * _norm_cap(out.get("central_attack_dominance", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("left_flank_dominance", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("right_flank_dominance", 0.0), 1.0)
        + 0.20 * _norm_cap(out.get("fixture_attack_pressure_score", 0.0), 1.0)
        + 0.15 * _norm_cap(out.get("fixture_corner_pressure_score", 0.0), 1.0)
    )
    out["gsi_finisher"] = finisher
    out["gsi_team_goal_env"] = team_goal_env
    out["gsi_role_shape"] = role_shape
    out["gsi_role_bonus"] = _goal_role_bonus(out)
    out["gsi_attack_style_bonus"] = _attack_style_bonus(out)
    out["player_goal_index"] = (
        0.44 * finisher
        + 0.26 * team_goal_env
        + 0.18 * role_shape
        + out["gsi_role_bonus"]
        + out["gsi_attack_style_bonus"]
    ) * 100.0
    score = out["player_goal_index"].fillna(0.0)
    out["confidence_label"] = pd.cut(score, bins=[-1, 30, 50, 200], labels=["Low", "Moderate", "High"]).astype("string")
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
    ranked = df.sort_values("player_goal_index", ascending=False).copy()
    lines = [
        f"# BAWA Player To Score Prediction: {home} vs. {away}",
        "",
        "## Fixture Overview",
        f"- fixture key: `{fixture_key}`",
        f"- league: {league}",
        f"- attacking style: {_scalar(first.get('fixture_attacking_style_label', ''), 'UNSET')}",
        f"- goal environment: {float(pd.to_numeric(pd.Series([first.get('og_goal_environment_score', 0.0)]), errors='coerce').fillna(0.0).iloc[0]):.3f}",
        f"- battle on score: {float(pd.to_numeric(pd.Series([first.get('og_battle_on_score', 0.0)]), errors='coerce').fillna(0.0).iloc[0]):.3f}",
        "",
        "## Ranked Goal Candidates",
    ]
    for idx, row in enumerate(ranked.head(8).itertuples(index=False), start=1):
        lines.extend([
            f"{idx}. {row.player_name} ({row.team_name})",
            f"   - confidence: {row.confidence_label}",
            f"   - GSI: {row.player_goal_index:.1f}",
            f"   - role: {getattr(row, 'tactical_role', '') or getattr(row, 'position_group', '') or 'n/a'}",
            f"   - goals/90: {getattr(row, 'goals_per90', 'n/a')} | shots/90: {getattr(row, 'shots_per90', 'n/a')} | SOT/90: {getattr(row, 'shots_on_target_per90', 'n/a')}",
        ])
    lines.extend(["", "## Final Recommendation", "- beta shortlist only; not a priced anytime goalscorer model yet", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a beta player-to-score report from player-event fixture inputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixture-key", default="")
    parser.add_argument("--outdir", default="reports/player_events/goalscorer_beta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    fixture_df = _pick_fixture(df, args.fixture_key)
    scored = _build_gsi(fixture_df)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixture_key = _string(scored["fixture_key"]).iloc[0]
    safe_key = fixture_key.replace("/", "_").replace(" ", "_")
    out_path = outdir / f"{safe_key}__player_to_score_report.md"
    out_path.write_text(render_report(scored))
    print(f"WROTE: {out_path}")
    print(f"rows: {len(scored)}")
    print("top_candidates:", scored.sort_values("player_goal_index", ascending=False)[["player_name", "team_name", "player_goal_index"]].head(5).to_dict(orient="records"))


if __name__ == "__main__":
    main()
