from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_player_shots_report import _build_psi
from render_player_shots_on_target_report import _build_soti


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _build_fixture_gate(base: pd.DataFrame, min_fixture_quality_score: float = 0.0) -> pd.DataFrame:
    fixture = (
        base[
            [
                "fixture_key",
                "fixture_attack_pressure_score",
                "fixture_corner_pressure_score",
                "fixture_territorial_stress_score",
                "fixture_attacking_style_label",
                "fixture_style_label",
                "og_goal_environment_score",
                "og_battle_on_score",
                "og_goal_environment_label",
                "home_team_shots_l5",
                "away_team_shots_l5",
                "home_team_shots_on_goal_l5",
                "away_team_shots_on_goal_l5",
                "home_team_corners_for_l5",
                "away_team_corners_for_l5",
                "home_team_corners_against_l5",
                "away_team_corners_against_l5",
                "h2h_total_shots_l5",
                "h2h_total_shots_on_goal_l5",
                "h2h_total_corners_l5",
                "starting_xi_team_quality_score",
                "starting_xi_attack_quality_score",
                "starting_xi_quality_edge",
                "player_quality_score_l5",
            ]
        ]
        .drop_duplicates(subset=["fixture_key"])
        .copy()
    )
    fixture["fixture_attack_quality_score"] = (
        0.22 * _norm_cap(fixture["fixture_attack_pressure_score"], 1.0)
        + 0.14 * _norm_cap(fixture["fixture_corner_pressure_score"], 1.0)
        + 0.14 * _norm_cap(fixture["fixture_territorial_stress_score"], 1.0)
        + 0.18 * _norm_cap(fixture["og_goal_environment_score"], 1.0)
        + 0.12 * _norm_cap(fixture["og_battle_on_score"], 1.0)
        + 0.08 * _norm_cap(fixture["home_team_shots_l5"] + fixture["away_team_shots_l5"], 30.0)
        + 0.06 * _norm_cap(fixture["home_team_shots_on_goal_l5"] + fixture["away_team_shots_on_goal_l5"], 10.0)
        + 0.06 * _norm_cap(fixture["h2h_total_shots_l5"], 32.0)
        + 0.05 * _norm_cap(fixture["starting_xi_team_quality_score"], 90.0)
        + 0.05 * _norm_cap(fixture["starting_xi_attack_quality_score"], 90.0)
        + 0.03 * _norm_cap(fixture["player_quality_score_l5"], 100.0)
        + 0.03 * _norm_cap(fixture["starting_xi_quality_edge"].abs(), 20.0)
    )
    fixture["strong_attack_support_flag"] = (
        fixture["fixture_attack_pressure_score"].ge(0.66)
        | fixture["fixture_corner_pressure_score"].ge(0.66)
        | fixture["fixture_territorial_stress_score"].ge(0.66)
        | fixture["fixture_attacking_style_label"].astype("string").str.upper().isin(["ATTACK_WAVE", "CORNER_SIEGE", "TERRITORY_TILT"])
    ).astype(int)
    fixture["strong_goal_env_support_flag"] = fixture["og_goal_environment_score"].ge(0.60).astype(int)
    fixture["strong_lineup_quality_support_flag"] = (
        _norm_cap(fixture["starting_xi_attack_quality_score"], 90.0).ge(0.68)
        | _norm_cap(fixture["player_quality_score_l5"], 100.0).ge(0.68)
    ).astype(int)
    fixture["fixture_attack_quality_pass_flag"] = fixture["fixture_attack_quality_score"].ge(min_fixture_quality_score).astype(int)
    reasons = []
    for row in fixture.itertuples(index=False):
        toks = []
        if float(getattr(row, "fixture_attack_pressure_score", 0.0)) >= 0.66:
            toks.append("ATTACK_PRESSURE_SUPPORT")
        if float(getattr(row, "fixture_corner_pressure_score", 0.0)) >= 0.66:
            toks.append("CORNER_PRESSURE_SUPPORT")
        if float(getattr(row, "fixture_territorial_stress_score", 0.0)) >= 0.66:
            toks.append("TERRITORY_STRESS_SUPPORT")
        if float(getattr(row, "og_goal_environment_score", 0.0)) >= 0.60:
            toks.append("GOAL_ENV_SUPPORT")
        if float(getattr(row, "og_battle_on_score", 0.0)) >= 0.55:
            toks.append("BATTLE_ON_SUPPORT")
        att = str(getattr(row, "fixture_attacking_style_label", "") or "").upper()
        if att and att != "BALANCED_ATTACK":
            toks.append(f"ATTACK_STYLE_{att}")
        contact = str(getattr(row, "fixture_style_label", "") or "").upper()
        if contact and contact != "BALANCED_CONTACT":
            toks.append(f"CONTACT_STYLE_{contact}")
        if float(getattr(row, "starting_xi_attack_quality_score", 0.0)) >= 61.0:
            toks.append("LINEUP_ATTACK_QUALITY")
        if not toks:
            toks.append("LOW_ATTACK_CONTEXT")
        reasons.append("|".join(toks))
    fixture["fixture_attack_reason_codes"] = reasons
    return fixture


def _prep_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "shots":
        scored = _build_psi(df.copy())
        scored = scored.rename(columns={"player_shot_index": "market_score", "confidence_label": "market_confidence"})
        scored["market"] = "shots"
    elif market == "shots_on_target":
        scored = _build_soti(df.copy())
        scored = scored.rename(columns={"player_sot_index": "market_score", "confidence_label": "market_confidence"})
        scored["market"] = "shots_on_target"
    else:
        raise ValueError(f"Unsupported market: {market}")
    return scored


def build_combined_board(
    input_csv: str,
    output_csv: str,
    output_md: str,
    shot_min_score: float = 84.0,
    sot_min_score: float = 88.0,
    max_shots_per_fixture: int = 1,
    max_sot_per_fixture: int = 1,
    leagues_filter: list[str] | None = None,
    min_fixture_quality_score: float = 0.72,
    require_attack_support: bool = True,
    require_goal_env_support: bool = True,
    allowed_attacking_style_labels: list[str] | None = None,
    dual_trigger_only: bool = False,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv)
    if leagues_filter:
        lf = {x.strip().lower() for x in leagues_filter if x.strip()}
        base = base[base["league"].astype("string").str.lower().isin(lf)].copy()

    fixture_gate = _build_fixture_gate(base, min_fixture_quality_score=min_fixture_quality_score)
    gate_cols = [
        "fixture_key",
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "fixture_territorial_stress_score",
        "fixture_attacking_style_label",
        "fixture_style_label",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_goal_environment_label",
        "fixture_attack_quality_score",
        "strong_attack_support_flag",
        "strong_goal_env_support_flag",
        "strong_lineup_quality_support_flag",
        "fixture_attack_quality_pass_flag",
        "fixture_attack_reason_codes",
    ]

    shots = _prep_market(base, "shots")
    sot = _prep_market(base, "shots_on_target")
    overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in shots.columns]
    if overlap_cols:
        shots = shots.drop(columns=overlap_cols)
    overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in sot.columns]
    if overlap_cols:
        sot = sot.drop(columns=overlap_cols)
    shots = shots.merge(fixture_gate[gate_cols], on="fixture_key", how="left")
    sot = sot.merge(fixture_gate[gate_cols], on="fixture_key", how="left")

    shots = shots[shots["market_score"].ge(shot_min_score)].copy()
    sot = sot[sot["market_score"].ge(sot_min_score)].copy()
    shots = shots[shots["fixture_attack_quality_pass_flag"].eq(1)].copy()
    sot = sot[sot["fixture_attack_quality_pass_flag"].eq(1)].copy()
    if require_attack_support:
        shots = shots[shots["strong_attack_support_flag"].eq(1)].copy()
        sot = sot[sot["strong_attack_support_flag"].eq(1)].copy()
    if require_goal_env_support:
        shots = shots[shots["strong_goal_env_support_flag"].eq(1)].copy()
        sot = sot[sot["strong_goal_env_support_flag"].eq(1)].copy()
    if allowed_attacking_style_labels:
        allowed = {x.strip().upper() for x in allowed_attacking_style_labels if x.strip()}
        shots = shots[shots["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()
        sot = sot[sot["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()

    shots = shots.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_shots_per_fixture)
    sot = sot.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_sot_per_fixture)

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
        "shots_per90",
        "shots_on_target_per90",
        "fouls_won_per90",
        "expected_minutes",
        "fixture_style_label",
        "fixture_attacking_style_label",
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "fixture_territorial_stress_score",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_goal_environment_label",
        "fixture_attack_quality_score",
        "strong_attack_support_flag",
        "strong_goal_env_support_flag",
        "strong_lineup_quality_support_flag",
        "fixture_attack_quality_pass_flag",
        "fixture_attack_reason_codes",
        "player_quality_score_l5",
        "player_form_tier",
        "player_quality_rank_in_position",
        "starting_xi_team_quality_score",
        "starting_xi_attack_quality_score",
        "starting_xi_quality_edge",
        "market",
        "market_score",
        "market_confidence",
        "analyst_notes",
    ]
    out = pd.concat([shots[keep_cols], sot[keep_cols]], ignore_index=True)
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
        out["combo_reason_bucket"] = (
            out["fixture_attacking_style_label"].astype("string").str.upper()
            + "__"
            + out["trigger_markets"].astype("string").str.replace("|", "_", regex=False).str.upper()
        )
    else:
        out["trigger_count"] = pd.Series(dtype=int)
        out["trigger_markets"] = pd.Series(dtype="string")
        out["same_player_dual_trigger_flag"] = pd.Series(dtype=int)
        out["combo_reason_bucket"] = pd.Series(dtype="string")

    if dual_trigger_only:
        out = out[out["same_player_dual_trigger_flag"].eq(1)].copy()

    out = out.sort_values(["match_date", "fixture_key", "market", "market_score"], ascending=[False, True, True, False]).reset_index(drop=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    lines = [
        "# Combined Attacking Board",
        "",
        "## Rules",
        f"- max shots names per fixture: {max_shots_per_fixture}",
        f"- max SOT names per fixture: {max_sot_per_fixture}",
        f"- shots minimum score: {shot_min_score}",
        f"- SOT minimum score: {sot_min_score}",
        f"- minimum attack quality score: {min_fixture_quality_score}",
        f"- require attack support: {require_attack_support}",
        f"- require goal environment support: {require_goal_env_support}",
        f"- allowed attacking styles: {', '.join(allowed_attacking_style_labels) if allowed_attacking_style_labels else 'all'}",
        f"- dual-trigger only: {dual_trigger_only}",
        "",
        "## Board Summary",
        f"- rows: {len(out)}",
        f"- fixtures covered: {out['fixture_key'].nunique() if not out.empty else 0}",
        f"- shots rows: {int((out['market'] == 'shots').sum()) if not out.empty else 0}",
        f"- shots on target rows: {int((out['market'] == 'shots_on_target').sum()) if not out.empty else 0}",
        f"- dual-trigger rows: {int(out['same_player_dual_trigger_flag'].sum()) if not out.empty else 0}",
        f"- average attack quality score: {round(float(out['fixture_attack_quality_score'].mean()), 3) if not out.empty else 0.0}",
        "",
        "## Fixture Samples",
    ]
    sample_cols = ["fixture_key", "market", "player_name", "team_name", "tactical_role", "market_score", "market_confidence"]
    for fixture_key, group in out.groupby("fixture_key", sort=False):
        lines.append(f"### {fixture_key}")
        fx = group.iloc[0]
        lines.append(
            f"- attack quality: {fx['fixture_attack_quality_score']:.3f} | battle_on: {fx['og_battle_on_score']:.3f} | goal_env: {fx['og_goal_environment_label']} | attack: {fx['fixture_attacking_style_label']} | contact: {fx['fixture_style_label']} | reasons: {fx['fixture_attack_reason_codes']}"
        )
        for row in group[sample_cols].itertuples(index=False):
            row0 = group[group["player_name"].eq(row.player_name)].iloc[0]
            extra = f" | dual:{row0['trigger_markets']}" if int(row0['same_player_dual_trigger_flag']) == 1 else ""
            lines.append(f"- `{row.market}`: {row.player_name} ({row.team_name}) | {row.tactical_role} | score={row.market_score:.1f} | {row.market_confidence}{extra}")
        lines.append("")
        if len(lines) > 120:
            break
    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined same-fixture shots + shots-on-target beta board.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--shot-min-score", type=float, default=84.0)
    parser.add_argument("--sot-min-score", type=float, default=88.0)
    parser.add_argument("--max-shots-per-fixture", type=int, default=1)
    parser.add_argument("--max-sot-per-fixture", type=int, default=1)
    parser.add_argument("--leagues", default="")
    parser.add_argument("--min-fixture-quality-score", type=float, default=0.72)
    parser.add_argument("--require-attack-support", action="store_true")
    parser.add_argument("--require-goal-env-support", action="store_true")
    parser.add_argument("--allowed-attacking-style-labels", default="")
    parser.add_argument("--dual-trigger-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    styles = [x.strip() for x in args.allowed_attacking_style_labels.split(",") if x.strip()]
    out = build_combined_board(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        shot_min_score=args.shot_min_score,
        sot_min_score=args.sot_min_score,
        max_shots_per_fixture=args.max_shots_per_fixture,
        max_sot_per_fixture=args.max_sot_per_fixture,
        leagues_filter=leagues or None,
        min_fixture_quality_score=args.min_fixture_quality_score,
        require_attack_support=args.require_attack_support,
        require_goal_env_support=args.require_goal_env_support,
        allowed_attacking_style_labels=styles or None,
        dual_trigger_only=args.dual_trigger_only,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
    print(f"fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
    print(f"shots_rows: {int((out['market'] == 'shots').sum()) if not out.empty else 0}")
    print(f"sot_rows: {int((out['market'] == 'shots_on_target').sum()) if not out.empty else 0}")
    print(f"dual_trigger_rows: {int(out['same_player_dual_trigger_flag'].sum()) if not out.empty else 0}")


if __name__ == "__main__":
    main()
