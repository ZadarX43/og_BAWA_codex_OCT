from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_fouls_committed_report import _build_fci
from render_player_tackles_report import _build_tki
from render_player_shots_report import _build_psi
from render_player_shots_on_target_report import _build_soti

DEFAULT_TARGET_FORMATIONS = {"4-2-3-1 vs 4-4-2", "4-4-2 vs 4-2-3-1"}


def _prep(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "fouls_committed":
        out = _build_fci(df).copy()
        out["market_score"] = out["foul_commitment_index"]
        out["market_confidence"] = out["confidence_label"]
    elif market == "tackles":
        out = _build_tki(df).copy()
        out["market_score"] = out["player_tackle_index"]
        out["market_confidence"] = out["confidence_label"]
    elif market == "shots":
        out = _build_psi(df).copy()
        out["market_score"] = out["player_shot_index"]
        out["market_confidence"] = out["confidence_label"]
    elif market == "shots_on_target":
        out = _build_soti(df).copy()
        out["market_score"] = out["player_sot_index"]
        out["market_confidence"] = out["confidence_label"]
    else:
        raise ValueError(market)
    out["market"] = market
    return out


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


def build_board(
    input_csv: str,
    output_csv: str,
    output_md: str,
    target_formations: set[str] | None = None,
    title: str = "4-2-3-1 vs 4-4-2 Weekend Board",
    attack_quality_edge_min: float = 2.5,
    contact_quality_edge_min: float = 2.5,
    fixture_quality_min: float = 0.70,
    shots_min_score: float = 84.0,
    sot_min_score: float = 88.0,
    fouls_min_score: float = 64.0,
    tackles_min_score: float = 66.0,
    max_shots_per_fixture: int = 2,
    max_sot_per_fixture: int = 2,
    max_fouls_per_fixture: int = 1,
    max_tackles_per_fixture: int = 1,
    include_attacking: bool = True,
    include_contact: bool = True,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv, low_memory=False)
    target_formations = target_formations or set(DEFAULT_TARGET_FORMATIONS)
    base = base[base["formation_matchup_label"].isin(target_formations)].copy()
    base = _ensure_fixture_quality(base)
    base = base[pd.to_numeric(base["fixture_quality_score"], errors="coerce").fillna(0.0).ge(fixture_quality_min)].copy()
    if base.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text(f"# {title}\n\nNo rows matched.\n")
        return out

    fouls = _prep(base, "fouls_committed")
    tackles = _prep(base, "tackles")
    shots = _prep(base, "shots")
    sot = _prep(base, "shots_on_target")

    fouls = fouls[
        fouls["market_score"].ge(fouls_min_score)
        & fouls["fixture_style_label"].astype("string").str.upper().isin(["AGGRESSIVE_BOTH", "MIDFIELD_GRIND"])
        & pd.to_numeric(fouls["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-contact_quality_edge_min)
    ].copy()
    tackles = tackles[
        tackles["market_score"].ge(tackles_min_score)
        & tackles["fixture_style_label"].astype("string").str.upper().isin(["AGGRESSIVE_BOTH", "MIDFIELD_GRIND"])
        & pd.to_numeric(tackles["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-contact_quality_edge_min)
    ].copy()
    shots = shots[
        shots["market_score"].ge(shots_min_score)
        & shots["fixture_attacking_style_label"].astype("string").str.upper().isin(["BALANCED_ATTACK", "ATTACK_WAVE", "CORNER_SIEGE"])
        & pd.to_numeric(shots["starting_xi_quality_edge"], errors="coerce").fillna(0.0).ge(attack_quality_edge_min)
    ].copy()
    sot = sot[
        sot["market_score"].ge(sot_min_score)
        & sot["fixture_attacking_style_label"].astype("string").str.upper().isin(["BALANCED_ATTACK", "ATTACK_WAVE", "CORNER_SIEGE"])
        & pd.to_numeric(sot["starting_xi_quality_edge"], errors="coerce").fillna(0.0).ge(attack_quality_edge_min)
    ].copy()

    fouls = fouls.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", group_keys=False).head(max_fouls_per_fixture)
    tackles = tackles.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", group_keys=False).head(max_tackles_per_fixture)
    shots = shots.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", group_keys=False).head(max_shots_per_fixture)
    sot = sot.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", group_keys=False).head(max_sot_per_fixture)

    keep = [
        "fixture_key","match_date","competition","league","home_team_name","away_team_name",
        "team_name","player_name","player_team_side","position_group","tactical_role",
        "manual_pitch_side","manual_overload_target_side","formation_matchup_label",
        "fixture_style_label","fixture_attacking_style_label","formation_pressure_score","fixture_quality_score",
        "starting_xi_quality_edge","player_quality_score_l5","market","market_score","market_confidence",
    ]
    frames = []
    if include_attacking:
        frames.extend([shots[keep], sot[keep]])
    if include_contact:
        frames.extend([fouls[keep], tackles[keep]])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keep)
    out = out.sort_values(["match_date","fixture_key","market","market_score"], ascending=[True, True, True, False]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        f"# {title}",
        "",
        f"- target formations: {', '.join(sorted(target_formations))}",
        f"- include attacking: {include_attacking}",
        f"- include contact: {include_contact}",
        "- filters:",
        f"  - shots/SOT require quality edge >= {attack_quality_edge_min:.1f}",
        f"  - fouls/tackles require quality edge <= -{contact_quality_edge_min:.1f}",
        f"  - fixture quality floor = {fixture_quality_min:.2f}",
        "",
    ]
    for fixture_key, fx in out.groupby("fixture_key", sort=False):
        first = fx.iloc[0]
        lines.extend([
            f"## {fixture_key}",
            f"- {first['home_team_name']} vs {first['away_team_name']} | {first['competition']} | {first['formation_matchup_label']} | contact {first['fixture_style_label']} | attack {first['fixture_attacking_style_label']} | fixture_quality {first['fixture_quality_score']:.3f}",
        ])
        for _, row in fx.iterrows():
            side = f" | manual_side={row['manual_pitch_side']}->{row['manual_overload_target_side']}" if str(row.get('manual_pitch_side','')) else ""
            lines.append(
                f"- {row['market']}: {row['player_name']} ({row['team_name']}) | score={row['market_score']:.1f} | edge={row['starting_xi_quality_edge']:.1f} | quality={row['player_quality_score_l5']:.1f}{side}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weekend-ready 4-2-3-1 vs 4-4-2 shortlist board.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--target-formations", default="4-2-3-1 vs 4-4-2,4-4-2 vs 4-2-3-1")
    parser.add_argument("--title", default="4-2-3-1 vs 4-4-2 Weekend Board")
    parser.add_argument("--attack-quality-edge-min", type=float, default=2.5)
    parser.add_argument("--contact-quality-edge-min", type=float, default=2.5)
    parser.add_argument("--fixture-quality-min", type=float, default=0.70)
    parser.add_argument("--shots-min-score", type=float, default=84.0)
    parser.add_argument("--sot-min-score", type=float, default=88.0)
    parser.add_argument("--fouls-min-score", type=float, default=64.0)
    parser.add_argument("--tackles-min-score", type=float, default=66.0)
    parser.add_argument("--max-shots-per-fixture", type=int, default=2)
    parser.add_argument("--max-sot-per-fixture", type=int, default=2)
    parser.add_argument("--max-fouls-per-fixture", type=int, default=1)
    parser.add_argument("--max-tackles-per-fixture", type=int, default=1)
    parser.add_argument("--include-attacking", action="store_true")
    parser.add_argument("--include-contact", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    include_attacking = args.include_attacking or (not args.include_attacking and not args.include_contact)
    include_contact = args.include_contact or (not args.include_attacking and not args.include_contact)
    target_formations = {x.strip() for x in args.target_formations.split(",") if x.strip()}
    df = build_board(
        args.input,
        args.output_csv,
        args.output_md,
        target_formations=target_formations,
        title=args.title,
        attack_quality_edge_min=args.attack_quality_edge_min,
        contact_quality_edge_min=args.contact_quality_edge_min,
        fixture_quality_min=args.fixture_quality_min,
        shots_min_score=args.shots_min_score,
        sot_min_score=args.sot_min_score,
        fouls_min_score=args.fouls_min_score,
        tackles_min_score=args.tackles_min_score,
        max_shots_per_fixture=args.max_shots_per_fixture,
        max_sot_per_fixture=args.max_sot_per_fixture,
        max_fouls_per_fixture=args.max_fouls_per_fixture,
        max_tackles_per_fixture=args.max_tackles_per_fixture,
        include_attacking=include_attacking,
        include_contact=include_contact,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")
