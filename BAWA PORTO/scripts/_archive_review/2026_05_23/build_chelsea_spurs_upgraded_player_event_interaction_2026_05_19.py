#!/usr/bin/env python3
"""Run a Chelsea-Spurs one-fixture upgraded player-event interaction pack.

This is research-only. It creates a one-fixture player-event input from the
possible XI, runs the weekend-upgraded player-event hit-rate and interaction
chain, then compares those outputs with the earlier light profile board.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTDIR = ROOT / "reports/latest/chelsea_spurs_2026_05_19_odds_genius_preview/upgraded_interaction_confirmed_lineups"
INPUT_DIR = BASE_OUTDIR / "fixture_inputs"
LIGHT_BOARD = ROOT / "reports/latest/chelsea_spurs_2026_05_19_odds_genius_preview/chelsea_spurs_player_event_board.csv"
PLAYER_STATS = ROOT / "Players/England Premier League/england-premier-league-players-2025-to-2026-stats.csv"
POLICY = ROOT / (
    "reports/2026-05-06/player_event_recent_form_opponent_allowance_interaction_audit/"
    "player_event_recent_form_opponent_allowance_top_candidates.csv"
)


POSSIBLE_XI = [
    ("Chelsea", "Robert Sanchez", "Goalkeeper", "Goalkeeper", 90),
    ("Chelsea", "Wesley Fofana", "Defender", "Centre-back enforcer", 90),
    ("Chelsea", "Josh Acheampong", "Defender", "Centre-back enforcer", 90),
    ("Chelsea", "Jorrel Hato", "Defender", "Centre-back enforcer", 90),
    ("Chelsea", "Marc Cucurella", "Defender", "Wide defender / wing-back", 90),
    ("Chelsea", "Andrey Santos", "Midfielder", "Holding midfielder", 90),
    ("Chelsea", "Moisés Caicedo", "Midfielder", "Holding midfielder", 90),
    ("Chelsea", "Cole Palmer", "Midfielder", "Wide midfielder / winger", 90),
    ("Chelsea", "Enzo Fernández", "Midfielder", "Central attacking midfielder", 90),
    ("Chelsea", "Pedro Neto", "Forward", "Wide forward", 80),
    ("Chelsea", "Liam Delap", "Forward", "Central striker", 80),
    ("Tottenham Hotspur", "Antonin Kinsky", "Goalkeeper", "Goalkeeper", 90),
    ("Tottenham Hotspur", "Pedro Porro", "Defender", "Wide defender / wing-back", 90),
    ("Tottenham Hotspur", "Kevin Danso", "Defender", "Centre-back enforcer", 90),
    ("Tottenham Hotspur", "Micky van de Ven", "Defender", "Centre-back enforcer", 90),
    ("Tottenham Hotspur", "Destiny Udogie", "Defender", "Wide defender / wing-back", 90),
    ("Tottenham Hotspur", "Rodrigo Bentancur", "Midfielder", "Holding midfielder", 90),
    ("Tottenham Hotspur", "João Palhinha", "Midfielder", "Holding midfielder", 90),
    ("Tottenham Hotspur", "Randal Kolo Muani", "Forward", "Wide forward", 80),
    ("Tottenham Hotspur", "Conor Gallagher", "Midfielder", "Central midfielder", 90),
    ("Tottenham Hotspur", "Mathys Tel", "Forward", "Wide forward", 80),
    ("Tottenham Hotspur", "Richarlison", "Forward", "Central striker", 85),
]

TIPS = [
    ("Wesley Fofana", "PLAYER_CARDS", 0.5, "Wesley booked"),
    ("Randal Kolo Muani", "PLAYER_CARDS", 0.5, "Muani booked"),
    ("Cole Palmer", "PLAYER_SOT", 0.5, "Palmer 1+ SOT"),
    ("Richarlison", "PLAYER_SOT", 0.5, "Richarlison 1+ SOT"),
    ("Conor Gallagher", "PLAYER_SHOTS", 0.5, "Gallagher 1+ shot"),
    ("Randal Kolo Muani", "PLAYER_SHOTS", 0.5, "Muani 1+ shot"),
    ("Marc Cucurella", "PLAYER_FOULED", 0.5, "Cucurella 1+ foul won"),
    ("João Palhinha", "PLAYER_FOULED", 0.5, "Palhinha 1+ foul won"),
]


def norm(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("í", "i")
        .replace("é", "e")
        .replace("ã", "a")
        .replace("ó", "o")
        .replace("ö", "o")
        .strip()
    )


def find_player(stats: pd.DataFrame, player: str) -> pd.Series | None:
    target = norm(player)
    names = stats["full_name"].map(norm)
    exact = stats[names.eq(target)]
    if not exact.empty:
        return exact.iloc[0]
    parts = [p for p in target.split() if len(p) > 2]
    if parts:
        mask = names.apply(lambda name: all(part in name for part in parts[-2:]))
        found = stats[mask]
        if not found.empty:
            return found.iloc[0]
        found = stats[names.str.contains(parts[-1], regex=False)]
        if not found.empty:
            return found.iloc[0]
    return None


def num(row: pd.Series | None, col: str, default: float = 0.0) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return default
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return default


def form_tier(rating: float) -> str:
    if rating >= 7.0:
        return "ELITE"
    if rating >= 6.65:
        return "STRONG"
    if rating >= 6.25:
        return "SOLID"
    return "BASELINE"


def build_input() -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(PLAYER_STATS)
    rows: list[dict[str, Any]] = []
    for team, player, position, role, minutes in POSSIBLE_XI:
        row = find_player(stats, player)
        rating = num(row, "average_rating_overall")
        side = "HOME" if team == "Chelsea" else "AWAY"
        rows.append(
            {
                "fixture_key": "2026_05_19_Chelsea_Tottenham_Hotspur",
                "match_date": "2026-05-19",
                "competition": "England Premier League",
                "league": "England Premier League",
                "home_team_name": "Chelsea",
                "away_team_name": "Tottenham Hotspur",
                "team_name": team,
                "player_name": player,
                "player_team_side": side,
                "position_group": position,
                "tactical_role": role,
                "expected_start_flag": 1,
                "expected_minutes": minutes,
                "referee_name": "Stuart Attwell",
                "ref_cards_per_match": 4.50,
                "fixture_style_label": "LONDON_DERBY_CONTACT",
                "fixture_attacking_style_label": "BALANCED_TRANSITION_ATTACK",
                "formation_matchup_label": "4-2-3-1 vs 4-2-3-1",
                "formation_pressure_score": 0.68,
                "fixture_foul_density_score": 0.76,
                "fixture_tackle_density_score": 0.72,
                "fixture_midfield_grind_score": 0.74,
                "fixture_wide_duel_score": 0.70,
                "fixture_attack_pressure_score": 0.64 if team == "Chelsea" else 0.56,
                "fixture_corner_pressure_score": 0.58 if team == "Chelsea" else 0.48,
                "fixture_territorial_stress_score": 0.66,
                "og_goal_environment_score": 0.56,
                "og_battle_on_score": 0.76,
                "player_quality_score_l5": rating,
                "player_form_rating_l5": rating,
                "player_form_tier": form_tier(rating),
                "minutes_last_3_matches": min(num(row, "minutes_played_overall") / max(num(row, "appearances_overall"), 1) * 3, 270),
                "days_rest": 3,
                "recent_injury_return_flag": 0,
                "suspension_risk_flag": 1 if num(row, "cards_per_90_overall") >= 0.30 else 0,
                "match_stakes_score": 0.86,
                "rivalry_flag": 1,
                "shots_per90": num(row, "shots_per_90_overall"),
                "shots_on_target_per90": num(row, "shots_on_target_per_90_overall"),
                "tackles_per90": num(row, "tackles_per_90_overall"),
                "fouls_per90": num(row, "fouls_committed_per_90_overall"),
                "fouls_won_per90": num(row, "fouls_drawn_per_90_overall"),
                "yellow_cards_per90": num(row, "cards_per_90_overall"),
                "source_status": "matched_player_stats" if row is not None else "lineup_context_only",
            }
        )
    out = INPUT_DIR / "player_events_fixture_input__England_Premier_League__2026.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def clean_player(value: Any) -> str:
    return norm(value).replace("joao", "joao").replace("r", "r")


def load_light_rows() -> pd.DataFrame:
    light = pd.read_csv(LIGHT_BOARD)
    light["board_version"] = "LIGHT_PROFILE_BOARD"
    return light


def compile_comparison() -> None:
    hitrate = pd.read_csv(BASE_OUTDIR / "player_event_hitrate_band_board/PLAYER_EVENT_HITRATE_BAND_BOARD.csv")
    dashboard = pd.read_csv(BASE_OUTDIR / "player_event_hitrate_band_board/PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv")
    interaction = pd.read_csv(
        BASE_OUTDIR
        / "player_event_interaction_live_shadow_board_exact/PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
    )
    light = load_light_rows()

    rows: list[dict[str, Any]] = []
    for player, market, threshold, tip in TIPS:
        hmask = (
            hitrate["player_name"].map(norm).eq(norm(player))
            & hitrate["market_family"].astype(str).eq(market)
            & pd.to_numeric(hitrate["threshold"], errors="coerce").round(3).eq(threshold)
        )
        hrow = hitrate[hmask].sort_values(["predicted_hit_rate", "support_score"], ascending=False).head(1)
        irow = interaction[
            interaction["player_name"].map(norm).eq(norm(player))
            & (
                (interaction["source_market"].astype(str).eq(market))
                | (
                    (market == "PLAYER_SOT")
                    & interaction["shadow_stage"].astype(str).str.contains("SOT_0_5", na=False)
                )
                | (
                    (market == "PLAYER_FOULED")
                    & interaction["shadow_stage"].astype(str).str.contains("FOULED_0_5", na=False)
                )
            )
        ].head(1)
        lrow = light[
            light["player"].map(norm).eq(norm(player))
            & light["event_market"].astype(str).str.contains(
                "card" if market == "PLAYER_CARDS" else "sot" if market == "PLAYER_SOT" else "foul" if market == "PLAYER_FOULED" else "shots",
                case=False,
                na=False,
            )
        ].head(1)

        if hrow.empty:
            upgraded_rating = "NO_UPGRADED_ROW"
            prob = ""
            confidence = ""
            support = ""
            context = ""
        else:
            rec = hrow.iloc[0]
            prob = rec.get("predicted_hit_rate_pct", "")
            confidence = rec.get("confidence_label", "")
            support = rec.get("support_score", "")
            context = rec.get("context_reason_codes", "")
            if str(confidence) in {"SHADOW_CORE", "STRONG_WATCH"}:
                upgraded_rating = "SURVIVES_STRONG"
            elif str(confidence) in {"WATCH", "ALT_WATCH"}:
                upgraded_rating = "SURVIVES_WATCH"
            elif market == "PLAYER_CARDS" and pd.to_numeric(pd.Series([rec.get("predicted_hit_rate", 0)]), errors="coerce").iloc[0] >= 0.25:
                upgraded_rating = "CARD_CONTEXT_WATCH"
            else:
                upgraded_rating = "DOES_NOT_SURVIVE"
        interaction_mode = "" if irow.empty else str(irow.iloc[0].get("interaction_match_mode", ""))
        exact_mode = interaction_mode == "EXACT_INTERACTION"

        rows.append(
            {
                "tip": tip,
                "player": player,
                "market_family": market,
                "threshold": threshold,
                "light_board_grade": "" if lrow.empty else lrow.iloc[0].get("grade", ""),
                "upgraded_board_version": "UPGRADED_INTERACTION_BOARD",
                "upgraded_rating": upgraded_rating,
                "upgraded_hit_rate_pct": prob,
                "upgraded_confidence_label": confidence,
                "upgraded_support_score": support,
                "upgraded_context": context,
                "upgraded_interaction_row": "YES" if not irow.empty else "NO",
                "exact_interaction_mode": "YES" if exact_mode else "NO",
                "interaction_priority": "" if irow.empty else irow.iloc[0].get("watch_priority", ""),
                "interaction_match_mode": interaction_mode,
                "interaction_backtest_hit_rate": "" if irow.empty else irow.iloc[0].get("backtest_hit_rate", ""),
                "notes": (
                    "True exact recent-form x opponent-allowance interaction mode."
                    if exact_mode
                    else "Interaction watch row exists, but live features fell back to proof-label mode."
                    if not irow.empty
                    else "Upgraded hit-rate/profile context only; exact interaction policy does not cover this exact market or did not pass."
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(BASE_OUTDIR / "LIGHT_PROFILE_BOARD_vs_UPGRADED_INTERACTION_BOARD.csv", index=False)

    top_cols = [
        "player_name",
        "team_name",
        "market_family",
        "threshold_name",
        "predicted_hit_rate_pct",
        "confidence_label",
        "support_score",
        "context_reason_codes",
        "lineup_watch_flags",
    ]
    dashboard[[c for c in top_cols if c in dashboard.columns]].to_csv(
        BASE_OUTDIR / "UPGRADED_INTERACTION_BOARD_TOP_ROWS.csv", index=False
    )

    report_lines = [
        "# Chelsea-Spurs Upgraded Player-Event Interaction Comparison",
        "",
        "Research-only comparison between the light profile board and the weekend-upgraded interaction stack.",
        "",
        "## Verdict By Tip",
        markdown_table(comparison),
        "",
        "## Interaction Rows",
        markdown_table(interaction.head(30)) if not interaction.empty else "_No exact interaction rows._",
        "",
        "## Files",
        "- `LIGHT_PROFILE_BOARD_vs_UPGRADED_INTERACTION_BOARD.csv`",
        "- `UPGRADED_INTERACTION_BOARD_TOP_ROWS.csv`",
        "- `player_event_hitrate_band_board/PLAYER_EVENT_HITRATE_BAND_BOARD.csv`",
        "- `player_event_live_feature_join/PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv`",
        "- `player_event_interaction_live_shadow_board_exact/PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv`",
    ]
    (BASE_OUTDIR / "LIGHT_PROFILE_BOARD_vs_UPGRADED_INTERACTION_BOARD.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df if max_rows is None else df.head(max_rows)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        vals = [str(row.get(col, "")).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    build_input()
    py = sys.executable
    run(
        [
            py,
            "scripts/build_player_event_hitrate_band_board.py",
            "--input-dir",
            str(INPUT_DIR),
            "--outdir",
            str(BASE_OUTDIR / "player_event_hitrate_band_board"),
            "--leagues",
            "England Premier League",
            "--min-minutes",
            "45",
            "--min-prob",
            "0.20",
        ]
    )
    run(
        [
            py,
            "scripts/build_player_event_live_interaction_features.py",
            "--input-dir",
            str(INPUT_DIR),
            "--leagues",
            "England_Premier_League",
            "--target-seasons",
            "2026",
            "--history-seasons",
            "2023,2024,2025",
            "--outdir",
            str(BASE_OUTDIR / "player_event_live_interaction_features"),
        ]
    )
    run(
        [
            py,
            "scripts/build_player_event_live_feature_join.py",
            "--player-event-board",
            str(BASE_OUTDIR / "player_event_hitrate_band_board/PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv"),
            "--recent-form",
            str(BASE_OUTDIR / "player_event_live_interaction_features/player_attacker_recent_form_live_features.csv"),
            "--opponent-allowance",
            str(
                BASE_OUTDIR
                / "player_event_live_interaction_features/player_event_opponent_attack_allowance_live_features.csv"
            ),
            "--outdir",
            str(BASE_OUTDIR / "player_event_live_feature_join"),
        ]
    )
    run(
        [
            py,
            "scripts/build_player_event_interaction_live_shadow_board.py",
            "--player-event-board",
            str(
                BASE_OUTDIR
                / "player_event_live_feature_join/PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv"
            ),
            "--policy",
            str(POLICY),
            "--outdir",
            str(BASE_OUTDIR / "player_event_interaction_live_shadow_board_exact"),
            "--per-market-limit",
            "200",
            "--allow-proof-label-only",
        ]
    )
    compile_comparison()
    print(f"WROTE {BASE_OUTDIR}")
    print(f"comparison={BASE_OUTDIR / 'LIGHT_PROFILE_BOARD_vs_UPGRADED_INTERACTION_BOARD.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
