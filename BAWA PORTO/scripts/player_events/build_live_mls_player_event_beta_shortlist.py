#!/usr/bin/env python3
"""Build a beta player-event shortlist for the live MLS review board.

This is a research/manual-review layer. It does not price player props and it
does not promote anything into deploy products.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAYER_RATINGS = ROOT / "frontend/public/data/player_intelligence/club_squad_ratings.json"
DEFAULT_MODEL_INTEL_ROWS = (
    ROOT
    / "reports/latest/live_mls_model_intelligence_compare_2026_05_13"
    / "live_mls_model_intelligence_rows.csv"
)
DEFAULT_FIXTURE_CARDS = (
    ROOT
    / "reports/latest/live_mls_model_intelligence_compare_2026_05_13"
    / "live_mls_fixture_intelligence_cards.csv"
)
DEFAULT_INJURIES = (
    ROOT
    / "reports/latest/api_current_context_overlay_window_mls_2026_05_13_to_2026_05_14"
    / "normalized/injuries__USA_MLS__2026.csv"
)
DEFAULT_LINEUPS = None
DEFAULT_FIXTURE_MASTER = None
DEFAULT_OUTDIR = ROOT / "reports/latest/live_mls_player_event_beta_shortlist_2026_05_13"


TEAM_ALIASES = {
    "la galaxy": "LA Galaxy",
    "los angeles galaxy": "LA Galaxy",
    "los angeles fc": "Los Angeles FC",
    "lafc": "Los Angeles FC",
    "new york red bulls": "New York RB",
    "new york rb": "New York RB",
    "dc united": "DC United",
    "d.c. united": "DC United",
    "cf montreal": "Montreal Impact",
    "montreal impact": "Montreal Impact",
    "san jose earthquakes": "SJ Earthquakes",
    "sj earthquakes": "SJ Earthquakes",
}


EVENT_CONFIG = {
    "shots": {
        "label": "Shots",
        "base_line": "SHOTS_1_5",
        "strong_line": "SHOTS_2_5",
        "strong_cut": 78.0,
        "positions": {"forward", "winger", "attacking_midfielder", "central_midfielder"},
        "weights": {
            "shot_threat": 0.42,
            "goal_threat": 0.24,
            "xg_threat": 0.18,
            "team_attack_context": 0.16,
        },
    },
    "shots_on_target": {
        "label": "Shots On Target",
        "base_line": "SOT_0_5",
        "strong_line": "SOT_1_5",
        "strong_cut": 82.0,
        "positions": {"forward", "winger", "attacking_midfielder"},
        "weights": {
            "shot_threat": 0.30,
            "goal_threat": 0.30,
            "xg_threat": 0.25,
            "team_attack_context": 0.15,
        },
    },
    "key_passes": {
        "label": "Key Passes",
        "base_line": "KEY_PASSES_0_5",
        "strong_line": "KEY_PASSES_1_5",
        "strong_cut": 76.0,
        "positions": {"forward", "winger", "attacking_midfielder", "central_midfielder", "full_back"},
        "weights": {
            "creative_spark": 0.38,
            "xa_threat": 0.30,
            "ball_progression": 0.16,
            "team_attack_context": 0.16,
        },
    },
    "tackles": {
        "label": "Tackles",
        "base_line": "TACKLES_0_5",
        "strong_line": "TACKLES_1_5",
        "strong_cut": 68.0,
        "positions": {"central_midfielder", "defensive_midfielder", "full_back", "centre_back", "defender", "midfielder"},
        "weights": {
            "defensive_lock": 0.46,
            "pressing_heat": 0.28,
            "midfield_engine": 0.12,
            "opponent_attack_context": 0.14,
        },
    },
    "fouls_committed": {
        "label": "Fouls Committed",
        "base_line": "FOULS_0_5",
        "strong_line": "FOULS_1_5",
        "strong_cut": 70.0,
        "positions": {"central_midfielder", "defensive_midfielder", "full_back", "centre_back", "defender", "midfielder"},
        "weights": {
            "pressing_heat": 0.32,
            "discipline_risk": 0.28,
            "defensive_lock": 0.20,
            "opponent_attack_context": 0.20,
        },
    },
    "player_fouled": {
        "label": "Player Fouled",
        "base_line": "FOULED_0_5",
        "strong_line": "FOULED_1_5",
        "strong_cut": 72.0,
        "positions": {"forward", "winger", "attacking_midfielder", "central_midfielder"},
        "weights": {
            "ball_progression": 0.34,
            "creative_spark": 0.28,
            "goal_threat": 0.12,
            "opponent_defensive_context": 0.26,
        },
    },
    "bookings": {
        "label": "Bookings",
        "base_line": "BOOKING",
        "strong_line": "BOOKING",
        "strong_cut": 74.0,
        "positions": {"central_midfielder", "defensive_midfielder", "full_back", "centre_back", "defender", "midfielder"},
        "weights": {
            "booking_heat": 0.44,
            "discipline_risk": 0.28,
            "pressing_heat": 0.10,
            "fixture_chaos_context": 0.18,
        },
    },
    "goalkeeper_saves": {
        "label": "Goalkeeper Saves",
        "base_line": "GK_SAVES_1_5",
        "strong_line": "GK_SAVES_2_5",
        "strong_cut": 76.0,
        "positions": {"goalkeeper"},
        "weights": {
            "goalkeeper_shield": 0.10,
            "opponent_attack_context": 0.48,
            "opponent_goal_heat": 0.32,
            "team_defensive_exposure": 0.10,
        },
    },
}


def norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_team(value: Any) -> str:
    text = norm(value)
    return TEAM_ALIASES.get(text, str(value or "").strip())


def metric(row: pd.Series | dict[str, Any], key: str, default: float = 50.0) -> float:
    try:
        value = row.get(key, default)
    except AttributeError:
        value = default
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def load_player_ratings(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    by_team: dict[str, list[dict[str, Any]]] = {}
    for club in data:
        if club.get("competition_key") != "usa_mls" or str(club.get("season")) not in {"2026", "2025/2026"}:
            continue
        team = canonical_team(club.get("club"))
        by_team[team] = club.get("players", [])
    return by_team


def load_fixture_context(rows_path: Path, fixture_cards_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(rows_path)
    cards = pd.read_csv(fixture_cards_path)
    context = rows.drop_duplicates("fixture_key").copy()
    keep_cols = [
        "fixture_key",
        "provider_fixture_id",
        "provider_kickoff_ts_utc",
        "referee_name",
        "home_injuries",
        "away_injuries",
    ]
    cards_keep = cards[[c for c in keep_cols if c in cards.columns]].drop_duplicates("fixture_key")
    context = context.drop(columns=[c for c in cards_keep.columns if c in context.columns and c != "fixture_key"])
    context = context.merge(cards_keep, on="fixture_key", how="left")
    context["home_team"] = context["home_team"].map(canonical_team)
    context["away_team"] = context["away_team"].map(canonical_team)
    return context


def injury_match_key(player_name: str) -> tuple[str, str]:
    clean = norm(player_name)
    parts = clean.split()
    if not parts:
        return "", ""
    return parts[-1], parts[0][0]


def load_injury_flags(path: Path, fixture_context: pd.DataFrame) -> dict[tuple[str, str], bool]:
    if not path.exists():
        return {}
    injuries = pd.read_csv(path)
    fixture_to_teams = {
        int(row.provider_fixture_id): (row.home_team, row.away_team)
        for row in fixture_context.itertuples()
        if not pd.isna(row.provider_fixture_id)
    }
    flags: dict[tuple[str, str], bool] = {}
    for row in injuries.itertuples():
        teams = fixture_to_teams.get(int(row.fixture_id))
        if not teams:
            continue
        surname, initial = injury_match_key(row.player_name)
        if not surname:
            continue
        for team in teams:
            flags[(team, f"{initial}:{surname}")] = True
    return flags


def load_confirmed_starters(
    lineups_path: Path | None,
    fixture_context: pd.DataFrame,
    fixture_master_path: Path | None = None,
) -> dict[tuple[str, str], dict[str, set[str]]]:
    if not lineups_path or not lineups_path.exists():
        return {}
    lineups = pd.read_csv(lineups_path)
    if lineups.empty:
        return {}
    team_id_to_name: dict[tuple[int, int], str] = {}
    if fixture_master_path and fixture_master_path.exists():
        fixtures = pd.read_csv(fixture_master_path)
        for fx in fixtures.itertuples():
            team_id_to_name[(int(fx.fixture_id), int(fx.home_team_id))] = canonical_team(fx.home_team_name)
            team_id_to_name[(int(fx.fixture_id), int(fx.away_team_id))] = canonical_team(fx.away_team_name)
    fixture_teams = {}
    for row in fixture_context.itertuples():
        if pd.isna(row.provider_fixture_id):
            continue
        fixture_teams[int(row.provider_fixture_id)] = {row.home_team, row.away_team}

    starter_rows = lineups[lineups["is_starting_xi"].eq(1)].copy()
    starters: dict[tuple[str, str], dict[str, set[str]]] = {}
    for (fixture_id, team_id), group in starter_rows.groupby(["fixture_id", "team_id"]):
        team_name = team_id_to_name.get((int(fixture_id), int(team_id)))
        teams = {team_name} if team_name else fixture_teams.get(int(fixture_id))
        if not teams:
            continue
        names = [str(name) for name in group["player_name"].dropna().tolist()]
        full = {norm(name) for name in names if norm(name)}
        surnames = [injury_match_key(name)[0] for name in names if injury_match_key(name)[0]]
        surname_counts = defaultdict(int)
        for surname in surnames:
            surname_counts[surname] += 1
        unique_surnames = {surname for surname, count in surname_counts.items() if count == 1}
        for team in teams:
            key = (str(fixture_id), team)
            existing = starters.setdefault(key, {"full": set(), "surname": set()})
            existing["full"].update(full)
            existing["surname"].update(unique_surnames)
    return starters


def is_confirmed_starter(
    fixture_id: Any,
    team: str,
    player: dict[str, Any],
    confirmed_starters: dict[tuple[str, str], dict[str, set[str]]],
) -> bool:
    if not confirmed_starters:
        return False
    keys = confirmed_starters.get((str(int(fixture_id)), team))
    if not keys:
        return False
    player_name = str(player.get("name") or "")
    player_full = norm(player_name)
    player_surname, _initial = injury_match_key(player_name)
    return player_full in keys["full"] or bool(player_surname and player_surname in keys["surname"])


def player_is_flagged_injured(team: str, player: dict[str, Any], injury_flags: dict[tuple[str, str], bool]) -> bool:
    surname, initial = injury_match_key(player.get("name", ""))
    return injury_flags.get((team, f"{initial}:{surname}"), False)


def team_context(row: pd.Series, side: str) -> dict[str, float]:
    opp = "away" if side == "home" else "home"
    return {
        "team_attack_context": (metric(row, f"{side}_attack_flow") + metric(row, f"{side}_goal_heat")) / 2,
        "opponent_attack_context": (metric(row, f"{opp}_attack_flow") + metric(row, f"{opp}_goal_heat")) / 2,
        "opponent_goal_heat": metric(row, f"{opp}_goal_heat"),
        "opponent_defensive_context": (metric(row, f"{opp}_defensive_lock") + metric(row, f"{opp}_chaos")) / 2,
        "fixture_chaos_context": (metric(row, "home_chaos") + metric(row, "away_chaos")) / 2,
        "team_defensive_exposure": 100 - metric(row, f"{side}_defensive_lock"),
    }


def event_score(player: dict[str, Any], config: dict[str, Any], context: dict[str, float]) -> float:
    ratings = player.get("ratings", {})
    score = 0.0
    for key, weight in config["weights"].items():
        if key in context:
            value = context[key]
        else:
            value = float(ratings.get(key, 50))
        score += value * weight
    minutes = player.get("minutes_confidence", {})
    minutes_played = float(minutes.get("minutes_played") or 0)
    if minutes_played < 450:
        score -= 7.5
    elif minutes_played < 900:
        score -= 3.0
    return max(0.0, min(100.0, score))


def bucket(score: float) -> str:
    if score >= 72:
        return "BETA_STRONG"
    if score >= 64:
        return "BETA_WATCH"
    return "BETA_DEPTH"


def line_hint(score: float, config: dict[str, Any]) -> str:
    if score >= float(config["strong_cut"]):
        return str(config["strong_line"])
    return str(config["base_line"])


def reason(player: dict[str, Any], event_key: str, score: float, row: pd.Series, side: str) -> str:
    ratings = player.get("ratings", {})
    if event_key in {"shots", "shots_on_target"}:
        return (
            f"Shot {ratings.get('shot_threat')} / goal {ratings.get('goal_threat')} profile with "
            f"{side} attack context {metric(row, f'{side}_attack_flow'):.0f}."
        )
    if event_key == "key_passes":
        return (
            f"Creative {ratings.get('creative_spark')} / xA {ratings.get('xa_threat')} profile against "
            f"opponent defensive context {metric(row, ('away' if side == 'home' else 'home') + '_defensive_lock'):.0f}."
        )
    if event_key in {"tackles", "fouls_committed"}:
        opp = "away" if side == "home" else "home"
        return (
            f"Defensive {ratings.get('defensive_lock')} / press {ratings.get('pressing_heat')} profile with "
            f"opponent attack flow {metric(row, f'{opp}_attack_flow'):.0f}."
        )
    if event_key == "player_fouled":
        return (
            f"Progression {ratings.get('ball_progression')} / creative {ratings.get('creative_spark')} profile, "
            "use only if starter role confirms."
        )
    if event_key == "bookings":
        return (
            f"Booking heat {ratings.get('booking_heat')} / discipline risk {ratings.get('discipline_risk')} "
            f"inside fixture chaos {(metric(row, 'home_chaos') + metric(row, 'away_chaos')) / 2:.0f}."
        )
    if event_key == "goalkeeper_saves":
        opp = "away" if side == "home" else "home"
        return (
            f"Keeper profile plus opponent attack flow {metric(row, f'{opp}_attack_flow'):.0f} "
            f"and goal heat {metric(row, f'{opp}_goal_heat'):.0f}."
        )
    return f"Beta score {score:.1f}; confirm role before use."


def build_rows(
    fixture_context: pd.DataFrame,
    players_by_team: dict[str, list[dict[str, Any]]],
    injury_flags: dict[tuple[str, str], bool],
    generated_at: str,
    confirmed_starters: dict[tuple[str, str], dict[str, set[str]]] | None = None,
    confirmed_starters_only: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fixture in fixture_context.itertuples(index=False):
        row = pd.Series(fixture._asdict())
        kickoff = str(row.get("provider_kickoff_ts_utc") or "")
        kickoff_dt = pd.to_datetime(kickoff, errors="coerce", utc=True)
        generated_dt = pd.to_datetime(generated_at, utc=True)
        pre_kickoff = bool(not pd.isna(kickoff_dt) and generated_dt < kickoff_dt)
        for side, team in (("home", row["home_team"]), ("away", row["away_team"])):
            players = players_by_team.get(team, [])
            context = team_context(row, side)
            for event_key, config in EVENT_CONFIG.items():
                candidates = []
                for player in players:
                    starter_confirmed = is_confirmed_starter(
                        row.get("provider_fixture_id"),
                        team,
                        player,
                        confirmed_starters or {},
                    )
                    if confirmed_starters_only and confirmed_starters and not starter_confirmed:
                        continue
                    pos_group = str(player.get("position_group") or "").lower()
                    pos = str(player.get("position") or "").lower()
                    if pos_group not in config["positions"] and pos not in config["positions"]:
                        continue
                    if player_is_flagged_injured(team, player, injury_flags):
                        continue
                    score = event_score(player, config, context)
                    if score < 58:
                        continue
                    candidates.append((score, player))
                candidates.sort(key=lambda item: item[0], reverse=True)
                max_rows = 2 if event_key != "goalkeeper_saves" else 1
                for rank, (score, player) in enumerate(candidates[:max_rows], start=1):
                    ratings = player.get("ratings", {})
                    out.append(
                        {
                            "fixture_key": row["fixture_key"],
                            "match_date": row["match_date"],
                            "fixture_kickoff_at": kickoff,
                            "capture_generated_at": generated_at,
                            "source_data_cutoff_at": generated_at,
                            "pre_kickoff_eligible": int(pre_kickoff),
                            "snapshot_phase": "pre_kickoff_beta",
                            "starter_confirmed": int(starter_confirmed),
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "team": team,
                            "side": side,
                            "player_key": player.get("player_id"),
                            "player_name": player.get("name"),
                            "position": player.get("position"),
                            "position_group": player.get("position_group"),
                            "minutes_label": player.get("minutes_confidence", {}).get("label"),
                            "minutes_played": player.get("minutes_confidence", {}).get("minutes_played"),
                            "event_key": event_key,
                            "event_label": config["label"],
                            "line_hint": line_hint(score, config),
                            "beta_score": round(score, 2),
                            "beta_bucket": bucket(score),
                            "event_rank_in_team": rank,
                            "og_player_power": ratings.get("og_player_power"),
                            "goal_threat": ratings.get("goal_threat"),
                            "shot_threat": ratings.get("shot_threat"),
                            "xg_threat": ratings.get("xg_threat"),
                            "creative_spark": ratings.get("creative_spark"),
                            "xa_threat": ratings.get("xa_threat"),
                            "defensive_lock": ratings.get("defensive_lock"),
                            "pressing_heat": ratings.get("pressing_heat"),
                            "discipline_risk": ratings.get("discipline_risk"),
                            "booking_heat": ratings.get("booking_heat"),
                            "reason": reason(player, event_key, score, row, side),
                            "review_status": "MANUAL_REVIEW_ONLY",
                            "not_deployable_flag": 1,
                        }
                    )
    out.sort(key=lambda r: (r["fixture_key"], r["event_key"], -float(r["beta_score"])))
    return out


def write_outputs(rows: list[dict[str, Any]], outdir: Path, confirmed_starters_only: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "live_mls_player_event_beta_rows.csv"
    review_rows_path = outdir / "live_mls_player_event_beta_review_rows.csv"
    if rows:
        with rows_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        review_rows = [row for row in rows if row["beta_bucket"] in {"BETA_STRONG", "BETA_WATCH"}]
        with review_rows_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(review_rows)
    else:
        rows_path.write_text("")
        review_rows_path.write_text("")

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fixture: dict[str, int] = defaultdict(int)
    for row in rows:
        by_event[row["event_key"]].append(row)
        by_fixture[row["fixture_key"]] += 1

    summary = {
        "rows": len(rows),
        "review_rows": sum(1 for row in rows if row["beta_bucket"] in {"BETA_STRONG", "BETA_WATCH"}),
        "fixtures": len(by_fixture),
        "confirmed_starters_only": bool(confirmed_starters_only),
        "strong_rows": sum(1 for row in rows if row["beta_bucket"] == "BETA_STRONG"),
        "watch_rows": sum(1 for row in rows if row["beta_bucket"] == "BETA_WATCH"),
        "events": {event: len(event_rows) for event, event_rows in sorted(by_event.items())},
        "top_20": sorted(rows, key=lambda r: float(r["beta_score"]), reverse=True)[:20],
        "warning": "Beta shortlist only. No priced player-prop odds. Confirm lineups before any use.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Live MLS Player Event Beta Shortlist",
        "",
        "Beta/manual-review output only. These rows are not deployable picks and are not priced player-prop odds.",
        "",
        f"- Rows: {summary['rows']}",
        f"- Review rows: {summary['review_rows']}",
        f"- Fixtures: {summary['fixtures']}",
        f"- BETA_STRONG rows: {summary['strong_rows']}",
        f"- BETA_WATCH rows: {summary['watch_rows']}",
        "",
        "## Event Coverage",
    ]
    for event, count in summary["events"].items():
        lines.append(f"- {event}: {count}")
    lines.extend(["", "## Top Manual Review Rows"])
    for row in summary["top_20"]:
        lines.append(
            f"- {row['fixture_key']} | {row['event_label']} {row['line_hint']} | "
            f"{row['player_name']} ({row['team']}) | {row['beta_bucket']} {row['beta_score']} | {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Operating Notes",
            (
                "- Confirmed-starter filter is active; rows are restricted to players matched into the provider starting XIs."
                if confirmed_starters_only
                else "- Confirmed lineups were not used; rows are pre-lineup watchlist candidates."
            ),
            "- Match player-stat feeds were not used for projection; use actuals only for later audit.",
            "- Injury flags are used as exclusion hints where provider names match the player rating identities.",
            "- Re-run after confirmed lineups to remove non-starters and sharpen role confidence.",
        ]
    )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-ratings", type=Path, default=DEFAULT_PLAYER_RATINGS)
    parser.add_argument("--model-intel-rows", type=Path, default=DEFAULT_MODEL_INTEL_ROWS)
    parser.add_argument("--fixture-cards", type=Path, default=DEFAULT_FIXTURE_CARDS)
    parser.add_argument("--injuries", type=Path, default=DEFAULT_INJURIES)
    parser.add_argument("--lineups", type=Path, default=DEFAULT_LINEUPS)
    parser.add_argument("--fixture-master", type=Path, default=DEFAULT_FIXTURE_MASTER)
    parser.add_argument("--kickoff-ts", default="")
    parser.add_argument("--confirmed-starters-only", action="store_true")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    players_by_team = load_player_ratings(args.player_ratings)
    fixture_context = load_fixture_context(args.model_intel_rows, args.fixture_cards)
    if args.kickoff_ts:
        fixture_context = fixture_context[fixture_context["provider_kickoff_ts_utc"].eq(args.kickoff_ts)].copy()
    injury_flags = load_injury_flags(args.injuries, fixture_context)
    confirmed_starters = load_confirmed_starters(args.lineups, fixture_context, args.fixture_master)
    rows = build_rows(
        fixture_context,
        players_by_team,
        injury_flags,
        generated_at,
        confirmed_starters=confirmed_starters,
        confirmed_starters_only=args.confirmed_starters_only,
    )
    write_outputs(rows, args.outdir, confirmed_starters_only=args.confirmed_starters_only)
    print(json.dumps({"rows": len(rows), "outdir": str(args.outdir)}, indent=2))


if __name__ == "__main__":
    main()
