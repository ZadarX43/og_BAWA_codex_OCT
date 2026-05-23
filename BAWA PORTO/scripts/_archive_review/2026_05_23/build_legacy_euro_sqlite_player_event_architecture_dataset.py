from __future__ import annotations

import argparse
import math
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SQLITE = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/database.sqlite")
DEFAULT_OUTDIR = Path("data_sources/legacy_euro_sqlite/player_event_architecture")
DEFAULT_LEAGUES = [
    "England Premier League",
    "Spain LIGA BBVA",
    "Italy Serie A",
    "Germany 1. Bundesliga",
    "France Ligue 1",
]


PLAYER_ATTR_COLS = [
    "overall_rating",
    "potential",
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]

TEAM_ATTR_COLS = [
    "buildUpPlaySpeed",
    "buildUpPlayDribbling",
    "buildUpPlayPassing",
    "chanceCreationPassing",
    "chanceCreationCrossing",
    "chanceCreationShooting",
    "defencePressure",
    "defenceAggression",
    "defenceTeamWidth",
]

BOOKS = ["B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS"]


def safe_int(value: Any) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return int(value)
    except Exception:
        return 0


def parse_xml_values(xml_text: Any) -> list[dict[str, Any]]:
    if xml_text is None or (isinstance(xml_text, float) and math.isnan(xml_text)):
        return []
    text = str(xml_text).strip()
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    values = []
    for value in root.findall("value"):
        stats = {}
        stats_node = value.find("stats")
        if stats_node is not None:
            for child in list(stats_node):
                stats[child.tag] = safe_int(child.text)
        row = {
            "event_id": safe_int(value.findtext("id")),
            "elapsed": safe_int(value.findtext("elapsed")),
            "team_api_id": safe_int(value.findtext("team")),
            "player1_api_id": safe_int(value.findtext("player1")),
            "player2_api_id": safe_int(value.findtext("player2")),
            "type": value.findtext("type") or "",
            "subtype": value.findtext("subtype") or "",
            "comment": value.findtext("comment") or "",
            "card_type": value.findtext("card_type") or "",
            "stats": stats,
        }
        values.append(row)
    return values


def implied_prob(odds: float | int | None) -> float:
    try:
        odds_f = float(odds)
    except Exception:
        return 0.0
    if odds_f <= 1.0:
        return 0.0
    return 1.0 / odds_f


def consensus_1x2(row: pd.Series) -> dict[str, float]:
    probs_h = []
    probs_d = []
    probs_a = []
    for book in BOOKS:
        h = implied_prob(row.get(f"{book}H"))
        d = implied_prob(row.get(f"{book}D"))
        a = implied_prob(row.get(f"{book}A"))
        if h and d and a:
            total = h + d + a
            if total > 0:
                probs_h.append(h / total)
                probs_d.append(d / total)
                probs_a.append(a / total)
    return {
        "book_count_1x2": len(probs_h),
        "consensus_home_prob": sum(probs_h) / len(probs_h) if probs_h else 0.0,
        "consensus_draw_prob": sum(probs_d) / len(probs_d) if probs_d else 0.0,
        "consensus_away_prob": sum(probs_a) / len(probs_a) if probs_a else 0.0,
    }


def side_prob(consensus: dict[str, float], side: str) -> float:
    return consensus["consensus_home_prob"] if side == "home" else consensus["consensus_away_prob"]


def position_bucket(slot: int, x: int, y: int) -> str:
    # In this legacy DB the Y line is the better depth proxy:
    # GK ~= 1, defensive line ~= 3, midfield ~= 5-7, forwards ~= 9+.
    if slot == 1 or y <= 1:
        return "GK"
    if y <= 3:
        return "DEF"
    if y <= 7:
        return "MID"
    return "FWD"


def load_matches(conn: sqlite3.Connection, leagues: list[str], limit_matches: int | None) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(leagues))
    limit_clause = f"LIMIT {int(limit_matches)}" if limit_matches else ""
    sql = f"""
        SELECT
            c.name AS country,
            l.name AS league_name,
            m.*
        FROM Match m
        JOIN League l ON m.league_id = l.id
        JOIN Country c ON m.country_id = c.id
        WHERE l.name IN ({placeholders})
          AND m.shoton IS NOT NULL
          AND length(m.shoton) > 0
        ORDER BY m.date, m.id
        {limit_clause}
    """
    return pd.read_sql_query(sql, conn, params=leagues, parse_dates=["date"])


def build_player_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, match in matches.iterrows():
        event_counts: dict[int, Counter] = defaultdict(Counter)
        player_team: dict[int, int] = {}
        consensus = consensus_1x2(match)
        for side in ["home", "away"]:
            team_id = safe_int(match[f"{side}_team_api_id"])
            opp_team_id = safe_int(match["away_team_api_id" if side == "home" else "home_team_api_id"])
            for slot in range(1, 12):
                player_id = safe_int(match.get(f"{side}_player_{slot}"))
                if not player_id:
                    continue
                x = safe_int(match.get(f"{side}_player_X{slot}"))
                y = safe_int(match.get(f"{side}_player_Y{slot}"))
                player_team[player_id] = team_id
                rows.append(
                    {
                        "match_id": safe_int(match["id"]),
                        "match_api_id": safe_int(match["match_api_id"]),
                        "country": match["country"],
                        "league_name": match["league_name"],
                        "season": match["season"],
                        "match_date": match["date"],
                        "team_api_id": team_id,
                        "opponent_team_api_id": opp_team_id,
                        "player_api_id": player_id,
                        "side": side,
                        "slot": slot,
                        "x_coord": x,
                        "y_coord": y,
                        "position_bucket": position_bucket(slot, x, y),
                        "is_home": 1 if side == "home" else 0,
                        "team_goals_for": safe_int(match[f"{side}_team_goal"]),
                        "team_goals_against": safe_int(match["away_team_goal" if side == "home" else "home_team_goal"]),
                        "book_count_1x2": consensus["book_count_1x2"],
                        "team_consensus_win_prob": side_prob(consensus, side),
                        "opponent_consensus_win_prob": side_prob(consensus, "away" if side == "home" else "home"),
                        "consensus_draw_prob": consensus["consensus_draw_prob"],
                    }
                )

        # Goals count as a shot and a shot on target for the scoring player, unless
        # the old XML marks the event as an own goal.
        for event in parse_xml_values(match.get("goal")):
            player = event["player1_api_id"]
            if player and event.get("comment") != "o":
                event_counts[player]["goals"] += 1
                event_counts[player]["shots_total"] += 1
                event_counts[player]["shots_on_target"] += 1

        for event in parse_xml_values(match.get("shoton")):
            player = event["player1_api_id"]
            if not player:
                continue
            event_counts[player]["shots_total"] += 1
            if safe_int((event.get("stats") or {}).get("shoton")) > 0:
                event_counts[player]["shots_on_target"] += 1
            if safe_int((event.get("stats") or {}).get("blocked")) > 0:
                event_counts[player]["shots_blocked"] += 1

        for event in parse_xml_values(match.get("shotoff")):
            player = event["player1_api_id"]
            if player:
                event_counts[player]["shots_total"] += 1
                event_counts[player]["shots_off_target"] += 1

        for event in parse_xml_values(match.get("foulcommit")):
            committed_by = event["player1_api_id"]
            drawn_by = event["player2_api_id"]
            if committed_by:
                event_counts[committed_by]["fouls_committed"] += 1
            if drawn_by:
                event_counts[drawn_by]["fouls_drawn"] += 1

        for event in parse_xml_values(match.get("card")):
            player = event["player1_api_id"]
            if not player:
                continue
            if event.get("card_type") == "r" or safe_int((event.get("stats") or {}).get("rcards")) > 0:
                event_counts[player]["red_cards"] += 1
            if event.get("card_type") == "y" or safe_int((event.get("stats") or {}).get("ycards")) > 0:
                event_counts[player]["yellow_cards"] += 1

        # Attach counts to the current match rows only.
        start_idx = len(rows) - len(player_team)
        for idx in range(start_idx, len(rows)):
            counts = event_counts.get(rows[idx]["player_api_id"], Counter())
            for col in [
                "goals",
                "shots_total",
                "shots_on_target",
                "shots_off_target",
                "shots_blocked",
                "fouls_committed",
                "fouls_drawn",
                "yellow_cards",
                "red_cards",
            ]:
                rows[idx][col] = int(counts.get(col, 0))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["card_any"] = (out["yellow_cards"].fillna(0) + out["red_cards"].fillna(0)).gt(0).astype(int)
    out["shots_ge1"] = out["shots_total"].ge(1).astype(int)
    out["shots_ge2"] = out["shots_total"].ge(2).astype(int)
    out["sot_ge1"] = out["shots_on_target"].ge(1).astype(int)
    out["fouls_committed_ge1"] = out["fouls_committed"].ge(1).astype(int)
    out["fouls_committed_ge2"] = out["fouls_committed"].ge(2).astype(int)
    out["fouls_drawn_ge1"] = out["fouls_drawn"].ge(1).astype(int)
    out["fouls_drawn_ge2"] = out["fouls_drawn"].ge(2).astype(int)
    return out


def merge_names(conn: sqlite3.Connection, rows: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_sql_query("SELECT player_api_id, player_name, height, weight FROM Player", conn)
    teams = pd.read_sql_query("SELECT team_api_id, team_long_name, team_short_name FROM Team", conn)
    out = rows.merge(players, on="player_api_id", how="left")
    out = out.merge(teams.rename(columns={"team_api_id": "team_api_id", "team_long_name": "team_name"}), on="team_api_id", how="left")
    out = out.merge(
        teams.rename(
            columns={
                "team_api_id": "opponent_team_api_id",
                "team_long_name": "opponent_team_name",
                "team_short_name": "opponent_team_short_name",
            }
        ),
        on="opponent_team_api_id",
        how="left",
    )
    return out


def merge_asof_group(left: pd.DataFrame, right: pd.DataFrame, by: str, left_date: str, right_date: str) -> pd.DataFrame:
    parts = []
    for key, group in left.groupby(by, sort=False):
        attrs = right[right[by].eq(key)].sort_values(right_date)
        if attrs.empty:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values(left_date),
            attrs,
            left_on=left_date,
            right_on=right_date,
            direction="backward",
            suffixes=("", "_attr"),
        )
        if f"{by}_attr" in merged.columns:
            merged = merged.drop(columns=[f"{by}_attr"])
        parts.append(merged)
    if not parts:
        return left
    return pd.concat(parts, ignore_index=True)


def merge_attributes(conn: sqlite3.Connection, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["match_date"] = pd.to_datetime(out["match_date"])

    player_attrs = pd.read_sql_query(
        "SELECT player_api_id, date, " + ", ".join(PLAYER_ATTR_COLS) + " FROM Player_Attributes",
        conn,
        parse_dates=["date"],
    ).dropna(subset=["player_api_id", "date"])
    player_attrs = player_attrs.rename(columns={"date": "player_attr_date"})
    out = merge_asof_group(out, player_attrs, "player_api_id", "match_date", "player_attr_date")

    team_attrs = pd.read_sql_query(
        "SELECT team_api_id, date, " + ", ".join(TEAM_ATTR_COLS) + " FROM Team_Attributes",
        conn,
        parse_dates=["date"],
    ).dropna(subset=["team_api_id", "date"])
    team_attrs = team_attrs.rename(columns={"date": "team_attr_date"})
    out = merge_asof_group(out, team_attrs, "team_api_id", "match_date", "team_attr_date")

    opp_attrs = team_attrs.rename(
        columns={
            "team_api_id": "opponent_team_api_id",
            "team_attr_date": "opponent_team_attr_date",
            **{col: f"opp_{col}" for col in TEAM_ATTR_COLS},
        }
    )
    out = merge_asof_group(out, opp_attrs, "opponent_team_api_id", "match_date", "opponent_team_attr_date")
    return out


def write_audit(outdir: Path, df: pd.DataFrame, leagues: list[str], sqlite_path: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in [
        "shots_ge1",
        "shots_ge2",
        "sot_ge1",
        "fouls_committed_ge1",
        "fouls_committed_ge2",
        "fouls_drawn_ge1",
        "fouls_drawn_ge2",
        "card_any",
    ]:
        rows.append(
            {
                "market": market,
                "rows": len(df),
                "hit_rate": round(float(df[market].mean()), 4) if len(df) else 0.0,
                "hits": int(df[market].sum()) if len(df) else 0,
            }
        )
    market_summary = pd.DataFrame(rows)
    league_summary = (
        df.groupby("league_name", dropna=False)
        .agg(
            rows=("match_id", "size"),
            matches=("match_id", "nunique"),
            shots_ge1=("shots_ge1", "mean"),
            sot_ge1=("sot_ge1", "mean"),
            fouls_committed_ge1=("fouls_committed_ge1", "mean"),
            fouls_drawn_ge1=("fouls_drawn_ge1", "mean"),
            card_any=("card_any", "mean"),
        )
        .reset_index()
    )
    role_summary = (
        df.groupby("position_bucket", dropna=False)
        .agg(
            rows=("match_id", "size"),
            shots_ge1=("shots_ge1", "mean"),
            sot_ge1=("sot_ge1", "mean"),
            fouls_committed_ge1=("fouls_committed_ge1", "mean"),
            fouls_drawn_ge1=("fouls_drawn_ge1", "mean"),
            card_any=("card_any", "mean"),
            avg_aggression=("aggression", "mean"),
            avg_finishing=("finishing", "mean"),
            avg_long_shots=("long_shots", "mean"),
        )
        .reset_index()
    )
    market_summary.to_csv(outdir / "legacy_euro_player_event_market_summary.csv", index=False)
    league_summary.to_csv(outdir / "legacy_euro_player_event_league_summary.csv", index=False)
    role_summary.to_csv(outdir / "legacy_euro_player_event_role_summary.csv", index=False)

    report = [
        "# Legacy Euro SQLite Player-Event Architecture Dataset",
        "",
        f"Source: `{sqlite_path}`",
        "",
        "This is a research-only dataset for testing player-event feature architecture. It is not a live deploy source.",
        "",
        "## Build Scope",
        "",
        f"- Leagues: {', '.join(leagues)}",
        f"- Player-match rows: {len(df):,}",
        f"- Matches: {df['match_id'].nunique():,}" if len(df) else "- Matches: 0",
        f"- Seasons: {df['season'].nunique():,}" if len(df) else "- Seasons: 0",
        "",
        "## Market Hit Rates",
        "",
        market_summary.to_csv(index=False),
        "",
        "## Key Feature Families",
        "",
        "- Player attributes: aggression, finishing, long shots, positioning, marking, standing tackle, stamina, speed.",
        "- Team attributes: chance creation, crossing, shooting, defensive pressure, defensive aggression.",
        "- Market priors: normalized 1X2 bookmaker consensus where available.",
        "- Role features: lineup slot and X/Y coordinate buckets.",
        "",
        "## Caveats",
        "",
        "- This data is 2008-2016, so use it for feature-shape validation only.",
        "- Tackle and keeper-save markets are not directly available.",
        "- Fouls drawn are derived from `foulcommit.player2`; this needs QA before model conclusions.",
        "- Closing odds are benchmark/market-prior research only, not early pre-match features.",
    ]
    (outdir / "LEGACY_EURO_PLAYER_EVENT_ARCHITECTURE_DATASET.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only player-event architecture dataset from the legacy Euro SQLite database.")
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    parser.add_argument("--limit-matches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leagues = [x.strip() for x in str(args.leagues).split(",") if x.strip()]
    conn = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    try:
        matches = load_matches(conn, leagues, args.limit_matches or None)
        rows = build_player_match_rows(matches)
        rows = merge_names(conn, rows)
        rows = merge_attributes(conn, rows)
    finally:
        conn.close()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "legacy_euro_player_event_architecture_dataset.csv"
    rows.to_csv(out_csv, index=False)
    write_audit(args.outdir, rows, leagues, args.sqlite)
    print(f"[ok] matches={rows['match_id'].nunique() if len(rows) else 0} rows={len(rows)}")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {args.outdir / 'LEGACY_EURO_PLAYER_EVENT_ARCHITECTURE_DATASET.md'}")


if __name__ == "__main__":
    main()
