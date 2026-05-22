#!/usr/bin/env python3
"""Build a confirmed-lineup player-event shortlist from the live shadow board.

Research/demo only. This filters existing player-event shadow rows to players
who are present in API-Football confirmed starting XIs. It does not price odds,
promote deploy tiers, or alter production routing.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_EXPRESSIONS = (
    "Player ",
    "Keeper Saves",
    "Team Most Cards",
    "Total Shots",
    "Total SOT",
)

MARKET_RANK = {
    "Player Fouled 0.5+": 1,
    "Player Tackles 1.5+": 2,
    "Player Tackles 2.5+": 3,
    "Player Shots 1.5+": 4,
    "Player Shots 2.5+": 5,
    "Player SOT 0.5+": 6,
    "Player SOT 1.5+": 7,
    "Keeper Saves 1.5+": 8,
    "Keeper Saves 2.5+": 9,
    "Keeper Saves 3.5+": 10,
    "Player key passes 0.5+": 11,
    "Player key passes 1.5+": 12,
    "Player assist watch 0.5+": 13,
}

EXPECTED_MARKET_FAMILIES = [
    "PLAYER_SHOTS",
    "PLAYER_SOT",
    "PLAYER_FOULED",
    "PLAYER_FOULS_COMMITTED",
    "PLAYER_TACKLES",
    "PLAYER_CARDS_BOOKINGS",
    "KEEPER_SAVES",
    "PLAYER_KEY_PASSES",
    "PLAYER_ASSIST_WATCH",
    "TEAM_CARDS",
    "TOTAL_SHOTS",
    "TOTAL_SOT",
]

LEAGUE_ALIASES = {
    "Premier League": "England Premier League",
    "Bundesliga": "Germany Bundesliga",
}

TEAM_ALIASES = {
    "1899 hoffenheim": "hoffenheim",
    "afc bournemouth": "bournemouth",
    "bayern m nchen": "bayern munchen",
    "bayern munich": "bayern munchen",
    "borussia m gladbach": "borussia monchengladbach",
    "borussia mönchengladbach": "borussia monchengladbach",
    "brighton hove albion": "brighton",
    "fc augsburg": "augsburg",
    "fc st pauli": "st pauli",
    "rb leipzig": "rb leipzig",
    "vfb stuttgart": "stuttgart",
    "vfl wolfsburg": "wolfsburg",
    "werder bremen": "werder bremen",
    "wolverhampton wanderers": "wolves",
}


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def team_key(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"\b(fc|afc|vfb|vfl|1899)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def player_key(value: Any) -> str:
    return norm_text(value)


def name_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.88
    lt = set(left.split())
    rt = set(right.split())
    if not lt or not rt:
        return 0.0
    return len(lt & rt) / max(len(lt), len(rt))


def read_many_csv(paths_csv: str) -> pd.DataFrame:
    if not paths_csv.strip():
        return pd.DataFrame()
    frames = []
    for item in paths_csv.split(","):
        path = Path(item.strip())
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def is_player_event_frame(df: pd.DataFrame) -> pd.Series:
    expression = df.get("expression", pd.Series("", index=df.index)).fillna("").astype(str)
    return expression.map(lambda value: value.startswith(TARGET_EXPRESSIONS))


def market_family(expression: Any) -> str:
    text = str(expression or "").lower()
    if "shots on target" in text or "player sot" in text:
        return "PLAYER_SOT"
    if "player shots" in text:
        return "PLAYER_SHOTS"
    if "player fouled" in text:
        return "PLAYER_FOULED"
    if "fouls committed" in text:
        return "PLAYER_FOULS_COMMITTED"
    if "tackles" in text:
        return "PLAYER_TACKLES"
    if "cards" in text or "booking" in text:
        if text.startswith("team most"):
            return "TEAM_CARDS"
        return "PLAYER_CARDS_BOOKINGS"
    if "keeper saves" in text:
        return "KEEPER_SAVES"
    if "key passes" in text:
        return "PLAYER_KEY_PASSES"
    if "assist watch" in text:
        return "PLAYER_ASSIST_WATCH"
    if text.startswith("total shots"):
        return "TOTAL_SHOTS"
    if text.startswith("total sot"):
        return "TOTAL_SOT"
    return "OTHER"


def build_fixture_matches(board: pd.DataFrame, fixtures: pd.DataFrame, kickoff_times: set[str]) -> pd.DataFrame:
    api = fixtures.copy()
    api["league"] = api["league"].map(lambda value: LEAGUE_ALIASES.get(str(value), str(value)))
    api["match_date"] = pd.to_datetime(api["match_date"], errors="coerce").dt.date.astype(str)
    api["kickoff_local"] = pd.to_datetime(api["kickoff_ts_utc"], errors="coerce").dt.strftime("%H:%M")
    api["home_key"] = api["home_team_name"].map(team_key)
    api["away_key"] = api["away_team_name"].map(team_key)
    if kickoff_times:
        api = api[api["kickoff_local"].isin(kickoff_times)].copy()

    board_fixtures = (
        board[["fixture_key", "match_date", "league", "home_team_name", "away_team_name"]]
        .drop_duplicates("fixture_key")
        .copy()
    )
    board_fixtures["match_date"] = pd.to_datetime(board_fixtures["match_date"], errors="coerce").dt.date.astype(str)
    board_fixtures["home_key"] = board_fixtures["home_team_name"].map(team_key)
    board_fixtures["away_key"] = board_fixtures["away_team_name"].map(team_key)

    rows = []
    for _, b in board_fixtures.iterrows():
        candidates = api[(api["league"].eq(b["league"])) & (api["match_date"].eq(b["match_date"]))].copy()
        best = None
        best_score = 0.0
        for _, a in candidates.iterrows():
            score = 0.5 * name_score(str(b["home_key"]), str(a["home_key"])) + 0.5 * name_score(str(b["away_key"]), str(a["away_key"]))
            if score > best_score:
                best_score = score
                best = a
        rec = b.to_dict()
        if best is not None and best_score >= 0.55:
            rec.update(
                {
                    "api_fixture_id": int(best["fixture_id"]),
                    "api_fixture_key": best["fixture_key"],
                    "api_home_team_name": best["home_team_name"],
                    "api_away_team_name": best["away_team_name"],
                    "api_kickoff_local": best["kickoff_local"],
                    "api_match_score": round(best_score, 4),
                }
            )
        else:
            rec.update(
                {
                    "api_fixture_id": pd.NA,
                    "api_fixture_key": "",
                    "api_home_team_name": "",
                    "api_away_team_name": "",
                    "api_kickoff_local": "",
                    "api_match_score": 0.0,
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = [str(row.get(col, "")).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | showing {max_rows} of {len(df)} rows |" + " |" * max(0, len(columns) - 2))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-csv", required=True)
    parser.add_argument("--fixtures-csvs", required=True, help="Comma-separated API normalized fixtures CSVs.")
    parser.add_argument("--lineups-csvs", required=True, help="Comma-separated API normalized lineups CSVs.")
    parser.add_argument(
        "--referee-overlay-csvs",
        default="",
        help="Optional comma-separated referee fixture overlay CSVs from build_referee_profile_engine.py.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target-leagues", default="England Premier League,Germany Bundesliga")
    parser.add_argument("--target-date", default="2026-05-09")
    parser.add_argument("--kickoff-times", default="14:30,15:00")
    parser.add_argument("--max-md-rows", type=int, default=80)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    board = pd.read_csv(args.dashboard_csv, low_memory=False)
    fixtures = read_many_csv(args.fixtures_csvs)
    lineups = read_many_csv(args.lineups_csvs)
    referee_overlay = read_many_csv(args.referee_overlay_csvs)
    if board.empty or fixtures.empty or lineups.empty:
        raise SystemExit("Dashboard, fixtures, and lineups inputs must all be non-empty.")

    target_leagues = {item.strip() for item in args.target_leagues.split(",") if item.strip()}
    kickoff_times = {item.strip() for item in args.kickoff_times.split(",") if item.strip()}

    board["match_date"] = pd.to_datetime(board["match_date"], errors="coerce").dt.date.astype(str)
    board = board[board["league"].isin(target_leagues) & board["match_date"].eq(args.target_date)].copy()
    player_rows = board[is_player_event_frame(board)].copy()
    player_rows["market_family"] = player_rows["expression"].map(market_family)
    player_rows["requires_player_confirmation_flag"] = player_rows["player_name"].fillna("").astype(str).str.strip().ne("").astype(int)

    fixture_matches = build_fixture_matches(board, fixtures, kickoff_times)

    lineups = lineups.copy()
    lineups["is_starting_xi"] = pd.to_numeric(lineups["is_starting_xi"], errors="coerce").fillna(0).astype(int)
    lineups["player_key"] = lineups["player_name"].map(player_key)
    starter_keys = (
        lineups[lineups["is_starting_xi"].eq(1)]
        .groupby("fixture_id")["player_key"]
        .apply(set)
        .to_dict()
    )
    lineup_counts = lineups.groupby("fixture_id").size().rename("api_lineup_rows").reset_index()
    starter_counts = (
        lineups[lineups["is_starting_xi"].eq(1)]
        .groupby("fixture_id")
        .size()
        .rename("api_starter_rows")
        .reset_index()
    )
    fixture_matches = (
        fixture_matches.merge(lineup_counts, left_on="api_fixture_id", right_on="fixture_id", how="left")
        .merge(starter_counts, left_on="api_fixture_id", right_on="fixture_id", how="left", suffixes=("", "_starter"))
        .drop(columns=["fixture_id", "fixture_id_starter"], errors="ignore")
    )
    fixture_matches["api_lineup_rows"] = pd.to_numeric(fixture_matches["api_lineup_rows"], errors="coerce").fillna(0).astype(int)
    fixture_matches["api_starter_rows"] = pd.to_numeric(fixture_matches["api_starter_rows"], errors="coerce").fillna(0).astype(int)
    fixture_matches["lineup_coverage_status"] = fixture_matches["api_starter_rows"].map(
        lambda value: "API_CONFIRMED_LINEUP" if int(value) >= 18 else "API_LINEUP_MISSING_OR_INCOMPLETE"
    )

    enriched = player_rows.merge(
        fixture_matches[
            [
                "fixture_key",
                "api_fixture_id",
                "api_fixture_key",
                "api_home_team_name",
                "api_away_team_name",
                "api_kickoff_local",
                "api_match_score",
                "api_lineup_rows",
                "api_starter_rows",
                "lineup_coverage_status",
            ]
        ],
        on="fixture_key",
        how="left",
    )
    referee_cols = [
        "fixture_key",
        "referee_name",
        "sample_matches",
        "cards_per_match_l20",
        "fouls_per_match_l20",
        "cards_per_foul_l20",
        "late_cards_per_match_l20",
        "first_half_cards_per_match_l20",
        "second_half_cards_per_match_l20",
        "strictness_score",
        "strictness_band",
        "profile_confidence",
        "open_play_foul_tolerance_score",
        "tactical_foul_punishment_score",
        "late_card_risk_flag",
        "penalty_risk_flag",
        "open_play_allowed_flag",
        "card_market_live_flag",
        "fouls_market_live_flag",
        "bookings_player_event_multiplier",
    ]
    if not referee_overlay.empty and "fixture_key" in referee_overlay.columns:
        overlay = referee_overlay[[col for col in referee_cols if col in referee_overlay.columns]].copy()
        overlay = overlay.rename(
            columns={
                "fixture_key": "api_fixture_key",
                "referee_name": "overlay_referee_name",
            }
        )
        enriched = enriched.merge(overlay, on="api_fixture_key", how="left")
    else:
        for col in referee_cols:
            if col == "fixture_key":
                continue
            out_col = "overlay_referee_name" if col == "referee_name" else col
            enriched[out_col] = pd.NA
    enriched["player_key"] = enriched["player_name"].map(player_key)

    def confirmation_status(row: pd.Series) -> str:
        if int(row.get("requires_player_confirmation_flag", 0) or 0) == 0:
            return "FIXTURE_LEVEL_NO_PLAYER_CONFIRMATION_REQUIRED"
        fixture_id = row.get("api_fixture_id")
        if pd.isna(fixture_id):
            return "NO_API_FIXTURE_MATCH"
        fixture_id = int(fixture_id)
        starters = starter_keys.get(fixture_id, set())
        if not starters:
            return "API_LINEUP_MISSING"
        if row.get("player_key") in starters:
            return "CONFIRMED_STARTER"
        return "API_LINEUP_AVAILABLE_NOT_STARTING"

    enriched["lineup_confirmation_status"] = enriched.apply(confirmation_status, axis=1)
    enriched["confirmed_starter_flag"] = enriched["lineup_confirmation_status"].eq("CONFIRMED_STARTER").astype(int)
    enriched["action_status"] = "PENDING_REVIEW"
    enriched.loc[enriched["lineup_confirmation_status"].eq("CONFIRMED_STARTER"), "action_status"] = "CONFIRMED_STARTER_REVIEW"
    enriched.loc[
        enriched["lineup_confirmation_status"].eq("FIXTURE_LEVEL_NO_PLAYER_CONFIRMATION_REQUIRED"),
        "action_status",
    ] = "FIXTURE_CONTEXT_REVIEW"
    enriched.loc[
        enriched["lineup_confirmation_status"].isin(["API_LINEUP_MISSING", "NO_API_FIXTURE_MATCH"]),
        "action_status",
    ] = "LINEUP_PENDING"
    enriched.loc[
        enriched["lineup_confirmation_status"].eq("API_LINEUP_AVAILABLE_NOT_STARTING"),
        "action_status",
    ] = "REMOVE_NOT_STARTING"
    enriched["market_rank"] = enriched["expression"].map(MARKET_RANK).fillna(99).astype(int)
    enriched["predicted_hit_rate_pct_num"] = pd.to_numeric(enriched.get("predicted_hit_rate_pct"), errors="coerce")

    sort_cols = ["confirmed_starter_flag", "watch_priority", "market_rank", "predicted_hit_rate_pct_num"]
    enriched = enriched.sort_values(sort_cols, ascending=[False, True, True, False]).reset_index(drop=True)
    shortlist = enriched[
        enriched["lineup_confirmation_status"].isin(
            ["CONFIRMED_STARTER", "FIXTURE_LEVEL_NO_PLAYER_CONFIRMATION_REQUIRED"]
        )
    ].copy()

    family_summary = (
        enriched.groupby("market_family", dropna=False)
        .agg(
            rows=("expression", "count"),
            confirmed_starter_rows=("lineup_confirmation_status", lambda s: int((s == "CONFIRMED_STARTER").sum())),
            fixture_context_rows=(
                "lineup_confirmation_status",
                lambda s: int((s == "FIXTURE_LEVEL_NO_PLAYER_CONFIRMATION_REQUIRED").sum()),
            ),
            lineup_pending_rows=("action_status", lambda s: int((s == "LINEUP_PENDING").sum())),
            remove_not_starting_rows=("action_status", lambda s: int((s == "REMOVE_NOT_STARTING").sum())),
            best_predicted_hit_rate_pct=("predicted_hit_rate_pct_num", "max"),
        )
        .reset_index()
    )
    missing_families = pd.DataFrame(
        [
            {
                "market_family": family,
                "rows": 0,
                "confirmed_starter_rows": 0,
                "fixture_context_rows": 0,
                "lineup_pending_rows": 0,
                "remove_not_starting_rows": 0,
                "best_predicted_hit_rate_pct": pd.NA,
            }
            for family in EXPECTED_MARKET_FAMILIES
            if family not in set(family_summary["market_family"].astype(str))
        ]
    )
    if not missing_families.empty:
        family_summary = pd.concat([family_summary, missing_families], ignore_index=True, sort=False)
    family_summary["live_wire_status"] = family_summary["rows"].map(lambda value: "WIRED" if int(value) > 0 else "NOT_IN_CURRENT_LIVE_BOARD")
    family_summary = family_summary.sort_values(["live_wire_status", "market_family"], ascending=[False, True]).reset_index(drop=True)

    coverage_path = outdir / "CONFIRMED_LINEUP_PLAYER_EVENT_FIXTURE_COVERAGE.csv"
    all_path = outdir / "CONFIRMED_LINEUP_PLAYER_EVENT_ROWS_ALL.csv"
    shortlist_path = outdir / "CONFIRMED_LINEUP_PLAYER_EVENT_SHORTLIST.csv"
    family_summary_path = outdir / "CONFIRMED_LINEUP_PLAYER_EVENT_FAMILY_SUMMARY.csv"
    md_path = outdir / "CONFIRMED_LINEUP_PLAYER_EVENT_SHORTLIST.md"

    fixture_matches.to_csv(coverage_path, index=False)
    enriched.to_csv(all_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)

    shortlist_cols = [
        "api_kickoff_local",
        "market_family",
        "league",
        "home_team_name",
        "away_team_name",
        "expression",
        "player_name",
        "team_name",
        "confidence_label",
        "predicted_hit_rate_pct",
        "watch_priority",
        "tactical_role",
        "overlay_referee_name",
        "strictness_band",
        "profile_confidence",
        "cards_per_match_l20",
        "fouls_per_match_l20",
        "late_card_risk_flag",
        "bookings_player_event_multiplier",
        "lineup_confirmation_status",
        "action_status",
    ]
    coverage_cols = [
        "api_kickoff_local",
        "league",
        "home_team_name",
        "away_team_name",
        "api_home_team_name",
        "api_away_team_name",
        "api_lineup_rows",
        "api_starter_rows",
        "lineup_coverage_status",
    ]
    lines = [
        "# Confirmed-Lineup Player-Event Shortlist Demo",
        "",
        "Research/demo only. Existing live-shadow rows are filtered to API-confirmed starters.",
        "",
        "## Counts",
        f"- all-family shadow rows checked: `{len(enriched)}`",
        f"- confirmed-starter / fixture-context shortlist rows: `{len(shortlist)}`",
        f"- fixtures with API confirmed/incomplete status rows: `{fixture_matches['fixture_key'].nunique()}`",
        f"- market families checked: `{enriched['market_family'].nunique() if not enriched.empty else 0}`",
        "",
        "## Market Family Summary",
        markdown_table(
            family_summary,
            [
                "market_family",
                "live_wire_status",
                "rows",
                "confirmed_starter_rows",
                "fixture_context_rows",
                "lineup_pending_rows",
                "remove_not_starting_rows",
                "best_predicted_hit_rate_pct",
            ],
            max_rows=args.max_md_rows,
        ),
        "",
        "## Confirmed Starter Shortlist",
        markdown_table(shortlist, shortlist_cols, max_rows=args.max_md_rows),
        "",
        "## Fixture Lineup Coverage",
        markdown_table(
            fixture_matches.sort_values(["api_kickoff_local", "league", "home_team_name"]),
            coverage_cols,
            max_rows=args.max_md_rows,
        ),
        "",
        "## Guardrails",
        "- No deploy_tier or tier mutation.",
        "- No priced player-event odds.",
        "- Referee overlay is a multiplier/context layer only; it does not create standalone player picks.",
        "- Rows marked API_LINEUP_MISSING should not be used as confirmed-lineup picks.",
        "- NOT_IN_CURRENT_LIVE_BOARD means the family has historical/audit work but no current live sprint rows yet.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {shortlist_path}")
    print(f"confirmed_starter_rows={len(shortlist)} checked_rows={len(enriched)}")
    if not shortlist.empty:
        print(shortlist[shortlist_cols].head(args.max_md_rows).to_string(index=False))


if __name__ == "__main__":
    main()
