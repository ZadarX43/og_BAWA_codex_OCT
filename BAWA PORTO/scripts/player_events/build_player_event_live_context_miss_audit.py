#!/usr/bin/env python3
"""Audit live player-event hits, misses, and missed context lanes.

Research-only. This consumes the official outcome tracker plus settled
API-Football actuals and highlights where the beta player-event board landed,
missed, or left an obvious context lane on the table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_many(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(pattern)):
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def summarize_tracker(tracker: pd.DataFrame, group_cols: list[str], clean_only: bool = False) -> pd.DataFrame:
    work = tracker.copy()
    if clean_only:
        work = work[
            work.get("outcome_status", "").astype(str).eq("GRADED")
            & work.get("red_card_context_flag", "").astype(str).eq("NO_RED_CARD_CONTEXT")
            & ~work.get("substitution_context_flag", "").astype(str).isin(
                ["EARLY_SUB_OFF_CONTEXT", "SUB_APPEARANCE_CONTEXT", "DID_NOT_PLAY_CONTEXT"]
            )
        ].copy()
    if work.empty:
        return pd.DataFrame(columns=group_cols + ["rows", "graded", "hits", "hit_rate", "pending"])
    rows = []
    for col in group_cols:
        if col not in work.columns:
            work[col] = ""
    for key, group in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded = group[group.get("outcome_status", "").astype(str).eq("GRADED")]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": int(len(group)),
                "graded": int(len(graded)),
                "hits": int(hits),
                "hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "pending": int(len(group) - len(graded)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["graded", "hits", "rows"], ascending=[False, False, False]).reset_index(drop=True)


def fixture_actuals(fixtures: pd.DataFrame, teams: pd.DataFrame | None = None) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    work = fixtures.copy()
    work["match_date"] = pd.to_datetime(work.get("match_date"), errors="coerce").dt.date.astype("string")
    if ("home_goals" not in work.columns or "away_goals" not in work.columns) and teams is not None and not teams.empty:
        team_scores = teams.copy()
        team_scores["is_home"] = num(team_scores.get("is_home", np.nan))
        team_scores["goals_for"] = num(team_scores.get("goals_for", np.nan))
        home_scores = (
            team_scores[team_scores["is_home"].eq(1)][["fixture_id", "goals_for"]]
            .rename(columns={"goals_for": "home_goals"})
            .drop_duplicates("fixture_id")
        )
        away_scores = (
            team_scores[team_scores["is_home"].eq(0)][["fixture_id", "goals_for"]]
            .rename(columns={"goals_for": "away_goals"})
            .drop_duplicates("fixture_id")
        )
        work = work.merge(home_scores, on="fixture_id", how="left").merge(away_scores, on="fixture_id", how="left")
    work["home_goals"] = num(work.get("home_goals", np.nan))
    work["away_goals"] = num(work.get("away_goals", np.nan))
    work["goal_diff"] = work["home_goals"] - work["away_goals"]
    work["actual_winner_side"] = np.select(
        [work["goal_diff"].gt(0), work["goal_diff"].lt(0)],
        ["HOME", "AWAY"],
        default="DRAW",
    )
    work["actual_losing_side"] = np.select(
        [work["goal_diff"].gt(0), work["goal_diff"].lt(0)],
        ["AWAY", "HOME"],
        default="BALANCED",
    )
    work["actual_state_shape"] = np.select(
        [work["goal_diff"].abs().ge(2), work["goal_diff"].eq(0)],
        ["DOMINANCE_MARGIN_2_PLUS", "BALANCED_SCORELINE"],
        default="ONE_GOAL_MARGIN",
    )
    keep = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "actual_winner_side",
        "actual_losing_side",
        "actual_state_shape",
        "status",
    ]
    return work[[col for col in keep if col in work.columns]].drop_duplicates("fixture_id")


def card_context(fixtures: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame, dashboard: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty or players.empty:
        return pd.DataFrame()
    fx = fixture_actuals(fixtures, teams)
    player_cards = players.copy()
    player_cards["yellow_cards"] = num(player_cards.get("yellow_cards", 0)).fillna(0)
    player_cards["red_cards"] = num(player_cards.get("red_cards", 0)).fillna(0)
    player_cards = player_cards[player_cards["yellow_cards"].gt(0) | player_cards["red_cards"].gt(0)].copy()
    if player_cards.empty:
        return pd.DataFrame()

    team_lookup = teams[["fixture_id", "team_id", "team_name", "is_home", "yellow_cards", "red_cards"]].copy() if not teams.empty else pd.DataFrame()
    if not team_lookup.empty:
        for col in ["fixture_id", "team_id"]:
            team_lookup[col] = num(team_lookup[col]).astype("Int64")
        team_lookup["team_side"] = np.where(num(team_lookup.get("is_home", 0)).eq(1), "HOME", "AWAY")
        team_lookup["team_yellow_cards"] = num(team_lookup.get("yellow_cards", 0)).fillna(0)
        team_lookup["team_red_cards"] = num(team_lookup.get("red_cards", 0)).fillna(0)
    else:
        team_lookup = pd.DataFrame(columns=["fixture_id", "team_id", "team_name", "team_side", "team_yellow_cards", "team_red_cards"])

    for col in ["fixture_id", "team_id"]:
        player_cards[col] = num(player_cards[col]).astype("Int64")
    rows = player_cards.merge(
        team_lookup[["fixture_id", "team_id", "team_name", "team_side", "team_yellow_cards", "team_red_cards"]],
        on=["fixture_id", "team_id"],
        how="left",
    ).merge(fx, on="fixture_id", how="left", suffixes=("", "_fixture"))

    predicted_cards = dashboard[
        dashboard.get("shadow_stage", pd.Series("", index=dashboard.index)).astype(str).str.contains("CARD|BOOK", case=False, na=False)
        | dashboard.get("expression", pd.Series("", index=dashboard.index)).astype(str).str.contains("card|booking", case=False, na=False)
    ].copy()
    predicted_card_fixture_keys = set(predicted_cards.get("fixture_key", pd.Series(dtype=str)).astype(str))
    rows["predicted_card_lane_present"] = rows.get("fixture_key", pd.Series("", index=rows.index)).astype(str).isin(predicted_card_fixture_keys)
    rows["card_side_vs_result"] = np.where(
        rows["team_side"].eq(rows["actual_losing_side"]),
        "LOSING_OR_UNDERDOG_SIDE_CARD",
        np.where(rows["actual_losing_side"].eq("BALANCED"), "BALANCED_SCORELINE_CARD", "WINNING_OR_DOMINANT_SIDE_CARD"),
    )
    rows["missed_context_lane"] = np.where(
        ~rows["predicted_card_lane_present"] & rows["card_side_vs_result"].eq("LOSING_OR_UNDERDOG_SIDE_CARD"),
        "SIDE_AWARE_UNDERDOG_CARD_LANE_MISSING",
        np.where(~rows["predicted_card_lane_present"], "CARD_LANE_NOT_WIRED_FOR_FIXTURE", ""),
    )
    keep = [
        "match_date",
        "home_team_name",
        "away_team_name",
        "actual_state_shape",
        "player_name",
        "team_name",
        "team_side",
        "yellow_cards",
        "red_cards",
        "team_yellow_cards",
        "team_red_cards",
        "card_side_vs_result",
        "predicted_card_lane_present",
        "missed_context_lane",
    ]
    return rows[[col for col in keep if col in rows.columns]].sort_values(
        ["match_date", "home_team_name", "team_side", "player_name"]
    )


def write_report(
    outdir: Path,
    tracker: pd.DataFrame,
    stage_summary: pd.DataFrame,
    clean_summary: pd.DataFrame,
    red_summary: pd.DataFrame,
    sub_summary: pd.DataFrame,
    hits: pd.DataFrame,
    misses: pd.DataFrame,
    card_rows: pd.DataFrame,
) -> None:
    graded = tracker[tracker.get("outcome_status", "").astype(str).eq("GRADED")].copy()
    hit_count = int(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0
    lines = [
        "# Player Event Live Context Miss Audit",
        "",
        "Research-only official-grading read for the first weekend player-event beta.",
        "",
        "## Overall",
        f"- tracked rows: `{len(tracker)}`",
        f"- officially graded rows from settled API actuals: `{len(graded)}`",
        f"- hits: `{hit_count}`",
        f"- hit rate on graded rows: `{(hit_count / len(graded)):.4f}`" if len(graded) else "- hit rate on graded rows: `n/a`",
        "",
        "## Stage Summary",
        markdown_table(stage_summary, max_rows=80),
        "",
        "## Clean 11v11 / Normal-Minute Summary",
        markdown_table(clean_summary, max_rows=80),
        "",
        "## Red-Card Context",
        markdown_table(red_summary, max_rows=80),
        "",
        "## Substitution Context",
        markdown_table(sub_summary, max_rows=80),
        "",
        "## Landed Rows",
        markdown_table(hits, max_rows=80),
        "",
        "## Missed Rows",
        markdown_table(misses, max_rows=80),
        "",
        "## Card / Booking Context Miss Audit",
        markdown_table(card_rows, max_rows=100),
        "",
        "## Research Read",
        "- The strongest clean beta signals from this settled pass were team corners, keeper saves 1.5+, tackles 1.5+/2.5+, and selected key-pass rows.",
        "- Confirmed-starter attack-watch produced real SOT wins, but raw shots 1.5+/2.5+ need tighter dominance/game-state handling.",
        "- Card/bookings logic should become side-aware: dominance mismatches need underdog/chasing-side pressure first.",
        "- Red-card and early-sub rows should stay in separate review buckets rather than being treated as clean misses.",
        "- Player Sub Swap markets require separate role-chain grading before being mixed into named-player evidence.",
    ]
    (outdir / "PLAYER_EVENT_LIVE_CONTEXT_MISS_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-rows", type=Path, required=True)
    parser.add_argument("--shadow-board", type=Path, required=True)
    parser.add_argument("--actuals-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tracker = pd.read_csv(args.tracker_rows, low_memory=False)
    dashboard = pd.read_csv(args.shadow_board, low_memory=False)
    fixtures = read_many(args.actuals_root, "fixtures_master__*.csv")
    players = read_many(args.actuals_root, "match_player_stats__*.csv")
    teams = read_many(args.actuals_root, "match_team_stats__*.csv")

    stage_summary = summarize_tracker(tracker, ["shadow_stage", "watch_priority"])
    clean_summary = summarize_tracker(tracker, ["shadow_stage", "watch_priority"], clean_only=True)
    red_summary = summarize_tracker(tracker, ["shadow_stage", "red_card_context_flag"])
    sub_summary = summarize_tracker(tracker, ["shadow_stage", "substitution_context_flag"])

    graded = tracker[tracker.get("outcome_status", "").astype(str).eq("GRADED")].copy()
    focus_cols = [
        "shadow_stage",
        "league",
        "home_team_name",
        "away_team_name",
        "player_name",
        "team_name",
        "expression",
        "actual_stat_col",
        "actual_threshold",
        "actual_stat_value",
        "actual_hit",
        "red_card_context_flag",
        "substitution_context_flag",
        "player_sub_swap_review_mode",
    ]
    hits = graded[num(graded.get("actual_hit", 0)).eq(1)][[col for col in focus_cols if col in graded.columns]].copy()
    misses = graded[num(graded.get("actual_hit", 0)).eq(0)][[col for col in focus_cols if col in graded.columns]].copy()
    card_rows = card_context(fixtures, players, teams, dashboard)

    stage_summary.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_STAGE_SUMMARY.csv", index=False)
    clean_summary.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_CLEAN_SUMMARY.csv", index=False)
    red_summary.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_RED_CARD_SUMMARY.csv", index=False)
    sub_summary.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_SUBSTITUTION_SUMMARY.csv", index=False)
    hits.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_LANDED_ROWS.csv", index=False)
    misses.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_MISSED_ROWS.csv", index=False)
    card_rows.to_csv(args.outdir / "PLAYER_EVENT_CONTEXT_CARD_MISS_AUDIT.csv", index=False)
    write_report(args.outdir, tracker, stage_summary, clean_summary, red_summary, sub_summary, hits, misses, card_rows)

    print(f"WROTE {args.outdir}")
    print(f"graded={len(graded)} hits={int(num(graded.get('actual_hit', 0)).sum()) if not graded.empty else 0}")
    if not stage_summary.empty:
        print(stage_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
