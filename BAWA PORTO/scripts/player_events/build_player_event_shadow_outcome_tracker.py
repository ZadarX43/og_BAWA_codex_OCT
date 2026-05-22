#!/usr/bin/env python3
"""Track outcomes for player-event live shadow rows.

Research-only sidecar. Joins PLAYER_EVENT_INTERACTION watch rows to normalized
API-Football match-player actuals when available. It does not create priced
odds, deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import norm_text


DEFAULT_SHADOW_BOARD = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_exact_interaction_shadow_refresh_current_projection_exact_fouled_strict_policy"
    / "player_event_interaction_live_shadow_board_exact"
    / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_event_shadow_outcome_tracker"
DEFAULT_ACTUAL_ROOTS = (
    ROOT / "reports" / "2026-05-07" / "api_current_player_window_2026_04_24_to_2026_05_04" / "normalized",
    ROOT / "data_sources" / "api_football" / "normalized",
)


MARKET_TARGETS = {
    "PLAYER_FOULED_0_5_INTERACTION_WATCH": ("fouls_drawn", 1, "actual_fouled_ge1"),
    "PLAYER_FOULED_1_5_INTERACTION_WATCH": ("fouls_drawn", 2, "actual_fouled_ge2"),
    "PLAYER_SHOTS_1_5_INTERACTION_WATCH": ("shots_total", 2, "actual_shots_ge2"),
    "PLAYER_SHOTS_2_5_INTERACTION_WATCH": ("shots_total", 3, "actual_shots_ge3"),
    "PLAYER_SOT_0_5_INTERACTION_WATCH": ("shots_on_target", 1, "actual_sot_ge1"),
    "PLAYER_TACKLES_1_5_LIVE_SHADOW": ("tackles", 2, "actual_tackles_ge2"),
    "PLAYER_TACKLES_2_5_LIVE_SHADOW": ("tackles", 3, "actual_tackles_ge3"),
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_team(value: Any, league_tag: Any = None) -> str:
    return norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))


def parse_stats_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"match_player_stats__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_selected(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in cols if col in header.columns]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_actuals(actual_roots: list[Path]) -> pd.DataFrame:
    stats_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "fouls_drawn",
        "fouls_committed",
        "tackles",
        "yellow_cards",
    ]
    fixture_cols = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "status",
    ]
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, int, int]] = set()
    for root in actual_roots:
        for stats_path in sorted(root.glob("match_player_stats__*.csv")):
            parsed = parse_stats_tag(stats_path)
            if parsed is None:
                continue
            league_tag, season_tag = parsed
            fixtures_path = root / f"fixtures_master__{league_tag}__{season_tag}.csv"
            stats = read_selected(stats_path, stats_cols)
            fixtures = read_selected(fixtures_path, fixture_cols)
            if stats.empty or fixtures.empty:
                continue
            merged = stats.merge(fixtures, on="fixture_id", how="left")
            merged["league_tag"] = league_tag
            merged["season_tag"] = season_tag
            for col in ["fixture_id", "team_id", "home_team_id", "away_team_id"]:
                merged[col] = num(merged[col]).astype("Int64")
            merged["team_name"] = np.where(
                merged["team_id"].eq(merged["home_team_id"]),
                merged["home_team_name"],
                merged["away_team_name"],
            )
            merged["player_team_side"] = np.where(merged["team_id"].eq(merged["home_team_id"]), "HOME", "AWAY")
            merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce").dt.date.astype("string")
            merged["home_team_norm"] = [norm_team(team) for team in merged["home_team_name"]]
            merged["away_team_norm"] = [norm_team(team) for team in merged["away_team_name"]]
            merged["team_name_norm"] = [norm_team(team) for team in merged["team_name"]]
            merged["player_name_norm"] = merged["player_name"].map(norm_text)
            for col in [
                "minutes",
                "started_flag",
                "shots_total",
                "shots_on_target",
                "fouls_drawn",
                "fouls_committed",
                "tackles",
                "yellow_cards",
            ]:
                if col not in merged.columns:
                    merged[col] = 0.0
                merged[col] = num(merged[col]).fillna(0.0)
            key = ["fixture_id", "team_id", "player_id"]
            merged["_source_priority"] = 0 if "reports/2026-05-07/api_current_player_window" in str(root) else 1
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values("_source_priority").drop_duplicates(["fixture_id", "team_id", "player_id"], keep="first")
    return out.drop(columns=["_source_priority"], errors="ignore")


def prepare_shadow(shadow: pd.DataFrame) -> pd.DataFrame:
    out = shadow[shadow.get("shadow_stage", pd.Series("", index=shadow.index)).isin(MARKET_TARGETS)].copy()
    if out.empty:
        return out
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.date.astype("string")
    out["home_team_norm"] = [norm_team(team) for team in out["home_team_name"]]
    out["away_team_norm"] = [norm_team(team) for team in out["away_team_name"]]
    out["team_name_norm"] = [norm_team(team) for team in out["team_name"]]
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["_shadow_row_id"] = np.arange(len(out))
    return out


def join_actuals(shadow: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if shadow.empty:
        return shadow
    if actuals.empty:
        out = shadow.copy()
        out["outcome_status"] = "PENDING_NO_ACTUALS"
        return out

    actual_cols = [
        "fixture_key",
        "match_date",
        "home_team_norm",
        "away_team_norm",
        "team_name_norm",
        "player_name_norm",
        "fixture_id",
        "league_tag",
        "season_tag",
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "fouls_drawn",
        "fouls_committed",
        "tackles",
        "yellow_cards",
        "status",
    ]
    actual_keep = actuals[[c for c in actual_cols if c in actuals.columns]].copy()
    by_fixture_key = shadow.merge(
        actual_keep,
        on=["fixture_key", "team_name_norm", "player_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    matched = by_fixture_key["fixture_id"].notna()
    leftovers = by_fixture_key[~matched].drop(columns=[c for c in by_fixture_key.columns if c.endswith("_actual")], errors="ignore")
    exact = by_fixture_key[matched].copy()

    fallback = leftovers.merge(
        actual_keep.drop(columns=["fixture_key"], errors="ignore"),
        on=["match_date", "home_team_norm", "away_team_norm", "team_name_norm", "player_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    out = pd.concat([exact, fallback], ignore_index=True, sort=False)
    for col in [
        "fixture_id",
        "league_tag",
        "season_tag",
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "fouls_drawn",
        "fouls_committed",
        "tackles",
        "yellow_cards",
        "status",
    ]:
        actual_col = f"{col}_actual"
        if actual_col in out.columns:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = out[col].where(out[col].notna(), out[actual_col])
    out = out.sort_values("_shadow_row_id").drop_duplicates("_shadow_row_id", keep="first")
    out["outcome_status"] = np.where(out["fixture_id"].notna(), "GRADED", "PENDING_NO_MATCH")
    return out


def score_outcomes(tracked: pd.DataFrame) -> pd.DataFrame:
    out = tracked.copy()
    out["actual_stat_col"] = ""
    out["actual_threshold"] = np.nan
    out["actual_stat_value"] = np.nan
    out["actual_hit"] = np.nan
    for stage, (stat_col, threshold, hit_col) in MARKET_TARGETS.items():
        mask = out["shadow_stage"].astype(str).eq(stage)
        out.loc[mask, "actual_stat_col"] = stat_col
        out.loc[mask, "actual_threshold"] = threshold
        if stat_col not in out.columns:
            out[stat_col] = np.nan
        values = num(out.loc[mask, stat_col])
        out.loc[mask, "actual_stat_value"] = values
        out.loc[mask & out["outcome_status"].eq("GRADED"), "actual_hit"] = values.ge(threshold).astype(float)
        out.loc[mask, hit_col] = out.loc[mask, "actual_hit"]
    return out


def summarize(tracked: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if tracked.empty:
        return pd.DataFrame(columns=group_cols + ["rows", "graded", "hits", "hit_rate", "pending"])
    rows = []
    for key, group in tracked.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded_mask = group["outcome_status"].eq("GRADED")
        graded = group[graded_mask]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": int(len(group)),
                "graded": int(len(graded)),
                "hits": int(hits),
                "hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "pending": int((~graded_mask).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows)
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in work.columns:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 4)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    tracked: pd.DataFrame,
    summary: pd.DataFrame,
    league_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
    formation_summary: pd.DataFrame,
    context_cell_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Player Event Shadow Outcome Tracker",
        "",
        "Research-only outcome tracker for exact player-event shadow rows.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Pending rows mean API player actuals were not joinable yet.",
        "",
        "## Overall",
        f"- tracked rows: `{len(tracked)}`",
        f"- graded rows: `{int(tracked['outcome_status'].eq('GRADED').sum()) if not tracked.empty else 0}`",
        f"- pending rows: `{int(tracked['outcome_status'].ne('GRADED').sum()) if not tracked.empty else 0}`",
        "",
        "## By Stage / Priority",
        markdown_table(summary),
        "",
        "## By Stage / League",
        markdown_table(league_summary, max_rows=80),
        "",
        "## By Stage / Tactical Role",
        markdown_table(role_summary, max_rows=80),
        "",
        "## By Stage / Formation Matchup",
        markdown_table(formation_summary, max_rows=80),
        "",
        "## By Stage / Fouled Context Cell Label",
        markdown_table(context_cell_summary, max_rows=80),
    ]
    (outdir / "PLAYER_EVENT_SHADOW_OUTCOME_TRACKER.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-board", type=Path, default=DEFAULT_SHADOW_BOARD)
    parser.add_argument("--actuals-root", type=Path, action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fouled-only", action="store_true")
    args = parser.parse_args()

    actual_roots = args.actuals_root or list(DEFAULT_ACTUAL_ROOTS)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.shadow_board.exists():
        raise SystemExit(f"Missing shadow board: {args.shadow_board}")
    shadow = prepare_shadow(pd.read_csv(args.shadow_board, low_memory=False))
    if args.fouled_only and not shadow.empty:
        shadow = shadow[shadow["shadow_stage"].astype(str).str.startswith("PLAYER_FOULED_")].copy()

    actuals = load_actuals(actual_roots)
    tracked = score_outcomes(join_actuals(shadow, actuals))
    summary = summarize(tracked, ["shadow_stage", "watch_priority"])
    league_summary = summarize(tracked, ["shadow_stage", "league"])
    role_summary = summarize(tracked, ["shadow_stage", "tactical_role"]) if "tactical_role" in tracked.columns else pd.DataFrame()
    formation_summary = (
        summarize(tracked, ["shadow_stage", "formation_matchup_label"])
        if "formation_matchup_label" in tracked.columns
        else pd.DataFrame()
    )
    context_cell_summary = (
        summarize(tracked, ["shadow_stage", "fouled_context_cell_label"])
        if "fouled_context_cell_label" in tracked.columns
        else pd.DataFrame()
    )

    tracked.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_TRACKER_ROWS.csv", index=False)
    summary.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_SUMMARY.csv", index=False)
    league_summary.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_LEAGUE_SUMMARY.csv", index=False)
    role_summary.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_ROLE_SUMMARY.csv", index=False)
    formation_summary.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_FORMATION_SUMMARY.csv", index=False)
    context_cell_summary.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_CONTEXT_CELL_SUMMARY.csv", index=False)
    write_report(args.outdir, tracked, summary, league_summary, role_summary, formation_summary, context_cell_summary)

    print(f"WROTE {args.outdir}")
    print(f"tracked_rows={len(tracked)} graded={int(tracked['outcome_status'].eq('GRADED').sum()) if not tracked.empty else 0}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
