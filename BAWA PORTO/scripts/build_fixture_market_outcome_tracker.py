#!/usr/bin/env python3
"""Track outcomes for fixture-market intelligence shadow rows.

Research-only sidecar for Bet365-style fixture/team markets. It consumes the
unified live shadow dashboard, grades supported fixture-market rows against
normalized API-Football team actuals when available, and writes compact summary
tables. It does not create priced odds, deploy picks, slips, or production
routing changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import norm_text


DEFAULT_SHADOW_BOARD = (
    ROOT
    / "reports"
    / "2026-05-08"
    / "live_shadow_research_dashboard_with_fixture_markets"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__LIVE_SHADOW_RESEARCH_DASHBOARD.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-08" / "fixture_market_outcome_tracker"
DEFAULT_ACTUAL_ROOTS = (
    ROOT / "reports" / "2026-05-07" / "api_current_player_window_2026_04_24_to_2026_05_04" / "normalized",
    ROOT / "data_sources" / "api_football" / "normalized",
)

SUPPORTED_STAGES = {
    "GOAL_RANGE_SHADOW",
    "WINNING_MARGIN_2_PLUS_SHADOW",
    "WINNING_MARGIN_3_PLUS_WATCH",
    "TOTAL_SHOTS_SHADOW",
    "TOTAL_SOT_SHADOW",
    "TEAM_MOST_SHOTS_SHADOW",
    "TEAM_MOST_SOT_SHADOW",
    "TEAM_MOST_CORNERS_SHADOW",
    "TEAM_MOST_CARDS_WATCH",
}

TOTAL_MARKET_LINES = {
    "TOTAL_SHOTS_SHADOW": {"HIGH": ("actual_total_shots", 24.5, "ge"), "LOW": ("actual_total_shots", 20.0, "le")},
    "TOTAL_SOT_SHADOW": {"HIGH": ("actual_total_sot", 8.0, "ge"), "LOW": ("actual_total_sot", 6.0, "le")},
}

TEAM_MOST_STAT_COLS = {
    "TEAM_MOST_SHOTS_SHADOW": "actual_shots_total",
    "TEAM_MOST_SOT_SHADOW": "actual_shots_on_goal",
    "TEAM_MOST_CORNERS_SHADOW": "actual_corners_for",
    "TEAM_MOST_CARDS_WATCH": "actual_cards",
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def parse_file_tag(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})\.csv$", path.name)
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


def source_priority(root: Path) -> int:
    return 0 if "reports/2026-05-07/api_current_player_window" in str(root) else 1


def norm_team(value: Any, league_tag: Any = None) -> str:
    return norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))


def load_fixtures(root: Path, league_tag: str, season_tag: int) -> pd.DataFrame:
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
    return read_selected(root / f"fixtures_master__{league_tag}__{season_tag}.csv", fixture_cols)


def load_fixture_team_actuals(actual_roots: list[Path]) -> pd.DataFrame:
    stats_cols = [
        "fixture_id",
        "team_id",
        "team_name",
        "is_home",
        "goals_for",
        "goals_against",
        "shots_total",
        "shots_on_goal",
        "corners_for",
        "yellow_cards",
        "red_cards",
    ]
    frames: list[pd.DataFrame] = []
    for root in actual_roots:
        for stats_path in sorted(root.glob("match_team_stats__*.csv")):
            parsed = parse_file_tag(stats_path, "match_team_stats")
            if parsed is None:
                continue
            league_tag, season_tag = parsed
            stats = read_selected(stats_path, stats_cols)
            fixtures = load_fixtures(root, league_tag, season_tag)
            if stats.empty or fixtures.empty:
                continue
            merged = stats.merge(fixtures, on="fixture_id", how="left", suffixes=("", "_fixture"))
            merged["league_tag"] = league_tag
            merged["season_tag"] = season_tag
            merged["_source_priority"] = source_priority(root)
            for col in [
                "fixture_id",
                "team_id",
                "home_team_id",
                "away_team_id",
                "goals_for",
                "goals_against",
                "shots_total",
                "shots_on_goal",
                "corners_for",
                "yellow_cards",
                "red_cards",
            ]:
                if col not in merged.columns:
                    merged[col] = np.nan
                merged[col] = num(merged[col])
            merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce").dt.date.astype("string")
            merged["home_team_norm"] = [norm_team(team, league_tag) for team in merged["home_team_name"]]
            merged["away_team_norm"] = [norm_team(team, league_tag) for team in merged["away_team_name"]]
            merged["team_name_norm"] = [norm_team(team, league_tag) for team in merged["team_name"]]
            merged["team_side"] = np.where(num(merged["is_home"]).eq(1), "HOME", "AWAY")
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values("_source_priority").drop_duplicates(["fixture_id", "team_id"], keep="first")
    return out.drop(columns=["_source_priority"], errors="ignore")


def prepare_fixture_actuals(team_actuals: pd.DataFrame) -> pd.DataFrame:
    if team_actuals.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for fixture_id, group in team_actuals.groupby("fixture_id", dropna=False):
        if pd.isna(fixture_id) or len(group) < 2:
            continue
        home = group[num(group["is_home"]).eq(1)]
        away = group[num(group["is_home"]).eq(0)]
        if home.empty or away.empty:
            continue
        home_row = home.iloc[0]
        away_row = away.iloc[0]
        home_goals = float(home_row.get("goals_for", np.nan))
        away_goals = float(away_row.get("goals_for", np.nan))
        if not np.isfinite(home_goals) or not np.isfinite(away_goals):
            continue
        home_cards = float(home_row.get("yellow_cards", 0.0) or 0.0) + float(home_row.get("red_cards", 0.0) or 0.0)
        away_cards = float(away_row.get("yellow_cards", 0.0) or 0.0) + float(away_row.get("red_cards", 0.0) or 0.0)
        rows.append(
            {
                "fixture_id": fixture_id,
                "api_fixture_key": home_row.get("fixture_key", ""),
                "match_date": home_row.get("match_date", ""),
                "league_tag": home_row.get("league_tag", ""),
                "season_tag": home_row.get("season_tag", ""),
                "status": home_row.get("status", ""),
                "home_team_name_actual": home_row.get("home_team_name", ""),
                "away_team_name_actual": home_row.get("away_team_name", ""),
                "home_team_norm": home_row.get("home_team_norm", ""),
                "away_team_norm": home_row.get("away_team_norm", ""),
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
                "actual_total_goals": home_goals + away_goals,
                "actual_goal_diff": home_goals - away_goals,
                "actual_home_shots_total": float(home_row.get("shots_total", np.nan)),
                "actual_away_shots_total": float(away_row.get("shots_total", np.nan)),
                "actual_home_shots_on_goal": float(home_row.get("shots_on_goal", np.nan)),
                "actual_away_shots_on_goal": float(away_row.get("shots_on_goal", np.nan)),
                "actual_home_corners_for": float(home_row.get("corners_for", np.nan)),
                "actual_away_corners_for": float(away_row.get("corners_for", np.nan)),
                "actual_home_cards": home_cards,
                "actual_away_cards": away_cards,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["actual_total_shots"] = out["actual_home_shots_total"] + out["actual_away_shots_total"]
    out["actual_total_sot"] = out["actual_home_shots_on_goal"] + out["actual_away_shots_on_goal"]
    return out


def prepare_shadow(shadow: pd.DataFrame) -> pd.DataFrame:
    stage = shadow.get("shadow_stage", pd.Series("", index=shadow.index)).astype(str)
    out = shadow[stage.isin(SUPPORTED_STAGES)].copy()
    if out.empty:
        return out
    for col in ["fixture_key", "match_date", "league", "home_team_name", "away_team_name", "source_selection", "watch_priority"]:
        if col not in out.columns:
            out[col] = ""
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.date.astype("string")
    out["home_team_norm"] = [norm_team(team) for team in out["home_team_name"]]
    out["away_team_norm"] = [norm_team(team) for team in out["away_team_name"]]
    out["_shadow_row_id"] = np.arange(len(out))
    return out


def join_actuals(shadow: pd.DataFrame, fixture_actuals: pd.DataFrame) -> pd.DataFrame:
    if shadow.empty:
        return shadow
    if fixture_actuals.empty:
        out = shadow.copy()
        out["outcome_status"] = "PENDING_NO_FIXTURE_ACTUALS"
        return out
    keep = fixture_actuals.copy()
    shadow_cols = list(shadow.columns)
    by_key = shadow.merge(
        keep,
        left_on="fixture_key",
        right_on="api_fixture_key",
        how="left",
        suffixes=("", "_actual"),
    )
    exact = by_key[by_key["fixture_id"].notna()].copy()
    leftovers = by_key.loc[by_key["fixture_id"].isna(), shadow_cols].copy()
    fallback = leftovers.merge(
        keep.drop(columns=["api_fixture_key"], errors="ignore"),
        on=["match_date", "home_team_norm", "away_team_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    out = pd.concat([exact, fallback], ignore_index=True, sort=False)
    out = out.sort_values("_shadow_row_id").drop_duplicates("_shadow_row_id", keep="first")
    out["outcome_status"] = np.where(out["fixture_id"].notna(), "GRADED", "PENDING_NO_FIXTURE_MATCH")
    return out


def goal_range_hit(selection: str, total_goals: float) -> tuple[str, float]:
    if not np.isfinite(total_goals):
        return "", np.nan
    if total_goals <= 1:
        actual = "0-1 goals"
    elif total_goals <= 3:
        actual = "2-3 goals"
    else:
        actual = "4+ goals"
    return actual, float(actual == selection)


def margin_hit(stage: str, selection: str, goal_diff: float) -> tuple[str, float]:
    if not np.isfinite(goal_diff):
        return "", np.nan
    if stage == "WINNING_MARGIN_2_PLUS_SHADOW":
        if goal_diff >= 2:
            actual = "HOME 2+"
        elif goal_diff <= -2:
            actual = "AWAY 2+"
        else:
            actual = "NO 2+"
    else:
        if goal_diff >= 3:
            actual = "HOME 3+"
        elif goal_diff <= -3:
            actual = "AWAY 3+"
        else:
            actual = "NO 3+"
    return actual, float(actual == selection)


def total_line_hit(stage: str, selection: str, row: pd.Series) -> tuple[str, float, float]:
    config = TOTAL_MARKET_LINES.get(stage, {}).get(selection)
    if config is None:
        return "", np.nan, np.nan
    stat_col, threshold, direction = config
    value = float(row.get(stat_col, np.nan))
    if not np.isfinite(value):
        return "", threshold, np.nan
    if direction == "ge":
        actual = "HIGH" if value >= threshold else "NOT_HIGH"
        return actual, threshold, float(value >= threshold)
    actual = "LOW" if value <= threshold else "NOT_LOW"
    return actual, threshold, float(value <= threshold)


def selected_side_from_team(row: pd.Series, selected_team: str) -> str:
    selected_norm = norm_team(selected_team, row.get("league_tag", ""))
    if selected_norm and selected_norm == row.get("home_team_norm", ""):
        return "HOME"
    if selected_norm and selected_norm == row.get("away_team_norm", ""):
        return "AWAY"
    return ""


def team_most_hit(stage: str, selection: str, row: pd.Series) -> tuple[str, float]:
    stat_col = TEAM_MOST_STAT_COLS.get(stage, "")
    if not stat_col:
        return "", np.nan
    home_value = float(row.get(f"actual_home_{stat_col.replace('actual_', '')}", np.nan))
    away_value = float(row.get(f"actual_away_{stat_col.replace('actual_', '')}", np.nan))
    if not np.isfinite(home_value) or not np.isfinite(away_value):
        return "", np.nan
    if home_value > away_value:
        actual = "HOME"
    elif away_value > home_value:
        actual = "AWAY"
    else:
        actual = "TIE"
    selected_side = selected_side_from_team(row, selection)
    return actual, float(selected_side == actual)


def score_outcomes(tracked: pd.DataFrame) -> pd.DataFrame:
    out = tracked.copy()
    out["actual_selection"] = ""
    out["actual_stat_col"] = ""
    out["actual_threshold"] = np.nan
    out["actual_stat_value"] = np.nan
    out["actual_hit"] = np.nan
    if out.empty:
        return out

    for idx, row in out.iterrows():
        if row.get("outcome_status") != "GRADED":
            continue
        stage = str(row.get("shadow_stage", ""))
        selection = str(row.get("source_selection", ""))
        if stage == "GOAL_RANGE_SHADOW":
            actual, hit = goal_range_hit(selection, float(row.get("actual_total_goals", np.nan)))
            out.at[idx, "actual_selection"] = actual
            out.at[idx, "actual_stat_col"] = "actual_total_goals"
            out.at[idx, "actual_stat_value"] = row.get("actual_total_goals", np.nan)
            out.at[idx, "actual_hit"] = hit
        elif stage in {"WINNING_MARGIN_2_PLUS_SHADOW", "WINNING_MARGIN_3_PLUS_WATCH"}:
            actual, hit = margin_hit(stage, selection, float(row.get("actual_goal_diff", np.nan)))
            out.at[idx, "actual_selection"] = actual
            out.at[idx, "actual_stat_col"] = "actual_goal_diff"
            out.at[idx, "actual_stat_value"] = row.get("actual_goal_diff", np.nan)
            out.at[idx, "actual_hit"] = hit
        elif stage in TOTAL_MARKET_LINES:
            actual, threshold, hit = total_line_hit(stage, selection, row)
            stat_col = TOTAL_MARKET_LINES.get(stage, {}).get(selection, ("", np.nan, ""))[0]
            out.at[idx, "actual_selection"] = actual
            out.at[idx, "actual_stat_col"] = stat_col
            out.at[idx, "actual_threshold"] = threshold
            out.at[idx, "actual_stat_value"] = row.get(stat_col, np.nan)
            out.at[idx, "actual_hit"] = hit
        elif stage in TEAM_MOST_STAT_COLS:
            actual, hit = team_most_hit(stage, selection, row)
            out.at[idx, "actual_selection"] = actual
            out.at[idx, "actual_stat_col"] = TEAM_MOST_STAT_COLS[stage]
            out.at[idx, "actual_hit"] = hit
    return out


def summarize(tracked: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if tracked.empty:
        return pd.DataFrame(columns=group_cols + ["rows", "graded", "hits", "hit_rate", "pending"])
    rows = []
    for key, group in tracked.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded_mask = group["outcome_status"].astype(str).eq("GRADED")
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
    out = pd.DataFrame(rows)
    return out.sort_values(["graded", "rows"], ascending=[False, False]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_report(
    outdir: Path,
    tracked: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    shadow_board: Path,
    actual_roots: list[Path],
    fixture_actuals: pd.DataFrame,
) -> None:
    graded = int(tracked["outcome_status"].astype(str).eq("GRADED").sum()) if not tracked.empty else 0
    pending = int(len(tracked) - graded)
    actual_min_date = fixture_actuals["match_date"].min() if not fixture_actuals.empty and "match_date" in fixture_actuals.columns else ""
    actual_max_date = fixture_actuals["match_date"].max() if not fixture_actuals.empty and "match_date" in fixture_actuals.columns else ""
    actual_fixtures = int(len(fixture_actuals)) if not fixture_actuals.empty else 0
    lines = [
        "# Fixture Market Outcome Tracker",
        "",
        "Research-only outcome tracker for fixture-market intelligence shadow rows.",
        "",
        "## Safety",
        "- No priced odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Pending rows mean API fixture/team actuals were not joinable yet.",
        "- Team-most cards are graded as API yellow plus red cards; keep as watch-only until bookmaker rule parity is confirmed.",
        "",
        "## Inputs",
        f"- shadow board: `{shadow_board}`",
        *[f"- actual root: `{root}`" for root in actual_roots],
        "",
        "## Supported Shadow Stages",
        *[f"- `{stage}`" for stage in sorted(SUPPORTED_STAGES)],
        "",
        "## Grading Definitions",
        "- `GOAL_RANGE_SHADOW`: exact range from final total goals: 0-1, 2-3, or 4+.",
        "- `WINNING_MARGIN_2_PLUS_SHADOW`: selected side must win by 2+.",
        "- `WINNING_MARGIN_3_PLUS_WATCH`: selected side must win by 3+.",
        "- `TOTAL_SHOTS_SHADOW`: HIGH grades at actual total shots >= 24.5; LOW grades at <= 20.0.",
        "- `TOTAL_SOT_SHADOW`: HIGH grades at actual total SOT >= 8.0; LOW grades at <= 6.0.",
        "- Team-most markets: selected team must strictly lead the opponent; ties are misses for now.",
        "",
        "## Overall",
        f"- tracked rows: `{len(tracked)}`",
        f"- graded rows: `{graded}`",
        f"- pending rows: `{pending}`",
        f"- fixture actuals loaded: `{actual_fixtures}`",
        f"- fixture actuals date range: `{actual_min_date}` to `{actual_max_date}`",
        "",
        "## By Stage / Priority",
        markdown_table(summaries.get("stage_priority", pd.DataFrame()), max_rows=100),
        "",
        "## By Stage / League",
        markdown_table(summaries.get("stage_league", pd.DataFrame()), max_rows=120),
        "",
        "## By Selection",
        markdown_table(summaries.get("selection", pd.DataFrame()), max_rows=120),
        "",
        "## Outcome Status",
        markdown_table(summaries.get("status", pd.DataFrame()), max_rows=80),
        "",
        "## By Tactical Family",
        markdown_table(summaries.get("tactical_family", pd.DataFrame()), max_rows=80),
    ]
    (outdir / "FIXTURE_MARKET_OUTCOME_TRACKER.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-board", type=Path, default=DEFAULT_SHADOW_BOARD)
    parser.add_argument("--actuals-root", type=Path, action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    actual_roots = args.actuals_root or list(DEFAULT_ACTUAL_ROOTS)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.shadow_board.exists():
        raise SystemExit(f"Missing shadow board: {args.shadow_board}")

    shadow = prepare_shadow(pd.read_csv(args.shadow_board, low_memory=False))
    team_actuals = load_fixture_team_actuals(actual_roots)
    fixture_actuals = prepare_fixture_actuals(team_actuals)
    tracked = join_actuals(shadow, fixture_actuals)
    tracked = score_outcomes(tracked)
    if not tracked.empty:
        tracked = tracked.sort_values("_shadow_row_id").reset_index(drop=True)

    summaries = {
        "stage_priority": summarize(tracked, ["shadow_stage", "watch_priority"]),
        "stage_league": summarize(tracked, ["shadow_stage", "league"]),
        "selection": summarize(tracked, ["shadow_stage", "source_selection"]),
        "status": summarize(tracked, ["shadow_stage", "outcome_status"]),
        "tactical_family": summarize(tracked, ["shadow_stage", "tactical_feature_families"])
        if "tactical_feature_families" in tracked.columns
        else pd.DataFrame(),
    }

    tracked.to_csv(args.outdir / "FIXTURE_MARKET_OUTCOME_TRACKER_ROWS.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(args.outdir / f"FIXTURE_MARKET_OUTCOME_{name.upper()}_SUMMARY.csv", index=False)
    write_report(args.outdir, tracked, summaries, args.shadow_board, actual_roots, fixture_actuals)

    graded = int(tracked["outcome_status"].astype(str).eq("GRADED").sum()) if not tracked.empty else 0
    print(f"WROTE {args.outdir}")
    print(f"tracked_rows={len(tracked)} graded={graded}")
    if not summaries["stage_priority"].empty:
        print(summaries["stage_priority"].to_string(index=False))


if __name__ == "__main__":
    main()
