#!/usr/bin/env python3
"""Audit FootyStats World Cup drops before any ingest or training.

Research-only. This script reads the Desktop drop folder and writes a compact
coverage/readiness report into reports/latest. It does not move, copy, or ingest
the source CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("reports/latest/footystats_world_cup_drop_audit_2026_05_19")

FILE_RE = re.compile(
    r"^international-fifa-world-cup-(?P<host>[a-z0-9-]+)-(?P<kind>matches|teams|players)-(?P<start>\d{4})-to-(?P<end>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$",
    re.IGNORECASE,
)

MATCH_REQUIRED = [
    "timestamp",
    "date_GMT",
    "status",
    "home_team_name",
    "away_team_name",
    "home_team_goal_count",
    "away_team_goal_count",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over15",
    "odds_ft_over25",
    "odds_btts_yes",
    "odds_btts_no",
]

MATCH_PREMATCH = [
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "Home Team Pre-Match xG",
    "Away Team Pre-Match xG",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_15_percentage_pre_match",
    "over_25_percentage_pre_match",
]

MATCH_ACTUAL_STATS = [
    "home_team_corner_count",
    "away_team_corner_count",
    "home_team_shots",
    "away_team_shots",
    "home_team_shots_on_target",
    "away_team_shots_on_target",
    "home_team_possession",
    "away_team_possession",
    "team_a_xg",
    "team_b_xg",
]

TEAM_REQUIRED = [
    "team_name",
    "common_name",
    "season",
    "matches_played",
    "wins",
    "draws",
    "losses",
    "goals_scored",
    "goals_conceded",
]

PLAYER_REQUIRED = [
    "full_name",
    "age",
    "position",
    "Current Club",
    "minutes_played_overall",
    "appearances_overall",
    "games_started",
    "goals_overall",
    "assists_overall",
    "average_rating_overall",
    "xg_per_game_overall",
    "shots_per_90_overall",
    "shots_on_target_per_90_overall",
    "key_passes_per_90_overall",
    "tackles_per_90_overall",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_file(path: Path) -> dict[str, Any] | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return {
        "path": path,
        "file": path.name,
        "host_slug": match.group("host"),
        "kind": match.group("kind").lower(),
        "season": int(match.group("start")),
        "end_year": int(match.group("end")),
        "duplicate_suffix": match.group("dup") or "",
    }


def missing_columns(cols: list[str], wanted: list[str]) -> list[str]:
    have = set(cols)
    return [col for col in wanted if col not in have]


def pct_nonnull(df: pd.DataFrame, cols: list[str]) -> float:
    present = [col for col in cols if col in df.columns]
    if not present:
        return 0.0
    return float(df[present].notna().mean().mean())


def norm_team(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+national\s+team$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def audit_file(path: Path) -> dict[str, Any]:
    parsed = parse_file(path)
    if parsed is None:
        return {}
    df = read_csv(path)
    cols = list(df.columns)
    kind = parsed["kind"]
    row: dict[str, Any] = {
        **{k: v for k, v in parsed.items() if k != "path"},
        "rows": len(df),
        "columns": len(cols),
        "sha256": sha256(path),
    }
    if kind == "matches":
        row.update(
            {
                "missing_required": "|".join(missing_columns(cols, MATCH_REQUIRED)),
                "missing_prematch": "|".join(missing_columns(cols, MATCH_PREMATCH)),
                "missing_actual_stats": "|".join(missing_columns(cols, MATCH_ACTUAL_STATS)),
                "required_nonnull_rate": pct_nonnull(df, MATCH_REQUIRED),
                "prematch_nonnull_rate": pct_nonnull(df, MATCH_PREMATCH),
                "actual_stats_nonnull_rate": pct_nonnull(df, MATCH_ACTUAL_STATS),
                "unique_home_teams": df.get("home_team_name", pd.Series(dtype=object)).nunique(dropna=True),
                "unique_away_teams": df.get("away_team_name", pd.Series(dtype=object)).nunique(dropna=True),
                "complete_status_rows": int(df.get("status", pd.Series(dtype=object)).astype(str).str.lower().eq("complete").sum()),
                "knockout_stage_label_gap_rows": int(df.get("Game Week", pd.Series(dtype=object)).isna().sum()) if "Game Week" in df.columns else len(df),
            }
        )
        if "timestamp" in df.columns:
            dt = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            row["min_match_date"] = dt.min().date().isoformat() if dt.notna().any() else ""
            row["max_match_date"] = dt.max().date().isoformat() if dt.notna().any() else ""
        elif "date_GMT" in df.columns:
            dt = pd.to_datetime(df["date_GMT"], errors="coerce")
            row["min_match_date"] = dt.min().date().isoformat() if dt.notna().any() else ""
            row["max_match_date"] = dt.max().date().isoformat() if dt.notna().any() else ""
    elif kind == "teams":
        row.update(
            {
                "missing_required": "|".join(missing_columns(cols, TEAM_REQUIRED)),
                "required_nonnull_rate": pct_nonnull(df, TEAM_REQUIRED),
                "unique_teams": df.get("common_name", df.get("team_name", pd.Series(dtype=object))).nunique(dropna=True),
                "total_team_matches_played": int(pd.to_numeric(df.get("matches_played", pd.Series(dtype=object)), errors="coerce").sum()),
            }
        )
    elif kind == "players":
        team_col = "Current Club"
        team_values = df.get(team_col, pd.Series(dtype=object)).dropna().map(norm_team)
        row.update(
            {
                "missing_required": "|".join(missing_columns(cols, PLAYER_REQUIRED)),
                "required_nonnull_rate": pct_nonnull(df, PLAYER_REQUIRED),
                "unique_players": df.get("full_name", pd.Series(dtype=object)).nunique(dropna=True),
                "unique_player_teams": team_values.nunique(dropna=True),
                "players_with_minutes": int((pd.to_numeric(df.get("minutes_played_overall", pd.Series(dtype=object)), errors="coerce").fillna(0) > 0).sum()),
                "players_with_rating": int(pd.to_numeric(df.get("average_rating_overall", pd.Series(dtype=object)), errors="coerce").notna().sum()),
            }
        )
    return row


def choose_canonical(files: pd.DataFrame) -> pd.DataFrame:
    if files.empty:
        return files
    out = files.copy()
    out["_dup_rank"] = out["duplicate_suffix"].astype(str).replace("", "0").astype(int)
    out["_has_clean_name"] = out["duplicate_suffix"].astype(str).eq("").astype(int)
    out = out.sort_values(["season", "kind", "_has_clean_name", "_dup_rank"], ascending=[True, True, False, True])
    out["canonical_file"] = False
    keep_idx = out.groupby(["season", "kind"], dropna=False).head(1).index
    out.loc[keep_idx, "canonical_file"] = True
    return out.drop(columns=["_dup_rank", "_has_clean_name"])


def season_readiness(files: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if files.empty:
        return pd.DataFrame()
    canonical = files[files["canonical_file"]].copy()
    for season, group in canonical.groupby("season"):
        by_kind = {str(row["kind"]): row for _, row in group.iterrows()}
        matches = by_kind.get("matches")
        teams = by_kind.get("teams")
        players = by_kind.get("players")
        reasons: list[str] = []
        status = "READY_FOR_WORLD_CUP_RESEARCH_ESTATE"
        if matches is None:
            reasons.append("missing_matches")
        elif int(matches.get("rows") or 0) != 64:
            reasons.append("match_rows_not_64")
        elif str(matches.get("missing_required") or ""):
            reasons.append("match_required_columns_missing")
        if teams is None:
            reasons.append("missing_teams")
        elif int(teams.get("rows") or 0) != 32:
            reasons.append("team_rows_not_32")
        if players is None:
            reasons.append("missing_players")
        elif int(players.get("rows") or 0) < 700:
            reasons.append("low_player_rows")
        if matches is not None and int(matches.get("knockout_stage_label_gap_rows") or 0) > 0:
            reasons.append("knockout_round_labels_need_inference")
        if reasons:
            status = "READY_WITH_REVIEW_NOTES" if not any(r.startswith("missing") for r in reasons) else "BLOCKED_MISSING_FILE"
        rows.append(
            {
                "season": int(season),
                "readiness_status": status,
                "notes": "|".join(reasons),
                "match_rows": int(matches.get("rows") or 0) if matches is not None else 0,
                "team_rows": int(teams.get("rows") or 0) if teams is not None else 0,
                "player_rows": int(players.get("rows") or 0) if players is not None else 0,
                "players_with_minutes": int(players.get("players_with_minutes") or 0) if players is not None else 0,
                "match_required_nonnull_rate": float(matches.get("required_nonnull_rate") or 0) if matches is not None else 0,
                "match_prematch_nonnull_rate": float(matches.get("prematch_nonnull_rate") or 0) if matches is not None else 0,
                "match_actual_stats_nonnull_rate": float(matches.get("actual_stats_nonnull_rate") or 0) if matches is not None else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("season")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", default=str(DEFAULT_DROP))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    drop = Path(args.drop)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(path for path in drop.glob("international-fifa-world-cup*.csv") if parse_file(path))
    file_rows = [audit_file(path) for path in paths]
    files = pd.DataFrame([row for row in file_rows if row])
    files = choose_canonical(files)
    readiness = season_readiness(files)

    duplicate_summary = pd.DataFrame()
    if not files.empty:
        duplicate_summary = (
            files.groupby(["season", "kind", "sha256"], dropna=False)
            .agg(files=("file", lambda s: " | ".join(sorted(s))), count=("file", "size"))
            .reset_index()
            .sort_values(["season", "kind", "count"], ascending=[True, True, False])
        )
        duplicate_summary = duplicate_summary[duplicate_summary["count"] > 1].copy()

    schema_rows: list[dict[str, Any]] = []
    for _, row in files[files["canonical_file"]].iterrows():
        path = drop / str(row["file"])
        frame = pd.read_csv(path, nrows=0)
        for col in frame.columns:
            schema_rows.append({"season": row["season"], "kind": row["kind"], "column": col})
    schema = pd.DataFrame(schema_rows)

    files.to_csv(outdir / "footystats_world_cup_file_inventory.csv", index=False)
    readiness.to_csv(outdir / "footystats_world_cup_season_readiness.csv", index=False)
    duplicate_summary.to_csv(outdir / "footystats_world_cup_duplicate_files.csv", index=False)
    schema.to_csv(outdir / "footystats_world_cup_schema_columns.csv", index=False)

    summary = [
        "# FootyStats World Cup Drop Audit",
        "",
        f"Source drop: `{drop}`",
        "",
        "## Readiness",
        markdown_table(readiness),
        "",
        "## Canonical Files",
        markdown_table(files[files["canonical_file"]][["season", "kind", "file", "rows", "columns", "missing_required"]]),
        "",
        "## Duplicate Files",
        markdown_table(duplicate_summary[["season", "kind", "count", "files"]] if not duplicate_summary.empty else duplicate_summary),
        "",
        "## Key Interpretation",
        "- The match files include canonical market odds for FTR, BTTS, OU1.5, and OU2.5 plus final goals, xG, shots, corners, cards, possession, stadium, referee, and pre-match percentage fields.",
        "- Team and player files are tournament-level aggregate exports. They are useful for player/team intelligence, but they must be converted into lagged pre-match snapshots before walk-forward modeling to avoid future leakage.",
        "- `Game Week` labels cover group matches, but knockout rows are blank and need deterministic round inference from match order/date.",
        "- Existing `footystats_drop_ingest.py` will not ingest these files yet because `international-fifa-world-cup-*` is not in the approved league map and historical years are outside the latest-season guard.",
        "",
        "## Outputs",
        f"- `{outdir / 'footystats_world_cup_file_inventory.csv'}`",
        f"- `{outdir / 'footystats_world_cup_season_readiness.csv'}`",
        f"- `{outdir / 'footystats_world_cup_duplicate_files.csv'}`",
        f"- `{outdir / 'footystats_world_cup_schema_columns.csv'}`",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Files: {len(files)}")
    print(f"Canonical files: {int(files['canonical_file'].sum()) if not files.empty else 0}")
    print(f"Seasons: {len(readiness)}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
