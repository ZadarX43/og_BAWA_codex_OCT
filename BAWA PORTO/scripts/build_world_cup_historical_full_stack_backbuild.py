#!/usr/bin/env python3
"""Backbuild timestamp-safe World Cup historical intelligence sidecars.

This creates 2018/2022 fixture-level research sidecars for recent national-team
history, qualifier-style context, local H2H, and venue scaffolding. It only uses
addition matches strictly before each fixture kickoff.

Player aggregate backbuild is intentionally documented as a gap unless a
timestamp-safe pre-tournament player snapshot source is supplied.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_ADDITIONS_MATCH_SPINE = Path(
    "data_sources/footystats_world_cup/recent_history_h2h_2026/world_cup_additions_match_spine_for_h2h.csv"
)
DEFAULT_FOOTYSTATS_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/historical_full_stack_backbuild")

COMPETITION_HINTS = {
    "wc-qualification-africa": "WC_QUALIFICATION_AFRICA",
    "wc-qualification-asia": "WC_QUALIFICATION_ASIA",
    "wc-qualification-concacaf": "WC_QUALIFICATION_CONCACAF",
    "wc-qualification-europe": "WC_QUALIFICATION_EUROPE",
    "wc-qualification-south-america": "WC_QUALIFICATION_SOUTH_AMERICA",
    "wc-qualification-oceania": "WC_QUALIFICATION_OFC",
    "wc-qualification-ofc": "WC_QUALIFICATION_OFC",
    "wc-qualification-intercontinental-playoffs": "WC_QUALIFICATION_INTERCONTINENTAL_PLAYOFFS",
    "fifa-world-cup": "WORLD_CUP",
    "international-friendlies": "INTERNATIONAL_FRIENDLIES",
    "concacaf-nations-league": "CONCACAF_NATIONS_LEAGUE",
    "concacaf-gold-cup-qualification": "CONCACAF_GOLD_CUP_QUALIFICATION",
    "concacaf-gold-cup": "CONCACAF_GOLD_CUP",
    "copa-america": "COPA_AMERICA",
    "asian-cup-qualification": "ASIAN_CUP_QUALIFICATION",
    "afc-asian-cup": "AFC_ASIAN_CUP",
    "african-nations-championship": "AFRICAN_NATIONS_CHAMPIONSHIP",
}


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.to_numeric(df["season"], errors="coerce").isin([2018, 2022])].copy()
    df["fixture_timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["home_team_slug"] = df["home_team_name"].map(slugify)
    df["away_team_slug"] = df["away_team_name"].map(slugify)
    return df.sort_values(["season", "fixture_timestamp", "fixture_key"]).reset_index(drop=True)


def load_additions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in [
        "timestamp",
        "home_goals",
        "away_goals",
        "total_goals",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "addition_source_end",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "home_team_slug", "away_team_slug"]).copy()


def competition_from_path(path: Path) -> str:
    name = path.name.lower()
    for hint, group in COMPETITION_HINTS.items():
        if hint in name:
            return group
    text = re.sub(r"^international-", "", name)
    text = re.sub(r"-matches-\d{4}-to-\d{4}-stats.*$", "", text)
    return slugify(text).upper()


def source_end_from_path(path: Path) -> int | None:
    m = re.search(r"-matches-(\d{4})-to-(\d{4})-stats", path.name.lower())
    if not m:
        return None
    return int(m.group(2))


def iter_footystats_match_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for path in root.rglob("international-*-matches-*-stats*.csv"):
        if "/archive/" in str(path):
            continue
        files.append(path)
    return sorted(files)


def load_footystats_drop_matches(root: Path) -> pd.DataFrame:
    frames = []
    for path in iter_footystats_match_files(root):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "home_team_name" not in df.columns or "away_team_name" not in df.columns:
            continue
        source_end = source_end_from_path(path)
        if source_end is None:
            continue
        local = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(df.get("timestamp"), errors="coerce"),
                "date_GMT": df.get("date_GMT"),
                "addition_competition": competition_from_path(path),
                "addition_source_end": source_end,
                "home_team_name": df["home_team_name"],
                "away_team_name": df["away_team_name"],
                "home_team_slug": df["home_team_name"].map(slugify),
                "away_team_slug": df["away_team_name"].map(slugify),
                "home_goals": pd.to_numeric(df.get("home_team_goal_count"), errors="coerce"),
                "away_goals": pd.to_numeric(df.get("away_team_goal_count"), errors="coerce"),
                "total_goals": pd.to_numeric(df.get("total_goal_count"), errors="coerce"),
                "home_shots": pd.to_numeric(df.get("home_team_shots"), errors="coerce"),
                "away_shots": pd.to_numeric(df.get("away_team_shots"), errors="coerce"),
                "home_sot": pd.to_numeric(df.get("home_team_shots_on_target"), errors="coerce"),
                "away_sot": pd.to_numeric(df.get("away_team_shots_on_target"), errors="coerce"),
                "source_file": str(path),
            }
        )
        local["btts"] = (local["home_goals"].gt(0) & local["away_goals"].gt(0)).astype(int)
        local["over25"] = local["total_goals"].gt(2).astype(int)
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["timestamp", "home_team_slug", "away_team_slug"])
    dedupe_cols = [
        "timestamp",
        "addition_competition",
        "home_team_slug",
        "away_team_slug",
        "home_goals",
        "away_goals",
    ]
    return out.drop_duplicates(dedupe_cols).reset_index(drop=True)


def load_backbuild_matches(spine_path: Path, drop_root: Path, source_mode: str) -> pd.DataFrame:
    frames = []
    if source_mode in {"spine", "both"} and spine_path.exists():
        frames.append(load_additions(spine_path))
    if source_mode in {"drop", "both"}:
        drop = load_footystats_drop_matches(drop_root)
        if not drop.empty:
            frames.append(drop)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_cols = [
        "timestamp",
        "addition_competition",
        "home_team_slug",
        "away_team_slug",
        "home_goals",
        "away_goals",
    ]
    return out.drop_duplicates(dedupe_cols).reset_index(drop=True)


def team_side_rows(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        local = pd.DataFrame(
            {
                "team_slug": matches[f"{side}_team_slug"],
                "opponent_slug": matches[f"{opp}_team_slug"],
                "timestamp": matches["timestamp"],
                "addition_competition": matches["addition_competition"],
                "addition_source_end": matches["addition_source_end"],
                "goals_for": matches[f"{side}_goals"],
                "goals_against": matches[f"{opp}_goals"],
                "shots_for": matches[f"{side}_shots"],
                "shots_against": matches[f"{opp}_shots"],
                "sot_for": matches[f"{side}_sot"],
                "sot_against": matches[f"{opp}_sot"],
                "btts": matches["btts"],
                "over25": matches["over25"],
            }
        )
        local["win"] = local["goals_for"].gt(local["goals_against"]).astype(int)
        local["draw"] = local["goals_for"].eq(local["goals_against"]).astype(int)
        local["loss"] = local["goals_for"].lt(local["goals_against"]).astype(int)
        local["goal_diff"] = local["goals_for"] - local["goals_against"]
        rows.append(local)
    return pd.concat(rows, ignore_index=True, sort=False)


def aggregate(prefix: str, rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            f"{prefix}_matches": 0,
            f"{prefix}_source_status": "NO_PRE_KICKOFF_ROWS",
        }
    return {
        f"{prefix}_matches": int(len(rows)),
        f"{prefix}_win_rate": float(rows["win"].mean()),
        f"{prefix}_draw_rate": float(rows["draw"].mean()),
        f"{prefix}_loss_rate": float(rows["loss"].mean()),
        f"{prefix}_goals_for_per_match": float(rows["goals_for"].mean()),
        f"{prefix}_goals_against_per_match": float(rows["goals_against"].mean()),
        f"{prefix}_goal_diff_per_match": float(rows["goal_diff"].mean()),
        f"{prefix}_btts_rate": float(rows["btts"].mean()),
        f"{prefix}_over25_rate": float(rows["over25"].mean()),
        f"{prefix}_shots_for_per_match": float(rows["shots_for"].mean()) if rows["shots_for"].notna().any() else np.nan,
        f"{prefix}_shots_against_per_match": float(rows["shots_against"].mean()) if rows["shots_against"].notna().any() else np.nan,
        f"{prefix}_sot_for_per_match": float(rows["sot_for"].mean()) if rows["sot_for"].notna().any() else np.nan,
        f"{prefix}_sot_against_per_match": float(rows["sot_against"].mean()) if rows["sot_against"].notna().any() else np.nan,
        f"{prefix}_latest_source_end": int(rows["addition_source_end"].max()) if rows["addition_source_end"].notna().any() else np.nan,
        f"{prefix}_source_status": "PRE_KICKOFF_ROWS_READY",
    }


def h2h_metrics(home: str, away: str, kickoff_ts: float, matches: pd.DataFrame) -> dict:
    pair = matches[
        (matches["timestamp"].lt(kickoff_ts))
        & (
            ((matches["home_team_slug"].eq(home)) & (matches["away_team_slug"].eq(away)))
            | ((matches["home_team_slug"].eq(away)) & (matches["away_team_slug"].eq(home)))
        )
    ].copy()
    if pair.empty:
        return {
            "historical_local_h2h_match_count": 0,
            "historical_local_h2h_source_status": "NO_PRE_KICKOFF_H2H",
        }
    home_goals = pair["home_goals"].where(pair["home_team_slug"].eq(home), pair["away_goals"])
    away_goals = pair["away_goals"].where(pair["home_team_slug"].eq(home), pair["home_goals"])
    return {
        "historical_local_h2h_match_count": int(len(pair)),
        "historical_local_h2h_home_side_win_rate": float(home_goals.gt(away_goals).mean()),
        "historical_local_h2h_draw_rate": float(home_goals.eq(away_goals).mean()),
        "historical_local_h2h_away_side_win_rate": float(home_goals.lt(away_goals).mean()),
        "historical_local_h2h_total_goals_per_match": float((home_goals + away_goals).mean()),
        "historical_local_h2h_btts_rate": float((home_goals.gt(0) & away_goals.gt(0)).mean()),
        "historical_local_h2h_over25_rate": float((home_goals + away_goals).gt(2).mean()),
        "historical_local_h2h_latest_source_end": int(pair["addition_source_end"].max()) if pair["addition_source_end"].notna().any() else np.nan,
        "historical_local_h2h_source_status": "PRE_KICKOFF_H2H_READY",
    }


def side_features(side_rows: pd.DataFrame, team_slug: str, kickoff_ts: float, lookback_days: int, side: str) -> dict:
    prior = side_rows[(side_rows["team_slug"].eq(team_slug)) & (side_rows["timestamp"].lt(kickoff_ts))].copy()
    lookback_start = kickoff_ts - lookback_days * 86400
    recent = prior[prior["timestamp"].ge(lookback_start)]
    qualifier = prior[prior["addition_competition"].astype(str).str.contains("WC_QUALIFICATION", na=False)]
    out = {}
    for prefix, frame in [
        (f"{side}_backbuilt_all_prior", prior),
        (f"{side}_backbuilt_recent", recent),
        (f"{side}_backbuilt_qualifier", qualifier),
    ]:
        out.update(aggregate(prefix, frame))
    return out


def add_deltas(row: dict, left_prefix: str, right_prefix: str, output_prefix: str) -> None:
    for metric in [
        "matches",
        "win_rate",
        "draw_rate",
        "goals_for_per_match",
        "goals_against_per_match",
        "goal_diff_per_match",
        "btts_rate",
        "over25_rate",
        "sot_for_per_match",
        "sot_against_per_match",
    ]:
        left = pd.to_numeric(pd.Series([row.get(f"{left_prefix}_{metric}")]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([row.get(f"{right_prefix}_{metric}")]), errors="coerce").iloc[0]
        row[f"{output_prefix}_{metric}_delta"] = left - right


def build_sidecar(fixtures: pd.DataFrame, matches: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    side_rows = team_side_rows(matches)
    rows = []
    for fx in fixtures.itertuples(index=False):
        row = {
            "fixture_key": fx.fixture_key,
            "season": fx.season,
            "match_date": fx.match_date,
            "home_team_name": fx.home_team_name,
            "away_team_name": fx.away_team_name,
            "historical_venue_name": fx.stadium_name,
            "historical_venue_known_flag": int(not pd.isna(fx.stadium_name)),
            "historical_venue_weather_truth_status": "NOT_BACKFILLED",
            "historical_player_intel_backbuild_status": "NEEDS_TIME_SAFE_PRE_TOURNAMENT_PLAYER_SNAPSHOT",
        }
        row.update(side_features(side_rows, fx.home_team_slug, fx.fixture_timestamp, lookback_days, "home"))
        row.update(side_features(side_rows, fx.away_team_slug, fx.fixture_timestamp, lookback_days, "away"))
        row.update(h2h_metrics(fx.home_team_slug, fx.away_team_slug, fx.fixture_timestamp, matches))
        add_deltas(row, "home_backbuilt_recent", "away_backbuilt_recent", "backbuilt_recent")
        add_deltas(row, "home_backbuilt_qualifier", "away_backbuilt_qualifier", "backbuilt_qualifier")
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(sidecar: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("recent_history_both_sides", "home_backbuilt_recent_matches", "away_backbuilt_recent_matches"),
        ("qualifier_history_both_sides", "home_backbuilt_qualifier_matches", "away_backbuilt_qualifier_matches"),
        ("all_prior_history_both_sides", "home_backbuilt_all_prior_matches", "away_backbuilt_all_prior_matches"),
    ]
    rows = []
    for name, home_col, away_col in checks:
        flag = pd.to_numeric(sidecar[home_col], errors="coerce").gt(0) & pd.to_numeric(sidecar[away_col], errors="coerce").gt(0)
        rows.append({"coverage_check": name, "fixtures": int(flag.sum()), "total": len(sidecar), "coverage_rate": float(flag.mean())})
    h2h = pd.to_numeric(sidecar["historical_local_h2h_match_count"], errors="coerce").gt(0)
    rows.append({"coverage_check": "local_h2h_available", "fixtures": int(h2h.sum()), "total": len(sidecar), "coverage_rate": float(h2h.mean())})
    venue = pd.to_numeric(sidecar["historical_venue_known_flag"], errors="coerce").gt(0)
    rows.append({"coverage_check": "historical_venue_known", "fixtures": int(venue.sum()), "total": len(sidecar), "coverage_rate": float(venue.mean())})
    return pd.DataFrame(rows)


def competition_inventory(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["addition_competition", "addition_source_end", "rows"])
    return (
        matches.groupby(["addition_competition", "addition_source_end"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["addition_competition", "addition_source_end"])
    )


def gap_manifest(sidecar: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "player_additions",
                "status": "NOT_MODEL_READY",
                "reason": "Available player files are aggregate exports; need a source timestamp or pre-tournament snapshot for each 2018/2022 fixture.",
                "affected_fixtures": len(sidecar),
            },
            {
                "layer": "weather_truth",
                "status": "NOT_BACKFILLED",
                "reason": "Historical venue names are available, but weather snapshots have not been joined.",
                "affected_fixtures": len(sidecar),
            },
            {
                "layer": "api_h2h",
                "status": "MANIFEST_NEEDED",
                "reason": "Local FootyStats H2H is available only where additions files contain a pre-kickoff meeting.",
                "affected_fixtures": int(pd.to_numeric(sidecar["historical_local_h2h_match_count"], errors="coerce").fillna(0).eq(0).sum()),
            },
        ]
    )


def write_summary(
    outdir: Path,
    sidecar: pd.DataFrame,
    coverage: pd.DataFrame,
    gaps: pd.DataFrame,
    inventory: pd.DataFrame,
    lookback_days: int,
    source_mode: str,
) -> None:
    lines = [
        "# Historical World Cup Full-Stack Backbuild",
        "",
        "Timestamp-safe research sidecar for testing full-stack World Cup ablations on 2018/2022 fixtures.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_historical_backbuilt_fixture_intelligence_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_historical_backbuilt_coverage.csv'}`",
        f"- `{outdir / 'world_cup_historical_backbuild_gap_manifest.csv'}`",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Source Inventory",
        "",
        markdown_table(inventory.head(80)),
        "",
        "## Known Gaps",
        "",
        markdown_table(gaps),
        "",
        "## Notes",
        "",
        f"- Source mode: {source_mode}.",
        f"- Recent-history lookback window: {lookback_days} days.",
        "- Every additions/history row is filtered to `source_timestamp < fixture_timestamp`.",
        "- This is a research input for ablation testing; it is not a production training artifact.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--additions-match-spine", type=Path, default=DEFAULT_ADDITIONS_MATCH_SPINE)
    parser.add_argument("--footystats-drop", type=Path, default=DEFAULT_FOOTYSTATS_DROP)
    parser.add_argument("--source-mode", choices=["spine", "drop", "both"], default="both")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--lookback-days", type=int, default=1460)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(args.merged)
    matches = load_backbuild_matches(args.additions_match_spine, args.footystats_drop, args.source_mode)
    sidecar = build_sidecar(fixtures, matches, args.lookback_days)
    coverage = coverage_table(sidecar)
    gaps = gap_manifest(sidecar)
    inventory = competition_inventory(matches)
    sidecar.to_csv(args.outdir / "world_cup_historical_backbuilt_fixture_intelligence_sidecar.csv", index=False)
    coverage.to_csv(args.outdir / "world_cup_historical_backbuilt_coverage.csv", index=False)
    gaps.to_csv(args.outdir / "world_cup_historical_backbuild_gap_manifest.csv", index=False)
    inventory.to_csv(args.outdir / "world_cup_historical_backbuilt_source_inventory.csv", index=False)
    write_summary(args.outdir, sidecar, coverage, gaps, inventory, args.lookback_days, args.source_mode)
    print(f"[ok] fixtures={len(sidecar)} coverage_checks={len(coverage)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
