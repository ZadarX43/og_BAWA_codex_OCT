#!/usr/bin/env python3
"""Audit the FootyStats `World Cup Additions` folder against 2026 teams."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_ADDITIONS = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/World Cup Additions")
DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_BASE_COVERAGE = Path("reports/latest/world_cup_2026_coverage_gaps_2026_05_19/world_cup_2026_team_coverage_gaps.csv")
DEFAULT_OUTDIR = Path("reports/latest/world_cup_additions_coverage_audit_2026_05_19")

FILE_RE = re.compile(
    r"^international-(?P<competition>.+)-(?P<kind>league|matches|teams|players)-"
    r"(?P<start>\d{4})-to-(?P<end>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$"
)

TEAM_SLUG_ALIASES = {
    "cape_verde": "cape_verde",
    "cape_verde_islands": "cape_verde",
    "congo_dr": "dr_congo",
    "dr_congo": "dr_congo",
    "curacao": "curacao",
    "cura_ao": "curacao",
    "czechia": "czech_republic",
    "korea_republic": "south_korea",
    "south_korea": "south_korea",
    "cote_d_ivoire": "ivory_coast",
    "ivory_coast": "ivory_coast",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
    "united_states": "usa",
    "united_states_men_s": "usa",
    "united_states_men_s_national": "usa",
    "usmnt": "usa",
    "usa": "usa",
}

TARGET_COMPETITIONS = {
    "afc-asian-cup": "AFC_ASIAN_CUP",
    "african-nations-championship": "AFRICAN_NATIONS_CHAMPIONSHIP",
    "asian-cup-qualification": "ASIAN_CUP_QUALIFICATION",
    "concacaf-gold-cup": "CONCACAF_GOLD_CUP",
    "concacaf-gold-cup-qualification": "CONCACAF_GOLD_CUP_QUALIFICATION",
    "concacaf-nations-league": "CONCACAF_NATIONS_LEAGUE",
    "copa-america": "COPA_AMERICA",
    "international-friendlies": "INTERNATIONAL_FRIENDLIES",
    "wc-qualification-asia": "WC_QUALIFICATION_ASIA",
    "wc-qualification-concacaf": "WC_QUALIFICATION_CONCACAF",
    "wc-qualification-intercontinental-playoffs": "WC_QUALIFICATION_INTERCONTINENTAL_PLAYOFFS",
}

TEAM_NUMERIC_COLS = [
    "matches_played",
    "points_per_game",
    "wins",
    "draws",
    "losses",
    "goals_scored",
    "goals_conceded",
    "goal_difference",
    "goals_scored_per_match",
    "goals_conceded_per_match",
    "xg_for_avg_overall",
    "xg_against_avg_overall",
]

PLAYER_NUMERIC_COLS = [
    "minutes_played_overall",
    "appearances_overall",
    "goals_overall",
    "assists_overall",
    "average_rating_overall",
    "market_value",
    "xg_total_overall",
    "xa_total_overall",
    "games_started",
]


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"national team", "", text)
    text = re.sub(r"fc$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team_slug(value: object) -> str:
    slug = slugify(value)
    return TEAM_SLUG_ALIASES.get(slug, slug)


def scheduled_teams(path: Path) -> pd.DataFrame:
    launch = pd.read_csv(path, low_memory=False)
    frames = []
    for side in ["home", "away"]:
        frames.append(
            launch[[f"api_{side}_team_name", f"{side}_team_slug"]].rename(
                columns={f"api_{side}_team_name": "team_name", f"{side}_team_slug": "team_slug"}
            )
        )
    teams = pd.concat(frames, ignore_index=True)
    teams["team_slug"] = teams["team_slug"].map(canonical_team_slug)
    return teams.drop_duplicates("team_slug").sort_values("team_slug").reset_index(drop=True)


def iter_files(root: Path) -> list[dict[str, object]]:
    out = []
    seen_hashes = set()
    for path in sorted(root.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        content_hash = hash(path.read_bytes())
        duplicate_content = content_hash in seen_hashes
        seen_hashes.add(content_hash)
        meta = match.groupdict()
        meta["path"] = str(path)
        meta["file_name"] = path.name
        meta["start"] = int(meta["start"])
        meta["end"] = int(meta["end"])
        meta["competition_group"] = TARGET_COMPETITIONS.get(meta["competition"], meta["competition"].upper().replace("-", "_"))
        meta["duplicate_content"] = int(duplicate_content)
        try:
            meta["rows"] = len(pd.read_csv(path, usecols=[0], low_memory=False))
        except Exception:
            meta["rows"] = -1
        out.append(meta)
    return out


def compact_competitions(values: pd.Series, limit: int = 8) -> str:
    unique = sorted(v for v in values.dropna().astype(str).unique() if v)
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";+{len(unique) - limit}_more"
    return ";".join(unique)


def build_team_coverage(files: list[dict[str, object]]) -> pd.DataFrame:
    frames = []
    for meta in files:
        if meta["kind"] != "teams" or meta["duplicate_content"]:
            continue
        df = pd.read_csv(meta["path"], low_memory=False)
        name = df.get("common_name", df.get("team_name"))
        if name is None:
            continue
        local = pd.DataFrame(
            {
                "team_slug": name.map(canonical_team_slug),
                "source_team_name": name,
                "competition": meta["competition"],
                "competition_group": meta["competition_group"],
                "start": meta["start"],
                "end": meta["end"],
            }
        )
        for col in TEAM_NUMERIC_COLS:
            if col in df.columns:
                local[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True, sort=False)
    agg = rows.groupby("team_slug", dropna=False).agg(
        additions_team_rows=("team_slug", "size"),
        additions_team_competitions=("competition_group", compact_competitions),
        additions_team_latest_end=("end", "max"),
        additions_team_matches_played=("matches_played", "sum"),
        additions_team_avg_ppg=("points_per_game", "mean"),
        additions_team_goal_diff=("goal_difference", "sum"),
    ).reset_index()
    return agg


def build_player_coverage(files: list[dict[str, object]]) -> pd.DataFrame:
    frames = []
    for meta in files:
        if meta["kind"] != "players" or meta["duplicate_content"]:
            continue
        df = pd.read_csv(meta["path"], low_memory=False)
        team_source = df.get("Current Club", df.get("nationality"))
        if team_source is None:
            continue
        local = pd.DataFrame(
            {
                "team_slug": team_source.map(canonical_team_slug),
                "competition": meta["competition"],
                "competition_group": meta["competition_group"],
                "start": meta["start"],
                "end": meta["end"],
                "player_name": df.get("full_name"),
            }
        )
        for col in PLAYER_NUMERIC_COLS:
            if col in df.columns:
                local[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows["_rating"] = pd.to_numeric(rows.get("average_rating_overall"), errors="coerce")
    rows["_market_value"] = pd.to_numeric(rows.get("market_value"), errors="coerce")
    agg = rows.groupby("team_slug", dropna=False).agg(
        additions_player_rows=("team_slug", "size"),
        additions_named_players=("player_name", lambda s: int(s.notna().sum())),
        additions_rated_players=("_rating", lambda s: int(s.notna().sum())),
        additions_player_competitions=("competition_group", compact_competitions),
        additions_player_latest_end=("end", "max"),
        additions_top11_avg_rating=("_rating", lambda s: s.dropna().sort_values(ascending=False).head(11).mean()),
        additions_top5_avg_rating=("_rating", lambda s: s.dropna().sort_values(ascending=False).head(5).mean()),
        additions_squad_market_value=("_market_value", "sum"),
        additions_player_minutes=("minutes_played_overall", "sum"),
        additions_player_goals=("goals_overall", "sum"),
        additions_player_xg=("xg_total_overall", "sum"),
        additions_player_xa=("xa_total_overall", "sum"),
    ).reset_index()
    return agg


def build_match_coverage(files: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for meta in files:
        if meta["kind"] != "matches" or meta["duplicate_content"]:
            continue
        df = pd.read_csv(meta["path"], low_memory=False)
        if "home_team_name" not in df.columns or "away_team_name" not in df.columns:
            continue
        for side in ["home", "away"]:
            goals_for = f"{side}_team_goal_count"
            goals_against = "away_team_goal_count" if side == "home" else "home_team_goal_count"
            local = pd.DataFrame(
                {
                    "team_slug": df[f"{side}_team_name"].map(canonical_team_slug),
                    "competition_group": meta["competition_group"],
                    "end": meta["end"],
                    "match_rows": 1,
                    "goals_for": pd.to_numeric(df.get(goals_for), errors="coerce"),
                    "goals_against": pd.to_numeric(df.get(goals_against), errors="coerce"),
                }
            )
            rows.append(local)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True, sort=False)
    return data.groupby("team_slug", dropna=False).agg(
        additions_match_rows=("match_rows", "sum"),
        additions_match_competitions=("competition_group", compact_competitions),
        additions_match_latest_end=("end", "max"),
        additions_match_goals_for=("goals_for", "sum"),
        additions_match_goals_against=("goals_against", "sum"),
    ).reset_index()


def write_summary(outdir: Path, inventory: pd.DataFrame, coverage: pd.DataFrame) -> None:
    comp = inventory.groupby(["competition_group", "kind"], dropna=False).agg(
        files=("file_name", "count"),
        non_duplicate_files=("duplicate_content", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).sum())),
        rows=("rows", "sum"),
        min_start=("start", "min"),
        max_end=("end", "max"),
    ).reset_index()
    comp.to_csv(outdir / "world_cup_additions_competition_inventory.csv", index=False)
    gap_cols = [
        "team_name",
        "team_slug",
        "player_prior_ready",
        "additions_player_rows",
        "additions_rated_players",
        "additions_player_competitions",
        "qualifier_2026_ready",
        "additions_team_rows",
        "additions_team_competitions",
        "additions_match_rows",
        "additions_match_competitions",
    ]
    fillable_players = coverage[
        coverage["player_prior_ready"].fillna(0).astype(int).eq(0)
        & coverage["additions_player_rows"].fillna(0).gt(0)
    ][gap_cols]
    fillable_players.to_csv(outdir / "world_cup_additions_player_gap_closers.csv", index=False)
    host_context = coverage[coverage["team_slug"].isin(["canada", "mexico", "usa"])][gap_cols]
    host_context.to_csv(outdir / "world_cup_additions_host_context_coverage.csv", index=False)

    lines = [
        "# World Cup Additions Coverage Audit",
        "",
        "Research-only audit of `/Users/hughwade/Desktop/FOOTYSTATS_DROP/World Cup Additions`.",
        "",
        "## Source Inventory",
        "",
        f"- Files parsed: {len(inventory)}",
        f"- Non-duplicate files: {int(inventory['duplicate_content'].eq(0).sum())}",
        f"- Duplicate-content files: {int(inventory['duplicate_content'].eq(1).sum())}",
        f"- Competitions: {inventory['competition_group'].nunique()}",
        "",
        "## 2026 Team Coverage Lift",
        "",
        f"- Scheduled teams with additions team rows: {int(coverage['additions_team_rows'].fillna(0).gt(0).sum())} / {len(coverage)}",
        f"- Scheduled teams with additions player rows: {int(coverage['additions_player_rows'].fillna(0).gt(0).sum())} / {len(coverage)}",
        f"- Scheduled teams with additions rated players: {int(coverage['additions_rated_players'].fillna(0).gt(0).sum())} / {len(coverage)}",
        f"- Previously missing player-prior teams now fillable from additions: {len(fillable_players)}",
        "",
        "## Player Gap Closers",
        "",
    ]
    if fillable_players.empty:
        lines.append("- None found.")
    else:
        for row in fillable_players.itertuples(index=False):
            lines.append(
                f"- {row.team_name}: player rows={int(row.additions_player_rows)}, "
                f"rated={int(row.additions_rated_players)}, comps={row.additions_player_competitions}"
            )
    lines.extend(
        [
            "",
            "## Host Context Coverage",
            "",
        ]
    )
    for row in host_context.itertuples(index=False):
        lines.append(
            f"- {row.team_name}: team rows={int(row.additions_team_rows or 0)}, "
            f"player rows={int(row.additions_player_rows or 0)}, "
            f"team comps={row.additions_team_competitions}, player comps={row.additions_player_competitions}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Additions should become a separate pre-tournament sidecar, not overwrite World Cup qualifier aggregates.",
            "- Friendlies and regional tournaments are useful for host/form/player intelligence, but should carry source-competition flags.",
            "- Player rows can fill current squad-quality proxies while official 2026 squads/API player data remains partial.",
        ]
    )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additions-root", type=Path, default=DEFAULT_ADDITIONS)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--base-coverage", type=Path, default=DEFAULT_BASE_COVERAGE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    files = iter_files(args.additions_root)
    inventory = pd.DataFrame(files)
    teams = scheduled_teams(args.launch_scaffold)
    if args.base_coverage.exists():
        base = pd.read_csv(args.base_coverage, low_memory=False)
        base["team_slug"] = base["team_slug"].map(canonical_team_slug)
        teams = teams.merge(
            base[
                [
                    "team_slug",
                    "qualifier_2026_ready",
                    "player_prior_ready",
                    "fjelstul_historical_ready",
                    "fjelstul_recent_ready",
                ]
            ],
            on="team_slug",
            how="left",
        )
    team_cov = build_team_coverage(files)
    player_cov = build_player_coverage(files)
    match_cov = build_match_coverage(files)
    coverage = teams.merge(team_cov, on="team_slug", how="left")
    coverage = coverage.merge(player_cov, on="team_slug", how="left")
    coverage = coverage.merge(match_cov, on="team_slug", how="left")
    inventory.to_csv(args.outdir / "world_cup_additions_file_inventory.csv", index=False)
    coverage.to_csv(args.outdir / "world_cup_additions_2026_team_coverage.csv", index=False)
    write_summary(args.outdir, inventory, coverage)
    print(f"[ok] files={len(inventory)} teams={len(coverage)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
