#!/usr/bin/env python3
"""Report World Cup 2026 sidecar coverage gaps and next actions."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_QUALIFIER = Path("data_sources/footystats_world_cup/qualification_foundation/world_cup_2026_qualifier_latest_team_features.csv")
DEFAULT_PLAYER = Path("data_sources/footystats_world_cup/player_intelligence_2026/world_cup_2026_team_player_intelligence_scaffold.csv")
DEFAULT_HISTORY = Path("data_sources/footystats_world_cup/fjelstul_historical_priors/world_cup_2026_team_historical_prior_sidecar.csv")
DEFAULT_MATRIX = Path("data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_matrix.csv")
DEFAULT_OUTDIR = Path("reports/latest/world_cup_2026_coverage_gaps_2026_05_19")

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
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
    "united_states": "usa",
    "usmnt": "usa",
    "usa": "usa",
}

HOSTS = {"canada", "mexico", "usa"}


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team_slug(value: object) -> str:
    slug = slugify(value)
    return TEAM_SLUG_ALIASES.get(slug, slug)


def team_list(launch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["home", "away"]:
        local = launch[[f"api_{side}_team_name", f"{side}_team_slug"]].rename(
            columns={f"api_{side}_team_name": "team_name", f"{side}_team_slug": "team_slug"}
        )
        rows.append(local)
    out = pd.concat(rows, ignore_index=True)
    out["team_slug"] = out["team_slug"].map(canonical_team_slug)
    out["team_name"] = out["team_name"].fillna(out["team_slug"])
    return out.drop_duplicates("team_slug").sort_values("team_slug").reset_index(drop=True)


def merge_qualifier(teams: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        teams["qualifier_2026_ready"] = 0
        return teams
    df = pd.read_csv(path, low_memory=False)
    df["team_slug"] = df["team_slug"].map(canonical_team_slug)
    keep = [
        "team_slug",
        "footystats_qualifier_2026_joined_flag",
        "matches_played",
        "points_per_game",
        "goal_difference",
    ]
    out = teams.merge(df[[c for c in keep if c in df.columns]], on="team_slug", how="left")
    out["qualifier_2026_ready"] = out["footystats_qualifier_2026_joined_flag"].fillna(0).astype(int)
    return out


def merge_player(teams: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        teams["player_prior_ready"] = 0
        return teams
    df = pd.read_csv(path, low_memory=False)
    df["team_slug"] = df["team_slug"].map(canonical_team_slug)
    if "team_name" in df.columns:
        df = df.sort_values("team_name", na_position="last").drop_duplicates("team_slug", keep="first")
    keep = ["team_slug", "team_name", "player_intel_status"]
    out = teams.merge(df[[c for c in keep if c in df.columns]], on="team_slug", how="left", suffixes=("", "_player"))
    out["player_intel_status"] = out["player_intel_status"].fillna("MISSING_PLAYER_PRIOR")
    out["player_prior_ready"] = out["player_intel_status"].ne("MISSING_PLAYER_PRIOR").astype(int)
    return out


def merge_history(teams: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        teams["fjelstul_historical_ready"] = 0
        teams["fjelstul_recent_ready"] = 0
        return teams
    df = pd.read_csv(path, low_memory=False)
    df["team_slug"] = df["team_slug"].map(canonical_team_slug)
    keep = [
        "team_slug",
        "fjelstul_historical_prior_ready_flag",
        "fjelstul_recent_prior_ready_flag",
        "all_wc_matches",
        "recent_wc_matches",
    ]
    out = teams.merge(df[[c for c in keep if c in df.columns]], on="team_slug", how="left")
    out["fjelstul_historical_ready"] = out["fjelstul_historical_prior_ready_flag"].fillna(0).astype(int)
    out["fjelstul_recent_ready"] = out["fjelstul_recent_prior_ready_flag"].fillna(0).astype(int)
    return out


def action_for(row: pd.Series) -> str:
    gaps = []
    if int(row["qualifier_2026_ready"]) == 0:
        gaps.append("host_auto_or_missing_qualifier")
    if int(row["player_prior_ready"]) == 0:
        gaps.append("player_squad_prior")
    if int(row["fjelstul_historical_ready"]) == 0:
        gaps.append("world_cup_history")
    if not gaps:
        return "ready_for_research_matrix"
    if set(gaps) == {"host_auto_or_missing_qualifier"} and row["team_slug"] in HOSTS:
        return "use_host_competitive_context_gold_cup_nations_league_friendlies"
    if "player_squad_prior" in gaps:
        return "refresh_api_players_squads_and_add_aliases_or_external_player_rating_template"
    if "world_cup_history" in gaps:
        return "use_qualifier_and_confederation_priors_no_historical_wc_prior"
    return "manual_source_review"


def write_summary(outdir: Path, coverage: pd.DataFrame, matrix: pd.DataFrame | None) -> None:
    def count(col: str) -> int:
        return int(coverage[col].fillna(0).astype(int).sum())

    missing_player = ", ".join(coverage.loc[coverage["player_prior_ready"].eq(0), "team_name"].tolist()) or "None"
    missing_qual = ", ".join(coverage.loc[coverage["qualifier_2026_ready"].eq(0), "team_name"].tolist()) or "None"
    missing_hist = ", ".join(coverage.loc[coverage["fjelstul_historical_ready"].eq(0), "team_name"].tolist()) or "None"
    missing_recent = ", ".join(coverage.loc[coverage["fjelstul_recent_ready"].eq(0), "team_name"].tolist()) or "None"

    lines = [
        "# World Cup 2026 Coverage Gaps",
        "",
        "Research-only coverage view for current 2026 World Cup sidecars.",
        "",
        "## Team Coverage",
        "",
        f"- Scheduled teams: {len(coverage)}",
        f"- FootyStats 2026 qualifier aggregate coverage: {count('qualifier_2026_ready')} / {len(coverage)}",
        f"- Player/squad prior coverage: {count('player_prior_ready')} / {len(coverage)}",
        f"- Fjelstul all-time World Cup history coverage: {count('fjelstul_historical_ready')} / {len(coverage)}",
        f"- Fjelstul recent 2010-2018 World Cup history coverage: {count('fjelstul_recent_ready')} / {len(coverage)}",
        "",
        "## Fixture Coverage",
        "",
    ]
    if matrix is not None:
        lines.extend(
            [
                f"- Scheduled fixtures in API group-stage scaffold: {len(matrix)}",
                f"- Both-side qualifier coverage: {int(matrix['qualifier_form_both_sides_flag'].sum())} / {len(matrix)}",
                f"- Both-side Fjelstul historical coverage: {int(matrix['fjelstul_historical_both_sides_flag'].sum())} / {len(matrix)}",
                f"- Both-side Fjelstul recent coverage: {int(matrix['fjelstul_recent_both_sides_flag'].sum())} / {len(matrix)}",
            ]
        )
    else:
        lines.append("- Fixture research matrix not found.")
    lines.extend(
        [
            "",
            "## Missing By Layer",
            "",
            f"- Missing 2026 qualifier aggregates: {missing_qual}",
            f"- Missing player/squad priors: {missing_player}",
            f"- Missing all-time World Cup history: {missing_hist}",
            f"- Missing recent 2010-2018 World Cup history: {missing_recent}",
            "",
            "## Solve Path",
            "",
            "1. Treat Canada, Mexico, and USA qualifier gaps as host-auto contexts, not failed joins.",
            "2. Refresh API-Football 2026 players/squads/injuries repeatedly as squads publish.",
            "3. Patch country aliases in player intelligence outputs so Curaçao, Türkiye, Cape Verde, DR Congo, and USA resolve consistently.",
            "4. Fill missing player/squad priors with the external player rating template until official squad/player data is complete.",
            "5. For teams with no historical World Cup priors, use qualifier strength, confederation priors, player ratings, market odds, and venue/weather context instead.",
            "6. Build host-form sidecars from Gold Cup, Nations League, Copa América/CONCACAF, friendlies, and API head-to-head data.",
        ]
    )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--qualifier", type=Path, default=DEFAULT_QUALIFIER)
    parser.add_argument("--player", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    coverage = team_list(launch)
    coverage = merge_qualifier(coverage, args.qualifier)
    coverage = merge_player(coverage, args.player)
    coverage = merge_history(coverage, args.history)
    coverage["is_host"] = coverage["team_slug"].isin(HOSTS).astype(int)
    coverage["coverage_action"] = coverage.apply(action_for, axis=1)
    matrix = pd.read_csv(args.matrix, low_memory=False) if args.matrix.exists() else None
    coverage.to_csv(args.outdir / "world_cup_2026_team_coverage_gaps.csv", index=False)
    write_summary(args.outdir, coverage, matrix)
    print(f"[ok] teams={len(coverage)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
