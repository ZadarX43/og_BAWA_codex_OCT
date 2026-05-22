#!/usr/bin/env python3
"""Build a 2026 World Cup additions context sidecar from extra FootyStats files.

This uses regional tournaments, friendlies, and extra qualification files as a
pre-tournament research layer. It intentionally does not overwrite World Cup
qualifier aggregates or canonical merged training files.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_ADDITIONS = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/World Cup Additions")
DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/additions_context_2026")

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

COMPETITION_GROUPS = {
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
    "btts_percentage",
    "over25_percentage",
    "clean_sheet_percentage",
    "fts_percentage",
]

PLAYER_NUMERIC_COLS = [
    "minutes_played_overall",
    "appearances_overall",
    "goals_overall",
    "assists_overall",
    "average_rating_overall",
    "ratings_total_overall",
    "market_value",
    "xg_total_overall",
    "xa_total_overall",
    "games_started",
    "yellow_cards_overall",
    "red_cards_overall",
]

HOSTS = {"canada", "mexico", "usa"}


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


def compact(values: pd.Series, limit: int = 12) -> str:
    unique = sorted(v for v in values.dropna().astype(str).unique() if v)
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";+{len(unique) - limit}_more"
    return ";".join(unique)


def iter_files(root: Path) -> list[dict[str, object]]:
    files = []
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
        meta["competition_group"] = COMPETITION_GROUPS.get(meta["competition"], meta["competition"].upper().replace("-", "_"))
        meta["duplicate_content"] = int(duplicate_content)
        files.append(meta)
    return files


def scheduled_teams(path: Path) -> pd.DataFrame:
    launch = pd.read_csv(path, low_memory=False)
    rows = []
    for side in ["home", "away"]:
        rows.append(
            launch[[f"api_{side}_team_name", f"{side}_team_slug"]].rename(
                columns={f"api_{side}_team_name": "team_name", f"{side}_team_slug": "team_slug"}
            )
        )
    teams = pd.concat(rows, ignore_index=True)
    teams["team_slug"] = teams["team_slug"].map(canonical_team_slug)
    teams["is_host"] = teams["team_slug"].isin(HOSTS).astype(int)
    return teams.drop_duplicates("team_slug").sort_values("team_slug").reset_index(drop=True)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    wts = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = vals.notna() & wts.gt(0)
    if not mask.any():
        return float(vals.mean()) if vals.notna().any() else float("nan")
    return float((vals[mask] * wts[mask]).sum() / wts[mask].sum())


def build_team_rows(files: list[dict[str, object]]) -> pd.DataFrame:
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
                "addition_competition": meta["competition_group"],
                "addition_source_start": meta["start"],
                "addition_source_end": meta["end"],
            }
        )
        for col in TEAM_NUMERIC_COLS:
            if col in df.columns:
                local[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(local)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_player_rows(files: list[dict[str, object]]) -> pd.DataFrame:
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
                "player_name": df.get("full_name"),
                "position": df.get("position"),
                "addition_competition": meta["competition_group"],
                "addition_source_start": meta["start"],
                "addition_source_end": meta["end"],
            }
        )
        for col in PLAYER_NUMERIC_COLS:
            if col in df.columns:
                local[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(local)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_match_rows(files: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for meta in files:
        if meta["kind"] != "matches" or meta["duplicate_content"]:
            continue
        df = pd.read_csv(meta["path"], low_memory=False)
        if "home_team_name" not in df.columns or "away_team_name" not in df.columns:
            continue
        for side in ["home", "away"]:
            opp = "away" if side == "home" else "home"
            rows.append(
                pd.DataFrame(
                    {
                        "team_slug": df[f"{side}_team_name"].map(canonical_team_slug),
                        "addition_competition": meta["competition_group"],
                        "addition_source_end": meta["end"],
                        "match_rows": 1,
                        "goals_for": pd.to_numeric(df.get(f"{side}_team_goal_count"), errors="coerce"),
                        "goals_against": pd.to_numeric(df.get(f"{opp}_team_goal_count"), errors="coerce"),
                        "shots_for": pd.to_numeric(df.get(f"{side}_team_shots"), errors="coerce"),
                        "shots_against": pd.to_numeric(df.get(f"{opp}_team_shots"), errors="coerce"),
                        "shots_on_target_for": pd.to_numeric(df.get(f"{side}_team_shots_on_target"), errors="coerce"),
                        "shots_on_target_against": pd.to_numeric(df.get(f"{opp}_team_shots_on_target"), errors="coerce"),
                        "cards_for": pd.to_numeric(df.get(f"{side}_team_yellow_cards"), errors="coerce").fillna(0)
                        + pd.to_numeric(df.get(f"{side}_team_red_cards"), errors="coerce").fillna(0),
                        "cards_against": pd.to_numeric(df.get(f"{opp}_team_yellow_cards"), errors="coerce").fillna(0)
                        + pd.to_numeric(df.get(f"{opp}_team_red_cards"), errors="coerce").fillna(0),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def aggregate_team_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    if team_rows.empty:
        return pd.DataFrame(columns=["team_slug"])
    grouped = team_rows.groupby("team_slug", dropna=False)
    rows = []
    for team_slug, g in grouped:
        matches = pd.to_numeric(g.get("matches_played"), errors="coerce").fillna(0)
        rows.append(
            {
                "team_slug": team_slug,
                "additions_team_source_rows": len(g),
                "additions_team_source_competitions": compact(g["addition_competition"]),
                "additions_team_latest_source_end": int(pd.to_numeric(g["addition_source_end"], errors="coerce").max()),
                "additions_team_matches_played": float(matches.sum()),
                "additions_team_weighted_ppg": weighted_average(g.get("points_per_game"), matches),
                "additions_team_goals_for_per_match": weighted_average(g.get("goals_scored_per_match"), matches),
                "additions_team_goals_against_per_match": weighted_average(g.get("goals_conceded_per_match"), matches),
                "additions_team_goal_diff_per_match": weighted_average(g.get("goal_difference") / matches.replace(0, pd.NA), matches),
                "additions_team_xg_for_avg": weighted_average(g.get("xg_for_avg_overall"), matches),
                "additions_team_xg_against_avg": weighted_average(g.get("xg_against_avg_overall"), matches),
                "additions_team_btts_pct": weighted_average(g.get("btts_percentage"), matches),
                "additions_team_over25_pct": weighted_average(g.get("over25_percentage"), matches),
                "additions_team_clean_sheet_pct": weighted_average(g.get("clean_sheet_percentage"), matches),
                "additions_team_fts_pct": weighted_average(g.get("fts_percentage"), matches),
            }
        )
    return pd.DataFrame(rows)


def aggregate_player_features(player_rows: pd.DataFrame) -> pd.DataFrame:
    if player_rows.empty:
        return pd.DataFrame(columns=["team_slug"])
    rows = player_rows.copy()
    rows["_rating_raw"] = pd.to_numeric(rows.get("average_rating_overall"), errors="coerce")
    rows["_ratings_total"] = pd.to_numeric(rows.get("ratings_total_overall"), errors="coerce")
    rows["_appearances"] = pd.to_numeric(rows.get("appearances_overall"), errors="coerce")
    rows["_rating_per_app"] = rows["_ratings_total"] / rows["_appearances"].replace(0, pd.NA)
    rows["_rating_repaired_flag"] = (
        rows["_rating_raw"].notna()
        & ~rows["_rating_raw"].between(0, 10)
        & rows["_rating_per_app"].between(0, 10)
    ).astype(int)
    rows["_rating_capped_flag"] = (
        rows["_rating_raw"].notna()
        & ~rows["_rating_raw"].between(0, 10)
        & rows["_rating_per_app"].gt(10)
        & rows["_rating_per_app"].le(10.2)
    ).astype(int)
    rows["_invalid_rating_flag"] = (rows["_rating_raw"].notna() & ~rows["_rating_raw"].between(0, 10)).astype(int)
    rows["_valid_rating"] = rows["_rating_raw"].where(rows["_rating_raw"].between(0, 10)).astype(float)
    rows.loc[rows["_rating_repaired_flag"].eq(1), "_valid_rating"] = rows.loc[
        rows["_rating_repaired_flag"].eq(1), "_rating_per_app"
    ].astype(float)
    rows.loc[rows["_rating_capped_flag"].eq(1), "_valid_rating"] = 10.0
    rows["_unusable_rating_flag"] = (rows["_rating_raw"].notna() & rows["_valid_rating"].isna()).astype(int)
    rows["_market_value"] = pd.to_numeric(rows.get("market_value"), errors="coerce")
    player_level = (
        rows.groupby(["team_slug", "player_name"], dropna=False)
        .agg(
            latest_source_end=("addition_source_end", "max"),
            competitions=("addition_competition", compact),
            minutes=("minutes_played_overall", "sum"),
            appearances=("appearances_overall", "sum"),
            goals=("goals_overall", "sum"),
            assists=("assists_overall", "sum"),
            xg=("xg_total_overall", "sum"),
            xa=("xa_total_overall", "sum"),
            starts=("games_started", "sum"),
            rating=("_valid_rating", "max"),
            invalid_rating_rows=("_invalid_rating_flag", "sum"),
            repaired_rating_rows=("_rating_repaired_flag", "sum"),
            capped_rating_rows=("_rating_capped_flag", "sum"),
            unusable_rating_rows=("_unusable_rating_flag", "sum"),
            market_value=("_market_value", "max"),
        )
        .reset_index()
    )
    out_rows = []
    for team_slug, g in player_level.groupby("team_slug", dropna=False):
        ratings = pd.to_numeric(g["rating"], errors="coerce").dropna().sort_values(ascending=False)
        market_values = pd.to_numeric(g["market_value"], errors="coerce").fillna(0)
        out_rows.append(
            {
                "team_slug": team_slug,
                "additions_player_unique_players": int(g["player_name"].notna().sum()),
                "additions_player_rated_players": int(ratings.count()),
                "additions_player_invalid_rating_rows": int(pd.to_numeric(g["invalid_rating_rows"], errors="coerce").fillna(0).sum()),
                "additions_player_repaired_rating_rows": int(pd.to_numeric(g["repaired_rating_rows"], errors="coerce").fillna(0).sum()),
                "additions_player_capped_rating_rows": int(pd.to_numeric(g["capped_rating_rows"], errors="coerce").fillna(0).sum()),
                "additions_player_unusable_rating_rows": int(pd.to_numeric(g["unusable_rating_rows"], errors="coerce").fillna(0).sum()),
                "additions_player_source_competitions": compact(g["competitions"].str.split(";").explode()),
                "additions_player_latest_source_end": int(pd.to_numeric(g["latest_source_end"], errors="coerce").max()),
                "additions_player_top11_avg_rating": float(ratings.head(11).mean()) if not ratings.empty else pd.NA,
                "additions_player_top5_avg_rating": float(ratings.head(5).mean()) if not ratings.empty else pd.NA,
                "additions_player_squad_market_value_proxy": float(market_values.sum()),
                "additions_player_minutes": float(pd.to_numeric(g["minutes"], errors="coerce").fillna(0).sum()),
                "additions_player_goals": float(pd.to_numeric(g["goals"], errors="coerce").fillna(0).sum()),
                "additions_player_assists": float(pd.to_numeric(g["assists"], errors="coerce").fillna(0).sum()),
                "additions_player_xg": float(pd.to_numeric(g["xg"], errors="coerce").fillna(0).sum()),
                "additions_player_xa": float(pd.to_numeric(g["xa"], errors="coerce").fillna(0).sum()),
                "additions_player_starts": float(pd.to_numeric(g["starts"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(out_rows)


def build_rating_anomalies(player_rows: pd.DataFrame, scheduled: pd.DataFrame) -> pd.DataFrame:
    if player_rows.empty:
        return pd.DataFrame()
    teams = set(scheduled["team_slug"])
    out = player_rows.copy()
    out["_rating_raw"] = pd.to_numeric(out.get("average_rating_overall"), errors="coerce")
    out["_ratings_total"] = pd.to_numeric(out.get("ratings_total_overall"), errors="coerce")
    out["_appearances"] = pd.to_numeric(out.get("appearances_overall"), errors="coerce")
    out["rating_total_per_appearance"] = out["_ratings_total"] / out["_appearances"].replace(0, pd.NA)
    out["rating_repair_action"] = "UNUSABLE_OUT_OF_RANGE"
    out.loc[out["rating_total_per_appearance"].between(0, 10), "rating_repair_action"] = "USE_RATINGS_TOTAL_PER_APPEARANCE"
    out.loc[
        out["rating_total_per_appearance"].gt(10) & out["rating_total_per_appearance"].le(10.2),
        "rating_repair_action",
    ] = "CAP_RATINGS_TOTAL_PER_APPEARANCE_TO_10"
    out = out[
        out["team_slug"].isin(teams)
        & out["_rating_raw"].notna()
        & ~out["_rating_raw"].between(0, 10)
    ].copy()
    keep = [
        "team_slug",
        "player_name",
        "position",
        "addition_competition",
        "addition_source_end",
        "average_rating_overall",
        "ratings_total_overall",
        "appearances_overall",
        "games_started",
        "minutes_played_overall",
        "rating_total_per_appearance",
        "rating_repair_action",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(
        ["team_slug", "addition_competition", "average_rating_overall"],
        ascending=[True, True, False],
    )


def aggregate_match_features(match_rows: pd.DataFrame) -> pd.DataFrame:
    if match_rows.empty:
        return pd.DataFrame(columns=["team_slug"])
    grouped = match_rows.groupby("team_slug", dropna=False)
    out = grouped.agg(
        additions_match_rows=("match_rows", "sum"),
        additions_match_source_competitions=("addition_competition", compact),
        additions_match_latest_source_end=("addition_source_end", "max"),
        additions_match_goals_for_per_match=("goals_for", "mean"),
        additions_match_goals_against_per_match=("goals_against", "mean"),
        additions_match_shots_for_per_match=("shots_for", "mean"),
        additions_match_shots_against_per_match=("shots_against", "mean"),
        additions_match_sot_for_per_match=("shots_on_target_for", "mean"),
        additions_match_sot_against_per_match=("shots_on_target_against", "mean"),
        additions_match_cards_for_per_match=("cards_for", "mean"),
        additions_match_cards_against_per_match=("cards_against", "mean"),
    ).reset_index()
    return out


def build_sidecar(files: list[dict[str, object]], launch_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = scheduled_teams(launch_path)
    team_rows = build_team_rows(files)
    player_rows = build_player_rows(files)
    match_rows = build_match_rows(files)
    sidecar = teams.merge(aggregate_team_features(team_rows), on="team_slug", how="left")
    sidecar = sidecar.merge(aggregate_player_features(player_rows), on="team_slug", how="left")
    sidecar = sidecar.merge(aggregate_match_features(match_rows), on="team_slug", how="left")
    sidecar["additions_team_context_ready_flag"] = sidecar["additions_team_matches_played"].fillna(0).gt(0).astype(int)
    sidecar["additions_player_context_ready_flag"] = sidecar["additions_player_rated_players"].fillna(0).gt(0).astype(int)
    sidecar["additions_match_context_ready_flag"] = sidecar["additions_match_rows"].fillna(0).gt(0).astype(int)
    sidecar["additions_context_ready_flag"] = (
        sidecar["additions_team_context_ready_flag"].eq(1)
        & sidecar["additions_player_context_ready_flag"].eq(1)
        & sidecar["additions_match_context_ready_flag"].eq(1)
    ).astype(int)
    sidecar["additions_host_context_flag"] = (
        sidecar["is_host"].eq(1) & sidecar["additions_context_ready_flag"].eq(1)
    ).astype(int)
    sidecar["additions_context_source_status"] = sidecar["additions_context_ready_flag"].map(
        {1: "FOOTYSTATS_ADDITIONS_TEAM_PLAYER_MATCH_CONTEXT", 0: "NO_FOOTYSTATS_ADDITIONS_CONTEXT"}
    )
    return sidecar, team_rows, player_rows, match_rows


def write_summary(outdir: Path, files: pd.DataFrame, sidecar: pd.DataFrame) -> None:
    comp = files.groupby(["competition_group", "kind"], dropna=False).agg(
        files=("file_name", "count"),
        non_duplicate_files=("duplicate_content", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).sum())),
        min_start=("start", "min"),
        max_end=("end", "max"),
    ).reset_index()
    comp.to_csv(outdir / "world_cup_additions_context_competition_inventory.csv", index=False)
    coverage = pd.DataFrame(
        [
            {"coverage": "team_context", "teams": int(sidecar["additions_team_context_ready_flag"].sum())},
            {"coverage": "player_context", "teams": int(sidecar["additions_player_context_ready_flag"].sum())},
            {"coverage": "match_context", "teams": int(sidecar["additions_match_context_ready_flag"].sum())},
            {"coverage": "all_context", "teams": int(sidecar["additions_context_ready_flag"].sum())},
            {"coverage": "host_context", "teams": int(sidecar["additions_host_context_flag"].sum())},
        ]
    )
    coverage.to_csv(outdir / "world_cup_additions_context_coverage_counts.csv", index=False)
    top = sidecar.sort_values("additions_player_top11_avg_rating", ascending=False).head(12)
    top_lines = [
        f"- {r.team_name}: top11={r.additions_player_top11_avg_rating:.2f}, "
        f"rated={int(r.additions_player_rated_players)}, comps={r.additions_player_source_competitions}"
        for r in top.itertuples()
        if pd.notna(getattr(r, "additions_player_top11_avg_rating", pd.NA))
    ]
    lines = [
        "# World Cup Additions Context Sidecar",
        "",
        "Research-only pre-tournament sidecar from FootyStats regional tournaments, friendlies, and extra qualification files.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_additions_context_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_additions_team_source_rows.csv'}`",
        f"- `{outdir / 'world_cup_additions_player_source_rows.csv'}`",
        f"- `{outdir / 'world_cup_additions_match_source_rows.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Teams with team context: {int(sidecar['additions_team_context_ready_flag'].sum())} / {len(sidecar)}",
        f"- Teams with player context: {int(sidecar['additions_player_context_ready_flag'].sum())} / {len(sidecar)}",
        f"- Teams with match context: {int(sidecar['additions_match_context_ready_flag'].sum())} / {len(sidecar)}",
        f"- Teams with all additions context: {int(sidecar['additions_context_ready_flag'].sum())} / {len(sidecar)}",
        f"- Host teams with additions context: {int(sidecar['additions_host_context_flag'].sum())} / 3",
        f"- Out-of-range player rating rows detected: {int(sidecar['additions_player_invalid_rating_rows'].fillna(0).sum())}",
        f"- Out-of-range rows repaired using `ratings_total_overall / appearances_overall`: {int(sidecar['additions_player_repaired_rating_rows'].fillna(0).sum())}",
        f"- Out-of-range rows capped after repair because value was only slightly above 10: {int(sidecar['additions_player_capped_rating_rows'].fillna(0).sum())}",
        f"- Out-of-range rows still unusable for rating aggregates: {int(sidecar['additions_player_unusable_rating_rows'].fillna(0).sum())}",
        "",
        "## Top Additions Player Rating Proxies",
        "",
        *top_lines,
        "",
        "## Notes",
        "",
        "- These fields are pre-tournament context proxies and should remain distinct from verified World Cup 2026 squad/API layers.",
        "- Friendlies and regional tournaments carry different competitive intensity, so source-competition columns must stay attached.",
        "- This sidecar does not alter `Matches/__merged__/World_Cup__merged.csv` or production deploy routing.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additions-root", type=Path, default=DEFAULT_ADDITIONS)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    files = iter_files(args.additions_root)
    sidecar, team_rows, player_rows, match_rows = build_sidecar(files, args.launch_scaffold)
    file_inventory = pd.DataFrame(files)
    rating_anomalies = build_rating_anomalies(player_rows, sidecar)
    file_inventory.to_csv(args.outdir / "world_cup_additions_context_file_inventory.csv", index=False)
    sidecar.to_csv(args.outdir / "world_cup_additions_context_sidecar.csv", index=False)
    team_rows.to_csv(args.outdir / "world_cup_additions_team_source_rows.csv", index=False)
    player_rows.to_csv(args.outdir / "world_cup_additions_player_source_rows.csv", index=False)
    match_rows.to_csv(args.outdir / "world_cup_additions_match_source_rows.csv", index=False)
    rating_anomalies.to_csv(args.outdir / "world_cup_additions_player_rating_anomalies.csv", index=False)
    write_summary(args.outdir, file_inventory, sidecar)
    print(f"[ok] sidecar={len(sidecar)} team_rows={len(team_rows)} player_rows={len(player_rows)} match_rows={len(match_rows)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
