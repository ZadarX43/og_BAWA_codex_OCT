#!/usr/bin/env python3
"""Build FootyStats World Cup qualification feature sidecars."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/qualification_foundation")

QUAL_RE = re.compile(
    r"^international-wc-qualification-(?P<region>.+?)-(?P<kind>league|matches|teams|players)-"
    r"(?P<start>\d{4})-to-(?P<end>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$"
)

REGION_TO_CONFED = {
    "africa": "CAF",
    "asia": "AFC",
    "concacaf": "CONCACAF",
    "europe": "UEFA",
    "south-america": "CONMEBOL",
    "intercontinental-playoffs": "INTERCONTINENTAL",
}

TEAM_SLUG_ALIASES = {
    "cape_verde": "cape_verde",
    "cape_verde_islands": "cape_verde",
    "congo_dr": "dr_congo",
    "dr_congo": "dr_congo",
    "curacao": "curacao",
    "cura_ao": "curacao",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
}


TEAM_FEATURE_COLS = [
    "matches_played",
    "wins",
    "draws",
    "losses",
    "points_per_game",
    "league_position",
    "performance_rank",
    "goals_scored",
    "goals_conceded",
    "goal_difference",
    "goals_scored_per_match",
    "goals_conceded_per_match",
    "average_total_goals_per_match",
    "xg_for_avg_overall",
    "xg_against_avg_overall",
    "win_percentage",
    "draw_percentage_overall",
    "loss_percentage_ovearll",
    "btts_percentage",
    "over25_percentage",
    "clean_sheet_percentage",
    "fts_percentage",
    "corners_per_match",
    "cards_per_match",
    "prediction_risk",
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
    "yellow_cards_overall",
    "red_cards_overall",
]


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"national team", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team_slug(value: object) -> str:
    slug = slugify(value)
    return TEAM_SLUG_ALIASES.get(slug, slug)


def read_qualification_files(drop: Path, kind: str) -> list[tuple[dict[str, object], pd.DataFrame]]:
    out = []
    seen_hashes = set()
    for path in sorted(drop.glob("international-wc-qualification-*")):
        match = QUAL_RE.match(path.name)
        if not match or match.group("kind") != kind:
            continue
        # Prefer original file over duplicate suffix by skipping repeated content.
        content_key = path.read_bytes().__hash__()
        if content_key in seen_hashes:
            continue
        seen_hashes.add(content_key)
        meta = match.groupdict()
        df = pd.read_csv(path, low_memory=False)
        meta["path"] = str(path)
        meta["start"] = int(meta["start"])
        meta["end"] = int(meta["end"])
        meta["confederation"] = REGION_TO_CONFED.get(meta["region"], "UNKNOWN")
        out.append((meta, df))
    return out


def build_team_features(drop: Path) -> pd.DataFrame:
    frames = []
    for meta, df in read_qualification_files(drop, "teams"):
        local = df.copy()
        name = local.get("common_name", local.get("team_name"))
        local["team_slug"] = name.map(canonical_team_slug)
        local["team_name_clean"] = name
        local["qualification_region"] = meta["region"]
        local["confederation"] = meta["confederation"]
        local["qualification_cycle_start"] = meta["start"]
        local["qualification_cycle_end"] = meta["end"]
        keep = [
            "team_slug",
            "team_name_clean",
            "qualification_region",
            "confederation",
            "qualification_cycle_start",
            "qualification_cycle_end",
        ] + [c for c in TEAM_FEATURE_COLS if c in local.columns]
        frames.append(local[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in TEAM_FEATURE_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=["team_slug", "qualification_region", "qualification_cycle_start", "qualification_cycle_end"]).reset_index(drop=True)


def build_player_features(drop: Path) -> pd.DataFrame:
    rows = []
    for meta, df in read_qualification_files(drop, "players"):
        local = df.copy()
        local["team_slug"] = local.get("Current Club", local.get("nationality", "")).map(canonical_team_slug)
        local["qualification_region"] = meta["region"]
        local["confederation"] = meta["confederation"]
        local["qualification_cycle_start"] = meta["start"]
        local["qualification_cycle_end"] = meta["end"]
        for col in PLAYER_NUMERIC_COLS:
            if col in local.columns:
                local[col] = pd.to_numeric(local[col], errors="coerce")
        local["_rating"] = local.get("average_rating_overall", pd.Series(dtype=float))
        local["_market_value"] = local.get("market_value", pd.Series(dtype=float))
        grouped = local.groupby(["team_slug", "qualification_region", "confederation", "qualification_cycle_start", "qualification_cycle_end"], dropna=False)
        agg = grouped.agg(
            qualifier_player_rows=("team_slug", "size"),
            qualifier_named_players=("full_name", lambda s: int(s.notna().sum()) if "full_name" in local.columns else 0),
            qualifier_rated_players=("_rating", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            qualifier_avg_player_rating=("_rating", "mean"),
            qualifier_top11_avg_rating=("_rating", lambda s: pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False).head(11).mean()),
            qualifier_top5_avg_rating=("_rating", lambda s: pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False).head(5).mean()),
            qualifier_squad_market_value=("_market_value", "sum"),
            qualifier_player_minutes=("minutes_played_overall", "sum"),
            qualifier_player_goals=("goals_overall", "sum"),
            qualifier_player_assists=("assists_overall", "sum"),
            qualifier_player_xg=("xg_total_overall", "sum"),
            qualifier_player_xa=("xa_total_overall", "sum"),
            qualifier_player_starts=("games_started", "sum"),
        ).reset_index()
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def build_match_spine(drop: Path) -> pd.DataFrame:
    frames = []
    keep_cols = [
        "timestamp",
        "date_GMT",
        "status",
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
        "total_goal_count",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "team_a_xg",
        "team_b_xg",
        "average_goals_per_match_pre_match",
        "btts_percentage_pre_match",
        "over_25_percentage_pre_match",
        "odds_ft_home_team_win",
        "odds_ft_draw",
        "odds_ft_away_team_win",
        "odds_ft_over25",
        "odds_btts_yes",
        "odds_btts_no",
        "stadium_name",
    ]
    for meta, df in read_qualification_files(drop, "matches"):
        local = df.copy()
        local["qualification_region"] = meta["region"]
        local["confederation"] = meta["confederation"]
        local["qualification_cycle_start"] = meta["start"]
        local["qualification_cycle_end"] = meta["end"]
        local["home_team_slug"] = local["home_team_name"].map(canonical_team_slug)
        local["away_team_slug"] = local["away_team_name"].map(canonical_team_slug)
        keep = [
            "qualification_region",
            "confederation",
            "qualification_cycle_start",
            "qualification_cycle_end",
            "home_team_slug",
            "away_team_slug",
        ] + [c for c in keep_cols if c in local.columns]
        frames.append(local[keep])
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def latest_2026_features(teams: pd.DataFrame, players: pd.DataFrame, launch: pd.DataFrame) -> pd.DataFrame:
    scheduled = sorted(
        set(launch["home_team_slug"].map(canonical_team_slug)).union(
            set(launch["away_team_slug"].map(canonical_team_slug))
        )
    )
    latest = teams[teams["qualification_cycle_end"].eq(2026)].copy()
    latest = latest.sort_values(["team_slug", "qualification_cycle_start"], ascending=[True, False]).drop_duplicates("team_slug")
    if not players.empty:
        p26 = players[players["qualification_cycle_end"].eq(2026)].copy()
        p26 = p26.sort_values(["team_slug", "qualification_cycle_start"], ascending=[True, False]).drop_duplicates("team_slug")
        latest = latest.merge(
            p26.drop(columns=["qualification_region", "confederation", "qualification_cycle_start", "qualification_cycle_end"], errors="ignore"),
            on="team_slug",
            how="left",
        )
    out = pd.DataFrame({"team_slug": scheduled}).merge(latest, on="team_slug", how="left")
    out["footystats_qualifier_2026_joined_flag"] = out["qualification_cycle_end"].notna().astype(int)
    return out


def build_override(latest: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["team_slug"] = latest["team_slug"]
    out["team_name"] = latest.get("team_name_clean")
    out["confederation"] = latest.get("confederation")
    out["qualification_route_verified_flag"] = 0
    out["qualifier_matches_played"] = latest.get("matches_played")
    out["qualifier_ppg"] = latest.get("points_per_game")
    out["qualifier_goal_diff_per_match"] = latest.get("goal_difference") / latest.get("matches_played").replace(0, pd.NA)
    out["qualifier_goals_for_per_match"] = latest.get("goals_scored_per_match")
    out["qualifier_goals_against_per_match"] = latest.get("goals_conceded_per_match")
    out["qualification_position"] = latest.get("league_position")
    out["qualification_goal_diff"] = latest.get("goal_difference")
    out["qualification_context_source_status"] = latest["footystats_qualifier_2026_joined_flag"].map(
        {1: "FOOTYSTATS_QUALIFIER_TEAM_STATS_2026", 0: "NO_FOOTYSTATS_QUALIFIER_TEAM_STATS_2026"}
    )
    out["qualifier_context_notes"] = "FootyStats qualifier aggregate joined; exact qualification route still requires verified override."
    return out


def write_summary(outdir: Path, teams: pd.DataFrame, players: pd.DataFrame, matches: pd.DataFrame, latest: pd.DataFrame) -> None:
    coverage = latest["footystats_qualifier_2026_joined_flag"].value_counts().rename_axis("joined").reset_index(name="teams")
    confed = latest.groupby("confederation", dropna=False).agg(teams=("team_slug", "count"), joined=("footystats_qualifier_2026_joined_flag", "sum")).reset_index()
    coverage.to_csv(outdir / "world_cup_2026_qualifier_join_coverage.csv", index=False)
    confed.to_csv(outdir / "world_cup_2026_qualifier_join_by_confederation.csv", index=False)
    top = latest.sort_values("points_per_game", ascending=False).head(12)
    top_lines = [
        f"- {r.team_name_clean}: PPG={r.points_per_game}, GD={r.goal_difference}, matches={r.matches_played}, confed={r.confederation}"
        for r in top.itertuples()
        if pd.notna(getattr(r, "points_per_game", pd.NA))
    ]
    lines = [
        "# World Cup FootyStats Qualification Foundation",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'footystats_wc_qualification_team_features.csv'}`",
        f"- `{outdir / 'footystats_wc_qualification_player_team_features.csv'}`",
        f"- `{outdir / 'footystats_wc_qualification_match_spine.csv'}`",
        f"- `{outdir / 'world_cup_2026_qualifier_latest_team_features.csv'}`",
        f"- `{outdir / 'world_cup_2026_qualification_override_from_footystats.csv'}`",
        "",
        "## Source Rows",
        "",
        f"- Team feature rows: {len(teams)}",
        f"- Player-team feature rows: {len(players)}",
        f"- Match spine rows: {len(matches)}",
        "",
        "## 2026 Scheduled-Team Join",
        "",
        f"- Joined teams: {int(latest['footystats_qualifier_2026_joined_flag'].sum())} / {len(latest)}",
        "",
        "## Top 2026 Qualifier PPG Rows",
        "",
        *top_lines,
        "",
        "## Notes",
        "",
        "- These are aggregate qualification stats and must be treated as pre-tournament sidecar features.",
        "- Qualification route remains unverified unless supplied separately.",
        "- Teams with no FootyStats qualifier row are usually hosts or naming/coverage gaps.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-root", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    launch["home_team_slug"] = launch["home_team_slug"].map(canonical_team_slug)
    launch["away_team_slug"] = launch["away_team_slug"].map(canonical_team_slug)
    teams = build_team_features(args.drop_root)
    players = build_player_features(args.drop_root)
    matches = build_match_spine(args.drop_root)
    latest = latest_2026_features(teams, players, launch)
    override = build_override(latest)
    teams.to_csv(args.outdir / "footystats_wc_qualification_team_features.csv", index=False)
    players.to_csv(args.outdir / "footystats_wc_qualification_player_team_features.csv", index=False)
    matches.to_csv(args.outdir / "footystats_wc_qualification_match_spine.csv", index=False)
    latest.to_csv(args.outdir / "world_cup_2026_qualifier_latest_team_features.csv", index=False)
    override.to_csv(args.outdir / "world_cup_2026_qualification_override_from_footystats.csv", index=False)
    write_summary(args.outdir, teams, players, matches, latest)
    print(f"[ok] teams={len(teams)} players={len(players)} matches={len(matches)} latest={len(latest)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
