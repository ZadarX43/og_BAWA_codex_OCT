#!/usr/bin/env python3
"""Build recent national-team history, local H2H, and API H2H manifest sidecars."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_ADDITIONS = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/World Cup Additions")
DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/recent_history_h2h_2026")

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


def iter_match_files(root: Path) -> list[dict[str, object]]:
    out = []
    seen_hashes = set()
    for path in sorted(root.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if not match or match.group("kind") != "matches":
            continue
        h = hash(path.read_bytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        meta = match.groupdict()
        meta["path"] = str(path)
        meta["file_name"] = path.name
        meta["start"] = int(meta["start"])
        meta["end"] = int(meta["end"])
        meta["competition_group"] = COMPETITION_GROUPS.get(meta["competition"], meta["competition"].upper().replace("-", "_"))
        out.append(meta)
    return out


def load_launch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["home_team_slug"] = df["home_team_slug"].map(canonical_team_slug)
    df["away_team_slug"] = df["away_team_slug"].map(canonical_team_slug)
    return df


def build_match_spine(root: Path) -> pd.DataFrame:
    rows = []
    for meta in iter_match_files(root):
        df = pd.read_csv(meta["path"], low_memory=False)
        if "home_team_name" not in df.columns or "away_team_name" not in df.columns:
            continue
        local = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(df.get("timestamp"), errors="coerce"),
                "date_GMT": df.get("date_GMT"),
                "addition_competition": meta["competition_group"],
                "addition_source_end": meta["end"],
                "home_team_name": df["home_team_name"],
                "away_team_name": df["away_team_name"],
                "home_team_slug": df["home_team_name"].map(canonical_team_slug),
                "away_team_slug": df["away_team_name"].map(canonical_team_slug),
                "home_goals": pd.to_numeric(df.get("home_team_goal_count"), errors="coerce"),
                "away_goals": pd.to_numeric(df.get("away_team_goal_count"), errors="coerce"),
                "total_goals": pd.to_numeric(df.get("total_goal_count"), errors="coerce"),
                "home_shots": pd.to_numeric(df.get("home_team_shots"), errors="coerce"),
                "away_shots": pd.to_numeric(df.get("away_team_shots"), errors="coerce"),
                "home_sot": pd.to_numeric(df.get("home_team_shots_on_target"), errors="coerce"),
                "away_sot": pd.to_numeric(df.get("away_team_shots_on_target"), errors="coerce"),
            }
        )
        local["btts"] = (local["home_goals"].gt(0) & local["away_goals"].gt(0)).astype(int)
        local["over25"] = local["total_goals"].gt(2).astype(int)
        rows.append(local)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def team_side_rows(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        local = pd.DataFrame(
            {
                "team_slug": matches[f"{side}_team_slug"],
                "opponent_slug": matches[f"{opp}_team_slug"],
                "addition_competition": matches["addition_competition"],
                "addition_source_end": matches["addition_source_end"],
                "timestamp": matches["timestamp"],
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
        rows.append(local)
    return pd.concat(rows, ignore_index=True, sort=False)


def aggregate_team_history(side_rows: pd.DataFrame, scheduled: set[str], min_year: int = 2023) -> pd.DataFrame:
    scoped = side_rows[side_rows["team_slug"].isin(scheduled)].copy()
    scoped_recent = scoped[pd.to_numeric(scoped["addition_source_end"], errors="coerce").ge(min_year)]
    rows = []
    for label, frame in [("all_additions", scoped), (f"since_{min_year}", scoped_recent)]:
        grouped = frame.groupby("team_slug", dropna=False)
        local = grouped.agg(
            matches=("team_slug", "size"),
            win_rate=("win", "mean"),
            draw_rate=("draw", "mean"),
            loss_rate=("loss", "mean"),
            goals_for_per_match=("goals_for", "mean"),
            goals_against_per_match=("goals_against", "mean"),
            btts_rate=("btts", "mean"),
            over25_rate=("over25", "mean"),
            shots_for_per_match=("shots_for", "mean"),
            shots_against_per_match=("shots_against", "mean"),
            sot_for_per_match=("sot_for", "mean"),
            sot_against_per_match=("sot_against", "mean"),
        ).reset_index()
        local["goal_diff_per_match"] = local["goals_for_per_match"] - local["goals_against_per_match"]
        rename = {c: f"recent_history_{label}_{c}" for c in local.columns if c != "team_slug"}
        rows.append(local.rename(columns=rename))
    out = rows[0].merge(rows[1], on="team_slug", how="outer") if len(rows) == 2 else rows[0]
    return out


def build_h2h(launch: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fx in launch.itertuples(index=False):
        home = fx.home_team_slug
        away = fx.away_team_slug
        pair = matches[
            ((matches["home_team_slug"] == home) & (matches["away_team_slug"] == away))
            | ((matches["home_team_slug"] == away) & (matches["away_team_slug"] == home))
        ].copy()
        if pair.empty:
            rows.append(
                {
                    "api_fixture_id": fx.api_fixture_id,
                    "local_h2h_match_count": 0,
                    "local_h2h_home_side_win_rate": pd.NA,
                    "local_h2h_draw_rate": pd.NA,
                    "local_h2h_away_side_win_rate": pd.NA,
                    "local_h2h_total_goals_per_match": pd.NA,
                    "local_h2h_btts_rate": pd.NA,
                    "local_h2h_over25_rate": pd.NA,
                    "local_h2h_source_status": "NO_LOCAL_FOOTYSTATS_H2H",
                }
            )
            continue
        home_goals = pair["home_goals"].where(pair["home_team_slug"].eq(home), pair["away_goals"])
        away_goals = pair["away_goals"].where(pair["home_team_slug"].eq(home), pair["home_goals"])
        rows.append(
            {
                "api_fixture_id": fx.api_fixture_id,
                "local_h2h_match_count": len(pair),
                "local_h2h_home_side_win_rate": float(home_goals.gt(away_goals).mean()),
                "local_h2h_draw_rate": float(home_goals.eq(away_goals).mean()),
                "local_h2h_away_side_win_rate": float(home_goals.lt(away_goals).mean()),
                "local_h2h_total_goals_per_match": float((home_goals + away_goals).mean()),
                "local_h2h_btts_rate": float((home_goals.gt(0) & away_goals.gt(0)).mean()),
                "local_h2h_over25_rate": float((home_goals + away_goals).gt(2).mean()),
                "local_h2h_latest_source_end": int(pd.to_numeric(pair["addition_source_end"], errors="coerce").max()),
                "local_h2h_source_status": "LOCAL_FOOTYSTATS_ADDITIONS_H2H",
            }
        )
    return pd.DataFrame(rows)


def build_api_h2h_manifest(launch: pd.DataFrame) -> pd.DataFrame:
    out = launch[
        [
            "api_fixture_id",
            "api_home_team_id",
            "api_home_team_name",
            "api_away_team_id",
            "api_away_team_name",
            "api_date",
            "api_round",
        ]
    ].copy()
    out["api_h2h_param"] = out["api_home_team_id"].astype(int).astype(str) + "-" + out["api_away_team_id"].astype(int).astype(str)
    out["api_h2h_endpoint"] = "/fixtures/headtohead"
    out["api_h2h_status"] = "READY_TO_FETCH"
    return out


def side_join(fixtures: pd.DataFrame, teams: pd.DataFrame, side: str) -> pd.DataFrame:
    side_df = teams.rename(columns={c: f"{side}_{c}" for c in teams.columns if c != "team_slug"})
    side_df = side_df.rename(columns={"team_slug": f"{side}_team_slug"})
    return fixtures.merge(side_df, on=f"{side}_team_slug", how="left")


def build_fixture_recent(launch: pd.DataFrame, team_history: pd.DataFrame) -> pd.DataFrame:
    base = launch[["api_fixture_id", "home_team_slug", "away_team_slug"]].copy()
    out = side_join(base, team_history, "home")
    out = side_join(out, team_history, "away")
    for col in [
        "recent_history_since_2023_win_rate",
        "recent_history_since_2023_goals_for_per_match",
        "recent_history_since_2023_goals_against_per_match",
        "recent_history_since_2023_goal_diff_per_match",
        "recent_history_since_2023_btts_rate",
        "recent_history_since_2023_over25_rate",
        "recent_history_since_2023_sot_for_per_match",
        "recent_history_since_2023_sot_against_per_match",
    ]:
        out[f"{col}_delta"] = pd.to_numeric(out.get(f"home_{col}"), errors="coerce") - pd.to_numeric(
            out.get(f"away_{col}"), errors="coerce"
        )
    return out


def write_summary(outdir: Path, team_history: pd.DataFrame, fixture_recent: pd.DataFrame, h2h: pd.DataFrame, manifest: pd.DataFrame) -> None:
    lines = [
        "# World Cup Recent History and H2H Sidecars",
        "",
        "Research-only sidecars from local FootyStats additions, plus API-Football H2H fetch manifest.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_recent_team_history_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_2026_fixture_recent_history_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_2026_local_h2h_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_api_h2h_manifest.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Teams with recent history rows: {len(team_history)}",
        f"- Fixtures with recent-history sidecar rows: {len(fixture_recent)}",
        f"- Fixtures with local FootyStats H2H matches: {int(h2h['local_h2h_match_count'].fillna(0).gt(0).sum())} / {len(h2h)}",
        f"- API H2H manifest rows ready to fetch: {len(manifest)}",
        "",
        "## Notes",
        "",
        "- Local H2H is sparse by nature; API H2H should be fetched later for a fuller history.",
        "- Recent-history deltas are stronger than direct H2H for most group fixtures.",
        "- No network calls are made by this builder.",
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
    launch = load_launch(args.launch_scaffold)
    scheduled = set(launch["home_team_slug"]).union(set(launch["away_team_slug"]))
    matches = build_match_spine(args.additions_root)
    side_rows = team_side_rows(matches)
    team_history = aggregate_team_history(side_rows, scheduled)
    fixture_recent = build_fixture_recent(launch, team_history)
    h2h = build_h2h(launch, matches)
    manifest = build_api_h2h_manifest(launch)
    matches.to_csv(args.outdir / "world_cup_additions_match_spine_for_h2h.csv", index=False)
    team_history.to_csv(args.outdir / "world_cup_recent_team_history_sidecar.csv", index=False)
    fixture_recent.to_csv(args.outdir / "world_cup_2026_fixture_recent_history_sidecar.csv", index=False)
    h2h.to_csv(args.outdir / "world_cup_2026_local_h2h_sidecar.csv", index=False)
    manifest.to_csv(args.outdir / "world_cup_api_h2h_manifest.csv", index=False)
    write_summary(args.outdir, team_history, fixture_recent, h2h, manifest)
    print(f"[ok] teams={len(team_history)} fixtures={len(fixture_recent)} h2h={len(h2h)} manifest={len(manifest)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
