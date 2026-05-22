#!/usr/bin/env python3
"""Build research-only World Cup historical priors from the Fjelstul archive.

The Fjelstul database covers World Cups through 2018, so these features are
macro historical priors rather than current-squad signals. They are useful for
World Cup model scaffolding and QA, but should not be treated as live 2026
player intelligence.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_ARCHIVE = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/archive")
DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/fjelstul_historical_priors")

TEAM_SLUG_ALIASES = {
    "cape_verde": "cape_verde",
    "cape_verde_islands": "cape_verde",
    "congo_dr": "dr_congo",
    "dr_congo": "dr_congo",
    "curacao": "curacao",
    "cura_ao": "curacao",
    "czechia": "czech_republic",
    "czech_republic": "czech_republic",
    "ivory_coast": "ivory_coast",
    "cote_d_ivoire": "ivory_coast",
    "korea_republic": "south_korea",
    "south_korea": "south_korea",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
    "united_states": "usa",
    "usa": "usa",
}


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


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Missing Fjelstul archive file: {path}")
    return pd.read_csv(path, low_memory=False)


def add_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tournament_id" in out.columns:
        out["world_cup_year"] = pd.to_numeric(out["tournament_id"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    return out


def team_window_features(team_apps: pd.DataFrame, standings: pd.DataFrame, min_year: int | None, prefix: str) -> pd.DataFrame:
    apps = team_apps.copy()
    st = standings.copy()
    if min_year is not None:
        apps = apps[apps["world_cup_year"].ge(min_year)]
        st = st[st["world_cup_year"].ge(min_year)]
    if apps.empty:
        return pd.DataFrame(columns=["team_slug"])

    grouped = apps.groupby(["team_slug", "team_name"], dropna=False)
    features = grouped.agg(
        **{
            f"{prefix}_wc_matches": ("match_id", "nunique"),
            f"{prefix}_wc_tournaments": ("tournament_id", "nunique"),
            f"{prefix}_wc_win_rate": ("win", "mean"),
            f"{prefix}_wc_draw_rate": ("draw", "mean"),
            f"{prefix}_wc_loss_rate": ("lose", "mean"),
            f"{prefix}_wc_goals_for_per_match": ("goals_for", "mean"),
            f"{prefix}_wc_goals_against_per_match": ("goals_against", "mean"),
            f"{prefix}_wc_goal_diff_per_match": ("goal_differential", "mean"),
            f"{prefix}_wc_group_stage_match_rate": ("group_stage", "mean"),
            f"{prefix}_wc_knockout_match_rate": ("knockout_stage", "mean"),
        }
    ).reset_index()

    if not st.empty:
        st_grouped = st.groupby(["team_slug"], dropna=False).agg(
            **{
                f"{prefix}_wc_tournament_rows": ("tournament_id", "nunique"),
                f"{prefix}_wc_avg_finish_position": ("position", "mean"),
                f"{prefix}_wc_best_finish_position": ("position", "min"),
            }
        ).reset_index()
        features = features.merge(st_grouped, on="team_slug", how="left")
    return features


def build_team_priors(root: Path, launch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    apps = add_year(read_csv(root, "team_appearances"))
    standings = add_year(read_csv(root, "tournament_standings"))
    apps["team_slug"] = apps["team_name"].map(canonical_team_slug)
    standings["team_slug"] = standings["team_name"].map(canonical_team_slug)

    all_features = team_window_features(apps, standings, None, "all")
    modern_features = team_window_features(apps, standings, 1998, "modern")
    recent_features = team_window_features(apps, standings, 2010, "recent")
    out = all_features.merge(modern_features.drop(columns=["team_name"], errors="ignore"), on="team_slug", how="outer")
    out = out.merge(recent_features.drop(columns=["team_name"], errors="ignore"), on="team_slug", how="outer")
    if "team_name" not in out.columns:
        out["team_name"] = pd.NA
    for col in ["team_name_x", "team_name_y"]:
        if col in out.columns:
            out["team_name"] = out["team_name"].combine_first(out[col])
    out = out.drop(columns=[c for c in ["team_name_x", "team_name_y"] if c in out.columns], errors="ignore")

    scheduled = pd.DataFrame(
        {
            "team_slug": sorted(
                set(launch["home_team_slug"].map(canonical_team_slug)).union(
                    set(launch["away_team_slug"].map(canonical_team_slug))
                )
            )
        }
    )
    sidecar = scheduled.merge(out, on="team_slug", how="left")
    sidecar["fjelstul_historical_prior_ready_flag"] = sidecar["all_wc_matches"].notna().astype(int)
    sidecar["fjelstul_recent_prior_ready_flag"] = sidecar["recent_wc_matches"].notna().astype(int)
    return out.sort_values("team_slug").reset_index(drop=True), sidecar.sort_values("team_slug").reset_index(drop=True)


def build_stage_event_priors(root: Path) -> pd.DataFrame:
    matches = add_year(read_csv(root, "matches"))
    goals = read_csv(root, "goals")
    bookings = read_csv(root, "bookings")
    goal_counts = goals.groupby("match_id").agg(
        wc_goals=("goal_id", "count"),
        wc_penalty_goals=("penalty", "sum"),
        wc_own_goals=("own_goal", "sum"),
    ).reset_index()
    booking_counts = bookings.groupby("match_id").agg(
        wc_bookings=("booking_id", "count"),
        wc_yellow_cards=("yellow_card", "sum"),
        wc_red_cards=("red_card", "sum"),
        wc_second_yellows=("second_yellow_card", "sum"),
        wc_sending_offs=("sending_off", "sum"),
    ).reset_index()
    base = matches.merge(goal_counts, on="match_id", how="left").merge(booking_counts, on="match_id", how="left")
    count_cols = [c for c in base.columns if c.startswith("wc_")]
    base[count_cols] = base[count_cols].fillna(0)
    base["modern_window_flag"] = base["world_cup_year"].ge(1998).astype(int)
    rows = []
    for label, frame in [("all", base), ("modern_1998_2018", base[base["modern_window_flag"].eq(1)])]:
        grouped = frame.groupby(["stage_name", "group_stage", "knockout_stage"], dropna=False)
        local = grouped.agg(
            window=("match_id", lambda s: label),
            matches=("match_id", "nunique"),
            goals_per_match=("wc_goals", "mean"),
            penalty_goals_per_match=("wc_penalty_goals", "mean"),
            own_goals_per_match=("wc_own_goals", "mean"),
            bookings_per_match=("wc_bookings", "mean"),
            yellow_cards_per_match=("wc_yellow_cards", "mean"),
            red_cards_per_match=("wc_red_cards", "mean"),
            sending_offs_per_match=("wc_sending_offs", "mean"),
            draw_rate=("draw", "mean"),
            home_win_rate=("home_team_win", "mean"),
            away_win_rate=("away_team_win", "mean"),
        ).reset_index()
        rows.append(local)
    return pd.concat(rows, ignore_index=True, sort=False)


def build_referee_priors(root: Path) -> pd.DataFrame:
    refs = read_csv(root, "referee_appearances")
    goals = read_csv(root, "goals")
    bookings = read_csv(root, "bookings")
    matches = read_csv(root, "matches")
    goal_counts = goals.groupby("match_id").agg(goals=("goal_id", "count"), penalty_goals=("penalty", "sum")).reset_index()
    booking_counts = bookings.groupby("match_id").agg(
        bookings=("booking_id", "count"),
        yellow_cards=("yellow_card", "sum"),
        red_cards=("red_card", "sum"),
        second_yellows=("second_yellow_card", "sum"),
        sending_offs=("sending_off", "sum"),
    ).reset_index()
    base = refs.merge(matches[["match_id", "draw", "home_team_win", "away_team_win"]], on="match_id", how="left")
    base = base.merge(goal_counts, on="match_id", how="left").merge(booking_counts, on="match_id", how="left")
    for col in ["goals", "penalty_goals", "bookings", "yellow_cards", "red_cards", "second_yellows", "sending_offs"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)
    out = base.groupby(
        ["referee_id", "family_name", "given_name", "country_name", "confederation_code"], dropna=False
    ).agg(
        wc_referee_matches=("match_id", "nunique"),
        wc_referee_goals_per_match=("goals", "mean"),
        wc_referee_penalty_goals_per_match=("penalty_goals", "mean"),
        wc_referee_bookings_per_match=("bookings", "mean"),
        wc_referee_yellow_cards_per_match=("yellow_cards", "mean"),
        wc_referee_red_cards_per_match=("red_cards", "mean"),
        wc_referee_sending_offs_per_match=("sending_offs", "mean"),
        wc_referee_draw_rate=("draw", "mean"),
    ).reset_index()
    return out.sort_values(["wc_referee_matches", "wc_referee_bookings_per_match"], ascending=[False, False])


def write_summary(outdir: Path, team_priors: pd.DataFrame, sidecar: pd.DataFrame, stage: pd.DataFrame, refs: pd.DataFrame) -> None:
    ready = sidecar["fjelstul_historical_prior_ready_flag"].value_counts().rename_axis("ready").reset_index(name="teams")
    recent = sidecar["fjelstul_recent_prior_ready_flag"].value_counts().rename_axis("ready").reset_index(name="teams")
    ready.to_csv(outdir / "world_cup_2026_fjelstul_team_prior_coverage.csv", index=False)
    recent.to_csv(outdir / "world_cup_2026_fjelstul_recent_team_prior_coverage.csv", index=False)
    top = sidecar.sort_values(["recent_wc_goal_diff_per_match", "recent_wc_matches"], ascending=[False, False]).head(12)
    top_lines = [
        f"- {r.team_slug}: recent GD/match={r.recent_wc_goal_diff_per_match:.2f}, "
        f"recent matches={int(r.recent_wc_matches)}, recent win={r.recent_wc_win_rate:.2f}"
        for r in top.itertuples()
        if pd.notna(getattr(r, "recent_wc_goal_diff_per_match", pd.NA))
    ]
    lines = [
        "# Fjelstul World Cup Historical Priors",
        "",
        "Research-only macro priors from the Fjelstul World Cup Database archive.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_fjelstul_team_historical_priors.csv'}`",
        f"- `{outdir / 'world_cup_2026_team_historical_prior_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_fjelstul_stage_event_priors.csv'}`",
        f"- `{outdir / 'world_cup_fjelstul_referee_style_priors.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Historical team priors joined to 2026 field: {int(sidecar['fjelstul_historical_prior_ready_flag'].sum())} / {len(sidecar)}",
        f"- Recent 2010-2018 team priors joined to 2026 field: {int(sidecar['fjelstul_recent_prior_ready_flag'].sum())} / {len(sidecar)}",
        f"- Historical team prior rows: {len(team_priors)}",
        f"- Stage/event prior rows: {len(stage)}",
        f"- Referee style prior rows: {len(refs)}",
        "",
        "## Top Recent World Cup Team Priors",
        "",
        *top_lines,
        "",
        "## License / Attribution",
        "",
        "- Source: Fjelstul World Cup Database by Joshua C. Fjelstul, Ph.D.",
        "- Copyright notice from source: © 2022 Joshua C. Fjelstul, Ph.D.",
        "- Source repository: https://www.github.com/jfjelstul/worldcup",
        "- License stated by source: CC-BY-SA 4.0.",
        "- These derived sidecars should keep this attribution and need license review before commercial product packaging.",
        "",
        "## Notes",
        "",
        "- Archive coverage ends at 2018, so this is not a 2022 or 2026 squad layer.",
        "- Use as macro World Cup priors and QA/context features only.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    team_priors, sidecar = build_team_priors(args.archive_root, launch)
    stage_priors = build_stage_event_priors(args.archive_root)
    referee_priors = build_referee_priors(args.archive_root)

    team_priors.to_csv(args.outdir / "world_cup_fjelstul_team_historical_priors.csv", index=False)
    sidecar.to_csv(args.outdir / "world_cup_2026_team_historical_prior_sidecar.csv", index=False)
    stage_priors.to_csv(args.outdir / "world_cup_fjelstul_stage_event_priors.csv", index=False)
    referee_priors.to_csv(args.outdir / "world_cup_fjelstul_referee_style_priors.csv", index=False)
    write_summary(args.outdir, team_priors, sidecar, stage_priors, referee_priors)
    print(f"[ok] teams={len(team_priors)} sidecar={len(sidecar)} stage={len(stage_priors)} refs={len(referee_priors)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
