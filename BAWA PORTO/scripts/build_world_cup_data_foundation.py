#!/usr/bin/env python3
"""Build an offline World Cup data-foundation report and feature-frame scaffold.

This script scans local data for FIFA/World Cup coverage, writes a readiness
report, and creates a shadow-model schema. It does not fetch remote data or make
production predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_API_ROOT = Path("data_sources/api_football")
DEFAULT_OUTDIR = Path("reports/2026-05-06/world_cup_data_foundation")

WORLD_CUP_KEYWORDS = ["world_cup", "world cup", "fifa"]

SHADOW_SCHEMA = [
    "fixture_id",
    "fixture_key",
    "competition",
    "season",
    "stage",
    "group",
    "match_date",
    "kickoff_ts_utc",
    "venue_name",
    "venue_city",
    "neutral_site_flag",
    "home_team_name",
    "away_team_name",
    "home_confederation",
    "away_confederation",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_elo",
    "away_elo",
    "rest_days_home",
    "rest_days_away",
    "travel_proxy_home",
    "travel_proxy_away",
    "squad_value_home",
    "squad_value_away",
    "injury_absence_count_home",
    "injury_absence_count_away",
    "projected_xi_strength_home",
    "projected_xi_strength_away",
    "recent_attack_index_home",
    "recent_attack_index_away",
    "recent_defence_index_home",
    "recent_defence_index_away",
    "od_home",
    "od_draw",
    "od_away",
    "od_over25",
    "od_under25",
    "od_btts_yes",
    "od_btts_no",
    "p_shadow_ftr_home",
    "p_shadow_ftr_draw",
    "p_shadow_ftr_away",
    "p_shadow_btts_yes",
    "p_shadow_ou25",
    "p_shadow_home_tg15",
    "p_shadow_away_tg15",
    "confidence_tier",
    "shadow_only_reason",
]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def file_matches_world_cup(path: Path) -> bool:
    text = str(path).lower().replace("-", "_")
    return any(keyword in text for keyword in WORLD_CUP_KEYWORDS)


def summarize_local_files(api_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(api_root.glob("**/*.csv")):
        if not file_matches_world_cup(path):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
            frame = pd.read_csv(path, low_memory=False)
            rows.append(
                {
                    "path": str(path),
                    "rows": len(frame),
                    "columns": len(header.columns),
                    "fixture_id_present": int("fixture_id" in header.columns),
                    "team_present": int(bool({"team_id", "home_team_name", "away_team_name"} & set(header.columns))),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "rows": 0,
                    "columns": 0,
                    "fixture_id_present": 0,
                    "team_present": 0,
                    "read_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default=str(DEFAULT_API_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    api_root = Path(args.api_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    local_wc = summarize_local_files(api_root)
    local_wc.to_csv(outdir / "world_cup_local_data_inventory.csv", index=False)

    schema = pd.DataFrame({"column": SHADOW_SCHEMA})
    schema.to_csv(outdir / "world_cup_shadow_feature_frame_schema.csv", index=False)

    empty_frame = pd.DataFrame(columns=SHADOW_SCHEMA)
    empty_frame.to_csv(outdir / "world_cup_shadow_feature_frame_template.csv", index=False)

    readiness = "NO_LOCAL_WORLD_CUP_DATA_FOUND"
    if not local_wc.empty and int(local_wc["rows"].sum()) > 0:
        readiness = "LOCAL_WORLD_CUP_DATA_PRESENT_NEEDS_NORMALIZATION"

    data_requirements = pd.DataFrame(
        [
            {
                "family": "fixtures_groups_venues",
                "purpose": "fixture identity, neutral venue, group/stage, rest/travel calendar",
                "status": "required_first",
            },
            {
                "family": "odds_prematch",
                "purpose": "market prior and value reference for FTR, BTTS, OU25, TG15",
                "status": "required_first",
            },
            {
                "family": "squads_lineups_injuries",
                "purpose": "availability and projected XI strength",
                "status": "required_for_confidence",
            },
            {
                "family": "team_form_style",
                "purpose": "national-team attack/defence shape and tournament goal threat",
                "status": "required_for_model_signal",
            },
            {
                "family": "travel_rest_weather",
                "purpose": "tournament-specific fatigue and venue context",
                "status": "shadow_context",
            },
        ]
    )
    data_requirements.to_csv(outdir / "world_cup_data_requirements.csv", index=False)

    copy_blocks = [
        {
            "module": "Tournament Intelligence",
            "copy": "Shadow-only World Cup layer combining squad availability, travel/rest, team goal threat, and market confidence.",
        },
        {
            "module": "Confidence Tiers",
            "copy": "Every tournament pick stays in shadow until fixture data, odds, and availability agree cleanly.",
        },
        {
            "module": "Team Goal Threat",
            "copy": "TG15 highlights sides whose attacking profile, opponent defence, and market shape point to sustained pressure.",
        },
        {
            "module": "Safety Rule",
            "copy": "Tournament outputs are research signals first; no production promotion before live-board sanity checks.",
        },
    ]
    pd.DataFrame(copy_blocks).to_csv(outdir / "world_cup_dashboard_copy_blocks.csv", index=False)

    summary = [
        "# World Cup Data Foundation",
        "",
        f"Readiness: `{readiness}`",
        "",
        "FIFA World Cup 2026 runs from 2026-06-11 to 2026-07-19. "
        "This report is an offline scaffold only; no API calls were made.",
        "",
        "Official schedule reference: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums",
        "",
        "## Local World Cup Inventory",
        markdown_table(local_wc.head(30)),
        "",
        "## Required Data Families",
        markdown_table(data_requirements),
        "",
        "## Shadow Markets",
        "- FTR shadow probabilities",
        "- BTTS shadow probabilities",
        "- OU25 shadow probabilities",
        "- TG15 home/away shadow probabilities",
        "",
        "## Dashboard Copy Direction",
        markdown_table(pd.DataFrame(copy_blocks)),
        "",
        "## Next Implementation Step",
        (
            "Once local World Cup fixture/odds/squad files exist, normalize them into the "
            "shadow feature-frame schema and keep all outputs OBSERVE/shadow-only."
        ),
    ]
    (outdir / "world_cup_data_foundation.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"readiness={readiness} local_files={len(local_wc)}")


if __name__ == "__main__":
    main()
