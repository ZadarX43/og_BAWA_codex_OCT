#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REPORTS_ROOT = ROOT / "reports"
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
MANIFEST_PATH = NORMALIZED_DIR / "current_site_normalized_import_manifest.json"
REPORT_PATH = ROOT / "reports" / "latest" / "CURRENT_SITE_NORMALIZED_IMPORT_REPORT.md"

FAMILIES = ("fixtures_master", "lineups", "match_player_stats", "match_team_stats")
DEFAULT_LEAGUES = {
    "Australia_A_League": {"league_id": 188, "season": 2025},
    "Austria_Bundesliga": {"league_id": 218, "season": 2025},
    "Belgium_Pro": {"league_id": 144, "season": 2025},
    "Brazil_Serie_A": {"league_id": 71, "season": 2026},
    "Denmark_Superliga": {"league_id": 119, "season": 2025},
    "England_Championship": {"league_id": 40, "season": 2025},
    "England_EFL_League_1": {"league_id": 41, "season": 2025},
    "England_Premier_League": {"league_id": 39, "season": 2025},
    "France_Ligue_1": {"league_id": 61, "season": 2025},
    "Germany_Bundesliga": {"league_id": 78, "season": 2025},
    "Germany_Bundesliga_2": {"league_id": 79, "season": 2025},
    "Italy_Serie_A": {"league_id": 135, "season": 2025},
    "Netherlands_Eredivisie": {"league_id": 88, "season": 2025},
    "Norway_Eliteserien": {"league_id": 103, "season": 2026},
    "Portugal_Liga": {"league_id": 94, "season": 2025},
    "Saudi_Pro_League": {"league_id": 307, "season": 2025},
    "Scotland_Premiership": {"league_id": 179, "season": 2025},
    "South_Korea_K_League": {"league_id": 292, "season": 2026},
    "Spain_La_Liga": {"league_id": 140, "season": 2025},
    "Switzerland_Super_League": {"league_id": 207, "season": 2025},
    "Turkey_Super_Lig": {"league_id": 203, "season": 2025},
    "USA_MLS": {"league_id": 253, "season": 2026},
}
NORMALIZED_SCHEMAS = {
    "fixtures_master": [
        "fixture_id",
        "fixture_key",
        "league",
        "league_id",
        "season",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "kickoff_ts_utc",
        "status",
        "venue_id",
        "venue_name",
        "referee_name",
    ],
    "match_team_stats": [
        "fixture_id",
        "team_id",
        "team_name",
        "is_home",
        "goals_for",
        "goals_against",
        "ht_goals_for",
        "ht_goals_against",
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "shots_outside_box",
        "blocked_shots",
        "possession_pct",
        "passes_total",
        "passes_accurate",
        "corners_for",
        "fouls_for",
        "yellow_cards",
        "red_cards",
    ],
    "match_player_stats": [
        "fixture_id",
        "player_id",
        "team_id",
        "player_name",
        "position",
        "minutes",
        "started_flag",
        "subbed_on_flag",
        "subbed_off_flag",
        "rating",
        "goals",
        "assists",
        "shots_total",
        "shots_on_target",
        "passes_total",
        "passes_key",
        "passes_accurate",
        "tackles",
        "interceptions",
        "blocks",
        "duels_total",
        "duels_won",
        "dribbles_attempted",
        "dribbles_successful",
        "dribbled_past",
        "fouls_drawn",
        "fouls_committed",
        "yellow_cards",
        "red_cards",
        "saves",
        "goals_conceded",
    ],
    "lineups": [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "formation",
        "is_starting_xi",
        "position",
        "lineup_known_pre_kickoff_flag",
        "lineup_published_ts_utc",
    ],
}
DEDUP_KEYS = {
    "fixtures_master": ("fixture_id",),
    "lineups": ("fixture_id", "team_id", "player_id", "is_starting_xi"),
    "match_player_stats": ("fixture_id", "team_id", "player_id"),
    "match_team_stats": ("fixture_id", "team_id"),
}


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge current report-scoped API-Football normalized lineups/stats into the canonical "
            "data_sources/api_football/normalized store for the active site competitions."
        )
    )
    parser.add_argument("--write", action="store_true", help="Write merged canonical CSVs and manifest.")
    parser.add_argument("--league-tags", default="", help="Comma-separated tags. Defaults to active site tags.")
    parser.add_argument("--families", default=",".join(FAMILIES), help="Comma-separated normalized families to import.")
    parser.add_argument(
        "--include-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the existing canonical CSV as the first merge source.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def read_rows(path: Path, schema: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: str(row.get(column, "") or "") for column in schema})
        return rows


def write_rows(path: Path, schema: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(rows)


def dedup_key(family: str, row: dict[str, str]) -> tuple[str, ...]:
    keys = DEDUP_KEYS[family]
    values = tuple(str(row.get(key, "") or "").strip() for key in keys)
    if any(values):
        return values
    return tuple(str(row.get(column, "") or "").strip() for column in NORMALIZED_SCHEMAS[family])


def discover_sources(family: str, tag: str, season: int, include_existing: bool) -> list[Path]:
    filename = f"{family}__{tag}__{season}.csv"
    sources = []
    existing = NORMALIZED_DIR / filename
    if include_existing and existing.exists():
        sources.append(existing)
    reports = sorted(
        REPORTS_ROOT.glob(f"**/normalized/{filename}"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    sources.extend(path for path in reports if path not in sources)
    return sources


def merge_sources(family: str, sources: list[Path]) -> list[dict[str, str]]:
    schema = NORMALIZED_SCHEMAS[family]
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    for source in sources:
        for row in read_rows(source, schema):
            merged[dedup_key(family, row)] = row
    return list(merged.values())


def build_report(records: list[dict[str, Any]], copied: list[dict[str, Any]]) -> str:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    lines = [
        "# CURRENT_SITE_NORMALIZED_IMPORT_REPORT",
        "",
        f"Generated: `{datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}`",
        f"Canonical destination: `{NORMALIZED_DIR.relative_to(ROOT)}`",
        "",
        "## Summary",
    ]
    for family in FAMILIES:
        family_records = by_family.get(family, [])
        lines.append(
            f"- `{family}`: `{sum(1 for record in family_records if record['source_count'])}` files with sources, "
            f"`{sum(record['merged_rows'] for record in family_records)}` merged rows"
        )

    lines.extend(["", "## Active Competition Files"])
    for record in records:
        status = next((entry["status"] for entry in copied if entry["filename"] == record["filename"]), "dry-run")
        lines.append(
            f"- `{record['filename']}`: `{record['merged_rows']}` rows from `{record['source_count']}` sources ({status})"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Report-scoped CSVs are merged rather than selecting the newest file only.",
            "- This preserves wider current-window coverage while allowing newer fixture rows to overwrite duplicates.",
            "- Missing leagues or zero-row team-stat files require a fresh current-context API refresh.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    selected = parse_csv_set(args.league_tags) if args.league_tags else set(DEFAULT_LEAGUES)
    families = tuple(family for family in parse_csv_set(args.families) if family in FAMILIES)
    unknown = selected - set(DEFAULT_LEAGUES)
    if unknown:
        raise SystemExit(f"Unknown league tags: {sorted(unknown)}")
    if not families:
        raise SystemExit("No valid families selected.")

    records: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []
    for tag in sorted(selected):
        season = int(DEFAULT_LEAGUES[tag]["season"])
        for family in families:
            filename = f"{family}__{tag}__{season}.csv"
            sources = discover_sources(family, tag, season, include_existing=args.include_existing)
            merged_rows = merge_sources(family, sources)
            destination = NORMALIZED_DIR / filename
            record = {
                "family": family,
                "tag": tag,
                "season": season,
                "filename": filename,
                "canonical_path": str(destination.relative_to(ROOT)),
                "source_count": len(sources),
                "sources": [
                    {
                        "path": str(source.relative_to(ROOT)),
                        "row_count": row_count(source),
                        "sha256": sha256_file(source),
                    }
                    for source in sources
                ],
                "merged_rows": len(merged_rows),
            }
            records.append(record)

            if args.write:
                status = "no_sources"
                if sources:
                    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
                    write_rows(tmp_path, NORMALIZED_SCHEMAS[family], merged_rows)
                    status = "unchanged"
                    if not destination.exists() or sha256_file(destination) != sha256_file(tmp_path):
                        shutil.move(str(tmp_path), str(destination))
                        status = "written"
                    else:
                        tmp_path.unlink(missing_ok=True)
                copied.append({"filename": filename, "status": status})

    if args.write:
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "families": list(families),
            "league_tags": sorted(selected),
            "files": records,
        }
        MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(records, copied), encoding="utf-8")

    print(f"Active tags: {len(selected)}")
    print(f"Families: {', '.join(families)}")
    print(f"Records inspected: {len(records)}")
    if args.write:
        print(f"Manifest written: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Report written: {REPORT_PATH.relative_to(ROOT)}")
    for family in families:
        family_records = [record for record in records if record["family"] == family]
        print(
            f"{family}: {sum(record['merged_rows'] for record in family_records)} rows "
            f"across {sum(1 for record in family_records if record['source_count'])} sourced files"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
