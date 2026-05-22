from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.fetch_current_player_window import DEFAULT_LEAGUES, parse_csv_set


NORMALIZED_RE = re.compile(r"^(?P<prefix>fixtures_master|lineups|match_player_stats|injuries|sidelined)__(?P<tag>.+)__(?P<season>\d{4})\.csv$")
RAW_RE = re.compile(
    r"^(?P<tag>.+)__league_(?P<league_id>\d+)__season_(?P<season>\d{4})__"
    r"(?P<from_date>\d{4}-\d{2}-\d{2})_to_(?P<to_date>\d{4}-\d{2}-\d{2})__"
    r"(?P<kind>fixtures|fixtures_bundle|injuries|sidelined)\.jsonl$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def csv_row_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except OSError:
        return 0


def jsonl_line_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return sum(1 for line in path.open("r", encoding="utf-8", errors="ignore") if line.strip())
    except OSError:
        return 0


def load_existing_manifest(outdir: Path) -> dict[str, Any]:
    path = outdir / "CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_normalized(outdir: Path) -> dict[tuple[str, int], dict[str, Path]]:
    normalized_dir = outdir / "normalized"
    assets: dict[tuple[str, int], dict[str, Path]] = {}
    if not normalized_dir.exists():
        return assets
    field_by_prefix = {
        "fixtures_master": "fixtures_csv",
        "lineups": "lineups_csv",
        "match_player_stats": "player_stats_csv",
        "injuries": "injuries_csv",
        "sidelined": "sidelined_csv",
    }
    for path in sorted(normalized_dir.glob("*.csv")):
        match = NORMALIZED_RE.match(path.name)
        if not match:
            continue
        key = (match.group("tag"), int(match.group("season")))
        assets.setdefault(key, {})[field_by_prefix[match.group("prefix")]] = path
    return assets


def discover_raw(outdir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    raw_dir = outdir / "raw"
    assets: dict[tuple[str, int], dict[str, Any]] = {}
    if not raw_dir.exists():
        return assets
    field_by_kind = {
        "fixtures": "fixtures_raw",
        "fixtures_bundle": "bundle_raw",
        "injuries": "injuries_raw",
        "sidelined": "sidelined_raw",
    }
    for path in sorted(raw_dir.glob("*.jsonl")):
        match = RAW_RE.match(path.name)
        if not match:
            continue
        tag = match.group("tag")
        season = int(match.group("season"))
        key = (tag, season)
        rec = assets.setdefault(
            key,
            {
                "tag": tag,
                "league_id": int(match.group("league_id")),
                "season": season,
                "from_date": match.group("from_date"),
                "to_date": match.group("to_date"),
            },
        )
        rec[field_by_kind[match.group("kind")]] = path
    return assets


def cfg_for(tag: str) -> dict[str, Any]:
    cfg = DEFAULT_LEAGUES.get(tag, {})
    return {"league_id": int(cfg.get("league_id", 0) or 0), "season": int(cfg.get("season", 0) or 0)}


def build_combined_manifest(
    outdir: Path,
    *,
    from_date: str = "",
    to_date: str = "",
    injury_query_scopes: set[str] | None = None,
    include_sidelined: bool | None = None,
    seen_registry_csv: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = discover_normalized(outdir)
    raw = discover_raw(outdir)
    existing = load_existing_manifest(outdir)
    keys = sorted(set(normalized) | set(raw), key=lambda item: item[0])

    existing_scopes = set(existing.get("injury_query_scopes") or [])
    scopes = sorted(injury_query_scopes if injury_query_scopes is not None else existing_scopes)
    if include_sidelined is None:
        include_sidelined = bool(existing.get("include_sidelined", False))
    seen_registry_csv = seen_registry_csv or str(existing.get("seen_registry_csv") or "")

    bundles: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for tag, season in keys:
        raw_rec = raw.get((tag, season), {})
        norm_rec = normalized.get((tag, season), {})
        cfg = cfg_for(tag)
        league_id = int(raw_rec.get("league_id") or cfg.get("league_id") or 0)
        row_from = from_date or str(raw_rec.get("from_date") or "")
        row_to = to_date or str(raw_rec.get("to_date") or "")
        paths = {
            "fixtures_raw": raw_rec.get("fixtures_raw"),
            "bundle_raw": raw_rec.get("bundle_raw"),
            "injuries_raw": raw_rec.get("injuries_raw"),
            "sidelined_raw": raw_rec.get("sidelined_raw"),
            "fixtures_csv": norm_rec.get("fixtures_csv"),
            "lineups_csv": norm_rec.get("lineups_csv"),
            "player_stats_csv": norm_rec.get("player_stats_csv"),
            "injuries_csv": norm_rec.get("injuries_csv"),
            "sidelined_csv": norm_rec.get("sidelined_csv"),
        }
        counts = {
            "fixtures_rows": csv_row_count(paths["fixtures_csv"]),
            "lineup_rows": csv_row_count(paths["lineups_csv"]),
            "player_stat_rows": csv_row_count(paths["player_stats_csv"]),
            "injury_rows": csv_row_count(paths["injuries_csv"]),
            "sidelined_rows": csv_row_count(paths["sidelined_csv"]),
        }
        bundle = {
            "tag": tag,
            "league_id": league_id,
            "season": season,
            "from_date": row_from,
            "to_date": row_to,
            "fixture_ids": counts["fixtures_rows"],
            "bundle_requests": jsonl_line_count(paths["bundle_raw"]),
            "injury_requests": jsonl_line_count(paths["injuries_raw"]),
            "sidelined_requests": jsonl_line_count(paths["sidelined_raw"]),
            "paths": {key: str(value) if value else "" for key, value in paths.items()},
            "counts": counts,
        }
        bundles.append(bundle)
        summary_rows.append(
            {
                "league_tag": tag,
                "league_id": league_id,
                "season": season,
                "from_date": row_from,
                "to_date": row_to,
                "fixture_ids": bundle["fixture_ids"],
                "bundle_requests": bundle["bundle_requests"],
                "injury_requests": bundle["injury_requests"],
                "sidelined_requests": bundle["sidelined_requests"],
                **counts,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = outdir / "CURRENT_INJURY_LINEUP_WINDOW_SUMMARY.csv"
    summary.to_csv(summary_csv, index=False)

    manifest = {
        "generated_at": utc_now(),
        "manifest_source": "combined_from_output_files",
        "outdir": str(outdir),
        "league_tags": [bundle["tag"] for bundle in bundles],
        "summary_csv": str(summary_csv),
        "bundle_count": len(bundles),
        "total_fixtures": int(summary["fixtures_rows"].sum()) if not summary.empty else 0,
        "total_injury_rows": int(summary["injury_rows"].sum()) if not summary.empty else 0,
        "total_sidelined_rows": int(summary["sidelined_rows"].sum()) if not summary.empty else 0,
        "total_lineup_rows": int(summary["lineup_rows"].sum()) if not summary.empty else 0,
        "total_player_stat_rows": int(summary["player_stat_rows"].sum()) if not summary.empty else 0,
        "injury_query_scopes": scopes,
        "include_sidelined": bool(include_sidelined),
        "seen_registry_csv": seen_registry_csv,
        "bundles": bundles,
    }
    manifest_path = outdir / "CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a combined current injury-lineup window manifest from an output folder.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--from-date", default="")
    parser.add_argument("--to-date", default="")
    parser.add_argument("--injury-query-scopes", default="", help="Optional comma-separated override.")
    parser.add_argument("--include-sidelined", action="store_true", default=None)
    parser.add_argument("--seen-registry-csv", default="")
    args = parser.parse_args()

    scopes = parse_csv_set(args.injury_query_scopes) if args.injury_query_scopes else None
    summary, manifest = build_combined_manifest(
        args.outdir,
        from_date=args.from_date,
        to_date=args.to_date,
        injury_query_scopes=scopes,
        include_sidelined=args.include_sidelined,
        seen_registry_csv=args.seen_registry_csv,
    )
    print(f"WROTE {args.outdir / 'CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json'}")
    print(f"WROTE {args.outdir / 'CURRENT_INJURY_LINEUP_WINDOW_SUMMARY.csv'}")
    print(f"leagues={len(summary)} fixtures={manifest['total_fixtures']} injuries={manifest['total_injury_rows']} sidelined={manifest['total_sidelined_rows']}")
    if not summary.empty:
        print(summary[["league_tag", "season", "fixture_ids", "injury_rows", "sidelined_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
