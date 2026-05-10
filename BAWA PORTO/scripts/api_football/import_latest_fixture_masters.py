#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = ROOT / "reports"
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
PUBLIC_FIXTURE_INTELLIGENCE_PATH = ROOT / "frontend" / "public" / "data" / "fixture_intelligence_public.json"
MANIFEST_PATH = NORMALIZED_DIR / "fixture_master_import_manifest.json"
REPORT_PATH = ROOT / "reports" / "latest" / "FIXTURE_MASTER_IMPORT_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the newest report-scoped API-Football fixtures_master files into the canonical normalized store."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Copy the newest fixtures_master files into data_sources/api_football/normalized and update the manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def discover_report_fixture_masters() -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(REPORTS_ROOT.glob("**/normalized/fixtures_master__*.csv")):
        grouped[path.name].append(path)
    return grouped


def load_public_missing_leagues() -> Counter[str]:
    if not PUBLIC_FIXTURE_INTELLIGENCE_PATH.exists():
        return Counter()
    payload = json.loads(PUBLIC_FIXTURE_INTELLIGENCE_PATH.read_text(encoding="utf-8"))
    rows = payload.get("fixtures", []) if isinstance(payload, dict) else []
    missing = [row for row in rows if isinstance(row, dict) and row.get("api_fixture_id") is None]
    return Counter(str(row.get("league", "") or "").strip() for row in missing if str(row.get("league", "") or "").strip())


def select_latest_sources(grouped: dict[str, list[Path]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename, paths in sorted(grouped.items()):
        latest = max(paths, key=lambda path: (path.stat().st_mtime, str(path)))
        records.append(
            {
                "filename": filename,
                "source_path": latest,
                "all_candidate_paths": [str(path.relative_to(ROOT)) for path in paths],
                "mtime": latest.stat().st_mtime,
                "row_count": load_csv_row_count(latest),
                "sha256": sha256_file(latest),
            }
        )
    return records


def write_manifest(records: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "files": [
            {
                "filename": record["filename"],
                "canonical_path": str((NORMALIZED_DIR / record["filename"]).relative_to(ROOT)),
                "source_path": str(record["source_path"].relative_to(ROOT)),
                "row_count": record["row_count"],
                "sha256": record["sha256"],
            }
            for record in records
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_report(records: list[dict[str, Any]], copied: list[dict[str, Any]], missing_leagues: Counter[str]) -> str:
    season_counts = Counter()
    for record in records:
        parts = record["filename"].removesuffix(".csv").split("__")
        if len(parts) >= 3:
            season_counts[parts[2]] += 1

    lines = [
        "# FIXTURE_MASTER_IMPORT_REPORT",
        "",
        f"Canonical destination: `{NORMALIZED_DIR.relative_to(ROOT)}`",
        f"Discovered report fixture-master files: `{len(records)}`",
        "",
        "## Imported Files",
    ]
    if copied:
        lines.extend(
            f"- `{entry['filename']}` from `{entry['source_path'].relative_to(ROOT)}` ({entry['status']})"
            for entry in copied
        )
    else:
        lines.append("- no file copies were performed")

    lines.extend(
        [
            "",
            "## Current-Season Coverage Imported",
            *(f"- season `{season}`: `{count}` files" for season, count in sorted(season_counts.items())),
            "",
            "## Published Fixture Identity Gaps",
        ]
    )
    if missing_leagues:
        lines.extend(f"- `{league}`: `{count}` fixtures still unresolved in published artifact" for league, count in missing_leagues.most_common())
    else:
        lines.append("- no unresolved published fixtures")

    lines.extend(
        [
            "",
            "## Notes",
            "- This import promotes the newest report-scoped normalized fixture masters into the canonical normalized directory.",
            "- It does not fetch new API data by itself; leagues still missing after import require a fresh API-Football refresh window.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    grouped = discover_report_fixture_masters()
    records = select_latest_sources(grouped)
    copied: list[dict[str, Any]] = []

    if args.write:
        NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
        for record in records:
            destination = NORMALIZED_DIR / record["filename"]
            status = "unchanged"
            if not destination.exists() or sha256_file(destination) != record["sha256"]:
                shutil.copy2(record["source_path"], destination)
                status = "copied"
            copied.append(
                {
                    "filename": record["filename"],
                    "source_path": record["source_path"],
                    "status": status,
                }
            )
        write_manifest(records)

    missing_leagues = load_public_missing_leagues()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(records, copied, missing_leagues), encoding="utf-8")

    print(f"Discovered latest report fixture masters: {len(records)}")
    if args.write:
        copied_count = sum(1 for entry in copied if entry["status"] == "copied")
        print(f"Canonical fixture-master files copied: {copied_count}")
        print(f"Manifest written: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Report written: {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
