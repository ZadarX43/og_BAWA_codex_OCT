#!/usr/bin/env python3
"""Audit website/static data footprint for Cloudflare planning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "frontend/public/data"
REPORT_DIR = ROOT / "reports/latest/site_data_footprint_audit"
DOC_PATH = ROOT / "docs/SITE_DATA_FOOTPRINT_AUDIT_2026-05-14.md"

GROUPS = {
    "public_core": [
        PUBLIC_DATA / "publish_summary.json",
        PUBLIC_DATA / "public_predictions.json",
        PUBLIC_DATA / "premium_predictions.json",
        PUBLIC_DATA / "weekly_results.json",
        PUBLIC_DATA / "results_archive.json",
        PUBLIC_DATA / "live_results_feed.json",
    ],
    "fixture_decision_intelligence": [PUBLIC_DATA / "fixture_decision_intelligence"],
    "fixture_lineup_intelligence": [PUBLIC_DATA / "fixture_lineup_intelligence"],
    "fixture_h2h_support": [PUBLIC_DATA / "fixture_h2h_support"],
    "site_data": [PUBLIC_DATA / "site_data"],
    "team_intelligence": [PUBLIC_DATA / "team_intelligence"],
    "player_intelligence": [PUBLIC_DATA / "player_intelligence"],
    "external_content": [PUBLIC_DATA / "external_content"],
    "weather_context": [PUBLIC_DATA / "weather_context"],
    "logo_assets": [PUBLIC_DATA / "api_football_logo_asset_manifest.json"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def walk_paths(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    files.append(Path(dirpath) / filename)
    return files


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def summarize_paths(paths: list[Path]) -> dict[str, Any]:
    files = walk_paths(paths)
    size = sum(file_size(path) for path in files)
    json_files = [path for path in files if path.suffix.lower() == ".json"]
    return {
        "files": len(files),
        "json_files": len(json_files),
        "bytes": size,
        "human_size": human_bytes(size),
        "largest_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": file_size(path),
                "human_size": human_bytes(file_size(path)),
            }
            for path in sorted(files, key=file_size, reverse=True)[:10]
        ],
    }


def main() -> None:
    group_summaries = {name: summarize_paths(paths) for name, paths in GROUPS.items()}
    total_files = walk_paths([PUBLIC_DATA])
    total_bytes = sum(file_size(path) for path in total_files)
    report = {
        "generated_at": utc_now(),
        "public_data_root": str(PUBLIC_DATA),
        "total": {
            "files": len(total_files),
            "json_files": sum(1 for path in total_files if path.suffix.lower() == ".json"),
            "bytes": total_bytes,
            "human_size": human_bytes(total_bytes),
        },
        "groups": group_summaries,
        "hosting_notes": [
            "Static JSON is fine for public previews and compact route payloads.",
            "D1/KV should own broader current-season lookup tables and cached route payloads.",
            "Historical/deep rows should stay gated and fetched on demand for higher tiers.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report)
    print(json.dumps(report["total"], indent=2))


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Site Data Footprint Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Public data root: `{Path(report['public_data_root']).relative_to(ROOT)}`",
        "",
        "## Total Static Public Data",
        "",
        f"- Files: {report['total']['files']}",
        f"- JSON files: {report['total']['json_files']}",
        f"- Size: {report['total']['human_size']} ({report['total']['bytes']} bytes)",
        "",
        "## Group Breakdown",
        "",
        "| Group | Files | JSON | Size |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in sorted(report["groups"].items(), key=lambda pair: pair[1]["bytes"], reverse=True):
        lines.append(f"| {name} | {item['files']} | {item['json_files']} | {item['human_size']} |")
    lines.extend(["", "## Largest Files By Group", ""])
    for name, item in sorted(report["groups"].items(), key=lambda pair: pair[1]["bytes"], reverse=True):
        if not item["largest_files"]:
            continue
        lines.extend([f"### {name}", ""])
        for file_item in item["largest_files"][:5]:
            lines.append(f"- `{file_item['path']}` — {file_item['human_size']}")
        lines.append("")
    lines.extend(
        [
            "## Hosting Notes",
            "",
            "- Keep public proof/results as compact JSON.",
            "- Keep current-season active competition data in D1/KV-backed route payloads.",
            "- Reserve deep historical rows, full event logs, and downloadable payloads for higher tiers.",
        ]
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
