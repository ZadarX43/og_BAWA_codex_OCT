#!/usr/bin/env python3
"""Backfill the publish-safe website estate with explicit snapshot timestamps.

This is a metadata-only website publish utility. It does not change model output,
deploy routing, or prediction policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from publish_snapshot_metadata import SNAPSHOT_FIELDS, metadata_from_fixture, utc_now_iso
from scripts.audit_pre_kickoff_intelligence_snapshots import (
    DEFAULT_PROVIDER_RESULTS,
    iso_or_blank,
    load_provider_index,
    provider_kickoff_for_fixture,
)

DEFAULT_DATA_ROOT = Path("frontend/public/data")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def annotate_fixture_feed(data_root: Path, provider_results: Path | None) -> tuple[dict[str, dict[str, Any]], int]:
    path = data_root / "fixture_intelligence_public.json"
    payload = read_json(path, {})
    fixtures = payload.get("fixtures") if isinstance(payload, dict) else None
    if not isinstance(fixtures, list):
        return {}, 0

    feed_generated_at = payload.get("capture_generated_at") or payload.get("generated_at") or utc_now_iso()
    provider_index = load_provider_index(provider_results) if provider_results and provider_results.exists() else {}
    fixture_map: dict[str, dict[str, Any]] = {}
    updated = 0
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        provider_kickoff, provider_method = provider_kickoff_for_fixture(fixture, provider_index) if provider_index else (None, "")
        if provider_kickoff is not None:
            fixture["fixture_kickoff_at"] = iso_or_blank(provider_kickoff)
            fixture["fixture_kickoff_source"] = "provider_results"
            fixture["fixture_kickoff_match_method"] = provider_method
        elif not fixture.get("fixture_kickoff_at"):
            fixture["fixture_kickoff_at"] = fixture.get("kickoff_time")
            fixture["fixture_kickoff_source"] = "site_feed"
        fixture.update(
            metadata_from_fixture(
                fixture,
                capture_generated_at=fixture.get("capture_generated_at") or fixture.get("updated_at") or feed_generated_at,
                source_data_cutoff_at=fixture.get("source_data_cutoff_at") or fixture.get("updated_at") or feed_generated_at,
            )
        )
        fixture_map[fixture_key] = fixture
        updated += 1
    write_json(path, payload)
    return fixture_map, updated


def annotate_payload(payload: dict[str, Any], fixture: dict[str, Any], capture_generated_at: str) -> bool:
    had_capture = bool(payload.get("capture_generated_at"))
    metadata = metadata_from_fixture(
        fixture,
        capture_generated_at=payload.get("capture_generated_at") or capture_generated_at,
        source_data_cutoff_at=payload.get("source_data_cutoff_at") or fixture.get("source_data_cutoff_at") or fixture.get("updated_at"),
        snapshot_phase=payload.get("snapshot_phase") if had_capture else "backfill",
    )
    before = {field: payload.get(field) for field in SNAPSHOT_FIELDS}
    payload.update(metadata)
    after = {field: payload.get(field) for field in SNAPSHOT_FIELDS}
    return before != after


def annotate_fixture_directory(data_root: Path, relative_dir: str, fixture_map: dict[str, dict[str, Any]], capture_generated_at: str) -> int:
    root = data_root / relative_dir
    if not root.exists():
        return 0
    updated = 0
    payloads_by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        fixture_key = str(payload.get("fixture_key") or path.stem).strip()
        fixture = fixture_map.get(fixture_key)
        if not fixture:
            continue
        if annotate_payload(payload, fixture, capture_generated_at):
            updated += 1
            write_json(path, payload)
        payloads_by_key[fixture_key] = payload

    index_path = root / "index.json"
    index_payload = read_json(index_path, [])
    if isinstance(index_payload, list):
        changed = False
        for row in index_payload:
            if not isinstance(row, dict):
                continue
            fixture_key = str(row.get("fixture_key") or "").strip()
            payload = payloads_by_key.get(fixture_key)
            if not payload:
                continue
            before = {field: row.get(field) for field in SNAPSHOT_FIELDS}
            for field in SNAPSHOT_FIELDS:
                row[field] = payload.get(field)
            changed = changed or before != {field: row.get(field) for field in SNAPSHOT_FIELDS}
        if changed:
            write_json(index_path, index_payload)
    return updated


def annotate_external_fixture_media(data_root: Path, fixture_map: dict[str, dict[str, Any]], capture_generated_at: str) -> int:
    root = data_root / "external_content" / "fixture_media"
    if not root.exists():
        return 0
    updated = 0
    for path in sorted(root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        fixture_key = str(payload.get("fixture_key") or path.stem).strip()
        fixture = fixture_map.get(fixture_key)
        if not fixture:
            continue
        if annotate_payload(payload, fixture, capture_generated_at):
            updated += 1
            write_json(path, payload)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--provider-results", type=Path, default=DEFAULT_PROVIDER_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_generated_at = utc_now_iso()
    fixture_map, feed_updated = annotate_fixture_feed(args.data_root, args.provider_results)
    counts = {
        "fixture_feed_rows": feed_updated,
        "fixture_decision_intelligence": annotate_fixture_directory(args.data_root, "fixture_decision_intelligence", fixture_map, capture_generated_at),
        "fixture_lineup_intelligence": annotate_fixture_directory(args.data_root, "fixture_lineup_intelligence", fixture_map, capture_generated_at),
        "fixture_h2h_support": annotate_fixture_directory(args.data_root, "fixture_h2h_support", fixture_map, capture_generated_at),
        "external_fixture_media": annotate_external_fixture_media(args.data_root, fixture_map, capture_generated_at),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
