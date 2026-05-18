#!/usr/bin/env python3
"""Shared timestamp contract for publish-safe website payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SNAPSHOT_FIELDS = (
    "capture_generated_at",
    "source_data_cutoff_at",
    "fixture_kickoff_at",
    "pre_kickoff_eligible",
    "snapshot_phase",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        if len(text) == 10:
            try:
                parsed = datetime.fromisoformat(f"{text}T00:00:00+00:00")
            except ValueError:
                return None
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_snapshot_metadata(
    *,
    capture_generated_at: Any,
    fixture_kickoff_at: Any,
    source_data_cutoff_at: Any | None = None,
    snapshot_phase: str | None = None,
) -> dict[str, Any]:
    capture_ts = parse_utc(capture_generated_at)
    kickoff_ts = parse_utc(fixture_kickoff_at)
    cutoff_ts = parse_utc(source_data_cutoff_at) or capture_ts

    pre_kickoff_eligible = bool(
        capture_ts is not None
        and cutoff_ts is not None
        and kickoff_ts is not None
        and capture_ts <= kickoff_ts
        and cutoff_ts <= kickoff_ts
    )
    if snapshot_phase:
        phase = snapshot_phase
    elif pre_kickoff_eligible:
        phase = "pre_kickoff"
    elif capture_ts is not None and kickoff_ts is not None and capture_ts > kickoff_ts:
        phase = "post_kickoff"
    else:
        phase = "unknown"

    return {
        "capture_generated_at": iso_utc(capture_ts),
        "source_data_cutoff_at": iso_utc(cutoff_ts),
        "fixture_kickoff_at": iso_utc(kickoff_ts),
        "pre_kickoff_eligible": pre_kickoff_eligible,
        "snapshot_phase": phase,
    }


def metadata_from_fixture(
    fixture: dict[str, Any],
    *,
    capture_generated_at: Any | None = None,
    source_data_cutoff_at: Any | None = None,
    snapshot_phase: str | None = None,
) -> dict[str, Any]:
    return build_snapshot_metadata(
        capture_generated_at=capture_generated_at
        or fixture.get("capture_generated_at")
        or fixture.get("updated_at")
        or fixture.get("generated_at"),
        source_data_cutoff_at=source_data_cutoff_at
        or fixture.get("source_data_cutoff_at")
        or fixture.get("updated_at")
        or fixture.get("generated_at"),
        fixture_kickoff_at=fixture.get("fixture_kickoff_at") or fixture.get("kickoff_time") or fixture.get("kickoff_at"),
        snapshot_phase=snapshot_phase,
    )


def extract_snapshot_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in SNAPSHOT_FIELDS if field in payload}
