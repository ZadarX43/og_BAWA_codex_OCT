#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FEED_PATH = ROOT / "frontend" / "public" / "data" / "fixture_intelligence_public.json"
DECISION_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_decision_intelligence"
LINEUP_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_lineup_intelligence"
H2H_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_h2h_support"
DEFAULT_PROVIDER_RESULTS = ROOT / "reports" / "latest" / "weekend_prediction_intelligence_scoring" / "raw_provider_results.json"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "pre_kickoff_snapshot_audit"


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_date_key(value: Any) -> str:
    text = str(value or "").strip()
    return text[:10].replace("-", "_") if text else ""


def club_tokens(value: Any) -> set[str]:
    aliases = {"saint": "st", "sint": "st", "st": "st"}
    drop = {"a", "ac", "afc", "cf", "cd", "club", "fc", "fk", "krc", "kv", "kvc", "rc", "royal", "sc", "the", "va"}
    tokens = set()
    for token in normalize_text(value).split("_"):
        token = aliases.get(token, token)
        if token and token not in drop:
            tokens.add(token)
    return tokens


def token_match_score(left: Any, right: Any) -> float:
    left_tokens = club_tokens(left)
    right_tokens = club_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def provider_key(match_date: Any, home_team: Any, away_team: Any) -> str:
    return "_".join(
        part
        for part in (normalize_date_key(match_date), normalize_text(home_team), normalize_text(away_team))
        if part
    )


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        text = text.replace(" ", "T") + "+00:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = text + "T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_or_blank(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def provider_result_item(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    league = item.get("league") or {}
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    return {
        "api_fixture_id": fixture.get("id"),
        "kickoff_time": fixture.get("date"),
        "league": league.get("name"),
        "league_id": league.get("id"),
        "season": league.get("season"),
        "home_team": home,
        "away_team": away,
        "provider_key": provider_key(fixture.get("date"), home, away),
    }


def load_provider_index(provider_results_path: Path) -> dict[str, Any]:
    payloads = read_json(provider_results_path, []) or []
    by_api_id: dict[str, dict[str, Any]] = {}
    by_provider_key: dict[str, dict[str, Any]] = {}
    by_site_league_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_to_site_leagues: dict[tuple[int, int], set[str]] = defaultdict(set)

    feed = read_json(FIXTURE_FEED_PATH, {}) or {}
    for fx in feed.get("fixtures") or []:
        if fx.get("api_league_id") is not None and fx.get("api_season") is not None:
            pair_to_site_leagues[(int(fx["api_league_id"]), int(fx["api_season"]))].add(str(fx.get("league") or ""))

    for row in payloads:
        league_id = row.get("league_id")
        season = row.get("season")
        for item in ((row.get("payload") or {}).get("response") or []):
            result = provider_result_item(item)
            if result.get("api_fixture_id") is not None:
                by_api_id[str(result["api_fixture_id"])] = result
            if result.get("provider_key"):
                by_provider_key[str(result["provider_key"])] = result
            if league_id is not None and season is not None:
                for site_league in pair_to_site_leagues.get((int(league_id), int(season)), set()):
                    key = f"{normalize_text(site_league)}:{normalize_date_key(result.get('kickoff_time'))}"
                    by_site_league_date[key].append(result)

    return {
        "by_api_id": by_api_id,
        "by_provider_key": by_provider_key,
        "by_site_league_date": dict(by_site_league_date),
    }


def provider_kickoff_for_fixture(fixture: dict[str, Any], provider_index: dict[str, Any]) -> tuple[datetime | None, str]:
    api_fixture_id = fixture.get("api_fixture_id")
    result = None
    method = ""
    if api_fixture_id is not None:
        result = provider_index.get("by_api_id", {}).get(str(api_fixture_id))
        method = "api_fixture_id" if result else ""
    if result is None:
        key = provider_key(fixture.get("kickoff_time"), fixture.get("home_team"), fixture.get("away_team"))
        result = provider_index.get("by_provider_key", {}).get(key)
        method = "provider_key" if result else ""
    if result is None:
        league_date_key = f"{normalize_text(fixture.get('league'))}:{normalize_date_key(fixture.get('kickoff_time'))}"
        candidates = provider_index.get("by_site_league_date", {}).get(league_date_key) or []
        best_score = 0.0
        best = None
        for candidate in candidates:
            score = min(
                token_match_score(fixture.get("home_team"), candidate.get("home_team")),
                token_match_score(fixture.get("away_team"), candidate.get("away_team")),
            )
            if score > best_score:
                best_score = score
                best = candidate
        if best_score >= 0.66:
            result = best
            method = "site_league_date_alias"
    if result is None:
        return None, ""
    return parse_dt(result.get("kickoff_time")), method


def classify_snapshot(snapshot_ts: datetime | None, kickoff_ts: datetime | None, has_explicit_ts: bool) -> str:
    if not has_explicit_ts:
        return "not_temporally_auditable"
    if snapshot_ts is None or kickoff_ts is None:
        return "unknown"
    if snapshot_ts <= kickoff_ts:
        return "pre_kickoff"
    return "post_kickoff"


def minutes_after(snapshot_ts: datetime | None, kickoff_ts: datetime | None) -> int | None:
    if snapshot_ts is None or kickoff_ts is None:
        return None
    return int((snapshot_ts - kickoff_ts).total_seconds() // 60)


def audit_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feed = read_json(args.fixture_feed, {}) or {}
    fixtures = list(feed.get("fixtures") or [])
    provider_index = load_provider_index(args.provider_results)
    feed_generated_at = parse_dt(feed.get("generated_at"))

    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_key = str(fixture.get("fixture_key") or "")
        site_kickoff = parse_dt(fixture.get("kickoff_time"))
        provider_kickoff, provider_method = provider_kickoff_for_fixture(fixture, provider_index)
        kickoff = provider_kickoff or site_kickoff
        kickoff_source = "provider" if provider_kickoff is not None else "site_feed"

        feed_row_updated = parse_dt(fixture.get("capture_generated_at") or fixture.get("updated_at"))
        feed_snapshot = feed_row_updated or feed_generated_at
        feed_status = classify_snapshot(feed_snapshot, kickoff, feed_snapshot is not None)

        decision_path = DECISION_ROOT / f"{fixture_key}.json"
        lineup_path = LINEUP_ROOT / f"{fixture_key}.json"
        h2h_path = H2H_ROOT / f"{fixture_key}.json"
        decision_payload = read_json(decision_path, {}) or {}
        lineup_payload = read_json(lineup_path, {}) or {}
        h2h_payload = read_json(h2h_path, {}) or {}

        decision_explicit = parse_dt(decision_payload.get("capture_generated_at") or decision_payload.get("generated_at") or decision_payload.get("updated_at"))
        lineup_explicit = parse_dt(lineup_payload.get("capture_generated_at") or lineup_payload.get("generated_at") or lineup_payload.get("updated_at"))
        h2h_explicit = parse_dt(h2h_payload.get("capture_generated_at") or h2h_payload.get("generated_at") or h2h_payload.get("updated_at"))

        row = {
            "fixture_key": fixture_key,
            "kickoff_time_site": iso_or_blank(site_kickoff),
            "kickoff_time_provider": iso_or_blank(provider_kickoff),
            "kickoff_time_used": iso_or_blank(kickoff),
            "kickoff_source": kickoff_source,
            "provider_match_method": provider_method,
            "league": fixture.get("league"),
            "home_team": fixture.get("home_team"),
            "away_team": fixture.get("away_team"),
            "publish_class": str(fixture.get("publish_class") or fixture.get("fixture_class") or "").upper(),
            "market": (fixture.get("deploy_summary") or {}).get("market") or (fixture.get("signal_summary") or {}).get("market_family") or "",
            "pick": (fixture.get("deploy_summary") or {}).get("pick") or (fixture.get("signal_summary") or {}).get("deploy_pick") or "",
            "feed_generated_at": iso_or_blank(feed_generated_at),
            "feed_row_updated_at": iso_or_blank(feed_row_updated),
            "feed_snapshot_ts": iso_or_blank(feed_snapshot),
            "feed_snapshot_status": feed_status,
            "feed_pre_kickoff_eligible": fixture.get("pre_kickoff_eligible"),
            "feed_snapshot_phase": fixture.get("snapshot_phase") or "",
            "feed_minutes_after_kickoff": minutes_after(feed_snapshot, kickoff),
            "decision_explicit_ts": iso_or_blank(decision_explicit),
            "decision_file_mtime": iso_or_blank(file_mtime(decision_path)),
            "decision_snapshot_status": classify_snapshot(decision_explicit, kickoff, decision_explicit is not None),
            "decision_pre_kickoff_eligible": decision_payload.get("pre_kickoff_eligible"),
            "decision_snapshot_phase": decision_payload.get("snapshot_phase") or "",
            "lineup_explicit_ts": iso_or_blank(lineup_explicit),
            "lineup_file_mtime": iso_or_blank(file_mtime(lineup_path)),
            "lineup_snapshot_status": classify_snapshot(lineup_explicit, kickoff, lineup_explicit is not None),
            "lineup_pre_kickoff_eligible": lineup_payload.get("pre_kickoff_eligible"),
            "lineup_snapshot_phase": lineup_payload.get("snapshot_phase") or "",
            "lineup_coverage_status": lineup_payload.get("coverage_status") or lineup_payload.get("source_status") or "",
            "h2h_explicit_ts": iso_or_blank(h2h_explicit),
            "h2h_file_mtime": iso_or_blank(file_mtime(h2h_path)),
            "h2h_snapshot_status": classify_snapshot(h2h_explicit, kickoff, h2h_explicit is not None),
            "h2h_pre_kickoff_eligible": h2h_payload.get("pre_kickoff_eligible"),
            "h2h_snapshot_phase": h2h_payload.get("snapshot_phase") or "",
            "h2h_coverage_status": h2h_payload.get("coverage_status") or h2h_payload.get("fallback_mode") or "",
        }
        rows.append(row)

    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fixture_count": len(rows),
        "feed_generated_at": iso_or_blank(feed_generated_at),
        "feed_snapshot_status_counts": dict(Counter(row["feed_snapshot_status"] for row in rows)),
        "decision_snapshot_status_counts": dict(Counter(row["decision_snapshot_status"] for row in rows)),
        "lineup_snapshot_status_counts": dict(Counter(row["lineup_snapshot_status"] for row in rows)),
        "h2h_snapshot_status_counts": dict(Counter(row["h2h_snapshot_status"] for row in rows)),
        "by_publish_class_feed_status": {
            publish_class: dict(Counter(row["feed_snapshot_status"] for row in rows if row["publish_class"] == publish_class))
            for publish_class in sorted({str(row["publish_class"]) for row in rows})
        },
        "provider_kickoff_matched": sum(1 for row in rows if row["kickoff_source"] == "provider"),
        "site_feed_kickoff_used": sum(1 for row in rows if row["kickoff_source"] == "site_feed"),
        "notes": [
            "The fixture feed and downstream publish-safe payloads now use capture_generated_at/source_data_cutoff_at/fixture_kickoff_at/pre_kickoff_eligible/snapshot_phase when present.",
            "Backfilled payloads are auditable as post-kickoff/backfill snapshots, not proof of pre-kickoff availability.",
            "File mtimes are included only as forensic hints. They are not a publish-safe pre-kickoff timestamp contract.",
        ],
    }
    return rows, summary


def summary_markdown(summary: dict[str, Any]) -> str:
    return (
        "# Pre-Kickoff Snapshot Audit\n\n"
        f"Generated: {summary['generated_at']}\n\n"
        f"Fixture rows: {summary['fixture_count']}\n\n"
        f"Fixture feed generated at: `{summary['feed_generated_at']}`\n\n"
        "## Feed Snapshot Status\n\n"
        "```json\n"
        + json.dumps(summary["feed_snapshot_status_counts"], indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Publish Class Split\n\n"
        "```json\n"
        + json.dumps(summary["by_publish_class_feed_status"], indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Payload Timestamp Auditability\n\n"
        "```json\n"
        + json.dumps(
            {
                "decision": summary["decision_snapshot_status_counts"],
                "lineup": summary["lineup_snapshot_status_counts"],
                "h2h": summary["h2h_snapshot_status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n\n"
        "## Interpretation\n\n"
        "Payloads with `pre_kickoff_eligible=true` can be treated as genuine pre-kickoff evidence. "
        "Payloads marked `backfill` or `post_kickoff` remain useful for discovery and UI validation, but not for proving a signal existed before kickoff.\n\n"
        "Required contract: every fixture-level publish-safe payload should carry `capture_generated_at`, `source_data_cutoff_at`, `fixture_kickoff_at`, `pre_kickoff_eligible`, and `snapshot_phase`.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether website intelligence snapshots existed before kickoff.")
    parser.add_argument("--fixture-feed", type=Path, default=FIXTURE_FEED_PATH)
    parser.add_argument("--provider-results", type=Path, default=DEFAULT_PROVIDER_RESULTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = audit_rows(args)
    fields = [
        "fixture_key",
        "kickoff_time_site",
        "kickoff_time_provider",
        "kickoff_time_used",
        "kickoff_source",
        "provider_match_method",
        "league",
        "home_team",
        "away_team",
        "publish_class",
        "market",
        "pick",
        "feed_generated_at",
        "feed_row_updated_at",
        "feed_snapshot_ts",
        "feed_snapshot_status",
        "feed_pre_kickoff_eligible",
        "feed_snapshot_phase",
        "feed_minutes_after_kickoff",
        "decision_explicit_ts",
        "decision_file_mtime",
        "decision_snapshot_status",
        "decision_pre_kickoff_eligible",
        "decision_snapshot_phase",
        "lineup_explicit_ts",
        "lineup_file_mtime",
        "lineup_snapshot_status",
        "lineup_pre_kickoff_eligible",
        "lineup_snapshot_phase",
        "lineup_coverage_status",
        "h2h_explicit_ts",
        "h2h_file_mtime",
        "h2h_snapshot_status",
        "h2h_pre_kickoff_eligible",
        "h2h_snapshot_phase",
        "h2h_coverage_status",
    ]
    write_csv(args.outdir / "pre_kickoff_snapshot_audit_rows.csv", rows, fields)
    write_json(args.outdir / "summary.json", summary)
    (args.outdir / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(f"Fixture rows: {summary['fixture_count']}")
    print(f"Feed status: {summary['feed_snapshot_status_counts']}")
    print(f"Decision status: {summary['decision_snapshot_status_counts']}")
    print(f"Outputs: {args.outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
