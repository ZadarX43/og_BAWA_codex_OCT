#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_DIR = ROOT / "Matches" / "__merged__"

MARKET_ODDS = {
    "ftr": ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"],
    "ou25": ["odds_ft_over25", "odds_ft_under25"],
    "btts": ["odds_btts_yes", "odds_btts_no"],
}

SUMMARY_FIELDS = [
    "league",
    "source_tag",
    "status",
    "severity",
    "upstream_fixtures",
    "allmarkets_fixtures_any_market",
    "source_full_1x2",
    "source_missing_1x2",
    "allmarkets_ftr_fixtures",
    "ftr_source_eligible_emitted",
    "ftr_source_eligible_missing",
    "ftr_emitted_without_source_1x2",
    "source_paired_ou25",
    "source_missing_paired_ou25",
    "allmarkets_ou25_fixtures",
    "ou25_source_eligible_emitted",
    "ou25_source_eligible_missing",
    "ou25_emitted_without_source_pair",
    "source_paired_btts",
    "source_missing_paired_btts",
    "allmarkets_btts_fixtures",
    "btts_source_eligible_emitted",
    "btts_source_eligible_missing",
    "btts_emitted_without_source_pair",
    "notes",
]

DETAIL_FIELDS = [
    "league",
    "source_tag",
    "fixture_key",
    "match_date",
    "home_team_name",
    "away_team_name",
    "issue",
    "market",
    "odds_status",
    "allmarkets_status",
]


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def infer_window_from_filename(path: Path) -> tuple[date, date] | None:
    match = re.search(r"ALLMARKETS_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        return None
    return date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2))


def present_positive(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = float(text)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed > 1.0


def status_is_open(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text not in {"complete", "completed", "finished", "ft"}


def row_date(row: dict[str, str]) -> date | None:
    fixture_key = str(row.get("fixture_key", "") or "")
    match = re.match(r"(\d{4})_(\d{2})_(\d{2})_", fixture_key)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    timestamp = str(row.get("timestamp", "") or "").strip()
    if timestamp:
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date()
        except (OSError, OverflowError, ValueError):
            return None
    return None


def fixture_identity(row: dict[str, str]) -> dict[str, str]:
    return {
        "fixture_key": str(row.get("fixture_key", "") or "").strip(),
        "match_date": str(row_date(row) or ""),
        "home_team_name": str(row.get("home_team_name", "") or "").strip(),
        "away_team_name": str(row.get("away_team_name", "") or "").strip(),
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_allmarkets(path: Path) -> dict[str, dict[str, set[str]]]:
    by_league: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in load_csv_rows(path):
        league = str(row.get("league", "") or "").strip()
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        market = str(row.get("market", "") or "").strip().lower()
        if not league or not fixture_key:
            continue
        by_league[league]["any"].add(fixture_key)
        if market:
            by_league[league][market].add(fixture_key)
    return by_league


def load_upstream_fixtures(merged_dir: Path, date_from: date, date_to: date) -> dict[str, dict[str, Any]]:
    by_league: dict[str, dict[str, Any]] = {}
    for path in sorted(merged_dir.glob("*__merged.csv")):
        source_tag = path.name.replace("__merged.csv", "")
        league = source_tag.replace("_", " ")
        fixtures: dict[str, dict[str, str]] = {}
        market_ready: dict[str, set[str]] = {market: set() for market in MARKET_ODDS}

        for row in load_csv_rows(path):
            current_date = row_date(row)
            fixture_key = str(row.get("fixture_key", "") or "").strip()
            if not current_date or not fixture_key:
                continue
            if current_date < date_from or current_date > date_to:
                continue
            if not status_is_open(row.get("status", "")):
                continue

            fixtures[fixture_key] = fixture_identity(row)
            for market, cols in MARKET_ODDS.items():
                if all(present_positive(row.get(col)) for col in cols):
                    market_ready[market].add(fixture_key)

        if fixtures:
            by_league[league] = {
                "source_tag": source_tag,
                "fixtures": fixtures,
                "market_ready": market_ready,
            }
    return by_league


def build_reports(
    upstream: dict[str, dict[str, Any]],
    allmarkets: dict[str, dict[str, set[str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    leagues = sorted(set(upstream) | set(allmarkets))

    for league in leagues:
        source = upstream.get(league, {"source_tag": "", "fixtures": {}, "market_ready": {m: set() for m in MARKET_ODDS}})
        fixtures: dict[str, dict[str, str]] = source["fixtures"]
        ready: dict[str, set[str]] = source["market_ready"]
        emitted = allmarkets.get(league, defaultdict(set))

        fixture_keys = set(fixtures)
        emitted_any = set(emitted.get("any", set()))
        notes: list[str] = []
        severity = "OK"

        if fixture_keys and not emitted_any:
            severity = "CRITICAL"
            notes.append("No ALLMARKETS rows emitted despite upstream current-window fixtures.")
        elif emitted_any and not fixture_keys:
            severity = "WARN"
            notes.append("ALLMARKETS emitted rows but no matching upstream current-window fixtures were found.")

        metrics: dict[str, Any] = {
            "league": league,
            "source_tag": source.get("source_tag", ""),
            "upstream_fixtures": len(fixture_keys),
            "allmarkets_fixtures_any_market": len(emitted_any),
        }

        for market in ("ftr", "ou25", "btts"):
            eligible = set(ready.get(market, set()))
            emitted_market = set(emitted.get(market, set()))
            missing_odds = fixture_keys - eligible
            missing_emission = eligible - emitted_market
            emitted_without_source_pair = emitted_market - eligible

            if missing_odds:
                if market == "ftr" and not emitted_market:
                    severity = "CRITICAL"
                elif severity == "OK":
                    severity = "WARN"
                notes.append(f"{market.upper()} source odds missing for {len(missing_odds)} fixture(s).")
            if missing_emission:
                if severity == "OK":
                    severity = "WARN"
                notes.append(f"{market.upper()} missing ALLMARKETS emission for {len(missing_emission)} odds-ready fixture(s).")
            if emitted_without_source_pair:
                if severity == "OK":
                    severity = "WARN"
                notes.append(f"{market.upper()} emitted {len(emitted_without_source_pair)} fixture(s) without source paired odds; likely rescue/backfill.")

            if market == "ftr":
                metrics.update(
                    {
                        "source_full_1x2": len(eligible),
                        "source_missing_1x2": len(missing_odds),
                        "allmarkets_ftr_fixtures": len(emitted_market),
                        "ftr_source_eligible_emitted": len(eligible & emitted_market),
                        "ftr_source_eligible_missing": len(missing_emission),
                        "ftr_emitted_without_source_1x2": len(emitted_without_source_pair),
                    }
                )
            else:
                metrics.update(
                    {
                        f"source_paired_{market}": len(eligible),
                        f"source_missing_paired_{market}": len(missing_odds),
                        f"allmarkets_{market}_fixtures": len(emitted_market),
                        f"{market}_source_eligible_emitted": len(eligible & emitted_market),
                        f"{market}_source_eligible_missing": len(missing_emission),
                        f"{market}_emitted_without_source_pair": len(emitted_without_source_pair),
                    }
                )

            for key in sorted(missing_odds):
                detail_rows.append(
                    {
                        **{"league": league, "source_tag": source.get("source_tag", "")},
                        **fixtures.get(key, {"fixture_key": key, "match_date": "", "home_team_name": "", "away_team_name": ""}),
                        "issue": "SOURCE_ODDS_MISSING",
                        "market": market.upper(),
                        "odds_status": "MISSING_SOURCE_PAIR",
                        "allmarkets_status": "EMITTED" if key in emitted_market else "NOT_EMITTED",
                    }
                )
            for key in sorted(missing_emission):
                detail_rows.append(
                    {
                        **{"league": league, "source_tag": source.get("source_tag", "")},
                        **fixtures.get(key, {"fixture_key": key, "match_date": "", "home_team_name": "", "away_team_name": ""}),
                        "issue": "ALLMARKETS_EMISSION_MISSING",
                        "market": market.upper(),
                        "odds_status": "SOURCE_PAIR_OK",
                        "allmarkets_status": "NOT_EMITTED",
                    }
                )

        metrics["status"] = "PASS" if severity == "OK" else "REVIEW"
        metrics["severity"] = severity
        metrics["notes"] = " | ".join(dict.fromkeys(notes))
        summary_rows.append({field: metrics.get(field, "") for field in SUMMARY_FIELDS})

    severity_rank = {"CRITICAL": 0, "WARN": 1, "OK": 2}
    summary_rows.sort(key=lambda row: (severity_rank.get(str(row["severity"]), 9), str(row["league"])))
    return summary_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "/") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(
    path: Path,
    allmarkets_csv: Path,
    date_from: date,
    date_to: date,
    summary_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    csv_path: Path,
    detail_path: Path,
) -> None:
    critical = [row for row in summary_rows if row["severity"] == "CRITICAL"]
    warn = [row for row in summary_rows if row["severity"] == "WARN"]
    ok = [row for row in summary_rows if row["severity"] == "OK"]
    focus_fields = [
        "league",
        "severity",
        "upstream_fixtures",
        "allmarkets_fixtures_any_market",
        "source_full_1x2",
        "source_missing_1x2",
        "allmarkets_ftr_fixtures",
        "source_paired_ou25",
        "source_paired_btts",
        "notes",
    ]
    lines = [
        "# PRE_ALLMARKETS_FIXTURE_LOSS_REPORT",
        "",
        f"Generated: `{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}`",
        f"Window: `{date_from}` to `{date_to}` inclusive",
        f"ALLMARKETS source: `{allmarkets_csv}`",
        "",
        "## Prompt Summary",
        f"- CRITICAL leagues: `{len(critical)}`",
        f"- WARN leagues: `{len(warn)}`",
        f"- OK leagues: `{len(ok)}`",
        f"- Detail issue rows: `{len(detail_rows)}`",
        "",
    ]
    if critical:
        lines.extend(["## Coverage Prompts", ""])
        for row in critical:
            lines.append(f"- CRITICAL: `{row['league']}` - {row['notes']}")
        lines.append("")
    if warn:
        if not critical:
            lines.extend(["## Coverage Prompts", ""])
        for row in warn:
            lines.append(f"- WARN: `{row['league']}` - {row['notes']}")
        lines.append("")

    lines.extend(
        [
            "## League Summary",
            "",
            *markdown_table(summary_rows, focus_fields),
            "",
            "## Outputs",
            f"- Summary CSV: `{csv_path}`",
            f"- Detail CSV: `{detail_path}`",
            "",
            "## How To Read",
            "- `source_missing_1x2` means a current-window fixture exists upstream but lacks complete home/draw/away odds.",
            "- `ftr_source_eligible_missing` means a fixture had complete source 1X2 odds but no FTR row was emitted by ALLMARKETS.",
            "- `*_emitted_without_source_pair` usually means snapshot/backfill/proxy rescue supplied market odds after the raw merged row was blank.",
            "- This report is QA only. It does not promote, demote, or route any pick.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current-window merged fixtures against emitted ALLMARKETS rows by league/market."
    )
    parser.add_argument("--allmarkets-csv", required=True, help="BOOKIE_*_ALLMARKETS CSV to audit.")
    parser.add_argument("--merged-dir", default=str(DEFAULT_MERGED_DIR), help="Canonical merged fixture directory.")
    parser.add_argument("--date-from", type=parse_iso_date, default=None, help="Window start, YYYY-MM-DD.")
    parser.add_argument("--date-to", type=parse_iso_date, default=None, help="Window end, YYYY-MM-DD inclusive.")
    parser.add_argument("--outdir", default="", help="Output directory. Defaults to the ALLMARKETS CSV directory.")
    parser.add_argument("--fail-on-critical", action="store_true", help="Exit 2 when CRITICAL coverage gaps exist.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Exit 1 when WARN or CRITICAL coverage gaps exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allmarkets_csv = Path(args.allmarkets_csv).resolve()
    if not allmarkets_csv.exists():
        raise FileNotFoundError(f"ALLMARKETS CSV not found: {allmarkets_csv}")

    inferred = infer_window_from_filename(allmarkets_csv)
    date_from = args.date_from or (inferred[0] if inferred else None)
    date_to = args.date_to or (inferred[1] if inferred else None)
    if date_from is None or date_to is None:
        raise SystemExit("Provide --date-from and --date-to, or use an ALLMARKETS filename containing _YYYY-MM-DD_to_YYYY-MM-DD.")

    merged_dir = Path(args.merged_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else allmarkets_csv.parent
    slug = f"{date_from}_to_{date_to}"
    summary_path = outdir / f"PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_{slug}.csv"
    detail_path = outdir / f"PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_{slug}.csv"
    markdown_path = outdir / f"PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_{slug}.md"
    json_path = outdir / f"PRE_ALLMARKETS_FIXTURE_LOSS_SUMMARY_{slug}.json"

    upstream = load_upstream_fixtures(merged_dir, date_from, date_to)
    allmarkets = load_allmarkets(allmarkets_csv)
    summary_rows, detail_rows = build_reports(upstream, allmarkets)

    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(detail_path, detail_rows, DETAIL_FIELDS)
    write_markdown(markdown_path, allmarkets_csv, date_from, date_to, summary_rows, detail_rows, summary_path, detail_path)

    severity_counts = defaultdict(int)
    for row in summary_rows:
        severity_counts[str(row["severity"])] += 1
    json_payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "allmarkets_csv": str(allmarkets_csv),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "severity_counts": dict(sorted(severity_counts.items())),
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
        "markdown_report": str(markdown_path),
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    critical = [row for row in summary_rows if row["severity"] == "CRITICAL"]
    warn = [row for row in summary_rows if row["severity"] == "WARN"]
    print(f"PRE_ALLMARKETS_FIXTURE_LOSS_REPORT written: {summary_path.relative_to(ROOT)}")
    print(f"Detail report written: {detail_path.relative_to(ROOT)}")
    print(f"Markdown report written: {markdown_path.relative_to(ROOT)}")
    print(f"Severity counts: {dict(sorted(severity_counts.items()))}")
    for row in critical:
        print(f"CRITICAL|{row['league']}|{row['notes']}")
    for row in warn[:20]:
        print(f"WARN|{row['league']}|{row['notes']}")
    if len(warn) > 20:
        print(f"WARN|truncated|{len(warn) - 20} additional warning league(s); see markdown report.")

    if critical and args.fail_on_critical:
        return 2
    if (critical or warn) and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
