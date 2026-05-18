#!/usr/bin/env python3
"""Audit route authority against fixture decision context.

This does not change deploy routing. It checks whether the website decision
payloads distinguish published route state from contextual audit caution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "frontend" / "public" / "data"
DEFAULT_REPORT = ROOT / "reports" / "latest" / "FIXTURE_DECISION_ROUTE_CONFLICT_AUDIT.md"
DEFAULT_JSON_REPORT = ROOT / "reports" / "latest" / "fixture_decision_route_conflict_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_pick(value: object) -> str:
    pick = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        "OVER_25": "OVER25",
        "OVER2.5": "OVER25",
        "OVER_2.5": "OVER25",
        "UNDER_25": "UNDER25",
        "UNDER2.5": "UNDER25",
        "UNDER_2.5": "UNDER25",
        "BTTS_YES": "YES",
        "BTTS_NO": "NO",
        "HOME_WIN": "HOME",
        "AWAY_WIN": "AWAY",
    }
    return aliases.get(pick, pick)


def market_key_for_family(family: object) -> str:
    family_text = str(family or "").upper()
    if family_text == "FTR":
        return "ftr"
    if family_text == "BTTS":
        return "btts"
    if family_text == "OU25":
        return "ou25"
    if family_text in {"TG15", "TEAM_GOALS", "TEAMGOALS"}:
        return "team_goals"
    return family_text.lower()


def fixture_route(fixture: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    signal = fixture.get("signal_summary") or {}
    deploy = fixture.get("deploy_summary") or {}
    return {
        "route_state": str(decision.get("route_state") or fixture.get("publish_class") or signal.get("signal_state") or "").upper(),
        "route_market": str(decision.get("route_market") or signal.get("market_family") or deploy.get("market") or "").upper(),
        "route_pick": normalize_pick(decision.get("route_pick") or signal.get("deploy_pick") or deploy.get("pick")),
        "route_active": bool(decision.get("route_active")) or (
            str(fixture.get("publish_class") or "").upper() == "DEPLOY"
            and bool(signal.get("deploy_pick") or deploy.get("pick"))
        ),
        "route_bookie_od": decision.get("route_bookie_od") or deploy.get("bookie_od"),
    }


def premium_index(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_json(data_root / "premium_predictions.json", [])
    if isinstance(rows, dict):
        rows = rows.get("predictions") or rows.get("rows") or []
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fixture_key = str(row.get("fixture_key") or "").strip()
        if fixture_key:
            out.setdefault(fixture_key, []).append(row)
    return out


def matching_premium_route(rows: list[dict[str, Any]], route_market: str, route_pick: str) -> bool:
    for row in rows:
        if str(row.get("market") or "").upper() == route_market and normalize_pick(row.get("pick")) == route_pick:
            return True
    return False


def numeric_odds(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1.0 else None


def audit_fixture(
    fixture: dict[str, Any],
    decision: dict[str, Any],
    premium_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    fixture_key = fixture.get("fixture_key")
    route = fixture_route(fixture, decision)
    audit_state = str(decision.get("audit_state") or decision.get("signal_state") or "").upper()
    conflict_level = str(decision.get("conflict_level") or "").upper()
    market_key = market_key_for_family(route["route_market"])
    market_intel = (decision.get("market_intelligence") or {}).get(market_key) or {}
    model_lean = normalize_pick(market_intel.get("model_lean"))
    primary_signal = str(decision.get("primary_signal") or "").strip()
    base = {
        "fixture_key": fixture_key,
        "fixture": decision.get("fixture") or f"{fixture.get('home_team')} vs {fixture.get('away_team')}",
        "route_state": route["route_state"],
        "route_market": route["route_market"],
        "route_pick": route["route_pick"],
        "audit_state": audit_state,
        "agreement_score": decision.get("audit_agreement_score") or decision.get("agreement_score"),
        "conflict_level": conflict_level,
        "model_lean": model_lean,
    }

    if route["route_state"] == "DEPLOY":
        if not route["route_market"] or not route["route_pick"]:
            issues.append({**base, "issue": "DEPLOY_WITHOUT_EXPLICIT_ROUTE", "severity": "CRITICAL"})
        if audit_state in {"AVOID", "FRAGILE"}:
            issues.append({**base, "issue": "DEPLOY_WITH_CAUTION_AUDIT", "severity": "WARN"})
        if conflict_level == "HARD_CONFLICT":
            issues.append({**base, "issue": "HARD_CONFLICT_FLAGGED", "severity": "CRITICAL"})
        if model_lean and route["route_pick"] and model_lean != route["route_pick"]:
            issues.append({**base, "issue": "ROUTE_PICK_DIFFERS_FROM_ROUTE_MARKET_LEAN", "severity": "CRITICAL"})
        if route["route_market"] and route["route_pick"] and not matching_premium_route(premium_rows, route["route_market"], route["route_pick"]):
            issues.append({**base, "issue": "DEPLOY_ROUTE_MISSING_PREMIUM_ROW", "severity": "WARN"})
        if numeric_odds(route["route_bookie_od"]) is None:
            issues.append({**base, "issue": "DEPLOY_ROUTE_MISSING_ODDS", "severity": "WARN"})
    else:
        if primary_signal and primary_signal != "No published pick":
            issues.append({**base, "issue": "NON_DEPLOY_PRIMARY_SIGNAL_LOOKS_LIKE_PICK", "severity": "WARN"})

    return issues


def write_markdown(path: Path, rows: list[dict[str, Any]], total_fixtures: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["issue"] for row in rows)
    severities = Counter(row["severity"] for row in rows)
    lines = [
        "# Fixture Decision Route Conflict Audit",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Summary",
        "",
        f"- Fixtures audited: `{total_fixtures}`",
        f"- Issues found: `{len(rows)}`",
        f"- Critical: `{severities.get('CRITICAL', 0)}`",
        f"- Warnings: `{severities.get('WARN', 0)}`",
        "",
        "## Issue Counts",
        "",
    ]
    if counts:
        for issue, count in sorted(counts.items()):
            lines.append(f"- `{issue}`: `{count}`")
    else:
        lines.append("- No route/audit contract issues found.")

    lines.extend(["", "## Details", ""])
    if rows:
        lines.append("| Severity | Issue | Fixture | Route | Audit | Agreement | Model lean |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows[:200]:
            route = f"{row['route_state']} {row['route_market']} {row['route_pick']}".strip()
            audit = f"{row['audit_state']} / {row['conflict_level']}".strip()
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["severity"]),
                        f"`{row['issue']}`",
                        str(row["fixture"]).replace("|", "/"),
                        route,
                        audit,
                        str(row.get("agreement_score") or ""),
                        str(row.get("model_lean") or ""),
                    ]
                )
                + " |"
            )
        if len(rows) > 200:
            lines.append("")
            lines.append(f"_Truncated detail table to first 200 of {len(rows)} issues._")
    else:
        lines.append("No issues found.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixture route/audit decision conflicts.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON_REPORT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    fixtures = list((load_json(data_root / "fixture_intelligence_public.json", {}) or {}).get("fixtures") or [])
    premium = premium_index(data_root)
    decision_root = data_root / "fixture_decision_intelligence"

    issues: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_key = str(fixture.get("fixture_key") or "")
        decision = load_json(decision_root / f"{fixture_key}.json", {}) or {}
        issues.extend(audit_fixture(fixture, decision, premium.get(fixture_key, [])))

    report_path = Path(args.report)
    json_report_path = Path(args.json_report)
    write_markdown(report_path, issues, len(fixtures))
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "fixtures_audited": len(fixtures),
                "issue_count": len(issues),
                "issues_by_type": dict(Counter(row["issue"] for row in issues)),
                "issues_by_severity": dict(Counter(row["severity"] for row in issues)),
                "issues": issues,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": not any(row["severity"] == "CRITICAL" for row in issues),
                "fixtures_audited": len(fixtures),
                "issue_count": len(issues),
                "critical": sum(1 for row in issues if row["severity"] == "CRITICAL"),
                "warning": sum(1 for row in issues if row["severity"] == "WARN"),
                "report": str(report_path.relative_to(ROOT)),
                "json_report": str(json_report_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 1 if any(row["severity"] == "CRITICAL" for row in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
