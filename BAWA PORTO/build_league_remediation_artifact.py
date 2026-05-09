#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "frontend" / "public" / "data"
REPORTS_DIR = ROOT / "reports" / "latest"

AUDIT_PATH = DATA_DIR / "league_coverage_audit.json"
OUTPUT_JSON_PATH = DATA_DIR / "league_remediation_plan.json"
OUTPUT_REPORT_PATH = REPORTS_DIR / "LEAGUE_REMEDIATION_PLAN_REPORT.md"

PROVIDER_ALIAS_ONLY = {
    "Jupiler Pro League",
    "Major League Soccer",
    "Primeira Liga",
    "Serie A",
}

PRIORITY_OVERRIDES = {
    "Portugal Liga": 1,
    "Germany Bundesliga 2": 2,
    "Turkey Super Lig": 3,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_market_gaps(market_output: dict[str, Any]) -> dict[str, bool]:
    return {
        "ftr": not bool(market_output.get("ftr_present")),
        "btts": not bool(market_output.get("btts_present")),
        "ou25": not bool(market_output.get("ou25_present")),
    }


def issue_type(record: dict[str, Any]) -> tuple[str, str]:
    classification = record["classification"]
    counts = record["fixture_counts"]
    overlay = record["overlay_support"]
    markets = record["market_output"]

    if classification == "blind_spot":
        if overlay["historical_overlay"] or any(
            overlay[key]
            for key in ("current_odds", "current_injuries", "current_lineups", "current_team_stats", "current_match_events")
        ):
            return "overlay_context_enhancement", "context_only"
        return "model_expansion", "hidden_for_now"

    if classification == "partial_coverage":
        if counts["routed_total"] == 0 and counts["context_total"] > 0:
            return "overlay_context_enhancement", "context_only"
        if counts["routed_total"] > 0 and counts["non_routed_total"] > 0:
            return "routing_expansion", "full_coverage"
        if markets["ftr_present"] and not (markets["btts_present"] or markets["ou25_present"]):
            return "routing_expansion", "full_coverage"
        if not overlay["historical_overlay"] and not any(
            overlay[key]
            for key in ("current_odds", "current_injuries", "current_lineups", "current_team_stats", "current_match_events")
        ):
            return "overlay_context_enhancement", "partial_coverage"
        return "routing_expansion", "full_coverage"

    return "none", record["classification"]


def recommended_actions(record: dict[str, Any], primary_issue: str) -> list[str]:
    league = record["league"]
    counts = record["fixture_counts"]
    markets = record["market_output"]
    overlay = record["overlay_support"]
    actions: list[str] = []

    if primary_issue == "model_expansion":
        actions.append("Audit whether FootyStats-era goal-market models exist for this league.")
        actions.append("Audit whether bookmaker coverage should be feeding ALLMARKETS for this league.")
        actions.append("If no usable model estate exists, keep the league hidden until a minimal observe-safe base exists.")
    elif primary_issue == "routing_expansion":
        if counts["non_routed_total"] > 0:
            actions.append("Audit pre-ALLMARKETS fixture loss for the non-routed tail.")
        if markets["ftr_present"] and not markets["btts_present"]:
            actions.append("Audit why BTTS rows are not surviving routed publish for this league.")
        if markets["ftr_present"] and not markets["ou25_present"]:
            actions.append("Audit why OU25 rows are not surviving routed publish for this league.")
        if not markets["ftr_present"] and (markets["btts_present"] or markets["ou25_present"]):
            actions.append("Audit why FTR rows are not surviving while secondary markets do.")
        actions.append("Prefer widening observe-safe routed coverage before forcing new deploys.")
    elif primary_issue == "overlay_context_enhancement":
        actions.append("Improve CONTEXT note richness for non-routed fixtures in this league.")
        if not overlay["current_odds"]:
            actions.append("Feed current-window prematch odds into the context lane.")
        if not overlay["current_injuries"]:
            actions.append("Add current-window injury overlay support for this league.")
        if not overlay["current_lineups"]:
            actions.append("Add current-window lineup overlay support for this league.")
        actions.append("Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.")

    if league == "Portugal Liga":
        actions.insert(0, "Use Portugal Liga as the first partial-coverage strengthening target.")
    if league == "Germany Bundesliga 2":
        actions.insert(0, "Decide explicitly whether Bundesliga 2 should recover to context-only or stay hidden.")
    if league == "Turkey Super Lig":
        actions.insert(0, "Decide whether Turkey Super Lig is worth active recovery or deliberate deprioritisation.")

    return actions


def priority_rank(record: dict[str, Any], primary_issue: str) -> int:
    league = record["league"]
    if league in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[league]

    counts = record["fixture_counts"]
    score = 100
    if record["classification"] == "blind_spot":
        score -= 20
    if primary_issue == "routing_expansion":
        score -= 15
    if record["overlay_support"]["historical_overlay"]:
        score -= 10
    if any(
        record["overlay_support"][key]
        for key in ("current_odds", "current_injuries", "current_lineups", "current_team_stats", "current_match_events")
    ):
        score -= 10
    score -= min(int(counts["covered_total"]), 15)
    return score


def build_payload() -> dict[str, Any]:
    audit = load_json(AUDIT_PATH)
    source_window = audit.get("source_window", {})
    generated_at = audit.get("generated_at")
    source_run_id = audit.get("source_run_id")

    remediation_records: list[dict[str, Any]] = []
    for record in audit.get("leagues", []):
        league = str(record.get("league", "") or "").strip()
        if not league or record.get("classification") == "full_coverage" or league in PROVIDER_ALIAS_ONLY:
            continue
        primary_issue, target_state = issue_type(record)
        remediation = {
            "league": league,
            "current_classification": record["classification"],
            "target_classification": target_state,
            "primary_issue": primary_issue,
            "secondary_issue": "overlay_context_enhancement" if primary_issue == "routing_expansion" and record["fixture_counts"]["non_routed_total"] > 0 else None,
            "market_gaps": infer_market_gaps(record["market_output"]),
            "fixture_counts": record["fixture_counts"],
            "overlay_support": record["overlay_support"],
            "recommended_actions": recommended_actions(record, primary_issue),
        }
        remediation["priority_rank"] = priority_rank(record, primary_issue)
        remediation_records.append(remediation)

    remediation_records.sort(key=lambda row: (row["priority_rank"], row["league"]))
    for idx, record in enumerate(remediation_records, start=1):
        record["priority_rank"] = idx

    payload = {
        "generated_at": generated_at,
        "source_run_id": source_run_id,
        "source_window": source_window,
        "coverage_summary": {
            "total_weak_leagues": len(remediation_records),
            "blind_spot_count": sum(1 for row in remediation_records if row["current_classification"] == "blind_spot"),
            "partial_coverage_count": sum(1 for row in remediation_records if row["current_classification"] == "partial_coverage"),
            "model_expansion_count": sum(1 for row in remediation_records if row["primary_issue"] == "model_expansion"),
            "routing_expansion_count": sum(1 for row in remediation_records if row["primary_issue"] == "routing_expansion"),
            "overlay_context_enhancement_count": sum(1 for row in remediation_records if row["primary_issue"] == "overlay_context_enhancement"),
        },
        "leagues": remediation_records,
    }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["coverage_summary"]
    lines = [
        "# LEAGUE_REMEDIATION_PLAN_REPORT",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Source run id: `{payload.get('source_run_id', '')}`",
        f"Source window: `{payload.get('source_window', {}).get('date_from', '')}` to `{payload.get('source_window', {}).get('date_to', '')}`",
        "",
        "## Summary",
        f"- Weak leagues: `{summary['total_weak_leagues']}`",
        f"- Blind spots: `{summary['blind_spot_count']}`",
        f"- Partial coverage: `{summary['partial_coverage_count']}`",
        f"- `model_expansion`: `{summary['model_expansion_count']}`",
        f"- `routing_expansion`: `{summary['routing_expansion_count']}`",
        f"- `overlay_context_enhancement`: `{summary['overlay_context_enhancement_count']}`",
        "",
        "## Prioritised Weak-League Actions",
    ]
    for record in payload["leagues"]:
        lines.extend(
            [
                f"### {record['priority_rank']}. {record['league']}",
                f"- Current: `{record['current_classification']}`",
                f"- Target: `{record['target_classification']}`",
                f"- Primary issue: `{record['primary_issue']}`",
                f"- Secondary issue: `{record['secondary_issue']}`" if record.get("secondary_issue") else "- Secondary issue: `none`",
                (
                    f"- Market gaps: `FTR={record['market_gaps']['ftr']}` | "
                    f"`BTTS={record['market_gaps']['btts']}` | `OU25={record['market_gaps']['ou25']}`"
                ),
            ]
        )
        for action in record["recommended_actions"]:
            lines.append(f"- Action: {action}")
        lines.append("")

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_report(payload)
    print(f"League remediation artifact written: {OUTPUT_JSON_PATH.relative_to(ROOT)}")
    print(f"League remediation report written: {OUTPUT_REPORT_PATH.relative_to(ROOT)}")
    print(f"Weak leagues: {payload['coverage_summary']['total_weak_leagues']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
