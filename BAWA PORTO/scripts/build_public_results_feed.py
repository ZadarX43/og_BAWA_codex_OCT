#!/usr/bin/env python3
"""Build public-safe results feed JSON for the website.

This is a publishing adapter only. It reads already-scored audit outputs and
emits a compact public feed; it does not change model, deploy, or grading logic.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEEKEND_DIR = ROOT / "reports/latest/weekend_prediction_intelligence_scoring"
MLS_DIR = ROOT / "reports/latest/live_mls_night_audit_2026_05_14"
OUT_PATH = ROOT / "frontend/public/data/live_results_feed.json"
DOC_PATH = ROOT / "docs/PUBLIC_RESULTS_FEED_BUILD_NOTES_2026-05-14.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def is_hit(row: dict[str, Any]) -> bool:
    if str(row.get("hit", "")).strip() == "1":
        return True
    return str(row.get("result_status", "")).strip().lower() == "won"


def result_status(row: dict[str, Any]) -> str:
    status = str(row.get("result_status", "")).strip().lower()
    if status in {"won", "lost", "void", "cashed", "pending"}:
        return status
    if str(row.get("hit", "")).strip() == "1":
        return "won"
    if str(row.get("hit", "")).strip() == "0":
        return "lost"
    return "pending"


def hit_rate(wins: int, settled: int) -> float | None:
    return None if settled <= 0 else round(wins / settled, 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled_rows = [row for row in rows if result_status(row) in {"won", "lost", "void", "cashed"}]
    wins = sum(1 for row in settled_rows if result_status(row) in {"won", "cashed"})
    losses = sum(1 for row in settled_rows if result_status(row) == "lost")
    voids = sum(1 for row in settled_rows if result_status(row) == "void")
    return {
        "rows": len(rows),
        "settled": len(settled_rows),
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "hit_rate": hit_rate(wins, len(settled_rows)),
    }


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get(key) or "UNKNOWN").strip() or "UNKNOWN"
        groups[label].append(row)
    output = []
    for label, group_rows in sorted(groups.items()):
        item = summarize(group_rows)
        item[key] = label
        output.append(item)
    return output


def public_item(row: dict[str, Any], *, source_window: str, source_type: str) -> dict[str, Any]:
    score = str(row.get("score") or "").strip()
    if not score and row.get("home_score") not in (None, "") and row.get("away_score") not in (None, ""):
        score = f"{row.get('home_score')}-{row.get('away_score')}"
    return {
        "source_window": source_window,
        "source_type": source_type,
        "fixture_key": row.get("fixture_key", ""),
        "kickoff_time": row.get("kickoff_time") or row.get("kickoff_ts_utc") or "",
        "league": row.get("league") or "USA MLS",
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "score": score,
        "tier": row.get("tier") or row.get("deploy_tier") or row.get("primary_tier") or "",
        "publish_class": row.get("publish_class") or ("DEPLOY" if row.get("deploy_tier") in {"ELITE", "STANDARD"} else "OBSERVE"),
        "market": row.get("market", ""),
        "pick": row.get("pick") or row.get("model_pick") or "",
        "actual": row.get("actual", ""),
        "result_status": result_status(row),
        "bookie_od": as_float(row.get("bookie_od")),
        "model_prob": as_float(row.get("model_prob") or row.get("model_p_for_bookie") or row.get("model_prob")),
        "value_edge": as_float(row.get("value_edge")),
        "profit_units": as_float(row.get("profit_units")),
        "site_signal_alignment": row.get("site_signal_alignment", ""),
        "site_signal_pick": row.get("site_signal_pick", ""),
        "site_signal_state": row.get("site_signal_state", ""),
        "site_signal_score": as_float(row.get("site_signal_score")),
        "decision_state": row.get("decision_signal_state") or row.get("state") or "",
        "decision_primary_signal": row.get("decision_primary_signal", ""),
        "agreement_score": as_float(row.get("agreement_score") or row.get("alignment_score")),
        "snapshot_phase": row.get("snapshot_phase", ""),
        "pre_kickoff_eligible": row.get("pre_kickoff_eligible", ""),
    }


def build_weekend_window() -> dict[str, Any]:
    summary = read_json(WEEKEND_DIR / "summary.json")
    raw_rows = read_csv(WEEKEND_DIR / "primary_prediction_score_rows.csv")
    rows = [
        row
        for row in raw_rows
        if str(row.get("publish_class", "")).upper() in {"DEPLOY", "OBSERVE"}
        and str(row.get("result_status", "")).lower() in {"won", "lost"}
    ]
    deploy_rows = [row for row in rows if str(row.get("publish_class", "")).upper() == "DEPLOY"]
    observe_rows = [row for row in rows if str(row.get("publish_class", "")).upper() == "OBSERVE"]
    ev_rows = [row for row in deploy_rows if str(row.get("ev_positive", "")).lower() == "true"]
    items = [public_item(row, source_window="weekend_2026_05_09_11", source_type="primary_prediction") for row in rows]
    return {
        "window_id": "weekend_2026_05_09_11",
        "title": "Weekend prediction audit",
        "subtitle": "Settled deploy and observe rows from the 2026-05-09 to 2026-05-11 board.",
        "period_start": summary.get("period_start", "2026-05-09"),
        "period_end": summary.get("period_end", "2026-05-11"),
        "generated_at": summary.get("generated_at", ""),
        "proof_level": "settled_provider_results",
        "summary": {
            "all": summarize(rows),
            "deploy": summarize(deploy_rows),
            "observe": summarize(observe_rows),
            "ev_positive": summarize(ev_rows),
        },
        "by_market": group_summary(deploy_rows, "market"),
        "by_tier": group_summary(deploy_rows, "tier"),
        "featured_results": sorted(items, key=lambda item: (item["result_status"] != "won", item["kickoff_time"]))[:12],
        "items": items,
        "notes": [
            "DEPLOY rows are public proof for live action tiers.",
            "OBSERVE rows are watchlist/research context and are not counted as deployable picks.",
        ],
    }


def build_mls_window() -> dict[str, Any]:
    summary = read_json(MLS_DIR / "summary.json")
    raw_rows = read_csv(MLS_DIR / "model_intelligence_scored_rows.csv")
    rows = [row for row in raw_rows if str(row.get("deploy_tier", "")).upper() in {"ELITE", "STANDARD", "OBSERVE"}]
    deploy_rows = [row for row in rows if str(row.get("deploy_tier", "")).upper() in {"ELITE", "STANDARD"}]
    observe_rows = [row for row in rows if str(row.get("deploy_tier", "")).upper() == "OBSERVE"]
    items = [public_item(row, source_window="mls_live_2026_05_14", source_type="live_mls_model_intelligence") for row in rows]
    return {
        "window_id": "mls_live_2026_05_14",
        "title": "MLS live-system test night",
        "subtitle": "Live MLS board with model picks scored against final provider results.",
        "period_start": "2026-05-13",
        "period_end": "2026-05-14",
        "generated_at": utc_now(),
        "proof_level": "live_provider_final_audit",
        "summary": {
            "all": summarize(rows),
            "deploy": summarize(deploy_rows),
            "observe": summarize(observe_rows),
            "player_event_beta": {
                "rows": summary.get("player_event_review_rows"),
                "wins": summary.get("player_event_hits"),
                "hit_rate": hit_rate(int(summary.get("player_event_hits") or 0), int(summary.get("player_event_review_rows") or 0)),
                "research_only": True,
            },
        },
        "by_market": group_summary(deploy_rows, "market"),
        "by_tier": group_summary(deploy_rows, "deploy_tier"),
        "featured_results": [
            item
            for item in items
            if item["publish_class"] == "DEPLOY"
            or item["site_signal_alignment"] in {"conflicts_model", "supports_model"}
        ][:16],
        "items": items,
        "notes": [
            "Player event rows are beta/manual-review only.",
            "The Real Salt Lake 3-0 Houston BTTS miss is retained publicly as a validated intelligence conflict warning.",
        ],
    }


def aggregate_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    deploy_rows = sum(int(window["summary"]["deploy"]["rows"] or 0) for window in windows)
    deploy_settled = sum(int(window["summary"]["deploy"]["settled"] or 0) for window in windows)
    deploy_wins = sum(int(window["summary"]["deploy"]["wins"] or 0) for window in windows)
    observe_rows = sum(int(window["summary"]["observe"]["rows"] or 0) for window in windows)
    observe_settled = sum(int(window["summary"]["observe"]["settled"] or 0) for window in windows)
    observe_wins = sum(int(window["summary"]["observe"]["wins"] or 0) for window in windows)
    return {
        "windows": len(windows),
        "deploy_rows": deploy_rows,
        "deploy_settled": deploy_settled,
        "deploy_wins": deploy_wins,
        "deploy_hit_rate": hit_rate(deploy_wins, deploy_settled),
        "observe_rows": observe_rows,
        "observe_settled": observe_settled,
        "observe_wins": observe_wins,
        "observe_hit_rate": hit_rate(observe_wins, observe_settled),
    }


def write_notes(feed: dict[str, Any]) -> None:
    lines = [
        "# Public Results Feed Build Notes",
        "",
        f"Generated: {feed['generated_at']}",
        "",
        "## Outputs",
        "",
        f"- Website feed: `{OUT_PATH.relative_to(ROOT)}`",
        "",
        "## Windows",
        "",
    ]
    for window in feed["windows"]:
        deploy = window["summary"]["deploy"]
        observe = window["summary"]["observe"]
        lines.extend(
            [
                f"### {window['title']}",
                "",
                f"- Period: {window['period_start']} to {window['period_end']}",
                f"- Deploy: {deploy['wins']}/{deploy['settled']} ({deploy['hit_rate']})",
                f"- Observe: {observe['wins']}/{observe['settled']} ({observe['hit_rate']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Guardrails",
            "",
            "- This is a public publishing adapter only.",
            "- `OBSERVE` rows remain research/watchlist rows, not deployable picks.",
            "- Player-event rows remain beta/manual-review only.",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    windows = [build_mls_window(), build_weekend_window()]
    feed = {
        "generated_at": utc_now(),
        "contract_version": 1,
        "summary": aggregate_windows(windows),
        "windows": windows,
        "notes": [
            "Public proof feed generated from settled audit outputs.",
            "Deploy rows and observe rows are deliberately separated.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(feed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_notes(feed)
    print(json.dumps({"output": str(OUT_PATH), "windows": len(windows), **feed["summary"]}, indent=2))


if __name__ == "__main__":
    main()
