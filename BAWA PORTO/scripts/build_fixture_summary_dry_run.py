#!/usr/bin/env python3
"""Build deterministic local fixture-summary drafts from fixture brain inputs.

This is the pre-GPT contract smoke. It reads only `summary_inputs` from compact
fixture-brain payloads and writes local JSON/Markdown drafts for review.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAIN_DIR = ROOT / "build" / "site_brain" / "current"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "fixture_summary_dry_run"
TIERS = ("standard", "premium", "pro", "pro_plus")
MARKET_LABELS = {
    "ftr": "Full Time Result",
    "ou25": "Over 2.5 Match Goals",
    "btts": "BTTS",
    "team_goals": "Team Goals",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local summary drafts from fixture brain summary_inputs.")
    parser.add_argument("--fixture-brain-dir", type=Path, default=DEFAULT_BRAIN_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--tier", choices=TIERS, action="append", default=[])
    parser.add_argument("--fixture-key", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def slug(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "fixture"


def sentence_join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def fmt_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def human_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("headline", "profile", "main_caution"):
            if value.get(key):
                parts.append(str(value[key]))
        strengths = value.get("primary_strengths")
        if isinstance(strengths, list) and strengths:
            parts.append("Strengths: " + ", ".join(str(item) for item in strengths[:4]))
        return " ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:6])
    return str(value or "")


def market_line(market: dict[str, Any]) -> str:
    market_key = str(market.get("market_key") or "")
    label = MARKET_LABELS.get(market_key, market_key.upper())
    lean = market.get("model_lean") or market.get("team_context_lean") or "no clear lean"
    state = market.get("state") or market.get("band") or "context pending"
    support = fmt_value(market.get("support"))
    source = market.get("source_status") or "compiled context"
    support_text = f", support {support}" if support else ""
    return f"{label}: {lean} ({state}{support_text}, {source})."


def standard_summary(block: dict[str, Any]) -> str:
    fixture = block.get("fixture") or {}
    teams = f"{fixture.get('home_team', 'Home')} vs {fixture.get('away_team', 'Away')}"
    markets = block.get("markets") if isinstance(block.get("markets"), list) else []
    available = [market_line(item) for item in markets if item.get("status") == "available"][:4]
    freshness = block.get("freshness") or {}
    last_updated = freshness.get("last_updated") or "not stamped"
    return sentence_join(
        [
            f"{teams} has a compiled Standard market read.",
            " ".join(available) if available else "Market cards are still waiting for deploy-safe model context.",
            f"Last updated: {last_updated}.",
        ]
    )


def premium_summary(block: dict[str, Any]) -> str:
    base = standard_summary(block)
    h2h = block.get("h2h") or {}
    weather = block.get("weather") or {}
    team_context = block.get("team_context") or {}
    notes: list[str] = []
    if h2h.get("status") and h2h.get("status") != "missing":
        notes.append(f"H2H context is available with status {h2h.get('status')}.")
    else:
        notes.append("H2H context is missing or thin for this fixture.")
    if weather.get("status") and weather.get("status") != "missing":
        notes.append(f"Weather context status: {weather.get('status')}.")
    else:
        notes.append("Weather is currently a graceful fallback.")
    for side in ("home", "away"):
        team = team_context.get(side) or {}
        if team.get("status") == "available" and team.get("summary"):
            notes.append(f"{side.title()} team context: {human_text(team.get('summary'))}")
    return sentence_join([base, " ".join(notes)])


def pro_summary(block: dict[str, Any]) -> str:
    base = premium_summary(block)
    player_events = block.get("player_events") or {}
    injury = block.get("injury_context") or {}
    notes: list[str] = []
    if player_events.get("status") == "available":
        titles = [card.get("card_title") for card in player_events.get("cards", []) if card.get("card_title")]
        notes.append(f"Player-event beta cards available: {', '.join(titles[:6])}.")
        if player_events.get("lineup_status"):
            notes.append(f"Lineup phase: {player_events.get('lineup_status')}.")
    else:
        notes.append("Player-event beta cards are not available for this fixture yet.")
    if injury.get("market_adjustment"):
        notes.append(f"Injury shock market adjustment: {injury.get('market_adjustment')}.")
    key_players = injury.get("key_players") or []
    if key_players:
        notes.append(f"Key availability watch: {', '.join(str(player) for player in key_players[:6])}.")
    return sentence_join([base, " ".join(notes)])


def pro_plus_summary(block: dict[str, Any]) -> str:
    base = pro_summary(block)
    audit = block.get("audit") or {}
    coverage = audit.get("coverage") or {}
    missing = [key.replace("has_", "") for key, value in coverage.items() if not value]
    counts = audit.get("fixture_stats_counts") or {}
    audit_line = "Audit posture: "
    if missing:
        audit_line += "missing " + ", ".join(missing[:8]) + "."
    else:
        audit_line += "all compact coverage flags are present."
    if counts:
        audit_line += " Fixture stat counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()) + "."
    return sentence_join([base, audit_line])


def render_tier(block: dict[str, Any]) -> dict[str, Any]:
    tier = block.get("tier") or "standard"
    renderer = {
        "standard": standard_summary,
        "premium": premium_summary,
        "pro": pro_summary,
        "pro_plus": pro_plus_summary,
    }.get(tier, standard_summary)
    return {
        "tier": tier,
        "audience": block.get("audience") or "",
        "generator_mode": "deterministic_dry_run_no_gpt",
        "summary_text": renderer(block),
        "copy_rules": block.get("copy_rules") or [],
        "freshness": (block.get("freshness") or {}),
    }


def payload_paths(brain_dir: Path, fixture_keys: set[str], limit: int) -> list[Path]:
    manifest = read_json(brain_dir / "manifest.json", {})
    paths: list[Path] = []
    objects = manifest.get("objects") if isinstance(manifest, dict) else []
    if isinstance(objects, list):
        for item in objects:
            fixture_key = str(item.get("fixture_key") or "")
            if fixture_keys and fixture_key not in fixture_keys:
                continue
            rel = item.get("relative_path")
            if rel:
                paths.append(brain_dir / rel)
    if not paths:
        root = brain_dir / "payloads" / "fixtures"
        for path in sorted(root.glob("*.json")) if root.exists() else []:
            fixture_key = path.stem
            if fixture_keys and fixture_key not in fixture_keys:
                continue
            paths.append(path)
    return paths[:limit] if limit and limit > 0 else paths


def render_payload(path: Path, tiers: list[str]) -> dict[str, Any] | None:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return None
    summary_inputs = payload.get("summary_inputs") or {}
    tier_blocks = summary_inputs.get("tiers") if isinstance(summary_inputs, dict) else {}
    if not isinstance(tier_blocks, dict):
        return None
    selected = tiers or list(TIERS)
    drafts = {
        tier: render_tier(tier_blocks[tier])
        for tier in selected
        if isinstance(tier_blocks.get(tier), dict)
    }
    return {
        "schema": "fixture_summary_dry_run_v1",
        "generated_at": utc_now(),
        "fixture_key": payload.get("fixture_key") or path.stem,
        "source_payload": str(path),
        "source_contract": summary_inputs.get("schema") or "",
        "drafts": drafts,
    }


def write_report(path: Path, outputs: list[dict[str, Any]], outdir: Path) -> None:
    lines = [
        "# Fixture Summary Dry Run",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Fixtures rendered: `{len(outputs)}`",
        f"- Output dir: `{outdir}`",
        "",
    ]
    for output in outputs:
        fixture_key = output.get("fixture_key") or ""
        lines.extend([f"## {fixture_key}", ""])
        for tier, draft in (output.get("drafts") or {}).items():
            lines.extend([f"### {tier}", "", draft.get("summary_text") or "", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    brain_dir = resolve(args.fixture_brain_dir)
    outdir = resolve(args.outdir)
    tiers = args.tier or list(TIERS)
    fixture_keys = set(args.fixture_key or [])
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for path in payload_paths(brain_dir, fixture_keys, args.limit):
        rendered = render_payload(path, tiers)
        if not rendered:
            continue
        write_json(outdir / "fixtures" / f"{slug(rendered['fixture_key'])}.json", rendered)
        outputs.append(rendered)
    index = {
        "schema": "fixture_summary_dry_run_index_v1",
        "generated_at": utc_now(),
        "fixture_brain_dir": str(brain_dir),
        "outdir": str(outdir),
        "tiers": tiers,
        "fixtures_rendered": len(outputs),
        "fixtures": [{"fixture_key": item["fixture_key"], "draft_tiers": sorted((item.get("drafts") or {}).keys())} for item in outputs],
    }
    write_json(outdir / "index.json", index)
    write_report(outdir / "FIXTURE_SUMMARY_DRY_RUN_REPORT.md", outputs, outdir)
    print(json.dumps({"fixtures_rendered": len(outputs), "outdir": str(outdir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
