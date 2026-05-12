#!/usr/bin/env python3
"""Add deterministic preview copy to published fixture decision intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATA_ROOT_DEFAULT = Path("frontend/public/data")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def safe_title(value: object, fallback: str = "—") -> str:
    raw = str(value or "").replace("_", " ").strip()
    return raw.title() if raw else fallback


def public_reason_label(token: object) -> str:
    raw = str(token or "").strip().upper()
    if not raw:
        return "structural caution"
    raw = (
        raw.replace("H2H_", "H2H ")
        .replace("BTTS_", "BTTS ")
        .replace("OU25_", "OU25 ")
        .replace("TEAM_", "Team ")
        .replace("AWAY_", "Away ")
        .replace("HOME_", "Home ")
    )
    return safe_title(raw)


def best_markets(decision: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    markets = decision.get("market_intelligence") or {}
    rows = [(key, value) for key, value in markets.items() if isinstance(value, dict)]
    rows.sort(key=lambda item: (int(item[1].get("alignment_score") or 0), str(item[0])), reverse=True)
    return rows


def market_label(key: str) -> str:
    labels = {
        "ftr": "FTR",
        "btts": "BTTS",
        "ou25": "Over 2.5",
        "team_goals": "Team Goals",
        "correct_score": "Correct Score",
        "corners": "Corners",
        "cards": "Cards",
    }
    return labels.get(key, safe_title(key))


def preview_headline(decision: dict[str, Any]) -> str:
    fixture = decision.get("fixture") or "Fixture"
    signal = decision.get("primary_signal") or "Fixture read"
    state = safe_title(decision.get("signal_state"), "Pending")
    return f"{fixture}: {signal} {state.lower()}"


def preview_short_summary(decision: dict[str, Any]) -> str:
    support = public_reason_label((decision.get("supporting_layers") or [None])[0])
    caution = public_reason_label((decision.get("caution_layers") or [None])[0])
    base = decision.get("public_safe_summary") or "No public-safe fixture summary is available yet."
    return f"{base} Primary support comes from {support.lower()}, while the main caution remains {caution.lower()}."


def preview_market_summary(decision: dict[str, Any]) -> str:
    ranked = best_markets(decision)
    if not ranked:
        return "No market alignment summary is available for this fixture yet."
    best_key, best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    weakest_key, weakest = ranked[-1]
    parts = [
        f"Best aligned market: {market_label(best_key)} ({best.get('state', 'Pending').title()})",
    ]
    if second:
        parts.append(f"Secondary support: {market_label(second[0])}")
    parts.append(f"Weakest read: {market_label(weakest_key)}")
    return ". ".join(parts) + "."


def preview_caution_line(decision: dict[str, Any]) -> str:
    caution = public_reason_label((decision.get("caution_layers") or [None])[0])
    return f"Main caution: {caution}."


def preview_telegram_summary(decision: dict[str, Any]) -> str:
    signal = decision.get("primary_signal") or "Fixture read"
    state = safe_title(decision.get("signal_state"), "Pending")
    agreement = decision.get("agreement_score")
    support = public_reason_label((decision.get("supporting_layers") or [None])[0])
    caution = public_reason_label((decision.get("caution_layers") or [None])[0])
    fixture = decision.get("fixture") or "Fixture"
    return (
        f"{fixture} — {signal} ({state}, {agreement}% agreement). "
        f"Support: {support}. Caution: {caution}."
    )


def preview_premium_summary(decision: dict[str, Any]) -> str:
    signal = decision.get("primary_signal") or "Fixture read"
    agreement = decision.get("agreement_score")
    support = [public_reason_label(token).lower() for token in (decision.get("supporting_layers") or [])[:2]]
    caution = [public_reason_label(token).lower() for token in (decision.get("caution_layers") or [])[:2]]
    support_line = ", ".join(support) if support else "limited structural support"
    caution_line = ", ".join(caution) if caution else "no major caution published"
    return (
        f"{signal} is carrying a {agreement}% agreement read across the published decision layers. "
        f"Support is currently led by {support_line}, while caution is still coming from {caution_line}."
    )


def build_preview(decision: dict[str, Any]) -> dict[str, str]:
    return {
        "headline": preview_headline(decision),
        "short_summary": preview_short_summary(decision),
        "market_summary": preview_market_summary(decision),
        "caution_line": preview_caution_line(decision),
        "telegram_summary": preview_telegram_summary(decision),
        "premium_summary": preview_premium_summary(decision),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publish-safe fixture previews from fixture decision JSON.")
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--decision-root", default=None, help="Optional override for fixture decision directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    decision_root = Path(args.decision_root) if args.decision_root else data_root / "fixture_decision_intelligence"
    total = 0
    for path in sorted(decision_root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = load_json(path)
        payload["preview"] = build_preview(payload)
        write_json(path, payload)
        total += 1
    print(f"Added preview objects to {total} fixture decision payloads.")


if __name__ == "__main__":
    main()
