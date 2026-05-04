#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PREDICTIONS_ROOT = ROOT / "predictions_output"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"

DEPLOYABLE_TIERS = {"ELITE", "STANDARD"}
SUPPORTED_MARKETS = {"FTR", "BTTS", "OU25"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a public-safe weekly results JSON from scored deploy outputs."
    )
    parser.add_argument(
        "--src",
        default="",
        help="Optional explicit DEPLOY_COMBINED_SCORED CSV path. If omitted, the newest scored file with settled deployable picks is used.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out


def normalize_market(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"FTR", "BTTS"}:
        return text
    if text in {"OU25", "OVER25", "UNDER25"}:
        return "OU25"
    return text


def normalize_tier(value: str) -> str:
    return str(value or "").strip().upper()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def has_settled_deployable_pick(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tier = normalize_tier(row.get("deploy_tier") or row.get("tier") or "")
            market = normalize_market(row.get("market", ""))
            selection = str(row.get("selection") or row.get("bookie_pick") or "").strip().upper()
            if tier not in DEPLOYABLE_TIERS or market not in SUPPORTED_MARKETS:
                continue
            if is_settled(row, market, selection):
                return True
    return False


def extract_hit(row: dict[str, str], market: str, selection: str) -> float | None:
    if market == "FTR":
        return parse_float(row.get("ftr_hit", ""))
    if market == "OU25":
        return parse_float(row.get("ou25_hit", ""))
    if market == "BTTS":
        selection = str(selection or "").strip().upper()
        if selection == "NO":
            hit = parse_float(row.get("btts_no_hit", ""))
            if hit is not None:
                return hit
        return parse_float(row.get("btts_yes_hit", ""))
    return None


def is_settled(row: dict[str, str], market: str, selection: str) -> bool:
    return extract_hit(row, market, selection) is not None


def resolve_scored_source(src: str | None) -> Path:
    if src:
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Explicit --src file not found: {candidate}")
        if "SCORED" not in candidate.name.upper():
            raise ValueError(f"Explicit --src must be a scored deploy file: {candidate}")
        return candidate

    files = sorted(
        [path for path in PREDICTIONS_ROOT.rglob("DEPLOY_COMBINED_SCORED_*.csv") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        if has_settled_deployable_pick(path):
            return path

    if not files:
        raise FileNotFoundError("No scored deploy file with settled deployable picks was found.")
    raise FileNotFoundError("Scored deploy files exist, but none contain settled deployable picks yet.")


def safe_fixture_card(row: dict[str, str], hit: float) -> dict[str, Any]:
    return {
        "fixture_key": str(row.get("fixture_key", "") or "").strip(),
        "kickoff_time": str(row.get("match_date", "") or "").strip(),
        "league": str(row.get("league", "") or "").strip(),
        "home_team": str(row.get("home_team_name") or row.get("home_team") or "").strip(),
        "away_team": str(row.get("away_team_name") or row.get("away_team") or "").strip(),
        "market": normalize_market(row.get("market", "")),
        "pick": str(row.get("selection") or row.get("bookie_pick") or "").strip().upper(),
        "confidence_tier": normalize_tier(row.get("deploy_tier") or row.get("tier") or ""),
        "result": "WIN" if hit >= 0.5 else "MISS",
    }


def build_summary_block(total: int, settled: int, wins: int) -> dict[str, Any]:
    pending = total - settled
    hit_rate = round(wins / settled, 4) if settled else None
    return {
        "total_picks": total,
        "settled_picks": settled,
        "pending_picks": pending,
        "hit_rate": hit_rate,
    }


def main() -> int:
    args = parse_args()
    source_path = resolve_scored_source(str(args.src or "").strip() or None)
    rows = load_rows(source_path)

    deployable_rows: list[dict[str, str]] = []
    featured_wins: list[dict[str, Any]] = []
    featured_misses: list[dict[str, Any]] = []
    by_market_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_tier_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    notes: list[str] = []
    unsupported_markets = Counter()

    period_dates: list[str] = []
    total_picks = settled_picks = wins = 0

    for row in rows:
        tier = normalize_tier(row.get("deploy_tier") or row.get("tier") or "")
        market = normalize_market(row.get("market", ""))
        selection = str(row.get("selection") or row.get("bookie_pick") or "").strip().upper()

        if tier not in DEPLOYABLE_TIERS:
            continue
        if market not in SUPPORTED_MARKETS:
            unsupported_markets[market or "UNKNOWN"] += 1
            continue

        deployable_rows.append(row)
        period_dates.append(str(row.get("match_date", "") or "").strip())
        total_picks += 1
        by_market_rows[market].append(row)
        by_tier_rows[tier].append(row)

        hit = extract_hit(row, market, selection)
        if hit is None:
            continue

        settled_picks += 1
        if hit >= 0.5:
            wins += 1
            if len(featured_wins) < 5:
                featured_wins.append(safe_fixture_card(row, hit))
        else:
            if len(featured_misses) < 5:
                featured_misses.append(safe_fixture_card(row, hit))

    if unsupported_markets:
        notes.append(
            "Unsupported markets were excluded from public weekly results: "
            + ", ".join(f"{market} ({count})" for market, count in sorted(unsupported_markets.items()))
        )

    placeholder_mode = settled_picks == 0
    if placeholder_mode:
        notes.append(
            "No settled deployable picks were available in the selected scored source. "
            "weekly_results.json was generated in placeholder mode and needs a scored deploy file with hit fields."
        )

    period_dates = sorted([value for value in period_dates if value])
    period_start = period_dates[0] if period_dates else ""
    period_end = period_dates[-1] if period_dates else ""

    by_market = []
    for market, market_rows in sorted(by_market_rows.items()):
        market_total = len(market_rows)
        market_settled = 0
        market_wins = 0
        for row in market_rows:
            selection = str(row.get("selection") or row.get("bookie_pick") or "").strip().upper()
            hit = extract_hit(row, market, selection)
            if hit is None:
                continue
            market_settled += 1
            if hit >= 0.5:
                market_wins += 1
        block = build_summary_block(market_total, market_settled, market_wins)
        block["market"] = market
        by_market.append(block)

    by_tier = []
    for tier, tier_rows in sorted(by_tier_rows.items()):
        tier_total = len(tier_rows)
        tier_settled = 0
        tier_wins = 0
        for row in tier_rows:
            market = normalize_market(row.get("market", ""))
            selection = str(row.get("selection") or row.get("bookie_pick") or "").strip().upper()
            hit = extract_hit(row, market, selection)
            if hit is None:
                continue
            tier_settled += 1
            if hit >= 0.5:
                tier_wins += 1
        block = build_summary_block(tier_total, tier_settled, tier_wins)
        block["tier"] = tier
        by_tier.append(block)

    payload = {
        "period_start": period_start,
        "period_end": period_end,
        "generated_at": utc_now_iso(),
        "total_picks": total_picks,
        "settled_picks": settled_picks,
        "pending_picks": max(total_picks - settled_picks, 0),
        "overall_hit_rate": round(wins / settled_picks, 4) if settled_picks else None,
        "by_market": by_market,
        "by_tier": by_tier,
        "featured_wins": featured_wins,
        "featured_misses": featured_misses,
        "notes": notes,
    }

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FRONTEND_DATA_DIR / "weekly_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Results source CSV: {source_path.relative_to(ROOT)}")
    print(f"weekly_results.json written: {out_path.relative_to(ROOT)}")
    print(f"Total picks: {total_picks}")
    print(f"Settled picks: {settled_picks}")
    print(f"Pending picks: {max(total_picks - settled_picks, 0)}")
    print(f"Placeholder mode: {placeholder_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
