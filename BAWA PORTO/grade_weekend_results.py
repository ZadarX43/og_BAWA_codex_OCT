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
WEEKLY_RESULTS_PATH = FRONTEND_DATA_DIR / "weekly_results.json"
RESULTS_ARCHIVE_PATH = FRONTEND_DATA_DIR / "results_archive.json"

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


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def normalize_market(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"FTR", "BTTS"}:
        return text
    if text in {"OU25", "OVER25", "UNDER25"}:
        return "OU25"
    return text


def normalize_tier(value: str) -> str:
    return str(value or "").strip().upper()


def normalize_pick(value: Any) -> str:
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


def settlement_status(row: dict[str, str], market: str, selection: str) -> str:
    status_text = str(row.get("status") or "").strip().lower()
    if any(token in status_text for token in ("void", "cancel", "postpon", "abandon")):
        return "void"
    hit = extract_hit(row, market, selection)
    if hit is None:
        return "pending"
    return "won" if hit >= 0.5 else "lost"


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


def build_summary_block(total: int, settled: int, wins: int, losses: int, voids: int, profit_units: float) -> dict[str, Any]:
    pending = total - settled
    hit_rate = round(wins / settled, 4) if settled else None
    roi = round(profit_units / settled, 4) if settled else None
    return {
        "total_picks": total,
        "settled_picks": settled,
        "pending_picks": pending,
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "hit_rate": hit_rate,
        "roi": roi,
        "profit_units": round(profit_units, 2),
    }


def compute_profit_units(result_status: str, bookie_od: float | None) -> float:
    if result_status == "won":
        return (bookie_od - 1.0) if bookie_od is not None else 0.0
    if result_status == "lost":
        return -1.0
    return 0.0


def safe_archive_row(
    row: dict[str, str],
    source_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    market = normalize_market(row.get("market", ""))
    selection = normalize_pick(row.get("selection") or row.get("bookie_pick") or "")
    result_status = settlement_status(row, market, selection)
    bookie_od = parse_float(row.get("bookie_od", ""))
    model_prob = parse_float(row.get("model_p_for_bookie") or row.get("p_pick") or "")
    bookie_implied_prob = parse_float(
        row.get("bookie_implied_used")
        or row.get("bookie_implied_novig")
        or row.get("bookie_implied")
        or ""
    )
    value_edge = parse_float(row.get("value_edge") or row.get("gap") or row.get("gap_novig") or "")
    final_home_score = parse_float(row.get("home_team_goal_count", ""))
    final_away_score = parse_float(row.get("away_team_goal_count", ""))
    confidence_tier = normalize_tier(row.get("deploy_tier") or row.get("tier") or "")
    return {
        "fixture_key": str(row.get("fixture_key", "") or "").strip(),
        "kickoff_time": str(row.get("match_date", "") or "").strip(),
        "league": str(row.get("league", "") or "").strip(),
        "home_team": str(row.get("home_team_name") or row.get("home_team") or "").strip(),
        "away_team": str(row.get("away_team_name") or row.get("away_team") or "").strip(),
        "market": market,
        "pick": selection,
        "confidence_tier": confidence_tier,
        "tier": confidence_tier,
        "premium_tier": confidence_tier,
        "bookie_od": round_or_none(bookie_od, 4),
        "model_prob": round_or_none(model_prob, 4),
        "bookie_implied_prob": round_or_none(bookie_implied_prob, 4),
        "value_edge": round_or_none(value_edge, 4),
        "result_status": result_status,
        "profit_units": round(compute_profit_units(result_status, bookie_od), 2),
        "final_home_score": int(final_home_score) if final_home_score is not None else None,
        "final_away_score": int(final_away_score) if final_away_score is not None else None,
        "settled_at": generated_at if result_status != "pending" else "",
        "published_run_id": source_path.stem,
    }


def build_chart_points(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("result_status") == "pending":
            continue
        date_key = str(item.get("kickoff_time") or "").strip()
        if not date_key:
            continue
        grouped[date_key].append(item)

    chart_points: list[dict[str, Any]] = []
    cumulative_profit = 0.0
    cumulative_settled = 0
    cumulative_wins = 0

    for date_key in sorted(grouped):
        rows = grouped[date_key]
        settled = len(rows)
        wins = sum(1 for row in rows if row.get("result_status") == "won")
        losses = sum(1 for row in rows if row.get("result_status") == "lost")
        voids = sum(1 for row in rows if row.get("result_status") == "void")
        profit = round(sum(float(row.get("profit_units") or 0.0) for row in rows), 2)
        cumulative_profit = round(cumulative_profit + profit, 2)
        cumulative_settled += settled
        cumulative_wins += wins
        chart_points.append(
            {
                "date": date_key,
                "settled_picks": settled,
                "wins": wins,
                "losses": losses,
                "voids": voids,
                "profit_units": profit,
                "cumulative_profit_units": cumulative_profit,
                "rolling_hit_rate": round(cumulative_wins / cumulative_settled, 4) if cumulative_settled else None,
                "cumulative_roi": round(cumulative_profit / cumulative_settled, 4) if cumulative_settled else None,
            }
        )
    return chart_points


def summarize_archive_rows(items: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get(label_key) or "").strip()].append(item)

    blocks: list[dict[str, Any]] = []
    for label in sorted(key for key in grouped if key):
        rows = grouped[label]
        total = len(rows)
        settled_rows = [row for row in rows if row.get("result_status") != "pending"]
        settled = len(settled_rows)
        wins = sum(1 for row in settled_rows if row.get("result_status") == "won")
        losses = sum(1 for row in settled_rows if row.get("result_status") == "lost")
        voids = sum(1 for row in settled_rows if row.get("result_status") == "void")
        profit_units = sum(float(row.get("profit_units") or 0.0) for row in settled_rows)
        block = build_summary_block(total, settled, wins, losses, voids, profit_units)
        block[label_key] = label
        blocks.append(block)
    return blocks


def main() -> int:
    args = parse_args()
    source_path = resolve_scored_source(str(args.src or "").strip() or None)
    rows = load_rows(source_path)
    generated_at = utc_now_iso()

    archive_rows: list[dict[str, Any]] = []
    featured_wins: list[dict[str, Any]] = []
    featured_misses: list[dict[str, Any]] = []
    notes: list[str] = []
    unsupported_markets = Counter()

    period_dates: list[str] = []
    total_picks = settled_picks = wins = losses = voids = 0
    total_profit_units = 0.0

    for row in rows:
        tier = normalize_tier(row.get("deploy_tier") or row.get("tier") or "")
        market = normalize_market(row.get("market", ""))
        selection = normalize_pick(row.get("selection") or row.get("bookie_pick") or "")

        if tier not in DEPLOYABLE_TIERS:
            continue
        if market not in SUPPORTED_MARKETS:
            unsupported_markets[market or "UNKNOWN"] += 1
            continue

        period_dates.append(str(row.get("match_date", "") or "").strip())
        total_picks += 1
        archive_item = safe_archive_row(row, source_path, generated_at)
        archive_rows.append(archive_item)

        result_status = str(archive_item.get("result_status") or "")
        if result_status == "pending":
            continue

        settled_picks += 1
        total_profit_units += float(archive_item.get("profit_units") or 0.0)

        if result_status == "won":
            wins += 1
            hit = extract_hit(row, market, selection) or 1.0
            if len(featured_wins) < 5:
                featured_wins.append(safe_fixture_card(row, hit))
        elif result_status == "lost":
            losses += 1
            hit = extract_hit(row, market, selection) or 0.0
            if len(featured_misses) < 5:
                featured_misses.append(safe_fixture_card(row, hit))
        elif result_status == "void":
            voids += 1

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

    archive_rows.sort(
        key=lambda item: (
            str(item.get("kickoff_time") or ""),
            str(item.get("league") or ""),
            str(item.get("home_team") or ""),
            str(item.get("away_team") or ""),
            str(item.get("market") or ""),
            str(item.get("confidence_tier") or ""),
        )
    )

    by_market = summarize_archive_rows(archive_rows, "market")
    by_tier = summarize_archive_rows(archive_rows, "tier")
    by_league = summarize_archive_rows(archive_rows, "league")
    chart_points = build_chart_points(archive_rows)

    payload = {
        "period_start": period_start,
        "period_end": period_end,
        "generated_at": generated_at,
        "total_picks": total_picks,
        "settled_picks": settled_picks,
        "pending_picks": max(total_picks - settled_picks, 0),
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "overall_hit_rate": round(wins / settled_picks, 4) if settled_picks else None,
        "overall_roi": round(total_profit_units / settled_picks, 4) if settled_picks else None,
        "overall_profit_units": round(total_profit_units, 2),
        "by_market": by_market,
        "by_tier": by_tier,
        "by_league": by_league,
        "chart_points": chart_points,
        "featured_wins": featured_wins,
        "featured_misses": featured_misses,
        "notes": notes,
    }
    archive_payload = {
        "period_start": period_start,
        "period_end": period_end,
        "generated_at": generated_at,
        "source_file": source_path.name,
        "published_run_id": source_path.stem,
        "total_picks": total_picks,
        "settled_picks": settled_picks,
        "pending_picks": max(total_picks - settled_picks, 0),
        "items": archive_rows,
    }

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEEKLY_RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    RESULTS_ARCHIVE_PATH.write_text(
        json.dumps(archive_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Results source CSV: {source_path.relative_to(ROOT)}")
    print(f"weekly_results.json written: {WEEKLY_RESULTS_PATH.relative_to(ROOT)}")
    print(f"results_archive.json written: {RESULTS_ARCHIVE_PATH.relative_to(ROOT)}")
    print(f"Total picks: {total_picks}")
    print(f"Settled picks: {settled_picks}")
    print(f"Pending picks: {max(total_picks - settled_picks, 0)}")
    print(f"Placeholder mode: {placeholder_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
