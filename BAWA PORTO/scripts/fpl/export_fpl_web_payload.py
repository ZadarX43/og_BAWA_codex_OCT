#!/usr/bin/env python3
"""Compile derived Odds Genius FPL web payloads.

This is the website boundary for Fantasy Intelligence. It reads internal
research outputs and writes a compact public-safe JSON payload containing only
derived OG recommendations and user-squad demo context. It does not expose raw
FPL bootstrap tables, fixtures, or private manager data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRANSFER_BOARD = Path("reports/latest/fpl_transfer_advisor/FPL_TRANSFER_ADVISOR_BOARD.csv")
DEFAULT_MARKET_BOARD = Path("reports/latest/fpl_market_state_enrichment_2026_05_26/FPL_ACTION_MARKET_BOARD_MARKET_ENRICHED.csv")
DEFAULT_PERSONAL_BOARD = Path("reports/latest/fpl_personal_squad_decisions_2026_05_26_sample/FPL_PERSONAL_SQUAD_DECISION_BOARD.csv")
DEFAULT_PERSONAL_SUMMARY = Path("reports/latest/fpl_personal_squad_decisions_2026_05_26_sample/FPL_PERSONAL_SQUAD_DECISION_SUMMARY.csv")
DEFAULT_BRIEFING = Path("reports/latest/fpl_briefing/FPL_GAMEWEEK_BRIEFING.json")
DEFAULT_OUT = Path("frontend/public/data/og_fpl/fpl_edge_payload.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return float(number)


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, default)))


def safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_id(value: Any) -> str:
    text = safe_text(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


def price_m(row: pd.Series) -> float:
    if "price_m" in row and safe_float(row.get("price_m")) > 0:
        return round(safe_float(row.get("price_m")), 1)
    if "price_tenths" in row and safe_int(row.get("price_tenths")) > 0:
        return round(safe_int(row.get("price_tenths")) / 10, 1)
    return estimate_price(row)


def estimate_price(row: pd.Series) -> float:
    position = safe_text(row.get("position")).upper()
    base = {"GK": 4.5, "DEF": 4.8, "MID": 6.0, "FWD": 6.5}.get(position, 5.0)
    premium = min(7.0, safe_float(row.get("next_5_ppg", row.get("expected_fpl_points_next_5_gw", 0))) * 0.55)
    risk_discount = safe_float(row.get("risk_score")) * 0.8
    return round(max(4.0, min(15.0, base + premium - risk_discount)), 1)


def player_id(row: pd.Series) -> str:
    value = clean_id(row.get("fpl_player_id")) or clean_id(row.get("player_id"))
    return value


def player_payload(row: pd.Series) -> dict[str, Any]:
    expected_1 = safe_float(row.get("expected_fpl_points_next_1_gw", row.get("next_1_expected_points", row.get("expected_points", 0))))
    expected_3 = safe_float(row.get("expected_fpl_points_next_3_gw", row.get("next_3_expected_points", expected_1 * 3)))
    expected_5 = safe_float(row.get("expected_fpl_points_next_5_gw", row.get("next_5_expected_points", expected_1 * 5)))
    start_probability = safe_float(row.get("start_probability_est"), 0)
    if start_probability <= 1:
        start_probability *= 100
    label = (
        safe_text(row.get("final_action_label"))
        or safe_text(row.get("transfer_in_label"))
        or safe_text(row.get("transfer_label"))
        or "WATCHLIST"
    )
    return {
        "id": player_id(row),
        "fpl_player_id": player_id(row),
        "name": safe_text(row.get("player_name")),
        "position": safe_text(row.get("position")).upper(),
        "club": safe_text(row.get("team_name")),
        "team_id": clean_id(row.get("team_id")),
        "fixture": f"vs {safe_text(row.get('opponent_team_name'))}" if safe_text(row.get("opponent_team_name")) else safe_text(row.get("fixture_key")),
        "price": price_m(row),
        "xpts1": round(expected_1, 1),
        "xpts3": round(expected_3, 1),
        "xpts5": round(expected_5, 1),
        "startPct": int(round(start_probability or 76)),
        "risk": safe_text(row.get("deadline_alert_label")) or safe_text(row.get("risk_flags")) or "Stable",
        "label": label,
        "swing": safe_text(row.get("fixture_swing_label")) or f"{safe_float(row.get('fixture_swing_score')):+.1f}",
        "ownership": safe_text(row.get("ownership_band")) or safe_text(row.get("template_status")) or "Unknown",
        "difficulty": safe_text(row.get("fixture_swing_label")) or "Unknown",
        "note": safe_text(row.get("action_reason")) or safe_text(row.get("transfer_reason")) or safe_text(row.get("reason_tokens")),
        "data_status": safe_text(row.get("data_status")) or "derived",
    }


def build_player_index(market: pd.DataFrame, transfer: pd.DataFrame, personal: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    personal_players = []
    if not personal.empty:
        personal_players = [player_payload(row) for _, row in personal.iterrows()]
    frames = [df for df in [market, transfer] if not df.empty]
    if not frames:
        return [player for player in personal_players if player["id"] and player["name"]]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["__player_id"] = combined.apply(player_id, axis=1)
    combined = combined[combined["__player_id"].astype(str).ne("")]
    score_col = "action_score" if "action_score" in combined.columns else "transfer_score"
    combined[score_col] = pd.to_numeric(combined.get(score_col), errors="coerce").fillna(0)
    combined = combined.sort_values([score_col], ascending=False).drop_duplicates("__player_id")
    players = personal_players + [player_payload(row) for _, row in combined.head(limit).iterrows()]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for player in players:
        if not player["id"] or not player["name"] or player["id"] in seen:
            continue
        seen.add(player["id"])
        out.append(player)
    return out[: max(limit, len(personal_players))]


def build_sample_squad(personal: pd.DataFrame, players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del personal
    by_position: dict[str, list[dict[str, Any]]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    seen_names: set[str] = set()
    for player in players:
      position = safe_text(player.get("position")).upper()
      name_key = safe_text(player.get("name")).lower()
      if position in by_position and name_key not in seen_names:
          seen_names.add(name_key)
          by_position[position].append(player)
    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    selected: dict[str, list[dict[str, Any]]] = {
        position: by_position[position][:count] for position, count in required.items()
    }
    if any(len(selected[position]) < count for position, count in required.items()):
        return []
    starters = [
        selected["GK"][0],
        *selected["DEF"][:3],
        *selected["MID"][:4],
        *selected["FWD"][:3],
    ]
    bench = [selected["GK"][1], selected["DEF"][3], selected["MID"][4], selected["DEF"][4]]
    captain_id = starters[0]["id"]
    vice_id = starters[1]["id"]
    captain_candidates = sorted(starters, key=lambda player: safe_float(player.get("xpts1")), reverse=True)
    if captain_candidates:
        captain_id = captain_candidates[0]["id"]
    if len(captain_candidates) > 1:
        vice_id = captain_candidates[1]["id"]
    out = [
        {
            "id": player["id"],
            "role": "starter",
            "benchOrder": 0,
            "captain": player["id"] == captain_id,
            "vice": player["id"] == vice_id,
        }
        for player in starters
    ]
    out.extend(
        {
            "id": player["id"],
            "role": "bench",
            "benchOrder": index,
            "captain": False,
            "vice": False,
        }
        for index, player in enumerate(bench, start=1)
    )
    return out


def build_summary(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {}
    row = summary_df.iloc[0]
    return {
        "manager_id": safe_text(row.get("manager_id")),
        "as_of_gameweek": safe_text(row.get("as_of_gameweek")),
        "strategy_mode": safe_text(row.get("strategy_mode")),
        "bank": round(safe_int(row.get("bank_tenths")) / 10, 1),
        "free_transfers": safe_int(row.get("free_transfers"), 1),
        "squad_health_score": safe_float(row.get("squad_health_score")),
        "transfer_decision_label": safe_text(row.get("transfer_decision_label")),
        "recommended_captain_name": safe_text(row.get("recommended_captain_name")),
        "recommended_vice_name": safe_text(row.get("recommended_vice_name")),
        "best_transfer_route": safe_text(row.get("best_transfer_route")),
        "decision_reason": safe_text(row.get("decision_reason")),
    }


def build_autocomplete(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for player in players[:120]:
        for kind, value in [
            ("player", player["name"]),
            ("team", player["club"]),
            ("position", player["position"]),
            ("label", player["label"]),
        ]:
            key = (kind, str(value).lower())
            if not value or key in seen:
                continue
            seen.add(key)
            suggestions.append({"type": kind, "label": value, "value": value, "player_id": player["id"] if kind == "player" else ""})
    return suggestions[:160]


def load_briefing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    transfer = read_csv(args.transfer_board)
    market = read_csv(args.market_board)
    personal = read_csv(args.personal_board)
    summary_df = read_csv(args.personal_summary)
    players = build_player_index(market, transfer, personal, args.player_limit)
    payload = {
        "generated_at_utc": utc_now(),
        "schema_version": 1,
        "product": "Odds Genius Fantasy Intelligence",
        "data_boundary": "derived_og_recommendations_only",
        "source_status": {
            "transfer_board": str(args.transfer_board),
            "market_board": str(args.market_board),
            "personal_board": str(args.personal_board),
            "personal_summary": str(args.personal_summary),
        },
        "summary": build_summary(summary_df),
        "players": players,
        "sample_squad": build_sample_squad(personal, players),
        "autocomplete": build_autocomplete(players),
        "briefing": load_briefing(args.briefing),
        "reason_tokens": ["OG_DERIVED_OUTPUT_ONLY", "RAW_FPL_DATA_NOT_EXPOSED", "FRONTEND_SAFE_FPL_PAYLOAD"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {"out": str(args.out), "players": len(players), "sample_squad": len(payload["sample_squad"]), "autocomplete": len(payload["autocomplete"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact derived OG FPL payload for website.")
    parser.add_argument("--transfer-board", type=Path, default=DEFAULT_TRANSFER_BOARD)
    parser.add_argument("--market-board", type=Path, default=DEFAULT_MARKET_BOARD)
    parser.add_argument("--personal-board", type=Path, default=DEFAULT_PERSONAL_BOARD)
    parser.add_argument("--personal-summary", type=Path, default=DEFAULT_PERSONAL_SUMMARY)
    parser.add_argument("--briefing", type=Path, default=DEFAULT_BRIEFING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--player-limit", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
