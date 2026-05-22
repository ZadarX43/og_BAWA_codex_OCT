#!/usr/bin/env python3
"""Publish a lightweight, public-safe H2H support layer for fixture pages."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from publish_snapshot_metadata import metadata_from_fixture, utc_now_iso
from ratings_engine_utils import clean_columns


def clamp_percent(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    return int(round(max(0.0, min(100.0, float(numeric) * 100.0 if float(numeric) <= 1.0 else float(numeric)))))


def safe_int(value: object) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def pct_avg(*values: object) -> int:
    numeric = [clamp_percent(value) for value in values if value not in (None, "")]
    numeric = [value for value in numeric if value > 0]
    if not numeric:
        return 0
    return int(round(sum(numeric) / len(numeric)))


def h2h_signal_fields(row: dict) -> dict:
    sample = int(row.get("sample_size") or 0)
    if sample <= 0:
        return {
            "status": "fallback",
            "goal_heat": 0,
            "over25_heat": 0,
            "btts_pressure": 0,
            "attack_flow": 0,
            "defensive_lock": 0,
            "chaos_rating": 0,
            "first_strike": 0,
            "first_strike_side": "",
        }
    goal_heat = clamp_percent(row.get("goal_environment"))
    over25_heat = clamp_percent(row.get("over25_rate"))
    btts_pressure = clamp_percent(row.get("btts_regime"))
    booking_heat = clamp_percent(row.get("booking_heat"))
    foul_intensity = clamp_percent(row.get("foul_intensity"))
    style_conflict = clamp_percent(row.get("style_conflict_index"))
    attack_flow = pct_avg(goal_heat, over25_heat, btts_pressure)
    defensive_lock = max(0, min(100, 100 - pct_avg(goal_heat, btts_pressure)))
    chaos_rating = pct_avg(style_conflict, booking_heat, foul_intensity)
    home_win = clamp_percent(row.get("home_win_rate"))
    draw_rate = clamp_percent(row.get("draw_rate"))
    away_win = max(0, min(100, 100 - home_win - draw_rate))
    return {
        "status": "available" if int(row.get("sample_size") or 0) > 0 else "fallback",
        "goal_heat": goal_heat,
        "over25_heat": over25_heat,
        "btts_pressure": btts_pressure,
        "attack_flow": attack_flow,
        "defensive_lock": defensive_lock,
        "chaos_rating": chaos_rating,
        "first_strike": max(home_win, away_win),
        "first_strike_side": "home" if home_win >= away_win else "away",
    }


def finalize_payload(payload: dict, *, fallback_mode: str = "direct_fixture_history") -> dict:
    payload.update(h2h_signal_fields(payload))
    sample = int(payload.get("sample_size") or 0)
    payload.setdefault("coverage_status", "available" if sample > 0 else "thin_history")
    payload.setdefault("fallback_mode", fallback_mode if sample > 0 else "thin_history")
    payload["summary"] = summary_for_row(payload)
    return payload


def summary_for_row(row: dict) -> str:
    sample = int(row.get("sample_size") or 0)
    if sample <= 0:
      return "Historic matchup regime is not established enough yet, so this block stays supporting-only."
    goal = int(row.get("goal_heat") or row.get("goal_environment") or 0)
    btts = int(row.get("btts_pressure") or row.get("btts_regime") or 0)
    over = int(row.get("over25_heat") or row.get("over25_rate") or 0)
    attack = int(row.get("attack_flow") or 0)
    chaos = int(row.get("chaos_rating") or 0)
    return (
        f"Last-{sample} meeting regime shows {goal}% goal environment, {btts}% BTTS pressure, "
        f"{over}% Over 2.5 frequency, {attack}% attack flow, and {chaos}% chaos rating. "
        "Treat this as supporting context rather than the primary read."
    )


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower().replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def competition_key(value: object) -> str:
    text = normalize_text(value)
    aliases = {
        "swiss super league": "switzerland super league",
    }
    return aliases.get(text, text)


def team_key(value: object) -> str:
    text = normalize_text(value)
    club_tokens = {"fc", "cf", "sc", "afc", "club", "vfl", "sv", "as", "kv", "kvc", "krc"}
    tokens = [token for token in text.split() if token not in club_tokens]
    return " ".join(tokens)


def load_fixture_feed_rows(fixture_feed_path: Path | None) -> list[dict]:
    if not fixture_feed_path or not fixture_feed_path.exists():
        return []
    fixture_feed = json.loads(fixture_feed_path.read_text(encoding="utf-8"))
    fixtures = fixture_feed.get("fixtures") if isinstance(fixture_feed, dict) else None
    return list(fixtures) if isinstance(fixtures, list) else []


def feature_competition_year(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})$", path.stem)
    if not match:
        return None
    return competition_key(match.group(1).replace("_", " ")), int(match.group(2))


def latest_feature_paths(features_root: Path, prefix: str, active_competitions: set[str] | None = None) -> list[Path]:
    if not active_competitions:
        return sorted(features_root.glob(f"{prefix}__*.csv"))

    latest: dict[str, tuple[int, Path]] = {}
    for path in sorted(features_root.glob(f"{prefix}__*.csv")):
        parsed = feature_competition_year(path, prefix)
        if not parsed:
            continue
        comp_key, year = parsed
        if comp_key not in active_competitions:
            continue
        current = latest.get(comp_key)
        if current is None or year > current[0]:
            latest[comp_key] = (year, path)
    return [item[1] for item in sorted(latest.values(), key=lambda item: str(item[1]))]


def build_payloads(features_root: Path, active_competitions: set[str] | None = None) -> tuple[dict[str, dict], list[dict]]:
    payloads: dict[str, dict] = {}
    index_rows: list[dict] = []
    for csv_path in latest_feature_paths(features_root, "api_h2h_regime_features", active_competitions):
        parsed_path = feature_competition_year(csv_path, "api_h2h_regime_features")
        source_competition_key = parsed_path[0] if parsed_path else ""
        df = clean_columns(pd.read_csv(csv_path))
        if "fixture_key" not in df.columns:
            continue
        for row in df.to_dict(orient="records"):
            fixture_key = str(row.get("fixture_key") or "").strip()
            if not fixture_key:
                continue
            payload = {
                "fixture_key": fixture_key,
                "fixture_id": int(row.get("fixture_id") or 0) if pd.notna(row.get("fixture_id")) else None,
                "competition": row.get("league"),
                "competition_key": source_competition_key or competition_key(row.get("league")),
                "season": str(row.get("season") or ""),
                "home_team": row.get("home_team_name"),
                "away_team": row.get("away_team_name"),
                "sample_size": int(row.get("h2h_n_l5") or 0),
                "goal_environment": clamp_percent(row.get("h2h_goal_environment")),
                "btts_regime": clamp_percent(row.get("h2h_btts_regime")),
                "booking_heat": clamp_percent(row.get("h2h_booking_heat")),
                "foul_intensity": clamp_percent(row.get("h2h_foul_intensity")),
                "style_conflict_index": clamp_percent(row.get("h2h_style_conflict_index")),
                "home_win_rate": clamp_percent(row.get("h2h_home_win_rate_last5")),
                "draw_rate": clamp_percent(row.get("h2h_draw_rate_last5")),
                "over25_rate": clamp_percent(row.get("h2h_over25_rate_last5")),
                "high_cards_rate": clamp_percent(row.get("h2h_high_cards_rate_last5")),
                "same_referee_overlap": bool(row.get("h2h_same_referee_overlap")),
                "same_referee_count": int(row.get("h2h_same_referee_count_l5") or 0),
            }
            payload = finalize_payload(payload)
            payloads[fixture_key] = payload
            index_rows.append(
                {
                    "fixture_key": fixture_key,
                    "fixture_id": payload["fixture_id"],
                    "competition": payload["competition"],
                    "competition_key": payload["competition_key"],
                    "season": payload["season"],
                    "sample_size": payload["sample_size"],
                }
            )
    return payloads, index_rows


def placeholder_payload(fixture: dict) -> dict:
    fixture_key = str(fixture.get("fixture_key") or "").strip()
    payload = {
        "fixture_key": fixture_key,
        "fixture_id": safe_int(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
        "competition": fixture.get("league"),
        "competition_key": competition_key(fixture.get("league")),
        "season": str(fixture.get("api_season") or fixture.get("season") or ""),
        "home_team": fixture.get("home_team"),
        "away_team": fixture.get("away_team"),
        "sample_size": 0,
        "goal_environment": 0,
        "btts_regime": 0,
        "booking_heat": 0,
        "foul_intensity": 0,
        "style_conflict_index": 0,
        "home_win_rate": 0,
        "draw_rate": 0,
        "over25_rate": 0,
        "high_cards_rate": 0,
        "same_referee_overlap": False,
        "same_referee_count": 0,
        "coverage_status": "unpublished",
        "fallback_mode": "unpublished",
    }
    payload.update(h2h_signal_fields(payload))
    payload["summary"] = "No publish-safe H2H regime snapshot is available yet for this fixture, so H2H remains supporting-only."
    return payload


def augment_with_fixture_feed(
    payloads: dict[str, dict],
    index_rows: list[dict],
    fixture_feed_rows: list[dict],
) -> tuple[dict[str, dict], list[dict]]:
    if not fixture_feed_rows:
        return payloads, index_rows

    by_pair: dict[tuple[str, str, str], list[dict]] = {}
    for payload in payloads.values():
        competition = payload.get("competition_key") or competition_key(payload.get("competition"))
        home = team_key(payload.get("home_team"))
        away = team_key(payload.get("away_team"))
        if not competition or not home or not away:
            continue
        by_pair.setdefault((competition, home, away), []).append(payload)
        by_pair.setdefault((competition, away, home), []).append(payload)

    existing_keys = set(payloads.keys())
    augmented = dict(payloads)
    augmented_index = list(index_rows)
    capture_generated_at = utc_now_iso()
    for fixture in fixture_feed_rows:
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        if fixture_key in existing_keys:
            if fixture_key in augmented:
                augmented[fixture_key].update(metadata_from_fixture(fixture, capture_generated_at=capture_generated_at))
            continue
        pair = (
            competition_key(fixture.get("league")),
            team_key(fixture.get("home_team")),
            team_key(fixture.get("away_team")),
        )
        candidates = by_pair.get(pair) or []
        if candidates:
            source = sorted(
                candidates,
                key=lambda item: (str(item.get("season") or ""), int(item.get("sample_size") or 0), str(item.get("fixture_key") or "")),
                reverse=True,
            )[0]
            payload = dict(source)
            payload.update(
                {
                    "fixture_key": fixture_key,
                    "fixture_id": safe_int(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
                    "competition": fixture.get("league") or source.get("competition"),
                    "competition_key": source.get("competition_key") or competition_key(fixture.get("league")),
                    "season": str(fixture.get("api_season") or fixture.get("season") or source.get("season") or ""),
                    "home_team": fixture.get("home_team") or source.get("home_team"),
                    "away_team": fixture.get("away_team") or source.get("away_team"),
                    "fallback_mode": "historical_team_pair",
                    "source_fixture_key": source.get("fixture_key"),
                }
            )
            payload.update(h2h_signal_fields(payload))
            sample = int(payload.get("sample_size") or 0)
            payload["summary"] = (
                f"Using the latest available same-team-pair historical regime ({sample} prior meeting"
                f"{'' if sample == 1 else 's'}) as supporting context for this current fixture."
            )
        else:
            payload = placeholder_payload(fixture)
        payload.update(metadata_from_fixture(fixture, capture_generated_at=capture_generated_at))
        augmented[fixture_key] = payload
        augmented_index.append(
            {
                "fixture_key": fixture_key,
                "fixture_id": payload["fixture_id"],
                "competition": payload["competition"],
                "competition_key": payload.get("competition_key"),
                "season": payload["season"],
                "sample_size": payload["sample_size"],
                "fallback_mode": payload.get("fallback_mode"),
                "coverage_status": payload.get("coverage_status"),
                "capture_generated_at": payload.get("capture_generated_at"),
                "source_data_cutoff_at": payload.get("source_data_cutoff_at"),
                "fixture_kickoff_at": payload.get("fixture_kickoff_at"),
                "pre_kickoff_eligible": payload.get("pre_kickoff_eligible"),
                "snapshot_phase": payload.get("snapshot_phase"),
            }
        )
        existing_keys.add(fixture_key)
    current_keys = {str(row.get("fixture_key") or "").strip() for row in fixture_feed_rows if row.get("fixture_key")}
    if current_keys:
        augmented_index = [
            {
                "fixture_key": key,
                "fixture_id": payload.get("fixture_id"),
                "competition": payload.get("competition"),
                "competition_key": payload.get("competition_key"),
                "season": payload.get("season"),
                "sample_size": payload.get("sample_size"),
                "fallback_mode": payload.get("fallback_mode"),
                "coverage_status": payload.get("coverage_status"),
                "capture_generated_at": payload.get("capture_generated_at"),
                "source_data_cutoff_at": payload.get("source_data_cutoff_at"),
                "fixture_kickoff_at": payload.get("fixture_kickoff_at"),
                "pre_kickoff_eligible": payload.get("pre_kickoff_eligible"),
                "snapshot_phase": payload.get("snapshot_phase"),
            }
            for key, payload in augmented.items()
            if key in current_keys
        ]
    return augmented, augmented_index


def publish(payloads: dict[str, dict], index_rows: list[dict], output_root: Path, prune_output: bool = False) -> None:
    target_dir = output_root / "fixture_h2h_support"
    target_dir.mkdir(parents=True, exist_ok=True)
    for fixture_key, payload in payloads.items():
        (target_dir / f"{fixture_key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if prune_output:
        keep = {f"{fixture_key}.json" for fixture_key in payloads}
        keep.add("index.json")
        for path in target_dir.glob("*.json"):
            if path.name not in keep:
                path.unlink()
    stable_index = sorted(index_rows, key=lambda row: str(row.get("fixture_key") or ""))
    (target_dir / "index.json").write_text(json.dumps(stable_index, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish public-safe fixture H2H support JSON.")
    parser.add_argument("--features-root", default="data_sources/api_football/features")
    parser.add_argument("--fixture-feed", default=None, help="Optional fixture feed JSON to backfill current fixture keys.")
    parser.add_argument("--output-root", default="frontend/public/data")
    parser.add_argument("--prune-output", action="store_true", help="Remove stale fixture JSON files outside the generated active set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture_feed_rows = load_fixture_feed_rows(Path(args.fixture_feed) if args.fixture_feed else None)
    active_competitions = {competition_key(row.get("league")) for row in fixture_feed_rows if row.get("league")} or None
    payloads, index_rows = build_payloads(Path(args.features_root), active_competitions)
    payloads, index_rows = augment_with_fixture_feed(payloads, index_rows, fixture_feed_rows)
    if fixture_feed_rows:
        current_keys = {str(row.get("fixture_key") or "").strip() for row in fixture_feed_rows if row.get("fixture_key")}
        payloads = {key: payload for key, payload in payloads.items() if key in current_keys}
        index_rows = [row for row in index_rows if str(row.get("fixture_key") or "").strip() in current_keys]
    publish(payloads, index_rows, Path(args.output_root), prune_output=bool(fixture_feed_rows) or args.prune_output)
    print(f"Published {len(payloads)} fixture H2H support payloads.")


if __name__ == "__main__":
    main()
