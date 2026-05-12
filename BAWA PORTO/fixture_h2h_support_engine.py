#!/usr/bin/env python3
"""Publish a lightweight, public-safe H2H support layer for fixture pages."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

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


def summary_for_row(row: dict) -> str:
    sample = int(row.get("sample_size") or 0)
    if sample <= 0:
      return "Historic matchup regime is not established enough yet, so this block stays supporting-only."
    goal = int(row.get("goal_environment") or 0)
    btts = int(row.get("btts_regime") or 0)
    over = int(row.get("over25_rate") or 0)
    booking = int(row.get("booking_heat") or 0)
    return (
        f"Last-{sample} meeting regime shows {goal}% goal environment, {btts}% BTTS pressure, "
        f"{over}% Over 2.5 frequency, and {booking}% booking heat. Treat this as supporting context rather than the primary read."
    )


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower().replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.replace("&", " and ")


def team_key(value: object) -> str:
    text = normalize_text(value)
    for token in (" fc ", " cf ", " sc ", " afc ", " club ", " vfl ", " sv ", " as "):
        text = text.replace(token, " ")
    return " ".join(text.split())


def build_payloads(features_root: Path) -> tuple[dict[str, dict], list[dict]]:
    payloads: dict[str, dict] = {}
    index_rows: list[dict] = []
    for csv_path in sorted(features_root.glob("api_h2h_regime_features__*.csv")):
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
            payload["summary"] = summary_for_row(payload)
            payloads[fixture_key] = payload
            index_rows.append(
                {
                    "fixture_key": fixture_key,
                    "fixture_id": payload["fixture_id"],
                    "competition": payload["competition"],
                    "season": payload["season"],
                    "sample_size": payload["sample_size"],
                }
            )
    return payloads, index_rows


def augment_with_fixture_feed(
    payloads: dict[str, dict],
    index_rows: list[dict],
    fixture_feed_path: Path | None,
) -> tuple[dict[str, dict], list[dict]]:
    if not fixture_feed_path or not fixture_feed_path.exists():
        return payloads, index_rows

    fixture_feed = json.loads(fixture_feed_path.read_text(encoding="utf-8"))
    fixtures = fixture_feed.get("fixtures") if isinstance(fixture_feed, dict) else None
    if not isinstance(fixtures, list):
        return payloads, index_rows

    by_pair: dict[tuple[str, str], list[dict]] = {}
    for payload in payloads.values():
        home = team_key(payload.get("home_team"))
        away = team_key(payload.get("away_team"))
        if not home or not away:
            continue
        by_pair.setdefault((home, away), []).append(payload)
        by_pair.setdefault((away, home), []).append(payload)

    existing_keys = set(payloads.keys())
    augmented = dict(payloads)
    augmented_index = list(index_rows)
    for fixture in fixtures:
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key or fixture_key in existing_keys:
            continue
        pair = (team_key(fixture.get("home_team")), team_key(fixture.get("away_team")))
        candidates = by_pair.get(pair) or []
        if not candidates:
            payload = {
                "fixture_key": fixture_key,
                "fixture_id": safe_int(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
                "competition": fixture.get("league"),
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
                "fallback_mode": "unpublished",
                "summary": "No publish-safe H2H regime summary has been produced for this fixture yet, so this layer stays supporting-only.",
            }
            augmented[fixture_key] = payload
            augmented_index.append(
                {
                    "fixture_key": fixture_key,
                    "fixture_id": payload["fixture_id"],
                    "competition": payload["competition"],
                    "season": payload["season"],
                    "sample_size": 0,
                    "fallback_mode": "unpublished",
                }
            )
            existing_keys.add(fixture_key)
            continue
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
                "season": str(fixture.get("api_season") or fixture.get("season") or source.get("season") or ""),
                "home_team": fixture.get("home_team") or source.get("home_team"),
                "away_team": fixture.get("away_team") or source.get("away_team"),
                "fallback_mode": "historical_team_pair",
                "source_fixture_key": source.get("fixture_key"),
            }
        )
        sample = int(payload.get("sample_size") or 0)
        payload["summary"] = (
            f"Using the latest available same-team-pair historical regime ({sample} prior meeting"
            f"{'' if sample == 1 else 's'}) as supporting context for this current fixture."
        )
        augmented[fixture_key] = payload
        augmented_index.append(
            {
                "fixture_key": fixture_key,
                "fixture_id": payload["fixture_id"],
                "competition": payload["competition"],
                "season": payload["season"],
                "sample_size": payload["sample_size"],
                "fallback_mode": "historical_team_pair",
            }
        )
    return augmented, augmented_index


def publish(payloads: dict[str, dict], index_rows: list[dict], output_root: Path) -> None:
    target_dir = output_root / "fixture_h2h_support"
    target_dir.mkdir(parents=True, exist_ok=True)
    for fixture_key, payload in payloads.items():
        (target_dir / f"{fixture_key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (target_dir / "index.json").write_text(json.dumps(index_rows, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish public-safe fixture H2H support JSON.")
    parser.add_argument("--features-root", default="data_sources/api_football/features")
    parser.add_argument("--fixture-feed", default=None, help="Optional fixture feed JSON to backfill current fixture keys.")
    parser.add_argument("--output-root", default="frontend/public/data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads, index_rows = build_payloads(Path(args.features_root))
    payloads, index_rows = augment_with_fixture_feed(payloads, index_rows, Path(args.fixture_feed) if args.fixture_feed else None)
    publish(payloads, index_rows, Path(args.output_root))
    print(f"Published {len(payloads)} fixture H2H support payloads.")


if __name__ == "__main__":
    main()
