from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.build_enriched_fixture_features import build_enriched_fixture_features
from scripts.api_football.build_event_features import build_event_features
from scripts.api_football.build_injury_features import build_injury_features
from scripts.api_football.build_lineup_features import build_lineup_features
from scripts.api_football.build_matchup_interaction_features import build_matchup_interaction_features
from scripts.api_football.build_odds_features import build_odds_features
from scripts.api_football.build_player_rolling_features import build_player_rolling_features
from scripts.api_football.build_referee_profile_features import build_referee_profile_features
from scripts.api_football.build_team_identity_features import build_team_identity_features
from scripts.api_football.build_team_rolling_features import build_team_rolling_features
from scripts.api_football.client import APIFootballClient
from scripts.api_football.fetch_current_player_window import (
    DEFAULT_LEAGUES,
    clean_status,
    fixture_ids,
    parse_csv_set,
    write_jsonl,
)
from scripts.api_football.normalize_fixtures_master import build_fixtures_master
from scripts.api_football.normalize_injuries import build_injuries
from scripts.api_football.normalize_lineups import build_lineups
from scripts.api_football.normalize_match_events import build_match_events
from scripts.api_football.normalize_match_player_stats import build_match_player_stats
from scripts.api_football.normalize_match_team_stats import build_match_team_stats
from scripts.api_football.normalize_odds_prematch_long import build_odds_prematch_long
from scripts.api_football.utils import chunk_list


DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "api_current_context_overlay_window"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fixture_item_index(bundle_payloads: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for payload in bundle_payloads:
        for item in payload.get("response", []) or []:
            fixture = item.get("fixture") or {}
            fixture_id = fixture.get("id")
            if fixture_id is not None:
                index[int(fixture_id)] = item
    return index


def _wrap_statistics_payload(base_item: dict[str, Any], statistics_payload: dict[str, Any]) -> dict[str, Any]:
    wrapped = {
        "fixture": deepcopy(base_item.get("fixture") or {}),
        "league": deepcopy(base_item.get("league") or {}),
        "teams": deepcopy(base_item.get("teams") or {}),
        "goals": deepcopy(base_item.get("goals") or {}),
        "score": deepcopy(base_item.get("score") or {}),
        "statistics": deepcopy(statistics_payload.get("response") or []),
    }
    return {"response": [wrapped]}


def _wrap_events_payload(base_item: dict[str, Any], events_payload: dict[str, Any]) -> dict[str, Any]:
    wrapped = {
        "fixture": deepcopy(base_item.get("fixture") or {}),
        "league": deepcopy(base_item.get("league") or {}),
        "teams": deepcopy(base_item.get("teams") or {}),
        "events": deepcopy(events_payload.get("response") or []),
    }
    return {"response": [wrapped]}


def _to_float(value: object) -> float:
    try:
        return float(str(value or "").strip() or 0.0)
    except ValueError:
        return 0.0


def build_current_overlay_summary(
    fixtures_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    injury_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
    team_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    referee_df: pd.DataFrame,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    player_fixture_ids = set()
    if not player_stats_df.empty and "fixture_id" in player_stats_df.columns:
        player_fixture_ids = {int(v) for v in player_stats_df["fixture_id"].dropna().tolist()}
    injury_index = {str(row["fixture_key"]): row for _, row in injury_df.iterrows()} if not injury_df.empty else {}
    lineup_index = {str(row["fixture_key"]): row for _, row in lineup_df.iterrows()} if not lineup_df.empty else {}
    team_index = {str(row["fixture_key"]): row for _, row in team_df.iterrows()} if not team_df.empty else {}
    matchup_index = {str(row["fixture_key"]): row for _, row in matchup_df.iterrows()} if not matchup_df.empty else {}
    odds_index = {str(row["fixture_key"]): row for _, row in odds_df.iterrows()} if not odds_df.empty else {}
    referee_index = {str(row["fixture_key"]): row for _, row in referee_df.iterrows()} if not referee_df.empty else {}

    rows: list[dict[str, Any]] = []
    for _, fx in fixtures_df.iterrows():
        fixture_key = str(fx["fixture_key"])
        injury = injury_index.get(fixture_key)
        lineup = lineup_index.get(fixture_key)
        team = team_index.get(fixture_key)
        matchup = matchup_index.get(fixture_key)
        odds = odds_index.get(fixture_key)
        referee = referee_index.get(fixture_key)
        fixture_id = int(fx["fixture_id"])

        injury_note = ""
        if injury is not None:
            home_abs = _to_float(injury.get("home_absence_severity_score"))
            away_abs = _to_float(injury.get("away_absence_severity_score"))
            if abs(home_abs - away_abs) >= 1.0:
                heavier = "home side" if home_abs > away_abs else "away side"
                injury_note = f"Current injury overlay shows the heavier absence burden on the {heavier}."
            elif max(home_abs, away_abs) >= 1.0:
                injury_note = "Current injury overlay shows notable availability drag around this fixture."

        lineup_note = ""
        if lineup is not None:
            attack_delta = _to_float(lineup.get("formation_attack_delta"))
            if abs(attack_delta) >= 0.75:
                stronger = "home side" if attack_delta > 0 else "away side"
                lineup_note = f"Current lineup overlay points to a more aggressive attacking shape from the {stronger}."

        form_note = ""
        if team is not None:
            ppg_delta = _to_float(team.get("ppg_diff_l5"))
            if abs(ppg_delta) >= 0.35:
                stronger = "home side" if ppg_delta > 0 else "away side"
                form_note = f"Current rolling team form remains stronger on the {stronger}."
            else:
                form_note = "Current rolling team form remains fairly balanced between both sides."

        style_note = ""
        if matchup is not None:
            press_gap = _to_float(matchup.get("press_mismatch_index"))
            goal_env = _to_float(matchup.get("goal_environment_interaction"))
            if goal_env >= 0.5:
                style_note = "Current matchup overlay points to a more active goal environment."
            elif press_gap >= 0.2:
                style_note = "Current matchup overlay shows a notable pressure-style mismatch."

        referee_note = ""
        if referee is not None:
            strictness = _to_float(referee.get("ref_strictness_score"))
            if strictness >= 0.7:
                referee_note = "Current referee overlay points to a stricter booking environment."

        availability = {
                "injuries": injury is not None,
                "lineups": lineup is not None,
                "team_stats": team is not None,
                "player_stats": fixture_id in player_fixture_ids,
                "match_events": matchup is not None or referee is not None,
                "prematch_odds": odds is not None,
                "current_overlay": any([injury is not None, lineup is not None, team is not None, matchup is not None, referee is not None]),
            }
        rows.append(
            {
                "fixture_id": fixture_id,
                "fixture_key": fixture_key,
                "league": fx["league"],
                "season": int(fx["season"]),
                "match_date": fx["match_date"],
                "home_team_name": fx["home_team_name"],
                "away_team_name": fx["away_team_name"],
                "availability": availability,
                "summary": {
                    "injury_note": injury_note,
                    "lineup_note": lineup_note,
                    "form_note": form_note,
                    "style_note": style_note,
                    "referee_note": referee_note,
                },
            }
        )

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_fixtures": len(rows),
        "fixtures": rows,
    }
    _write_json(output_json, payload)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return payload


def fetch_league_bundle(
    tag: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    client: APIFootballClient,
    raw_dir: Path,
    normalized_dir: Path,
    features_dir: Path,
) -> dict[str, Any]:
    league_id = int(cfg["league_id"])
    season = int(cfg["season"])
    from_date = args.from_date or str(cfg["start"])
    to_date = args.to_date
    stem = f"{tag}__league_{league_id}__season_{season}__{from_date}_to_{to_date}"

    params = {
        "league": league_id,
        "season": season,
        "from": from_date,
        "to": to_date,
        "status": clean_status(args.status),
        "timezone": args.timezone,
    }
    fixtures_payload = client.get_json("/fixtures", params)
    ids = fixture_ids(fixtures_payload)
    if args.max_fixtures_per_league and len(ids) > args.max_fixtures_per_league:
        ids = ids[: args.max_fixtures_per_league]

    fixtures_raw = raw_dir / f"{stem}__fixtures.jsonl"
    write_jsonl(fixtures_raw, [fixtures_payload])

    bundle_payloads: list[dict[str, Any]] = []
    for fixture_chunk in chunk_list(ids, max(1, min(int(args.chunk_size), 20))):
        ids_param = "-".join(str(fid) for fid in fixture_chunk)
        bundle_payloads.append(client.get_json("/fixtures", {"ids": ids_param}))
    bundle_raw = raw_dir / f"{stem}__fixtures_bundle.jsonl"
    write_jsonl(bundle_raw, bundle_payloads)
    bundle_index = _fixture_item_index(bundle_payloads)

    stats_wrapped: list[dict[str, Any]] = []
    events_wrapped: list[dict[str, Any]] = []
    injuries_payloads: list[dict[str, Any]] = []
    odds_payloads: list[dict[str, Any]] = []
    for fixture_id in ids:
        base_item = bundle_index.get(int(fixture_id))
        if base_item is None:
            continue
        stats_payload = client.get_json("/fixtures/statistics", {"fixture": fixture_id})
        stats_wrapped.append(_wrap_statistics_payload(base_item, stats_payload))
        events_payload = client.get_json("/fixtures/events", {"fixture": fixture_id})
        events_wrapped.append(_wrap_events_payload(base_item, events_payload))
        injuries_payloads.append(client.get_json("/injuries", {"fixture": fixture_id}))
        odds_payloads.extend(client.paged_get("/odds", {"fixture": fixture_id}, max_pages=args.max_pages_per_fixture))

    stats_raw = raw_dir / f"{stem}__fixtures_statistics_wrapped.jsonl"
    events_raw = raw_dir / f"{stem}__fixtures_events_wrapped.jsonl"
    injuries_raw = raw_dir / f"{stem}__injuries.jsonl"
    odds_raw = raw_dir / f"{stem}__odds_prematch.jsonl"
    write_jsonl(stats_raw, stats_wrapped)
    write_jsonl(events_raw, events_wrapped)
    write_jsonl(injuries_raw, injuries_payloads)
    write_jsonl(odds_raw, odds_payloads)

    fixtures_csv = normalized_dir / f"fixtures_master__{tag}__{season}.csv"
    lineups_csv = normalized_dir / f"lineups__{tag}__{season}.csv"
    player_stats_csv = normalized_dir / f"match_player_stats__{tag}__{season}.csv"
    team_stats_csv = normalized_dir / f"match_team_stats__{tag}__{season}.csv"
    events_csv = normalized_dir / f"match_events__{tag}__{season}.csv"
    injuries_csv = normalized_dir / f"injuries__{tag}__{season}.csv"
    odds_csv = normalized_dir / f"odds_prematch_long__{tag}__{season}.csv"

    fixtures_df = build_fixtures_master(str(fixtures_raw), str(fixtures_csv))
    lineups_df = build_lineups(str(bundle_raw), str(lineups_csv))
    player_stats_df = build_match_player_stats(str(bundle_raw), str(player_stats_csv))
    team_stats_df = build_match_team_stats(str(stats_raw), str(team_stats_csv))
    events_df = build_match_events(str(events_raw), str(events_csv))
    injuries_df = build_injuries(str(injuries_raw), str(injuries_csv))
    odds_df = build_odds_prematch_long(str(odds_raw), str(odds_csv))

    team_features_csv = features_dir / f"api_team_rolling_features__{tag}__{season}.csv"
    player_features_csv = features_dir / f"api_player_rolling_features__{tag}__{season}.csv"
    lineup_features_csv = features_dir / f"api_lineup_features__{tag}__{season}.csv"
    injury_features_csv = features_dir / f"api_injury_features__{tag}__{season}.csv"
    event_features_csv = features_dir / f"api_event_features__{tag}__{season}.csv"
    odds_features_csv = features_dir / f"api_odds_features__{tag}__{season}.csv"
    enriched_csv = features_dir / f"api_enriched_fixture_features__{tag}__{season}.csv"
    identity_csv = features_dir / f"api_team_identity_features__{tag}__{season}.csv"
    matchup_csv = features_dir / f"api_matchup_interaction_features__{tag}__{season}.csv"
    referee_csv = features_dir / f"api_referee_profile_features__{tag}__{season}.csv"

    team_features_df = build_team_rolling_features(str(fixtures_csv), str(team_stats_csv), str(team_features_csv))
    player_features_df = build_player_rolling_features(str(fixtures_csv), str(player_stats_csv), str(player_features_csv))
    lineup_features_df = build_lineup_features(str(fixtures_csv), str(lineups_csv), str(player_stats_csv), str(lineup_features_csv))
    injury_features_df = build_injury_features(str(fixtures_csv), str(injuries_csv), str(player_stats_csv), str(injury_features_csv))
    event_features_df = build_event_features(str(fixtures_csv), str(events_csv), str(team_stats_csv), str(event_features_csv))
    odds_features_df = build_odds_features(str(fixtures_csv), str(odds_csv), str(odds_features_csv))
    enriched_df = build_enriched_fixture_features(
        str(fixtures_csv),
        str(team_features_csv),
        str(event_features_csv),
        str(lineup_features_csv),
        str(injury_features_csv),
        str(odds_features_csv),
        str(enriched_csv),
    )
    identity_df = build_team_identity_features(str(enriched_csv), str(identity_csv))
    matchup_df = build_matchup_interaction_features(str(identity_csv), str(enriched_csv), str(matchup_csv))
    referee_df = build_referee_profile_features(str(fixtures_csv), str(team_stats_csv), str(events_csv), str(enriched_csv), str(referee_csv))

    return {
        "tag": tag,
        "league_id": league_id,
        "season": season,
        "from_date": from_date,
        "to_date": to_date,
        "fixture_ids": len(ids),
        "paths": {
            "fixtures_raw": str(fixtures_raw),
            "bundle_raw": str(bundle_raw),
            "stats_raw": str(stats_raw),
            "events_raw": str(events_raw),
            "injuries_raw": str(injuries_raw),
            "odds_raw": str(odds_raw),
            "fixtures_csv": str(fixtures_csv),
            "lineups_csv": str(lineups_csv),
            "player_stats_csv": str(player_stats_csv),
            "team_stats_csv": str(team_stats_csv),
            "events_csv": str(events_csv),
            "injuries_csv": str(injuries_csv),
            "odds_csv": str(odds_csv),
            "team_features_csv": str(team_features_csv),
            "player_features_csv": str(player_features_csv),
            "lineup_features_csv": str(lineup_features_csv),
            "injury_features_csv": str(injury_features_csv),
            "event_features_csv": str(event_features_csv),
            "odds_features_csv": str(odds_features_csv),
            "enriched_csv": str(enriched_csv),
            "identity_csv": str(identity_csv),
            "matchup_csv": str(matchup_csv),
            "referee_csv": str(referee_csv),
        },
        "counts": {
            "fixtures_rows": len(fixtures_df),
            "lineup_rows": len(lineups_df),
            "player_stat_rows": len(player_stats_df),
            "team_stat_rows": len(team_stats_df),
            "event_rows": len(events_df),
            "injury_rows": len(injuries_df),
            "odds_rows": len(odds_df),
            "team_feature_rows": len(team_features_df),
            "player_feature_rows": len(player_features_df),
            "lineup_feature_rows": len(lineup_features_df),
            "injury_feature_rows": len(injury_features_df),
            "event_feature_rows": len(event_features_df),
            "odds_feature_rows": len(odds_features_df),
            "enriched_rows": len(enriched_df),
            "identity_rows": len(identity_df),
            "matchup_rows": len(matchup_df),
            "referee_rows": len(referee_df),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and build a current-window API-Football overlay bundle for CONTEXT/MONITOR enrichment "
            "without touching production normalized season files."
        )
    )
    parser.add_argument("--league-tags", default="", help="Comma-separated tags. Defaults to the 14 board leagues.")
    parser.add_argument("--from-date", default="", help="Override lower bound YYYY-MM-DD for every league.")
    parser.add_argument("--to-date", default=date.today().isoformat(), help="Upper bound YYYY-MM-DD.")
    parser.add_argument("--status", default="", help="Fixture status filter. Empty string fetches all statuses in the date window.")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-fixtures-per-league", type=int, default=0)
    parser.add_argument("--max-pages-per-fixture", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = parse_csv_set(args.league_tags) if args.league_tags else set(DEFAULT_LEAGUES)
    unknown = selected - set(DEFAULT_LEAGUES)
    if unknown:
        raise SystemExit(f"Unknown league tags: {sorted(unknown)}")

    raw_dir = args.outdir / "raw"
    normalized_dir = args.outdir / "normalized"
    features_dir = args.outdir / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    bundles: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    for tag in sorted(selected):
        bundle = fetch_league_bundle(tag, DEFAULT_LEAGUES[tag], args, client, raw_dir, normalized_dir, features_dir)
        bundles.append(bundle)
        summary_frames.append(
            pd.DataFrame(
                [
                    {
                        "league_tag": bundle["tag"],
                        "league_id": bundle["league_id"],
                        "season": bundle["season"],
                        **bundle["counts"],
                    }
                ]
            )
        )

    all_fixtures = []
    all_injuries = []
    all_lineups = []
    all_team_features = []
    all_matchups = []
    all_referees = []
    for bundle in bundles:
        paths = bundle["paths"]
        all_fixtures.append(pd.read_csv(paths["fixtures_csv"]))
        all_injuries.append(pd.read_csv(paths["injury_features_csv"]))
        all_lineups.append(pd.read_csv(paths["lineup_features_csv"]))
        all_team_features.append(pd.read_csv(paths["team_features_csv"]))
        all_matchups.append(pd.read_csv(paths["matchup_csv"]))
        all_referees.append(pd.read_csv(paths["referee_csv"]))

    fixtures_df = pd.concat(all_fixtures, ignore_index=True) if all_fixtures else pd.DataFrame()
    injuries_df = pd.concat(all_injuries, ignore_index=True) if all_injuries else pd.DataFrame()
    lineups_df = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
    team_df = pd.concat(all_team_features, ignore_index=True) if all_team_features else pd.DataFrame()
    matchup_df = pd.concat(all_matchups, ignore_index=True) if all_matchups else pd.DataFrame()
    odds_df = pd.concat([pd.read_csv(bundle["paths"]["odds_features_csv"]) for bundle in bundles], ignore_index=True) if bundles else pd.DataFrame()
    player_stats_df = pd.concat([pd.read_csv(bundle["paths"]["player_stats_csv"]) for bundle in bundles], ignore_index=True) if bundles else pd.DataFrame()
    referee_df = pd.concat(all_referees, ignore_index=True) if all_referees else pd.DataFrame()

    summary_csv = args.outdir / "CURRENT_CONTEXT_OVERLAY_BUILD_SUMMARY.csv"
    build_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    build_summary.to_csv(summary_csv, index=False)

    summary_json = args.outdir / "CURRENT_CONTEXT_OVERLAY_SUMMARY.json"
    summary_records_csv = args.outdir / "CURRENT_CONTEXT_OVERLAY_SUMMARY.csv"
    overlay_summary = build_current_overlay_summary(
        fixtures_df,
        player_stats_df,
        injuries_df,
        lineups_df,
        team_df,
        matchup_df,
        odds_df,
        referee_df,
        summary_json,
        summary_records_csv,
    )

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "outdir": str(args.outdir),
        "league_tags": sorted(selected),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_records_csv": str(summary_records_csv),
        "bundle_count": len(bundles),
        "total_fixtures": int(overlay_summary.get("total_fixtures", 0)),
        "bundles": bundles,
    }
    manifest_path = args.outdir / "CURRENT_CONTEXT_OVERLAY_MANIFEST.json"
    _write_json(manifest_path, manifest)

    print(f"WROTE {args.outdir}")
    print(f"Manifest: {manifest_path}")
    print(f"Overlay summary: {summary_json}")
    if not build_summary.empty:
        print(build_summary.to_string(index=False))


if __name__ == "__main__":
    main()
