#!/usr/bin/env python3
"""Build canonical referee profiles and fixture overlays for player-event beta.

Research/beta only. This standardizes the older referee profile scripts into a
single builder that can run over one or more league/season normalized API-
Football files. It does not alter deploy tiers, slips, or production routing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_FEATURE_DIR = ROOT / "data_sources" / "api_football" / "features" / "referee_profiles"


def _num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def _safe_div(num: float, den: float) -> float:
    return 0.0 if not den else float(num) / float(den)


def _mean(records: list[dict[str, float]], key: str, n: int) -> float:
    sample = records[:n]
    return float(np.mean([float(row.get(key, 0.0) or 0.0) for row in sample])) if sample else 0.0


def _sum(records: list[dict[str, float]], key: str, n: int) -> float:
    sample = records[:n]
    return float(np.sum([float(row.get(key, 0.0) or 0.0) for row in sample])) if sample else 0.0


def _band(score: float) -> str:
    if score >= 0.75:
        return "STRICT"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "UNKNOWN"


def _confidence(sample: int) -> str:
    if sample >= 20:
        return "HIGH_SAMPLE"
    if sample >= 10:
        return "MEDIUM_SAMPLE"
    if sample >= 5:
        return "LOW_SAMPLE"
    if sample > 0:
        return "TINY_SAMPLE"
    return "NO_SAMPLE"


def _slug_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _default_path(name: str, league_tag: str, season: int) -> Path:
    return NORMALIZED_DIR / f"{name}__{league_tag}__{season}.csv"


def load_league_seasons(league_tag: str, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures_frames: list[pd.DataFrame] = []
    stats_frames: list[pd.DataFrame] = []
    events_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for season in seasons:
        fixtures_path = _default_path("fixtures_master", league_tag, season)
        stats_path = _default_path("match_team_stats", league_tag, season)
        events_path = _default_path("match_events", league_tag, season)

        fixtures = _read_csv(fixtures_path)
        stats = _read_csv(stats_path)
        events = _read_csv(events_path)

        if not fixtures.empty:
            fixtures["source_league_tag"] = league_tag
            fixtures["source_season"] = season
            fixtures_frames.append(fixtures)
        if not stats.empty:
            stats["source_league_tag"] = league_tag
            stats["source_season"] = season
            stats_frames.append(stats)
        if not events.empty:
            events["source_league_tag"] = league_tag
            events["source_season"] = season
            events_frames.append(events)

        audit_rows.append(
            {
                "league_tag": league_tag,
                "season": season,
                "fixtures_path": str(fixtures_path),
                "fixtures_rows": len(fixtures),
                "team_stats_path": str(stats_path),
                "team_stats_rows": len(stats),
                "events_path": str(events_path),
                "event_rows": len(events),
            }
        )

    fixtures_all = pd.concat(fixtures_frames, ignore_index=True, sort=False) if fixtures_frames else pd.DataFrame()
    stats_all = pd.concat(stats_frames, ignore_index=True, sort=False) if stats_frames else pd.DataFrame()
    events_all = pd.concat(events_frames, ignore_index=True, sort=False) if events_frames else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    return fixtures_all, stats_all, events_all, audit


def fixture_totals(team_stats: pd.DataFrame) -> pd.DataFrame:
    if team_stats.empty:
        return pd.DataFrame(columns=["fixture_id"])
    work = team_stats.copy()
    for col in ["fouls_for", "yellow_cards", "red_cards", "is_home"]:
        if col not in work.columns:
            work[col] = 0
    work["fouls_for"] = _num(work["fouls_for"])
    work["yellow_cards"] = _num(work["yellow_cards"])
    work["red_cards"] = _num(work["red_cards"])
    work["is_home"] = _num(work["is_home"]).astype(int)
    grouped = work.groupby("fixture_id", as_index=False).agg(
        total_fouls=("fouls_for", "sum"),
        total_yellows=("yellow_cards", "sum"),
        total_reds=("red_cards", "sum"),
    )
    home = (
        work[work["is_home"].eq(1)]
        .groupby("fixture_id", as_index=False)
        .agg(home_cards=("yellow_cards", "sum"), home_fouls=("fouls_for", "sum"))
    )
    away = (
        work[work["is_home"].eq(0)]
        .groupby("fixture_id", as_index=False)
        .agg(away_cards=("yellow_cards", "sum"), away_fouls=("fouls_for", "sum"))
    )
    out = grouped.merge(home, on="fixture_id", how="left").merge(away, on="fixture_id", how="left")
    for col in ["home_cards", "home_fouls", "away_cards", "away_fouls"]:
        out[col] = _num(out[col])
    out["total_cards"] = out["total_yellows"] + out["total_reds"]
    return out


def event_totals(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["fixture_id"])
    work = events.copy()
    for col in ["event_type", "event_detail", "minute"]:
        if col not in work.columns:
            work[col] = ""
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    event_detail = work["event_detail"].fillna("").astype(str).str.lower()
    minute = _num(work["minute"])

    cards = work[event_type.eq("card") | event_detail.str.contains("yellow|red", regex=True, na=False)].copy()
    cards["minute_num"] = _num(cards.get("minute", pd.Series(dtype=float)))
    card_summary = cards.groupby("fixture_id", as_index=False).agg(
        raw_card_events=("fixture_id", "count"),
        first_half_card_events=("minute_num", lambda s: int((s <= 45).sum())),
        second_half_card_events=("minute_num", lambda s: int((s > 45).sum())),
        late_card_events=("minute_num", lambda s: int((s >= 75).sum())),
    )

    penalties = work[
        event_type.str.contains("penalty|var|goal", regex=True, na=False)
        & event_detail.str.contains("penalty", regex=False, na=False)
    ].copy()
    penalty_summary = penalties.groupby("fixture_id").size().rename("penalty_events").reset_index()
    out = card_summary.merge(penalty_summary, on="fixture_id", how="outer")
    for col in ["raw_card_events", "first_half_card_events", "second_half_card_events", "late_card_events", "penalty_events"]:
        out[col] = _num(out[col])
    return out


def score_from_history(prev: list[dict[str, float]], history_window: int) -> dict[str, Any]:
    sample = min(len(prev), history_window)
    total_cards = _sum(prev, "total_cards", history_window)
    total_fouls = _sum(prev, "total_fouls", history_window)
    cards_per_match = _mean(prev, "total_cards", history_window)
    yellows_per_match = _mean(prev, "total_yellows", history_window)
    reds_per_match = _mean(prev, "total_reds", history_window)
    fouls_per_match = _mean(prev, "total_fouls", history_window)
    cards_per_foul = _safe_div(total_cards, total_fouls)
    foul_to_card_ratio = _safe_div(total_fouls, total_cards)
    late_cards = _mean(prev, "late_card_events", history_window)
    first_half_cards = _mean(prev, "first_half_card_events", history_window)
    second_half_cards = _mean(prev, "second_half_card_events", history_window)
    penalties = _mean(prev, "penalty_events", history_window)
    home_cards = _mean(prev, "home_cards", history_window)
    away_cards = _mean(prev, "away_cards", history_window)
    home_card_tilt = _safe_div(home_cards, max(away_cards, 1e-9))
    away_card_tilt = _safe_div(away_cards, max(home_cards, 1e-9))

    strictness = (
        0.35 * min(cards_per_match / 6.0, 1.0)
        + 0.25 * min(fouls_per_match / 30.0, 1.0)
        + 0.20 * min(cards_per_foul / 0.30, 1.0)
        + 0.10 * min(late_cards / 1.5, 1.0)
        + 0.10 * min(reds_per_match / 0.30, 1.0)
    ) if sample else 0.0
    open_play_tolerance = (
        1.0 - min(cards_per_foul / 0.30, 1.0)
    ) * min(fouls_per_match / 30.0, 1.0) if sample else 0.0
    tactical_foul_punishment = (
        0.55 * min(cards_per_foul / 0.30, 1.0)
        + 0.25 * min(second_half_cards / 3.5, 1.0)
        + 0.20 * min(late_cards / 1.5, 1.0)
    ) if sample else 0.0

    return {
        "sample_matches": sample,
        "cards_per_match_l20": round(cards_per_match, 4),
        "yellows_per_match_l20": round(yellows_per_match, 4),
        "reds_per_match_l20": round(reds_per_match, 4),
        "fouls_per_match_l20": round(fouls_per_match, 4),
        "foul_to_card_ratio_l20": round(foul_to_card_ratio, 4),
        "cards_per_foul_l20": round(cards_per_foul, 4),
        "late_cards_per_match_l20": round(late_cards, 4),
        "first_half_cards_per_match_l20": round(first_half_cards, 4),
        "second_half_cards_per_match_l20": round(second_half_cards, 4),
        "penalties_per_match_l20": round(penalties, 4),
        "home_cards_per_match_l20": round(home_cards, 4),
        "away_cards_per_match_l20": round(away_cards, 4),
        "home_card_tilt": round(home_card_tilt, 4),
        "away_card_tilt": round(away_card_tilt, 4),
        "open_play_foul_tolerance_score": round(open_play_tolerance, 4),
        "tactical_foul_punishment_score": round(tactical_foul_punishment, 4),
        "dissent_strictness_score": round(strictness, 4),
        "timewasting_strictness_score": round(min(1.0, late_cards / 1.5), 4) if sample else 0.0,
        "strictness_score": round(strictness, 4),
        "strictness_band": _band(strictness),
        "profile_confidence": _confidence(sample),
    }


def build_profiles(
    league_tags: list[str],
    seasons: list[int],
    outdir: Path,
    history_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    all_profile_rows: list[dict[str, Any]] = []
    all_overlay_rows: list[dict[str, Any]] = []
    load_audits: list[pd.DataFrame] = []

    for league_tag in league_tags:
        fixtures, team_stats, events, load_audit = load_league_seasons(league_tag, seasons)
        load_audits.append(load_audit)
        if fixtures.empty:
            continue

        fx_totals = fixture_totals(team_stats)
        ev_totals = event_totals(events)
        work = fixtures.merge(fx_totals, on="fixture_id", how="left").merge(ev_totals, on="fixture_id", how="left")
        for col in [
            "total_fouls",
            "total_yellows",
            "total_reds",
            "total_cards",
            "home_cards",
            "away_cards",
            "home_fouls",
            "away_fouls",
            "first_half_card_events",
            "second_half_card_events",
            "late_card_events",
            "penalty_events",
        ]:
            if col not in work.columns:
                work[col] = 0.0
            work[col] = _num(work[col])
        work["kickoff_ts_utc"] = pd.to_datetime(work.get("kickoff_ts_utc"), errors="coerce", utc=True)
        work = work.sort_values(["kickoff_ts_utc", "fixture_id"]).reset_index(drop=True)

        history: dict[str, list[dict[str, float]]] = defaultdict(list)
        latest_by_ref: dict[str, dict[str, Any]] = {}

        for _, row in work.iterrows():
            referee_name = str(row.get("referee_name", "") or "").strip()
            prev = list(reversed(history.get(referee_name, []))) if referee_name else []
            prof = score_from_history(prev, history_window)
            fixture_profile = {
                "fixture_id": row.get("fixture_id"),
                "fixture_key": row.get("fixture_key", ""),
                "league": row.get("league", ""),
                "league_tag": league_tag,
                "league_id": row.get("league_id", ""),
                "season": row.get("source_season", row.get("season", "")),
                "match_date": row.get("match_date", ""),
                "home_team_id": row.get("home_team_id", ""),
                "away_team_id": row.get("away_team_id", ""),
                "home_team_name": row.get("home_team_name", ""),
                "away_team_name": row.get("away_team_name", ""),
                "kickoff_ts_utc": row.get("kickoff_ts_utc", ""),
                "referee_name": referee_name,
                **prof,
            }
            fixture_profile.update(
                {
                    "expected_total_cards_ref_adjusted": round(prof["cards_per_match_l20"], 3),
                    "expected_total_fouls_ref_adjusted": round(prof["fouls_per_match_l20"], 3),
                    "expected_first_half_cards_ref_adjusted": round(prof["first_half_cards_per_match_l20"], 3),
                    "expected_second_half_cards_ref_adjusted": round(prof["second_half_cards_per_match_l20"], 3),
                    "late_card_risk_flag": int(prof["late_cards_per_match_l20"] >= 1.0 and prof["sample_matches"] >= 5),
                    "penalty_risk_flag": int(prof["penalties_per_match_l20"] >= 0.35 and prof["sample_matches"] >= 5),
                    "open_play_allowed_flag": int(prof["open_play_foul_tolerance_score"] >= 0.45 and prof["sample_matches"] >= 5),
                    "card_market_live_flag": int(prof["strictness_score"] >= 0.55 and prof["sample_matches"] >= 5),
                    "fouls_market_live_flag": int(prof["fouls_per_match_l20"] >= 24 and prof["sample_matches"] >= 5),
                    "bookings_player_event_multiplier": round(0.85 + (0.30 * prof["strictness_score"]), 4) if prof["sample_matches"] else 1.0,
                }
            )
            all_overlay_rows.append(fixture_profile)
            if referee_name:
                latest_by_ref[referee_name] = {
                    "league_tag": league_tag,
                    "league": row.get("league", ""),
                    "season_window": "_".join(str(season) for season in seasons),
                    "referee_name": referee_name,
                    "last_seen_match_date": row.get("match_date", ""),
                    **prof,
                }
                history[referee_name].append(
                    {
                        "total_cards": float(row.get("total_cards", 0.0) or 0.0),
                        "total_yellows": float(row.get("total_yellows", 0.0) or 0.0),
                        "total_reds": float(row.get("total_reds", 0.0) or 0.0),
                        "total_fouls": float(row.get("total_fouls", 0.0) or 0.0),
                        "home_cards": float(row.get("home_cards", 0.0) or 0.0),
                        "away_cards": float(row.get("away_cards", 0.0) or 0.0),
                        "first_half_card_events": float(row.get("first_half_card_events", 0.0) or 0.0),
                        "second_half_card_events": float(row.get("second_half_card_events", 0.0) or 0.0),
                        "late_card_events": float(row.get("late_card_events", 0.0) or 0.0),
                        "penalty_events": float(row.get("penalty_events", 0.0) or 0.0),
                    }
                )

        all_profile_rows.extend(latest_by_ref.values())

    profiles = pd.DataFrame(all_profile_rows)
    overlay = pd.DataFrame(all_overlay_rows)
    audit = pd.concat(load_audits, ignore_index=True, sort=False) if load_audits else pd.DataFrame()
    if not profiles.empty:
        profiles = profiles.sort_values(["league_tag", "referee_name"]).reset_index(drop=True)
    if not overlay.empty:
        overlay = overlay.sort_values(["league_tag", "kickoff_ts_utc", "fixture_id"]).reset_index(drop=True)
    return profiles, overlay, audit


def _current_fixture_overlay(current_fixtures_csvs: str, profiles: pd.DataFrame, league_tags: list[str]) -> pd.DataFrame:
    if not current_fixtures_csvs.strip():
        return pd.DataFrame()
    frames = []
    for item in _slug_list(current_fixtures_csvs):
        path = Path(item)
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    fixtures = pd.concat(frames, ignore_index=True, sort=False)
    if fixtures.empty:
        return pd.DataFrame()
    fixtures["referee_name"] = fixtures.get("referee_name", pd.Series("", index=fixtures.index)).fillna("").astype(str).str.strip()
    profile_cols = [
        "league_tag",
        "referee_name",
        "sample_matches",
        "cards_per_match_l20",
        "yellows_per_match_l20",
        "reds_per_match_l20",
        "fouls_per_match_l20",
        "foul_to_card_ratio_l20",
        "cards_per_foul_l20",
        "late_cards_per_match_l20",
        "first_half_cards_per_match_l20",
        "second_half_cards_per_match_l20",
        "penalties_per_match_l20",
        "home_cards_per_match_l20",
        "away_cards_per_match_l20",
        "home_card_tilt",
        "away_card_tilt",
        "open_play_foul_tolerance_score",
        "tactical_foul_punishment_score",
        "dissent_strictness_score",
        "timewasting_strictness_score",
        "strictness_score",
        "strictness_band",
        "profile_confidence",
    ]
    profile = profiles[[col for col in profile_cols if col in profiles.columns]].copy() if not profiles.empty else pd.DataFrame()
    if "source_league_tag" in fixtures.columns:
        fixtures["league_tag"] = fixtures["source_league_tag"]
    elif len(league_tags) == 1:
        fixtures["league_tag"] = league_tags[0]
    else:
        fixtures["league_tag"] = ""
    out = fixtures.merge(profile, on=["league_tag", "referee_name"], how="left")
    defaults = score_from_history([], 20)
    for key, value in defaults.items():
        if key not in out.columns:
            out[key] = value
        out[key] = out[key].fillna(value)
    out["expected_total_cards_ref_adjusted"] = pd.to_numeric(out["cards_per_match_l20"], errors="coerce").fillna(0.0).round(3)
    out["expected_total_fouls_ref_adjusted"] = pd.to_numeric(out["fouls_per_match_l20"], errors="coerce").fillna(0.0).round(3)
    out["expected_first_half_cards_ref_adjusted"] = pd.to_numeric(out["first_half_cards_per_match_l20"], errors="coerce").fillna(0.0).round(3)
    out["expected_second_half_cards_ref_adjusted"] = pd.to_numeric(out["second_half_cards_per_match_l20"], errors="coerce").fillna(0.0).round(3)
    out["late_card_risk_flag"] = (
        pd.to_numeric(out["late_cards_per_match_l20"], errors="coerce").fillna(0.0).ge(1.0)
        & pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).ge(5)
    ).astype(int)
    out["penalty_risk_flag"] = (
        pd.to_numeric(out["penalties_per_match_l20"], errors="coerce").fillna(0.0).ge(0.35)
        & pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).ge(5)
    ).astype(int)
    out["open_play_allowed_flag"] = (
        pd.to_numeric(out["open_play_foul_tolerance_score"], errors="coerce").fillna(0.0).ge(0.45)
        & pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).ge(5)
    ).astype(int)
    out["card_market_live_flag"] = (
        pd.to_numeric(out["strictness_score"], errors="coerce").fillna(0.0).ge(0.55)
        & pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).ge(5)
    ).astype(int)
    out["fouls_market_live_flag"] = (
        pd.to_numeric(out["fouls_per_match_l20"], errors="coerce").fillna(0.0).ge(24.0)
        & pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).ge(5)
    ).astype(int)
    out["bookings_player_event_multiplier"] = (
        0.85 + (0.30 * pd.to_numeric(out["strictness_score"], errors="coerce").fillna(0.0))
    ).round(4)
    out.loc[pd.to_numeric(out["sample_matches"], errors="coerce").fillna(0).eq(0), "bookings_player_event_multiplier"] = 1.0
    return out


def write_outputs(
    profiles: pd.DataFrame,
    overlay: pd.DataFrame,
    audit: pd.DataFrame,
    outdir: Path,
    league_tags: list[str],
    seasons: list[int],
    current_fixtures_csvs: str = "",
) -> dict[str, Path]:
    slug = "__".join(league_tags)
    season_slug = "_".join(str(season) for season in seasons)
    paths = {
        "profiles": outdir / f"referee_profiles__{slug}__{season_slug}.csv",
        "overlay": outdir / f"referee_fixture_overlay__{slug}__{season_slug}.csv",
        "current_overlay": outdir / f"referee_current_fixture_overlay__{slug}__{season_slug}.csv",
        "audit": outdir / f"referee_profile_engine_load_audit__{slug}__{season_slug}.csv",
        "markdown": outdir / f"REFEREE_PROFILE_ENGINE_SUMMARY__{slug}__{season_slug}.md",
    }
    current_overlay = _current_fixture_overlay(current_fixtures_csvs, profiles, league_tags)
    profiles.to_csv(paths["profiles"], index=False)
    overlay.to_csv(paths["overlay"], index=False)
    current_overlay.to_csv(paths["current_overlay"], index=False)
    audit.to_csv(paths["audit"], index=False)

    lines = [
        "# Referee Profile Engine Summary",
        "",
        "Research/beta support layer. No production deploy routing changed.",
        "",
        "## Counts",
        f"- league_tags: `{','.join(league_tags)}`",
        f"- seasons: `{','.join(str(season) for season in seasons)}`",
        f"- referee_profiles: `{len(profiles)}`",
        f"- fixture_overlay_rows: `{len(overlay)}`",
        f"- current_fixture_overlay_rows: `{len(current_overlay)}`",
        "",
        "## Confidence",
    ]
    if not profiles.empty:
        counts = profiles["profile_confidence"].value_counts().to_dict()
        lines.append(f"- profile_confidence_counts: `{counts}`")
        top = profiles.sort_values(["strictness_score", "sample_matches"], ascending=[False, False]).head(20)
        cols = ["league_tag", "referee_name", "sample_matches", "cards_per_match_l20", "fouls_per_match_l20", "strictness_score", "strictness_band", "profile_confidence"]
        lines.append("")
        lines.append("## Strictest Referees")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in top.iterrows():
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    else:
        lines.append("- No referee profile rows built.")
    lines += [
        "",
        "## Outputs",
        f"- profiles: `{paths['profiles']}`",
        f"- overlay: `{paths['overlay']}`",
        f"- current overlay: `{paths['current_overlay']}`",
        f"- audit: `{paths['audit']}`",
    ]
    paths["markdown"].write_text("\n".join(lines) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-tags", required=True, help="Comma-separated league tags, e.g. England_Premier_League")
    parser.add_argument("--seasons", required=True, help="Comma-separated seasons, e.g. 2024,2025")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--history-window", type=int, default=20)
    parser.add_argument(
        "--current-fixtures-csvs",
        default="",
        help="Optional comma-separated current fixtures CSVs to enrich using latest referee profiles.",
    )
    args = parser.parse_args()

    league_tags = _slug_list(args.league_tags)
    seasons = [int(value) for value in _slug_list(args.seasons)]
    profiles, overlay, audit = build_profiles(league_tags, seasons, args.outdir, args.history_window)
    paths = write_outputs(profiles, overlay, audit, args.outdir, league_tags, seasons, args.current_fixtures_csvs)
    print(f"WROTE {paths['profiles']}")
    print(f"WROTE {paths['overlay']}")
    print(f"WROTE {paths['current_overlay']}")
    print(f"WROTE {paths['audit']}")
    print(f"profiles={len(profiles)} overlay_rows={len(overlay)}")


if __name__ == "__main__":
    main()
