#!/usr/bin/env python3
"""Publish a canonical fixture decision intelligence layer."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_ROOT_DEFAULT = Path("frontend/public/data")

TEAM_RATING_LABELS = {
    "og_power_rating": "OG Power Rating",
    "attack_flow_rating": "Attack Flow",
    "defensive_lock_rating": "Defensive Lock",
    "goal_heat_rating": "Goal Heat",
    "btts_pressure_rating": "BTTS Pressure",
    "over25_heat_rating": "Over 2.5 Heat",
    "control_rating": "Control Rating",
    "first_strike_rating": "First Strike",
    "corner_pressure_rating": "Corner Pressure",
    "card_heat_rating": "Card Heat",
    "chaos_rating": "Chaos Rating",
    "home_fortress_rating": "Home Fortress",
    "away_threat_rating": "Away Threat",
}

REASON_LABELS = {
    "TEAM_POWER_ADVANTAGE": "Team power advantage",
    "TEAM_POWER_NOT_DECISIVE": "Team power not decisive",
    "HOME_FORTRESS_ADVANTAGE": "Home fortress advantage",
    "AWAY_THREAT_ADVANTAGE": "Away threat advantage",
    "AWAY_TRAVEL_THREAT": "Away travel threat",
    "HOME_RESISTANCE_HOLDS": "Home resistance holds",
    "POWER_PARITY": "Power parity",
    "POWER_IMBALANCE": "Power imbalance",
    "CONTROL_SUPPORT": "Control support",
    "BTTS_PRESSURE_SUPPORT": "BTTS pressure support",
    "BTTS_PRESSURE_WEAK": "BTTS pressure weak",
    "BTTS_SUPPRESSION_SUPPORT": "BTTS suppression support",
    "BTTS_ACCESS_RISK": "BTTS access risk",
    "GOAL_ENVIRONMENT_SUPPORT": "Goal environment support",
    "GOAL_ENVIRONMENT_SOFT": "Goal environment soft",
    "DEFENSIVE_SUPPRESSION_RISK": "Defensive suppression risk",
    "DEFENSIVE_LOCK_SUPPORT": "Defensive lock support",
    "DEFENSIVE_LOCK_SOFT": "Defensive lock soft",
    "OVER25_HEAT_SUPPORT": "Over 2.5 heat support",
    "OVER25_HEAT_SOFT": "Over 2.5 heat soft",
    "CHAOS_ENVIRONMENT_SUPPORT": "Chaos environment support",
    "CHAOS_ENVIRONMENT_SOFT": "Chaos environment soft",
    "CONTROL_SUPPRESSION_RISK": "Control suppression risk",
    "CONTROL_NOT_STRONG_ENOUGH": "Control not strong enough",
    "OVER25_ENVIRONMENT_RISK": "Over 2.5 environment risk",
    "ATTACK_VS_DEFENCE_MISMATCH": "Attack versus defence mismatch",
    "AWAY_ATTACK_MISMATCH": "Away attack mismatch",
    "LINEUP_UNIT_EDGE_SOFT": "Lineup unit edge soft",
    "BOTH_ATTACK_UNITS_LIVE": "Both attack units live",
    "DEFENSIVE_UNIT_SUPPRESSION_RISK": "Defensive unit suppression risk",
    "DEFENSIVE_UNIT_SUPPORT": "Defensive unit support",
    "LINEUP_GOAL_PRESSURE_SUPPORT": "Lineup goal pressure support",
    "LINEUP_SUPPRESSION_SUPPORT": "Lineup suppression support",
    "LINEUP_DATA_MISSING": "Lineup data missing",
    "H2H_BTTS_SUPPORT": "H2H BTTS support",
    "H2H_BTTS_SUPPRESSION": "H2H BTTS suppression",
    "H2H_OVER_SUPPORT": "H2H over support",
    "H2H_UNDER_SUPPORT": "H2H under support",
    "H2H_DRAW_SUPPORT": "H2H draw support",
    "H2H_UNAVAILABLE": "H2H unavailable",
}

MARKET_LABELS = {
    "ftr": "FTR",
    "btts": "BTTS",
    "ou25": "Over 2.5",
    "team_goals": "Team Goals",
    "correct_score": "Correct Score",
    "corners": "Corners",
    "cards": "Cards",
}

MARKET_REASON_HINTS = {
    "ftr": ("POWER", "FORTRESS", "THREAT", "CONTROL", "MISMATCH"),
    "btts": ("BTTS", "GOAL", "ATTACK", "SUPPRESSION", "DEFENSIVE"),
    "ou25": ("OVER25", "GOAL", "CHAOS", "CONTROL", "SUPPRESSION"),
    "team_goals": ("ATTACK", "GOAL", "FIRST_STRIKE", "MISMATCH", "LINEUP"),
    "correct_score": ("CONTROL", "POWER", "DEFENSIVE", "CHAOS", "GOAL"),
    "corners": ("CORNER", "WIDE", "PRESSURE"),
    "cards": ("CARD", "CHAOS", "BOOKING", "DISCIPLINE"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalize(value: object) -> str:
    return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())


def season_start(value: object) -> str:
    raw = str(value or "").strip()
    for token in raw.replace("-", "/").split("/"):
        if token.isdigit() and len(token) == 4:
            return token
    return raw


def clamp_0_100(value: object, fallback: int = 0) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0, min(100, int(round(numeric))))


def avg(values: list[object]) -> int | None:
    usable = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        usable.append(numeric)
    if not usable:
        return None
    return clamp_0_100(sum(usable) / len(usable))


def safe_title(value: object) -> str:
    raw = str(value or "").replace("_", " ").strip()
    return raw.title() if raw else "—"


def public_reason_label(token: object) -> str:
    key = str(token or "").strip().upper()
    if key in REASON_LABELS:
        return REASON_LABELS[key]
    key = (
        key.replace("H2H_", "H2H ")
        .replace("BTTS_", "BTTS ")
        .replace("OU25_", "OU25 ")
        .replace("TEAM_", "Team ")
        .replace("AWAY_", "Away ")
        .replace("HOME_", "Home ")
    )
    return safe_title(key)


def score_band(value: int | None) -> str:
    if value is None:
        return "Unspecified"
    if value >= 90:
        return "Elite"
    if value >= 80:
        return "Strong"
    if value >= 70:
        return "Positive"
    if value >= 55:
        return "Mixed"
    if value >= 40:
        return "Weak"
    return "Red Flag"


def confidence_band(agreement_score: int, confidence_tier: object = None) -> str:
    raw = str(confidence_tier or "").strip().upper()
    if raw:
        return raw
    if agreement_score >= 80:
        return "HIGH"
    if agreement_score >= 62:
        return "MEDIUM"
    return "LOW"


def market_key_for_family(family: str) -> str | None:
    family = str(family or "").upper()
    if family == "FTR":
        return "ftr"
    if family == "BTTS":
        return "btts"
    if family == "OU25":
        return "ou25"
    return None


def state_from_alignment(alignment_score: int) -> str:
    if alignment_score >= 80:
        return "SUPPORTED"
    if alignment_score >= 62:
        return "MIXED"
    if alignment_score >= 45:
        return "FRAGILE"
    return "AVOID"


def signal_profile(fixture: dict[str, Any]) -> tuple[str, str]:
    signal_summary = fixture.get("signal_summary") or {}
    family = str(signal_summary.get("market_family") or "").upper()
    pick = str(signal_summary.get("deploy_pick") or fixture.get("deploy_summary", {}).get("pick") or "").upper()
    copy = f"{signal_summary.get('headline', '')} {signal_summary.get('summary_text', '')}".lower()
    if not pick:
        if family == "BTTS":
            pick = "NO" if "btts no" in copy else "YES"
        elif family == "OU25":
            pick = "UNDER25" if "under" in copy else "OVER25"
        elif family == "FTR":
            if "draw" in copy:
                pick = "DRAW"
            elif "away" in copy:
                pick = "AWAY"
            else:
                pick = "HOME"
    return family, pick


def primary_signal_label(fixture: dict[str, Any]) -> str:
    family, pick = signal_profile(fixture)
    if family == "BTTS":
        return "BTTS No" if pick == "NO" else "BTTS Yes"
    if family == "OU25":
        return "Under 2.5" if pick == "UNDER25" else "Over 2.5"
    if family == "FTR":
        if pick == "AWAY":
            return f"{fixture.get('away_team', 'Away')} Win"
        if pick == "DRAW":
            return "Draw"
        return f"{fixture.get('home_team', 'Home')} Win"
    return str((fixture.get("signal_summary") or {}).get("signal_label") or "Fixture read")


def relevant_rating_keys(fixture: dict[str, Any]) -> list[str]:
    family, pick = signal_profile(fixture)
    if family == "BTTS":
        if pick == "NO":
            return ["defensive_lock_rating", "control_rating", "goal_heat_rating", "btts_pressure_rating", "chaos_rating"]
        return ["goal_heat_rating", "btts_pressure_rating", "attack_flow_rating", "defensive_lock_rating", "first_strike_rating", "chaos_rating"]
    if family == "OU25":
        if pick == "UNDER25":
            return ["control_rating", "defensive_lock_rating", "goal_heat_rating", "over25_heat_rating", "chaos_rating", "first_strike_rating"]
        return ["goal_heat_rating", "over25_heat_rating", "attack_flow_rating", "defensive_lock_rating", "chaos_rating", "first_strike_rating"]
    if family == "FTR":
        if pick == "AWAY":
            return ["og_power_rating", "away_threat_rating", "home_fortress_rating", "defensive_lock_rating", "first_strike_rating", "control_rating"]
        if pick == "DRAW":
            return ["control_rating", "defensive_lock_rating", "chaos_rating", "goal_heat_rating", "first_strike_rating"]
        return ["og_power_rating", "home_fortress_rating", "away_threat_rating", "defensive_lock_rating", "first_strike_rating", "control_rating"]
    return ["og_power_rating", "attack_flow_rating", "defensive_lock_rating", "goal_heat_rating"]


class DecisionPublisher:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.fixture_feed = load_json(data_root / "fixture_intelligence_public.json")
        self.fixtures = list(self.fixture_feed.get("fixtures") or [])
        self.team_index = list(load_json(data_root / "team_intelligence" / "team_ratings_index.json"))
        self.club_index = list(load_json(data_root / "player_intelligence" / "club_squad_ratings.json"))
        self.lineup_index = {row["fixture_key"]: row for row in load_json(data_root / "fixture_lineup_intelligence" / "index.json")}
        h2h_index_path = data_root / "fixture_h2h_support" / "index.json"
        self.h2h_index = {row["fixture_key"]: row for row in load_json(h2h_index_path)} if h2h_index_path.exists() else {}

    def find_best_team_entry(self, team_name: str, competition: str, season: object) -> dict[str, Any] | None:
        target_team = normalize(team_name)
        target_comp = normalize(competition)
        target_season = season_start(season)
        candidates = []
        for entry in self.team_index:
            if normalize(entry.get("team")) != target_team:
                continue
            score = 0
            if normalize(entry.get("competition")) == target_comp:
                score += 4
            if season_start(entry.get("season")) == target_season:
                score += 6
            candidates.append((score, str(entry.get("season") or ""), entry))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2] if candidates else None

    def find_best_club_entry(self, team_name: str, competition: str, season: object) -> dict[str, Any] | None:
        target_team = normalize(team_name)
        target_comp = normalize(competition)
        target_season = season_start(season)
        candidates = []
        for entry in self.club_index:
            if normalize(entry.get("club")) != target_team:
                continue
            score = 0
            if normalize(entry.get("competition")) == target_comp:
                score += 4
            if season_start(entry.get("season")) == target_season:
                score += 6
            candidates.append((score, str(entry.get("season") or ""), entry))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2] if candidates else None

    @lru_cache(maxsize=65536)
    def load_team_payload(self, competition_key: str, season: str, team_slug: str) -> dict[str, Any] | None:
        path = self.data_root / "team_intelligence" / "teams" / competition_key / season / f"{team_slug}.json"
        return load_json(path) if path.exists() else None

    @lru_cache(maxsize=65536)
    def load_club_payload(self, competition_key: str, season: str, club_slug: str) -> dict[str, Any] | None:
        path = self.data_root / "player_intelligence" / "clubs" / competition_key / season / f"{club_slug}.json"
        return load_json(path) if path.exists() else None

    @lru_cache(maxsize=65536)
    def load_lineup_payload(self, fixture_key: str) -> dict[str, Any] | None:
        path = self.data_root / "fixture_lineup_intelligence" / f"{fixture_key}.json"
        return load_json(path) if path.exists() else None

    @lru_cache(maxsize=65536)
    def load_h2h_payload(self, fixture_key: str) -> dict[str, Any] | None:
        path = self.data_root / "fixture_h2h_support" / f"{fixture_key}.json"
        return load_json(path) if path.exists() else None

    def build_team_faceoff_summary(self, fixture: dict[str, Any], home_team: dict[str, Any] | None, away_team: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not home_team or not away_team:
            return []
        summary = []
        for key in relevant_rating_keys(fixture):
            home_value = home_team.get("ratings", {}).get(key)
            away_value = away_team.get("ratings", {}).get(key)
            if home_value is None and away_value is None:
                continue
            delta = None
            if home_value is not None and away_value is not None:
                delta = clamp_0_100(abs(float(home_value) - float(away_value)), fallback=0)
            leader = "Parity"
            if home_value is not None and away_value is not None:
                if float(home_value) > float(away_value):
                    leader = fixture.get("home_team", "Home")
                elif float(away_value) > float(home_value):
                    leader = fixture.get("away_team", "Away")
            summary.append(
                {
                    "metric": key,
                    "label": TEAM_RATING_LABELS.get(key, safe_title(key)),
                    "home_value": clamp_0_100(home_value, fallback=0) if home_value is not None else None,
                    "away_value": clamp_0_100(away_value, fallback=0) if away_value is not None else None,
                    "delta": delta,
                    "leader": leader,
                }
            )
        return summary

    def build_support_and_caution(self, fixture: dict[str, Any], home_team: dict[str, Any] | None, away_team: dict[str, Any] | None, lineup: dict[str, Any] | None, h2h: dict[str, Any] | None) -> tuple[list[str], list[str], int]:
        support: list[str] = []
        caution: list[str] = []
        score = 50
        family, pick = signal_profile(fixture)
        home_ratings = (home_team or {}).get("ratings", {})
        away_ratings = (away_team or {}).get("ratings", {})
        home_units = (lineup or {}).get("home_units", {})
        away_units = (lineup or {}).get("away_units", {})

        def add_support(token: str, weight: int = 6) -> None:
            nonlocal score
            if token not in support:
                support.append(token)
                score += weight

        def add_caution(token: str, weight: int = 6) -> None:
            nonlocal score
            if token not in caution:
                caution.append(token)
                score -= weight

        if home_team and away_team:
            if family == "FTR":
                power_edge = float(home_ratings.get("og_power_rating", 50)) - float(away_ratings.get("og_power_rating", 50))
                if pick == "HOME":
                    if power_edge >= 8:
                        add_support("TEAM_POWER_ADVANTAGE")
                    else:
                        add_caution("TEAM_POWER_NOT_DECISIVE")
                    if float(home_ratings.get("home_fortress_rating", 50)) >= float(away_ratings.get("away_threat_rating", 50)) + 6:
                        add_support("HOME_FORTRESS_ADVANTAGE")
                    else:
                        add_caution("AWAY_TRAVEL_THREAT")
                elif pick == "AWAY":
                    if power_edge <= -8:
                        add_support("AWAY_POWER_ADVANTAGE")
                    else:
                        add_caution("AWAY_POWER_NOT_DECISIVE")
                    if float(away_ratings.get("away_threat_rating", 50)) >= float(home_ratings.get("home_fortress_rating", 50)) + 6:
                        add_support("AWAY_THREAT_ADVANTAGE")
                    else:
                        add_caution("HOME_RESISTANCE_HOLDS")
                else:
                    if abs(power_edge) <= 8:
                        add_support("POWER_PARITY")
                    else:
                        add_caution("POWER_IMBALANCE")
                    if avg([home_ratings.get("control_rating"), away_ratings.get("control_rating")]) and avg([home_ratings.get("control_rating"), away_ratings.get("control_rating")]) >= 58:
                        add_support("CONTROL_SUPPORT")
            elif family == "BTTS":
                combined_btts = avg([home_ratings.get("btts_pressure_rating"), away_ratings.get("btts_pressure_rating")]) or 0
                combined_heat = avg([home_ratings.get("goal_heat_rating"), away_ratings.get("goal_heat_rating")]) or 0
                max_lock = max(float(home_ratings.get("defensive_lock_rating", 0)), float(away_ratings.get("defensive_lock_rating", 0)))
                if pick == "YES":
                    if combined_btts >= 58:
                        add_support("BTTS_PRESSURE_SUPPORT")
                    else:
                        add_caution("BTTS_PRESSURE_WEAK")
                    if combined_heat >= 60:
                        add_support("GOAL_ENVIRONMENT_SUPPORT")
                    else:
                        add_caution("GOAL_ENVIRONMENT_SOFT")
                    if max_lock >= 78:
                        add_caution("DEFENSIVE_SUPPRESSION_RISK")
                else:
                    if combined_btts <= 44:
                        add_support("BTTS_SUPPRESSION_SUPPORT")
                    else:
                        add_caution("BTTS_ACCESS_RISK")
                    if max_lock >= 72:
                        add_support("DEFENSIVE_LOCK_SUPPORT")
                    else:
                        add_caution("DEFENSIVE_LOCK_SOFT")
            elif family == "OU25":
                combined_over = avg([home_ratings.get("over25_heat_rating"), away_ratings.get("over25_heat_rating")]) or 0
                combined_control = avg([home_ratings.get("control_rating"), away_ratings.get("control_rating")]) or 0
                combined_chaos = avg([home_ratings.get("chaos_rating"), away_ratings.get("chaos_rating")]) or 0
                if pick == "OVER25":
                    if combined_over >= 58:
                        add_support("OVER25_HEAT_SUPPORT")
                    else:
                        add_caution("OVER25_HEAT_SOFT")
                    if combined_chaos >= 52:
                        add_support("CHAOS_ENVIRONMENT_SUPPORT")
                    else:
                        add_caution("CHAOS_ENVIRONMENT_SOFT")
                    if combined_control >= 62:
                        add_caution("CONTROL_SUPPRESSION_RISK")
                else:
                    if combined_control >= 62:
                        add_support("CONTROL_SUPPORT")
                    else:
                        add_caution("CONTROL_NOT_STRONG_ENOUGH")
                    if combined_over >= 58:
                        add_caution("OVER25_ENVIRONMENT_RISK")

        if lineup:
            home_attack = float(home_units.get("attack_unit", 50))
            away_attack = float(away_units.get("attack_unit", 50))
            home_def = float(home_units.get("defensive_unit", 50))
            away_def = float(away_units.get("defensive_unit", 50))
            if family == "FTR":
                if pick == "HOME" and home_attack >= away_def + 8:
                    add_support("ATTACK_VS_DEFENCE_MISMATCH")
                elif pick == "AWAY" and away_attack >= home_def + 8:
                    add_support("AWAY_ATTACK_MISMATCH")
                else:
                    add_caution("LINEUP_UNIT_EDGE_SOFT")
            elif family == "BTTS":
                if pick == "YES" and home_attack >= 58 and away_attack >= 58:
                    add_support("BOTH_ATTACK_UNITS_LIVE")
                if pick == "YES" and (home_def >= 72 or away_def >= 72):
                    add_caution("DEFENSIVE_UNIT_SUPPRESSION_RISK")
                if pick == "NO" and (home_def >= 68 or away_def >= 68):
                    add_support("DEFENSIVE_UNIT_SUPPORT")
            elif family == "OU25":
                if pick == "OVER25" and home_attack >= 58 and away_attack >= 54:
                    add_support("LINEUP_GOAL_PRESSURE_SUPPORT")
                if pick == "UNDER25" and home_def >= 62 and away_def >= 62:
                    add_support("LINEUP_SUPPRESSION_SUPPORT")
        else:
            add_caution("LINEUP_DATA_MISSING", weight=4)

        if h2h:
            sample_size = int(h2h.get("sample_size") or 0)
            if sample_size > 0:
                if family == "BTTS" and pick == "YES" and clamp_0_100(h2h.get("btts_regime")) >= 58:
                    add_support("H2H_BTTS_SUPPORT", weight=4)
                elif family == "BTTS" and pick == "NO" and clamp_0_100(h2h.get("btts_regime")) <= 42:
                    add_support("H2H_BTTS_SUPPRESSION", weight=4)
                elif family == "OU25" and pick == "OVER25" and clamp_0_100(h2h.get("over25_rate")) >= 58:
                    add_support("H2H_OVER_SUPPORT", weight=4)
                elif family == "OU25" and pick == "UNDER25" and clamp_0_100(h2h.get("over25_rate")) <= 42:
                    add_support("H2H_UNDER_SUPPORT", weight=4)
                elif family == "FTR" and pick == "DRAW" and clamp_0_100(h2h.get("draw_rate")) >= 34:
                    add_support("H2H_DRAW_SUPPORT", weight=4)
            else:
                add_caution("H2H_UNAVAILABLE", weight=2)
        else:
            add_caution("H2H_UNAVAILABLE", weight=2)

        return support, caution, max(0, min(100, score))

    def build_unit_battle_summary(self, lineup: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not lineup:
            return []
        home = lineup.get("home_units") or {}
        away = lineup.get("away_units") or {}
        battles = [
            {
                "label": "Attack vs Defence",
                "home_value": clamp_0_100(home.get("attack_unit")),
                "away_value": clamp_0_100(away.get("defensive_unit")),
                "delta": clamp_0_100(float(home.get("attack_unit", 0)) - float(away.get("defensive_unit", 0)) + 50) - 50,
            },
            {
                "label": "Midfield Control",
                "home_value": clamp_0_100(home.get("midfield_control")),
                "away_value": clamp_0_100(away.get("midfield_control")),
                "delta": clamp_0_100(float(home.get("midfield_control", 0)) - float(away.get("midfield_control", 0)) + 50) - 50,
            },
            {
                "label": "Away Attack vs Home Defence",
                "home_value": clamp_0_100(away.get("attack_unit")),
                "away_value": clamp_0_100(home.get("defensive_unit")),
                "delta": clamp_0_100(float(away.get("attack_unit", 0)) - float(home.get("defensive_unit", 0)) + 50) - 50,
            },
        ]
        return battles

    def build_profile_narrative(self, fixture: dict[str, Any], home_team: dict[str, Any] | None, away_team: dict[str, Any] | None) -> str:
        home_tags = list((home_team or {}).get("profile_tags") or [])
        away_tags = list((away_team or {}).get("profile_tags") or [])
        family, pick = signal_profile(fixture)
        home_label = home_tags[0] if home_tags else "mixed home profile"
        away_label = away_tags[0] if away_tags else "mixed away profile"
        sentence = f"{fixture.get('home_team', 'Home')} bring a {home_label.lower()} into a matchup with {fixture.get('away_team', 'Away')}'s {away_label.lower()}."
        if family == "BTTS":
            sentence += " The cleaner read is two-way scoring access." if pick != "NO" else " The cleaner read is suppression and restricted access rather than an open trade."
        elif family == "OU25":
            sentence += " The shape points toward a live goal environment." if pick == "OVER25" else " The shape points toward control and a lower-event scoring profile."
        elif family == "FTR":
            if pick == "AWAY":
                sentence += f" The structural lean comes from {fixture.get('away_team', 'Away')}'s travelling profile against the home-side resistance."
            elif pick == "DRAW":
                sentence += " The matchup reads closer to parity than clear one-sided dominance."
            else:
                sentence += f" The structural lean comes from {fixture.get('home_team', 'Home')}'s home-side control against the away profile."
        return sentence

    def build_h2h_context(self, h2h: dict[str, Any] | None) -> dict[str, Any]:
        if not h2h:
            return {
                "available": False,
                "summary": "No publish-safe H2H regime summary is available for this fixture yet, so history is not being used as a supporting layer.",
                "sample_size": 0,
            }
        return {
            "available": bool(int(h2h.get("sample_size") or 0) > 0),
            "summary": h2h.get("summary") or "Historic meeting regime is shown as supporting context only.",
            "sample_size": int(h2h.get("sample_size") or 0),
            "goal_environment": clamp_0_100(h2h.get("goal_environment")),
            "btts_regime": clamp_0_100(h2h.get("btts_regime")),
            "over25_rate": clamp_0_100(h2h.get("over25_rate")),
            "draw_rate": clamp_0_100(h2h.get("draw_rate")),
            "booking_heat": clamp_0_100(h2h.get("booking_heat")),
        }

    def build_key_player_drivers(self, squad_payload: dict[str, Any] | None, lineup_profiles: list[dict[str, Any]] | None, team_label: str) -> list[dict[str, Any]]:
        drivers: list[dict[str, Any]] = []
        if lineup_profiles:
            sorted_profiles = sorted(lineup_profiles, key=lambda item: float(item.get("power", 0)), reverse=True)[:2]
            for profile in sorted_profiles:
                metrics = [
                    ("Goal Threat", clamp_0_100(profile.get("goal_threat"))),
                    ("Creative Spark", clamp_0_100(profile.get("creative_spark"))),
                    ("Defensive Lock", clamp_0_100(profile.get("defensive_lock"))),
                    ("Midfield Engine", clamp_0_100(profile.get("midfield_engine"))),
                ]
                metrics.sort(key=lambda item: item[1], reverse=True)
                drivers.append(
                    {
                        "team": team_label,
                        "player": profile.get("name") or profile.get("surname") or "Profile pending",
                        "role": safe_title(profile.get("position_group")),
                        "power": clamp_0_100(profile.get("power")),
                        "driver_metric": metrics[0][0],
                        "driver_value": metrics[0][1],
                    }
                )
        elif squad_payload and squad_payload.get("players"):
            for player in squad_payload["players"][:2]:
                ratings = player.get("ratings") or {}
                metrics = [
                    ("Goal Threat", clamp_0_100(ratings.get("goal_threat"))),
                    ("Creative Spark", clamp_0_100(ratings.get("creative_spark"))),
                    ("Defensive Lock", clamp_0_100(ratings.get("defensive_lock"))),
                    ("Midfield Engine", clamp_0_100(ratings.get("midfield_engine"))),
                ]
                metrics.sort(key=lambda item: item[1], reverse=True)
                drivers.append(
                    {
                        "team": team_label,
                        "player": player.get("name") or player.get("surname") or "Profile pending",
                        "role": safe_title(player.get("position_group")),
                        "power": clamp_0_100(ratings.get("og_player_power")),
                        "driver_metric": metrics[0][0],
                        "driver_value": metrics[0][1],
                    }
                )
        return drivers

    def build_market_suitability(self, fixture: dict[str, Any], home_team: dict[str, Any] | None, away_team: dict[str, Any] | None, lineup: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        home_ratings = (home_team or {}).get("ratings", {})
        away_ratings = (away_team or {}).get("ratings", {})
        home_goal = clamp_0_100(home_ratings.get("goal_heat_rating"))
        away_goal = clamp_0_100(away_ratings.get("goal_heat_rating"))
        home_btts = clamp_0_100(home_ratings.get("btts_pressure_rating"))
        away_btts = clamp_0_100(away_ratings.get("btts_pressure_rating"))
        home_power = clamp_0_100(home_ratings.get("og_power_rating"))
        away_power = clamp_0_100(away_ratings.get("og_power_rating"))
        home_corner = clamp_0_100(home_ratings.get("corner_pressure_rating"))
        away_corner = clamp_0_100(away_ratings.get("corner_pressure_rating"))
        home_card = clamp_0_100(home_ratings.get("card_heat_rating"))
        away_card = clamp_0_100(away_ratings.get("card_heat_rating"))
        combined_goal = avg([home_goal, away_goal]) or 0
        combined_btts = avg([home_btts, away_btts]) or 0
        combined_corner = avg([home_corner, away_corner]) or 0
        combined_card = avg([home_card, away_card]) or 0
        power_gap = abs(home_power - away_power)
        return {
            "ftr": {"rating": max(home_power, away_power), "label": score_band(max(home_power, away_power)), "read": "Stronger when one side owns a clear team-power edge."},
            "btts": {"rating": combined_btts, "label": score_band(combined_btts), "read": "Driven by two-way access and reduced suppression."},
            "ou25": {"rating": combined_goal, "label": score_band(combined_goal), "read": "Driven by combined goal environment and tempo."},
            "team_goals": {"rating": max(clamp_0_100((lineup or {}).get("home_units", {}).get("attack_unit")), clamp_0_100((lineup or {}).get("away_units", {}).get("attack_unit")), home_goal, away_goal), "label": "Strong" if max(home_goal, away_goal) >= 70 else score_band(max(home_goal, away_goal)), "read": "Useful when one attacking unit clearly dominates the defensive resistance opposite it."},
            "correct_score": {"rating": max(0, min(100, 100 - power_gap + abs(home_goal - away_goal))), "label": score_band(max(0, min(100, 100 - power_gap + abs(home_goal - away_goal)))), "read": "Better when the shape is structured and the likely scoring zone is narrow."},
            "corners": {"rating": combined_corner, "label": score_band(combined_corner), "read": "Driven by territory and wide-channel pressure."},
            "cards": {"rating": combined_card, "label": score_band(combined_card), "read": "Driven by discipline heat, fouling, and volatility."},
        }

    def market_model_lean(
        self,
        market_key: str,
        fixture: dict[str, Any],
        home_team: dict[str, Any] | None,
        away_team: dict[str, Any] | None,
        lineup: dict[str, Any] | None,
    ) -> str:
        family, pick = signal_profile(fixture)
        primary_market = market_key_for_family(family)
        if primary_market == market_key:
            return pick or "LEAN"
        home_ratings = (home_team or {}).get("ratings", {})
        away_ratings = (away_team or {}).get("ratings", {})
        if market_key == "ftr":
            home_power = clamp_0_100(home_ratings.get("og_power_rating"))
            away_power = clamp_0_100(away_ratings.get("og_power_rating"))
            if abs(home_power - away_power) <= 6:
                return "DRAW"
            return "HOME" if home_power >= away_power else "AWAY"
        if market_key == "btts":
            combined_btts = avg([home_ratings.get("btts_pressure_rating"), away_ratings.get("btts_pressure_rating")]) or 0
            return "YES" if combined_btts >= 52 else "NO"
        if market_key == "ou25":
            combined_goal = avg([home_ratings.get("goal_heat_rating"), away_ratings.get("goal_heat_rating")]) or 0
            return "OVER25" if combined_goal >= 55 else "UNDER25"
        if market_key == "team_goals":
            home_attack = clamp_0_100((lineup or {}).get("home_units", {}).get("attack_unit") or home_ratings.get("goal_heat_rating"))
            away_attack = clamp_0_100((lineup or {}).get("away_units", {}).get("attack_unit") or away_ratings.get("goal_heat_rating"))
            leader = fixture.get("home_team", "Home") if home_attack >= away_attack else fixture.get("away_team", "Away")
            return f"{leader} 1.5+"
        if market_key == "correct_score":
            return "STRUCTURED" if clamp_0_100(home_ratings.get("control_rating")) >= 58 or clamp_0_100(away_ratings.get("control_rating")) >= 58 else "OPEN"
        if market_key == "corners":
            return "ELEVATED" if avg([home_ratings.get("corner_pressure_rating"), away_ratings.get("corner_pressure_rating")]) or 0 >= 55 else "MIXED"
        if market_key == "cards":
            return "ELEVATED" if avg([home_ratings.get("card_heat_rating"), away_ratings.get("card_heat_rating")]) or 0 >= 55 else "MIXED"
        return "LEAN"

    def select_market_reason_tokens(self, market_key: str, tokens: list[str]) -> list[str]:
        hints = MARKET_REASON_HINTS.get(market_key, ())
        matched = [token for token in tokens if any(hint in token for hint in hints)]
        if matched:
            return matched[:3]
        return tokens[:2]

    def market_public_summary(self, market_label: str, state: str, support: list[str], caution: list[str]) -> str:
        support_text = public_reason_label(support[0]) if support else "structural support"
        caution_text = public_reason_label(caution[0]) if caution else "no major caution published"
        if state == "SUPPORTED":
            return f"{market_label} is well supported by {support_text.lower()}, with caution still coming from {caution_text.lower()}."
        if state == "WATCHLIST":
            return f"{market_label} is interesting enough to monitor, but current support is better treated as watch-first because of {caution_text.lower()}."
        if state == "MIXED":
            return f"{market_label} has partial support from {support_text.lower()}, but {caution_text.lower()} keeps it from reading clean."
        if state == "AVOID":
            return f"{market_label} is too contradicted by {caution_text.lower()} to present as a clean public read."
        return f"{market_label} has some structural support from {support_text.lower()}, but the overall layer fit remains fragile."

    def build_market_intelligence(
        self,
        fixture: dict[str, Any],
        signal_state: str,
        agreement_score: int,
        supporting_layers: list[str],
        caution_layers: list[str],
        market_suitability: dict[str, dict[str, Any]],
        home_team: dict[str, Any] | None,
        away_team: dict[str, Any] | None,
        lineup: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        family, _pick = signal_profile(fixture)
        primary_market = market_key_for_family(family)
        result: dict[str, dict[str, Any]] = {}
        for market_key, market in market_suitability.items():
            base_rating = clamp_0_100(market.get("rating"))
            support = self.select_market_reason_tokens(market_key, supporting_layers)
            caution = self.select_market_reason_tokens(market_key, caution_layers)
            if market_key == primary_market:
                alignment_score = clamp_0_100(round((agreement_score * 0.6) + (base_rating * 0.4)))
                state = signal_state
            else:
                alignment_score = clamp_0_100(round((base_rating * 0.75) + (len(support) * 6) - (len(caution) * 4) + 10))
                state = state_from_alignment(alignment_score)
            result[market_key] = {
                "alignment_score": alignment_score,
                "state": state,
                "model_lean": self.market_model_lean(market_key, fixture, home_team, away_team, lineup),
                "structural_support": support,
                "cautions": caution,
                "public_summary": self.market_public_summary(MARKET_LABELS.get(market_key, safe_title(market_key)), state, support, caution),
                "rating": base_rating,
                "band": market.get("label") or score_band(base_rating),
            }
        return result

    def build_watchlist(self, fixture: dict[str, Any], signal_state: str, caution_layers: list[str]) -> dict[str, Any]:
        family, pick = signal_profile(fixture)
        active = signal_state == "WATCHLIST"
        if family == "BTTS":
            triggers = [
                "Confirm both forward lines are intact before kickoff.",
                "Watch for early two-way shot volume inside the opening 15 minutes.",
                "Escalate only if suppression signs stay absent.",
            ]
        elif family == "OU25":
            triggers = [
                "Watch for early tempo and repeat penalty-box entries.",
                "Escalate only if control signals fail to settle the game.",
                "Treat a quiet opening phase as a downgrade rather than a delay.",
            ]
        else:
            triggers = [
                "Wait for the actual XI to confirm the structural edge.",
                "Watch whether midfield control matches the pre-match read.",
                "Treat short prices or late uncertainty as a reason to hold discipline.",
            ]
        summary = (
            f"{primary_signal_label(fixture)} is better treated as a watch-first structure because {public_reason_label(caution_layers[0]).lower()}."
            if active and caution_layers
            else "No watchlist layer is active for this fixture."
        )
        return {
            "active": active,
            "summary": summary,
            "trigger_signals": triggers if active else [],
            "public_state": "Watchlist: wait for live confirmation" if active else "No watchlist flag",
            "mode": f"{family}:{pick}" if active else "",
        }

    def derive_signal_state(self, fixture: dict[str, Any], agreement_score: int, caution_layers: list[str]) -> str:
        publish_class = str(fixture.get("publish_class") or "").upper()
        if agreement_score >= 80 and publish_class == "DEPLOY":
            return "SUPPORTED"
        if agreement_score >= 62 and publish_class == "DEPLOY":
            return "MIXED"
        if agreement_score >= 58 and publish_class != "DEPLOY":
            return "WATCHLIST"
        if agreement_score >= 45:
            return "FRAGILE"
        if len(caution_layers) >= 3:
            return "AVOID"
        return "FRAGILE"

    def public_summary(self, fixture: dict[str, Any], state: str, support: list[str], caution: list[str]) -> str:
        home = fixture.get("home_team", "Home")
        away = fixture.get("away_team", "Away")
        signal = primary_signal_label(fixture)
        if state == "SUPPORTED":
            return f"{signal} is structurally supported for {home} vs {away}, with multiple independent layers aligning behind the live read."
        if state == "WATCHLIST":
            return f"{signal} has useful structure for {home} vs {away}, but the current layer fit is better treated as watch-first than pre-match deploy."
        if state == "MIXED":
            return f"{signal} has live support in {home} vs {away}, but the supporting layers are not clean enough to read as fully aligned."
        if state == "AVOID":
            return f"{signal} is too contradicted across the current structural layers for {home} vs {away}, so the clean read is caution rather than action."
        return f"{signal} has some live shape in {home} vs {away}, but the supporting layers remain fragile and need discipline."

    def build_fixture_payload(self, fixture: dict[str, Any]) -> dict[str, Any]:
        competition = fixture.get("league")
        season = fixture.get("api_season")
        home_team_name = fixture.get("home_team")
        away_team_name = fixture.get("away_team")

        home_team_entry = self.find_best_team_entry(home_team_name, competition, season)
        away_team_entry = self.find_best_team_entry(away_team_name, competition, season)
        home_squad_entry = self.find_best_club_entry(home_team_name, competition, season)
        away_squad_entry = self.find_best_club_entry(away_team_name, competition, season)

        home_team = self.load_team_payload(home_team_entry["competition_key"], home_team_entry["season"], home_team_entry["team_slug"]) if home_team_entry else None
        away_team = self.load_team_payload(away_team_entry["competition_key"], away_team_entry["season"], away_team_entry["team_slug"]) if away_team_entry else None
        home_squad = self.load_club_payload(home_squad_entry["competition_key"], home_squad_entry["season"], home_squad_entry["club_slug"]) if home_squad_entry else None
        away_squad = self.load_club_payload(away_squad_entry["competition_key"], away_squad_entry["season"], away_squad_entry["club_slug"]) if away_squad_entry else None
        lineup = self.load_lineup_payload(fixture["fixture_key"])
        h2h = self.load_h2h_payload(fixture["fixture_key"])

        supporting_layers, caution_layers, agreement_score = self.build_support_and_caution(fixture, home_team, away_team, lineup, h2h)
        signal_state = self.derive_signal_state(fixture, agreement_score, caution_layers)
        lineup_home_profiles = (lineup or {}).get("home_lineup_profiles") or []
        lineup_away_profiles = (lineup or {}).get("away_lineup_profiles") or []
        market_suitability = self.build_market_suitability(fixture, home_team, away_team, lineup)
        market_intelligence = self.build_market_intelligence(
            fixture,
            signal_state,
            agreement_score,
            supporting_layers,
            caution_layers,
            market_suitability,
            home_team,
            away_team,
            lineup,
        )
        watchlist = self.build_watchlist(fixture, signal_state, caution_layers)

        return {
            "fixture_key": fixture["fixture_key"],
            "fixture": f"{home_team_name} vs {away_team_name}",
            "primary_signal": primary_signal_label(fixture),
            "signal_state": signal_state,
            "agreement_score": agreement_score,
            "confidence_band": confidence_band(agreement_score, (fixture.get("signal_summary") or {}).get("confidence_tier")),
            "supporting_layers": supporting_layers,
            "caution_layers": caution_layers,
            "profile_tags": {
                "home": list((home_team or {}).get("profile_tags") or [])[:4],
                "away": list((away_team or {}).get("profile_tags") or [])[:4],
            },
            "profile_narrative": self.build_profile_narrative(fixture, home_team, away_team),
            "team_faceoff_summary": self.build_team_faceoff_summary(fixture, home_team, away_team),
            "unit_battle_summary": self.build_unit_battle_summary(lineup),
            "key_player_drivers": (
                self.build_key_player_drivers(home_squad, lineup_home_profiles, home_team_name) +
                self.build_key_player_drivers(away_squad, lineup_away_profiles, away_team_name)
            )[:6],
            "key_mismatches": list((lineup or {}).get("key_mismatches") or [])[:4],
            "h2h_context": self.build_h2h_context(h2h),
            "market_suitability": market_suitability,
            "market_intelligence": market_intelligence,
            "watchlist": watchlist,
            "public_safe_summary": self.public_summary(fixture, signal_state, supporting_layers, caution_layers),
            "internal_reason_tokens": supporting_layers + caution_layers,
            "summary": {
                "support_count": len(supporting_layers),
                "caution_count": len(caution_layers),
                "lineup_available": bool(lineup),
                "h2h_available": bool(h2h and int(h2h.get("sample_size") or 0) > 0),
            },
        }

    def publish(self, output_root: Path | None = None) -> int:
        target_root = output_root or self.data_root
        target_dir = target_root / "fixture_decision_intelligence"
        target_dir.mkdir(parents=True, exist_ok=True)
        index_rows = []
        for fixture in self.fixtures:
            payload = self.build_fixture_payload(fixture)
            (target_dir / f"{payload['fixture_key']}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            index_rows.append(
                {
                    "fixture_key": payload["fixture_key"],
                    "fixture": payload["fixture"],
                    "primary_signal": payload["primary_signal"],
                    "signal_state": payload["signal_state"],
                    "agreement_score": payload["agreement_score"],
                    "confidence_band": payload["confidence_band"],
                }
            )
        (target_dir / "index.json").write_text(json.dumps(index_rows, indent=2, ensure_ascii=False))
        return len(index_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish canonical fixture decision intelligence JSON.")
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publisher = DecisionPublisher(Path(args.data_root))
    total = publisher.publish(Path(args.output_root) if args.output_root else None)
    print(f"Published {total} fixture decision intelligence payloads.")


if __name__ == "__main__":
    main()
