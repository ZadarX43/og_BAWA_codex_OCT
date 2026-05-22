#!/usr/bin/env python3
"""Recalculate injury-shock impact against FTR, BTTS, and OU25.

This is a pre-website reporting sidecar. It does not mutate predictions,
deploy routing, or `deploy_rulebook.py`; it converts player availability
signals into compact market-context fields for the fixture brain compiler.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "reports/latest/injury_shock_market_impact"
DEFAULT_COVERAGE_NAME = "INJURY_SHOCK_ELITE_STANDARD_COVERAGE.csv"
DEFAULT_PLAYER_NAME = "INJURY_SHOCK_ELITE_STANDARD_PLAYER_IMPACT_WITH_RATINGS.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build injury shock market impact sidecars for FTR, BTTS, and OU25.")
    parser.add_argument("--coverage-csv", type=Path, default=None)
    parser.add_argument("--player-impact-csv", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default="INJURY_SHOCK_MARKET_IMPACT")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_default_source(filename: str) -> Path:
    candidates = sorted(
        (ROOT / "reports/latest").glob(f"injury_shock_elite_standard_*/{filename}"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return ROOT / "reports/latest/injury_shock_elite_standard_2026_05_22_to_2026_05_26" / filename


def text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def upper_tokens(value: Any) -> set[str]:
    return {part.strip().upper() for part in text(value).replace(",", "|").split("|") if part.strip()}


def num(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_key_player(row: pd.Series) -> bool:
    if int(num(row.get("key_player_flag"))) == 1:
        return True
    power = num(row.get("og_player_power"))
    goal = num(row.get("goal_threat"))
    creative = num(row.get("creative_spark"))
    defensive = num(row.get("defensive_lock"))
    club_rank = num(row.get("club_rank"))
    return bool(
        power >= 68
        or goal >= 72
        or creative >= 72
        or defensive >= 72
        or (0 < club_rank <= 5)
    )


def availability_status(absence_type: Any) -> str:
    value = text(absence_type).lower()
    if "questionable" in value or "doubt" in value:
        return "QUESTIONABLE_NOT_START_RISK"
    if value:
        return "LIKELY_NOT_STARTING_ABSENT"
    return "UNKNOWN_AVAILABILITY_RISK"


def role_impact(row: pd.Series) -> str:
    position = text(row.get("position")).lower()
    goal = num(row.get("goal_threat"))
    creative = num(row.get("creative_spark"))
    defensive = num(row.get("defensive_lock"))
    club_rank = num(row.get("club_rank"))
    if "goalkeeper" in position:
        return "KEEPER_ABSENCE"
    if "defender" in position and (defensive >= 55 or (0 < club_rank <= 5)):
        return "DEFENSIVE_STRUCTURE_ABSENCE"
    if goal >= 62 or ("forward" in position and (goal >= 58 or (0 < club_rank <= 5))):
        return "GOAL_THREAT_ABSENCE"
    if creative >= 52 or ("midfielder" in position and (creative >= 50 or (0 < club_rank <= 5))):
        return "CREATION_MIDFIELD_ABSENCE"
    return "DEPTH_ABSENCE"


def severity(row: pd.Series, role: str) -> str:
    if role == "DEPTH_ABSENCE":
        return "LOW"
    power = num(row.get("og_player_power"))
    goal = num(row.get("goal_threat"))
    creative = num(row.get("creative_spark"))
    defensive = num(row.get("defensive_lock"))
    club_rank = num(row.get("club_rank"))
    availability = availability_status(row.get("absence_type"))
    high = power >= 75 or goal >= 85 or creative >= 75 or defensive >= 75 or (0 < club_rank <= 2)
    medium = power >= 58 or goal >= 62 or creative >= 58 or defensive >= 57 or (0 < club_rank <= 5)
    if high:
        return "HIGH_IF_OUT" if availability == "QUESTIONABLE_NOT_START_RISK" else "HIGH"
    if medium:
        return "MEDIUM_IF_OUT" if availability == "QUESTIONABLE_NOT_START_RISK" else "MEDIUM"
    return "LOW"


def player_market_note(row: pd.Series, role: str) -> str:
    markets = upper_tokens(row.get("deploy_markets"))
    notes: list[str] = []
    is_questionable = availability_status(row.get("absence_type")) == "QUESTIONABLE_NOT_START_RISK"
    suffix = " if ruled out" if is_questionable else ""
    attacking = role in {"GOAL_THREAT_ABSENCE", "CREATION_MIDFIELD_ABSENCE"}
    defensive = role in {"DEFENSIVE_STRUCTURE_ABSENCE", "KEEPER_ABSENCE"}
    if "FTR" in markets and attacking:
        notes.append(f"FTR attacking/creative availability caution{suffix}")
    if "FTR" in markets and defensive:
        notes.append(f"FTR defensive/keeper volatility caution{suffix}")
    if "OU25" in markets and attacking:
        notes.append(f"OU25 goal-ceiling caution{suffix}")
    if "OU25" in markets and defensive:
        notes.append(f"OU25 defensive-chaos support{suffix}")
    if "BTTS" in markets and attacking:
        notes.append(f"BTTS scoring-access caution{suffix}")
    if "BTTS" in markets and defensive:
        notes.append(f"BTTS defensive-volatility support{suffix}")
    return "; ".join(notes) if notes else "LOW_DIRECT_MARKET_IMPACT"


def summarize_market(rows: pd.DataFrame, market: str) -> str:
    if rows.empty:
        return "NO_KEY_PLAYER_SHOCK_DETECTED"
    roles = set(rows["role_impact"].dropna().astype(str))
    attacking = bool(roles & {"GOAL_THREAT_ABSENCE", "CREATION_MIDFIELD_ABSENCE"})
    defensive = bool(roles & {"DEFENSIVE_STRUCTURE_ABSENCE", "KEEPER_ABSENCE"})
    if market == "FTR":
        if attacking and defensive:
            return "FTR_TWO_WAY_AVAILABILITY_REVIEW"
        if attacking:
            return "FTR_ATTACK_AVAILABILITY_REVIEW"
        if defensive:
            return "FTR_DEFENSIVE_VOLATILITY_REVIEW"
        return "FTR_LOW_DIRECT_INJURY_IMPACT"
    if market == "BTTS":
        if attacking and defensive:
            return "BTTS_MIXED_INJURY_SHOCK"
        if attacking:
            return "BTTS_SCORING_ACCESS_CAUTION"
        if defensive:
            return "BTTS_DEFENSIVE_VOLATILITY_SUPPORT"
        return "BTTS_LOW_DIRECT_INJURY_IMPACT"
    if market == "OU25":
        if attacking and defensive:
            return "OU25_MIXED_INJURY_SHOCK"
        if attacking:
            return "OU25_DOWNWARD_GOAL_RISK"
        if defensive:
            return "OU25_VOLATILITY_SUPPORT"
        return "OU25_LOW_DIRECT_INJURY_IMPACT"
    return "NO_MARKET_RULE"


def overall_adjustment(row: pd.Series) -> str:
    markets = upper_tokens(row.get("deploy_markets"))
    if not markets:
        return "NO_DEPLOY_MARKET_CONTEXT"
    for market in ("FTR", "BTTS", "OU25"):
        if market in markets:
            value = text(row.get(f"{market.lower()}_injury_impact"))
            if value and value != "NO_KEY_PLAYER_SHOCK_DETECTED":
                return value
    return "NO_KEY_PLAYER_SHOCK_DETECTED"


def rating_summary(row: pd.Series) -> str:
    fields = [
        ("power", "og_player_power"),
        ("goal", "goal_threat"),
        ("creative", "creative_spark"),
        ("defensive", "defensive_lock"),
        ("club rank", "club_rank"),
    ]
    parts = [f"{label} {num(row.get(column)):.1f}" for label, column in fields if text(row.get(column))]
    return "; ".join(parts)


def enrich_player_rows(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return players
    enriched = players.copy()
    enriched["key_player_flag"] = enriched.apply(lambda row: int(is_key_player(row)), axis=1)
    enriched["availability_impact_status"] = enriched["absence_type"].apply(availability_status)
    enriched["role_impact"] = enriched.apply(role_impact, axis=1)
    enriched["market_impact_note"] = enriched.apply(lambda row: player_market_note(row, text(row.get("role_impact"))), axis=1)
    enriched["injury_shock_severity"] = enriched.apply(lambda row: severity(row, text(row.get("role_impact"))), axis=1)
    enriched["player_rating_summary"] = enriched.apply(rating_summary, axis=1)
    return enriched


def build_fixture_impacts(coverage: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_players = players[players.get("key_player_flag", 0).fillna(0).astype(int) == 1] if not players.empty else pd.DataFrame()
    grouped = {key: group for key, group in key_players.groupby("fixture_key")} if not key_players.empty else {}
    for _, fixture in coverage.iterrows():
        fixture_key = text(fixture.get("fixture_key"))
        group = grouped.get(fixture_key, pd.DataFrame())
        high_or_medium = 0
        if not group.empty:
            high_or_medium = int(group["injury_shock_severity"].astype(str).str.contains("HIGH|MEDIUM", regex=True).sum())
        item = fixture.to_dict()
        item.update(
            {
                "key_absences": int(len(group)),
                "high_or_medium": high_or_medium,
                "key_players": "; ".join(group.get("player_name", pd.Series(dtype=str)).dropna().astype(str).tolist()),
                "shock_roles": "|".join(sorted(set(group.get("role_impact", pd.Series(dtype=str)).dropna().astype(str).tolist()))),
                "market_impact": "; ".join(
                    sorted(set(note for note in group.get("market_impact_note", pd.Series(dtype=str)).dropna().astype(str).tolist() if note))
                ),
                "ftr_injury_impact": summarize_market(group, "FTR"),
                "btts_injury_impact": summarize_market(group, "BTTS"),
                "ou25_injury_impact": summarize_market(group, "OU25"),
            }
        )
        item["fixture_injury_market_adjustment"] = overall_adjustment(pd.Series(item))
        rows.append(item)
    return pd.DataFrame(rows)


def write_report(path: Path, fixture_impacts: pd.DataFrame, player_impacts: pd.DataFrame, coverage_csv: Path, player_csv: Path) -> None:
    key_player_rows = player_impacts[player_impacts.get("key_player_flag", 0).fillna(0).astype(int) == 1] if not player_impacts.empty else pd.DataFrame()
    lines = [
        "# Injury Shock Market Impact",
        "",
        "Reporting layer only. This recalculates injury and availability impacts against FTR, BTTS, and Over 2.5 before website-brain compilation.",
        "",
        "## Source Inputs",
        "",
        f"- coverage csv: `{coverage_csv}`",
        f"- player impact csv: `{player_csv}`",
        f"- generated at: `{utc_now()}`",
        "",
        "## Fixture Market Impact Summary",
        "",
    ]
    display_cols = [
        "match_date",
        "league",
        "home_team",
        "away_team",
        "deploy_markets",
        "deploy_picks",
        "ftr_injury_impact",
        "btts_injury_impact",
        "ou25_injury_impact",
        "fixture_injury_market_adjustment",
        "key_absences",
        "key_players",
        "market_impact",
    ]
    cols = [col for col in display_cols if col in fixture_impacts.columns]
    lines.append(markdown_table(fixture_impacts[cols]) if cols else "_No fixture impact rows._")
    lines.extend(["", "## Key Player Absence Detail", ""])
    player_cols = [
        "match_date",
        "league",
        "home_team",
        "away_team",
        "deploy_markets",
        "deploy_picks",
        "side",
        "team",
        "player_name",
        "availability_impact_status",
        "absence_type",
        "reason",
        "injury_shock_severity",
        "role_impact",
        "position",
        "player_rating_summary",
        "market_impact_note",
    ]
    cols = [col for col in player_cols if col in key_player_rows.columns]
    lines.append(markdown_table(key_player_rows[cols]) if cols else "_No key-player injury impact rows._")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = text(row.get(col)).replace("|", "/")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    coverage_arg = args.coverage_csv or latest_default_source(DEFAULT_COVERAGE_NAME)
    player_arg = args.player_impact_csv or latest_default_source(DEFAULT_PLAYER_NAME)
    coverage_csv = coverage_arg if coverage_arg.is_absolute() else ROOT / coverage_arg
    player_csv = player_arg if player_arg.is_absolute() else ROOT / player_arg
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    coverage = read_csv(coverage_csv)
    players = enrich_player_rows(read_csv(player_csv))
    fixture_impacts = build_fixture_impacts(coverage, players)

    fixture_out = outdir / f"{args.prefix}_FIXTURE.csv"
    player_out = outdir / f"{args.prefix}_PLAYER.csv"
    report_out = outdir / f"{args.prefix}_REPORT.md"
    fixture_impacts.to_csv(fixture_out, index=False)
    players.to_csv(player_out, index=False)
    write_report(report_out, fixture_impacts, players, coverage_csv, player_csv)

    print(f"WROTE {fixture_out}")
    print(f"WROTE {player_out}")
    print(f"WROTE {report_out}")
    if not fixture_impacts.empty:
        cols = [
            "fixture_key",
            "ftr_injury_impact",
            "btts_injury_impact",
            "ou25_injury_impact",
            "fixture_injury_market_adjustment",
            "key_absences",
            "key_players",
        ]
        print(fixture_impacts[[col for col in cols if col in fixture_impacts.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
