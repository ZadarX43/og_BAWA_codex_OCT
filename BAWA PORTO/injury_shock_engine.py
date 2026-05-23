from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MOBILITY_TERMS = {
    "hamstring",
    "calf",
    "thigh",
    "ankle",
    "groin",
    "muscle",
    "knee",
}

VOLATILITY_TERMS = {
    "doubt",
    "doubtful",
    "late fitness",
    "fitness test",
    "not 100",
    "knock",
    "returning",
    "illness",
}

ABSENT_TERMS = {
    "out",
    "injury",
    "injured",
    "suspension",
    "suspended",
    "unavailable",
}


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        if isinstance(value, float) and math.isnan(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def role_from_position(position: Any) -> str:
    text = str(position or "").upper().strip()
    if text.startswith("G"):
        return "goalkeeper"
    if text.startswith("D"):
        return "defender"
    if text.startswith("M"):
        return "midfielder"
    if text.startswith("F"):
        return "attacker"
    return "unknown"


def absence_state(row: pd.Series) -> str:
    text = " ".join(
        [
            norm_text(row.get("absence_type")),
            norm_text(row.get("reason")),
            norm_text(row.get("status")),
        ]
    )
    if any(term in text for term in VOLATILITY_TERMS):
        return "doubt_or_late_test"
    if any(term in text for term in ABSENT_TERMS):
        return "absent"
    return "reported"


def mobility_risk(row: pd.Series) -> float:
    text = " ".join(
        [
            norm_text(row.get("absence_type")),
            norm_text(row.get("reason")),
            norm_text(row.get("status")),
        ]
    )
    score = 0.0
    if any(term in text for term in MOBILITY_TERMS):
        score += 10.0
    if any(term in text for term in VOLATILITY_TERMS):
        score += 8.0
    return score


def player_recent_metrics(records: list[dict[str, Any]]) -> dict[str, float | str]:
    sample5 = records[-5:]
    sample10 = records[-10:]
    minutes5 = sum(safe_float(r.get("minutes")) for r in sample5)
    minutes10 = sum(safe_float(r.get("minutes")) for r in sample10)
    starts10 = sum(safe_float(r.get("started_flag")) for r in sample10)
    latest_position = next((r.get("position") for r in reversed(sample10) if str(r.get("position") or "").strip()), "")
    role = role_from_position(latest_position)
    return {
        "role": role,
        "minutes_l5_total": minutes5,
        "minutes_l10_total": minutes10,
        "starter_rate_l10": starts10 / len(sample10) if sample10 else 0.0,
        "goals_per90_l5": sum(safe_float(r.get("goals")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
        "assists_per90_l5": sum(safe_float(r.get("assists")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
        "shots_per90_l5": sum(safe_float(r.get("shots_total")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
        "sot_per90_l5": sum(safe_float(r.get("shots_on_target")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
        "tackles_per90_l5": sum(safe_float(r.get("tackles")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
        "saves_per90_l5": sum(safe_float(r.get("saves")) for r in sample5) * 90.0 / minutes5 if minutes5 else 0.0,
    }


def player_function_label(metrics: dict[str, float | str]) -> str:
    role = str(metrics.get("role") or "unknown")
    goals = safe_float(metrics.get("goals_per90_l5"))
    assists = safe_float(metrics.get("assists_per90_l5"))
    shots = safe_float(metrics.get("shots_per90_l5"))
    tackles = safe_float(metrics.get("tackles_per90_l5"))
    saves = safe_float(metrics.get("saves_per90_l5"))
    if role == "goalkeeper":
        return "goalkeeper" if saves >= 1.5 else "backup_goalkeeper"
    if role == "attacker":
        if goals >= 0.35 or shots >= 2.2:
            return "primary_goal_threat"
        if assists >= 0.25:
            return "creator_or_wide_forward"
        return "attacking_depth"
    if role == "midfielder":
        if tackles >= 2.0:
            return "midfield_ball_winner"
        if assists >= 0.2:
            return "central_creator"
        return "midfield_rotation"
    if role == "defender":
        if tackles >= 1.6:
            return "defensive_duel_anchor"
        return "defensive_depth"
    return "unknown_function"


def function_impact(metrics: dict[str, float | str]) -> float:
    minutes_weight = min(safe_float(metrics.get("minutes_l5_total")) / 450.0, 1.0)
    starter_weight = safe_float(metrics.get("starter_rate_l10"))
    return 0.55 + (0.30 * minutes_weight) + (0.25 * starter_weight)


def score_absence(row: pd.Series, metrics: dict[str, float | str]) -> dict[str, float | str]:
    role = str(metrics.get("role") or "unknown")
    state = absence_state(row)
    function = player_function_label(metrics)
    impact = function_impact(metrics)
    if state == "doubt_or_late_test":
        impact *= 0.72
    elif state == "reported":
        impact *= 0.55

    attack = 0.0
    midfield = 0.0
    defence = 0.0
    keeper = 0.0

    if function == "primary_goal_threat":
        attack += 28.0 * impact
    elif function == "creator_or_wide_forward":
        attack += 22.0 * impact
    elif function == "attacking_depth":
        attack += 11.0 * impact
    elif function == "central_creator":
        midfield += 15.0 * impact
        attack += 10.0 * impact
    elif function == "midfield_ball_winner":
        midfield += 23.0 * impact
    elif function == "midfield_rotation":
        midfield += 10.0 * impact
    elif function == "defensive_duel_anchor":
        defence += 22.0 * impact
    elif function == "defensive_depth":
        defence += 12.0 * impact
    elif function == "goalkeeper":
        keeper += 35.0 * impact
        defence += 12.0 * impact
    elif role == "attacker":
        attack += 8.0 * impact
    elif role == "midfielder":
        midfield += 8.0 * impact
    elif role == "defender":
        defence += 8.0 * impact

    mobility = mobility_risk(row) * impact
    return {
        "role": role,
        "function": function,
        "state": state,
        "impact_weight": round(impact, 3),
        "attack_score": attack,
        "midfield_score": midfield,
        "defence_score": defence,
        "keeper_score": keeper,
        "mobility_score": mobility,
    }


def read_optional_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def team_side(row: pd.Series, team_id: int) -> str:
    if int(row.get("home_team_id")) == team_id:
        return "home"
    if int(row.get("away_team_id")) == team_id:
        return "away"
    return "unknown"


def build_context_index(context_csv: str | None) -> dict[str, dict[str, Any]]:
    context = read_optional_csv(context_csv)
    if context.empty or "fixture_key" not in context.columns:
        return {}
    return {str(row["fixture_key"]): row.to_dict() for _, row in context.iterrows()}


def context_volatility_score(context: dict[str, Any]) -> tuple[float, list[str]]:
    if not context:
        return 0.0, []
    flags: list[str] = []
    score = 0.0
    flag_cols = [
        "title_won_flag",
        "cup_final_ahead_flag",
        "manager_exit_noise_flag",
        "europe_secured_flag",
        "relegation_safe_flag",
        "must_win_flag",
        "rotation_risk_flag",
        "farewell_match_flag",
    ]
    for col in flag_cols:
        value = safe_float(context.get(col))
        if value > 0:
            flags.append(col.upper())
            score += 8.0 if col != "must_win_flag" else 4.0
    text = norm_text(context.get("context_note") or context.get("notes"))
    for token in ["manager", "rotation", "rest", "secured", "safe", "final", "pressure", "farewell"]:
        if token in text:
            score += 3.0
    return clamp(score), flags


def market_adjustments(home: dict[str, float], away: dict[str, float], context_score: float) -> dict[str, float | str | int]:
    attack_total = home["attack_absence_score"] + away["attack_absence_score"]
    defence_total = home["defence_absence_score"] + away["defence_absence_score"] + home["keeper_absence_score"] + away["keeper_absence_score"]
    fav_side = "home" if home["team_absence_total"] < away["team_absence_total"] else "away"
    if abs(home["team_absence_total"] - away["team_absence_total"]) < 5:
        fav_side = "none"

    ou25 = (defence_total * 0.18) - (attack_total * 0.24) - (context_score * 0.04)
    btts = (defence_total * 0.14) - (max(home["attack_absence_score"], away["attack_absence_score"]) * 0.22)
    ftr_vol = max(home["team_absence_total"], away["team_absence_total"]) * 0.18 + context_score * 0.16
    goal_model = (defence_total * 0.10) - (attack_total * 0.20)

    shock_score = max(home["team_absence_total"], away["team_absence_total"], context_score)
    deploy_warning = int(shock_score >= 35 or max(home["attack_absence_score"], away["attack_absence_score"]) >= 28 or ftr_vol >= 12)

    return {
        "goal_model_adjustment": round(goal_model, 2),
        "btts_adjustment": round(btts, 2),
        "ou25_adjustment": round(ou25, 2),
        "ftr_volatility_adjustment": round(ftr_vol, 2),
        "deploy_warning_flag": deploy_warning,
        "absence_edge_side": fav_side,
    }


def build_team_summary(absences: pd.DataFrame, history: dict[int, list[dict[str, Any]]]) -> tuple[dict[str, float], list[str]]:
    summary = {
        "attack_absence_score": 0.0,
        "midfield_absence_score": 0.0,
        "defence_absence_score": 0.0,
        "keeper_absence_score": 0.0,
        "mobility_risk_score": 0.0,
        "lineup_confidence_score": 100.0,
        "injury_news_severity": 0.0,
        "team_absence_total": 0.0,
    }
    reasons: list[str] = []
    if absences.empty:
        return summary, reasons

    for _, injury in absences.iterrows():
        player_id = int(safe_float(injury.get("player_id"), -1))
        metrics = player_recent_metrics(history.get(player_id, []))
        scored = score_absence(injury, metrics)
        summary["attack_absence_score"] += safe_float(scored["attack_score"])
        summary["midfield_absence_score"] += safe_float(scored["midfield_score"])
        summary["defence_absence_score"] += safe_float(scored["defence_score"])
        summary["keeper_absence_score"] += safe_float(scored["keeper_score"])
        summary["mobility_risk_score"] += safe_float(scored["mobility_score"])
        player = str(injury.get("player_name") or "Unknown player")
        if safe_float(scored["impact_weight"]) >= 0.7 or scored["function"] in {"primary_goal_threat", "goalkeeper", "midfield_ball_winner"}:
            reasons.append(
                f"{player}:{scored['function']}:{scored['state']}:{scored['impact_weight']}"
            )

    summary["attack_absence_score"] = clamp(summary["attack_absence_score"])
    summary["midfield_absence_score"] = clamp(summary["midfield_absence_score"])
    summary["defence_absence_score"] = clamp(summary["defence_absence_score"])
    summary["keeper_absence_score"] = clamp(summary["keeper_absence_score"])
    summary["mobility_risk_score"] = clamp(summary["mobility_risk_score"])
    summary["team_absence_total"] = clamp(
        summary["attack_absence_score"] * 0.36
        + summary["midfield_absence_score"] * 0.26
        + summary["defence_absence_score"] * 0.26
        + summary["keeper_absence_score"] * 0.32
        + summary["mobility_risk_score"] * 0.12
    )
    summary["injury_news_severity"] = clamp(summary["team_absence_total"] + len(absences) * 2.0)
    summary["lineup_confidence_score"] = clamp(100.0 - summary["team_absence_total"] - summary["mobility_risk_score"] * 0.2)
    return summary, reasons[:6]


def dedupe_absence_sources(injuries: pd.DataFrame) -> pd.DataFrame:
    """Avoid triple-counting the same absence when fixture/team/league scopes overlap."""
    if injuries.empty:
        return injuries
    out = injuries.copy()
    if "availability_key" not in out.columns:
        out["availability_key"] = (
            out.get("team_id", "").astype(str)
            + ":"
            + out.get("player_id", "").astype(str)
            + ":"
            + out.get("absence_type", "").astype(str)
            + ":"
            + out.get("reason", "").astype(str)
        )
    source_rank = {
        "league_season": 0,
        "league_date": 1,
        "team_season": 2,
        "fixture": 3,
    }
    out["_scope_rank"] = out.get("source_scope", pd.Series("", index=out.index)).astype(str).map(source_rank).fillna(5)
    out["_first_seen_sort"] = pd.to_datetime(out.get("availability_first_seen_ts_utc", ""), errors="coerce", utc=True)
    out = out.sort_values(["fixture_id", "availability_key", "_first_seen_sort", "_scope_rank"], na_position="last")
    return out.drop_duplicates(subset=["fixture_id", "availability_key"], keep="first").drop(columns=["_scope_rank", "_first_seen_sort"], errors="ignore")


def build_injury_shock_board(
    fixtures_csv: str,
    injuries_csv: str,
    player_stats_csv: str,
    output_csv: str,
    output_md: str | None = None,
    context_csv: str | None = None,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv, low_memory=False)
    injuries = read_optional_csv(injuries_csv)
    player_stats = pd.read_csv(player_stats_csv, low_memory=False)
    context_by_fixture = build_context_index(context_csv)

    if injuries.empty:
        injuries = pd.DataFrame(columns=["fixture_id", "team_id", "player_id", "player_name", "absence_type", "reason", "status"])
    else:
        injuries = dedupe_absence_sources(injuries)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures.get("kickoff_ts_utc"), errors="coerce", utc=True)
    player_stats = player_stats.merge(fixtures[["fixture_id", "kickoff_ts_utc"]], on="fixture_id", how="left")
    player_stats = player_stats.sort_values(["kickoff_ts_utc", "fixture_id", "player_id"]).reset_index(drop=True)

    stats_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in player_stats.iterrows():
        stats_by_fixture[int(row["fixture_id"])].append(row.to_dict())

    injuries_by_fixture = {int(fid): df.copy() for fid, df in injuries.groupby("fixture_id")} if not injuries.empty else {}
    history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, fixture in fixtures.sort_values(["kickoff_ts_utc", "fixture_id"]).iterrows():
        fixture_id = int(fixture["fixture_id"])
        fx_inj = injuries_by_fixture.get(fixture_id, pd.DataFrame(columns=injuries.columns))
        home_team_id = int(fixture["home_team_id"])
        away_team_id = int(fixture["away_team_id"])

        home_summary, home_reasons = build_team_summary(fx_inj[fx_inj["team_id"] == home_team_id], history)
        away_summary, away_reasons = build_team_summary(fx_inj[fx_inj["team_id"] == away_team_id], history)
        context_score, context_flags = context_volatility_score(context_by_fixture.get(str(fixture.get("fixture_key")), {}))
        adjustments = market_adjustments(home_summary, away_summary, context_score)

        warning_tokens: list[str] = []
        if home_summary["attack_absence_score"] >= 28:
            warning_tokens.append("HOME_ATTACK_SHOCK")
        if away_summary["attack_absence_score"] >= 28:
            warning_tokens.append("AWAY_ATTACK_SHOCK")
        if home_summary["defence_absence_score"] + home_summary["keeper_absence_score"] >= 30:
            warning_tokens.append("HOME_DEFENCE_SPINE_SHOCK")
        if away_summary["defence_absence_score"] + away_summary["keeper_absence_score"] >= 30:
            warning_tokens.append("AWAY_DEFENCE_SPINE_SHOCK")
        if context_score >= 24:
            warning_tokens.append("MOTIVATION_VOLATILITY")
        if adjustments["deploy_warning_flag"]:
            warning_tokens.append("REQUIRE_LINEUP_CONFIRMATION")

        rows.append(
            {
                "fixture_id": fixture_id,
                "fixture_key": fixture.get("fixture_key"),
                "league": fixture.get("league"),
                "season": fixture.get("season"),
                "match_date": fixture.get("match_date"),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": fixture.get("home_team_name"),
                "away_team_name": fixture.get("away_team_name"),
                "home_attack_absence_score": round(home_summary["attack_absence_score"], 2),
                "away_attack_absence_score": round(away_summary["attack_absence_score"], 2),
                "home_midfield_absence_score": round(home_summary["midfield_absence_score"], 2),
                "away_midfield_absence_score": round(away_summary["midfield_absence_score"], 2),
                "home_defence_absence_score": round(home_summary["defence_absence_score"], 2),
                "away_defence_absence_score": round(away_summary["defence_absence_score"], 2),
                "home_keeper_absence_score": round(home_summary["keeper_absence_score"], 2),
                "away_keeper_absence_score": round(away_summary["keeper_absence_score"], 2),
                "home_mobility_risk_score": round(home_summary["mobility_risk_score"], 2),
                "away_mobility_risk_score": round(away_summary["mobility_risk_score"], 2),
                "home_lineup_confidence_score": round(home_summary["lineup_confidence_score"], 2),
                "away_lineup_confidence_score": round(away_summary["lineup_confidence_score"], 2),
                "home_injury_news_severity": round(home_summary["injury_news_severity"], 2),
                "away_injury_news_severity": round(away_summary["injury_news_severity"], 2),
                "motivation_volatility_score": round(context_score, 2),
                "goal_model_adjustment": adjustments["goal_model_adjustment"],
                "btts_adjustment": adjustments["btts_adjustment"],
                "ou25_adjustment": adjustments["ou25_adjustment"],
                "ftr_volatility_adjustment": adjustments["ftr_volatility_adjustment"],
                "deploy_warning_flag": adjustments["deploy_warning_flag"],
                "absence_edge_side": adjustments["absence_edge_side"],
                "warning_tokens": "|".join(dict.fromkeys(warning_tokens)),
                "home_absence_reasons": "|".join(home_reasons),
                "away_absence_reasons": "|".join(away_reasons),
                "context_flags": "|".join(context_flags),
            }
        )

        for rec in stats_by_fixture.get(fixture_id, []):
            history[int(rec["player_id"])].append(rec)

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    if output_md:
        write_markdown_summary(out, output_md)
    return out


def write_markdown_summary(df: pd.DataFrame, output_md: str) -> None:
    lines = [
        "# Injury Shock And Lineup Risk Board",
        "",
        "Research / pre-deploy warning layer only. This does not override `deploy_rulebook.py`.",
        "",
    ]
    warned = df[df["deploy_warning_flag"].astype(int).eq(1)].copy() if not df.empty else pd.DataFrame()
    lines.append(f"- fixtures_scored: {len(df)}")
    lines.append(f"- deploy_warning_fixtures: {len(warned)}")
    lines.append("")
    if warned.empty:
        lines.append("No deployment warning fixtures surfaced.")
    else:
        warned["sort_score"] = warned[
            [
                "home_injury_news_severity",
                "away_injury_news_severity",
                "motivation_volatility_score",
                "ftr_volatility_adjustment",
            ]
        ].max(axis=1)
        for _, row in warned.sort_values("sort_score", ascending=False).head(40).iterrows():
            lines.append(f"## {row['home_team_name']} vs {row['away_team_name']}")
            lines.append(f"- fixture_key: `{row['fixture_key']}`")
            lines.append(f"- warning_tokens: `{row['warning_tokens']}`")
            lines.append(
                "- scores: "
                f"home_attack={row['home_attack_absence_score']}, away_attack={row['away_attack_absence_score']}, "
                f"home_defence={row['home_defence_absence_score']}, away_defence={row['away_defence_absence_score']}, "
                f"motivation={row['motivation_volatility_score']}"
            )
            lines.append(
                "- market_adjustments: "
                f"goal={row['goal_model_adjustment']}, btts={row['btts_adjustment']}, "
                f"ou25={row['ou25_adjustment']}, ftr_vol={row['ftr_volatility_adjustment']}"
            )
            if str(row.get("home_absence_reasons") or ""):
                lines.append(f"- home_absence_reasons: `{row['home_absence_reasons']}`")
            if str(row.get("away_absence_reasons") or ""):
                lines.append(f"- away_absence_reasons: `{row['away_absence_reasons']}`")
            if str(row.get("context_flags") or ""):
                lines.append(f"- context_flags: `{row['context_flags']}`")
            lines.append("")
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OG Injury Shock & Lineup Risk warning features.")
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--injuries-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--context-csv", default=None, help="Optional fixture_key-level motivation/news flags CSV.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    board = build_injury_shock_board(
        fixtures_csv=args.fixtures_csv,
        injuries_csv=args.injuries_csv,
        player_stats_csv=args.player_stats_csv,
        context_csv=args.context_csv,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"WROTE: {args.output_csv} rows={len(board)}")
    if args.output_md:
        print(f"WROTE: {args.output_md}")
