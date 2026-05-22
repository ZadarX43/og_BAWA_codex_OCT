#!/usr/bin/env python3
"""Build a research-only Team Over 1.5 Goals shortlist.

This is not a deploy board and does not require bookmaker prices. It combines
model goal-mass fields from allmarkets with the publish-safe team intelligence
signals to produce a manual-review shortlist for acca construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLMARKETS = ROOT / "predictions_output" / "2026-05-14_imp20_may14_to_may19" / "BOOKIE_IMP20_ALLMARKETS_2026-05-14_to_2026-05-19.csv"
DEFAULT_INTEL_CARDS = ROOT / "reports" / "latest" / "live_weekend_model_intelligence_compare_2026_05_14_to_2026_05_19" / "live_fixture_intelligence_cards.csv"
DEFAULT_INTEL_ROWS = ROOT / "reports" / "latest" / "live_weekend_model_intelligence_compare_2026_05_14_to_2026_05_19" / "live_model_intelligence_rows.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "team_goals_15_shortlist_2026_05_14_to_2026_05_19"


def num(value: Any, default: float = 0.0) -> float:
    try:
        parsed = pd.to_numeric(value, errors="coerce")
    except Exception:
        return default
    if pd.isna(parsed):
        return default
    return float(parsed)


def scalar(row: pd.Series, key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def score_scale(value: Any, default: float = 0.0) -> float:
    parsed = num(value, default)
    if 0.0 < parsed <= 1.5:
        return parsed * 100.0
    return parsed


def tier_label(score: float, pois_ge2: float, intel_pick: str, conflict_flags: list[str]) -> str:
    if conflict_flags:
        return "AVOID"
    if score >= 74 and pois_ge2 >= 0.45 and intel_pick == "OVER15":
        return "A_TIER"
    if score >= 66 and pois_ge2 >= 0.38 and intel_pick in {"OVER15", "WATCH"}:
        return "B_TIER"
    if score >= 58 and pois_ge2 >= 0.32:
        return "WATCH"
    return "LOW_PRIORITY"


def scoreline_goals(scoreline: Any) -> tuple[int | None, int | None]:
    text = str(scoreline or "").strip()
    if "-" not in text:
        return None, None
    left, right = text.split("-", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None, None


def cs_side_mass(row: pd.Series, prefix: str) -> dict[str, Any]:
    is_home = prefix == "home"
    top_ge2_mass = 0.0
    top_exact1_mass = 0.0
    top_zero_mass = 0.0
    top_ge2_count = 0
    top_scorelines: list[str] = []

    for idx in (1, 2, 3):
        scoreline = scalar(row, f"cs{idx}", "")
        mass = num(row.get(f"cs{idx}_p"))
        home_goals, away_goals = scoreline_goals(scoreline)
        if home_goals is None or away_goals is None:
            continue
        team_goals = home_goals if is_home else away_goals
        if team_goals >= 2:
            top_ge2_mass += mass
            top_ge2_count += 1
            top_scorelines.append(str(scoreline))
        elif team_goals == 1:
            top_exact1_mass += mass
        else:
            top_zero_mass += mass

    return {
        "cs_side_ge2_top3_mass": top_ge2_mass,
        "cs_side_ge2_top3_count": top_ge2_count,
        "cs_side_exact1_top3_mass": top_exact1_mass,
        "cs_side_zero_top3_mass": top_zero_mass,
        "cs_side_ge2_top3_scorelines": ",".join(top_scorelines),
    }


def build_reason(row: dict[str, Any]) -> str:
    reasons = []
    if row["pois_team_ge2"] >= 0.45:
        reasons.append("POIS_GE2_STRONG")
    elif row["pois_team_ge2"] >= 0.38:
        reasons.append("POIS_GE2_SOLID")
    if row["team_ge2_confidence"] >= 65:
        reasons.append("MODEL_GE2_CONFIDENCE")
    if row["cs_side_ge2_top3_mass"] >= 0.08:
        reasons.append("CS_TOP3_GE2_SUPPORT")
    if row["mass_4plus_goals"] >= 0.28:
        reasons.append("CS_4PLUS_ENVIRONMENT")
    if row["both_teams_2plus_mass"] >= 0.16:
        reasons.append("CS_BOTH_TEAMS_2PLUS")
    if row["team_goals_15_signal_pick"] == "OVER15":
        reasons.append("INTEL_OVER15")
    if row["team_goals_15_signal_score"] >= 65:
        reasons.append("INTEL_SCORE_STRONG")
    if row["team_attack_flow"] >= 65:
        reasons.append("ATTACK_FLOW")
    if row["team_goal_heat"] >= 65:
        reasons.append("GOAL_HEAT")
    if row["team_first_strike"] >= 65:
        reasons.append("FIRST_STRIKE")
    if row["opponent_defensive_lock"] < 60:
        reasons.append("OPP_LOCK_LOW")
    if row["opponent_chaos"] >= 58:
        reasons.append("OPP_CHAOS")
    if row["shape_flags"]:
        reasons.append(str(row["shape_flags"]).replace(";", "|"))
    return "|".join(dict.fromkeys(reasons)) or "MODEL_CONTEXT_ONLY"


def side_row(base: pd.Series, card: pd.Series | None, side: str) -> dict[str, Any]:
    is_home = side == "home"
    prefix = "home" if is_home else "away"
    opp_prefix = "away" if is_home else "home"
    team = scalar(base, "home_team_name") if is_home else scalar(base, "away_team_name")
    opponent = scalar(base, "away_team_name") if is_home else scalar(base, "home_team_name")

    lambda_team = num(base.get(f"lambda_{prefix}"))
    pois_team_ge2 = max(num(base.get(f"pois_{prefix}_ge2")), num(base.get(f"p_{prefix}_ge2")))
    team_ge2_confidence = score_scale(base.get(f"{prefix}_ge2_confidence"))
    candidate_flag = int(num(base.get(f"{prefix}_team_ge2_candidate_flag")))
    cs_mass = cs_side_mass(base, prefix)
    cs_side_ge2_top3_mass = cs_mass["cs_side_ge2_top3_mass"]
    cs_side_ge2_top3_count = cs_mass["cs_side_ge2_top3_count"]
    cs_side_exact1_top3_mass = cs_mass["cs_side_exact1_top3_mass"]
    cs_side_zero_top3_mass = cs_mass["cs_side_zero_top3_mass"]
    cs_mass_over25 = num(base.get("cs_mass_over25"))
    both_teams_2plus_mass = num(base.get("both_teams_2plus_mass"))
    mass_4plus_goals = num(base.get("mass_4plus_goals"))
    mass_0_1_goals = num(base.get("mass_0_goals")) + num(base.get("mass_1_goal"))

    card = card if card is not None else pd.Series(dtype=object)
    intel_pick = str(card.get(f"{prefix}_team_goals_15", "WATCH") or "WATCH")
    intel_score = num(card.get(f"{prefix}_team_goals_15_score"), 50.0)
    attack_flow = num(card.get(f"{prefix}_attack_flow"), 50.0)
    goal_heat = num(card.get(f"{prefix}_goal_heat"), 50.0)
    first_strike = num(card.get(f"{prefix}_first_strike"), 50.0)
    opponent_lock = num(card.get(f"{opp_prefix}_defensive_lock"), 50.0)
    opponent_chaos = num(card.get(f"{opp_prefix}_chaos"), 50.0)

    model_score = (
        42.0
        + pois_team_ge2 * 45.0
        + min(lambda_team, 2.8) * 5.0
        + max(team_ge2_confidence - 50.0, 0.0) * 0.25
        + candidate_flag * 4.0
    )
    cs_component = (
        cs_side_ge2_top3_mass * 120.0
        + cs_side_ge2_top3_count * 3.5
        + max(cs_mass_over25 - 0.52, 0.0) * 18.0
        + max(mass_4plus_goals - 0.26, 0.0) * 24.0
        + max(both_teams_2plus_mass - 0.15, 0.0) * 16.0
        - cs_side_exact1_top3_mass * 32.0
        - cs_side_zero_top3_mass * 40.0
        - max(mass_0_1_goals - 0.30, 0.0) * 22.0
    )
    intel_component = (
        (intel_score - 50.0) * 0.42
        + (attack_flow - 60.0) * 0.18
        + (goal_heat - 60.0) * 0.14
        + (first_strike - 60.0) * 0.14
        - max(opponent_lock - 65.0, 0.0) * 0.24
        + max(opponent_chaos - 55.0, 0.0) * 0.10
    )
    shortlist_score = max(0.0, min(100.0, model_score + cs_component + intel_component))

    conflicts = []
    if intel_pick == "UNDER15":
        conflicts.append("INTEL_UNDER15")
    if opponent_lock >= 76 and pois_team_ge2 < 0.45:
        conflicts.append("HIGH_OPP_LOCK")
    if str(card.get("team_signal_ou25", "")).upper() == "UNDER25" and lambda_team < 1.65:
        conflicts.append("LOW_EVENT_TEAM_TOTAL")
    if cs_side_ge2_top3_count == 0 and pois_team_ge2 < 0.40 and lambda_team < 1.55:
        conflicts.append("CS_NO_TOP3_TEAM_GE2")
    if mass_0_1_goals >= 0.40 and pois_team_ge2 < 0.45:
        conflicts.append("LOW_GOAL_MASS_ENVIRONMENT")

    out = {
        "fixture_key": scalar(base, "fixture_key"),
        "match_date": scalar(base, "match_date"),
        "league": scalar(base, "league"),
        "side": side.upper(),
        "team": team,
        "opponent": opponent,
        "team_goals_15_pick": "OVER15",
        "lambda_team": round(lambda_team, 4),
        "pois_team_ge2": round(pois_team_ge2, 4),
        "team_ge2_confidence": round(team_ge2_confidence, 2),
        "team_ge2_candidate_flag": candidate_flag,
        "cs_side_ge2_top3_mass": round(cs_side_ge2_top3_mass, 4),
        "cs_side_ge2_top3_count": int(cs_side_ge2_top3_count),
        "cs_side_exact1_top3_mass": round(cs_side_exact1_top3_mass, 4),
        "cs_side_zero_top3_mass": round(cs_side_zero_top3_mass, 4),
        "cs_side_ge2_top3_scorelines": cs_mass["cs_side_ge2_top3_scorelines"],
        "cs_mass_over25": round(cs_mass_over25, 4),
        "both_teams_2plus_mass": round(both_teams_2plus_mass, 4),
        "mass_4plus_goals": round(mass_4plus_goals, 4),
        "mass_0_1_goals": round(mass_0_1_goals, 4),
        "team_goals_15_signal_pick": intel_pick,
        "team_goals_15_signal_score": round(intel_score, 2),
        "team_attack_flow": round(attack_flow, 2),
        "team_goal_heat": round(goal_heat, 2),
        "team_first_strike": round(first_strike, 2),
        "opponent_defensive_lock": round(opponent_lock, 2),
        "opponent_chaos": round(opponent_chaos, 2),
        "shape_flags": scalar(card, "shape_flags", ""),
        "team_signal_ftr": scalar(card, "team_signal_ftr", ""),
        "team_signal_ou25": scalar(card, "team_signal_ou25", ""),
        "team_signal_btts": scalar(card, "team_signal_btts", ""),
        "shortlist_score": round(shortlist_score, 2),
        "conflict_flags": "|".join(conflicts),
    }
    out["shortlist_tier"] = tier_label(shortlist_score, pois_team_ge2, intel_pick, conflicts)
    out["reason"] = build_reason(out)
    out["guardrail"] = "RESEARCH_ONLY|NO_BOOKMAKER_ODDS|NO_DEPLOY_PROMOTION"
    return out


def row_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or "fixture_key" not in df.columns:
        return {}
    return {
        str(row.fixture_key): row._asdict()
        for row in df.drop_duplicates("fixture_key", keep="first").itertuples(index=False)
    }


def build_shortlist(allmarkets: pd.DataFrame, cards: pd.DataFrame, intelligence_rows: pd.DataFrame) -> pd.DataFrame:
    fixtures = allmarkets.drop_duplicates("fixture_key", keep="first").copy()
    card_lookup = row_lookup(cards)
    metric_lookup = row_lookup(intelligence_rows)
    rows = []
    for _, fixture in fixtures.iterrows():
        fixture_key = str(fixture.get("fixture_key") or "")
        card_data = {
            **metric_lookup.get(fixture_key, {}),
            **card_lookup.get(fixture_key, {}),
        }
        card = pd.Series(card_data) if card_data else None
        rows.append(side_row(fixture, card, "home"))
        rows.append(side_row(fixture, card, "away"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    tier_rank = {"A_TIER": 4, "B_TIER": 3, "WATCH": 2, "LOW_PRIORITY": 1, "AVOID": 0}
    out["_tier_rank"] = out["shortlist_tier"].map(tier_rank).fillna(0)
    return out.sort_values(["_tier_rank", "shortlist_score", "pois_team_ge2"], ascending=[False, False, False]).drop(columns=["_tier_rank"])


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in cols) + " |")
    return "\n".join(lines)


def acca_pool(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows[
        rows["shortlist_tier"].isin(["A_TIER", "B_TIER"])
        & rows["team_goals_15_signal_pick"].eq("OVER15")
        & rows["team_signal_ou25"].ne("UNDER25")
        & rows["pois_team_ge2"].ge(0.45)
        & rows["lambda_team"].ge(1.55)
        & rows["opponent_defensive_lock"].le(70)
    ].copy()


def write_report(outdir: Path, rows: pd.DataFrame, allmarkets_path: Path, cards_path: Path, intelligence_rows_path: Path) -> None:
    counts = rows.groupby(["shortlist_tier"], dropna=False).size().reset_index(name="rows") if not rows.empty else pd.DataFrame()
    acca_rows = acca_pool(rows)
    top_cols = [
        "match_date",
        "league",
        "team",
        "opponent",
        "side",
        "shortlist_tier",
        "shortlist_score",
        "lambda_team",
        "pois_team_ge2",
        "team_ge2_confidence",
        "cs_side_ge2_top3_mass",
        "cs_side_ge2_top3_count",
        "cs_side_ge2_top3_scorelines",
        "cs_mass_over25",
        "both_teams_2plus_mass",
        "mass_4plus_goals",
        "team_goals_15_signal_pick",
        "team_goals_15_signal_score",
        "reason",
    ]
    lines = [
        "# Team Goals 1.5 Shortlist",
        "",
        "Research-only shortlist for Team Over 1.5 Goals. This is not a priced market board and must not be treated as deploy output.",
        "",
        "## Inputs",
        f"- allmarkets: `{allmarkets_path}`",
        f"- intelligence cards: `{cards_path}`",
        f"- intelligence rows: `{intelligence_rows_path}`",
        "",
        "## Guardrails",
        "- No bookmaker odds or EV claim.",
        "- No deploy routing, slips, or production rulebook mutation.",
        "- Use for manual acca research only until team-goals odds coverage exists.",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Acca Pool",
        "Stricter pool: A/B tier, intelligence OVER15, no UNDER25 fixture context, model GE2 mass >= 45%, lambda >= 1.55, opponent defensive lock <= 70.",
        "",
        markdown_table(acca_rows[[c for c in top_cols if c in acca_rows.columns]], max_rows=40),
        "",
        "## Top Shortlist",
        markdown_table(rows[[c for c in top_cols if c in rows.columns]], max_rows=60),
        "",
    ]
    (outdir / "TEAM_GOALS_15_SHORTLIST.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allmarkets", type=Path, default=DEFAULT_ALLMARKETS)
    parser.add_argument("--intelligence-cards", type=Path, default=DEFAULT_INTEL_CARDS)
    parser.add_argument("--intelligence-rows", type=Path, default=DEFAULT_INTEL_ROWS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-date", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allmarkets = pd.read_csv(args.allmarkets, low_memory=False)
    cards = pd.read_csv(args.intelligence_cards, low_memory=False) if args.intelligence_cards.exists() else pd.DataFrame()
    intelligence_rows = pd.read_csv(args.intelligence_rows, low_memory=False) if args.intelligence_rows.exists() else pd.DataFrame()
    rows = build_shortlist(allmarkets, cards, intelligence_rows)
    if args.min_date:
        rows["_match_date_dt"] = pd.to_datetime(rows["match_date"], errors="coerce")
        rows = rows[rows["_match_date_dt"].ge(pd.Timestamp(args.min_date))].drop(columns=["_match_date_dt"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.outdir / "TEAM_GOALS_15_SHORTLIST.csv", index=False)
    rows[rows["shortlist_tier"].isin(["A_TIER", "B_TIER"])].to_csv(args.outdir / "TEAM_GOALS_15_CORE_SHORTLIST.csv", index=False)
    acca_rows = acca_pool(rows)
    acca_rows.to_csv(args.outdir / "TEAM_GOALS_15_ACCA_POOL.csv", index=False)
    write_report(args.outdir, rows, args.allmarkets, args.intelligence_cards, args.intelligence_rows)
    summary = {
        "rows": int(len(rows)),
        "fixtures": int(rows["fixture_key"].nunique()) if not rows.empty else 0,
        "acca_pool_rows": int(len(acca_rows)),
        "acca_pool_fixtures": int(acca_rows["fixture_key"].nunique()) if not acca_rows.empty else 0,
        "tier_counts": rows["shortlist_tier"].value_counts().to_dict() if not rows.empty else {},
        "outputs": str(args.outdir.relative_to(ROOT)) if args.outdir.is_relative_to(ROOT) else str(args.outdir),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
