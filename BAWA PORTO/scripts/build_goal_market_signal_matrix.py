#!/usr/bin/env python3
"""Build a goal-market signal matrix from publish-safe fixture intelligence.

This is a research/reporting layer. It does not alter deploy routing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_ROWS = ROOT / "reports" / "latest" / "weekend_prediction_intelligence_scoring" / "primary_prediction_score_rows.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "goal_market_signal_matrix"
DECISION_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_decision_intelligence"

METRIC_KEYS = {
    "goal_heat_rating": "goal_heat",
    "btts_pressure_rating": "btts_pressure",
    "attack_flow_rating": "attack_flow",
    "defensive_lock_rating": "defensive_lock",
    "first_strike_rating": "first_strike",
    "chaos_rating": "chaos",
}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_map(decision: dict[str, Any]) -> dict[str, dict[str, float]]:
    values = {short: {"home": 50.0, "away": 50.0, "delta": 0.0} for short in METRIC_KEYS.values()}
    for item in decision.get("team_faceoff_summary") or []:
        if not isinstance(item, dict):
            continue
        short = METRIC_KEYS.get(str(item.get("metric") or ""))
        if not short:
            continue
        home = float(item.get("home_value") or 0)
        away = float(item.get("away_value") or 0)
        values[short] = {"home": home, "away": away, "delta": home - away}
    return values


def support_tokens(decision: dict[str, Any]) -> set[str]:
    tokens = set()
    for key in ("supporting_layers", "caution_layers", "internal_reason_tokens"):
        for token in decision.get(key) or []:
            tokens.add(str(token or "").upper())
    return tokens


def player_goal_threat(decision: dict[str, Any], team: str) -> float:
    values = []
    for player in decision.get("key_player_drivers") or []:
        if str(player.get("team") or "") == team:
            values.append(float(player.get("driver_value") or player.get("power") or 0))
    return max(values) if values else 50.0


def actual_ftr(row: pd.Series) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    if float(home) > float(away):
        return "HOME"
    if float(home) < float(away):
        return "AWAY"
    return "DRAW"


def actual_ou25(row: pd.Series) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    return "OVER25" if float(home) + float(away) > 2.5 else "UNDER25"


def actual_btts(row: pd.Series) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    return "YES" if float(home) > 0 and float(away) > 0 else "NO"


def actual_team_goals_15(row: pd.Series, side: str) -> str:
    goals = row.get("home_score") if side == "home" else row.get("away_score")
    if pd.isna(goals):
        return ""
    return "OVER15" if float(goals) >= 2 else "UNDER15"


def state_from_score(score: float, positive_threshold: float = 62.0, avoid_threshold: float = 45.0) -> str:
    if score >= positive_threshold:
        return "BOOST"
    if score < avoid_threshold:
        return "AVOID"
    return "WATCH"


def ftr_signal(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    edge = (
        0.30 * metrics["attack_flow"]["delta"]
        + 0.24 * metrics["defensive_lock"]["delta"]
        + 0.26 * metrics["first_strike"]["delta"]
        + 0.16 * metrics["goal_heat"]["delta"]
    )
    chaos_drag = max(0.0, ((metrics["chaos"]["home"] + metrics["chaos"]["away"]) / 2.0) - 62.0) * 0.20
    confidence = clamp(50.0 + abs(edge) * 1.8 - chaos_drag)
    if edge >= 6:
        pick = "HOME"
    elif edge <= -6:
        pick = "AWAY"
    else:
        pick = "NO_EDGE"
    return {
        "market": "FTR",
        "signal_pick": pick,
        "signal_score": round(confidence, 2),
        "signal_state": "BOOST" if pick != "NO_EDGE" and confidence >= 62 else "WATCH" if pick != "NO_EDGE" else "NO_EDGE",
        "primary_driver": "attack_flow/defensive_lock/first_strike edge",
        "ftr_edge": round(edge, 2),
    }


def ou25_signal(metrics: dict[str, dict[str, float]], tokens: set[str]) -> dict[str, Any]:
    goal_avg = (metrics["goal_heat"]["home"] + metrics["goal_heat"]["away"]) / 2.0
    attack_avg = (metrics["attack_flow"]["home"] + metrics["attack_flow"]["away"]) / 2.0
    defence_avg = (metrics["defensive_lock"]["home"] + metrics["defensive_lock"]["away"]) / 2.0
    chaos_avg = (metrics["chaos"]["home"] + metrics["chaos"]["away"]) / 2.0
    score = 50.0 + (goal_avg - 60.0) * 0.35 + (attack_avg - 60.0) * 0.30 + (chaos_avg - 50.0) * 0.20 - (defence_avg - 65.0) * 0.25
    if "OVER25_HEAT_SUPPORT" in tokens:
        score += 7
    if "LINEUP_GOAL_PRESSURE_SUPPORT" in tokens:
        score += 5
    if "OVER25_HEAT_SOFT" in tokens or "DEFENSIVE_SUPPRESSION_RISK" in tokens:
        score -= 5
    score = clamp(score)
    state = state_from_score(score)
    return {
        "market": "OU25",
        "signal_pick": "OVER25" if state == "BOOST" else "UNDER25" if state == "AVOID" else "WATCH",
        "signal_score": round(score, 2),
        "signal_state": state,
        "primary_driver": "combined goal heat/attack flow/chaos less defensive suppression",
    }


def btts_signal(metrics: dict[str, dict[str, float]], tokens: set[str]) -> dict[str, Any]:
    btts_avg = (metrics["btts_pressure"]["home"] + metrics["btts_pressure"]["away"]) / 2.0
    attack_floor = min(metrics["attack_flow"]["home"], metrics["attack_flow"]["away"])
    defence_avg = (metrics["defensive_lock"]["home"] + metrics["defensive_lock"]["away"]) / 2.0
    one_sided_attack = abs(metrics["attack_flow"]["delta"])
    score = 50.0 + (btts_avg - 55.0) * 0.45 + (attack_floor - 55.0) * 0.30 - (defence_avg - 65.0) * 0.25 - one_sided_attack * 0.16
    if "BTTS_PRESSURE_SUPPORT" in tokens:
        score += 8
    if "H2H_BTTS_SUPPORT" in tokens:
        score += 3
    if "BTTS_PRESSURE_WEAK" in tokens:
        score -= 9
    if "DEFENSIVE_SUPPRESSION_RISK" in tokens:
        score -= 6
    score = clamp(score)
    state = state_from_score(score)
    return {
        "market": "BTTS",
        "signal_pick": "YES" if state == "BOOST" else "NO" if state == "AVOID" else "WATCH",
        "signal_score": round(score, 2),
        "signal_state": state,
        "primary_driver": "combined BTTS pressure and two-team attack floor",
    }


def team_goals_signal(metrics: dict[str, dict[str, float]], decision: dict[str, Any], row: pd.Series, side: str) -> dict[str, Any]:
    home_team = str(row.get("home_team") or "Home")
    away_team = str(row.get("away_team") or "Away")
    team = home_team if side == "home" else away_team
    own = "home" if side == "home" else "away"
    opp = "away" if side == "home" else "home"
    attack_vs_lock = metrics["attack_flow"][own] - metrics["defensive_lock"][opp]
    threat = player_goal_threat(decision, team)
    score = (
        50.0
        + attack_vs_lock * 0.45
        + (metrics["goal_heat"][own] - 60.0) * 0.25
        + (metrics["first_strike"][own] - 60.0) * 0.20
        + (metrics["chaos"][opp] - 50.0) * 0.15
        + (threat - 65.0) * 0.20
    )
    score = clamp(score)
    state = state_from_score(score)
    return {
        "market": f"{'HOME' if side == 'home' else 'AWAY'}_TEAM_GOALS_15",
        "signal_pick": "OVER15" if state == "BOOST" else "UNDER15" if state == "AVOID" else "WATCH",
        "signal_score": round(score, 2),
        "signal_state": state,
        "primary_driver": "team attack flow versus opponent defensive lock",
        "team": team,
    }


def shape_flags(metrics: dict[str, dict[str, float]]) -> list[str]:
    flags = []
    attack_avg = (metrics["attack_flow"]["home"] + metrics["attack_flow"]["away"]) / 2.0
    defence_avg = (metrics["defensive_lock"]["home"] + metrics["defensive_lock"]["away"]) / 2.0
    goal_avg = (metrics["goal_heat"]["home"] + metrics["goal_heat"]["away"]) / 2.0
    chaos_avg = (metrics["chaos"]["home"] + metrics["chaos"]["away"]) / 2.0
    btts_avg = (metrics["btts_pressure"]["home"] + metrics["btts_pressure"]["away"]) / 2.0
    if attack_avg >= 65 and defence_avg < 65:
        flags.append("high_attack_low_defence")
    if attack_avg >= 65 and defence_avg >= 70:
        flags.append("high_attack_high_defence")
    if metrics["attack_flow"]["home"] >= 70 and metrics["defensive_lock"]["away"] < 62:
        flags.append("home_attack_vs_low_away_lock")
    if metrics["attack_flow"]["away"] >= 70 and metrics["defensive_lock"]["home"] < 62:
        flags.append("away_attack_vs_low_home_lock")
    if abs(metrics["first_strike"]["delta"]) >= 8 and abs(metrics["attack_flow"]["delta"]) >= 6:
        flags.append("first_strike_plus_team_edge")
    if goal_avg >= 68 and chaos_avg >= 58:
        flags.append("high_goal_heat_high_chaos")
    if defence_avg >= 70 and btts_avg < 55:
        flags.append("high_defensive_lock_low_btts_pressure")
    return flags


def signal_hit(row: pd.Series, signal: dict[str, Any]) -> str:
    market = signal["market"]
    pick = signal["signal_pick"]
    if pick in {"WATCH", "NO_EDGE"}:
        return "no_call"
    if market == "FTR":
        actual = actual_ftr(row)
    elif market == "OU25":
        actual = actual_ou25(row)
    elif market == "BTTS":
        actual = actual_btts(row)
    elif market == "HOME_TEAM_GOALS_15":
        actual = actual_team_goals_15(row, "home")
    elif market == "AWAY_TEAM_GOALS_15":
        actual = actual_team_goals_15(row, "away")
    else:
        actual = ""
    if not actual:
        return "missing_actual"
    return "won" if pick == actual else "lost"


def model_alignment(row: pd.Series, signal: dict[str, Any]) -> str:
    model_market = str(row.get("market") or "").upper()
    model_pick = str(row.get("pick") or "").upper()
    market = signal["market"]
    signal_pick = str(signal["signal_pick"] or "").upper()
    state = str(signal["signal_state"] or "").upper()
    if market != model_market:
        return "not_primary_market"
    if signal_pick in {"WATCH", "NO_EDGE"}:
        return "neutral"
    if model_pick == signal_pick and state != "AVOID":
        return "supports_model"
    if state == "AVOID" or model_pick != signal_pick:
        return "conflicts_model"
    return "neutral"


def build_rows(score_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows = []
    shape_rows = []
    for _, row in score_rows.iterrows():
        fixture_key = str(row.get("fixture_key") or "")
        decision = load_json(DECISION_ROOT / f"{fixture_key}.json")
        metrics = metric_map(decision)
        tokens = support_tokens(decision)
        flags = shape_flags(metrics)
        signals = [
            ftr_signal(metrics),
            ou25_signal(metrics, tokens),
            btts_signal(metrics, tokens),
            team_goals_signal(metrics, decision, row, "home"),
            team_goals_signal(metrics, decision, row, "away"),
        ]
        base = {
            "fixture_key": fixture_key,
            "kickoff_time": row.get("kickoff_time"),
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "publish_class": row.get("publish_class"),
            "tier": row.get("tier"),
            "model_market": row.get("market"),
            "model_pick": row.get("pick"),
            "model_result_status": row.get("result_status"),
            "home_score": row.get("home_score"),
            "away_score": row.get("away_score"),
            "shape_flags": ";".join(flags),
            "pre_kickoff_eligible": decision.get("pre_kickoff_eligible"),
            "snapshot_phase": decision.get("snapshot_phase"),
        }
        for short, values in metrics.items():
            base[f"home_{short}"] = values["home"]
            base[f"away_{short}"] = values["away"]
            base[f"{short}_delta"] = values["delta"]
        for signal in signals:
            out = dict(base)
            out.update(signal)
            out["signal_result_status"] = signal_hit(row, signal)
            out["model_alignment"] = model_alignment(row, signal)
            matrix_rows.append(out)
        for flag in flags:
            shape_rows.append({**base, "shape_flag": flag, "actual_ftr": actual_ftr(row), "actual_ou25": actual_ou25(row), "actual_btts": actual_btts(row)})
    return pd.DataFrame(matrix_rows), pd.DataFrame(shape_rows)


def summarize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    settled = matrix[matrix["signal_result_status"].isin(["won", "lost"])].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["is_win"] = settled["signal_result_status"].eq("won")
    return (
        settled.groupby(["market", "signal_state", "signal_pick"], dropna=False)
        .agg(rows=("fixture_key", "count"), wins=("is_win", "sum"), avg_score=("signal_score", "mean"))
        .reset_index()
        .assign(hit_rate=lambda df: (df["wins"] / df["rows"]).round(4), avg_score=lambda df: df["avg_score"].round(1))
        .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
    )


def summarize_alignment(matrix: pd.DataFrame) -> pd.DataFrame:
    primary = matrix[matrix["model_alignment"].ne("not_primary_market") & matrix["model_result_status"].isin(["won", "lost"])].copy()
    if primary.empty:
        return pd.DataFrame()
    primary["model_win"] = primary["model_result_status"].eq("won")
    return (
        primary.groupby(["model_market", "model_alignment", "signal_state", "signal_pick"], dropna=False)
        .agg(rows=("fixture_key", "count"), model_wins=("model_win", "sum"), avg_signal_score=("signal_score", "mean"))
        .reset_index()
        .assign(model_hit_rate=lambda df: (df["model_wins"] / df["rows"]).round(4), avg_signal_score=lambda df: df["avg_signal_score"].round(1))
        .sort_values(["model_alignment", "model_hit_rate", "rows"], ascending=[True, True, False])
    )


def summarize_shapes(shapes: pd.DataFrame) -> pd.DataFrame:
    if shapes.empty:
        return pd.DataFrame()
    rows = []
    for flag, group in shapes.groupby("shape_flag"):
        rows.append(
            {
                "shape_flag": flag,
                "fixtures": len(group),
                "home_win_rate": round(float(group["actual_ftr"].eq("HOME").mean()), 4),
                "away_win_rate": round(float(group["actual_ftr"].eq("AWAY").mean()), 4),
                "draw_rate": round(float(group["actual_ftr"].eq("DRAW").mean()), 4),
                "over25_rate": round(float(group["actual_ou25"].eq("OVER25").mean()), 4),
                "btts_yes_rate": round(float(group["actual_btts"].eq("YES").mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["over25_rate", "btts_yes_rate", "fixtures"], ascending=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], matrix_summary: pd.DataFrame, alignment_summary: pd.DataFrame, shape_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Goal Market Signal Matrix",
            "",
            "Research layer for comparing website intelligence-derived goal-market signals against settled outcomes and model deploy/observe outputs.",
            "",
            "## Scope",
            "",
            f"- Fixtures scored: {summary['fixture_rows']}",
            f"- Matrix rows: {summary['matrix_rows']}",
            f"- Settled signal rows: {summary['settled_signal_rows']}",
            f"- Model conflicts: {summary['model_conflicts']}",
            f"- Model supports: {summary['model_supports']}",
            "",
            "## Derived Signal Backtest",
            "",
            markdown_table(matrix_summary),
            "",
            "## Model Alignment",
            "",
            markdown_table(alignment_summary),
            "",
            "## Signal Shape Outcomes",
            "",
            markdown_table(shape_summary),
            "",
            "## Product Meaning",
            "",
            "- Use this as a bridge report, not as deploy policy.",
            "- Strong model/support alignment can become a confidence enhancer after larger validation.",
            "- Model/conflict rows are the highest-value review queue because they may prevent weak acca legs.",
            "- `pre_kickoff_eligible` must be true before any row can influence production deploy rules.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_SCORE_ROWS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    score_rows = pd.read_csv(args.score_rows)
    matrix, shapes = build_rows(score_rows)
    matrix_summary = summarize_matrix(matrix)
    alignment_summary = summarize_alignment(matrix)
    shape_summary = summarize_shapes(shapes)

    args.outdir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.outdir / "goal_market_signal_matrix_rows.csv", index=False)
    shapes.to_csv(args.outdir / "goal_market_signal_shape_rows.csv", index=False)
    matrix_summary.to_csv(args.outdir / "goal_market_signal_summary.csv", index=False)
    alignment_summary.to_csv(args.outdir / "goal_market_model_alignment_summary.csv", index=False)
    shape_summary.to_csv(args.outdir / "goal_market_shape_summary.csv", index=False)

    summary = {
        "fixture_rows": int(score_rows.shape[0]),
        "matrix_rows": int(matrix.shape[0]),
        "settled_signal_rows": int(matrix["signal_result_status"].isin(["won", "lost"]).sum()),
        "model_conflicts": int(matrix["model_alignment"].eq("conflicts_model").sum()),
        "model_supports": int(matrix["model_alignment"].eq("supports_model").sum()),
        "signal_result_counts": dict(Counter(matrix["signal_result_status"])),
        "outputs": str(args.outdir.relative_to(ROOT)),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.outdir / "SUMMARY.md").write_text(write_summary(summary, matrix_summary, alignment_summary, shape_summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
