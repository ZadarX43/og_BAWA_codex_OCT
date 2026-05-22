#!/usr/bin/env python3
"""Analyze pre-kickoff fixture-feed rows against published intelligence signals."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORE_ROWS = ROOT / "reports" / "latest" / "weekend_prediction_intelligence_scoring" / "primary_prediction_score_rows.csv"
MARKET_ROWS = ROOT / "reports" / "latest" / "weekend_prediction_intelligence_scoring" / "market_intelligence_score_rows.csv"
SNAPSHOT_ROWS = ROOT / "reports" / "latest" / "pre_kickoff_snapshot_audit" / "pre_kickoff_snapshot_audit_rows.csv"
DECISION_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_decision_intelligence"
OUTDIR = ROOT / "reports" / "latest" / "pre_kickoff_intelligence_signal_analysis"

TEAM_METRICS = [
    "goal_heat_rating",
    "btts_pressure_rating",
    "attack_flow_rating",
    "defensive_lock_rating",
    "first_strike_rating",
    "chaos_rating",
]


def load_decision(fixture_key: str) -> dict[str, Any]:
    path = DECISION_ROOT / f"{fixture_key}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ftr_actual(row: pd.Series) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    if float(home) > float(away):
        return "HOME"
    if float(home) < float(away):
        return "AWAY"
    return "DRAW"


def bool_actual(row: pd.Series, market: str) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    total = float(home) + float(away)
    if market == "OU25":
        return "OVER25" if total > 2.5 else "UNDER25"
    if market == "BTTS":
        return "YES" if float(home) > 0 and float(away) > 0 else "NO"
    return ""


def outcome_for_market(row: pd.Series, market: str, pick: str) -> str:
    pick = str(pick or "").upper()
    if market == "FTR":
        return "won" if pick == ftr_actual(row) else "lost"
    if market in {"OU25", "BTTS"}:
        return "won" if pick == bool_actual(row, market) else "lost"
    return ""


def metric_rows(primary: pd.DataFrame, pre_keys: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture_key in sorted(pre_keys):
        match = primary[primary["fixture_key"].eq(fixture_key)]
        if match.empty:
            continue
        score_row = match.iloc[0]
        decision = load_decision(fixture_key)
        faceoff = decision.get("team_faceoff_summary") or []
        actual_ftr = ftr_actual(score_row)
        for item in faceoff:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            if metric not in TEAM_METRICS:
                continue
            leader = str(item.get("leader") or "")
            if actual_ftr == "DRAW":
                ftr_hit = "draw_no_side"
            elif actual_ftr == "HOME":
                ftr_hit = "hit" if leader == str(score_row.get("home_team") or "") else "miss"
            elif actual_ftr == "AWAY":
                ftr_hit = "hit" if leader == str(score_row.get("away_team") or "") else "miss"
            else:
                ftr_hit = ""
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "league": score_row.get("league"),
                    "home_team": score_row.get("home_team"),
                    "away_team": score_row.get("away_team"),
                    "actual_ftr": actual_ftr,
                    "actual_btts": bool_actual(score_row, "BTTS"),
                    "actual_ou25": bool_actual(score_row, "OU25"),
                    "metric": metric,
                    "label": item.get("label"),
                    "home_value": item.get("home_value"),
                    "away_value": item.get("away_value"),
                    "delta": item.get("delta"),
                    "leader": leader,
                    "leader_ftr_result": ftr_hit,
                }
            )
    return rows


def summarize_metric_alignment(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for metric, group in rows.groupby("metric"):
        decisive = group[group["leader_ftr_result"].isin(["hit", "miss"])]
        high_delta = decisive[pd.to_numeric(decisive["delta"], errors="coerce").fillna(0) >= 8]
        summaries.append(
            {
                "metric": metric,
                "decisive_matches": len(decisive),
                "leader_ftr_hits": int(decisive["leader_ftr_result"].eq("hit").sum()),
                "leader_ftr_hit_rate": round(float(decisive["leader_ftr_result"].eq("hit").mean()), 4) if len(decisive) else None,
                "high_delta_matches": len(high_delta),
                "high_delta_hits": int(high_delta["leader_ftr_result"].eq("hit").sum()),
                "high_delta_hit_rate": round(float(high_delta["leader_ftr_result"].eq("hit").mean()), 4) if len(high_delta) else None,
            }
        )
    return pd.DataFrame(summaries).sort_values(["high_delta_hit_rate", "leader_ftr_hit_rate"], ascending=False, na_position="last")


def market_summary(markets: pd.DataFrame, pre_keys: set[str]) -> pd.DataFrame:
    rows = markets[markets["fixture_key"].isin(pre_keys)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["is_win"] = rows["result_status"].eq("won")
    return (
        rows.groupby(["market", "state"], dropna=False)
        .agg(rows=("fixture_key", "count"), wins=("is_win", "sum"), avg_alignment=("alignment_score", "mean"), avg_rating=("rating", "mean"))
        .reset_index()
        .assign(hit_rate=lambda df: (df["wins"] / df["rows"]).round(4), avg_alignment=lambda df: df["avg_alignment"].round(1), avg_rating=lambda df: df["avg_rating"].round(1))
        .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
    )


def threshold_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()
    rows = []
    numeric = metric_df.copy()
    numeric["home_value"] = pd.to_numeric(numeric["home_value"], errors="coerce")
    numeric["away_value"] = pd.to_numeric(numeric["away_value"], errors="coerce")
    numeric["max_value"] = numeric[["home_value", "away_value"]].max(axis=1)
    numeric["avg_value"] = numeric[["home_value", "away_value"]].mean(axis=1)
    numeric["sum_value"] = numeric["home_value"] + numeric["away_value"]
    checks = [
        ("max>=75", lambda frame: frame["max_value"] >= 75),
        ("avg>=65", lambda frame: frame["avg_value"] >= 65),
        ("sum>=140", lambda frame: frame["sum_value"] >= 140),
    ]
    for metric, group in numeric.groupby("metric"):
        for check_name, check in checks:
            sample = group[check(group)]
            if sample.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "threshold": check_name,
                    "fixtures": len(sample),
                    "ou25_over_rate": round(float(sample["actual_ou25"].eq("OVER25").mean()), 4),
                    "btts_yes_rate": round(float(sample["actual_btts"].eq("YES").mean()), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["ou25_over_rate", "fixtures"], ascending=False)


def write_markdown(
    summary: dict[str, Any],
    metric_summary_df: pd.DataFrame,
    market_summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
) -> str:
    metric_table = markdown_table(metric_summary_df) if not metric_summary_df.empty else "_No metric rows._"
    market_table = markdown_table(market_summary_df) if not market_summary_df.empty else "_No market rows._"
    threshold_table = markdown_table(threshold_summary_df) if not threshold_summary_df.empty else "_No threshold rows._"
    return "\n".join(
        [
            "# Pre-Kickoff Intelligence Signal Analysis",
            "",
            "Scope: fixture-feed rows that were available before kickoff. Downstream decision/lineup/H2H payloads are currently backfilled, so this is discovery analysis rather than deploy-policy evidence.",
            "",
            "## Counts",
            "",
            f"- Pre-kickoff fixture rows: {summary['pre_kickoff_fixtures']}",
            f"- Settled rows: {summary['settled_rows']}",
            f"- Deploy rows in scope: {summary['deploy_rows']}",
            f"- Observe rows in scope: {summary['observe_rows']}",
            "",
            "## Primary Row Results",
            "",
            "```json",
            json.dumps(summary["primary_result_counts"], indent=2, sort_keys=True),
            "```",
            "",
            "## Team Rating Leader vs FTR",
            "",
            metric_table,
            "",
            "## Market Intelligence States",
            "",
            market_table,
            "",
            "## Goal/BTTS Threshold Checks",
            "",
            threshold_table,
            "",
            "## Interpretation",
            "",
            "- Treat strong metric alignment as a shortlist signal, not proof of a deploy rule yet.",
            "- The next live run should generate team/player/fixture intelligence before kickoff so the same report can be rerun with downstream `pre_kickoff_eligible=true`.",
            "- The useful candidate checks from this sample are high-delta first strike, goal heat, attack flow, and defensive lock against FTR, plus market-state alignment for BTTS/OU25.",
            "",
        ]
    )


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> int:
    primary = pd.read_csv(SCORE_ROWS)
    markets = pd.read_csv(MARKET_ROWS)
    snapshots = pd.read_csv(SNAPSHOT_ROWS)
    pre_keys = set(snapshots[snapshots["feed_pre_kickoff_eligible"].eq(True)]["fixture_key"].astype(str))
    primary_pre = primary[primary["fixture_key"].isin(pre_keys)].copy()
    primary_pre["primary_result"] = primary_pre["result_status"].fillna("")
    metric_df = pd.DataFrame(metric_rows(primary, pre_keys))
    metric_summary_df = summarize_metric_alignment(metric_df) if not metric_df.empty else pd.DataFrame()
    market_summary_df = market_summary(markets, pre_keys)
    threshold_summary_df = threshold_summary(metric_df)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    primary_pre.to_csv(OUTDIR / "pre_kickoff_primary_rows.csv", index=False)
    metric_df.to_csv(OUTDIR / "team_rating_metric_rows.csv", index=False)
    metric_summary_df.to_csv(OUTDIR / "team_rating_metric_summary.csv", index=False)
    market_summary_df.to_csv(OUTDIR / "market_state_summary.csv", index=False)
    threshold_summary_df.to_csv(OUTDIR / "goal_btts_threshold_summary.csv", index=False)

    summary = {
        "pre_kickoff_fixtures": len(pre_keys),
        "settled_rows": int(primary_pre["result_status"].isin(["won", "lost"]).sum()),
        "deploy_rows": int(primary_pre["publish_class"].eq("DEPLOY").sum()),
        "observe_rows": int(primary_pre["publish_class"].eq("OBSERVE").sum()),
        "primary_result_counts": dict(Counter(primary_pre["result_status"].fillna(""))),
        "outputs": str(OUTDIR.relative_to(ROOT)),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTDIR / "SUMMARY.md").write_text(write_markdown(summary, metric_summary_df, market_summary_df, threshold_summary_df), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
