#!/usr/bin/env python3
"""Score the live MLS test night against model, intelligence, and beta props."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER_ROOT = ROOT / "reports/latest/api_current_context_overlay_window_mls_2026_05_13_to_2026_05_14_final"
DEFAULT_MODEL_ROWS = (
    ROOT
    / "reports/latest/live_mls_model_intelligence_compare_2026_05_13"
    / "live_mls_model_intelligence_rows.csv"
)
DEFAULT_PLAYER_BETA_ROWS = (
    ROOT
    / "reports/latest/live_mls_player_event_beta_shortlist_2026_05_13_wave_0030_confirmed"
    / "live_mls_player_event_beta_review_rows.csv"
)
DEFAULT_OUTDIR = ROOT / "reports/latest/live_mls_night_audit_2026_05_14"

ALIASES = {
    "CF Montreal": "Montreal Impact",
    "Orlando City SC": "Orlando City",
    "New York Red Bulls": "New York RB",
    "Sporting Kansas City": "Sporting KC",
    "Los Angeles Galaxy": "LA Galaxy",
    "San Jose Earthquakes": "SJ Earthquakes",
    "Minnesota United FC": "Minnesota United",
    "New York City FC": "New York City",
    "San Diego FC": "San Diego",
}


def canon(value: Any) -> str:
    return ALIASES.get(str(value), str(value))


def norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def score_for(team_stats: pd.DataFrame, fixture_id: int) -> tuple[int, int] | None:
    rows = team_stats[team_stats.fixture_id.eq(fixture_id)]
    home = rows[rows.is_home.eq(1)]
    away = rows[rows.is_home.eq(0)]
    if home.empty or away.empty:
        return None
    return int(home.iloc[0].goals_for), int(away.iloc[0].goals_for)


def ftr(score: tuple[int, int] | None) -> str | None:
    if score is None:
        return None
    if score[0] > score[1]:
        return "HOME"
    if score[1] > score[0]:
        return "AWAY"
    return "DRAW"


def btts(score: tuple[int, int] | None) -> str | None:
    if score is None:
        return None
    return "YES" if score[0] > 0 and score[1] > 0 else "NO"


def ou25(score: tuple[int, int] | None) -> str | None:
    if score is None:
        return None
    return "OVER25" if sum(score) > 2 else "UNDER25"


def actual_for_market(market: str, score: tuple[int, int] | None) -> str | None:
    market = market.upper()
    if market == "FTR":
        return ftr(score)
    if market == "BTTS":
        return btts(score)
    if market == "OU25":
        return ou25(score)
    return None


def load_provider(provider_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = provider_root / "normalized"
    fixtures = pd.read_csv(normalized / "fixtures_master__USA_MLS__2026.csv")
    team_stats = pd.read_csv(normalized / "match_team_stats__USA_MLS__2026.csv")
    player_stats = pd.read_csv(normalized / "match_player_stats__USA_MLS__2026.csv")
    return fixtures, team_stats, player_stats


def fixture_truth(fixtures: pd.DataFrame, team_stats: pd.DataFrame, model_rows: pd.DataFrame) -> pd.DataFrame:
    key_by_pair = {
        (row.home_team, row.away_team): row.fixture_key
        for row in model_rows.drop_duplicates("fixture_key").itertuples()
    }
    rows = []
    for fx in fixtures.itertuples():
        home = canon(fx.home_team_name)
        away = canon(fx.away_team_name)
        fixture_key = key_by_pair.get((home, away))
        if not fixture_key:
            continue
        score = score_for(team_stats, int(fx.fixture_id))
        rows.append(
            {
                "fixture_id": int(fx.fixture_id),
                "fixture_key": fixture_key,
                "kickoff_ts_utc": fx.kickoff_ts_utc,
                "status": fx.status,
                "home_team": home,
                "away_team": away,
                "home_goals": score[0] if score else None,
                "away_goals": score[1] if score else None,
                "total_goals": sum(score) if score else None,
                "actual_ftr": ftr(score),
                "actual_btts": btts(score),
                "actual_ou25": ou25(score),
            }
        )
    return pd.DataFrame(rows)


def score_model_rows(model_rows: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_by_key = truth.set_index("fixture_key").to_dict("index")
    scored = []
    for row in model_rows.itertuples():
        truth_row = truth_by_key.get(row.fixture_key)
        if not truth_row:
            continue
        actual = actual_for_market(row.market, (truth_row["home_goals"], truth_row["away_goals"]))
        if actual is None:
            continue
        scored.append(
            {
                "fixture_key": row.fixture_key,
                "kickoff_ts_utc": truth_row["kickoff_ts_utc"],
                "home_team": truth_row["home_team"],
                "away_team": truth_row["away_team"],
                "score": f"{truth_row['home_goals']}-{truth_row['away_goals']}",
                "deploy_tier": row.deploy_tier,
                "market": str(row.market).upper(),
                "model_pick": row.model_pick,
                "actual": actual,
                "hit": int(str(row.model_pick).upper() == str(actual).upper()),
                "site_signal_alignment": row.site_signal_alignment,
                "site_signal_pick": row.site_signal_pick,
                "site_signal_state": row.site_signal_state,
                "site_signal_score": row.site_signal_score,
                "model_prob": row.model_prob,
                "value_edge": row.value_edge,
            }
        )
    return pd.DataFrame(scored)


def player_event_actual(row: pd.Series, player_stats: pd.DataFrame) -> tuple[str, Any]:
    player_name = norm(row.player_name)
    surname = player_name.split()[-1] if player_name else ""
    candidates = player_stats[player_stats.player_name.map(norm).eq(player_name)]
    if candidates.empty and surname:
        candidates = player_stats[player_stats.player_name.map(lambda value: norm(value).split()[-1] if norm(value) else "").eq(surname)]
    if candidates.empty:
        return "NO_STAT", ""
    stat = candidates.iloc[0]
    event = row.event_key
    if event == "shots_on_target":
        value = int(stat.shots_on_target)
        return ("HIT" if value >= 1 else "MISS"), value
    if event == "shots":
        value = int(stat.shots_total)
        threshold = 2 if str(row.line_hint).endswith("1_5") else 3
        return ("HIT" if value >= threshold else "MISS"), value
    if event == "key_passes":
        value = int(stat.passes_key)
        return ("HIT" if value >= 1 else "MISS"), value
    if event == "goalkeeper_saves":
        value = int(stat.saves)
        return ("HIT" if value >= 2 else "MISS"), value
    return "UNSCORED", ""


def score_player_beta(beta_rows_path: Path, player_stats: pd.DataFrame) -> pd.DataFrame:
    if not beta_rows_path.exists():
        return pd.DataFrame()
    beta = pd.read_csv(beta_rows_path)
    rows = []
    for _, row in beta.iterrows():
        result, actual = player_event_actual(row, player_stats)
        rows.append({**row.to_dict(), "actual_value": actual, "event_result": result, "event_hit": int(result == "HIT")})
    return pd.DataFrame(rows)


def write_summary(outdir: Path, truth: pd.DataFrame, scored: pd.DataFrame, beta: pd.DataFrame, provider_root: Path) -> None:
    deploy = scored[scored.deploy_tier.isin(["ELITE", "STANDARD"])].copy()
    observe = scored[scored.deploy_tier.eq("OBSERVE")].copy()
    lines = [
        "# MLS Live Night Audit",
        "",
        f"Provider snapshot: `{provider_root}`",
        "",
        "## Final Scores",
    ]
    for row in truth.sort_values(["kickoff_ts_utc", "fixture_id"]).itertuples():
        lines.append(
            f"- {row.kickoff_ts_utc} | {row.home_team} {row.home_goals}-{row.away_goals} {row.away_team} "
            f"| OU25={row.actual_ou25} BTTS={row.actual_btts} FTR={row.actual_ftr}"
        )
    lines.extend(["", "## Deploy Score"])
    if deploy.empty:
        lines.append("- No deploy rows scored.")
    else:
        for (tier, market), group in deploy.groupby(["deploy_tier", "market"]):
            lines.append(f"- {tier} {market}: {int(group.hit.sum())}/{len(group)}")
        lines.append(f"- Total deploy rows: {int(deploy.hit.sum())}/{len(deploy)}")
    lines.extend(["", "## Intelligence Alignment On Deploy Rows"])
    if not deploy.empty:
        for alignment, group in deploy.groupby("site_signal_alignment"):
            lines.append(f"- {alignment}: {int(group.hit.sum())}/{len(group)}")
    lines.extend(["", "## Observe Rows"])
    if observe.empty:
        lines.append("- No observe rows scored.")
    else:
        for market, group in observe.groupby("market"):
            lines.append(f"- OBSERVE {market}: {int(group.hit.sum())}/{len(group)}")
    lines.extend(["", "## Player Event Beta"])
    if beta.empty:
        lines.append("- No beta player-event rows scored.")
    else:
        scored_beta = beta[beta.event_result.isin(["HIT", "MISS"])]
        lines.append(f"- Review rows scored: {int(scored_beta.event_hit.sum())}/{len(scored_beta)}")
        for event, group in scored_beta.groupby("event_key"):
            lines.append(f"- {event}: {int(group.event_hit.sum())}/{len(group)}")
        lines.append("")
        lines.append("### Top Beta Rows")
        for row in beta.sort_values("beta_score", ascending=False).head(20).itertuples():
            lines.append(
                f"- {row.event_label} {row.line_hint} | {row.player_name} ({row.team}) "
                f"| {row.beta_bucket} {row.beta_score} | actual={row.actual_value} | {row.event_result}"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- This audit scores prediction outputs and beta intelligence only; it does not change deploy policy.",
            "- Player-event rows are manual-review beta rows, not priced player-prop odds.",
        ]
    )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    parser.add_argument("--model-rows", type=Path, default=DEFAULT_MODEL_ROWS)
    parser.add_argument("--player-beta-rows", type=Path, default=DEFAULT_PLAYER_BETA_ROWS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    fixtures, team_stats, player_stats = load_provider(args.provider_root)
    model_rows = pd.read_csv(args.model_rows)
    truth = fixture_truth(fixtures, team_stats, model_rows)
    scored = score_model_rows(model_rows, truth)
    beta = score_player_beta(args.player_beta_rows, player_stats)

    args.outdir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.outdir / "fixture_truth.csv", index=False)
    scored.to_csv(args.outdir / "model_intelligence_scored_rows.csv", index=False)
    beta.to_csv(args.outdir / "player_event_beta_scored_rows.csv", index=False)
    summary = {
        "fixtures": len(truth),
        "deploy_rows": int(scored.deploy_tier.isin(["ELITE", "STANDARD"]).sum()) if not scored.empty else 0,
        "deploy_hits": int(scored[scored.deploy_tier.isin(["ELITE", "STANDARD"])].hit.sum()) if not scored.empty else 0,
        "observe_rows": int(scored.deploy_tier.eq("OBSERVE").sum()) if not scored.empty else 0,
        "observe_hits": int(scored[scored.deploy_tier.eq("OBSERVE")].hit.sum()) if not scored.empty else 0,
        "player_event_review_rows": len(beta),
        "player_event_hits": int(beta.event_hit.sum()) if not beta.empty else 0,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary(args.outdir, truth, scored, beta, args.provider_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
