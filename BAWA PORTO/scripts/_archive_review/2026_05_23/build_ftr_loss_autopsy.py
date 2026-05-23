#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_ROWS = ROOT / "reports/latest/weekend_deploy_tier_scoring_2026_05_14_to_2026_05_19/DEPLOY_TIER_SCORE_ROWS.csv"
DEFAULT_CROSS_LAYER = ROOT / "reports/latest/full_cross_layer_analysis_2026_05_15_to_2026_05_19/FULL_CROSS_LAYER_ANALYSIS.csv"
DEFAULT_LIVE_INTEL = ROOT / "reports/latest/live_weekend_model_intelligence_compare_2026_05_14_to_2026_05_19/live_model_intelligence_rows.csv"
DEFAULT_INJURY_SCAN = ROOT / "reports/latest/injury_shock_coverage_scan/INJURY_SHOCK_COVERAGE_SCAN.csv"
DEFAULT_OUTDIR = ROOT / "reports/latest/ftr_loss_autopsy_2026_05_14_to_2026_05_19"

FTR_LEAGUE_HEALTH_SEED = {
    # Shadow seed from current season/top-pick FTR health observations. This is audit-only.
    "Spain La Liga": 0.299,
    "France Ligue 1": 0.373,
    "USA MLS": 0.266,
    "Norway Eliteserien": 0.404,
    "Japan J1": 0.401,
    "England Premier League": 0.525,
    "Italy Serie A": 0.525,
    "Germany Bundesliga": 0.525,
}

MARKET_UNSAFE_LEAGUES = {
    "Turkey Super Lig": "NOT_TRAINED_UNSAFE_FTR",
}

END_SEASON_LEAGUES = {
    "Germany Bundesliga",
    "France Ligue 1",
    "Turkey Super Lig",
    "Netherlands Eredivisie",
    "Belgium Pro",
    "Italy Serie A",
    "Spain La Liga",
    "England Premier League",
}


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team(value: Any) -> str:
    text = f" {str(value or '').lower()} "
    replacements = {
        " sj earthquakes ": " san jose earthquakes ",
        " sporting kc ": " sporting kansas city ",
        " la galaxy ": " los angeles galaxy ",
        " lafc ": " los angeles fc ",
        " fc cincinnati ": " cincinnati ",
        " stade brestois 29 ": " brest ",
        " angers sco ": " angers ",
        " krc genk ": " genk ",
        " konyaspor ": " konyaspor ",
        " genclerbirligi ": " genclerbirligi ",
        " gençlerbirliği ": " genclerbirligi ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return norm(text)


def fixture_key(row: pd.Series, home_col: str = "home_team_name", away_col: str = "away_team_name") -> str:
    return "|".join([str(row.get("match_date") or "")[:10], canonical_team(row.get(home_col)), canonical_team(row.get(away_col))])


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool_flag(value: Any) -> int:
    try:
        return int(float(value or 0) > 0)
    except ValueError:
        return 0


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = to_float(row.get(col))
    return default if value is None else value


def league_health(league: str) -> tuple[str, float | None, str]:
    if league in MARKET_UNSAFE_LEAGUES:
        return "UNSAFE", None, MARKET_UNSAFE_LEAGUES[league]
    value = FTR_LEAGUE_HEALTH_SEED.get(league)
    if value is None:
        return "UNKNOWN", None, "NO_DYNAMIC_FTR_HEALTH_SEED"
    if value < 0.35:
        return "RED", value, "LOW_FTR_TOP_PICK_HEALTH"
    if value < 0.45:
        return "AMBER", value, "WEAK_FTR_TOP_PICK_HEALTH"
    if value < 0.50:
        return "WATCH", value, "BORDERLINE_FTR_TOP_PICK_HEALTH"
    return "GREEN", value, "ACCEPTABLE_FTR_TOP_PICK_HEALTH"


def match_injury(row: pd.Series, injury: pd.DataFrame) -> pd.Series | None:
    if injury.empty:
        return None
    key = fixture_key(row)
    matched = injury[injury["autopsy_join_key"].eq(key)]
    if not matched.empty:
        return matched.iloc[0]
    home = canonical_team(row.get("home_team_name"))
    away = canonical_team(row.get("away_team_name"))
    date = str(row.get("match_date") or "")[:10]
    candidates = injury[injury["match_date"].astype(str).str[:10].eq(date)].copy()
    if candidates.empty:
        return None
    candidates["score"] = candidates.apply(
        lambda item: int(home in canonical_team(item.get("home_team_name")) or canonical_team(item.get("home_team_name")) in home)
        + int(away in canonical_team(item.get("away_team_name")) or canonical_team(item.get("away_team_name")) in away),
        axis=1,
    )
    candidates = candidates.sort_values("score", ascending=False)
    if not candidates.empty and int(candidates.iloc[0]["score"]) >= 2:
        return candidates.iloc[0]
    return None


def selected_side(pick: str) -> str:
    pick = str(pick or "").upper()
    if pick == "HOME":
        return "home"
    if pick == "AWAY":
        return "away"
    return "draw"


def opposite(side: str) -> str:
    return "away" if side == "home" else "home" if side == "away" else ""


def drawish_scoreline(value: Any) -> bool:
    text = str(value or "").strip()
    if "-" not in text:
        return False
    parts = text.split("-", 1)
    try:
        return int(parts[0]) == int(parts[1])
    except ValueError:
        return False


def classify_row(row: pd.Series) -> dict[str, Any]:
    side = selected_side(row.get("pick"))
    opp = opposite(side)
    health_bucket, health_value, health_reason = league_health(str(row.get("league") or ""))

    picked_attack = safe_float(row, f"{side}_attack_absence_score") if side else 0.0
    picked_mid = safe_float(row, f"{side}_midfield_absence_score") if side else 0.0
    picked_def = safe_float(row, f"{side}_defence_absence_score") if side else 0.0
    picked_lineup_conf = safe_float(row, f"{side}_lineup_confidence_score", 100.0) if side else 100.0
    opp_attack = safe_float(row, f"{opp}_attack_absence_score") if opp else 0.0
    opp_def = safe_float(row, f"{opp}_defence_absence_score") if opp else 0.0

    cs_draw = safe_float(row, "cs_mass_draw")
    cs_over25 = safe_float(row, "cs_mass_over25")
    sportsmole_ftr = str(row.get("sportsmole_ftr") or "").upper()
    cs_alignment = str(row.get("cs_alignment") or "")
    tg15_read = str(row.get("tg15_read") or "")
    actual = str(row.get("actual") or "").upper()
    result_status = str(row.get("result_status") or "")
    league = str(row.get("league") or "")
    cross_read = str(row.get("cross_layer_read") or "")

    draw_risk = int(
        actual == "DRAW"
        or sportsmole_ftr == "DRAW"
        or cs_draw >= 0.25
        or drawish_scoreline(row.get("cs_top1_scoreline"))
        or drawish_scoreline(row.get("sportsmole_score"))
    )
    cs_conflict = int("CONFLICT" in cs_alignment.upper())
    tg15_no_help = int("DOES_NOT_HELP" in tg15_read.upper() or (side and "AVOID" in str(row.get(f"{side}_tg15") or "").upper()))
    preview_counter = int("COUNTER" in str(row.get("sportsmole_alignment") or "").upper() or "PREVIEW_COUNTER" in cross_read)
    injury_shock = int(
        safe_float(row, "ftr_volatility_adjustment") >= 0.35
        or picked_attack >= 10
        or picked_mid >= 10
        or to_bool_flag(row.get("deploy_warning_flag")) == 1
    )
    lineup_uncertainty = int(picked_lineup_conf < 96 or safe_float(row, "home_lineup_confidence_score", 100) < 96 or safe_float(row, "away_lineup_confidence_score", 100) < 96)
    motivation_volatility = int(safe_float(row, "motivation_volatility_score") >= 0.3 or league in END_SEASON_LEAGUES)
    weak_league = int(health_bucket in {"UNSAFE", "RED", "AMBER", "UNKNOWN"})
    market_drift = int(safe_float(row, "bookie_od") >= 1.75)
    stale_power = int("FULL_CONSENSUS" in cross_read and result_status == "lost" and (weak_league or injury_shock or motivation_volatility))

    reasons: list[str] = []
    if weak_league:
        reasons.append(health_reason)
    if draw_risk:
        reasons.append("DRAW_STALEMATE_RISK")
    if injury_shock:
        reasons.append("INJURY_LINEUP_SHOCK")
    if lineup_uncertainty:
        reasons.append("LINEUP_UNCERTAINTY")
    if motivation_volatility:
        reasons.append("END_SEASON_MOTIVATION_VOLATILITY")
    if cs_conflict:
        reasons.append("CS_CONFLICT")
    if tg15_no_help:
        reasons.append("TG15_NO_HELP_OR_PICK_SIDE_AVOID")
    if preview_counter:
        reasons.append("PREVIEW_COUNTER_SIGNAL")
    if market_drift:
        reasons.append("MARKET_PRICE_WEAKNESS")
    if stale_power:
        reasons.append("STALE_POWER_CONSENSUS_WARNING")

    preventable_loss = int(result_status == "lost" and len(reasons) > 0)
    preventability = "NOT_LOSS"
    if result_status == "lost":
        if weak_league or injury_shock or lineup_uncertainty or preview_counter or cs_conflict:
            preventability = "HIGH"
        elif draw_risk or motivation_volatility or tg15_no_help:
            preventability = "MEDIUM"
        else:
            preventability = "LOW"

    return {
        "selected_side": side.upper(),
        "ftr_league_health_bucket": health_bucket,
        "ftr_league_health_seed": health_value,
        "ftr_league_health_reason": health_reason,
        "weak_or_unsafe_league_flag": weak_league,
        "draw_stalemate_risk_flag": draw_risk,
        "injury_lineup_shock_flag": injury_shock,
        "lineup_uncertainty_flag": lineup_uncertainty,
        "motivation_volatility_flag": motivation_volatility,
        "cs_conflict_flag": cs_conflict,
        "tg15_no_help_flag": tg15_no_help,
        "preview_counter_flag": preview_counter,
        "market_price_weakness_flag": market_drift,
        "stale_power_consensus_flag": stale_power,
        "picked_side_attack_absence_score": picked_attack,
        "picked_side_midfield_absence_score": picked_mid,
        "picked_side_defence_absence_score": picked_def,
        "picked_side_lineup_confidence_score": picked_lineup_conf,
        "opponent_attack_absence_score": opp_attack,
        "opponent_defence_absence_score": opp_def,
        "preventable_loss_flag": preventable_loss,
        "preventability_bucket": preventability,
        "autopsy_reason_codes": "|".join(reasons) if reasons else "",
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        settled = [row for row in group_rows if row.get("result_status") in {"won", "lost"}]
        wins = sum(1 for row in settled if row.get("result_status") == "won")
        losses = sum(1 for row in settled if row.get("result_status") == "lost")
        item = {key: group_key[idx] for idx, key in enumerate(keys)}
        item.update(
            {
                "rows": len(group_rows),
                "settled": len(settled),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(wins / len(settled), 4) if settled else None,
                "preventable_losses": sum(1 for row in settled if row.get("preventable_loss_flag") == 1),
            }
        )
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FTR Loss Autopsy",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Window: `{summary['window']}`",
        "",
        "## Headline",
        "",
        f"- FTR rows: {summary['overall']['rows']}",
        f"- Settled: {summary['overall']['settled']}",
        f"- Wins: {summary['overall']['wins']}",
        f"- Losses: {summary['overall']['losses']}",
        f"- Hit rate: {summary['overall']['hit_rate']}",
        f"- Preventable losses flagged: {summary['overall']['preventable_losses']}",
        "",
        "## By Tier",
        "",
        "| Tier | Wins | Settled | Hit Rate | Preventable Losses |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["by_tier"]:
        hit = "n/a" if item["hit_rate"] is None else f"{item['hit_rate'] * 100:.1f}%"
        lines.append(f"| {item['score_tier']} | {item['wins']} | {item['settled']} | {hit} | {item['preventable_losses']} |")

    lines.extend(["", "## Loss Taxonomy", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in summary["loss_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Lost ELITE/STANDARD Rows",
            "",
            "| Tier | Fixture | Pick | Result | Preventability | Reason Codes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        if row.get("score_tier") not in {"ELITE", "STANDARD"} or row.get("result_status") != "lost":
            continue
        fixture = f"{row.get('home_team_name')} vs {row.get('away_team_name')}"
        result = f"{row.get('home_goals')}-{row.get('away_goals')}"
        lines.append(
            f"| {row.get('score_tier')} | {fixture} | {row.get('pick')} | {result} | "
            f"{row.get('preventability_bucket')} | {row.get('autopsy_reason_codes')} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a research sidecar only. It does not change deploy gates.",
            "- League health is seeded from current top-pick FTR weakness observations and should be replaced by a rolling league throttle backtest before production use.",
            "- Injury shock rows are only as good as the current injury/lineup refresh timing; post-kickoff refreshes prove attribution but not deploy-time warning quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an FTR miss/loss autopsy sidecar.")
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_SCORE_ROWS)
    parser.add_argument("--cross-layer", type=Path, default=DEFAULT_CROSS_LAYER)
    parser.add_argument("--live-intel", type=Path, default=DEFAULT_LIVE_INTEL)
    parser.add_argument("--injury-scan", type=Path, default=DEFAULT_INJURY_SCAN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--window", default="2026-05-14_to_2026-05-19")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    score = pd.read_csv(args.score_rows)
    cross = pd.read_csv(args.cross_layer) if args.cross_layer.exists() else pd.DataFrame()
    live = pd.read_csv(args.live_intel) if args.live_intel.exists() else pd.DataFrame()
    injury = pd.read_csv(args.injury_scan) if args.injury_scan.exists() else pd.DataFrame()

    ftr = score[score["score_market"].eq("FTR")].copy()
    ftr["autopsy_join_key"] = ftr.apply(fixture_key, axis=1)

    if not cross.empty:
        cross_ftr = cross[cross["market_l"].eq("ftr")].copy()
        cross_ftr["autopsy_join_key"] = cross_ftr.apply(fixture_key, axis=1)
        cross_cols = [
            "autopsy_join_key",
            "site_signal_alignment",
            "site_signal_pick",
            "site_signal_score",
            "site_signal_state",
            "tg15_read",
            "home_tg15",
            "away_tg15",
            "cs_alignment",
            "cs_support_label",
            "cs_top1_scoreline",
            "cs_top2_scoreline",
            "cs_top3_scoreline",
            "cs_mass_over25",
            "cs_mass_btts_yes",
            "cs_mass_home_win",
            "cs_mass_draw",
            "cs_mass_away_win",
            "sportsmole_score",
            "sportsmole_alignment",
            "sportsmole_ftr",
            "sportsmole_ou25",
            "sportsmole_btts",
            "layer_score",
            "cross_layer_read",
            "layer_notes",
            "shape_flags",
            "preview_note",
        ]
        ftr = ftr.merge(cross_ftr[[col for col in cross_cols if col in cross_ftr.columns]], on="autopsy_join_key", how="left")

    if not live.empty:
        live_ftr = live[live["market"].astype(str).str.upper().eq("FTR")].copy()
        live_ftr["home_team_name"] = live_ftr.get("home_team")
        live_ftr["away_team_name"] = live_ftr.get("away_team")
        live_ftr["autopsy_join_key"] = live_ftr.apply(fixture_key, axis=1)
        live_cols = [
            "autopsy_join_key",
            "competition_key",
            "team_intel_overlay_action",
            "team_intel_overlay_fit_score",
            "team_intel_overlay_reason",
            "home_absence_severity",
            "away_absence_severity",
            "bookie_over25_prob_norm",
            "bookie_btts_yes_prob_norm",
            "ftr_signal_pick",
            "ftr_signal_state",
            "ftr_signal_score",
            "ou25_signal_pick",
            "ou25_signal_state",
            "ou25_signal_score",
            "btts_signal_pick",
            "btts_signal_state",
            "btts_signal_score",
        ]
        ftr = ftr.merge(live_ftr[[col for col in live_cols if col in live_ftr.columns]], on="autopsy_join_key", how="left")

    if not injury.empty:
        injury = injury.copy()
        injury["autopsy_join_key"] = injury.apply(fixture_key, axis=1)

    rows: list[dict[str, Any]] = []
    for _, item in ftr.iterrows():
        base = item.to_dict()
        injury_row = match_injury(item, injury)
        if injury_row is not None:
            for col in [
                "home_attack_absence_score",
                "away_attack_absence_score",
                "home_midfield_absence_score",
                "away_midfield_absence_score",
                "home_defence_absence_score",
                "away_defence_absence_score",
                "home_keeper_absence_score",
                "away_keeper_absence_score",
                "home_lineup_confidence_score",
                "away_lineup_confidence_score",
                "motivation_volatility_score",
                "ftr_volatility_adjustment",
                "deploy_warning_flag",
                "warning_tokens",
                "home_player_impact_reasons",
                "away_player_impact_reasons",
                "injury_presence_review_flag",
                "injury_source_csv",
                "sidelined_source_csv",
            ]:
                base[col] = injury_row.get(col)
            base["injury_shock_join_status"] = "MATCH"
        else:
            base["injury_shock_join_status"] = "NO_MATCH"

        base.update(classify_row(pd.Series(base)))
        rows.append(base)

    loss_reason_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("result_status") != "lost":
            continue
        for reason in str(row.get("autopsy_reason_codes") or "").split("|"):
            if reason:
                loss_reason_counts[reason] += 1

    summary = {
        "generated_at": now_utc(),
        "window": args.window,
        "inputs": {
            "score_rows": str(args.score_rows.relative_to(ROOT) if args.score_rows.is_absolute() else args.score_rows),
            "cross_layer": str(args.cross_layer.relative_to(ROOT) if args.cross_layer.is_absolute() else args.cross_layer),
            "live_intel": str(args.live_intel.relative_to(ROOT) if args.live_intel.is_absolute() else args.live_intel),
            "injury_scan": str(args.injury_scan.relative_to(ROOT) if args.injury_scan.is_absolute() else args.injury_scan),
        },
        "overall": summarize(rows, [])[0],
        "by_tier": summarize(rows, ["score_tier"]),
        "by_league": summarize(rows, ["league"]),
        "by_tier_league": summarize(rows, ["score_tier", "league"]),
        "by_preventability": summarize(rows, ["preventability_bucket"]),
        "loss_reason_counts": dict(loss_reason_counts.most_common()),
        "outputs": {
            "csv": str((outdir / f"FTR_LOSS_AUTOPSY__{args.window}.csv").relative_to(ROOT)),
            "summary_json": str((outdir / "summary.json").relative_to(ROOT)),
            "summary_md": str((outdir / "SUMMARY.md").relative_to(ROOT)),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / f"FTR_LOSS_AUTOPSY__{args.window}.csv", rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text(markdown(summary, rows), encoding="utf-8")

    print(f"Rows: {summary['overall']['rows']}")
    print(f"Settled: {summary['overall']['settled']}")
    print(f"Wins/losses: {summary['overall']['wins']} / {summary['overall']['losses']}")
    print(f"Hit rate: {summary['overall']['hit_rate']}")
    print(f"Preventable losses: {summary['overall']['preventable_losses']}")
    print(f"Outputs: {outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
